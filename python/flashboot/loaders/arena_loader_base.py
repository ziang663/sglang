"""What the seed, the clone and the DP replicas all do the same way.

Namely: build the model exactly as the stock loader does, drop the parameters' own
storage and allocate ONE contiguous arena in checkpoint file layout, rebind every saved
parameter onto its slice of that arena, and — if anything goes wrong — actually release
what the attempt allocated before falling back to reading the checkpoint from disk.

The subclasses differ only in where the arena's bytes come from: this rank's own disk read
(:mod:`flashboot.loaders.seed`), another instance's arena
(:mod:`flashboot.loaders.clone`), or one segment from disk plus the rest from the other
replicas (:mod:`flashboot.loaders.data_parallel`).
"""
from __future__ import annotations

import dataclasses
import gc
import os
import time
import traceback
from typing import TYPE_CHECKING

import torch
from torch import nn

from sglang.srt.configs.load_config import LoadConfig
from sglang.srt.model_loader.loader import (
    BaseModelLoader,
    _get_quantization_config,
    _initialize_model,
)

from flashboot import model_binding, rendezvous, weight_preload
from flashboot.arena import ArenaHandle, ArenaRebinder, allocate_device_arena
from flashboot.arena_fill import ArenaFillConfig
from flashboot.loaders.coordination import CoordinationSettings
from flashboot.loaders.process_topology import ProcessTopology, global_shard_rank
from flashboot.loaders.split_fill import PENDING_SPLIT_FILLS
from flashboot.transport import ArenaRequirements, create_transport

if TYPE_CHECKING:
    from sglang.srt.configs.device_config import DeviceConfig
    from sglang.srt.configs.model_config import ModelConfig

class ShardedArenaLoaderBase(BaseModelLoader):
    """Machinery shared by the seed and clone loaders: model build, arena allocation,
    rebinding, and the invariants. Subclasses implement only ``load_model``."""

    SUPPORTED_ARCHITECTURES = (
        "DeepseekV4ForCausalLM", "GlmMoeDsaForCausalLM", "Glm4MoeLiteForCausalLM",
        "Qwen3MoeForCausalLM",
        # Speculative NEXTN/MTP draft architectures.
        "DeepseekV3ForCausalLMNextN", "DeepseekV4ForCausalLMNextN",
        "Glm4MoeForCausalLMNextN",
    )

    def __init__(self, load_config: LoadConfig):
        super().__init__(load_config)
        self.fill_config = ArenaFillConfig.from_environment()
        # Publish the arena on this rank's TCP rendezvous so a clone can pull it.
        self.serve_enabled = os.getenv("FB_FLASHSHARDED_SERVE", "0") == "1"
        # Deferred fill (FB_SPLIT_FILL=1): bind params onto a placeholder arena now and run
        # the real read+H2D at the very end of server startup (run_split_fill), so the
        # GPFS -> /dev/shm staging overlaps the whole startup instead of blocking the load.
        self.split_fill = os.getenv("FB_SPLIT_FILL", "0") == "1"

    def download_model(self, model_config: ModelConfig) -> None:
        pass  # the sharded checkpoint is always local

    # ── load entry with stock-loader fallback ──────────────────────────────────
    def load_model(self, *, model_config: ModelConfig,
                   device_config: DeviceConfig) -> nn.Module:
        """Entry point sglang calls. With FB_LOAD_FALLBACK=1 (default) any flash-load
        failure — seed disk/arena path or clone transport pull — releases the partial
        attempt and retries with the stock loader (FB_FALLBACK_LOAD_FORMAT, default
        ``sharded_state``: the on-disk format of flashboot checkpoints), so a broken
        NIC/preload/handshake degrades to a slower start instead of a dead server.
        FB_LOAD_FALLBACK=0 restores fail-fast."""
        pending_before = len(PENDING_SPLIT_FILLS)
        try:
            return self._load_flash_model(model_config=model_config,
                                          device_config=device_config)
        except Exception as error:
            if os.getenv("FB_LOAD_FALLBACK", "1") != "1":
                raise
            traceback.print_exc()
            fallback_format = os.getenv("FB_FALLBACK_LOAD_FORMAT", "sharded_state")
            # The REASON belongs in the banner line, not only in the stack dump above
            # it: a fallback is a silent success by design, and during bring-up this
            # one twice hid a real fault (a NIC probe that rejected RoCE and made the
            # rdma transport look absent; a seed that loaded but never published) —
            # both diagnosed as "the clone is slow" because the one line anybody greps
            # for said nothing about why.
            reason = f"{type(error).__name__}: {error}"
            print(f"[flashboot][fallback] {type(self).__name__} failed ({reason}) — "
                  f"falling back to stock load_format={fallback_format} "
                  f"(FB_LOAD_FALLBACK=0 to fail fast instead)", flush=True)
            self._release_partial_load(error, pending_before)
            return self._stock_fallback_load(fallback_format, model_config, device_config)

    def _load_flash_model(self, *, model_config: ModelConfig,
                          device_config: DeviceConfig) -> nn.Module:
        raise NotImplementedError

    def _release_partial_load(self, error: BaseException, pending_before: int) -> None:
        """ACTUALLY free what the failed attempt allocated, before the stock load runs.

        Subtle: while the except handler is live, ``error.__traceback__`` keeps every
        unwound frame — and their locals (the device-init model, the 40GB+ partial
        arena) — alive, so a bare gc.collect() frees NOTHING and the stock weights
        would land on top of a still-resident arena -> GPU OOM. Order matters:
          1. drain in-flight CUDA work (an async fill H2D must not outlive the arena),
          2. drop split-fill closures armed by the failed attempt (the module-level
             list would otherwise pin the arena forever and later run a stale fill),
          3. clear the traceback frames' locals (this is what actually drops the
             arena/model refs; the still-executing load_model frame is skipped),
          4. collect + return the CUDA caching-allocator pages,
          5. hand the tmpfs staging back to the stager (else 372GB/node of dead shm)."""
        if torch.cuda.is_available():
            try:
                torch.cuda.synchronize()
            except Exception:
                pass
        del PENDING_SPLIT_FILLS[pending_before:]
        # Clear the WHOLE exception chain: an exception raised while handling another
        # (__context__) or explicitly chained (__cause__) carries its own traceback
        # whose frames also pin the arena/model locals.
        chained, seen = error, set()
        while chained is not None and id(chained) not in seen:
            seen.add(id(chained))
            if chained.__traceback__ is not None:
                traceback.clear_frames(chained.__traceback__)
            chained = chained.__cause__ or chained.__context__
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            try:
                free_bytes, _total = torch.cuda.mem_get_info()
                print(f"[flashboot][fallback] rank{global_shard_rank()} partial state "
                      f"released, cuda free={free_bytes / 1e9:.1f}GB", flush=True)
            except Exception:
                pass
        if weight_preload.is_enabled():
            try:
                weight_preload.mark_consumed(weight_preload.preload_dir(),
                                             global_shard_rank())
            except Exception:
                pass

    def _stock_fallback_load(self, fallback_format: str, model_config: ModelConfig,
                             device_config: DeviceConfig) -> nn.Module:
        """Build the stock loader for ``fallback_format`` and load from the local
        checkpoint. Strips flashboot-only extra-config keys (role/seed_ip/seed_port —
        stock ShardedStateLoader rejects unknown keys)."""
        from sglang.srt.configs.load_config import LoadFormat
        from sglang.srt.model_loader.loader import get_model_loader

        extra = getattr(self.load_config, "model_loader_extra_config", None) or {}
        stock_extra = {k: v for k, v in extra.items() if k == "pattern"} \
            if isinstance(extra, dict) else {}
        stock_config = dataclasses.replace(
            self.load_config, load_format=LoadFormat(fallback_format),
            model_loader_extra_config=stock_extra)
        try:
            stock_loader = get_model_loader(stock_config, model_config)
        except TypeError:  # older get_model_loader(load_config) signature
            stock_loader = get_model_loader(stock_config)
        model = stock_loader.load_model(model_config=model_config,
                                        device_config=device_config)
        print(f"[flashboot][fallback] rank{global_shard_rank()} stock "
              f"{fallback_format} load OK — serving continues (no arena publish)",
              flush=True)
        return model

    @classmethod
    def _assert_architecture(cls, model_config: ModelConfig) -> None:
        architectures = list(getattr(model_config.hf_config, "architectures", None) or [])
        assert any(a in cls.SUPPORTED_ARCHITECTURES for a in architectures), (
            f"flashboot sharded loaders only support {cls.SUPPORTED_ARCHITECTURES}, "
            f"got architectures={architectures}. Use --load-format auto for other models."
        )

    # ── model construction ─────────────────────────────────────────────────────
    def _build_and_process_model(self, model_config: ModelConfig,
                                 target: torch.device) -> nn.Module:
        """device-init the model + run process_weights_after_loading, identical to stock
        ShardedStateLoader order. process runs on garbage init weights (the arena fill
        overwrites the values) and never runs again. Caller holds the
        set_default_torch_dtype context."""
        quant_config = _get_quantization_config(model_config, self.load_config)
        with target:
            model = _initialize_model(model_config, self.load_config, quant_config)
            for _, module in model.named_modules():
                quant_method = getattr(module, "quant_method", None)
                if quant_method is not None:
                    quant_method.process_weights_after_loading(module)
        return model

    # ── arena allocation + rebinding ───────────────────────────────────────────
    def _allocate_file_arena(self, model: nn.Module, tensor_placements, total_bytes: int,
                             gpu: int, have_cuda: bool,
                             arena_requirements: ArenaRequirements = None):
        """Drop the device-init storage of saved params (peak stays ~1x) and allocate ONE
        arena in checkpoint file layout. Returns ``(rebinder, arena_handle)`` — the caller
        fills the arena (disk block-H2D on the seed / transport pull on the clone), then
        calls :meth:`_rebind_saved_tensors`.

        ``arena_requirements`` comes from the transport that will serve or fill this
        arena (:meth:`flashboot.transport.contracts.Transport.arena_requirements`) — it
        is how "a fabric handle needs CUDA-VMM memory" reaches the allocation without
        this method knowing the word fabric.

        The three steps themselves live in :mod:`flashboot.model_binding`, which the
        ModelExpress front end drives directly."""
        arena_requirements = arena_requirements or ArenaRequirements()
        rebinder = model_binding.rebinder_for(model, tensor_placements)
        model_binding.drop_saved_parameter_storage(
            model, tensor_placements, torch.device("cuda", gpu), have_cuda)
        arena_handle = allocate_device_arena(
            total_bytes, gpu, arena_requirements.fabric_exportable)
        if os.getenv("FB_TEST_FAIL_AFTER_ALLOC") == "1":
            # Fault-injection hook for the fallback tests: fail at the point where the
            # most GPU memory is at stake (arena allocated, nothing filled yet), to
            # prove _release_partial_load really frees it before the stock reload.
            raise RuntimeError("[flashboot][test] injected failure after arena "
                               "allocation (FB_TEST_FAIL_AFTER_ALLOC=1)")
        return rebinder, arena_handle

    def _rebind_saved_tensors(self, model: nn.Module, rebinder: ArenaRebinder,
                              arena_handle: ArenaHandle, tensor_placements) -> None:
        """Point every saved param/buffer at its arena address (``set_``, keeps Parameter
        identity), redirect stale references, mark the already-applied post-load
        transforms done, and stash the arena_handle on the model (keepalive: the arena
        must outlive the views). The implementation is
        :func:`flashboot.model_binding.bind_placements`, shared with the ModelExpress
        front end."""
        model_binding.bind_placements(model, rebinder, arena_handle, tensor_placements)

    def _assert_arena_contiguity(self, model: nn.Module, arena_handle: ArenaHandle,
                                 tag: str, topology: ProcessTopology) -> None:
        """rank0 invariant: every bound param/buffer's data_ptr lies inside the arena.
        Raises if any escaped (a post-load step reallocated a weight out of the
        contiguous block)."""
        if topology.rank != 0 or not arena_handle.layout:
            return
        model_binding.assert_bound_tensors_live_in_the_arena(model, arena_handle, tag)

    # ── weight preload (seed only) ─────────────────────────────────────────────
    @staticmethod
    def _wait_preloaded_shards(staged_dir: str, shard_rank: int) -> None:
        """Block until the background stager (:mod:`flashboot.weight_preload`) has
        finished this rank's shards, signalled by the ``.rank{r}.ready`` marker (touched
        only after an atomic rename, so the subsequent glob never races a partial file)."""
        marker = weight_preload.rank_ready_marker(staged_dir, shard_rank)
        failed = weight_preload.failed_marker(staged_dir)
        timeout = float(os.getenv(weight_preload.TIMEOUT_ENV, "900"))
        started = time.perf_counter()
        while not os.path.exists(marker):
            if os.path.exists(failed):
                with open(failed, errors="ignore") as f:
                    reason = f.read().strip()
                raise RuntimeError(
                    f"[flashboot] weight-preload stager died:\n{reason}")
            if time.perf_counter() - started > timeout:
                raise TimeoutError(
                    f"[flashboot] weight-preload timeout waiting for {marker}")
            time.sleep(0.2)
        print(f"[flashboot][weight-preload] rank{shard_rank} shards ready, waited "
              f"{time.perf_counter() - started:.1f}s", flush=True)

    # ── serving the arena to a clone ───────────────────────────────────────────
    def _serve_arena_safely(self, model: nn.Module, arena_handle: ArenaHandle,
                            topology: ProcessTopology,
                            model_config: ModelConfig) -> None:
        """Arena serving is best-effort: the seed's own weights are already loaded, so
        a serve failure (unusable NIC, bad transport, port conflict) must degrade to
        "this seed cannot feed clones" — never to a dead seed. Clones that fail to pull
        fall back to a disk load on their side (see load_model). FB_SERVE_STRICT=1
        restores raising."""
        try:
            self._serve_arena(model, arena_handle, topology, model_config)
        except Exception:
            if os.getenv("FB_SERVE_STRICT", "0") == "1":
                raise
            traceback.print_exc()
            print(f"[flashboot][serve] rank{global_shard_rank()} arena serve FAILED — "
                  f"seed continues inference WITHOUT serving clones "
                  f"(FB_SERVE_STRICT=1 to make this fatal)", flush=True)

    def _serve_arena(self, model: nn.Module, arena_handle: ArenaHandle,
                     topology: ProcessTopology, model_config: ModelConfig) -> None:
        """Publish this rank's arena on its own TCP rendezvous
        (``seed_port + global_shard_rank``) so a clone can pull it over the selected
        transport. The keepalive dict (server thread) is stashed on the model — it
        must live as long as the server."""
        transport = create_transport(topology.gpu)  # fail fast if it is unusable
        settings = CoordinationSettings.from_load_config(self.load_config)
        if not settings.seed_base_port:
            raise RuntimeError(
                "[flashboot][serve] arena serve needs a TCP port: set "
                "--model-loader-extra-config '{\"seed_port\":N}'")
        # Key the port on the GLOBAL shard rank (PP-aware): with pipeline parallelism the
        # same tp_rank recurs on every PP stage, so tp_rank alone would collide the stages.
        port = rendezvous.rank_server_port(
            settings.seed_base_port, global_shard_rank(), model_config)
        # What is published, and which kind of server publishes it, is the transport's
        # business. ``broadcast_world`` is not: it is how many clone INSTANCES start
        # together, which the deployment decides — 1 for a single clone, K > 1 for a
        # chain — so it is passed down rather than branched on here.
        service = transport.serve_arena(arena_handle, topology.gpu, port,
                                        settings.clone_instance_count, settings.transfer_timeout_seconds)
        # Both halves outlive the load: the server is a daemon thread, and its backing
        # (an exported handle's arena, a registered memory region, a queue pair) is READ
        # by clones for as long as they are pulling.
        model._flashboot_arena_service = service
        model._flashboot_control_server = service.server
