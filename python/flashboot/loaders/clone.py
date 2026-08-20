"""The CLONE: build the same arena, then take its contents from an instance that has them.

The arena layout is deterministic from this rank's own part headers, so a clone needs no
weight read at all to know exactly how big its arena is and where every tensor goes. It
then reads another instance's arena into its own — one whole-arena transfer for a single
clone, or a chunk-pipelined chain when several instances start together — and rebinds.

The rebind marks every already-applied post-load transform as done (see
``mark_quant_weights_finalized`` and ``fixup_deepseek_arena_state`` in
:mod:`flashboot.model_binding`),
because the arena already holds post-process bytes and re-transforming them would be
silently wrong. Then w_kc/w_vc are re-derived exactly as on the seed -> byte-correct.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from torch import nn

from sglang.srt.model_loader.loader import _post_load_weights
from sglang.srt.model_loader.utils import set_default_torch_dtype

from flashboot import rendezvous
from flashboot.utils.checkpoint_layout import rank_part_files, read_checkpoint_layout
from flashboot.loaders.arena_loader_base import ShardedArenaLoaderBase
from flashboot.loaders.coordination import CoordinationSettings
from flashboot.loaders.process_topology import ProcessTopology, global_shard_rank
from flashboot.transport import create_transport

if TYPE_CHECKING:
    from sglang.srt.configs.device_config import DeviceConfig
    from sglang.srt.configs.model_config import ModelConfig

class ShardedStateCloneLoader(ShardedArenaLoaderBase):
    """CLONE: build the SAME file-layout arena (deterministic from this rank's own part
    headers — no weight read), pull the seed's published arena into it over NVLink, then
    finalize (re-derives w_kc/w_vc identically -> byte-correct)."""

    def _load_flash_model(self, *, model_config: ModelConfig,
                          device_config: DeviceConfig) -> nn.Module:
        self._assert_architecture(model_config)
        topology = ProcessTopology()
        shard_rank = global_shard_rank()
        target = torch.device(device_config.device)
        have_cuda = torch.cuda.is_available()
        settings = CoordinationSettings.from_load_config(self.load_config)
        # Both file selection AND the rendezvous port key on the global shard rank
        # (PP-aware): clone shard rank j pairs with seed shard rank j, so each (PP, TP)
        # shard connects to the seed process that holds the same shard.
        files = rank_part_files(model_config.model_path, shard_rank)

        # A 1-to-1 rdma clone no longer goes through the chain: it asks the seed's
        # advertisement server for a wire pull (one round trip, one whole-arena READ).
        if settings.clone_instance_count > 1:
            return self._clone_via_chain(settings, files, model_config, target,
                                         have_cuda, topology, shard_rank)

        # 1. OPEN the seed's arena. Deliberately before the model build: for the handle
        #    family this is where the mapping is made, and it costs seconds on a live
        #    serving GPU that cannot be overlapped in-process — so it hides behind the
        #    build that follows instead.
        seed_arena = self._open_seed_arena(settings, shard_rank, topology, model_config)

        try:
            return self._clone_from(seed_arena, files, model_config, target,
                                    have_cuda, topology)
        except Exception:
            try:
                seed_arena.close()
            except Exception:  # noqa: BLE001 — teardown must not mask the real failure
                pass
            raise

    def _clone_from(self, seed_arena, files, model_config: ModelConfig,
                    target: torch.device, have_cuda: bool,
                    topology: ProcessTopology) -> nn.Module:
        with set_default_torch_dtype(model_config.dtype):
            model = self._build_and_process_model(model_config, target)
            # 2. the same file-layout arena, deterministic from this rank's part headers.
            tensor_placements, _copy_runs, total_bytes = read_checkpoint_layout(files)
            rebinder, arena_handle = self._allocate_file_arena(
                model, tensor_placements, total_bytes, topology.gpu, have_cuda)
            # 3. drift guard: the clone arena MUST match the seed's published size.
            if seed_arena.byte_count and arena_handle.size != seed_arena.byte_count:
                raise RuntimeError(
                    f"[flashboot][clone] arena drift: clone size={arena_handle.size} != seed "
                    f"size={seed_arena.byte_count} — seed and clone must load the same "
                    f"checkpoint with identical flags")
            # 4. Read the seed's whole post-process arena into ours. How that happens —
            #    one copy-engine copy from an imported handle, or one-sided READs over the
            #    network — is the transport's business, including its own fallback.
            seed_arena.fetch_whole_arena_into(arena_handle)
            # 5. rebind (same offsets the seed used) — _rebind_saved_tensors marks all
            #    already-applied post-load transforms done (mark_quant_weights_finalized
            #    + fixup_deepseek_arena_state: fp8/expert flags + ape hotfix + correction
            #    bias), so the arena's post-process bytes are not re-transformed — then
            #    derive w_kc/w_vc from kv_b_proj.
            self._rebind_saved_tensors(model, rebinder, arena_handle, tensor_placements)
            _post_load_weights(model)

        self._assert_arena_contiguity(model, arena_handle, "clone-after-post-load", topology)
        if have_cuda:
            torch.cuda.synchronize(topology.gpu)
            torch.cuda.empty_cache()
        return model.eval()

    def _clone_via_chain(self, settings: CoordinationSettings, files,
                         model_config: ModelConfig, target: torch.device,
                         have_cuda: bool, topology: ProcessTopology,
                         shard_rank: int) -> nn.Module:
        """Chain-broadcast load — ``broadcast_world`` clone instances starting together,
        every hop streaming CONCURRENTLY rather than waiting for its upstream to finish.

        The arena is allocated before this instance takes its place in the chain, because
        taking that place is what publishes the arena to our successor. What "publishing"
        means, how a hop learns its upstream's chunk k is valid, and whether our position
        is assigned by arrival or pinned by the launcher are all the transport's business
        (see :class:`flashboot.transport.contracts.ChainParticipant`) — the two families
        differ in exactly those things and in nothing else here.
        """
        transport = create_transport(topology.gpu)
        host = rendezvous.seed_host_for_rank(settings.seed_host_list, shard_rank,
                                               topology.world)
        port = rendezvous.rank_server_port(settings.seed_base_port, shard_rank,
                                              model_config)
        with set_default_torch_dtype(model_config.dtype):
            model = self._build_and_process_model(model_config, target)
            tensor_placements, _copy_runs, total_bytes = read_checkpoint_layout(files)
            rebinder, arena_handle = self._allocate_file_arena(
                model, tensor_placements, total_bytes, topology.gpu, have_cuda,
                transport.arena_requirements())
            participant = transport.join_chain(arena_handle, topology.gpu,
                                               settings.clone_instance_count,
                                               settings.chain_position)
            # Registers at the seed and BLOCKS until every instance has, then fills our
            # arena from our upstream while publishing our own readiness per chunk.
            participant.receive_broadcast(host, port, settings.transfer_timeout_seconds,
                                          connect_timeout_s=settings.connect_timeout_seconds)
            # Our successor keeps reading our arena and our readiness signal for the rest
            # of its pull, so whatever backs them lives as long as the model.
            model._flashboot_chain_participant = participant
            self._rebind_saved_tensors(model, rebinder, arena_handle,
                                       tensor_placements)
            _post_load_weights(model)

        self._assert_arena_contiguity(model, arena_handle, "clone-after-post-load",
                                      topology)
        if have_cuda:
            torch.cuda.synchronize(topology.gpu)
            torch.cuda.empty_cache()
        return model.eval()

    def _open_seed_arena(self, settings: CoordinationSettings, shard_rank: int,
                         topology: ProcessTopology, model_config: ModelConfig):
        """Open the arena of the seed process holding global shard ``shard_rank``.

        ``seed_ip`` is a comma-separated per-node IP list in node-rank order; this rank's
        own world size (== the seed's, enforced by the drift guard) maps the global shard
        rank to the seed node that hosts it.

        The transport does the rest, including deciding what "open" costs and what to do
        when its preferred mechanism does not work here — a handle that cannot be
        imported falls through to the seed's wire offer inside the transport, so there is
        nothing to branch on at this level."""
        if not (settings.seed_host_list and settings.seed_base_port):
            raise RuntimeError(
                "[flashboot][clone] sharded_clone needs the seed address: set "
                "--model-loader-extra-config "
                "'{\"role\":\"sharded_clone\",\"seed_ip\":\"<ip0,ip1,...>\",\"seed_port\":N}'")
        host = rendezvous.seed_host_for_rank(settings.seed_host_list, shard_rank,
                                               topology.world)
        port = rendezvous.rank_server_port(settings.seed_base_port, shard_rank, model_config)
        return create_transport(topology.gpu).open_seed_arena(
            host, port, topology.gpu, timeout_s=settings.transfer_timeout_seconds,
            connect_timeout_s=settings.connect_timeout_seconds)
