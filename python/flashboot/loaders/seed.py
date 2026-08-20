"""The SEED: fill the arena from this rank's own checkpoint, then offer it to others.

Deliberately mirrors stock ``ShardedStateLoader.load_model``'s ORDER, so it is
byte-correct by construction — the only insertion is the arena rebind between
``process_weights_after_loading`` and the tensor copy:

    device-init model   (rotary/activation buffers get their real computed values)
      -> process_weights_after_loading           (final param shapes, like stock)
      -> allocate ONE aligned file-order arena (drop the device-init param storage first)
      -> fill: read each checkpoint copy run -> pinned host -> staged copy into the arena
      -> rebind every saved param/buffer onto its arena view (set_, zero-copy)
      -> _post_load_weights                      (derive w_kc/w_vc OUTSIDE the arena)
      -> optionally publish the arena (FB_FLASHSHARDED_SERVE=1).

Why device-init and not meta-init: GLM rotary caches and activation scales are
``register_buffer(persistent=False)`` — absent from the checkpoint AND not recomputed by
post_load — so meta-init would leave them on the meta device. device-init computes them.
"""
from __future__ import annotations

import os
import time
from typing import TYPE_CHECKING

import torch
from torch import nn

from sglang.srt.model_loader.loader import _post_load_weights
from sglang.srt.model_loader.utils import set_default_torch_dtype

from flashboot import weight_preload
from flashboot.arena import ArenaHandle
from flashboot.arena_fill import fill_arena_from_checkpoint
from flashboot.utils.checkpoint_layout import rank_part_files, read_checkpoint_layout
from flashboot.loaders.arena_loader_base import ShardedArenaLoaderBase
from flashboot.loaders.process_topology import ProcessTopology, global_shard_rank
from flashboot.loaders.split_fill import PENDING_SPLIT_FILLS
from flashboot.transport import ArenaRequirements, create_transport

if TYPE_CHECKING:
    from sglang.srt.configs.device_config import DeviceConfig
    from sglang.srt.configs.model_config import ModelConfig

class ShardedStateArenaLoader(ShardedArenaLoaderBase):
    """SEED: device-init -> process_weights -> allocate file-layout arena -> fill it from
    this rank's sharded_state safetensors -> rebind -> post_load -> optionally serve.

    Byte-identical to stock ShardedStateLoader (same op order) except the weights live in
    one contiguous arena instead of scattered storage."""

    def _load_flash_model(self, *, model_config: ModelConfig,
                          device_config: DeviceConfig) -> nn.Module:
        self._assert_architecture(model_config)
        topology = ProcessTopology()
        shard_rank = global_shard_rank()
        target = torch.device(device_config.device)
        have_cuda = torch.cuda.is_available()
        split = self.split_fill
        # With background weight preloading (FB_WEIGHT_PRELOAD=1) the shards are staged to
        # tmpfs; normally we wait here, but with split fill that wait moves to
        # run_split_fill (end of server startup) and the layout is read from the
        # original checkpoint headers (always present, cheap).
        if weight_preload.is_enabled() and not split:
            shard_dir = weight_preload.preload_dir()
            self._wait_preloaded_shards(shard_dir, shard_rank)
        else:
            shard_dir = model_config.model_path
        files = rank_part_files(shard_dir, shard_rank)

        # A transport can have requirements of the ALLOCATION itself (an NVLink-fabric
        # handle is exportable only from CUDA-VMM memory), so when this seed will serve,
        # resolve its transport before allocating and ask what it needs.
        arena_requirements = (create_transport(topology.gpu).arena_requirements()
                              if self.serve_enabled else ArenaRequirements())

        with set_default_torch_dtype(model_config.dtype):
            model = self._build_and_process_model(model_config, target)
            tensor_placements, copy_runs, total_bytes = read_checkpoint_layout(files)
            rebinder, arena_handle = self._allocate_file_arena(
                model, tensor_placements, total_bytes, topology.gpu, have_cuda,
                arena_requirements)
            if split:
                # Placeholder arena: fill a finite non-zero byte pattern (avoids zero fp8
                # scales in the dummy post_load), rebind, and run post_load ONCE so
                # derived buffers get fixed addresses for cuda-graph capture. The real
                # read+H2D and a second in-place post_load run at the end of server
                # startup — see _arm_split_fill / run_split_fill.
                arena_handle.byte_view.fill_(1)
                self._rebind_saved_tensors(model, rebinder, arena_handle, tensor_placements)
                _post_load_weights(model)
                self._arm_split_fill(model, arena_handle, shard_rank, topology, have_cuda,
                                     model_config)
            else:
                disk_started = time.perf_counter()
                fill_arena_from_checkpoint(copy_runs, arena_handle.base,
                                           topology.gpu, have_cuda, self.fill_config)
                # The DP split (role sharded_dp) exists to cut exactly this number by
                # having K replicas read 1/K each, and it prints the same line — without
                # one here there is nothing to compare it against.
                print(f"[flashboot][seed] shard{shard_rank}: {total_bytes / 1e9:.1f}GB "
                      f"read from disk in {time.perf_counter() - disk_started:.1f}s",
                      flush=True)
                if weight_preload.is_enabled():
                    # shards are in the GPU arena now; let the stager free the tmpfs copy
                    weight_preload.mark_consumed(shard_dir, shard_rank)
                self._rebind_saved_tensors(model, rebinder, arena_handle, tensor_placements)
                # post_load (MLA w_kc/w_vc absorb OUTSIDE the arena) — same as stock. No
                # second process_weights (it already ran before the fill), so the arena
                # bytes serve as-is and no skip flags are needed on the seed.
                _post_load_weights(model)

        self._assert_arena_contiguity(model, arena_handle, "after-post-load", topology)
        if have_cuda:
            torch.cuda.synchronize(topology.gpu)
        if self.serve_enabled and not split:
            # With split fill the arena still holds the placeholder here; serving now
            # would let a clone pull garbage. The finalize serves after the real
            # fill instead — the clone's connect retries until the port opens, so the
            # pull is ordered after the fill by construction.
            self._serve_arena_safely(model, arena_handle, topology, model_config)
        if have_cuda:
            torch.cuda.empty_cache()
        return model.eval()

    def _arm_split_fill(self, model: nn.Module, arena_handle: ArenaHandle,
                        shard_rank: int, topology: ProcessTopology, have_cuda: bool,
                        model_config: ModelConfig) -> None:
        """Register the split fill (real read+H2D + a second, in-place post_load) for
        :func:`run_split_fill`, which the server calls at the very end of startup. By
        then the background prestager has usually finished staging to /dev/shm, so the
        staging wait is hidden behind the whole startup. When serving to a clone
        (FB_FLASHSHARDED_SERVE=1) the serve also happens HERE, after the fill — never
        before, or the clone would pull the placeholder bytes."""
        loader = self

        def finalize() -> None:
            started = time.perf_counter()
            if weight_preload.is_enabled():
                fill_dir = weight_preload.preload_dir()
                ready = os.path.exists(
                    weight_preload.rank_ready_marker(fill_dir, shard_rank))
                print(f"[flashboot][split] rank{shard_rank} finalize: /dev/shm preload "
                      f"{'complete' if ready else 'still staging -> waiting'}", flush=True)
                if not ready:
                    loader._wait_preloaded_shards(fill_dir, shard_rank)
            else:
                fill_dir = model_config.model_path
            files = rank_part_files(fill_dir, shard_rank)
            _fill_placements, copy_runs, _total = read_checkpoint_layout(files)
            fill_arena_from_checkpoint(copy_runs, arena_handle.base,
                                       topology.gpu, have_cuda, loader.fill_config)
            if weight_preload.is_enabled():
                weight_preload.mark_consumed(fill_dir, shard_rank)
            # Re-derive w_kc/w_vc from the now-real kv_b_proj; bind_or_assign does an
            # in-place copy_ into the buffers capture already recorded, so addresses hold.
            _post_load_weights(model)
            if have_cuda:
                torch.cuda.synchronize(topology.gpu)
            print(f"[flashboot][split] rank{shard_rank} split fill done in "
                  f"{time.perf_counter() - started:.1f}s", flush=True)
            if loader.serve_enabled:
                loader._serve_arena_safely(model, arena_handle, topology, model_config)
                print(f"[flashboot][split] rank{shard_rank} arena served post-fill",
                      flush=True)

        PENDING_SPLIT_FILLS.append(finalize)
