"""The DP-COOPERATIVE load: K replicas each read 1/K of the checkpoint, then gather.

Each tensor-parallel rank exists once per data-parallel replica, and all K copies of rank
r need byte-identical arenas. So replica d fills only its native segment of rank r's
arena from disk — the fill pipeline is fed just the copy runs clipped to that byte range,
so it reads exactly its share — and one all-gather then makes every replica whole. One
full load's disk bandwidth split K ways, plus one segment transfer at link rate.

The result is byte-identical to a full disk load by construction: the segment boundaries
are computed from the same shared math on every replica, and the alignment they were cut
on rides the membership, so two replicas that disagree fail loudly instead of gathering
shifted segments.
"""
from __future__ import annotations

import time
from typing import TYPE_CHECKING

import torch
from torch import nn

from sglang.srt.model_loader.loader import _post_load_weights
from sglang.srt.model_loader.utils import set_default_torch_dtype

from flashboot import rendezvous, weight_preload
from flashboot.arena_fill import fill_arena_from_checkpoint
from flashboot.utils.checkpoint_layout import (
    clip_copy_runs,
    rank_part_files,
    read_checkpoint_layout,
)
from flashboot.collectives import all_gather
from flashboot.utils.arena_partitioning import (
    data_parallel_gather_group,
    segment_bounds,
)
from flashboot.loaders.arena_loader_base import ShardedArenaLoaderBase
from flashboot.loaders.coordination import CoordinationSettings
from flashboot.loaders.process_topology import ProcessTopology, global_shard_rank
from flashboot.transport import create_transport

if TYPE_CHECKING:
    from sglang.srt.configs.device_config import DeviceConfig
    from sglang.srt.configs.model_config import ModelConfig

class ShardedStateDpLoader(ShardedArenaLoaderBase):
    """DP-COOPERATIVE disk load (role ``sharded_dp``): K data-parallel replica
    INSTANCES of the same TP layout split every rank's disk read K ways and merge
    over the selected transport's all-gather.

    Each tp rank r exists once per replica; replica d fills only its NATIVE segment
    ``segment_bounds(d, K, total_bytes)`` of rank r's file-layout arena from
    disk — the fill pipeline is fed just the copy runs clipped to that range
    (:func:`flashboot.utils.checkpoint_layout.clip_copy_runs`), so it reads exactly its
    share of the checkpoint bytes. The K replicas of rank r then run ONE world=K
    all-gather (direct mode — the rendezvous is the readiness barrier) so every
    replica holds the full arena: one full load's disk bandwidth split K ways, plus
    one segment transfer at link rate. Rebind + post_load then proceed exactly as
    the seed's — the arena bytes are byte-identical to a full disk load by
    construction.

    The gather runs through :func:`flashboot.collectives.all_gather.all_gather`, one
    implementation for every transport: the transport supplies this replica's half
    (:meth:`flashboot.transport.contracts.Transport.join_all_gather`) and moves the
    bytes, while the ordering — join, rendezvous, connect, gather — is the same either
    way. A transport with requirements of the ALLOCATION (fabric needs CUDA-VMM memory so
    its handle can be exported) states them through ``arena_requirements``.

    Rendezvous addressing reuses the per-rank port scheme: replica 0 hosts the
    collective rendezvous server on ``seed_port + shard_rank`` and joins it over
    loopback (the peer-less server shape — see
    :func:`flashboot.rendezvous.start_dp_gather_server`); replicas > 0 dial
    ``seed_ip`` (replica 0's node). The group key
    (:func:`flashboot.utils.arena_partitioning.data_parallel_gather_group`) guards against a replica
    dialing the wrong port band.

    Activation (env fallbacks FB_DP_WORLD / FB_DP_RANK / SGLANG_FABRIC_SEED_IP /
    SGLANG_FABRIC_SEED_PORT mirror the other roles):

        --load-format flashload --model-loader-extra-config \\
            '{"role":"sharded_dp","dp_world":K,"dp_rank":d,
              "seed_ip":"<replica-0 ip>","seed_port":N}'
    """

    def _load_flash_model(self, *, model_config: ModelConfig,
                          device_config: DeviceConfig) -> nn.Module:
        self._assert_architecture(model_config)
        if self.split_fill:
            raise RuntimeError(
                "[flashboot][dp] FB_SPLIT_FILL is incompatible with sharded_dp: the "
                "all-gather needs every native segment's real bytes before the "
                "rendezvous, which split fill defers past startup")
        topology = ProcessTopology()
        shard_rank = global_shard_rank()
        target = torch.device(device_config.device)
        have_cuda = torch.cuda.is_available()
        settings = CoordinationSettings.from_load_config(self.load_config)
        self._validate_replica_settings(settings)
        # Same preload interplay as the seed: the stager stages whole shard files,
        # the clipped fill simply reads its share of them.
        if weight_preload.is_enabled():
            shard_dir = weight_preload.preload_dir()
            self._wait_preloaded_shards(shard_dir, shard_rank)
        else:
            shard_dir = model_config.model_path
        files = rank_part_files(shard_dir, shard_rank)

        # The transport is resolved BEFORE the arena allocation, because it can have
        # requirements of the allocation itself (a fabric arena must be CUDA-VMM for its
        # handle to be exportable). Which all-gather engine that implies is the
        # transport's business, not this loader's.
        transport = create_transport(topology.gpu)
        is_draft = bool(getattr(model_config, "is_draft_model", False))
        port = rendezvous.rank_server_port(settings.seed_base_port, shard_rank,
                                              model_config)
        group = data_parallel_gather_group(shard_rank, is_draft)
        with set_default_torch_dtype(model_config.dtype):
            model = self._build_and_process_model(model_config, target)
            tensor_placements, copy_runs, total_bytes = read_checkpoint_layout(files)
            rebinder, arena_handle = self._allocate_file_arena(
                model, tensor_placements, total_bytes, topology.gpu, have_cuda,
                transport.arena_requirements())
            # Replica 0 hosts the rendezvous BEFORE its fill: the port is open the
            # moment any replica finishes its disk read, so nobody burns connect
            # patience on replica 0's own fill — early registrations just wait in the
            # round (registering, and thereby declaring the segment ready, still
            # happens strictly after the local fill on every replica).
            if settings.replica_index == 0:
                model._flashboot_control_server = \
                    rendezvous.start_dp_gather_server(
                        port, world=settings.replica_count, timeout_s=settings.transfer_timeout_seconds)
                host = "127.0.0.1"
            else:
                host = rendezvous.seed_host_for_rank(settings.seed_host_list, shard_rank,
                                                        topology.world)
            # Segment split, on exact byte boundaries. Nothing below wants them
            # rounded: clip_copy_runs cuts a run anywhere and shifts its file offset
            # with it, plan_fill_chunks derives its span from the runs it is handed,
            # and an RDMA READ takes any offset on either end. Rounding only ever
            # moved bytes onto one replica as disk the others then waited for.
            start, end = segment_bounds(
                settings.replica_index, settings.replica_count, total_bytes)
            native_runs = clip_copy_runs(copy_runs, start, end)
            disk_started = time.perf_counter()
            if native_runs:  # a segment can still land entirely on arena padding
                fill_arena_from_checkpoint(native_runs, arena_handle.base,
                                           topology.gpu, have_cuda, self.fill_config)
            disk_seconds = time.perf_counter() - disk_started
            if weight_preload.is_enabled():
                weight_preload.mark_consumed(shard_dir, shard_rank)
            print(f"[flashboot][dp] shard{shard_rank} replica "
                  f"{settings.replica_index}/{settings.replica_count}: native segment "
                  f"[{start}, {end}) = {(end - start) / 1e9:.1f}GB of "
                  f"{total_bytes / 1e9:.1f}GB read from disk in {disk_seconds:.1f}s",
                  flush=True)
            # One all-gather, whatever the transport: join (which declares this
            # replica's segment ready), rendezvous, connect, gather.
            participant = all_gather.all_gather(
                transport, arena_handle, topology.gpu, rank=settings.replica_index,
                world=settings.replica_count, rendezvous_host=host, rendezvous_port=port,
                group=group, segment_alignment=1,
                timeout_s=settings.transfer_timeout_seconds,
                connect_timeout_s=settings.connect_timeout_seconds)
            # Peers may still be reading our native segment after our own gather
            # returned (the concurrent mode has no end barrier), so whatever backs
            # our publication lives as long as the model.
            model._flashboot_gather_participant = participant
            self._rebind_saved_tensors(model, rebinder, arena_handle,
                                       tensor_placements)
            _post_load_weights(model)

        self._assert_arena_contiguity(model, arena_handle, "dp-after-post-load",
                                      topology)
        if have_cuda:
            torch.cuda.synchronize(topology.gpu)
        if self.serve_enabled:
            # The gathered arena is a full post-process image — serving it to
            # clones works exactly as on the seed.
            self._serve_arena_safely(model, arena_handle, topology, model_config)
        if have_cuda:
            torch.cuda.empty_cache()
        return model.eval()

    @staticmethod
    def _validate_replica_settings(settings: CoordinationSettings) -> None:
        if settings.replica_count < 2:
            raise RuntimeError(
                f"[flashboot][dp] sharded_dp needs the replica count: set "
                f"--model-loader-extra-config '{{\"dp_world\":K}}' with K >= 2 "
                f"(or FB_DP_WORLD), got {settings.replica_count}")
        if not (0 <= settings.replica_index < settings.replica_count):
            raise RuntimeError(
                f"[flashboot][dp] sharded_dp needs this replica's fixed position: "
                f"set --model-loader-extra-config '{{\"dp_rank\":d}}' with "
                f"0 <= d < dp_world={settings.replica_count} (or FB_DP_RANK), got "
                f"{settings.replica_index}")
        if not settings.seed_base_port:
            raise RuntimeError(
                "[flashboot][dp] sharded_dp needs the rendezvous port: set "
                "--model-loader-extra-config '{\"seed_port\":N}' "
                "(or SGLANG_FABRIC_SEED_PORT)")
        if settings.replica_index > 0 and not settings.seed_host_list:
            raise RuntimeError(
                "[flashboot][dp] replicas > 0 need replica 0's address: set "
                "--model-loader-extra-config '{\"seed_ip\":\"<ip>\"}' "
                "(or SGLANG_FABRIC_SEED_IP)")
