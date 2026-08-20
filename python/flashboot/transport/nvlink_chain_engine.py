"""Pipelined chunked chain broadcast — the dedicated broadcast engine.

Ported from the GB300-verified main-branch design (RingReceiver): K clone instances
starting together form a chain ``seed -> clone0 -> ... -> cloneK-1`` in which every
clone pulls its predecessor's arena in fixed-size CHUNKS, gated by a per-chunk u32
flag array the predecessor publishes as each chunk lands (``flashboot._C.
ChainReceiver``: cuStreamWaitValue32 -> cudaMemcpyAsync -> cuStreamWriteValue32, all
hardware stream ops on the copy engine). Chunks therefore flow down the whole chain
concurrently — total time is ~arena_size/link_bw regardless of chain length, the
seed's egress carries exactly one stream, and no doorbell messages cross the
rendezvous during the transfer.

Split of responsibilities:
  * this module   — the per-clone data plane: flags allocation/zeroing, the
    registration payload, the import-and-pipelined-pull.
  * control — the rendezvous: collects the K registrations and releases every
    position AT ONCE (per-chunk flags provide the data-readiness ordering, so no
    sequential gating is needed; the seed's own advertisement carries no flags —
    position 0's hop runs without waits since the seed's bytes are always valid).
  * sharded_loader — orchestration (build model, allocate the arena, rebind).

Wire format: the chain registration is a normal ArenaAdvertisement wire dict plus
``{"flags_handle": <hex>, "num_chunks": N, "chunk_bytes": C}``. The seed forwards it
verbatim to the successor, so this module owns both ends of the extra keys.

Chunk geometry must be UNIFORM along the chain (a successor indexes its
predecessor's flags with its own chunk size); every instance reads the same
``FB_CHAIN_CHUNK_MB`` (default 256) and mismatches fail loudly at pull time.

Failure semantics: a clone dying mid-transfer stalls its successor on a flag that
never flips (hardware wait, no timeout) — the successor is then bounded only by its
outer load timeout. The chain trades the store-and-forward mode's per-link fallback
for full pipelining; run it with healthy instances.
"""
from __future__ import annotations

import dataclasses
import math
import time

from flashboot import _C, rendezvous
from flashboot.arena import ArenaHandle
from flashboot.utils.arena_partitioning import chain_chunk_bytes


@dataclasses.dataclass
class ChainMembership:
    """A clone's data-plane half of the chain registration: the zeroed per-chunk
    flags buffer (a native DeviceArena — its handle rides in ``registration``) and
    the wire payload for the rendezvous registration. KEEP the membership alive on the
    model: the successor reads our flags and arena for the duration of its pull."""
    flags_arena: object          # native _C.DeviceArena, num_chunks u32, zeroed
    registration: dict           # ArenaAdvertisement wire + flags/chunk keys
    chunk_bytes: int
    num_chunks: int


def prepare_membership(arena_handle: ArenaHandle, transport: str,
                       device: int) -> ChainMembership:
    """Allocate + ZERO the per-chunk flags buffer and build the registration
    payload. Zeroing must happen before the handle leaves this process: fresh
    device memory holds garbage, and any stale value >= 1 would release a chunk to
    the successor early."""
    chunk_bytes = chain_chunk_bytes()
    num_chunks = math.ceil(arena_handle.size / chunk_bytes)
    flags_arena = _C.DeviceArena()
    flags_arena.create(num_chunks * 4, device,
                       fabric_exportable=(transport == "fabric"))
    _C.chain_zero_flags(flags_arena.ptr(0), num_chunks)
    registration = rendezvous.ArenaAdvertisement(
        arena_size=int(arena_handle.size), device=int(device), transport=transport,
        chunk_bytes=chunk_bytes, num_chunks=num_chunks,
        credential={
            "memory_handle": arena_handle.arena.export_shareable_handle(transport).hex(),
            # Our successor waits on these per-chunk flags; the seed publishes none,
            # which is how position 1 knows its upstream's bytes are always valid.
            "flags_handle": flags_arena.export_shareable_handle(transport).hex(),
        }).to_wire()
    return ChainMembership(flags_arena=flags_arena, registration=registration,
                           chunk_bytes=chunk_bytes, num_chunks=num_chunks)


def pull_from_upstream(arena_handle: ArenaHandle, membership: ChainMembership,
                       upstream_wire: dict, device: int, position: int) -> None:
    """Import the upstream arena (+ flags, absent when the upstream is the seed)
    and run the pipelined pull into our arena, publishing our own flags as chunks
    land so our successor streams concurrently. Blocking until our whole arena is
    valid; the local mappings are dropped right after (the upstream keeps owning
    its memory, our successor keeps reading OUR arena/flags, which stay alive on
    the membership)."""
    upstream = rendezvous.ArenaAdvertisement.from_wire(upstream_wire)
    if upstream.arena_size != arena_handle.size:
        raise RuntimeError(
            f"[flashboot][chain] arena drift at position {position}: local "
            f"size={arena_handle.size} != upstream size={upstream.arena_size} — "
            f"every instance must load the same checkpoint with identical flags")
    upstream_chunk = upstream_wire.get("chunk_bytes")
    if upstream_chunk is not None and int(upstream_chunk) != membership.chunk_bytes:
        raise RuntimeError(
            f"[flashboot][chain] chunk mismatch at position {position}: ours "
            f"{membership.chunk_bytes} != upstream {upstream_chunk} — set the same "
            f"FB_CHAIN_CHUNK_MB on every instance")

    arena_importer = _C.PeerArenaImporter()
    flags_importer = None
    started = time.perf_counter()
    try:
        arena_importer.import_handle(
            upstream.transport, bytes.fromhex(upstream.credential["memory_handle"]),
            int(upstream.arena_size), device)
        upstream_flags_pointer = 0  # 0 = the seed: bytes always valid, no waits
        flags_handle = upstream_wire["credential"].get("flags_handle")
        if flags_handle:
            flags_importer = _C.PeerArenaImporter()
            flags_importer.import_handle(
                upstream.transport, bytes.fromhex(flags_handle),
                int(upstream_wire["num_chunks"]) * 4, device)
            upstream_flags_pointer = flags_importer.peer_arena_pointer(0)
        imported = time.perf_counter()

        receiver = _C.ChainReceiver(device)
        receiver.run(pred_arena=arena_importer.peer_arena_pointer(0),
                     pred_flags=upstream_flags_pointer,
                     my_arena=arena_handle.base,
                     my_flags=membership.flags_arena.ptr(0),
                     total_bytes=arena_handle.size,
                     chunk_bytes=membership.chunk_bytes)
        finished = time.perf_counter()
        print(f"[flashboot][chain] position {position}: "
              f"{arena_handle.size / 1e9:.1f}GB pipelined over "
              f"{upstream.transport} in {finished - imported:.2f}s "
              f"({arena_handle.size / 1e9 / max(finished - imported, 1e-9):.1f}GB/s, "
              f"{membership.num_chunks} chunks x {membership.chunk_bytes >> 20}MB, "
              f"import={imported - started:.2f}s, "
              f"{'flag-gated' if upstream_flags_pointer else 'seed-fed'})",
              flush=True)
    finally:
        for importer in (arena_importer, flags_importer):
            if importer is not None:
                try:
                    importer.release()
                except Exception:  # noqa: BLE001 — teardown must not mask the pull error
                    pass
