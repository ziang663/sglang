"""All-gather over the HANDLE transports (ipc / fabric) — import + copy-engine D2D.

The rdma all-gather's direct-mode narrative (:mod:`flashboot.transport.rdma_data_plane`:
every rank holds its 1/K native segment, the one-round rendezvous is the readiness
barrier, then each rank fetches every other owner's segment) carried to the
transports where a peer's arena can simply be MAPPED: instead of QP cards and
one-sided READs, each member's registration carries its arena's 64-byte shareable memory
handle (the same :class:`flashboot.rendezvous.ArenaAdvertisement` wire the clone
advertisement uses), and the data plane imports each owner's arena
(``flashboot._C.PeerArenaImporter``) and copies that owner's ``segment_bounds``
range with the copy engine (``copy_from_peer`` — NVLink D2D), then drops the
mapping (the owner keeps its memory; late importers stay safe because a gathered
segment is never written again).

Only DIRECT mode exists here, deliberately:

  * ring exists for fabrics that cannot carry concurrent all-to-all flows; the
    NVLink switch fabric can, at line rate, so ring buys no bandwidth;
  * ring also needs a remote progress primitive to gate step s on the neighbour's
    step s-1 — handles have none (the chain builds one from imported flag buffers
    and ``cuStreamWaitValue32``; ring would need that machinery for no gain).

Rendezvous, group keys and segment math are shared with the rdma engine — the
rendezvous through :func:`flashboot.rendezvous.collective_rendezvous`, the rest
through :mod:`flashboot.utils.arena_partitioning`, which belongs to NEITHER engine; a
membership is rejected on size/alignment/transport drift exactly like the rdma one. Unlike rdma there is
nothing to keep alive after the gather beyond the arena itself: the exported
handle stays valid as long as the arena lives.
"""
from __future__ import annotations

import dataclasses
import time

from flashboot import rendezvous
from flashboot.utils.arena_partitioning import segment_bounds

NVLINK_TRANSPORTS = ("ipc", "fabric")


def _new_peer_importer():
    """The native handle-import engine, deferred so this module (and its tests)
    stays importable without the extension/driver."""
    from flashboot import _C

    return _C.PeerArenaImporter()


@dataclasses.dataclass
class NvlinkGatherMembership:
    """One member's half of the handle all-gather: where its arena lives locally
    and the wire ``info`` for the rendezvous registration (the ArenaAdvertisement dict —
    arena size + shareable handle — plus the collective keys). The ARENA backs the
    exported handle: keep it alive until every peer's gather completed."""
    arena_ptr: int
    arena_size: int
    rank: int
    world: int
    transport: str
    device: int
    info: dict


def prepare_allgather_membership(arena_handle, transport: str, device: int,
                                 rank: int, world: int,
                                 segment_alignment: int = 1,
                                 extra_info: dict = None) -> NvlinkGatherMembership:
    """Export the arena's shareable handle and build the complete rendezvous registration
    (the membership broadcast is the last rendezvous message needed, exactly
    like the rdma engine). ``arena_handle`` is the loader's
    :class:`flashboot.arena.ArenaHandle` — for ``fabric`` its arena must have been
    allocated ``fabric_exportable`` (CUDA-VMM); ``ipc`` exports plain cudaMalloc.
    ``extra_info`` keys ride the wire verbatim (free-form telemetry)."""
    if transport not in NVLINK_TRANSPORTS:
        raise ValueError(f"[flashboot][gather] transport {transport!r} is not a "
                         f"handle transport (expected one of {NVLINK_TRANSPORTS}; "
                         f"rdma runs flashboot.transport.rdma_data_plane)")
    info = dict(extra_info or {})
    info.update(rendezvous.ArenaAdvertisement(
        arena_size=int(arena_handle.size), device=int(device), transport=transport,
        memory_handle=arena_handle.arena.export_shareable_handle(transport).hex(),
    ).to_wire())
    info.update(rank=int(rank), world=int(world), allgather_mode="direct",
                segment_alignment=max(1, int(segment_alignment)))
    return NvlinkGatherMembership(
        arena_ptr=int(arena_handle.base), arena_size=int(arena_handle.size),
        rank=int(rank), world=int(world), transport=transport, device=int(device),
        info=info)


def all_gather(membership: NvlinkGatherMembership, members: list,
               timeout_s: float = 0.0) -> None:
    """Make this rank hold the FULL arena from the per-rank native segments: for
    every other owner, import its arena handle, D2D-copy its ``segment_bounds``
    range into the same offsets of ours, release the mapping. Blocking; every step
    is a native driver call, so ``timeout_s`` is accepted only for interface
    symmetry with the rdma engine (the driver bounds each call, not a poll
    deadline). The rendezvous already proved every owner's segment valid."""
    sizes = {int(m["arena_size"]) for m in members}
    if sizes != {membership.arena_size}:
        raise RuntimeError(f"[flashboot][gather] arena drift across the all-gather "
                           f"membership: sizes {sorted(sizes)}")
    alignments = {int(m.get("segment_alignment", 1)) for m in members}
    if len(alignments) != 1:
        raise RuntimeError(f"[flashboot][gather] segment alignment drift across "
                           f"the all-gather membership: {sorted(alignments)} — "
                           f"every rank must split the arena identically")
    transports = {str(m.get("transport", "")) for m in members}
    if transports != {membership.transport}:
        raise RuntimeError(f"[flashboot][gather] transport drift across the "
                           f"all-gather membership: {sorted(transports)}")
    # The handles are imported field-by-field rather than through
    # ArenaAdvertisement.from_wire, so the wire-version check that path performs has to
    # happen here too: a mixed-build membership imports fine and yields wrong weights.
    versions = {int(m.get("version", -1)) for m in members}
    if versions != {rendezvous.ADVERTISEMENT_VERSION}:
        raise RuntimeError(f"[flashboot][gather] flashboot wire version drift across "
                           f"the all-gather membership: {sorted(versions)} against this "
                           f"process's {rendezvous.ADVERTISEMENT_VERSION} — every "
                           f"member must run the same build")
    alignment = alignments.pop()
    by_rank = {int(m["rank"]): m for m in members}
    started = time.perf_counter()
    import_seconds = 0.0
    for owner in range(membership.world):
        if owner == membership.rank:
            continue
        start, end = segment_bounds(owner, membership.world,
                                    membership.arena_size, alignment)
        if start >= end:
            continue  # a huge alignment starved this owner's segment to empty
        source = by_rank[owner]
        importer = _new_peer_importer()
        try:
            import_started = time.perf_counter()
            importer.import_handle(
                membership.transport, bytes.fromhex(source["credential"]["memory_handle"]),
                int(source["arena_size"]), membership.device)
            import_seconds += time.perf_counter() - import_started
            # copy_from_peer(local_dst, peer_src_offset, nbytes): same offsets on
            # both sides — the segment layout is identical by construction.
            importer.copy_from_peer(membership.arena_ptr + start, start,
                                    end - start)
        finally:
            try:
                importer.release()
            except Exception:  # noqa: BLE001 — teardown must not mask the copy error
                pass
    elapsed = time.perf_counter() - started
    gathered = membership.arena_size * (membership.world - 1) // membership.world
    print(f"[flashboot][gather] all-gather rank {membership.rank}: "
          f"{gathered / 1e9:.1f}GB gathered ({membership.transport}) in "
          f"{elapsed:.2f}s ({gathered / 1e9 / max(elapsed, 1e-9):.1f}GB/s ingress, "
          f"import={import_seconds:.2f}s, world={membership.world})", flush=True)
