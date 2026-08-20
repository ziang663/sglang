"""RDMA chain broadcast + all-gather — the cross-node collectives engine.

The endpoint underneath comes from the transport layer
(:func:`flashboot.transport.rdma_engine.create_endpoint`): which one-sided-READ
engine this process runs is a transport decision, and everything here works the
same on either.

The rdma sibling of :mod:`flashboot.transport.nvlink_chain_engine`: same chain narrative (seed ->
clone1 -> ... -> cloneK, every clone pulls its predecessor's arena in fixed-size
chunks, all hops streaming concurrently), but over one-sided RDMA READ instead of
NVLink handle imports — for fabrics with no IMEX domain (H100 + InfiniBand). Plus an
all-gather for the flashload_dp pattern: every rank loads 1/K of the arena from disk,
then the collective makes everyone hold the full arena.

Control plane — ONE collection round (:func:`rendezvous.collective_rendezvous`):
every member sends its complete rdma info (rank, arena rkey/addr/size, progress
counter, and a pre-built RC QP card per expected connection keyed by
``"from_rank->to_rank"``) to the rendezvous server; the server collects all ``world``
infos and broadcasts the FULL membership to everyone at once. Each member then pairs
QP cards with exactly the peers its topology needs — chain: predecessor/successor,
ring: neighbours, mesh: everyone — and both sides drive the pair straight to RTR/RTS
(:func:`flashboot.transport.infiniband.connect_endpoint`). No per-link handshake rounds: the cards
were minted before the rendezvous, so the membership broadcast IS the handshake.

Data plane — the chain's per-chunk flags become a PROGRESS COUNTER the downstream
polls with small READs (the ``cuStreamWaitValue32`` of RDMA):

  * every non-seed member keeps a u32 host-memory counter (its own MR) = how many
    chunks of its arena are valid, bumped as its own pull lands them;
  * the seed publishes NO counter (``progress_addr == 0`` — its bytes are always
    valid, exactly the ``pred_flags == 0`` convention of _C.ChainReceiver);
  * the downstream loop: READ the upstream counter -> big READs for the ready-but-
    unpulled chunks (GPUDirect straight into the local arena) -> bump the own counter
    for the next hop.

Chunk geometry must be UNIFORM along the chain — every member reads the same
``FB_CHAIN_CHUNK_MB`` knob (default 256, shared with the handle chain) and mismatches
fail loudly at pull time.

All-gather modes (``FB_ALLGATHER_MODE``):

  * ``direct`` (default) — every rank READs each native segment straight from its
    owner, all K-1 segment reads in flight concurrently. The rendezvous itself is the
    readiness barrier (a rank only registers after its own segment is filled), so no
    progress polling at all; on a non-blocking IB fabric the NIC ingests the K-1
    flows at line rate. Costs 2(K-1) QPs per rank.
  * ``ring`` — K-1 steps, step s READs from the left neighbour the segment it gained
    in step s-1, gated by the neighbour's progress counter (= completed steps). One
    ingress + one egress peer per rank (2 QPs), the classic choice when the fabric
    can't carry concurrent all-to-all flows.

Failure semantics match the chain broadcast: a member dying mid-transfer stalls its
downstream on a counter that never advances, bounded by ``timeout_s``.
"""
from __future__ import annotations

import ctypes
import dataclasses
import math
import os
import time

from flashboot import rendezvous
from flashboot.utils.arena_partitioning import (
    chain_chunk_bytes,
    chain_connections,
    connection_key,
    mesh_connections,
    parse_connection_key,
    ring_connections,
    segment_bounds,
)
from flashboot.transport import infiniband
from flashboot.transport.rdma_engine import bound_nic_info, rdma_backend

# Per-WR ceiling for the big data READs: safely under every mlx5 port's max_msg_sz;
# larger pulls are split into back-to-back WRs on the same QP (posted in one wave, so
# the wire never drains between them).
_MAX_READ_BYTES = 1 << 30


def all_gather_mode() -> str:
    """``FB_ALLGATHER_MODE``: ``direct`` (default — concurrent per-owner segment
    reads, see the module docstring) or ``ring``."""
    mode = os.getenv("FB_ALLGATHER_MODE", "direct").strip().lower()
    if mode not in ("direct", "ring"):
        raise ValueError(f"FB_ALLGATHER_MODE={mode!r} is not supported: "
                         f"expected 'direct' or 'ring'")
    return mode


# ============================== membership ==============================
class HostCounter:
    """A u32 counter in its own cache line of HOST memory, registered as an MR so
    peers poll it with small READs. Single writer, monotonic — an aligned 32-bit
    store can't tear, so the remote poll only ever sees a valid past value. Also
    doubles as the LANDING buffer for polling a peer's counter (the READ needs a
    registered local destination). Keep it alive as long as any peer may poll it."""

    def __init__(self, endpoint) -> None:
        self._buffer = (ctypes.c_uint32 * 16)()   # 64 bytes: one cache line, zeroed
        self.addr = ctypes.addressof(self._buffer)
        self.lkey, self.rkey = endpoint.reg(self.addr, ctypes.sizeof(self._buffer))

    def store(self, value: int) -> None:
        self._buffer[0] = int(value)

    def load(self) -> int:
        return int(self._buffer[0])


@dataclasses.dataclass
class RdmaMembership:
    """One member's data-plane half of the collective: the endpoint owning every MR
    and pre-built QP (keepalive — peers READ our arena and poll our counter for the
    duration of THEIR pulls), the local card per connection, and the wire ``info``
    for the rendezvous registration. KEEP the membership alive until the whole collective
    completed on every rank."""
    endpoint: object           # native _C.Endpoint
    rank: int
    world: int
    arena_ptr: int
    arena_size: int
    arena_lkey: int            # local key for READs landing in our own arena
    cards: dict                # connection key -> local card (with the "qp" index)
    progress: object           # HostCounter we publish, None when always-full (seed)
    scratch: object            # HostCounter used as the poll landing buffer
    info: dict                 # the rendezvous registration payload

    def renew_cards(self) -> dict:
        """Re-mint this member's QP cards for the NEXT round, and return the refreshed
        ``info``.

        A seed hosting round after round has to do this: an RC queue pair cannot be
        re-pointed at a new peer while it is in RTS, so the same card handed to a
        second clone would be useless. ``Endpoint.reset_qp`` is the legal re-target
        path (RTS -> RESET -> INIT with a fresh PSN) and deliberately keeps the qpn and
        every memory registration — the arena MR and its rkey survive, so serving N
        clones costs ONE registration, and this is safe only after the previous round's
        transfers completed (which is exactly when the rendezvous releases it).

        The nixl backend has no queue pair to re-target, but the mirror-image problem
        is worse: a SIGKILLed peer leaves its UCX worker unable to bring up a queue
        pair for the NEXT one, below anything the NIXL API can reset. So the whole
        agent is rebuilt (``renew_agent``) and the card re-issued, which republishes
        the new worker's metadata. See :meth:`flashboot.transport.nixl_endpoint.NixlEndpoint.
        renew_agent` for the measured failure.
        """
        if hasattr(self.endpoint, "renew_agent"):
            self.endpoint.renew_agent()
            ranges = (self.endpoint.local_ranges()
                      if hasattr(self.endpoint, "local_ranges") else None)
            for key, card in list(self.cards.items()):
                fresh = dict(self.endpoint.create_qp())
                fresh["qp"] = int(card.get("qp", 0))
                if ranges:
                    fresh["ranges"] = ranges
                self.cards[key] = fresh
            self.info["credential"]["qp_cards"] = self.cards
            return self.info
        if hasattr(self.endpoint, "reset_qp"):
            for key, card in list(self.cards.items()):
                fresh = dict(self.endpoint.reset_qp(int(card["qp"])))
                fresh.setdefault("qp", int(card["qp"]))
                if "ranges" in card:
                    fresh["ranges"] = card["ranges"]
                self.cards[key] = fresh
            self.info["credential"]["qp_cards"] = self.cards
            print(f"[flashboot][rdma] round boundary: rank {self.rank} reset "
                  f"{len(self.cards)} queue pair(s), arena registration kept",
                  flush=True)
        return self.info


def prepare_membership(endpoint, rank: int, world: int, arena_ptr: int,
                       arena_size: int, connections, publish_progress: bool,
                       extra_info: dict = None) -> RdmaMembership:
    """Register the arena + progress counter and pre-build ONE RC QP per expected
    connection, all BEFORE the rendezvous — the returned ``info`` is the complete
    registration, so the membership broadcast is the last rendezvous message needed.
    ``publish_progress=False`` marks this member always-full (``progress_addr == 0``,
    the seed / a direct-mode all-gather member); ``extra_info`` keys ride the wire
    verbatim (free-form telemetry such as bench checksums)."""
    chunk_bytes = chain_chunk_bytes()
    num_chunks = math.ceil(int(arena_size) / chunk_bytes)
    # Timed separately from the pull on purpose. Registration is not part of
    # direct_pull's window, and on every stack measured so far it is the LARGER of
    # the two: pinning the arena and handing the NIC its rkey costs multiples of what
    # reading the bytes back costs, and the NIC is idle for all of it. Reporting only
    # the transfer makes a clone look faster than it is.
    _reg_t0 = time.perf_counter()
    arena_lkey, arena_rkey = endpoint.reg(int(arena_ptr), int(arena_size))
    _reg_s = time.perf_counter() - _reg_t0
    print(f"[flashboot][rdma] arena registered: {int(arena_size) / 1e9:.1f}GB as 1 MR "
          f"in {_reg_s:.2f}s (backend={rdma_backend()})", flush=True)
    progress = HostCounter(endpoint) if publish_progress else None
    scratch = HostCounter(endpoint)
    _qp_t0 = time.perf_counter()
    cards = {connection_key(from_rank, to_rank): dict(endpoint.create_qp())
             for from_rank, to_rank in connections}
    print(f"[flashboot][rdma] {len(cards)} QP(s) created in "
          f"{time.perf_counter() - _qp_t0:.2f}s", flush=True)
    # The nixl backend addresses reads by (agent, address): tell peers which ranges
    # this agent serves so a member with several remotes can route each read.
    if hasattr(endpoint, "local_ranges"):
        ranges = endpoint.local_ranges()
        for card in cards.values():
            card["ranges"] = ranges
    # The publication both families send. What every member has goes in the fields;
    # how to READ this arena — addresses, keys, the counter, a card per link — goes in
    # the credential, which nothing outside this engine opens.
    credential = dict(bound_nic_info())   # nic / nic_count, for the divergence check
    credential.update(
        backend=rdma_backend(),
        arena_addr=int(arena_ptr), arena_rkey=int(arena_rkey),
        progress_addr=progress.addr if progress else 0,
        progress_rkey=progress.rkey if progress else 0,
        qp_cards=cards,
    )
    info = rendezvous.ArenaAdvertisement(
        arena_size=int(arena_size), rank=int(rank), world=int(world),
        transport="rdma", chunk_bytes=chunk_bytes, num_chunks=num_chunks,
        credential=credential).to_wire()
    info.update(extra_info or {})
    return RdmaMembership(endpoint=endpoint, rank=int(rank), world=int(world),
                          arena_ptr=int(arena_ptr), arena_size=int(arena_size),
                          arena_lkey=arena_lkey, cards=cards, progress=progress,
                          scratch=scratch, info=info)


def prepare_chain_membership(endpoint, rank, world, arena_ptr, arena_size,
                             extra_info=None) -> RdmaMembership:
    """Chain-broadcast membership: rank 0 is the SEED (always-full, publishes no
    progress), ranks 1..world-1 the clones in chain order."""
    return prepare_membership(
        endpoint, rank, world, arena_ptr, arena_size,
        connections=chain_connections(rank, world),
        publish_progress=(rank > 0), extra_info=extra_info)


def prepare_allgather_membership(endpoint, rank, world, arena_ptr, arena_size,
                                 extra_info=None,
                                 segment_alignment: int = 1) -> RdmaMembership:
    """All-gather membership for the resolved ``FB_ALLGATHER_MODE``: mesh QPs and no
    progress counter for ``direct`` (the rendezvous is the readiness barrier), ring
    QPs plus a completed-steps counter for ``ring``. The mode and the segment
    alignment ride the wire so a drifted knob fails loudly instead of deadlocking
    (or silently gathering shifted segments)."""
    mode = all_gather_mode()
    connections = (mesh_connections if mode == "direct" else ring_connections)(rank, world)
    info = dict(extra_info or {})
    info["allgather_mode"] = mode
    info["segment_alignment"] = max(1, int(segment_alignment))
    return prepare_membership(endpoint, rank, world, arena_ptr, arena_size,
                              connections=connections,
                              publish_progress=(mode == "ring"), extra_info=info)


def _report_nic_divergence(membership: RdmaMembership, by_rank: dict) -> None:
    """Warn when the two ends of a link derived their GPU->HCA binding from a DIFFERENT
    number of visible HCAs.

    The raw backend's map is positional — ``gpu // (num_gpus // num_nics)`` over the
    host's OWN sorted HCAs — so a peer that sees a different HCA count derives a
    different binding, silently, with no symptom other than bandwidth. Measured: a
    clone whose four ranks came out with a wide per-rank spread because the seed
    side saw only 2 HCAs and put two ranks on one card; the same code on another pair
    of machines was flat across ranks. Nothing here can repair it — both ends bound
    their endpoints before the rendezvous — so this exists purely to put the cause in
    the log next to the bandwidth it explains.

    Only the MANUALLY bound backend can suffer this, so an empty ``nic`` (the nixl
    backend, where UCX picks the rail) ends the check: differing HCA counts there are
    not a divergence, and warning about them would be a false alarm about the very
    thing that backend exists to avoid."""
    if not membership.info["credential"].get("nic"):
        return
    mine = membership.info["credential"].get("nic_count")
    if not mine:
        return
    my_rate = float(membership.info["credential"].get("nic_rate") or 0.0)
    for key in membership.cards:
        from_rank, to_rank = parse_connection_key(key)
        peer_rank = to_rank if from_rank == membership.rank else from_rank
        peer = by_rank.get(peer_rank) or {}
        theirs = peer.get("nic_count")
        if not theirs:
            continue
        their_rate = float(peer.get("nic_rate") or 0.0)
        if int(theirs) != int(mine):
            problem = (f"sees {mine} active HCA(s) but rank {peer_rank} sees {theirs}, "
                       f"so the positional map binds DIFFERENTLY on each side")
        elif my_rate and their_rate and my_rate != their_rate:
            problem = (f"bound a {my_rate:.0f}Gb/s rail while rank {peer_rank} bound "
                       f"{their_rate:.0f}Gb/s, so the link runs at the slower one")
        else:
            continue
        print(f"[flashboot][rdma] WARNING rank {membership.rank} {problem} (ours: "
              f"{membership.info.get('nic') or 'engine default'}@{my_rate:.0f}, theirs: "
              f"{peer.get('nic') or 'engine default'}@{their_rate:.0f}). Expect ranks "
              f"that do not finish together: pin the same HCA set on both ends, or run "
              f"FB_RDMA_BACKEND=nixl and let UCX choose the rail.", flush=True)


def connect_topology(membership: RdmaMembership, members: list) -> None:
    """Pair every pre-built local QP with the matching peer card from the broadcast
    membership and drive it to RTR/RTS. The peer of connection ``from->to`` is
    whichever side we are not; both ends run this right after the broadcast, so no
    further rendezvous messages are needed."""
    by_rank = {int(member["rank"]): member for member in members}
    # Same lockstep requirement as the handle transports: the arena layout mirrors the
    # engine's runtime tensor layout, so two ends on different flashboot builds move
    # bytes successfully and produce WRONG weights. Name it here, before any READ.
    versions = {int(member.get("version", -1)) for member in members}
    if versions != {rendezvous.ADVERTISEMENT_VERSION}:
        raise RuntimeError(
            f"[flashboot][rdma] flashboot wire version drift across the membership: "
            f"{sorted(versions)} against this process's "
            f"{rendezvous.ADVERTISEMENT_VERSION} — every member must run the same "
            f"build")
    backends = {str(member["credential"].get("backend", "")) for member in members}
    if len(backends) > 1:
        raise RuntimeError(
            f"[flashboot][rdma] mixed rdma backends across the membership: "
            f"{sorted(backends)} — every member must run the same engine (an unset "
            f"FB_RDMA_BACKEND resolves to nixl only where its bindings import, so pin "
            f"it explicitly on both ends)")
    try:
        _report_nic_divergence(membership, by_rank)
    except Exception:  # noqa: BLE001 — a diagnostic must never fail a load
        pass
    for key, card in membership.cards.items():
        from_rank, to_rank = parse_connection_key(key)
        peer = to_rank if from_rank == membership.rank else from_rank
        peer_cards = by_rank[peer]["credential"].get("qp_cards") or {}
        if key not in peer_cards:
            raise RuntimeError(
                f"[flashboot][rdma] rank {peer} advertised no QP card for connection "
                f"{key} — every member must derive the same topology "
                f"(world={membership.world}, its cards: {sorted(peer_cards)})")
        infiniband.connect_endpoint(membership.endpoint, peer_cards[key],
                                        qp=int(card["qp"]))


# ============================== direct 1-to-1 ==============================
def prepare_direct_membership(endpoint, rank: int, arena_ptr: int,
                              arena_size: int) -> RdmaMembership:
    """Membership for the DIRECT 1-to-1 pull: seed is rank 0, clone rank 1, one link.

    Publishes no progress counter — the seed's bytes are all valid before the clone
    arrives, so there is nothing to gate on and nothing to pipeline. Compare
    :func:`prepare_chain_membership`, which needs both because each hop reads an arena
    that is still being filled."""
    return prepare_membership(endpoint, rank, 2, arena_ptr, arena_size,
                              connections=[(0, 1)], publish_progress=False)


def direct_pull(membership: RdmaMembership, source: dict, timeout_s: float) -> None:
    """Pull the WHOLE of ``source``'s arena into ours in one READ wave (split only at
    the per-WR ceiling, so the wire never drains between the work requests).

    No chunking: measured on the same pair of machines, one whole-arena READ and 228
    chunked READs reach the same bandwidth. The chunking exists so a chain can overlap
    hops, and a 1-to-1 clone has no hop to overlap with."""
    size = int(source["arena_size"])
    if size != membership.arena_size:
        raise RuntimeError(
            f"[flashboot][rdma] arena size mismatch for the direct pull: the seed "
            f"offers {size} bytes, ours is {membership.arena_size} — seed and clone "
            f"must load the same shard of the same checkpoint")
    qp = int(membership.cards[connection_key(0, 1)]["qp"])
    started = time.perf_counter()
    posted = _post_segment_read(membership, source, 0, size, qp)
    membership.endpoint.wait_completions(posted, int(max(timeout_s, 1.0) * 1000))
    elapsed = time.perf_counter() - started
    print(f"[flashboot][rdma] direct pull: {size / 1e9:.1f}GB over rdma in "
          f"{elapsed:.2f}s ({size / 1e9 / max(elapsed, 1e-9):.1f}GB/s, {posted} WR, "
          f"1-to-1)", flush=True)


# ============================== chain data plane ==============================
def pull_chunks(num_chunks: int, upstream_chunks_ready, read_chunks,
                publish_progress, timeout_s: float,
                poll_interval_s: float = 0.0002) -> None:
    """The downstream progress loop, dependency-injected for testability:
    ``upstream_chunks_ready()`` -> how many chunks the upstream holds (a small READ
    of its counter; constant ``num_chunks`` for the seed), ``read_chunks(first,
    last)`` -> blocking READ of ``[first, last)``, ``publish_progress(n)`` -> bump our
    own counter for the next hop.

    One chunk per pass, deliberately. The loop must not read everything the upstream
    has and publish once at the end: against the seed, whose bytes are all valid from
    the start, that is the whole arena in a single pass and the successor waits out our
    entire pull instead of streaming behind us — a chain degenerating into a relay.
    Capping the pass IS the pipeline.

    The cap used to be a second knob in bytes (``FB_RDMA_BATCH_MB``, 256MB worth of
    chunks per pass) on top of the chunk size, which bought a granularity nothing could
    observe: progress was published per pass, so the pass — not the chunk — was already
    the real unit. One chunk per pass makes the chunk size the only knob, and its
    default carries what the batch knob used to."""
    done = 0
    deadline = time.monotonic() + float(timeout_s)
    while done < num_chunks:
        available = min(int(upstream_chunks_ready()), num_chunks)
        if available <= done:
            if time.monotonic() > deadline:
                raise TimeoutError(
                    f"[flashboot][rdma] upstream progress stalled at {done}/"
                    f"{num_chunks} chunks for {timeout_s:.0f}s")
            time.sleep(poll_interval_s)
            continue
        target = min(available, done + 1)
        read_chunks(done, target)
        done = target
        publish_progress(done)


def chain_pull(membership: RdmaMembership, members: list, timeout_s: float) -> None:
    """Pull the whole arena from the chain predecessor (rank - 1), chunk-pipelined:
    poll its progress counter, READ the ready chunks straight into our arena
    (GPUDirect), publish our own counter as they land so the successor streams
    concurrently. Blocking until our whole arena is valid; our successor keeps
    reading our arena and counter afterwards — the membership stays alive on the
    caller."""
    if membership.rank == 0:
        raise RuntimeError("[flashboot][rdma] the seed (chain rank 0) has nothing "
                           "to pull — only ranks > 0 run chain_pull")
    upstream = next(m for m in members if int(m["rank"]) == membership.rank - 1)
    if int(upstream["arena_size"]) != membership.arena_size:
        raise RuntimeError(
            f"[flashboot][rdma] arena drift at chain rank {membership.rank}: local "
            f"size={membership.arena_size} != upstream size={upstream['arena_size']} "
            f"— every instance must load the same checkpoint with identical flags")
    chunk_bytes = chain_chunk_bytes()
    if int(upstream["chunk_bytes"]) != chunk_bytes:
        raise RuntimeError(
            f"[flashboot][rdma] chunk mismatch at chain rank {membership.rank}: ours "
            f"{chunk_bytes} != upstream {upstream['chunk_bytes']} — set the same "
            f"FB_CHAIN_CHUNK_MB on every instance")

    endpoint, scratch = membership.endpoint, membership.scratch
    pull_qp = int(membership.cards[connection_key(membership.rank - 1,
                                                  membership.rank)]["qp"])
    timeout_ms = max(1, int(float(timeout_s) * 1000))
    num_chunks = int(membership.info["num_chunks"])
    upstream_progress_addr = int(upstream["credential"].get("progress_addr", 0))

    if upstream_progress_addr:
        def upstream_chunks_ready() -> int:
            endpoint.read(scratch.addr, scratch.lkey, upstream_progress_addr,
                          int(upstream["credential"]["progress_rkey"]), 4, timeout_ms, qp=pull_qp)
            return scratch.load()
    else:
        def upstream_chunks_ready() -> int:  # the seed: bytes always valid
            return num_chunks

    upstream_arena_addr = int(upstream["credential"]["arena_addr"])
    upstream_arena_rkey = int(upstream["credential"]["arena_rkey"])

    def read_chunks(first: int, last: int) -> None:
        posted = 0
        for chunk in range(first, last):
            offset = chunk * chunk_bytes
            length = min(chunk_bytes, membership.arena_size - offset)
            endpoint.post_read(membership.arena_ptr + offset, membership.arena_lkey,
                               upstream_arena_addr + offset, upstream_arena_rkey,
                               length, wr_id=chunk, qp=pull_qp)
            posted += 1
        endpoint.wait_completions(posted, timeout_ms)

    started = time.perf_counter()
    pull_chunks(num_chunks, upstream_chunks_ready, read_chunks,
                membership.progress.store, timeout_s)
    elapsed = time.perf_counter() - started
    print(f"[flashboot][rdma] chain rank {membership.rank}: "
          f"{membership.arena_size / 1e9:.1f}GB pulled over rdma in {elapsed:.2f}s "
          f"({membership.arena_size / 1e9 / max(elapsed, 1e-9):.1f}GB/s, "
          f"{num_chunks} chunks x {chunk_bytes >> 20}MB, "
          f"{'progress-gated' if upstream_progress_addr else 'seed-fed'})",
          flush=True)


# ============================== all-gather data plane ==============================
def _post_segment_read(membership: RdmaMembership, source: dict, start: int,
                       end: int, qp: int) -> int:
    """Post the READ wave for ``[start, end)`` of ``source``'s arena into the same
    offsets of ours, split at the per-WR ceiling. Returns the number of WRs."""
    endpoint, posted = membership.endpoint, 0
    source_addr, source_rkey = int(source["credential"]["arena_addr"]), int(source["credential"]["arena_rkey"])
    offset = start
    while offset < end:
        length = min(_MAX_READ_BYTES, end - offset)
        endpoint.post_read(membership.arena_ptr + offset, membership.arena_lkey,
                           source_addr + offset, source_rkey, length,
                           wr_id=offset, qp=qp)
        offset += length
        posted += 1
    return posted


def all_gather(membership: RdmaMembership, members: list, timeout_s: float) -> None:
    """Make this rank hold the FULL arena from the per-rank native segments
    (flashload_dp: rank r filled ``segment_bounds(r)`` before the rendezvous).
    Dispatches on the mode every member advertised — a drifted FB_ALLGATHER_MODE
    fails loudly here instead of deadlocking on missing cards."""
    modes = {str(m.get("allgather_mode", "direct")) for m in members}
    if len(modes) != 1:
        raise RuntimeError(f"[flashboot][rdma] FB_ALLGATHER_MODE drift across the "
                           f"membership: {sorted(modes)} — set the same mode on "
                           f"every rank")
    sizes = {int(m["arena_size"]) for m in members}
    if sizes != {membership.arena_size}:
        raise RuntimeError(f"[flashboot][rdma] arena drift across the all-gather "
                           f"membership: sizes {sorted(sizes)}")
    alignments = {int(m.get("segment_alignment", 1)) for m in members}
    if len(alignments) != 1:
        raise RuntimeError(f"[flashboot][rdma] segment alignment drift across the "
                           f"all-gather membership: {sorted(alignments)} — every "
                           f"rank must split the arena identically")
    mode, alignment = modes.pop(), alignments.pop()
    started = time.perf_counter()
    if mode == "direct":
        _all_gather_direct(membership, members, timeout_s, alignment)
    else:
        _all_gather_ring(membership, members, timeout_s, alignment)
    elapsed = time.perf_counter() - started
    gathered = membership.arena_size * (membership.world - 1) // membership.world
    print(f"[flashboot][rdma] all-gather rank {membership.rank}: "
          f"{gathered / 1e9:.1f}GB gathered ({mode}) in {elapsed:.2f}s "
          f"({gathered / 1e9 / max(elapsed, 1e-9):.1f}GB/s ingress, "
          f"world={membership.world})", flush=True)


def _all_gather_direct(membership: RdmaMembership, members: list,
                       timeout_s: float, alignment: int = 1) -> None:
    """Every peer's native segment READ concurrently from its owner: all waves post
    first (one per peer QP), then one completion wait drains them — the NIC ingests
    K-1 flows at once. No progress polling: the rendezvous already proved every
    owner's segment valid."""
    by_rank = {int(m["rank"]): m for m in members}
    posted = 0
    for owner in range(membership.world):
        if owner == membership.rank:
            continue
        start, end = segment_bounds(owner, membership.world, membership.arena_size,
                                    alignment)
        qp = int(membership.cards[connection_key(owner, membership.rank)]["qp"])
        posted += _post_segment_read(membership, by_rank[owner], start, end, qp)
    membership.endpoint.wait_completions(posted,
                                         max(1, int(float(timeout_s) * 1000)))


def _all_gather_ring(membership: RdmaMembership, members: list,
                     timeout_s: float, alignment: int = 1) -> None:
    """K-1 ring steps: step s READs from the left neighbour the segment it gained in
    step s-1 (its own for s == 1), gated by the neighbour's completed-steps counter.
    Our own counter bumps after each step for the right neighbour."""
    endpoint, scratch = membership.endpoint, membership.scratch
    world, rank = membership.world, membership.rank
    left = (rank - 1) % world
    left_info = next(m for m in members if int(m["rank"]) == left)
    pull_qp = int(membership.cards[connection_key(left, rank)]["qp"])
    timeout_ms = max(1, int(float(timeout_s) * 1000))
    left_progress_addr = int(left_info["credential"].get("progress_addr", 0))
    deadline = time.monotonic() + float(timeout_s)

    for step in range(1, world):
        # The left neighbour holds the segment we want once it completed step-1
        # steps; its own native segment (step 1) needs no wait.
        while step > 1 and left_progress_addr:
            endpoint.read(scratch.addr, scratch.lkey, left_progress_addr,
                          int(left_info["credential"]["progress_rkey"]), 4, timeout_ms, qp=pull_qp)
            if scratch.load() >= step - 1:
                break
            if time.monotonic() > deadline:
                raise TimeoutError(
                    f"[flashboot][rdma] ring all-gather stalled at step {step}/"
                    f"{world - 1}: left neighbour {left} stuck at "
                    f"{scratch.load()} completed steps for {timeout_s:.0f}s")
            time.sleep(0.0002)
        owner = (rank - step) % world
        start, end = segment_bounds(owner, world, membership.arena_size, alignment)
        posted = _post_segment_read(membership, left_info, start, end, pull_qp)
        endpoint.wait_completions(posted, timeout_ms)
        membership.progress.store(step)
