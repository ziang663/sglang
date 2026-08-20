"""Who talks to whom, which arena bytes each member owns, and how those bytes are cut
into chunks — pure integer math.

Every layer needs these answers and none of them owns them. The handle engine used to
borrow :func:`segment_bounds` from the rdma engine, which made two sibling
implementations depend on each other; both now reach down here, and nothing here reaches
back.

Stdlib only — no CUDA, no sockets, no flashboot imports at all. That is what lets a
member compute its own topology and its own segment BEFORE any rendezvous, which is in
turn why the rendezvous can be a single round: the credentials a peer needs are already
minted by the time the first message is sent.
"""
from __future__ import annotations

import os
from typing import List, Tuple


def chain_chunk_bytes() -> int:
    """The unit a pipelined broadcast moves and signals in (``FB_CHAIN_CHUNK_MB``,
    default 256MB) — one definition, both chain engines.

    A chain is only correct if every member agrees on this: a successor indexes its
    predecessor's readiness signal with its OWN chunk number, so a mismatch reads the
    wrong signal rather than failing. It rides the wire and both engines reject drift,
    because the two ends are different processes.

    It is also the only knob over the pipeline's shape. Small chunks let the successor
    start sooner; large chunks amortise the per-chunk signal (a device flag write on
    the handle engine, a completion wait plus the successor's counter READ on rdma).
    256MB is where a chunk is still a fraction of a second on either fabric while the
    signalling is lost in the noise: at 25GB/s one chunk is ~10ms against a completion
    turnaround measured in microseconds."""
    return int(os.getenv("FB_CHAIN_CHUNK_MB", "256")) << 20


# ── links ──────────────────────────────────────────────────────────────────────
def connection_key(from_rank: int, to_rank: int) -> str:
    """The membership key of one expected connection: ``"from->to"`` where ``from``
    is the data source (READ target) and ``to`` the puller. BOTH ends mint a QP card
    under this key; the rendezvous broadcast lets them pair without any handshake."""
    return f"{int(from_rank)}->{int(to_rank)}"


def parse_connection_key(key: str) -> Tuple[int, int]:
    from_rank, _, to_rank = key.partition("->")
    return int(from_rank), int(to_rank)


def chain_connections(rank: int, world: int) -> List[Tuple[int, int]]:
    """The connections a chain member participates in: pull from the predecessor
    (every rank but the seed), serve the successor (every rank but the last)."""
    connections = []
    if rank > 0:
        connections.append((rank - 1, rank))
    if rank < world - 1:
        connections.append((rank, rank + 1))
    return connections


def ring_connections(rank: int, world: int) -> List[Tuple[int, int]]:
    """The connections of a ring all-gather member: pull from the left neighbour,
    serve the right one (the same single link for world == 2)."""
    if world < 2:
        return []
    left, right = (rank - 1) % world, (rank + 1) % world
    return [(left, rank), (rank, right)]


def mesh_connections(rank: int, world: int) -> List[Tuple[int, int]]:
    """The connections of a direct all-gather member: pull from every other rank,
    serve every other rank — 2(K-1) queue pairs."""
    return ([(other, rank) for other in range(world) if other != rank]
            + [(rank, other) for other in range(world) if other != rank])


# ── segments ───────────────────────────────────────────────────────────────────
def segment_bounds(rank: int, world: int, total_bytes: int,
                   alignment: int = 1) -> Tuple[int, int]:
    """The ``[start, end)`` arena byte range rank owns natively in the all-gather
    (uniform split, the tail absorbing the remainder). Every member computes the
    same bounds from the membership, so only sizes cross the wire.

    ``alignment`` rounds every INTERIOR boundary up to a multiple (the outer ends
    stay 0 and ``total_bytes``), the tail absorbing whatever the rounding shifted.
    Callers want the default: the flashload_dp split ran at 64MB for a while on the
    assumption that a segment should coincide with the transfer chunking, and nothing
    below it ever wanted that — a run clips at any byte, the fill chunker spans
    whatever runs it is given, and an RDMA READ takes any offset. What the rounding
    did produce was imbalance, up to 10% of one replica's disk at eight replicas,
    which the others then wait for in the gather. The parameter stays because members
    must agree on it either way: it rides the membership as ``segment_alignment`` and
    the all-gather rejects drift."""
    align = max(1, int(alignment))

    def boundary(position: int) -> int:
        if position <= 0:
            return 0
        if position >= int(world):
            return int(total_bytes)
        uniform = position * int(total_bytes) // int(world)
        return min((uniform + align - 1) // align * align, int(total_bytes))

    return boundary(int(rank)), boundary(int(rank) + 1)


# ── rendezvous group keys ──────────────────────────────────────────────────────
def data_parallel_gather_group(shard_rank: int, is_draft_model: bool = False) -> str:
    """The rendezvous group key of one tp shard's DP all-gather (the ``sharded_dp``
    loader role): the port already separates shard ranks (and the draft band), so
    the group is a semantic guard — a member dialing the wrong port fails loudly on
    the group mismatch instead of joining a foreign collective."""
    return f"dp-gather/shard{int(shard_rank)}" + ("/draft" if is_draft_model else "")
