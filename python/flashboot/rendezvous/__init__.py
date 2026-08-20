"""Processes finding each other and swapping the credential to read each other's memory.

Every transfer flashboot does starts the same way: two or more processes that were
launched separately, know only a host and a base port, have to meet and hand over
whatever the other side needs in order to read an arena. That is all this package does.
It moves small JSON messages over TCP — never weights — and it never looks inside the
credential it forwards.

Three shapes of meeting, and the difference between them is about COORDINATION, not
about how bytes later move:

    a seed publishes, clones take     asymmetric, one round trip, the seed outlives them
    a chain forms                     K clones register, every position released at once
    a group arrives together          symmetric, everybody leaves with the full membership

    seed_server           hosts the first two: they share a port and are told apart by
                          the first message on the connection
    dp_gather_server      hosts the third: the DP gather, round after round
    collective_round      the round both symmetric servers run
    joining               the other side of all three: what a clone or member calls
    addressing            which port a shard serves on; which host a clone dials
    arena_advertisement   what a seed says about its arena, and the wire-version guard
    member_registration   what a member sends to enter one round, and what disqualifies it
    message_framing       length-prefixed JSON, capped so a stray peer cannot make a
                          server allocate

What this package deliberately does NOT know: CUDA, and which transport will carry the
bytes. A credential is an opaque dict here — a 64-byte memory handle, or an address plus
an rkey plus a queue-pair card, it makes no difference — which is exactly what lets a new
transport be added without touching a line of this code. A test enforces it
(tests/test_layering.py).

NOTE — UNAUTHENTICATED, by design and worth stating plainly: whatever can reach a seed's
rank port receives that rank's credential, and an ipc/fabric memory handle IS read access
to the arena, i.e. to the model weights (the rdma family's rkey likewise). It binds
0.0.0.0 because seed and clone are different nodes. Framing is length-capped and a bad
message can never take a server thread down — but there is no identity check. Run it on a
trusted cluster network.
"""
from __future__ import annotations

from flashboot.rendezvous.addressing import rank_server_port, seed_host_for_rank
from flashboot.rendezvous.arena_advertisement import (
    ADVERTISEMENT_VERSION,
    ArenaAdvertisement,
)
from flashboot.rendezvous.dp_gather_server import start_dp_gather_server
from flashboot.rendezvous.joining import (
    collective_complete,
    collective_rendezvous,
    fetch_arena_advertisement,
    rdma_pull_complete,
    request_rdma_pull,
)
from flashboot.rendezvous.member_registration import MemberRegistration
from flashboot.rendezvous.message_framing import (
    receive_exact_bytes,
    receive_json_message,
    send_json_message,
)
from flashboot.rendezvous.seed_server import start_seed_server

__all__ = [
    "ADVERTISEMENT_VERSION",
    "ArenaAdvertisement",
    "MemberRegistration",
    "collective_complete",
    "collective_rendezvous",
    "fetch_arena_advertisement",
    "rank_server_port",
    "rdma_pull_complete",
    "receive_exact_bytes",
    "receive_json_message",
    "request_rdma_pull",
    "seed_host_for_rank",
    "send_json_message",
    "start_dp_gather_server",
    "start_seed_server",
]
