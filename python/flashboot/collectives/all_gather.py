"""All-gather: K members each holding ``1/K`` of an arena end up holding all of it.

This is the collective the ``sharded_dp`` role uses. Each of K replicas of the same TP
layout reads only its native segment from disk — one full load's disk bandwidth split K
ways — and then one all-gather makes every replica whole, at link rate.

There is ONE implementation, for every transport. What differs between the transports is
how bytes cross (imported handles and the copy engine, or one-sided READs), and that is
behind :meth:`~flashboot.transport.contracts.ArenaGatherParticipant.gather_from`. What
does NOT differ, and is therefore here, is the ordering:

  1. join, which is what declares this member's segment ready — and for a transport whose
     credentials are queue-pair cards, mints them, because they have to ride the registration;
  2. one rendezvous round: everybody's publication in, the full membership out;
  3. connect this member's links BEFORE any byte moves;
  4. gather;
  5. report completion.

Step 2 is also the readiness barrier: a member only registers after its own segment is
filled, so by the time anybody holds the membership every owner's bytes are valid. That
is why no progress signalling is needed here at all.

The failure path matters as much as the order. If the gather throws, the rendezvous
connection is closed WITHOUT a completion, which is how the server learns the round
failed instead of waiting the whole timeout out.
"""
from __future__ import annotations

from flashboot import rendezvous


def all_gather(transport, arena_handle, device: int, rank: int, world: int,
               rendezvous_host: str, rendezvous_port: int, group: str,
               segment_alignment: int, timeout_s: float,
               connect_timeout_s: float = None):
    """Make this member hold the whole arena. Returns the participant, which the CALLER
    must keep alive: peers may still be reading this member's native segment after the
    local gather returned — the concurrent mode has no end barrier."""
    participant = transport.join_all_gather(arena_handle, device, rank, world,
                                           segment_alignment)
    members, connection = rendezvous.collective_rendezvous(
        rendezvous_host, rendezvous_port, participant.publication, world, group=group,
        timeout_s=timeout_s, connect_timeout_s=connect_timeout_s)
    try:
        participant.connect(members)
        participant.gather_from(members, timeout_s)
    except Exception:
        try:
            connection.close()  # no completion: the server logs this member as failed
        except Exception:  # noqa: BLE001
            pass
        raise
    rendezvous.collective_complete(connection)
    return participant
