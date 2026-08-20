"""One round of a symmetric rendezvous: release the membership, then wait it out.

Extracted because two servers run the same round and only the hooks differ. The DP
gather's host is not a member and has no links of its own; a broadcast seed is member
rank 0, has to bring its own queue pair up before anyone hears about it, and has to
re-mint its cards between rounds. Both of those are this round with different arguments,
so they are arguments — not two copies that drift.

Nothing here knows a transport. A member's publication is an opaque dict: it is merged,
ordered by rank, and handed back out verbatim.
"""
from __future__ import annotations

import logging

from flashboot.rendezvous.message_framing import (
    receive_json_message,
    send_json_message,
)

logger = logging.getLogger(__name__)


def abort_round(log_tag, port, waiting_members, problem, keepalive=None) -> None:
    """Give up on this round and TELL every member that already joined why.

    Sending the reason instead of just closing is the point: a member blocked in its
    rendezvous call raises immediately with this cause, rather than sitting out the
    full timeout. Send failures are ignored — a peer that is already gone cannot be
    informed, and the rest still must be.

    The reason is ALSO recorded on ``keepalive`` and printed, because the hosting
    process is the one place it would otherwise never surface: a broadcast seed's own
    load already succeeded, so a failed round leaves it serving happily while every
    clone reports an error the seed never saw. An aborted round does NOT take the
    server down — the next round may well be the fixed relaunch.
    """
    if keepalive is not None:
        keepalive["error"] = problem
    logger.error(f"[flashboot][{log_tag}] :{port} rendezvous failed: {problem}")
    print(f"[flashboot][{log_tag}] :{port} rendezvous FAILED: {problem} "
          f"({len(waiting_members)} member(s) told; the server stays up for a relaunch)",
          flush=True)
    for connection, _ in waiting_members:
        try:
            send_json_message(connection, {"error": problem})
        except Exception:  # noqa: BLE001 — peer may already be gone
            pass
        connection.close()


def complete_round(log_tag, port, waiting_members, host_publications=(),
                   connect_host_links=None, keepalive=None) -> bool:
    """Everyone has joined: hand the full membership out and wait them out. Returns
    True when the round completed, False when it was aborted (the members were told).

    Five steps, in this order for a reason:
      1. merge the host's own publication(s) with the ones that dialled in;
      2. check the ranks are exactly 0..N-1 — a duplicate or missing rank is a launch
         mistake, and waiting longer cannot fix it, so abort loudly;
      3. sort by rank so every member sees the SAME ordering (they pair queue-pair
         cards by "from->to" keys and both ends must agree);
      4. connect_host_links() — our own links go up BEFORE anyone is told about
         us, or the first remote READ lands on a queue pair that is not ready;
      5. broadcast the membership, then wait for each member's "done".
    """
    membership = list(host_publications) + [info for _, info in waiting_members]
    ranks = sorted(int(info.get("rank", -1)) for info in membership)
    if ranks != list(range(len(membership))):
        abort_round(log_tag, port, waiting_members,
                    f"member ranks must be exactly 0..{len(membership) - 1}, got "
                    f"{ranks}", keepalive)
        return False
    membership.sort(key=lambda info: int(info["rank"]))
    if connect_host_links is not None:
        # The local side (a broadcast seed) connects its queue pairs FIRST, so remote
        # members can post READs the moment they see the membership.
        try:
            connect_host_links(membership)
        except Exception as e:  # noqa: BLE001
            # Our links are down: nobody can read from us, so the round is over. Abort
            # it EXPLICITLY — letting this unwind would close the pending sockets on
            # the way out and leave every member reporting "peer closed mid-message"
            # with the real cause visible only in our own log. The traceback goes to
            # our log and the server stays up: a retry then gets this reason instead
            # of a closed port.
            logger.error(f"[flashboot][{log_tag}] :{port} host links failed",
                         exc_info=True)
            abort_round(log_tag, port, waiting_members,
                        f"the rendezvous host could not bring its own data-plane "
                        f"links up: {type(e).__name__}: {e}", keepalive)
            return False
    for position, (connection, _) in enumerate(waiting_members):
        try:
            send_json_message(connection, {"membership": membership})
        except Exception as e:  # noqa: BLE001 — one dead link must not strand the rest
            logger.error(f"[flashboot][{log_tag}] :{port} membership broadcast to "
                         f"member {position} failed: {e}")
    for position, (connection, _) in enumerate(waiting_members):
        try:
            completion = receive_json_message(connection)
            if not (isinstance(completion, dict) and completion.get("done")):
                raise RuntimeError(f"unexpected completion: {completion!r}")
        except Exception as e:  # noqa: BLE001
            logger.error(f"[flashboot][{log_tag}] :{port} member {position} never "
                         f"completed: {e}")
        finally:
            connection.close()
    if keepalive is not None:
        keepalive["rounds"] = keepalive.get("rounds", 0) + 1
        keepalive.pop("error", None)
    logger.info(f"[flashboot][{log_tag}] :{port} rendezvous complete "
                f"({len(membership)} members, {len(waiting_members)} remote)")
    return True
