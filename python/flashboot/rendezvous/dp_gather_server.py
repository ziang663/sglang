"""The DP gather's rendezvous: collect every replica's publication, broadcast them all.

The flashload_dp pattern's meeting point, and nothing else. Every replica of one tp
shard registers what it holds; the server collects all ``world`` of them and hands the
whole list back at once, so each replica can read the segments it is missing straight
from their owners. One collection round, no per-link handshakes, because the credentials
were minted before the registration that carried them.

The host is NOT a member here. Replica 0 happens to run this thread and then joins its
own server over loopback like everyone else, which is why there is no local member and
no link-connect hook: a peer-less server, serving peers. The broadcast seed's round is
the same round with those hooks filled in, and lives in
:mod:`flashboot.rendezvous.seed_server` where the seed's other session shapes are.

Rounds REPEAT. One failed attempt, or one replica that came and went, must not leave the
port closed so the retry gets ECONNREFUSED. A finished round resets the collection state
and the server keeps standing.
"""
from __future__ import annotations

import logging
import socket
import threading

from flashboot.rendezvous.collective_round import abort_round, complete_round
from flashboot.rendezvous.member_registration import MemberRegistration
from flashboot.rendezvous.message_framing import receive_json_message, send_json_message

logger = logging.getLogger(__name__)

_LABEL = "dp-gather"


def start_dp_gather_server(port, world=None, timeout_s=1800.0) -> dict:
    """DP gather rendezvous server (daemon thread on 0.0.0.0:``port``): collect every
    replica's info, broadcast the full membership to all, wait out the completions —
    then go back to accepting and serve the NEXT round.

    ``world`` is the membership size; leave it None to adopt the ``world`` of the first
    registration. Ranks must be exactly ``0..world-1`` — the server validates and
    rejects the round loudly otherwise (a mis-launched replica is a config error, not
    a wait).

    Returns the keepalive dict ``{"thread": ..., "rounds": N}`` — plus an ``"error"``
    key if the last round was aborted, so the hosting process can tell "nobody joined
    yet" from "this rendezvous already failed". Whatever backs a member's info
    (endpoint, memory registrations) must be kept alive by its own process for as long
    as peers read it.
    """
    keepalive = {"thread": None, "rounds": 0}

    def _serve():
        server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        server.bind(("0.0.0.0", int(port)))
        server.listen(16)
        server.settimeout(timeout_s)
        pending, expected_world, expected_group = [], world, None

        try:
            logger.info(f"[flashboot][{_LABEL}] serving rendezvous rounds on :{port} "
                        f"(world={world})")
            while True:
                try:
                    connection, _ = server.accept()
                except socket.timeout:
                    # Quiet for timeout_s. A half-collected round is stale — its
                    # members are told why so they stop waiting — but the server keeps
                    # standing for whatever comes next.
                    if pending:
                        abort_round(_LABEL, port, pending,
                                    f"rendezvous incomplete after {timeout_s:.0f}s "
                                    f"({len(pending)}/{expected_world})", keepalive)
                        pending, expected_world, expected_group = [], world, None
                    continue
                connection.settimeout(timeout_s)
                try:
                    payload = receive_json_message(connection)
                except Exception as e:  # noqa: BLE001 — a bad client must not kill serving
                    logger.error(f"[flashboot][{_LABEL}] :{port} unreadable "
                                 f"registration: {e}")
                    connection.close()
                    continue
                # Parsing decides whether this is a registration at all; admission
                # decides whether a well-formed one fits the round in progress. Both
                # turn into a reason the sender is told, never into an exception that
                # would end the serving thread and strand everyone already collected.
                try:
                    registration = MemberRegistration.from_wire(payload)
                    problem = registration.problem_joining(expected_world,
                                                           expected_group)
                except Exception as e:  # noqa: BLE001 — garbage in a field, not a crash
                    problem = f"malformed registration: {type(e).__name__}: {e}"
                if problem:
                    try:
                        send_json_message(connection, {"error": problem})
                    except Exception:  # noqa: BLE001 — peer may already be gone
                        pass
                    connection.close()
                    continue
                expected_world = expected_world or registration.world
                expected_group = expected_group if expected_group is not None \
                    else registration.group
                pending.append((connection, registration.info))
                logger.info(f"[flashboot][{_LABEL}] :{port} "
                            f"{len(pending)}/{expected_world} collected")
                if len(pending) == expected_world:
                    complete_round(_LABEL, port, pending, keepalive=keepalive)
                    pending, expected_world, expected_group = [], world, None
        except Exception as e:  # noqa: BLE001 — never crash the serving process
            logger.error(f"[flashboot][{_LABEL}] :{port} server failed: {e}",
                         exc_info=True)
        finally:
            server.close()

    thread = threading.Thread(target=_serve, name=f"flashboot-dp-gather-{port}",
                              daemon=True)
    thread.start()
    keepalive["thread"] = thread
    return keepalive
