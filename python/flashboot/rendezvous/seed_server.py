"""The seed's server: publish an arena, and host every shape of taking it.

One tiny TCP server per seed SHARD, on ``base_port + global_shard_rank``, publishing its
own arena. The clone holding the same shard rank connects straight to it — peer to peer,
no rank-0 demultiplexing funnel, no shared filesystem.

Every session shape arrives on the same port and says which it is in its first
message, so ONE server serves a single clone and a broadcast alike and the seed never
has to be told the clone count before anybody has dialled in:

    {"world": 1}                     a single clone: send the standing publication
    {"world": 1, "rdma": <its own>}  a direct rdma pull: one exchange, then it READs
    a MemberRegistration            K clones registering for a broadcast

The broadcast is ONE round whatever the transport — the same round the DP gather runs,
in :mod:`flashboot.rendezvous.collective_round` — and every member leaves with the full
membership. An nvlink clone reads only its predecessor out of it, because importing a
memory handle pairs with nobody; an rdma clone needs both neighbours, because its
queue-pair cards are minted per link and both ends have to pair them. That difference
belongs to the engines, not to this server: it hands everyone the same list.

This module knows nothing about how bytes move. What to publish comes from a transport's
publisher (:class:`flashboot.transport.contracts.ArenaPublisher`) and is forwarded
verbatim, which is what lets a transport be added without touching the rendezvous.

NOTE — this rendezvous is UNAUTHENTICATED, by design and worth stating plainly:
whatever can reach a seed's rank port receives that rank's credential, and an ipc/fabric
memory handle IS read access to the arena, i.e. to the model weights (the rdma family's
rkey likewise). It binds 0.0.0.0 because seed and clone are different nodes. Framing is
length-capped so a stray connection cannot make a server allocate, and a bad message can
never take a server thread down — but there is no identity check. Run it on a trusted
cluster network.
"""
from __future__ import annotations

import logging
import socket
import threading

from flashboot.rendezvous.collective_round import abort_round, complete_round
from flashboot.rendezvous.member_registration import MemberRegistration
from flashboot.rendezvous.message_framing import (
    receive_json_message,
    send_json_message,
)

logger = logging.getLogger(__name__)


def start_seed_server(port, gpu, publisher, timeout_s=1800.0, chain_seed=None) -> dict:
    """SEED: one daemon thread serving this rank's arena on 0.0.0.0:``port``.

    Returns immediately with a keepalive dict ``{"thread": ...}`` the caller holds for
    the server's lifetime.

    ``chain_seed`` is what makes one port serve BOTH cardinalities. A clone says which
    it wants in its first message — ``world == 1`` takes the standing publication or
    one direct exchange, ``world > 1`` registers for a broadcast — and the seed no
    longer has to be told the clone count when it starts. Pass a chain seed to allow
    the second shape; leave it None for a transport whose seed cannot join its own
    broadcast. It supplies three things, all of them per-round:

      * ``local_member(world)`` — this seed's own publication as member rank 0, once
        the first registration has revealed how long the chain is. Nothing expensive
        depends on that number: rank 0 has exactly one link (0->1) whatever the world,
        so the arena registration and the card are already made by the time this is
        called and only the advertised world is filled in.
      * ``connect(membership)`` — bring the seed's own link up. Called BEFORE the
        membership is broadcast, because a clone may post a READ the moment it sees it.
      * ``renew()`` — fresh publication for the next round. An RC queue pair cannot be
        re-pointed at a new peer while in RTS, so the card handed to one clone is
        useless for the next.

    ``publisher`` is the transport's seed half (see
    :class:`flashboot.transport.contracts.ArenaPublisher`) and is the ONLY thing here
    that knows how bytes move. This server does two transport-independent things with
    it:

      * ``publisher.standing_publication()`` — the credential every clone gets, built
        INSIDE the thread after bind+listen, so building it stays off the caller's
        critical path while the port is already open: a clone connects immediately and
        simply blocks until the publication is ready.
      * ``publisher.exchange_with_puller(clone_publication)`` — one clone's turn, for a
        transport whose credential cannot be standing (the rdma family: an RC queue pair
        cannot be re-pointed at a new peer while in RTS, so each clone gets a fresh
        card). The session WAITS for that clone's "done" before serving the next one,
        because a queue pair still being read must not be re-targeted.

    Whatever the publication is — a 64-byte memory handle, an address and an rkey — is
    forwarded verbatim. This module never looks inside, which is what lets a transport be
    added without touching it.
    """
    keepalive = {"thread": None, "rounds": 0}

    def _serve_broadcast_round(pending, world) -> None:
        """All ``world - 1`` clones have registered: run the collective round with this
        seed as member rank 0, then re-mint its card for the next one.

        The same round the DP gather runs (:mod:`flashboot.rendezvous.collective_round`),
        with the local-member hooks filled in — which is the whole difference between a
        broadcast and a gather at this layer, and the reason there is one implementation
        rather than two that drift."""
        seed_publication = chain_seed.local_member(world)
        complete_round("seed", port, pending, host_publications=[seed_publication],
                       connect_host_links=chain_seed.connect, keepalive=keepalive)
        try:
            chain_seed.renew()
        except Exception:  # noqa: BLE001 — must not take the server down
            # Keep serving: the next round's connect() will fail and abort THAT round
            # with a reason, which is more useful than a dead port.
            logger.error(f"[flashboot][seed] :{port} could not renew the seed's card; "
                         f"the next broadcast round will likely fail", exc_info=True)

    def _serve():
        server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        server.bind(("0.0.0.0", int(port)))
        server.listen(8)
        server.settimeout(timeout_s)
        try:
            # For the nvlink family this exports the 64-byte handle, importable by ANY
            # number of clones: built once, published forever. Whatever CUDA state that
            # needs on this thread is the publisher's business, not this module's.
            standing_publication = publisher.standing_publication()
            logger.info(f"[flashboot][seed] serving arena advertisement on :{port} "
                        f"(gpu{gpu} transport="
                        f"{standing_publication.get('transport')})")
            waiting_clones, broadcast_world, broadcast_group = [], 0, None
            while True:
                try:
                    connection, _ = server.accept()
                except socket.timeout:
                    # A half-collected broadcast round is stale: its clones are told
                    # why so they stop waiting and fall back, rather than sitting out
                    # their own load timeout on a round that will never fill.
                    if waiting_clones:
                        abort_round("seed", port, waiting_clones,
                                    f"broadcast rendezvous incomplete after "
                                    f"{timeout_s:.0f}s ({len(waiting_clones) + 1}/"
                                    f"{broadcast_world})", keepalive)
                        waiting_clones, broadcast_world, broadcast_group = [], 0, None
                        continue
                    logger.warning(f"[flashboot][seed] :{port} no clone within "
                                   f"{timeout_s}s; server exiting")
                    return
                connection.settimeout(timeout_s)
                # The first message on a connection says which session the client
                # wants, and is the only thing this loop dispatches on.
                try:
                    session_request = receive_json_message(connection)
                except Exception as e:  # noqa: BLE001 — a bad client must not kill serving
                    logger.error(f"[flashboot][seed] :{port} unreadable session "
                                 f"request: {e}")
                    connection.close()
                    continue
                # A member registration means the client wants a broadcast: it will
                # wait in the round until everyone has arrived. Anything else is a
                # single-instance session, answered on the spot.
                wants_broadcast = MemberRegistration.matches(session_request)
                if chain_seed is not None and wants_broadcast:
                    try:
                        registration = MemberRegistration.from_wire(session_request)
                        problem = registration.problem_joining(
                            broadcast_world or None, broadcast_group)
                    except Exception as e:  # noqa: BLE001 — garbage, not a crash
                        problem = f"malformed registration: {type(e).__name__}: {e}"
                    if problem:
                        try:
                            send_json_message(connection, {"error": problem})
                        except Exception:  # noqa: BLE001 — peer may already be gone
                            pass
                        connection.close()
                        continue
                    broadcast_world = broadcast_world or registration.world
                    broadcast_group = (broadcast_group if broadcast_group is not None
                                       else registration.group)
                    waiting_clones.append((connection, registration.info))
                    logger.info(f"[flashboot][seed] :{port} broadcast rendezvous "
                                f"{len(waiting_clones) + 1}/{broadcast_world} "
                                f"collected (the seed is member 0)")
                    if len(waiting_clones) + 1 == broadcast_world:
                        _serve_broadcast_round(waiting_clones, broadcast_world)
                        waiting_clones, broadcast_world, broadcast_group = [], 0, None
                    continue
                clone_publication = (session_request.get("rdma")
                                     if isinstance(session_request, dict) else None)
                if clone_publication:
                    # DIRECT 1-to-1 wire pull. One round trip: the clone's publication
                    # in, the seed's out, then the clone READs and reports done. Serving
                    # is sequential here, so waiting for that done is what keeps the next
                    # clone from re-targeting a queue pair still being read.
                    try:
                        if not publisher.offers_exchange():
                            raise RuntimeError(
                                "this seed published no rdma offer (its arena is not "
                                "registered for wire reads)")
                        send_json_message(
                            connection,
                            {"rdma": publisher.exchange_with_puller(
                                dict(clone_publication))})
                        completion = receive_json_message(connection)
                        if not (isinstance(completion, dict) and completion.get("done")):
                            raise RuntimeError(f"unexpected completion: {completion!r}")
                        logger.info(f"[flashboot][seed] :{port} direct rdma pull served")
                    except Exception as e:  # noqa: BLE001 — one clone must not end serving
                        logger.error(f"[flashboot][seed] :{port} direct rdma pull "
                                     f"failed: {e}")
                        try:
                            send_json_message(connection, {"error": str(e)})
                        except Exception:  # noqa: BLE001 — peer may already be gone
                            pass
                    finally:
                        connection.close()
                    continue
                try:
                    send_json_message(connection, standing_publication)
                    # the handle stays valid as long as the arena lives; nothing to
                    # wait for — the next clone can be served immediately.
                except Exception as e:  # noqa: BLE001 — a bad clone must not kill serving
                    # Tell the requester WHY (it try/excepts into its stock-loader
                    # fallback immediately instead of waiting out a dead socket).
                    try:
                        send_json_message(connection, {"error": str(e)})
                    except Exception:  # noqa: BLE001 — peer may already be gone
                        pass
                    logger.error(f"[flashboot][seed] :{port} clone session failed: {e}")
                finally:
                    connection.close()
        except Exception as e:  # noqa: BLE001 — never crash the serving process
            logger.error(f"[flashboot][seed] :{port} server failed: {e}", exc_info=True)
        finally:
            server.close()

    thread = threading.Thread(target=_serve, name=f"flashboot-seed-{port}", daemon=True)
    thread.start()
    keepalive["thread"] = thread
    return keepalive


# The rendezvous is engine-agnostic: the rdma engine's info carries rkeys and QP cards,
# the handle engine's a shareable memory handle. So the server validates only the three
# keys they have in common.
