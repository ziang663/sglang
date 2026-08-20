"""The joining side of every session shape — what a clone or a member calls.

Three shapes, and the differences between them are about COORDINATION, not about
transports: asymmetric (a seed publishes, clones take) against symmetric (everybody
arrives, everybody leaves with the full membership), and one round trip against a
collection window.

    fetch_arena_advertisement   take a seed's standing publication
    request_rdma_pull           one exchange for a credential that cannot stand
    collective_rendezvous       register for a symmetric round, get everyone's

Each has a completion call, which is a notification and a deterministic teardown — never
the ordering. Data readiness is carried by the transports' own signals.
"""
from __future__ import annotations

import logging
import socket
import time

from flashboot.rendezvous.arena_advertisement import ArenaAdvertisement
from flashboot.rendezvous.member_registration import MemberRegistration
from flashboot.rendezvous.message_framing import (
    receive_json_message,
    send_json_message,
)

logger = logging.getLogger(__name__)


def _connect_with_retry(seed_host, port, connect_timeout):
    """TCP-connect to the seed's rendezvous, retrying until it is up (it binds
    early, so this is quick) or ``connect_timeout`` elapses."""
    connection = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    connection.settimeout(connect_timeout)
    deadline = time.monotonic() + connect_timeout
    while True:
        try:
            connection.connect((str(seed_host), int(port)))
            return connection
        except (ConnectionRefusedError, OSError) as e:
            if time.monotonic() > deadline:
                raise TimeoutError(
                    f"[flashboot][clone] seed {seed_host}:{port} not reachable within "
                    f"{connect_timeout:.0f}s ({type(e).__name__}: {e}) — set "
                    f"FB_SEED_CONNECT_TIMEOUT_S to adjust") from e
            time.sleep(0.05)


def fetch_arena_advertisement(seed_host, port, timeout_s=1800.0, connect_timeout_s=None):
    """CLONE (single instance): connect to ``seed_host:port``, send the plain hello
    and receive the seed's ArenaAdvertisement. Returns ``(advertisement, open_socket)``
    — nothing further crosses the socket for the handle transports, so the caller may
    close it right after.

    ``connect_timeout_s`` bounds the CONNECT phase only (TCP connect retries + the
    rendezvous messages on the socket): a seed that never shows up fails the clone
    fast — surfacing as a TimeoutError the loader's fallback catches — instead of
    spinning for the full transfer ``timeout_s``. Defaults to ``timeout_s`` when unset.
    NOTE: if clones are launched before the seed is expected to be ready (the seed
    serves only after its own fill), size this to cover the seed's startup."""
    connect_timeout = float(connect_timeout_s) if connect_timeout_s else float(timeout_s)
    connection = _connect_with_retry(seed_host, port, connect_timeout)
    send_json_message(connection, {"world": 1})
    message = receive_json_message(connection)
    if isinstance(message, dict) and message.get("error"):
        raise RuntimeError(f"[flashboot][clone] seed {seed_host}:{port} reported: "
                           f"{message['error']}")
    advertisement = ArenaAdvertisement.from_wire(message)
    return advertisement, connection


def request_rdma_pull(seed_host, port, my_info, timeout_s=1800.0,
                      connect_timeout_s=None):
    """CLONE (direct 1-to-1 rdma pull): send our arena address, rkey and queue-pair card
    to the seed and receive the same about its arena. Returns ``(seed_info, socket)``.

    This is the WHOLE rendezvous of a 1-to-1 rdma pull — one round trip, because
    there is nothing to synchronise: the seed's bytes are all valid before we arrive, so
    no collective round, no progress counter, no chunk pipeline, and no fixed
    broadcast_rank. Both ends bring their link up from the exchanged cards and one READ
    moves the arena. Report :func:`rdma_pull_complete` when it lands — the seed serves
    clones sequentially and re-mints its card per clone, so it waits for that.

    Two callers: a clone whose ``FLASHBOOT_TRANSPORT`` is rdma with a single clone, and
    a clone whose handle import failed on a seed that advertised ``rdma_offered``."""
    connect_timeout = float(connect_timeout_s) if connect_timeout_s else float(timeout_s)
    connection = _connect_with_retry(seed_host, port, connect_timeout)
    send_json_message(connection, {"world": 1, "rdma": my_info})
    connection.settimeout(float(timeout_s))
    message = receive_json_message(connection)
    if isinstance(message, dict) and message.get("error"):
        connection.close()
        raise RuntimeError(f"[flashboot][clone] seed {seed_host}:{port} refused the "
                           f"direct rdma pull: {message['error']}")
    seed_info = dict(message["rdma"])
    logger.info(f"[flashboot][clone] direct rdma pull: seed offers "
                f"{float(seed_info.get('arena_size', 0)) / 1e9:.1f}GB")
    return seed_info, connection


def rdma_pull_complete(connection) -> None:
    """CLONE (direct 1-to-1 rdma pull): tell the seed the READ landed, so it may
    re-mint its card and serve the next clone. Closes the socket."""
    try:
        send_json_message(connection, {"done": 1})
    finally:
        connection.close()


def collective_rendezvous(host, port, my_info, world, group, timeout_s=1800.0,
                          connect_timeout_s=None):
    """MEMBER (rdma collectives): register ``my_info`` (rank + arena rkey/addr/size +
    progress counter + the pre-built QP cards — built by
    :mod:`flashboot.transport.rdma_data_plane`) at the collective rendezvous on
    ``host:port`` and block until the server collected all ``world`` infos, at which
    point the FULL membership is broadcast to every member at once. Returns
    ``(membership, open_socket)`` — ``membership`` is the info list sorted by rank.
    After the transfer report :func:`collective_complete` on the socket (completion
    notification + deterministic teardown, NOT ordering)."""
    connect_timeout = float(connect_timeout_s) if connect_timeout_s else float(timeout_s)
    connection = _connect_with_retry(host, port, connect_timeout)
    send_json_message(connection, MemberRegistration(
        world=int(world), group=str(group), info=my_info).to_wire())
    connection.settimeout(float(timeout_s))  # the wait spans the collection window
    message = receive_json_message(connection)
    if isinstance(message, dict) and message.get("error"):
        connection.close()
        raise RuntimeError(f"[flashboot][collective] {host}:{port} rejected the "
                           f"rendezvous: {message['error']}")
    membership = [dict(info) for info in message["membership"]]
    logger.info(f"[flashboot][collective] rank {my_info.get('rank')}/{world}: "
                f"membership of {len(membership)} received")
    return membership, connection


def collective_complete(connection) -> None:
    """MEMBER (rdma collectives): notify the rendezvous server our transfer finished
    (telemetry + deterministic teardown — data readiness is carried by the rdma
    progress counters). Closes the socket."""
    try:
        send_json_message(connection, {"done": 1})
    finally:
        connection.close()
