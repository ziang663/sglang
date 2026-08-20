"""Length-prefixed JSON over TCP — the whole rendezvous wire format.

Only small messages cross this: an arena's credential, a membership list, a completion
notice. The weights themselves never do.

The cap exists because a 4-byte length prefix otherwise lets any peer ask a server to
allocate up to 4GiB before a single payload byte has been validated. The only message
that grows with scale is the collective membership broadcast, quadratically: a mesh
all-gather of W members carries W*(W-1) queue-pair cards, ~1KB each with NIXL agent
metadata — 5MB at W=64. 64MiB therefore rejects garbage and mismatched protocols without
bounding any membership anyone would launch.
"""
from __future__ import annotations

import json

_MAX_MESSAGE_BYTES = 64 << 20


def receive_exact_bytes(connection, nbytes: int) -> bytes:
    buffer = b""
    while len(buffer) < nbytes:
        chunk = connection.recv(nbytes - len(buffer))
        if not chunk:
            raise ConnectionError("[flashboot] peer closed mid-message")
        buffer += chunk
    return buffer


def send_json_message(connection, obj) -> None:
    payload = json.dumps(obj).encode()
    connection.sendall(len(payload).to_bytes(4, "big") + payload)


def receive_json_message(connection):
    length = int.from_bytes(receive_exact_bytes(connection, 4), "big")
    if length > _MAX_MESSAGE_BYTES:
        raise ConnectionError(
            f"[flashboot] rendezvous message announces {length} bytes, over the "
            f"{_MAX_MESSAGE_BYTES} cap — the peer is not speaking this protocol "
            f"(wrong port? something else bound it?)")
    return json.loads(receive_exact_bytes(connection, length).decode())
