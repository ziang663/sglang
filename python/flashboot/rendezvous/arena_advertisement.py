"""What a member publishes about its arena — the one payload, for every transport.

Both families used to describe themselves differently: the nvlink family sent this
dataclass's fields plus a couple of pipeline keys, the rdma family a flat dict of
addresses, rkeys and queue-pair cards, and the two agreed on ``arena_size`` and ``rank``
only by habit — so anything reading a membership had to know which family had written
it. They send the same class now, and the split runs along the one line that is real:

  * the fields BELOW are what any member of any collective has: where it sits, how big
    its arena is, in what units it will be moved. The rendezvous reads exactly these,
    and every one of them means the same thing whatever carries the bytes.
  * ``credential`` is how to actually read those bytes, and it stays an opaque dict — a
    hex memory handle plus a flags handle for nvlink, an address, an rkey, a progress
    counter and a card per link for rdma. Nothing outside the owning engine looks in.

The version is checked on receipt because a mismatched PAIR is the one failure with no
symptom: the arena layout mirrors the engine's runtime tensor layout, so two ends on
different flashboot builds move every byte successfully and produce WRONG weights.
Failing here names the actual cause instead of letting it surface as garbage output.
"""
from __future__ import annotations

import dataclasses

# 7: the publication became one class for both families, with the transport-specific
# half moved under ``credential``. A 6 and a 7 cannot read each other's membership.
ADVERTISEMENT_VERSION = 7


@dataclasses.dataclass
class ArenaAdvertisement:
    """One member's publication: its place in the collective, its arena's size and
    chunking, and the credential a peer needs to read it."""
    arena_size: int
    rank: int = 0
    world: int = 1
    device: int = 0
    transport: str = "fabric"
    # The unit a pipelined broadcast moves and signals in. Zero for a member of a
    # collective that does not chunk (a 1-to-1 pull, an all-gather).
    chunk_bytes: int = 0
    num_chunks: int = 0
    credential: dict = dataclasses.field(default_factory=dict)
    version: int = ADVERTISEMENT_VERSION

    def to_wire(self) -> dict:
        return dataclasses.asdict(self)

    @classmethod
    def from_wire(cls, obj: dict) -> ArenaAdvertisement:
        version = int(obj.get("version", -1))
        if version != ADVERTISEMENT_VERSION:
            raise RuntimeError(
                f"[flashboot] arena advertisement wire version {version} != this "
                f"process's {ADVERTISEMENT_VERSION} — the two ends run different "
                f"flashboot builds, which the sharded_state layout does not allow "
                f"(update both, or re-export the checkpoint)")
        return cls(arena_size=int(obj["arena_size"]),
                   rank=int(obj.get("rank", 0)),
                   world=int(obj.get("world", 1)),
                   device=int(obj.get("device", 0)),
                   transport=str(obj.get("transport", "fabric")),
                   chunk_bytes=int(obj.get("chunk_bytes", 0)),
                   num_chunks=int(obj.get("num_chunks", 0)),
                   credential=dict(obj.get("credential") or {}),
                   version=version)
