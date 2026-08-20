"""What a member sends to enter one round of a collective, and what disqualifies it.

The sibling of :mod:`flashboot.rendezvous.arena_advertisement`: that one is what a seed
publishes about its arena, this one is what a member says when it registers for a
round. Both are one message, one dataclass, ``to_wire``/``from_wire``.

The envelope is deliberately thin. ``info`` is the member's publication — arena size
and whatever credential its transport needs, rkeys and queue-pair cards for the wire
family, a shareable memory handle for the nvlink family — and the rendezvous never
looks inside beyond the three keys both families share. Everything else here is what
the SERVER needs to run a round: which collective this is (``group``), how many members
it is collecting (``world``), and where this one sits (``rank``, read out of the
publication so a member cannot claim two different positions).

Registration is separated from admission on purpose. :meth:`MemberRegistration.
from_wire` decides whether a message is a registration at all; :meth:`MemberRegistration
.problem_joining` decides whether a well-formed one may enter THIS round. The first is
about the sender, the second about the round in progress, and a server that has already
collected members needs to tell those apart to say anything useful about a rejection.
"""
from __future__ import annotations

import dataclasses

# The rendezvous is engine-agnostic: the rdma family's publication carries rkeys and
# queue-pair cards, the nvlink family's a shareable memory handle. So only the three
# keys they have in common are ever validated here.
REQUIRED_INFO_KEYS = ("rank", "world", "arena_size")

# The key that says "this message is a registration". One place, because both the
# server routing a message and this class parsing one have to agree on it.
_REGISTRATION_MARKER = "collective"


@dataclasses.dataclass
class MemberRegistration:
    """One member asking to enter one round: its position, the membership size it
    believes it is joining, which collective, and its publication."""
    world: int
    group: str
    info: dict

    @property
    def rank(self) -> int:
        """Where the member sits, as stated by its publication — the only copy. The
        envelope does not carry a second one, so the two can never disagree."""
        return int(self.info["rank"])

    def to_wire(self) -> dict:
        return {_REGISTRATION_MARKER: 1, "world": int(self.world),
                "group": str(self.group), "info": self.info}

    @classmethod
    def matches(cls, payload) -> bool:
        """Whether ``payload`` is a registration at all — the question a server asks
        before it can route a message, and the reason it lives here: what a
        registration looks like is this class's business, and a caller that answered it
        for itself would be a second place that has to change when the shape does.

        Deliberately just the marker: a message that says it is a registration and gets
        the rest wrong should be REJECTED with a reason (:meth:`from_wire`), not quietly
        routed somewhere else as though it had been a different kind of message.
        """
        return bool(isinstance(payload, dict) and payload.get(_REGISTRATION_MARKER)
                    and isinstance(payload.get("info"), dict))

    @classmethod
    def from_wire(cls, obj) -> MemberRegistration:
        """Parse a registration, or raise ``ValueError`` naming what it is not.

        Structure only: whether this message is a registration and carries a
        publication with the keys every engine has. Whether it may join the round in
        progress is :meth:`problem_joining`.
        """
        if not cls.matches(obj):
            raise ValueError("expected a member registration carrying an info dict")
        info = obj["info"]
        missing = [key for key in REQUIRED_INFO_KEYS if key not in info]
        if missing:
            raise ValueError(f"member info is missing {missing} "
                             f"(it carries: {sorted(info)})")
        return cls(world=int(obj.get("world", 0)), group=obj.get("group"),
                   info=dict(info))

    def problem_joining(self, expected_world, expected_group) -> str | None:
        """Why this registration cannot enter the round in progress — a string — or
        None if it can.

        Every check RETURNS a reason instead of raising: the server sends it back so
        the member fails immediately with the cause, and a single mis-launched client
        can never take the server thread down and hang every member that already
        joined.

        All of these are launch mistakes rather than timing: waiting longer cannot
        turn a rank outside the membership, or a member counting a different world,
        into one that fits. Naming them at the door is what keeps them from surfacing
        as a KeyError deep in the data plane.
        """
        if expected_world is not None and self.world != expected_world:
            return (f"world mismatch: registration says {self.world} while collecting "
                    f"a membership of {expected_world}")
        if expected_group is not None and self.group != expected_group:
            return (f"group mismatch: registration says {self.group!r} while "
                    f"collecting {expected_group!r}")
        if int(self.info["world"]) != self.world:
            return (f"member {self.rank} says world={self.info['world']} in its "
                    f"publication but {self.world} in its registration")
        if not 0 <= self.rank < self.world:
            return f"member rank {self.rank} is outside 0..{self.world - 1}"
        arena_size = int(self.info["arena_size"])
        if arena_size <= 0:
            return f"member {self.rank} advertises arena_size={arena_size}"
        return None
