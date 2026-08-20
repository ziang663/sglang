"""The ordering a transfer follows, for every transport alike.

What is left here after the transports were given their own layer is the part that is
genuinely NOT about how bytes cross: the sequence a group of processes has to follow to
end up with the right bytes.

    all_gather      join, rendezvous, connect, gather, report — one implementation

Two things this package used to hold moved, and where they went says what each of them
really was:

  * the two ENGINES (one-sided READ; imported handles and the copy engine) are
    per-transport implementations of "move these bytes", so they sit in
    :mod:`flashboot.transport` beside the transports they belong to;
  * the segment split, the link topology and the chunk geometry are pure integer math
    that every layer needs and that depends on nothing — so they sit at the bottom, in
    :mod:`flashboot.utils.arena_partitioning`.

The chain broadcast has no module here, deliberately. Its ordering cannot be separated
from its readiness signal — a hardware CUDA flag against a polled host counter, with two
different rendezvous shapes following from the two credential models — so each transport
owns its own, behind
:class:`flashboot.transport.contracts.ChainParticipant`. Forcing a shared implementation
would have meant giving up either the concurrent hops or the ring mode, both measured.
"""
