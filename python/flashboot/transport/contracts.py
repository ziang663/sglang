"""What a transport promises the layers above it.

The point of these three protocols is that NOTHING above this package should have to
name a transport. Before them the loader asked "is this rdma?" in ten places — to
decide how to allocate the arena, which server to start, which engine to gather with,
and which pull path to take — so every new transport meant another branch in every one
of them.

The vocabulary is deliberately small, and each piece exists because a real difference
between the two families demanded it:

  * :class:`ArenaRequirements` — a fabric handle can only be exported from CUDA-VMM
    memory, so the ALLOCATION depends on the transport. The loader asks for
    requirements instead of asking whether the transport is called "fabric".
  * :class:`SeedArena` — opening a remote arena and reading from it are separate
    steps, because for the nvlink family the opening is expensive and must happen
    EARLY: ``cudaIpcOpenMemHandle`` measured ~1ms in a bare process but ~5s at the
    pull point (E00211), and while it is in flight the driver serializes every other
    CUDA call in the process, so it cannot be overlapped in-process. The clone
    therefore opens before it builds the model and reads afterwards.
  * :class:`ArenaPublisher` — a seed hands out two different kinds of credential. The
    nvlink family has a STANDING one: a 64-byte handle, built once, importable by any
    number of pullers, valid as long as the arena lives. The rdma family has none —
    the credential is a queue pair, and an RC queue pair cannot be re-pointed at a new
    peer while it is in RTS, so each puller needs a freshly minted card. Both shapes
    fit "a standing publication, plus a per-puller exchange for transports that need
    one".

A ``Publication`` is an opaque dict. The rendezvous forwards it without looking
inside, which is what lets a new transport be added without touching the rendezvous
or anything above it.
"""
from __future__ import annotations

import dataclasses
from typing import Dict, Optional, Protocol, runtime_checkable

# What a puller needs in order to read some other process's arena. Opaque above this
# package: a memory handle for the nvlink family, an address + rkey + queue-pair card
# for the rdma family.
Publication = Dict[str, object]


@dataclasses.dataclass(frozen=True)
class ArenaRequirements:
    """What a transport needs of the arena ALLOCATION itself.

    ``fabric_exportable`` switches the allocation from ``cudaMalloc`` to
    fabric-exportable CUDA-VMM memory. Only a seed serving over an NVLink fabric needs
    it; ipc exports plain cudaMalloc memory and the rdma family registers whatever it
    is given.
    """
    fabric_exportable: bool = False


@runtime_checkable
class SeedArena(Protocol):
    """Another process's arena, opened and ready to be read from.

    ``byte_count`` is what the owner says its arena is, so the puller can compare it
    with its own before moving anything — seed and clone derive the arena layout
    independently from their own checkpoint headers, and a size mismatch is the only
    cheap signal that they are not loading the same thing.
    """

    byte_count: int

    def fetch_whole_arena_into(self, arena_handle) -> None:
        """Copy the owner's entire arena into ``arena_handle``'s memory. Blocking."""

    def close(self) -> None:
        """Drop whatever this process holds for the read (a mapping, a queue pair).
        The owner keeps owning its memory."""


@runtime_checkable
class ArenaPublisher(Protocol):
    """The seed's half: what it hands out so others can read its arena.

    Whatever backs the publication — an exported handle's arena, a registered memory
    region, a queue pair — is owned by this object, so the caller keeps it alive for as
    long as any puller may still be reading.
    """

    def standing_publication(self) -> Publication:
        """The credential every puller gets, built once. Called AFTER the server's port
        is open, so building it stays off the seed's critical path while the port is
        already accepting."""

    def exchange_with_puller(self, puller_publication: Publication) -> Publication:
        """One puller's turn, for transports whose credential cannot be standing: pair
        with that puller and return what it needs. Raises when this transport offers no
        such exchange."""

    def offers_exchange(self) -> bool:
        """Whether :meth:`exchange_with_puller` will work. A handle seed answers True
        when it ALSO registered its arena for wire reads, which is what keeps a puller
        that cannot import the handle — different node, or a container sharing neither
        the IPC namespace nor the GPUs — off the read-it-from-disk path."""

    def close(self) -> None:
        """Release the publication's backing. After this no puller can read the arena."""


@runtime_checkable
class ArenaGatherParticipant(Protocol):
    """This process's half of an all-gather: every member holds ``1/K`` of the arena and
    ends up holding all of it.

    Three steps, in this order for a reason. The participant is BUILT before the
    rendezvous, because building it is what declares this member's segment ready — and
    for the rdma family the queue-pair cards have to be minted before the hello that
    carries them. It is CONNECTED once the membership is known, before any byte moves.
    Only then does it gather.

    The gather itself belongs to the transport, not to the caller: the rdma family posts
    every peer's segment read at once and waits once, so K-1 flows are in flight together
    (measured — a per-peer wait would serialize them), and it has a second, step-gated
    ring mode for fabrics that cannot carry concurrent all-to-all flows. The nvlink family
    imports and copies one owner at a time. Neither shape survives being flattened into
    the other.
    """

    publication: Publication

    def connect(self, members: list) -> None:
        """Pair this member's links with the broadcast membership. A no-op for a
        transport with nothing to pair."""

    def gather_from(self, members: list, timeout_s: float) -> None:
        """Fill every byte this member does not natively own from whoever owns it.
        Blocking; returns once this member's arena is whole."""

    def close(self) -> None:
        ...


@dataclasses.dataclass
class ArenaService:
    """A seed's live offer of its arena: the rendezvous server, plus whatever backs
    what that server hands out.

    Both halves have to outlive the load, and for different reasons — the server is a
    daemon thread, and the backing (an exported handle's arena, a registered memory
    region, a queue pair, a chain's per-chunk flags) is being READ by clones for as long
    as they are pulling. Returning them together is what lets the loader keep one
    reference and not think about which transport needs what.
    """
    server: dict
    backing: object


@runtime_checkable
class ChainParticipant(Protocol):
    """This process's place in a chain broadcast: ``seed -> clone1 -> ... -> cloneK``,
    where every hop streams CONCURRENTLY rather than waiting for its upstream to finish.

    Only ONE mechanism genuinely differs between the two families, and it is here: how a
    hop learns that its upstream's chunk k is valid. The nvlink family publishes a u32
    CUDA flag per chunk and the downstream copy stream waits on it in hardware
    (``cuStreamWaitValue32``, stream-ordered). The rdma family keeps a u32 counter in host
    memory and the downstream polls it with small READs. Both satisfy the same sentence:

        after waiting for chunk k, a read will not observe bytes past the upstream's
        chunk k.

    A second difference follows from the credential model, and is why the rendezvous is
    behind this interface too rather than in the collective above it. A handle is a
    STANDING credential, so a clone can export before any byte is valid and the seed can
    assign chain positions BY ARRIVAL. A queue-pair card is not: the rdma family mints one
    card per link before the registration that carries it, so a member's position must be fixed
    by the launcher and known before it contacts anyone. One shape cannot be expressed in
    the other, and pretending otherwise would break the pipelining that is the whole point
    of a chain.
    """

    def receive_broadcast(self, rendezvous_host: str, rendezvous_port: int,
                          timeout_s: float,
                          connect_timeout_s: Optional[float] = None) -> None:
        """Take our place in the chain and fill our arena from our upstream, publishing
        our own readiness per chunk as it lands so our successor streams concurrently.
        Blocking until our whole arena is valid."""

    def close(self) -> None:
        ...


@runtime_checkable
class Transport(Protocol):
    """One way to move an arena from the process that has it to one that wants it."""

    name: str

    def arena_requirements(self) -> ArenaRequirements:
        ...

    def publisher_for(self, arena_handle, device: int) -> ArenaPublisher:
        ...

    def serve_arena(self, arena_handle, device: int, port: int, clone_count: int,
                    timeout_s: float) -> ArenaService:
        """Seed side: start offering this arena on ``port``.

        ``clone_count`` is how many clone INSTANCES start together — 1 for the plain
        single-clone case, K > 1 for a chain broadcast. It is a property of the
        deployment, not of the transport, which is why the caller supplies it and the
        transport decides what that implies for its own machinery."""

    def join_chain(self, arena_handle, device: int, clone_count: int,
                   requested_position: int) -> ChainParticipant:
        """Clone side: build this instance's place in a chain of ``clone_count`` clones.

        ``requested_position`` is 1..clone_count where the launcher pinned it, or 0 to
        let the seed assign one by arrival — which a transport may refuse if its
        credentials are minted per link (see :class:`ChainParticipant`)."""

    def open_seed_arena(self, seed_host: str, seed_port: int, device: int,
                        timeout_s: float,
                        connect_timeout_s: Optional[float] = None) -> SeedArena:
        """Puller side: get the seed's credential over the rendezvous and turn it
        into something readable. Does the expensive part of opening NOW."""

    def join_all_gather(self, arena_handle, device: int, rank: int, world: int,
                        segment_alignment: int) -> ArenaGatherParticipant:
        """Build this member's half of an all-gather. Call it only once this member's own
        segment is filled: doing so is what declares the segment ready."""
