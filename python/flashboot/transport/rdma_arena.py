"""The rdma family: an arena read over one-sided RDMA READ.

No handle exists here. The owner registers its arena for remote reads and publishes an
address plus an rkey plus a queue-pair card; the puller reads it straight into its own
arena over InfiniBand or RoCE (GPUDirect). One round trip on the rendezvous is the
whole handshake, because there is nothing to synchronise: a 1-to-1 pull has no hop to
overlap, so no progress counter and no chunk pipeline — measured, one whole-arena READ
and 228 chunked READs reach the same bandwidth.

The credential cannot be a standing one. An RC queue pair cannot be re-pointed at a new
peer while it is in RTS, and the nixl backend has the mirror-image problem (a SIGKILLed
peer leaves its UCX worker unable to bring up a queue pair for the next one), so every
puller after the first gets a freshly minted card — see
:meth:`flashboot.transport.rdma_data_plane.RdmaMembership.renew_cards`. The arena's own
registration survives that, so serving N pullers costs ONE registration.
"""
from __future__ import annotations

from flashboot import rendezvous
from flashboot.transport import rdma_data_plane as rdma_collectives
from flashboot.transport.contracts import (
    ArenaRequirements,
    ArenaService,
    Publication,
)
from flashboot.transport.rdma_engine import create_endpoint


class RdmaSeedArena:
    """A seed's arena, readable over the wire once the cards have been exchanged.

    Registration of the LOCAL arena has to wait for the arena to exist, so unlike the
    nvlink family nothing expensive happens at open time: what open does is remember
    where the seed is. The round trip and the registration both happen in
    :meth:`fetch_whole_arena_into`, which the loader calls once its arena is allocated.
    """

    def __init__(self, seed_host: str, seed_port: int, device: int,
                 byte_count: int, timeout_s: float, connect_timeout_s: float,
                 reason: str = "") -> None:
        self.byte_count = int(byte_count)
        self.description = "rdma" + (f" ({reason})" if reason else "")
        self._seed_host = str(seed_host)
        self._seed_port = int(seed_port)
        self._device = int(device)
        self._timeout_s = float(timeout_s)
        self._connect_timeout_s = float(connect_timeout_s)

    def fetch_whole_arena_into(self, arena_handle) -> None:
        membership = rdma_collectives.prepare_direct_membership(
            create_endpoint(self._device), rank=1, arena_ptr=arena_handle.base,
            arena_size=arena_handle.size)
        seed_publication, connection = rendezvous.request_rdma_pull(
            self._seed_host, self._seed_port, membership.info,
            timeout_s=self._timeout_s, connect_timeout_s=self._connect_timeout_s)
        try:
            rdma_collectives.connect_topology(
                membership, [seed_publication, membership.info])
            rdma_collectives.direct_pull(membership, seed_publication,
                                         self._timeout_s)
        except Exception:
            try:
                connection.close()  # no "done": the seed logs this session as failed
            except Exception:  # noqa: BLE001
                pass
            raise
        rendezvous.rdma_pull_complete(connection)

    def close(self) -> None:
        # Nothing outlives the pull: unlike a chain hop, no downstream reads from us,
        # and the membership went out of scope with fetch_whole_arena_into.
        pass


class RdmaArenaPublisher:
    """The seed's half of a wire pull: the arena registered for remote reads, one queue
    pair, and one freshly minted card per puller.

    Costs one registration on the seed's critical path plus one queue pair. That buys two things: a same-node deployment stops
    being the only one a handle can serve, and a 1-to-1 puller no longer needs the
    collective machinery — no world size on the seed, no fixed position on the puller,
    no chunk pipeline for a transfer with nothing to overlap.
    """

    def __init__(self, arena_handle, device: int) -> None:
        self._arena_size = int(arena_handle.size)
        self._device = int(device)
        self._membership = rdma_collectives.prepare_direct_membership(
            create_endpoint(device), rank=0, arena_ptr=arena_handle.base,
            arena_size=arena_handle.size)
        self._pullers_served = 0

    @property
    def membership(self):
        """The registered arena and its queue pair — pullers READ this for as long as
        they are pulling, so the caller keeps this publisher alive."""
        return self._membership

    def standing_publication(self) -> Publication:
        """Size and device, and the offer itself. The credential for a wire read is an
        rkey bound to a queue pair, which is minted per puller, so there is nothing
        standing to publish beyond saying that remote reads are on offer here."""
        # This runs on the SERVER's thread, which has no current device of its own, so
        # bind it before touching the driver. First call on that thread, so it covers the
        # per-puller exchanges that follow too.
        _bind_current_device(self._device)
        return rendezvous.ArenaAdvertisement(
            arena_size=self._arena_size, device=self._device, transport="rdma",
            credential={"rdma_offered": True}).to_wire()

    def offers_exchange(self) -> bool:
        return True

    def exchange_with_puller(self, puller_publication: Publication) -> Publication:
        """Bring our end of the link up against this puller and return our card.

        Safe to re-mint here because the rendezvous serves pullers SEQUENTIALLY and
        waits for each one's completion: a queue pair still being read is never
        re-targeted.
        """
        if self._pullers_served:
            self._membership.renew_cards()
        self._pullers_served += 1
        rdma_collectives.connect_topology(
            self._membership, [self._membership.info, dict(puller_publication)])
        return self._membership.info

    def close(self) -> None:
        release_peers = getattr(self._membership.endpoint, "release_peers", None)
        if release_peers is not None:
            try:
                release_peers()
            except Exception:  # noqa: BLE001 — teardown must not raise into a load
                pass


class RdmaTransport:
    """Arenas moved over one-sided RDMA READ."""

    name = "rdma"

    def arena_requirements(self) -> ArenaRequirements:
        # A registered region can be plain cudaMalloc memory; nothing to export.
        return ArenaRequirements(fabric_exportable=False)

    def publisher_for(self, arena_handle, device: int) -> RdmaArenaPublisher:
        return RdmaArenaPublisher(arena_handle, device)

    def serve_arena(self, arena_handle, device: int, port: int, clone_count: int,
                    timeout_s: float) -> ArenaService:
        """ONE server, both cardinalities. Which one a clone wanted is in its opening
        message — a single instance takes one direct exchange, K instances register for
        a broadcast round — so the seed does not have to be told the count before anyone
        has dialled in. ``clone_count`` is accepted and ignored for exactly that reason:
        it used to select the server, and a deployment that still passes it is not
        wrong, just no longer heard."""
        del clone_count
        publisher = self.publisher_for(arena_handle, device)
        chain_seed = _RdmaChainSeed(arena_handle, device)
        server = rendezvous.start_seed_server(port, device, publisher, timeout_s,
                                              chain_seed=chain_seed)
        return ArenaService(server=server, backing=(publisher, chain_seed.membership))

    def join_chain(self, arena_handle, device: int, clone_count: int,
                   requested_position: int) -> "RdmaChainParticipant":
        return RdmaChainParticipant(arena_handle, device, clone_count,
                                    requested_position)

    def join_all_gather(self, arena_handle, device: int, rank: int, world: int,
                        segment_alignment: int) -> "RdmaGatherParticipant":
        return RdmaGatherParticipant(arena_handle, device, rank, world,
                                     segment_alignment)

    def open_seed_arena(self, seed_host: str, seed_port: int, device: int,
                        timeout_s: float, connect_timeout_s=None):
        """THE SEED'S CHOICE WINS, even over this process's own preference.

        A seed publishes its transport before any clone connects, so the clone follows
        what it finds rather than what it would have picked: a seed that published a
        memory handle is read through that handle (one copy-engine copy, no NIC), and
        this transport's own wire path is used only when the seed actually published a
        wire offer. Preferring the wire here would quietly take a same-node clone off the
        handle — the handle is fastest, the wire's default backend is close behind,
        and its raw backend is far behind on a same-node pair.

        The nvlink family does the mirror image of this: see
        :meth:`flashboot.transport.nvlink_arena.NvlinkTransport.open_seed_arena`.
        """
        from flashboot.transport.nvlink_arena import (
            NVLINK_TRANSPORTS,
            NvlinkTransport,
        )

        connect_timeout = float(connect_timeout_s or timeout_s)
        advertisement, connection = rendezvous.fetch_arena_advertisement(
            seed_host, seed_port, timeout_s, connect_timeout_s=connect_timeout)
        connection.close()   # the advertisement was the whole exchange
        if (advertisement.transport in NVLINK_TRANSPORTS
                and advertisement.credential.get("memory_handle")):
            return NvlinkTransport(advertisement.transport).open_seed_arena(
                seed_host, seed_port, device, timeout_s,
                connect_timeout_s=connect_timeout)
        if not advertisement.credential.get("rdma_offered"):
            raise RuntimeError(
                f"[flashboot][clone] seed {seed_host}:{seed_port} published neither a "
                f"memory handle nor a wire offer, so there is nothing to read: its "
                f"arena is not registered for remote reads and it exported no handle")
        return RdmaSeedArena(seed_host, seed_port, device,
                             byte_count=int(advertisement.arena_size),
                             timeout_s=timeout_s, connect_timeout_s=connect_timeout)


class RdmaGatherParticipant:
    """An all-gather member reading every other owner's segment over the wire.

    The mode (``FB_ALLGATHER_MODE``) is decided here because it is a WIRE concern: the
    default reads every owner's segment concurrently, which a non-blocking fabric ingests
    at line rate; ``ring`` trades that for one ingress and one egress peer per member,
    which is the right answer only where the fabric cannot carry concurrent all-to-all
    flows. The chosen mode rides the membership so a drifted knob fails loudly instead of
    deadlocking on cards nobody minted.
    """

    def __init__(self, arena_handle, device: int, rank: int, world: int,
                 segment_alignment: int) -> None:
        self._membership = rdma_collectives.prepare_allgather_membership(
            create_endpoint(device), rank=rank, world=world,
            arena_ptr=arena_handle.base, arena_size=arena_handle.size,
            segment_alignment=segment_alignment)

    @property
    def publication(self) -> Publication:
        return self._membership.info

    @property
    def membership(self):
        """Peers may still be READing our native segment after our own gather returned —
        direct mode has no end barrier — so the endpoint, its registrations and its queue
        pairs have to outlive this call. The caller keeps this object alive."""
        return self._membership

    def connect(self, members: list) -> None:
        rdma_collectives.connect_topology(self._membership, members)

    def gather_from(self, members: list, timeout_s: float) -> None:
        rdma_collectives.all_gather(self._membership, members, timeout_s)

    def close(self) -> None:
        release_peers = getattr(self._membership.endpoint, "release_peers", None)
        if release_peers is not None:
            try:
                release_peers()
            except Exception:  # noqa: BLE001 — teardown must not raise into a load
                pass


class RdmaChainParticipant:
    """A clone's place in a wire chain: pull the predecessor's arena chunk by chunk,
    gated by its progress counter, publishing our own as chunks land.

    The position is FIXED by the launcher, not assigned by arrival. One RC queue pair is
    pre-built per link before the rendezvous — the cards ride the registration, which is what
    makes the membership broadcast the entire handshake — so the position has to be known
    before this member contacts anybody. A single clone is the one case that needs no
    explicit position: it can only be position 1.
    """

    def __init__(self, arena_handle, device: int, clone_count: int,
                 requested_position: int) -> None:
        position = requested_position or (1 if clone_count == 1 else 0)
        if not 1 <= position <= clone_count:
            raise RuntimeError(
                f"[flashboot][clone] the rdma chain needs this clone's fixed position: "
                f"set --model-loader-extra-config '{{\"broadcast_rank\":i}}' with "
                f"1 <= i <= broadcast_world={clone_count}, got {requested_position}")
        self._position = position
        self._world = int(clone_count) + 1   # the seed is collective rank 0
        self._membership = rdma_collectives.prepare_chain_membership(
            create_endpoint(device), rank=position, world=self._world,
            arena_ptr=arena_handle.base, arena_size=arena_handle.size)

    @property
    def membership(self):
        """Our successor keeps READing our arena and our progress counter for the rest of
        its pull, so this has to outlive the load."""
        return self._membership

    def receive_broadcast(self, rendezvous_host: str, rendezvous_port: int,
                          timeout_s: float, connect_timeout_s=None) -> None:
        members, connection = rendezvous.collective_rendezvous(
            rendezvous_host, rendezvous_port, self._membership.info, self._world,
            group="chain", timeout_s=timeout_s, connect_timeout_s=connect_timeout_s)
        try:
            rdma_collectives.connect_topology(self._membership, members)
            rdma_collectives.chain_pull(self._membership, members, timeout_s)
        except Exception:
            try:
                connection.close()  # no completion: the seed logs this link as failed
            except Exception:  # noqa: BLE001
                pass
            raise
        rendezvous.collective_complete(connection)

    def close(self) -> None:
        pass


class _RdmaChainSeed:
    """The seed's half of a broadcast, as the three per-round things the seed server
    asks of it (see :func:`flashboot.rendezvous.start_seed_server`).

    The seed is collective rank 0: it registers its arena, pre-builds the queue pair for
    the (0, 1) link, publishes NO progress counter — its bytes are always valid, the same
    convention the nvlink chain expresses as "no flags" — and never posts a READ. Its
    queue pair reaches RTS inside the server's connect step, before any clone sees the
    membership.

    None of that depends on how long the chain is: rank 0 has exactly one link whatever
    the world, so the arena registration (the expensive part, and the one thing that must
    not be paid per round) happens once here at construction, and the world only arrives
    later, to be advertised. That is what lets one server decide per connection whether a
    clone wanted a single-instance pull or a broadcast, instead of the seed having to be
    told the clone count before anybody has dialled in.
    """

    def __init__(self, arena_handle, device: int) -> None:
        # world=2 is a placeholder for the advertisement only: the connections rank 0
        # takes part in are [(0, 1)] for every world, so nothing built here changes when
        # the real world arrives.
        self._membership = rdma_collectives.prepare_chain_membership(
            create_endpoint(device), rank=0, world=2,
            arena_ptr=arena_handle.base, arena_size=arena_handle.size)

    @property
    def membership(self):
        """Kept alive by the caller: clones READ this arena and its registration for as
        long as they are pulling."""
        return self._membership

    def local_member(self, world: int) -> dict:
        """This seed's publication as member rank 0 of a chain of ``world``."""
        self._membership.world = int(world)
        self._membership.info["world"] = int(world)
        return dict(self._membership.info)

    def connect(self, members) -> None:
        rdma_collectives.connect_topology(self._membership, members)

    def renew(self) -> dict:
        # Serving clone after clone: an RC queue pair cannot be re-pointed at a new peer
        # in RTS, so each round gets freshly reset cards (same queue-pair number, same
        # arena registration — see RdmaMembership.renew_cards).
        return self._membership.renew_cards()


def _bind_current_device(device: int) -> None:
    """Make ``device`` current on whatever thread is asking. The rendezvous serves from a
    thread of its own and deliberately knows nothing about CUDA, so binding it is the
    transport's job."""
    import torch

    torch.cuda.set_device(int(device))
