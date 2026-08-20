"""The nvlink family: an arena mapped from its owner and copied by the copy engine.

``ipc`` (same node, a ``cudaIpcGetMemHandle`` blob) and ``fabric`` (a CUDA-VMM
``CUmemFabricHandle``, importable by any GPU in the same IMEX domain, so it crosses
nodes) differ only in which 64-byte handle they export. The owner exports one; a puller
imports it and ONE device-to-device copy over NVLink moves the whole arena — no bounce
buffer, no NIC.

Two things this module keeps out of the layers above it.

**The import is done EARLY, not at pull time.** ``cudaIpcOpenMemHandle`` measured ~1ms
in a bare process but ~5s at the pull point (E00211: the slow state accumulates during
the model build), and it cannot be overlapped in-process — while it is in flight the
driver serializes every other CUDA call this process makes. So opening maps the arena
immediately and the mapping is carried until the copy.

**The fallback to the wire lives here.** A handle cannot always be imported: a puller on
another node, or in a container sharing neither the IPC namespace nor the GPUs, will
fail — and that failure IS the normal way of discovering it. If the seed also registered
its arena for remote reads, this module quietly returns a wire-backed source instead,
which beats the stock loader reading the whole checkpoint from disk by a wide margin.
Nothing above needs to know it happened; the reason goes in the log line.

One CUDA subtlety the failure path depends on: a failed ``cudaIpcOpenMemHandle`` leaves
its error in the CUDA context, and any later CUDA call in this process would report the
same error — which would kill the very recovery path we are taking. The native importer
consumes that error before raising, so the fallback is usable.
"""
from __future__ import annotations

import os
import time

from flashboot import rendezvous
from flashboot.transport import rdma_arena
from flashboot.transport.contracts import (
    ArenaRequirements,
    ArenaService,
    Publication,
)

NVLINK_TRANSPORTS = ("ipc", "fabric")


class ImportedSeedArena:
    """A seed's arena, already mapped into this process, waiting to be copied."""

    def __init__(self, transport_name: str, importer, byte_count: int) -> None:
        self.byte_count = int(byte_count)
        self.description = str(transport_name)
        self._importer = importer

    def fetch_whole_arena_into(self, arena_handle) -> None:
        """One copy-engine device-to-device copy of the whole arena.

        The mapping is released right after: releasing only drops THIS process's view of
        the seed's memory (the seed keeps owning it) and holds no state that cuda-graph
        capture could trip over, so nothing has to stay alive afterwards."""
        copy_started = time.perf_counter()
        self._importer.copy_from_peer(arena_handle.base, 0, self.byte_count)
        copy_seconds = time.perf_counter() - copy_started
        self.close()
        release_seconds = time.perf_counter() - copy_started - copy_seconds
        print(f"[flashboot][clone] {self.description} pull "
              f"{self.byte_count / 1e9:.1f}GB in {copy_seconds:.2f}s "
              f"({self.byte_count / 1e9 / max(copy_seconds, 1e-9):.1f}GB/s) "
              f"[release={release_seconds:.2f}s]", flush=True)

    def close(self) -> None:
        if self._importer is None:
            return
        importer, self._importer = self._importer, None
        try:
            importer.release()
        except Exception:  # noqa: BLE001 — teardown must not mask a copy error
            pass


class NvlinkArenaPublisher:
    """The seed's half: the 64-byte handle, plus — whenever this node can also serve
    remote reads — a wire offer beside it.

    The handle is exported ONCE and published forever: any number of pullers can import
    the same one, and it stays valid as long as the arena lives. The wire offer is an
    ADDITION for pullers that cannot import it, so a seed that can still hand out a
    perfectly good handle must not die because the optional half failed.
    """

    def __init__(self, transport_name: str, arena_handle, device: int,
                 wire_publisher=None) -> None:
        self._transport_name = str(transport_name)
        self._arena_handle = arena_handle
        self._device = int(device)
        self._wire_publisher = wire_publisher

    @property
    def wire_publisher(self):
        """The optional wire half, kept reachable so the caller can hold what backs it
        (a registered arena and a queue pair) alive for as long as pullers read it."""
        return self._wire_publisher

    def standing_publication(self) -> Publication:
        # This runs on the SERVER's thread, which has no current device of its own, so
        # bind it before touching the driver. First call on that thread, so it covers the
        # per-puller exchanges that follow too.
        _bind_current_device(self._device)
        native_arena = getattr(self._arena_handle, "arena", None)
        if native_arena is None:
            raise RuntimeError(
                f"[flashboot][seed] the {self._transport_name} transport needs the "
                f"native DeviceArena to export its shareable handle")
        return rendezvous.ArenaAdvertisement(
            arena_size=int(self._arena_handle.size), device=self._device,
            transport=self._transport_name,
            credential={
                "memory_handle": native_arena.export_shareable_handle(
                    self._transport_name).hex(),
                # True when this seed ALSO registered its arena for a one-sided READ.
                # A puller that cannot import the handle — another node, or a container
                # sharing neither the IPC namespace nor the GPUs — then pulls over rdma
                # instead of falling all the way back to reading the checkpoint.
                "rdma_offered": self._wire_publisher is not None,
            }).to_wire()

    def offers_exchange(self) -> bool:
        return self._wire_publisher is not None

    def exchange_with_puller(self, puller_publication: Publication) -> Publication:
        if self._wire_publisher is None:
            raise RuntimeError(
                "this seed published no wire offer (its arena is not registered for "
                "remote reads)")
        return self._wire_publisher.exchange_with_puller(puller_publication)

    def close(self) -> None:
        if self._wire_publisher is not None:
            self._wire_publisher.close()


class NvlinkTransport:
    """Arenas moved by exporting a shareable memory handle and copying over NVLink."""

    def __init__(self, name: str) -> None:
        if name not in NVLINK_TRANSPORTS:
            raise ValueError(f"[flashboot] {name!r} is not a handle transport "
                             f"(expected one of {NVLINK_TRANSPORTS})")
        self.name = str(name)

    def arena_requirements(self) -> ArenaRequirements:
        """A fabric handle can only be exported from CUDA-VMM memory; ipc exports plain
        cudaMalloc. This is the whole reason the arena ALLOCATION has to know which
        transport will serve it."""
        return ArenaRequirements(fabric_exportable=(self.name == "fabric"))

    def publisher_for(self, arena_handle, device: int) -> NvlinkArenaPublisher:
        return NvlinkArenaPublisher(self.name, arena_handle, device,
                                    wire_publisher=_optional_wire_publisher(
                                        arena_handle, device))

    def serve_arena(self, arena_handle, device: int, port: int, clone_count: int,
                    timeout_s: float) -> ArenaService:
        """One server for both shapes: which one a clone wants is in its opening
        message, so the count is the clones' business and never the seed's."""
        del clone_count   # the clones announce it in their registrations
        publisher = self.publisher_for(arena_handle, device)
        chain_seed = _NvlinkChainSeed(publisher)
        server = rendezvous.start_seed_server(port, device, publisher, timeout_s,
                                              chain_seed=chain_seed)
        return ArenaService(server=server, backing=publisher)

    def join_chain(self, arena_handle, device: int, clone_count: int,
                   requested_position: int) -> "NvlinkChainParticipant":
        position = int(requested_position)
        if not 1 <= position <= int(clone_count):
            raise ValueError(
                f"[flashboot][{self.name}] a broadcast position must be pinned by the "
                f"launcher: set --model-loader-extra-config '{{\"broadcast_rank\":i}}' "
                f"with 1 <= i <= broadcast_world={clone_count}, got {requested_position}"
                f" — positions used to be handed out by arrival order here, but a "
                f"member states its own rank now, the same way it does on rdma and in "
                f"the DP gather, so one rendezvous round serves all three")
        return NvlinkChainParticipant(self.name, arena_handle, device, clone_count,
                                      position)

    def join_all_gather(self, arena_handle, device: int, rank: int, world: int,
                        segment_alignment: int) -> "NvlinkGatherParticipant":
        return NvlinkGatherParticipant(self.name, arena_handle, device, rank, world,
                                       segment_alignment)

    def open_seed_arena(self, seed_host: str, seed_port: int, device: int,
                        timeout_s: float, connect_timeout_s=None):
        """Fetch the seed's advertisement and map its arena now. Falls through to the
        seed's wire offer when the handle cannot be imported here.

        ``FB_FABRIC_STRICT=1`` is the validation/deployment guard used by the GB300
        runner: the seed must advertise a fabric handle and that handle must import.
        It disables both ways this method can otherwise recover through RDMA, so a run
        cannot report success after silently leaving the NVLink fabric path.
        """
        from flashboot import _C

        connect_timeout = float(connect_timeout_s or timeout_s)
        advertisement, connection = rendezvous.fetch_arena_advertisement(
            seed_host, seed_port, timeout_s, connect_timeout_s=connect_timeout)
        connection.close()   # the advertisement was the whole handle handshake
        byte_count = int(advertisement.arena_size)
        strict_fabric = os.getenv("FB_FABRIC_STRICT", "0") == "1"
        if strict_fabric and advertisement.transport != "fabric":
            raise RuntimeError(
                "[flashboot][clone] FB_FABRIC_STRICT=1 but the seed advertised "
                f"transport={advertisement.transport!r}; start both seed and clone "
                "with FLASHBOOT_TRANSPORT=fabric in the same IMEX domain")
        if advertisement.transport == "rdma":
            # The seed decides the transport. It published no handle, so read its arena
            # over the wire instead — there is nothing here to import.
            return rdma_arena.RdmaSeedArena(
                seed_host, seed_port, device, byte_count=byte_count,
                timeout_s=timeout_s, connect_timeout_s=connect_timeout,
                reason="the seed publishes rdma, not a handle")
        importer = _C.PeerArenaImporter()
        import_started = time.perf_counter()
        try:
            importer.import_handle(advertisement.transport,
                                   bytes.fromhex(advertisement.credential["memory_handle"]),
                                   byte_count, int(device))
        except Exception as error:
            # This is where "the seed is not where I thought" actually shows up.
            if strict_fabric:
                raise RuntimeError(
                    "[flashboot][clone] fabric handle import failed while "
                    "FB_FABRIC_STRICT=1; refusing the RDMA wire fallback") from error
            if not advertisement.credential.get("rdma_offered"):
                raise
            reason = f"{type(error).__name__}: {error}"
            print(f"[flashboot][clone] {advertisement.transport} import failed "
                  f"({reason}) — the seed also offers a wire pull, switching to rdma",
                  flush=True)
            return rdma_arena.RdmaSeedArena(
                seed_host, seed_port, device, byte_count=byte_count,
                timeout_s=timeout_s, connect_timeout_s=connect_timeout,
                reason=f"{advertisement.transport} import failed: {reason}")
        print(f"[flashboot][clone] {advertisement.transport} import of "
              f"{byte_count / 1e9:.1f}GB seed arena in "
              f"{time.perf_counter() - import_started:.2f}s", flush=True)
        return ImportedSeedArena(advertisement.transport, importer, byte_count)


def _optional_wire_publisher(arena_handle, device: int):
    """A wire offer beside the handle, when this node can serve remote reads.

    Best-effort on purpose: it exists so a puller that cannot import the handle stays
    off the disk path, and a seed whose handle is perfectly serviceable must not fail
    because the optional half could not be built."""
    from flashboot.transport.selection import rdma_available

    # A strict GB300 deployment is measuring/serving the NVLink fabric path itself.
    # Do not spend time registering a second RDMA publication and, more importantly,
    # do not give a misplaced clone a mechanism through which it could appear healthy.
    if os.getenv("FB_FABRIC_STRICT", "0") == "1":
        return None
    if not rdma_available():
        return None
    try:
        return rdma_arena.RdmaArenaPublisher(arena_handle, device)
    except Exception as error:  # noqa: BLE001 — the offer is an addition, not a need
        print(f"[flashboot][seed] no wire offer this time "
              f"({type(error).__name__}: {error}) — serving the handle only, so a "
              f"clone that cannot import it will fall back to disk", flush=True)
        return None


class NvlinkGatherParticipant:
    """An all-gather member that maps each other owner's arena and copies its segment.

    Only the concurrent ("direct") shape exists here, deliberately. A ring exists for
    fabrics that cannot carry concurrent all-to-all flows, and the NVLink switch fabric
    can, at line rate — so a ring buys no bandwidth. It would also need a remote progress
    primitive to gate step s on a neighbour's step s-1, which handles do not have: the
    chain builds one from imported flag buffers and ``cuStreamWaitValue32``, and a ring
    would need all that machinery for no gain.

    Nothing has to be kept alive after the gather beyond the arena itself, which already
    backs the exported handle: a late importer stays safe because a gathered segment is
    never written again.
    """

    def __init__(self, transport_name: str, arena_handle, device: int, rank: int,
                 world: int, segment_alignment: int) -> None:
        from flashboot.transport import nvlink_all_gather_engine as handle_all_gather

        self._engine = handle_all_gather
        self._membership = handle_all_gather.prepare_allgather_membership(
            arena_handle, transport_name, device, rank=rank, world=world,
            segment_alignment=segment_alignment)

    @property
    def publication(self) -> Publication:
        return self._membership.info

    @property
    def membership(self):
        return self._membership

    def connect(self, members: list) -> None:
        # Nothing to pair: a handle is a standing credential, so the membership broadcast
        # already told us everything we need in order to read.
        del members

    def gather_from(self, members: list, timeout_s: float) -> None:
        self._engine.all_gather(self._membership, members, timeout_s)

    def close(self) -> None:
        pass


class NvlinkChainParticipant:
    """A clone's place in a handle-transport chain.

    Positions are assigned BY ARRIVAL, which a handle makes safe: it is a standing
    credential, so a clone can export its arena (and its per-chunk flags) before a single
    byte of either is valid. The seed collects all K registrations and releases every
    position at once; data-readiness ordering is per chunk, in hardware, so nothing has to
    be sequenced by the rendezvous.
    """

    def __init__(self, transport_name: str, arena_handle, device: int,
                 clone_count: int, position: int) -> None:
        from flashboot.transport import nvlink_chain_engine as nvlink_chain

        self._engine = nvlink_chain
        self._arena_handle = arena_handle
        self._device = int(device)
        self._clone_count = int(clone_count)
        self._position = int(position)
        self._world = int(clone_count) + 1   # the seed is member 0
        # Allocates and ZEROES the per-chunk flags before their handle can leave this
        # process: fresh device memory holds garbage, and any stale value would release a
        # chunk to our successor early.
        self._membership = nvlink_chain.prepare_membership(arena_handle, transport_name,
                                                           device)
        # What the rendezvous needs of every member, whatever its transport: where it
        # sits and how big its arena is. The credential below it stays this engine's.
        self._membership.registration.update(rank=self._position, world=self._world)

    @property
    def membership(self):
        """Our successor reads our arena AND our flags for the rest of its pull, so this
        has to outlive the load."""
        return self._membership

    def receive_broadcast(self, rendezvous_host: str, rendezvous_port: int,
                          timeout_s: float, connect_timeout_s=None) -> None:
        """Register for the round, then pull from whoever the membership says is ahead
        of us.

        The membership arrives whole, which is more than this engine needs — importing
        a memory handle pairs with nobody, so our predecessor's publication is the only
        one we read. We take it anyway because the round is the same round the rdma
        broadcast and the DP gather run, and the alternative was a release shape that
        existed only for this transport."""
        members, connection = rendezvous.collective_rendezvous(
            rendezvous_host, rendezvous_port, self._membership.registration,
            self._world, group="chain", timeout_s=timeout_s,
            connect_timeout_s=connect_timeout_s)
        upstream = next(m for m in members if int(m["rank"]) == self._position - 1)
        try:
            self._engine.pull_from_upstream(self._arena_handle, self._membership,
                                            upstream, self._device, self._position)
        except Exception:
            try:
                connection.close()  # no completion: the seed logs this link as failed
            except Exception:  # noqa: BLE001
                pass
            raise
        rendezvous.collective_complete(connection)

    def close(self) -> None:
        pass


class _NvlinkChainSeed:
    """The seed's half of an nvlink broadcast, in the shape the seed server asks for.

    Almost nothing to do, and that is the point: this family's credential is STANDING
    — one exported memory handle, importable by any number of clones, valid as long as
    the arena lives — so there is no link to raise before the membership goes out and
    nothing to re-mint between rounds. The hooks exist so that one server can run one
    round for both families; here two of the three are honestly empty."""

    def __init__(self, publisher) -> None:
        self._publisher = publisher

    def local_member(self, world: int) -> dict:
        """The seed as member rank 0. Its publication carries no pipeline flags: its
        bytes are valid before anybody arrives, which is what position 1 relies on."""
        publication = dict(self._publisher.standing_publication())
        publication.update(rank=0, world=int(world))
        return publication

    def connect(self, members) -> None:
        """Nothing to raise: an exported handle is readable the moment it exists."""

    def renew(self) -> None:
        """Nothing to re-mint: the same handle serves clone after clone."""


def _bind_current_device(device: int) -> None:
    """Make ``device`` current on whatever thread is asking. The rendezvous serves from a
    thread of its own and deliberately knows nothing about CUDA, so binding it is the
    transport's job."""
    import torch

    torch.cuda.set_device(int(device))
