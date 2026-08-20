"""How the bytes of one arena reach another — the transport layer.

Everything about "which mechanism moves the arena" lives here and nowhere above:

    selection       resolves FLASHBOOT_TRANSPORT to a concrete transport
    rdma_engine     which one-sided-READ engine this process uses, and its endpoint
    nixl_endpoint   the NIXL agent, wearing the native endpoint's method surface
    infiniband      GPU -> HCA mapping and queue-pair card pairing

Two families sit behind that vocabulary, with the same purpose and different shapes:

  * the NVLINK family — ``ipc`` (same node) and ``fabric`` (an NVLink fabric / IMEX
    domain, which crosses nodes): the owner exports a 64-byte shareable memory handle,
    the puller imports it and one copy-engine device-to-device copy moves the arena.
  * the RDMA family — ``rdma``: no handle exists. The owner registers its arena for
    one-sided READs and publishes an rkey; the puller reads it over InfiniBand or RoCE
    straight into its own arena (GPUDirect).

This package must not import the collectives or the loaders above it. Choosing an rdma
engine used to be asked UPWARDS of the collectives, which is why ``rdma_engine`` exists
as its own module here.
"""
from __future__ import annotations

from flashboot.transport.contracts import (
    ArenaPublisher,
    ArenaRequirements,
    Publication,
    SeedArena,
    Transport,
)
from flashboot.transport.selection import (
    SUPPORTED_TRANSPORTS,
    fabric_available,
    peer_import_available,
    rdma_available,
    select_transport,
)


def create_transport(device: int = 0) -> Transport:
    """The transport this process will use on ``device``, as an object.

    :func:`select_transport` resolves the NAME (which is what a preflight report and a
    log line want); this builds the thing that can actually move an arena. Callers above
    this package should use it and then never mention ipc, fabric or rdma again.
    """
    return transport_named(select_transport(device))


def transport_named(name: str) -> Transport:
    """The transport called ``name`` — no probing, no environment. Used when the choice
    has already been made somewhere else: the seed decides, and a clone follows the name
    it finds in the seed's advertisement."""
    from flashboot.transport.nvlink_arena import NVLINK_TRANSPORTS, NvlinkTransport
    from flashboot.transport.rdma_arena import RdmaTransport

    if name in NVLINK_TRANSPORTS:
        return NvlinkTransport(name)
    if name == "rdma":
        return RdmaTransport()
    raise ValueError(
        f"[flashboot] no transport called {name!r} (expected one of "
        f"{SUPPORTED_TRANSPORTS})")


__all__ = [
    "ArenaPublisher",
    "ArenaRequirements",
    "Publication",
    "SUPPORTED_TRANSPORTS",
    "SeedArena",
    "Transport",
    "create_transport",
    "fabric_available",
    "peer_import_available",
    "rdma_available",
    "select_transport",
    "transport_named",
]
