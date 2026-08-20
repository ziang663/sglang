"""Transport resolution for the flashboot weight pull.

Two handle transports with the same shape (the seed exports a 64-byte shareable
memory handle, the clone imports it and pulls the arena over NVLink with the copy
engine — ``flashboot._C.PeerArenaImporter``):

  * ``fabric`` — NVLink fabric / IMEX domain (GB300 NVL72): a CUDA VMM
    ``CUmemFabricHandle``, importable by any GPU in the same IMEX domain — crosses
    nodes inside the domain.
  * ``ipc``    — same node only: a ``cudaIpcGetMemHandle`` blob.

plus the wire transport for fabrics with neither:

  * ``rdma``   — raw-ibverbs one-sided RDMA READ over InfiniBand/RoCE
    (``flashboot._C.Endpoint``, GPUDirect): crosses nodes with no IMEX domain (H100
    IB clusters). The chain broadcast runs it through
    :mod:`flashboot.transport.rdma_data_plane` (progress counters instead of handle imports).

``FLASHBOOT_TRANSPORT`` picks the transport; ``auto`` (the default) resolves to
``fabric`` when the IMEX preflight passes (``flashboot._C.check_imex`` — driver >=
12.4, device fabric-handle support, an accessible nvidia-caps-imex-channels channel),
else to ``rdma`` when the engine is built and an active InfiniBand HCA exists, and
otherwise to ``ipc``. The seed's choice wins: the clone follows the transport
recorded in the seed's arena advertisement.

WHY rdma BEFORE ipc: the seed publishes its transport before any clone connects, so
nothing here knows whether the clones will share its node. ipc only works if they do;
rdma works either way, so ordering it first keeps ``auto`` working for the case this
transport exists for. Ordering ipc first would instead make a cross-node clone fail its
handle import — and, before the seed learned to publish an rdma offer beside the handle,
degrade all the way to reading the checkpoint from disk.

What that ordering costs, on ONE node with the same arena and the same 1-to-1 path:
preferring rdma over the handle is a small loss with the default nixl backend, because
UCX serves an intra-node pair without touching the NIC. With ``FB_RDMA_BACKEND=raw`` it
is not small — that backend has no intra-node path and loops every READ through the
HCA.

The ambiguity itself is handled where it belongs now: a seed on a handle transport ALSO
publishes an rdma offer, so a clone that cannot import the handle pulls over the wire
instead of from disk (see :meth:`flashboot.sharded_loader.ShardedArenaLoaderBase.
_prepare_rdma_offer`). The remaining gap is the mirror case — a same-node clone whose
seed chose rdma AND whose backend is raw — where a handle published beside the rdma
offer would recover the difference. Not implemented; a deployment that knows its
placement should still say so with FLASHBOOT_TRANSPORT.
"""
from __future__ import annotations

import logging
import os

logger = logging.getLogger(__name__)

SUPPORTED_TRANSPORTS = ("ipc", "fabric", "rdma")


def peer_import_available() -> bool:
    """True when the native handle-import engine is built
    (``flashboot._C.PeerArenaImporter`` — the ipc/fabric transports).

    Deliberately does NOT probe placement or IMEX: whether the clone actually shares a
    node (ipc) or an IMEX domain (fabric) with the seed is only known at import time,
    where the native engine raises with a transport-specific hint."""
    try:
        from flashboot import _C

        return hasattr(_C, "PeerArenaImporter")
    except Exception:  # noqa: BLE001 — probing must never crash the caller
        return False


def fabric_available(device: int = 0) -> bool:
    """True when the native handle engines are built AND this system passes the
    NVLink-fabric / IMEX preflight (``flashboot._C.check_imex``: CUDA driver >= 12.4,
    the device reports fabric-handle support, and an nvidia-caps-imex-channels channel
    is accessible — i.e. the nvidia-imex daemon manages this node's IMEX domain).

    This is the GB300-verified gate: a fabric handle exported on a passing system can
    be imported by any GPU in the same IMEX domain. Non-IMEX nodes fail the checks
    and fall through to rdma/ipc."""
    try:
        from flashboot import _C

        if not (hasattr(_C, "PeerArenaImporter") and hasattr(_C, "check_imex")):
            return False
        status = _C.check_imex(device)
        if not status["ok"]:
            logger.debug(f"[flashboot] fabric preflight failed: {status['detail']}")
        return bool(status["ok"])
    except Exception:  # noqa: BLE001 — probing must never crash the caller
        return False


def rdma_available() -> bool:
    """True when the selected rdma backend can run here AND this node has an active
    InfiniBand HCA. ``nixl`` (the default where its bindings import) needs only those
    bindings; ``raw`` needs the in-package ibverbs engine (``flashboot._C.Endpoint``,
    built with FB_BUILD_RDMA). Since an unset ``FB_RDMA_BACKEND`` already resolves to
    whichever is usable, this answers "can rdma run here at all"."""
    try:
        from flashboot.transport import infiniband
        from flashboot.transport.rdma_engine import rdma_backend

        if not infiniband.active_infiniband_devices():
            return False
        if rdma_backend() == "nixl":
            from flashboot.transport.nixl_endpoint import nixl_available

            return nixl_available()
        from flashboot import _C

        return hasattr(_C, "Endpoint")
    except Exception:  # noqa: BLE001 — probing must never crash the caller
        return False


def select_transport(device: int = 0) -> str:
    """Resolve ``FLASHBOOT_TRANSPORT`` to a concrete transport.

    ``auto`` (the default) prefers ``fabric`` when the IMEX preflight passes (GB300
    NVL72 and other MNNVL/IMEX-configured systems), then ``rdma`` when the engine and
    an active HCA exist (H100 IB clusters — crosses nodes), and falls back to ``ipc``
    (same-node rehearsals) — so the same launch command picks the working transport
    on either platform.

    Raises with a clear message when the requested transport is unknown or its native
    engine is unavailable, instead of failing later inside the load.
    """
    requested = os.getenv("FLASHBOOT_TRANSPORT", "auto").strip().lower()
    if requested in ("", "auto"):
        if fabric_available(device):
            transport = "fabric"
        elif rdma_available():
            transport = "rdma"
            # This used to print a warning built on a bad inference — that a same-node
            # clone would pay the CROSS-node rdma bandwidth. Measured on one node, same
            # arena, same direct path: ipc is fastest, rdma+nixl close behind (UCX picks
            # its intra-node transport, the NIC is not in the path), rdma+raw far
            # behind both (no intra-node path at all — every READ loops through the
            # HCA). So the pick is fine with the default backend and poor only with
            # raw, which is where the note now lives
            # (transport.rdma_engine.rdma_backend and nixl_endpoint).
        else:
            transport = "ipc"
        logger.info(f"[flashboot] FLASHBOOT_TRANSPORT=auto resolved to {transport!r}")
    else:
        transport = requested
    if transport not in SUPPORTED_TRANSPORTS:
        raise ValueError(
            f"FLASHBOOT_TRANSPORT={requested!r} is not supported: expected one of "
            f"{SUPPORTED_TRANSPORTS} or 'auto' (see flashboot.transport.selection)."
        )
    if transport == "rdma":
        if not rdma_available():
            raise RuntimeError(
                "flashboot rdma transport unavailable: the flashboot._C extension "
                "lacks the Endpoint engine (rebuild with FB_BUILD_RDMA=1) or this "
                "node has no active InfiniBand HCA."
            )
    elif not peer_import_available():
        raise RuntimeError(
            f"flashboot {transport} transport unavailable: the flashboot._C extension "
            f"was built without the PeerArenaImporter engine — rebuild it "
            f"(python setup.py build_ext --inplace)."
        )
    return transport
