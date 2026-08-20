"""RDMA plumbing shared by the collectives: GPU->NIC mapping + card pairing.

The heavy lifting (RC queue pairs, GPUDirect memory registration, one-sided
RDMA-READ) lives in the native ``flashboot._C.Endpoint`` (csrc/rdma_read.cpp). This
module provides the small pieces around it:

  * the GPU -> InfiniBand HCA mapping (``infiniband_device_for_gpu``): a FIXED
    positional map ``gpu // (num_gpus // num_nics)`` over the sorted active HCAs, so
    the node's GPUs spread evenly across all rails. Deterministic, no PCIe/sysfs
    depth probing — a nearest-NIC probe mis-handles GPUs that are all-SYS to every
    HCA and overloads some rails.
  * ``connect_endpoint``: drive one of the endpoint's RC queue pairs to RTS from a
    peer's QP business card (the dict ``Endpoint.create_qp`` returns, carried inside
    the collective-rendezvous membership — see :mod:`flashboot.transport.rdma_data_plane`).
"""
from __future__ import annotations

import logging
import os

logger = logging.getLogger(__name__)


# ── GPU -> InfiniBand HCA mapping ────────────────────────────────────────────────────
def _read_sysfs(path: str) -> str:
    try:
        with open(path) as f:
            return f.read().strip()
    except OSError:
        return ""


def infiniband_devices() -> list:
    """InfiniBand/RoCE devices visible to this host (sysfs), sorted by name."""
    try:
        return sorted(os.listdir("/sys/class/infiniband"))
    except OSError:
        return []


def _has_active_infiniband_port(device: str) -> bool:
    ports_dir = f"/sys/class/infiniband/{device}/ports"
    try:
        ports = os.listdir(ports_dir)
    except OSError:
        return False
    for port in ports:
        if "ACTIVE" not in _read_sysfs(f"{ports_dir}/{port}/state"):
            continue
        # Both link layers carry one-sided RDMA: "InfiniBand" and RoCE, which
        # reports "Ethernet". Probing for "InfiniBand" alone hides every HCA on a
        # RoCE fleet and makes the rdma transport look absent.
        link_layer = _read_sysfs(f"{ports_dir}/{port}/link_layer")
        if "InfiniBand" in link_layer or "Ethernet" in link_layer:
            return True
    return False


_ACTIVE_DEVICES_CACHE: "list[str] | None" = None


def active_infiniband_devices() -> list:
    """Active InfiniBand HCAs, sorted by name (mlx5_0, mlx5_1, ...) — the canonical NIC
    ordering the GPU->NIC map indexes into. Cached: the NIC set is fixed for the process
    lifetime, so the sysfs scan runs once even when called per GPU."""
    global _ACTIVE_DEVICES_CACHE
    if _ACTIVE_DEVICES_CACHE is None:
        _ACTIVE_DEVICES_CACHE = [d for d in infiniband_devices()
                                 if _has_active_infiniband_port(d)]
    return _ACTIVE_DEVICES_CACHE


def port_rate_gbps(device: str) -> float:
    """Nominal rate of ``device``'s first active port, in Gb/s (0.0 if unreadable).

    sysfs reports e.g. ``"400 Gb/sec (4X NDR)"``. This is published alongside the
    binding because the rails on one host are NOT necessarily equivalent: a measured
    run had a seed's two HCAs deliver very different aggregate rates, which looks
    exactly like "half the ranks are slow" until you can see which rail each rank got.
    """
    ports_dir = f"/sys/class/infiniband/{device}/ports"
    try:
        ports = sorted(os.listdir(ports_dir))
    except OSError:
        return 0.0
    for port in ports:
        if "ACTIVE" not in _read_sysfs(f"{ports_dir}/{port}/state"):
            continue
        rate = _read_sysfs(f"{ports_dir}/{port}/rate").split()
        if rate:
            try:
                return float(rate[0])
            except ValueError:
                return 0.0
    return 0.0


def port_numa_node(device: str) -> str:
    """NUMA node of ``device``'s PCI function as a string ("0", "1", ...), or "?" when
    sysfs does not say.

    Logged next to the binding because the positional map sorts HCAs BY NAME, and name
    order has nothing to do with topology: a measured node's four HCAs sort to NUMA
    0, 1, 0, 1. So consecutive GPUs can land on a cross-socket rail while a local one
    sits idle, which is invisible unless the affinity is printed."""
    node = _read_sysfs(f"/sys/class/infiniband/{device}/device/numa_node")
    return node if node and node != "-1" else "?"


def _cuda_device_count() -> int:
    """Number of CUDA devices on this node via cudart (ctypes — no torch import), for the
    GPU->NIC fan-out step. 0 if cudart cannot be loaded."""
    import ctypes

    cudart = None
    for library in ("libcudart.so", "libcudart.so.13", "libcudart.so.12"):
        try:
            cudart = ctypes.CDLL(library)
            break
        except OSError:
            continue
    if cudart is None:
        return 0
    count = ctypes.c_int(0)
    if cudart.cudaGetDeviceCount(ctypes.byref(count)) != 0:
        return 0
    return int(count.value)


def _gpu_pci_bdf(gpu: int) -> str:
    """PCI address of CUDA device ``gpu`` as sysfs spells it, or "" if cudart cannot say.

    Uses cudaDeviceGetPCIBusId, so it follows CUDA_VISIBLE_DEVICES the same way every
    other index in this process does."""
    import ctypes

    for library in ("libcudart.so", "libcudart.so.13", "libcudart.so.12"):
        try:
            cudart = ctypes.CDLL(library)
        except OSError:
            continue
        buf = ctypes.create_string_buffer(64)
        if cudart.cudaDeviceGetPCIBusId(buf, 64, ctypes.c_int(int(gpu))) == 0:
            return buf.value.decode().lower()
        return ""
    return ""


def gpu_numa_node(gpu: int) -> str:
    """NUMA node of CUDA device ``gpu``, or "?" when it cannot be determined."""
    bdf = _gpu_pci_bdf(gpu)
    if not bdf:
        return "?"
    node = _read_sysfs(f"/sys/bus/pci/devices/{bdf}/numa_node")
    return node if node not in ("", "-1") else "?"


def infiniband_device_for_gpu(gpu: int) -> str:
    """The InfiniBand HCA a GPU binds for RDMA, by the fixed positional map
    ``nic_index = gpu // (num_gpus // num_nics)`` over :func:`active_infiniband_devices`.

    On the common 8-GPU/4-NIC H100 node this is ``gpu // 2`` (GPU0,1->mlx5_0, ...,
    GPU6,7->mlx5_3) — a balanced 2-GPU/NIC fan-out that is NUMA-local on the
    contiguous-NUMA layout. On an 8-GPU/8-NIC node it is 1:1. Returns '' when no NIC is
    found (the native Endpoint then picks its own default).
    """
    try:
        nics = active_infiniband_devices()
        if not nics:
            return ""

        def positional() -> str:
            num_gpus = _cuda_device_count() or len(nics)
            gpus_per_nic = max(1, num_gpus // len(nics))
            return nics[(int(gpu) // gpus_per_nic) % len(nics)]

        if os.environ.get("FB_NIC_MAP", "numa").strip().lower() == "positional":
            return positional()

        # NUMA-affine first. The positional map divides by the number of VISIBLE GPUs,
        # which is the wrong denominator inside a container that was given a SUBSET of
        # the node's GPUs but still sees every HCA: a 4-GPU container on an 8-GPU /
        # 4-HCA node computes 4//4 = 1 and hands out a 1:1 map, so GPUs whose sockets
        # hold only two of the HCAs get bound to the other socket's. The symptom is a
        # clone whose ranks split into a fast group and a slow group, which is easy to
        # read as fabric noise.
        my_node = gpu_numa_node(gpu)
        if my_node != "?":
            local = [n for n in nics if port_numa_node(n) == my_node]
            if local:
                # Spread the GPUs that share this socket across its HCAs, keeping the
                # ordering stable so both ends of a transfer agree.
                same = [g for g in range(_cuda_device_count() or 1)
                        if gpu_numa_node(g) == my_node]
                slot = same.index(int(gpu)) if int(gpu) in same else 0
                return local[slot % len(local)]
        return positional()
    except Exception:  # noqa: BLE001 — NIC discovery must never crash a load
        return ""


def connect_endpoint(endpoint, peer_queue_pair: dict, qp: int = 0) -> None:
    """Drive queue pair ``qp`` of the native Endpoint to RTS using the peer's business
    card. ``qp`` is the LOCAL index ``Endpoint.create_qp`` handed out for this
    connection (the ``"qp"`` field of the peer's card is the peer's own index and is
    deliberately ignored here)."""
    if hasattr(endpoint, "connect_card"):
        # nixl backend: pairing is add_remote_agent, not a QP transition — the whole
        # card (agent metadata + served ranges) goes through instead of the RC fields.
        endpoint.connect_card(peer_queue_pair, qp=int(qp))
        return
    endpoint.connect(int(peer_queue_pair["qpn"]), int(peer_queue_pair["psn"]),
                     int(peer_queue_pair["lid"]), str(peer_queue_pair["gid"]),
                     int(peer_queue_pair["mtu"]), qp=int(qp))
