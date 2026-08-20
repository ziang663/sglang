"""Which rdma engine this process uses, and the endpoint it binds.

Choosing between the two one-sided-READ engines — a NIXL agent and the in-package
ibverbs one — is a TRANSPORT decision, so it lives here rather than in the collectives
that call it. Both engines present the same method surface (see
:mod:`flashboot.transport.nixl_endpoint`), which is why nothing above this module needs
to know which one it got.
"""
from __future__ import annotations

import os
import time

from flashboot.transport import infiniband

_RESOLVED_BACKEND: "str | None" = None


def rdma_backend() -> str:
    """``FB_RDMA_BACKEND``: ``nixl`` (a NIXL agent) or ``raw`` (the in-package ibverbs
    engine). Unset resolves to **nixl when its python bindings import**, else raw.

    nixl is the default for two measured reasons.

    CROSS-node, it picks its own rail. The raw backend's GPU->HCA map is positional over
    NAME-sorted HCAs, and HCA names do not follow topology — on a node whose rails do
    not line up with that ordering it hands some ranks a cross-socket rail, and a clone
    is only done when its slowest rank is. nixl leaves the rail choice to UCX, which
    tends to even the ranks out.

    SAME node the difference is larger and in the same direction: UCX serves an
    intra-node pair without touching the NIC at all, while raw has no intra-node path
    and loops every READ through the HCA. So ``FB_RDMA_BACKEND=raw`` is worth
    considering only when every clone is on another node — and worth measuring on your
    own topology before committing to it either way.

    An explicit value is honoured strictly and errors if that backend is unusable, so
    an A/B stays an A/B. Resolved once per process — both ends of a collective must
    agree, and :func:`flashboot.rdma_collectives.connect_topology` rejects a mixed
    membership by name.
    """
    global _RESOLVED_BACKEND
    requested = os.getenv("FB_RDMA_BACKEND", "").strip().lower()
    if requested:
        if requested not in ("raw", "nixl"):
            raise ValueError(f"FB_RDMA_BACKEND={requested!r} is not supported: "
                             f"expected 'raw' or 'nixl'.")
        return requested
    if _RESOLVED_BACKEND is None:
        from flashboot.transport.nixl_endpoint import nixl_available

        if nixl_available():
            _RESOLVED_BACKEND = "nixl"
        else:
            # Say so. Falling back here is not a neutral choice: it puts this rank back
            # on the positional GPU->HCA map, which is exactly where some ranks can
            # draw a cross-socket rail and finish materially slower — and the only
            # symptom is the clone taking longer.
            _RESOLVED_BACKEND = "raw"
            print("[flashboot][rdma] WARNING nixl is the default backend but its "
                  "python bindings do not import (pip install nixl) — falling back to "
                  "the raw ibverbs engine, whose GPU->HCA map is positional: check the "
                  "per-rank binding line below if the pull comes out lopsided. Set "
                  "FB_RDMA_BACKEND=raw to make this choice explicit and silence this.",
                  flush=True)
    return _RESOLVED_BACKEND


def _nic_inventory(nics) -> str:
    """``mlx5_15@400/numa0 mlx5_33@400/numa1`` — the host's rails with rate and NUMA
    affinity, for the binding log line.

    Both qualifiers earned their place from measurements. Rate: same count on both
    ends but different rates still means the ranks do not finish together. NUMA: the
    positional map sorts HCAs by NAME, and on a measured node the four HCAs sort to
    NUMA 0, 1, 0, 1 — so this line is what tells you whether a rank got a rail across
    the socket from its GPU, which is the leading explanation for two ranks reaching
    line rate while two sharing an equally fast rail reached a third of it."""
    return " ".join(
        f"{n}@{infiniband.port_rate_gbps(n):.0f}/numa{infiniband.port_numa_node(n)}"
        for n in nics) or "none"


# What create_endpoint bound in this process, keyed by GPU. Rides the wire so the far
# end of a link can spot a divergent GPU->HCA map (see the collectives' divergence
# check).
_BOUND_NICS: dict = {}


def bound_nic_info() -> dict:
    """The HCA this process's endpoint bound, plus how many active HCAs its host sees.
    ``{}`` when that is ambiguous — no endpoint yet, or several on different GPUs; one
    process per rank is the norm, so in practice this is exact."""
    if len(_BOUND_NICS) != 1:
        return {}
    return dict(next(iter(_BOUND_NICS.values())))


def create_endpoint(gpu: int):
    """The rdma endpoint for ``gpu``. ``FB_RDMA_BACKEND=raw`` binds the native ibverbs
    engine to the GPU's positionally-mapped HCA; ``nixl`` (the default where its
    bindings import) returns a NIXL agent wearing the same method surface (see
    :mod:`flashboot.transport.nixl_endpoint`), so every collective above is unchanged.

    Either way the binding is RECORDED and printed: it is derived from what this host
    happens to see, it decides this rank's bandwidth, and it was invisible during
    bring-up (deriving it needed a separate script poking at
    ``infiniband_device_for_gpu``)."""
    nics = infiniband.active_infiniband_devices()
    if rdma_backend() == "nixl":
        from flashboot.transport.nixl_endpoint import NixlEndpoint, nixl_available

        if not nixl_available():
            raise RuntimeError(
                "[flashboot][rdma] FB_RDMA_BACKEND=nixl but the nixl python bindings "
                "are not importable (pip install nixl)")
        # UCX picks the rail: no manual binding to diverge (this backend's main
        # operational argument), so only the count is worth publishing.
        _BOUND_NICS[int(gpu)] = {"nic": "", "nic_count": len(nics), "nic_rate": 0.0}
        print(f"[flashboot][rdma] gpu{gpu} -> nixl/UCX chooses among {len(nics)} "
              f"active HCA(s): {_nic_inventory(nics)}", flush=True)
        _t0 = time.perf_counter()
        _ep = NixlEndpoint(int(gpu))
        print(f"[flashboot][rdma] gpu{gpu} nixl agent created in "
              f"{time.perf_counter() - _t0:.2f}s", flush=True)
        return _ep
    from flashboot import _C

    if not hasattr(_C, "Endpoint"):
        raise RuntimeError(
            "[flashboot][rdma] the flashboot._C extension was built without the "
            "rdma engine — rebuild with FB_BUILD_RDMA=1 (needs libibverbs-dev)")
    nic = infiniband.infiniband_device_for_gpu(gpu)
    rate = infiniband.port_rate_gbps(nic) if nic else 0.0
    _BOUND_NICS[int(gpu)] = {"nic": nic, "nic_count": len(nics), "nic_rate": rate}
    _gpu_node = infiniband.gpu_numa_node(gpu)
    _nic_node = infiniband.port_numa_node(nic) if nic else "?"
    _how = os.environ.get("FB_NIC_MAP", "numa").strip().lower()
    # Print BOTH sides of the affinity, not just the pick. A cross-socket binding is
    # not an error and produces no warning; it just makes this rank several times
    # slower than its siblings, and without these two numbers side by side that reads
    # as fabric noise rather than as a mapping bug.
    print(f"[flashboot][rdma] gpu{gpu}(numa{_gpu_node}) -> "
          f"{nic or '(engine default)'}@{rate:.0f}Gb/s(numa{_nic_node}) "
          f"[map={_how}{'' if _gpu_node == _nic_node or _nic_node == '?' else ' CROSS-SOCKET'}] "
          f"over {len(nics)} active HCA(s): {_nic_inventory(nics)}", flush=True)
    _t0 = time.perf_counter()
    _ep = _C.Endpoint(int(gpu), nic)
    print(f"[flashboot][rdma] gpu{gpu} raw ibverbs endpoint created in "
          f"{time.perf_counter() - _t0:.2f}s", flush=True)
    return _ep
