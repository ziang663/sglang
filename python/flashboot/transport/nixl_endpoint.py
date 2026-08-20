"""NIXL-agent RDMA backend — a drop-in for the native ``flashboot._C.Endpoint``.

``FB_RDMA_BACKEND=nixl`` (the DEFAULT wherever the bindings import — see
:func:`flashboot.transport.rdma_engine.rdma_backend` for the measurements behind that choice,
the sharpest being a same-node pull, which UCX serves without the NIC while the raw
engine, because UCX serves an intra-node pair without going near the NIC) swaps the
raw-ibverbs engine for a NIXL agent while every
collective above it (chain broadcast, all-gather, 1-to-1 clone) keeps running
unchanged: this class exposes exactly the six methods
:mod:`flashboot.transport.rdma_data_plane` calls on an endpoint.

    reg(addr, size) -> (lkey, rkey)
    create_qp() -> card dict
    connect(...) / connect_card(card, qp=)   <- the pairing step
    post_read(local, lkey, remote, rkey, length, wr_id=, qp=)
    read(local, lkey, remote, rkey, length, timeout_ms, qp=)
    wait_completions(count, timeout_ms)

Two shape differences from ibverbs, both absorbed here:

* **No queue pairs.** NIXL manages connections inside its own backend (UCX), so
  ``create_qp`` mints a card that carries this agent's metadata instead of a QP
  business card, and pairing is ``add_remote_agent``. The ``qp=`` argument the
  collectives pass through is kept in the signature and ignored — one agent serves
  every peer.
* **No rkeys.** A NIXL read is addressed by ``(remote agent, addr, len, dev_id)``,
  so ``reg`` returns placeholder keys; the addresses already travel in the
  rendezvous membership, which is why the collectives need no change. What ``reg``
  does do is register the range with the agent (host or device memory, detected
  from the pointer) — a read only lands in memory the agent knows.

The peer name for a read is resolved from the card the collectives paired us with;
with a single remote agent (chain hop, 1-to-1 clone) that is unambiguous, and for
the all-gather every peer is added by name at pairing time and selected per read
via the remote address ranges it advertised.
"""
from __future__ import annotations

import base64
import logging
import os
import time

logger = logging.getLogger(__name__)

_DEFAULT_MEM_TYPE = "VRAM"


def nixl_available() -> bool:
    """True when the NIXL python bindings import (no agent is created here)."""
    try:
        from nixl._api import nixl_agent  # noqa: F401

        return True
    except Exception:  # noqa: BLE001 — probing must never crash the caller
        return False


def _memory_type(addr: int) -> str:
    """"VRAM" for device pointers, "DRAM" for host — NIXL registers them differently.

    The collectives register two kinds of memory through the same ``reg``: the CUDA
    arena and the host-memory progress counters (:class:`flashboot.transport.rdma_data_plane.HostCounter`).
    """
    try:
        from cuda.bindings import driver as cu

        err, kind = cu.cuPointerGetAttribute(
            cu.CUpointer_attribute.CU_POINTER_ATTRIBUTE_MEMORY_TYPE, addr)
        if err == cu.CUresult.CUDA_SUCCESS and int(kind) == int(
                cu.CUmemorytype.CU_MEMORYTYPE_DEVICE):
            return "VRAM"
    except Exception:  # noqa: BLE001 — a non-CUDA pointer simply is not device memory
        pass
    return "DRAM"


class NixlEndpoint:
    """One NIXL agent presented with the native Endpoint's method surface."""

    def __init__(self, gpu: int, agent_name: str | None = None) -> None:
        from nixl._api import nixl_agent, nixl_agent_config

        self.gpu = int(gpu)
        # The name gains a -rN suffix on every renew_agent(), so a peer never confuses
        # this round's agent with the previous round's dead one.
        self._base_name = agent_name or f"flashboot-{os.getpid()}-gpu{self.gpu}"
        self._generation = 0
        self._name = self._base_name
        config = nixl_agent_config(backends=["UCX"])
        self._agent = nixl_agent(self._name, config)
        self._registrations: list = []          # keepalive: dereg would revoke reads
        self._peers: list[str] = []             # remote agent names, pairing order
        self._peer_ranges: dict[str, list] = {} # peer name -> [(addr, size), ...]
        self._inflight: list = []               # (handle, wr_id)
        self._local_ranges: list = []           # (addr, size, mem_type)
        logger.info("[flashboot][nixl] agent %s ready on gpu %d", self._name, self.gpu)

    # ---------------- registration ----------------
    def reg(self, addr: int, size: int):
        """Register ``[addr, addr+size)`` with the agent. Returns placeholder
        ``(lkey, rkey)``: NIXL addresses reads by agent + address, so the collectives'
        key fields simply travel unused."""
        mem_type = _memory_type(int(addr))
        descs = self._agent.get_reg_descs(
            [(int(addr), int(size), self.gpu, "")], mem_type)
        self._registrations.append(self._agent.register_memory(descs))
        self._local_ranges.append((int(addr), int(size), mem_type))
        return (0, 0)

    # ---------------- pairing ----------------
    def create_qp(self) -> dict:
        """The NIXL equivalent of a QP business card: this agent's metadata. Carries
        the ibverbs field names with zero values so the existing card plumbing (and
        any logging of it) stays happy."""
        return {
            "qp": 0, "qpn": 0, "psn": 0, "lid": 0, "gid": "", "mtu": 0,
            "backend": "nixl",
            "agent": self._name,
            "meta": base64.b64encode(self._agent.get_agent_metadata()).decode(),
        }

    def connect_card(self, peer_card: dict, qp: int = 0) -> None:
        """Pair with the peer described by ``peer_card`` (its agent metadata)."""
        del qp  # one agent serves every peer
        meta = peer_card.get("meta")
        if not meta:
            raise RuntimeError(
                "[flashboot][nixl] peer card carries no agent metadata — the peer is "
                "probably running FB_RDMA_BACKEND=raw; both ends must use the same "
                "rdma backend")
        name = self._agent.add_remote_agent(base64.b64decode(meta))
        name = name.decode() if isinstance(name, bytes) else str(name)
        if name not in self._peers:
            self._peers.append(name)
        ranges = peer_card.get("ranges") or []
        if ranges:
            self._peer_ranges[name] = [(int(a), int(n)) for a, n in ranges]
        logger.info("[flashboot][nixl] paired with remote agent %s", name)

    def release_peers(self) -> int:
        """Best-effort ``remove_remote_agent`` for every peer paired so far; returns how
        many were dropped."""
        for name in self._peers:
            try:
                self._agent.remove_remote_agent(name)
            except Exception as e:  # noqa: BLE001 — a dead peer may refuse; that is the point
                logger.info("[flashboot][nixl] remove_remote_agent(%s): %s", name, e)
        released, self._peers, self._peer_ranges = len(self._peers), [], {}
        return released

    def renew_agent(self) -> None:
        """Rebuild this endpoint's agent for a NEW round — the NIXL counterpart of
        ``Endpoint.reset_qp``, called at a round boundary by
        :meth:`flashboot.transport.rdma_data_plane.RdmaMembership.renew_cards`.

        A seed that serves clone after clone needs this, and dropping the remote agents
        is NOT enough. A finished clone is normally SIGKILLed, so it never disconnects
        politely; the seed's UCX then cannot bring up a queue pair for the next peer —
        measured on the seed, one per TP rank:

            ib_mlx5_dv.c:433 UCX ERROR mlx5dv_devx_obj_modify(opcode=0x503) failed,
                                       syndrome 0x5d668c: Remote I/O error

        which reaches the next clone as NIXL_ERR_REMOTE_DISCONNECT from its first
        completion poll (clone #1 pulled fine; clone #2 connected, posted, then died).
        That state lives in the UCX worker, below anything remove_remote_agent touches,
        so the whole agent is replaced: new agent, new worker, and the SAME memory
        ranges re-registered — the arena itself never moves, so a clone still pulls
        from the same addresses. Dropping the last reference to the old agent is what
        clears the poisoned state, and its registrations go with it (safe: the round
        completed, so every peer already reported its pull done).

        Costs one re-registration of the arena per round (~0.2s for a 15GB arena), and
        the caller must re-issue the business card afterwards — :meth:`create_qp` then
        publishes the NEW worker's metadata, which is the whole point.
        """
        from nixl._api import nixl_agent, nixl_agent_config

        released = self.release_peers()
        ranges, self._registrations = list(self._local_ranges), []
        self._local_ranges, self._inflight = [], []
        previous = self._name
        self._generation += 1
        self._name = f"{self._base_name}-r{self._generation}"
        # Destroy the old agent BEFORE building the new one, not as a side effect of
        # rebinding self._agent: the poisoned worker is what we are getting rid of, and
        # two live agents in one process may also contend for the same listener port.
        # Dropping the last reference here runs the C++ destructor now.
        old_agent, self._agent = self._agent, None
        del old_agent
        self._agent = nixl_agent(self._name, nixl_agent_config(backends=["UCX"]))
        for addr, size, _mem_type in ranges:
            self.reg(int(addr), int(size))
        print(f"[flashboot][nixl] round boundary on gpu{self.gpu}: agent {previous} -> "
              f"{self._name} ({released} peer(s) released, {len(ranges)} range(s) "
              f"re-registered)", flush=True)

    def connect(self, qpn, psn, lid, gid, mtu, qp: int = 0) -> None:  # noqa: D401
        """ibverbs-shaped signature, kept so nothing above needs a branch. The NIXL
        path pairs through :meth:`connect_card`; reaching here means a raw card was
        handed to a NIXL endpoint."""
        del qpn, psn, lid, gid, mtu, qp
        raise RuntimeError(
            "[flashboot][nixl] got a raw-ibverbs QP card: both ends must run the "
            "same FB_RDMA_BACKEND")

    # ---------------- data plane ----------------
    def _peer_for(self, remote_addr: int, length: int) -> str:
        if not self._peers:
            raise RuntimeError(
                "[flashboot][nixl] no remote agent paired yet — connect_topology must "
                "run before any read")
        if len(self._peers) == 1:
            return self._peers[0]
        for name, ranges in self._peer_ranges.items():
            for base, size in ranges:
                if base <= remote_addr and remote_addr + length <= base + size:
                    return name
        raise RuntimeError(
            f"[flashboot][nixl] no paired agent advertises "
            f"[0x{remote_addr:x}, +{length}) — peers: {sorted(self._peers)}")

    def post_read(self, local_addr: int, local_lkey: int, remote_addr: int,
                  remote_rkey: int, length: int, wr_id: int = 0, qp: int = 0) -> None:
        """Start one READ; completion is collected by :meth:`wait_completions`."""
        del local_lkey, remote_rkey, qp
        peer = self._peer_for(int(remote_addr), int(length))
        local_mem = _memory_type(int(local_addr))
        local = self._agent.get_xfer_descs(
            [(int(local_addr), int(length), self.gpu)], local_mem)
        remote = self._agent.get_xfer_descs(
            [(int(remote_addr), int(length), self.gpu)], local_mem)
        handle = self._agent.initialize_xfer("READ", local, remote, peer)
        state = self._agent.transfer(handle)
        if state == "ERR":
            raise RuntimeError(
                f"[flashboot][nixl] READ of {length}B from {peer} failed to start")
        self._inflight.append((handle, int(wr_id)))

    def wait_completions(self, count: int, timeout_ms: int) -> None:
        """Block until ``count`` posted READs are done (or the timeout expires)."""
        deadline = time.perf_counter() + max(1, int(timeout_ms)) / 1000.0
        pending = self._inflight[-int(count):] if count else []
        remaining = list(pending)
        while remaining:
            still = []
            for handle, wr_id in remaining:
                state = self._agent.check_xfer_state(handle)
                if state == "ERR":
                    raise RuntimeError(
                        f"[flashboot][nixl] READ wr_id={wr_id} failed in flight")
                if state != "DONE":
                    still.append((handle, wr_id))
            remaining = still
            if remaining and time.perf_counter() > deadline:
                raise TimeoutError(
                    f"[flashboot][nixl] {len(remaining)} of {count} READs did not "
                    f"complete within {timeout_ms} ms")
        for handle, _wr in pending:
            try:
                self._agent.release_xfer_handle(handle)
            except Exception:  # noqa: BLE001 — a released handle must not fail a pull
                logger.debug("[flashboot][nixl] release_xfer_handle failed",
                             exc_info=True)
        if count:
            del self._inflight[-int(count):]

    def read(self, local_addr: int, local_lkey: int, remote_addr: int,
             remote_rkey: int, length: int, timeout_ms: int, qp: int = 0) -> None:
        """Blocking READ — the progress-counter poll of the chain."""
        self.post_read(local_addr, local_lkey, remote_addr, remote_rkey, length,
                       wr_id=0, qp=qp)
        self.wait_completions(1, timeout_ms)

    # ---------------- introspection used when building the card ----------------
    def local_ranges(self) -> list:
        """(addr, size) of everything registered — rides the card so a peer with
        several remotes can route a read to the right agent."""
        return [(addr, size) for addr, size, _mem in self._local_ranges]
