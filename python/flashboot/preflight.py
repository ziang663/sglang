"""Node preflight for the flashboot clone transports — run BEFORE a multinode launch.

    python -m flashboot.preflight [--device N]

Prints, per probe, what this node can do and why:

  * fabric — the NVLink-fabric / IMEX preflight (``flashboot._C.check_imex``, ported
    from the GB300-verified implementation): CUDA driver >= 12.4, the device reports
    CU_MEM_HANDLE_TYPE_FABRIC support, the nvidia-caps-imex-channels device class
    exists, and at least one channel is R/W accessible. All four pass on a correctly
    configured GB300 NVL72 node (nvidia-imex daemon running); plain H100 nodes fail
    the IMEX checks by design.
  * ipc    — the handle-import engine is built (same-node clones only).

Exits 0 when the transport ``auto`` resolution succeeds, 1 otherwise — so launch
scripts can gate on it.
"""
from __future__ import annotations

import argparse
import sys


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description="flashboot node preflight (fabric/IMEX + ipc probes)")
    parser.add_argument("--device", type=int, default=0,
                        help="CUDA device ordinal to probe (default 0)")
    arguments = parser.parse_args(argv)

    # The _C extension links libtorch — load torch first so its symbols resolve.
    try:
        import torch  # noqa: F401
    except Exception as error:  # noqa: BLE001 — report, do not trace
        print(f"[preflight] FAIL: torch import failed: {error}")
        return 1
    try:
        from flashboot import _C
    except Exception as error:  # noqa: BLE001
        print(f"[preflight] FAIL: flashboot._C import failed: {error}\n"
              f"  -> build it first: python setup.py build_ext --inplace")
        return 1

    from flashboot import transport

    print(f"[preflight] flashboot._C loaded "
          f"(PeerArenaImporter={'yes' if hasattr(_C, 'PeerArenaImporter') else 'NO'}, "
          f"ChainReceiver={'yes' if hasattr(_C, 'ChainReceiver') else 'NO'}, "
          f"check_imex={'yes' if hasattr(_C, 'check_imex') else 'NO'})")

    if hasattr(_C, "check_imex"):
        status = _C.check_imex(arguments.device)
        print(f"[preflight] fabric/IMEX: {'PASS' if status['ok'] else 'fail'} — "
              f"{status['detail']}")
    else:
        print("[preflight] fabric/IMEX: fail — _C built without check_imex (rebuild)")

    ipc_ok = transport.peer_import_available()
    print(f"[preflight] ipc: {'PASS' if ipc_ok else 'fail'} — "
          f"{'handle-import engine present (same-node clones)' if ipc_ok else 'PeerArenaImporter engine missing'}")

    try:
        resolved = transport.select_transport(arguments.device)
    except Exception as error:  # noqa: BLE001
        print(f"[preflight] FLASHBOOT_TRANSPORT resolution FAILED: {error}")
        return 1
    print(f"[preflight] transport resolution -> {resolved!r}")
    if resolved == "ipc":
        print("[preflight] note: ipc serves same-node clones only")
    return 0


if __name__ == "__main__":
    sys.exit(main())
