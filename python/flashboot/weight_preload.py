"""Background GPFS -> tmpfs weight prestager for cold-start overlap (flashboot.weight_preload).

The cold read of a GLM sharded_state checkpoint from GPFS dominates a fresh pod's startup
(bounded by the shared filesystem's read ceiling). This module copies *only this
node's* shards to a tmpfs directory (``/dev/shm``) in a short-lived side process launched as the
side process started before the server (env gate ``FB_WEIGHT_PRELOAD=1``), so the read
overlaps the heavy import / NCCL rendezvous / model build that follow. The loader
(:mod:`flashboot.sharded_loader`, same gate) blocks per rank on a ``.rank{r}.ready`` marker
and then reads the tmpfs copy at RAM speed.

Design constraints:
  * High GPFS concurrency: all of the node's ranks are staged *concurrently* (many files in
    flight), because GPFS only reaches its aggregate bandwidth under multi-file concurrency —
    copying one file at a time leaves it far below the ceiling and defeats the overlap.
  * NUMA-local: shards for the first half of a node's GPUs bind their threads to socket 0 and
    the second half to socket 1 (H100 topology), so first-touch places the tmpfs pages local to
    the GPU that later H2D-copies them.
  * Bounded threads: ``FB_WEIGHT_PRELOAD_THREADS`` (64 by default) is a node-wide budget split
    evenly across the concurrent ranks — the measured GPFS read sweet spot, not per-rank.
  * Self-terminating: the process exits as soon as its shards are staged, releasing all CPU
    before the compute-heavy capture / warmup phases; it never lingers.

Only stdlib is imported at module load so the launcher can spawn this before touching sglang.
"""
from __future__ import annotations

import glob
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Optional, Sequence

from flashboot.utils.launch_arguments import argument_int, argument_value, node_shard_ranks

ENABLE_ENV = "FB_WEIGHT_PRELOAD"
DIR_ENV = "FB_WEIGHT_PRELOAD_DIR"
THREADS_ENV = "FB_WEIGHT_PRELOAD_THREADS"
CHUNK_MB_ENV = "FB_WEIGHT_PRELOAD_CHUNK_MB"
TIMEOUT_ENV = "FB_WEIGHT_PRELOAD_TIMEOUT"
KEEP_ENV = "FB_WEIGHT_PRELOAD_KEEP"                 # =1 -> keep tmpfs shards (skip auto-free; for HOLD reuse)
CLEANUP_TIMEOUT_ENV = "FB_WEIGHT_PRELOAD_CLEANUP_TIMEOUT"  # max wait for loader's .consumed
_DEFAULT_DIR = "/dev/shm/fb_weights"


def is_enabled() -> bool:
    """True when the caller opted into background weight preloading (``FB_WEIGHT_PRELOAD=1``)."""
    return os.environ.get(ENABLE_ENV) == "1"


def preload_dir() -> str:
    """tmpfs directory the shards are staged into and that the loader reads back from."""
    return os.environ.get(DIR_ENV, _DEFAULT_DIR)


def rank_ready_marker(staged_dir: str, shard_rank: int) -> str:
    """Marker the stager touches once every part of ``shard_rank`` has been staged (loader waits)."""
    return os.path.join(staged_dir, f".rank{shard_rank}.ready")


def failed_marker(staged_dir: str) -> str:
    """Marker the stager writes (with the traceback) when staging dies — waiters check it
    to fail fast instead of sitting out their whole timeout on a marker that will never
    appear."""
    return os.path.join(staged_dir, ".preload_failed")


def consumed_marker(staged_dir: str, shard_rank: int) -> str:
    """Marker the loader touches once it has finished reading ``shard_rank`` from tmpfs (the
    stager waits on it, then frees that rank's staged files)."""
    return os.path.join(staged_dir, f".rank{shard_rank}.consumed")


def mark_consumed(staged_dir: str, shard_rank: int) -> None:
    """Loader-side: signal that this rank's shards have been fully read from tmpfs and may be
    freed. Best-effort (a failure just means the stager frees on timeout instead)."""
    try:
        open(consumed_marker(staged_dir, shard_rank), "w").close()
    except OSError:
        pass


def _stamp() -> str:
    """Wall-clock 'HH:MM:SS' for aligning stager progress with the loader's log."""
    return time.strftime("%H:%M:%S")


# ── staging ─────────────────────────────────────────────────────────────────────────────────
_ext_cache: object = None  # lazily-imported flashboot._C (or False if unavailable)


def _ext():
    """Lazy handle to the compiled extension. Imported only on the copy path (never from
    ``spawn_background``), so the launcher's first-line spawn stays stdlib-only and fast.

    ``import torch`` first: _C.so links libtorch (libc10 & co.) without an rpath, so a
    bare ``python -m flashboot.weight_preload`` process must load torch's shared libs
    before _C can resolve. Costs ~10s in the stager side process only."""
    global _ext_cache
    if _ext_cache is None:
        try:
            import torch  # noqa: F401  (loads libc10/libtorch for the _C import below)
            from flashboot import _C as mod
            _ext_cache = mod if hasattr(mod, "stage_file_to_shm") else False
        except Exception:
            _ext_cache = False
    return _ext_cache or None


def _stage_file(src: str, staged_dir: str, threads: int, chunk: int, numa_node: int) -> int:
    """Copy one shard GPFS->tmpfs via the C++ ``stage_file_to_shm`` (pread straight into the
    mmap'd tmpfs — one copy, NUMA-local threads): write ``<name>.partial`` then atomically rename
    so a reader never observes a partial file. Skips if already staged."""
    dst = os.path.join(staged_dir, os.path.basename(src))
    if os.path.exists(dst + ".ready"):
        return os.path.getsize(dst)
    ext = _ext()
    if ext is None:
        raise RuntimeError("flashboot._C.stage_file_to_shm unavailable — cannot preload weights")
    tmp = dst + ".partial"
    ext.stage_file_to_shm(src, tmp, threads, chunk, numa_node)
    os.rename(tmp, dst)
    return os.path.getsize(src)


def _stage_rank(shard_rank: int, numa_node: int, src_dir: str, staged_dir: str,
                threads: int, chunk: int) -> None:
    """Stage one rank's shards NUMA-locally, then touch its ready marker. Run concurrently with
    the node's other ranks so many files are in flight at once — GPFS delivers its aggregate
    bandwidth only under that concurrency (one-file-at-a-time leaves it far below the ceiling)."""
    pat = os.path.join(src_dir, f"model-rank-{shard_rank}-part-*.safetensors")
    part_paths = sorted(glob.glob(pat))
    t0 = time.perf_counter()
    nbytes = sum(_stage_file(p, staged_dir, threads, chunk, numa_node) for p in part_paths)
    open(rank_ready_marker(staged_dir, shard_rank), "w").close()  # signal loader
    dt = time.perf_counter() - t0
    print(f"[weight-preload][{_stamp()}][numa{numa_node}] rank{shard_rank}: {len(part_paths)} parts "
          f"{nbytes / 1e9:.1f}GB in {dt:.1f}s ({nbytes / 1e9 / max(dt, 1e-6):.1f}GB/s)",
          flush=True)


def _free_rank(staged_dir: str, shard_rank: int) -> int:
    """Delete one rank's staged shards + its markers from tmpfs. Returns bytes freed."""
    freed = 0
    for p in glob.glob(os.path.join(staged_dir, f"model-rank-{shard_rank}-part-*.safetensors")):
        try:
            freed += os.path.getsize(p)
            os.remove(p)
        except OSError:
            pass
    for m in (rank_ready_marker(staged_dir, shard_rank), consumed_marker(staged_dir, shard_rank)):
        try:
            os.remove(m)
        except OSError:
            pass
    return freed


def _free_when_consumed(staged_dir: str, ranks: Sequence[int], timeout: float) -> int:
    """For each staged rank, wait for the loader's ``.consumed`` marker then free that rank's
    tmpfs shards (reclaims RAM incrementally as ranks finish). Stops waiting after ``timeout``
    (leaves the not-yet-consumed shards rather than deleting data still being read)."""
    t0 = time.perf_counter()
    total = 0
    for shard_rank in ranks:
        marker = consumed_marker(staged_dir, shard_rank)
        while not os.path.exists(marker):
            if time.perf_counter() - t0 > timeout:
                print(f"[weight-preload][{_stamp()}] cleanup timeout waiting rank{shard_rank} "
                      f"consumed; leaving its shards", flush=True)
                return total
            time.sleep(0.5)
        freed = _free_rank(staged_dir, shard_rank)
        total += freed
        print(f"[weight-preload][{_stamp()}] rank{shard_rank} consumed -> freed "
              f"{freed / 1e9:.1f}GB", flush=True)
    return total


LOG_ENV = "FB_WEIGHT_PRELOAD_LOG"


def _log_path() -> str:
    """Where the stager's stdout goes. Defaults to the (pod-local) tmpfs dir; set
    ``FB_WEIGHT_PRELOAD_LOG`` to a shared path to inspect timing off-node."""
    node_rank = os.environ.get("PET_NODE_RANK", "0")
    override = os.environ.get(LOG_ENV)
    if override:
        return override.replace("{rank}", node_rank)
    return os.path.join(preload_dir(), f"preload_node{node_rank}.log")


def spawn_background(argv: Sequence[str]) -> Optional[subprocess.Popen]:
    """Launch the stager as a detached side process (non-blocking), before the server so
    the copy overlaps the launcher's own heavy work. Returns the Popen, or None when
    disabled; never raises into the launch path."""
    if not is_enabled():
        return None
    os.makedirs(preload_dir(), exist_ok=True)
    log_path = _log_path()
    os.makedirs(os.path.dirname(log_path) or ".", exist_ok=True)
    log = open(log_path, "a")
    return subprocess.Popen(
        [sys.executable, "-m", "flashboot.weight_preload", *argv],
        stdout=log, stderr=subprocess.STDOUT, start_new_session=True,
    )


def main(argv: Sequence[str]) -> int:
    """Stage this node's shards, then exit. Reads the layout from the same launch args the
    server gets (``--model-path`` / ``--tp-size`` / ``--pp-size`` / ``--nnodes`` /
    ``--node-rank``), falling back to the ``PET_*`` env for node identity."""
    if not is_enabled():
        return 0
    src = argument_value(argv, "--model-path", "--model")
    if not src:
        print("[weight-preload] no --model-path in argv; nothing to stage", flush=True)
        return 0
    tp_size = argument_int(argv, "--tp-size", "--tp", default=1)
    pp_size = argument_int(argv, "--pp-size", "--pp", default=1)
    nnodes = argument_int(argv, "--nnodes", default=int(os.environ.get("PET_NNODES", "1")))
    node_rank = argument_int(argv, "--node-rank", default=int(os.environ.get("PET_NODE_RANK", "0")))
    staged = preload_dir()
    os.makedirs(staged, exist_ok=True)
    total_threads = int(os.environ.get(THREADS_ENV, "64"))  # node-wide budget, split across ranks
    chunk = int(os.environ.get(CHUNK_MB_ENV, "64")) << 20

    ranks = node_shard_ranks(nnodes, pp_size, tp_size, node_rank)
    n = len(ranks)
    per_rank_threads = max(total_threads // max(n, 1), 1)
    # Local GPU index = position in the sorted node rank list; first half -> socket 0, rest -> 1.
    half = max(n // 2, 1)
    print(f"[weight-preload][{_stamp()}] START node{node_rank}: staging shards {ranks} src={src} "
          f"-> {staged} ({n} ranks concurrent x {per_rank_threads} threads, {chunk >> 20}MB chunks)",
          flush=True)

    # Stage every rank concurrently so many files are in flight (matches the loader's own
    # multi-rank GPFS concurrency); each rank binds to its GPU's socket for NUMA-local pages.
    t0 = time.perf_counter()
    try:
        with ThreadPoolExecutor(max_workers=n) as pool:
            futures = [pool.submit(_stage_rank, r, 0 if i < half else 1, src, staged,
                                   per_rank_threads, chunk)
                       for i, r in enumerate(ranks)]
            for fut in futures:
                fut.result()
    except BaseException:
        # Leave the reason where the waiting loaders can see it, so they fail fast
        # instead of sitting out their whole timeout on markers that will never appear.
        import traceback
        try:
            with open(failed_marker(staged), "w") as f:
                f.write(traceback.format_exc())
        except OSError:
            pass
        raise
    print(f"[weight-preload][{_stamp()}] DONE node{node_rank}: {n} ranks staged in "
          f"{time.perf_counter() - t0:.1f}s (staging CPU released)", flush=True)

    # Auto-free: wait for the loader to signal each rank consumed, then delete that rank's tmpfs
    # shards so the ~48GB/rank of RAM is reclaimed (tmpfs never frees on its own). The staging
    # threads are already gone, so this phase is a light poll (no CPU). FB_WEIGHT_PRELOAD_KEEP=1
    # skips it (keep the /dev/shm copy for HOLD-gang reuse).
    if os.environ.get(KEEP_ENV) == "1":
        print(f"[weight-preload][{_stamp()}] {KEEP_ENV}=1 -> keeping tmpfs shards; exiting", flush=True)
        return 0
    freed = _free_when_consumed(staged, ranks, float(os.environ.get(CLEANUP_TIMEOUT_ENV, "1800")))
    try:
        os.rmdir(staged)  # best-effort; only succeeds if now empty
    except OSError:
        pass
    print(f"[weight-preload][{_stamp()}] node{node_rank}: freed {freed / 1e9:.1f}GB from "
          f"{staged} after loader consumed; exiting", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
