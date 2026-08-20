"""The deferred fill: bind the model to a placeholder arena now, read the weights last.

With ``FB_SPLIT_FILL=1`` a seed binds its parameters onto a placeholder arena during the
load and does the real read plus device copy at the very END of server startup, so the
storage-to-tmpfs staging overlaps the whole startup instead of blocking the load. The
server calls :func:`run_split_fill` once, from the scheduler process that owns the model.

The pending list is module-level because that call site is a different one: sglang's
scheduler imports ``run_split_fill`` directly and has no reference to the loader that
armed it.
"""
from __future__ import annotations


# Fills armed by the loader under FB_SPLIT_FILL=1 — one per model loaded in this process
# (the target model and, with speculative decoding, the draft) — executed by
# run_split_fill(), which the server calls once at the very end of startup.
PENDING_SPLIT_FILLS: list = []


def run_split_fill() -> bool:
    """Standalone finalize step for FB_SPLIT_FILL=1: the server calls this ONCE at the
    very end of startup (right before the scheduler reports ready), from the scheduler
    process that owns the model. Each pending fill checks whether the background
    /dev/shm preload (:mod:`flashboot.weight_preload`) has finished — if so the real
    read + H2D into the placeholder arena starts immediately; if not it waits for the
    marker first (that wait is exactly the staging time NOT hidden behind startup).
    Returns True when at least one pending fill ran, False when nothing was armed."""
    if not PENDING_SPLIT_FILLS:
        return False
    while PENDING_SPLIT_FILLS:
        PENDING_SPLIT_FILLS.pop(0)()
    return True
