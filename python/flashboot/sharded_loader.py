"""Compatibility shell — the loaders live in :mod:`flashboot.loaders` now.

This module path is part of flashboot's EXTERNAL interface: sglang is a different
repository and hard-codes two imports,

    from flashboot.sglang_plugin import get_flashboot_loader     # the loader entry point
    from flashboot.sharded_loader import run_split_fill          # the scheduler's finalize

so moving the implementation is only safe while this path keeps re-exporting it. Every
deployment would otherwise have to upgrade both repositories in lockstep.

New code should import from :mod:`flashboot.loaders` directly.
"""
from __future__ import annotations

from flashboot.loaders import (  # noqa: F401 — re-exported for the pinned import paths
    CoordinationSettings,
    ProcessTopology,
    ShardedStateArenaLoader,
    ShardedStateCloneLoader,
    ShardedStateDpLoader,
    global_shard_rank,
    run_split_fill,
)

__all__ = [
    "CoordinationSettings",
    "ProcessTopology",
    "ShardedStateArenaLoader",
    "ShardedStateCloneLoader",
    "ShardedStateDpLoader",
    "global_shard_rank",
    "run_split_fill",
]
