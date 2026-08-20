"""flashboot: the single entry point sglang's get_model_loader calls for the
flashload / flashclone load-formats.

sglang's loader.py keeps ONE thin hook:

    if load_config.load_format in (LoadFormat.FLASHLOAD, LoadFormat.FLASHCLONE):
        from flashboot.sglang_plugin import get_flashboot_loader
        return get_flashboot_loader(load_config, model_config)

Role dispatch lives here (not in sglang). The role comes from
``--model-loader-extra-config '{"role": ...}'``:

    "sharded"       (default) -> ShardedStateArenaLoader  (seed: load from disk)
    "sharded_clone"           -> ShardedStateCloneLoader  (clone: NVLink-pull the seed)
    "sharded_dp"              -> ShardedStateDpLoader     (DP replicas: split the
                                 disk read, merge over an all-gather)

flashboot (and thereby torch) is imported lazily, only when a flash load format is
actually requested.
"""
from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


def _loader_class(role: str):
    from flashboot.sharded_loader import (
        ShardedStateArenaLoader,
        ShardedStateCloneLoader,
        ShardedStateDpLoader,
    )

    return {
        "sharded": ShardedStateArenaLoader,
        "sharded_clone": ShardedStateCloneLoader,
        "sharded_dp": ShardedStateDpLoader,
    }.get(role)


def get_flashboot_loader(load_config, model_config=None):
    extra_config = getattr(load_config, "model_loader_extra_config", None)
    role = str(extra_config["role"]).lower() \
        if isinstance(extra_config, dict) and extra_config.get("role") else "sharded"
    loader_class = _loader_class(role)
    if loader_class is None:
        raise ValueError(
            f"[flashboot] unknown role={role!r} "
            f"(expected sharded|sharded_clone|sharded_dp)")
    logger.info(
        f"[flashboot] load_format="
        f"{getattr(load_config.load_format, 'value', load_config.load_format)} "
        f"role={role} -> {loader_class.__name__}")
    return loader_class(load_config)
