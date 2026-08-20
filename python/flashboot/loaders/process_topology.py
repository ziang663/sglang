"""Where this process sits, and which shard of the checkpoint is therefore its own."""
from __future__ import annotations

import os

import torch
import torch.distributed

from sglang.srt.distributed import (
    get_pp_group,
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
)


class ProcessTopology:
    """This process's placement (rank/world from torch.distributed or env, current GPU)."""

    def __init__(self) -> None:
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            self.world = torch.distributed.get_world_size()
            self.rank = torch.distributed.get_rank()
        else:
            self.world = int(os.environ.get("WORLD_SIZE", "1"))
            self.rank = int(os.environ.get("RANK", "0"))
        self.gpu = torch.cuda.current_device()

    def __repr__(self) -> str:
        return f"ProcessTopology(rank={self.rank}/{self.world} gpu={self.gpu})"

def global_shard_rank() -> int:
    """Global shard id in the ``model-rank-{rank}-part-*`` file name: the rank across the
    (PP, TP) grid = ``pp_rank * tp_world + tp_rank``. With pipeline parallelism each PP
    stage holds different layers but reuses the same tp_rank range, so selecting files by
    tp_rank alone would read the wrong stage's layers. For pp_size==1 this reduces to
    tp_rank. Must match the rank scheme in sglang ShardedStateLoader.save_model (and
    :func:`flashboot.utils.launch_arguments.node_shard_ranks`)."""
    tp_rank = get_tensor_model_parallel_rank()
    pp_group = get_pp_group()
    if pp_group.world_size > 1:
        return pp_group.rank_in_group * get_tensor_model_parallel_world_size() + tp_rank
    return tp_rank
