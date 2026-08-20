"""stdlib-only launch-argv parsing + rank topology math.

:mod:`flashboot.weight_preload` (the background shard prestager) must read a few sglang
server arguments BEFORE any heavy ``import sglang`` happens, so this module imports only
the stdlib and re-implements just the argv forms sglang accepts (``--flag value`` and
``--flag=value``).

The rank math here mirrors sglang's ``_calculate_rank_ranges`` and the
``model-rank-{rank}-part-*`` file naming of ShardedStateLoader.save_model; keeping ONE
copy prevents the launcher and the prestager from drifting apart on which shards belong
to a node.
"""
from __future__ import annotations

from typing import List, Optional, Sequence, Tuple


def argument_value(argv: Sequence[str], *names: str, default: Optional[str] = None) -> Optional[str]:
    """Value of the first of ``names`` found in ``argv``. Accepts BOTH sglang argv forms:
    ``--flag value`` and ``--flag=value``."""
    for name in names:
        for i, token in enumerate(argv):
            if token == name and i + 1 < len(argv):
                return argv[i + 1]
            if token.startswith(name + "="):
                return token.split("=", 1)[1]
    return default


def argument_int(argv: Sequence[str], *names: str, default: Optional[int] = None) -> Optional[int]:
    value = argument_value(argv, *names, default=None)
    try:
        return int(value) if value is not None else default
    except (TypeError, ValueError):
        return default


def node_rank_ranges(nnodes: int, pp_size: int, tp_size: int,
                     node_rank: int) -> Tuple[range, range]:
    """(pp_rank_range, tp_rank_range) hosted by ``node_rank``.

    Mirrors sglang ``entrypoints.engine._calculate_rank_ranges``: a PP stage either owns
    whole nodes (pp_size <= nnodes) or shares a node with other stages (pp_size > nnodes),
    and each node hosts a contiguous slice of the stage's TP ranks.
    """
    pp_size_per_node = max(pp_size // nnodes, 1)
    nnodes_per_pp_rank = max(nnodes // pp_size, 1)
    pp_rank_range = range(
        pp_size_per_node * (node_rank // nnodes_per_pp_rank),
        pp_size_per_node * (node_rank // nnodes_per_pp_rank + 1),
    )
    tp_size_per_node = tp_size // nnodes_per_pp_rank
    tp_rank_range = range(
        tp_size_per_node * (node_rank % nnodes_per_pp_rank),
        tp_size_per_node * (node_rank % nnodes_per_pp_rank + 1),
    )
    return pp_rank_range, tp_rank_range


def node_shard_ranks(nnodes: int, pp_size: int, tp_size: int, node_rank: int) -> List[int]:
    """Global shard ids (``pp_rank * tp_size + tp_rank``) whose GPUs live on this node —
    the ranks in the ``model-rank-{rank}-part-*`` file names this node's loaders will
    read. Must match :func:`flashboot.sharded_loader.global_shard_rank`."""
    pp_rank_range, tp_rank_range = node_rank_ranges(nnodes, pp_size, tp_size, node_rank)
    return sorted(pp_rank * tp_size + tp_rank
                  for pp_rank in pp_rank_range for tp_rank in tp_rank_range)
