"""The roles: who reads the checkpoint, and who takes an arena from somebody who did.

    seed            reads its own shard from storage and offers the arena to others
    clone           reads no checkpoint at all — it takes a seed's arena
    data_parallel   K replicas each read 1/K and gather the rest from each other

All three build the SAME arena and rebind the model onto it the same way, which is in
arena_loader_base; they differ only in where the bytes come from. Two supporting modules:
process_topology (where this process sits, and which shard is therefore its own) and
coordination (how this instance is placed relative to the others, and where it meets
them).

This is the top layer. It uses the collectives, the transports and the rendezvous; it
is the only layer sglang talks to, through :mod:`flashboot.sglang_plugin`.

Nothing here names a transport. Which mechanism moves an arena, what it publishes, how a
chain hop learns its upstream is ready — all of that is behind the contracts in
:mod:`flashboot.transport.contracts`, so adding a transport touches one new file in that
package and nothing in this one.
"""
from __future__ import annotations

from flashboot.loaders.clone import ShardedStateCloneLoader
from flashboot.loaders.coordination import CoordinationSettings
from flashboot.loaders.data_parallel import ShardedStateDpLoader
from flashboot.loaders.process_topology import ProcessTopology, global_shard_rank
from flashboot.loaders.seed import ShardedStateArenaLoader
from flashboot.loaders.split_fill import run_split_fill

__all__ = [
    "CoordinationSettings",
    "ProcessTopology",
    "ShardedStateArenaLoader",
    "ShardedStateCloneLoader",
    "ShardedStateDpLoader",
    "global_shard_rank",
    "run_split_fill",
]
