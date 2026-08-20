"""Which port a shard serves on, and which host a clone dials.

Both answers key on the GLOBAL shard rank (``pipeline_rank * tensor_parallel_world +
tensor_parallel_rank``), never on the tensor-parallel rank alone: with pipeline
parallelism the same tensor-parallel rank recurs on every stage, so keying on it would
collapse every stage onto one port. For a single pipeline stage the global shard rank IS
the tensor-parallel rank, so TP-only checkpoints are unaffected.
"""
from __future__ import annotations

# The NEXTN/MTP draft worker would otherwise collide with the target model on
# base+shard_rank (same shard rank, same gpu). Far above any real shard rank, well
# within the 16-bit port range when the base is chosen sanely.
_DRAFT_MODEL_PORT_OFFSET = 1000


# Bump when the advertisement wire changes meaning. Checked on receipt (see
# ArenaAdvertisement.from_wire): seed and clone must run the SAME flashboot, which the
# sharded_state contract already requires — the arena layout mirrors the engine's
# runtime tensor layout, so a mismatched pair produces wrong weights, not an error.
# Failing here names the actual cause instead of letting it surface as garbage output.


def rank_server_port(base_port, shard_rank, model_config=None) -> int:
    """Per-rank TCP server port: ``base_port + shard_rank`` (+ the draft-model offset for a
    NEXTN/MTP draft worker, so target and draft on the same rank never share a port).

    ``shard_rank`` is the GLOBAL shard rank (``pp_rank * tp_world + tp_rank``), not the
    TP rank: with pipeline parallelism the same tp_rank recurs on every PP stage, so
    keying the port on tp_rank alone would collapse all PP stages onto one port. Keying on
    the global shard rank gives each (PP, TP) shard its own port. For pp_size==1 the global
    shard rank equals the tp_rank, so this is unchanged for TP-only checkpoints."""
    is_draft = bool(getattr(model_config, "is_draft_model", False)) if model_config is not None else False
    return int(base_port) + int(shard_rank) + (_DRAFT_MODEL_PORT_OFFSET if is_draft else 0)


def seed_host_for_rank(seed_ip, shard_rank, world_size) -> str:
    """Pick the seed node hosting global shard ``shard_rank`` from a comma-separated
    per-node IP list (node-rank order).

    The seed lays its ``world_size`` global shard ranks out contiguously across its nodes
    (node k owns ``[k*ranks_per_node, (k+1)*ranks_per_node)`` — see
    :func:`flashboot.utils.launch_arguments.node_shard_ranks`), so with ``n`` seed hosts and
    ``ranks_per_node = world_size // n`` the rank lives on ``ip[shard_rank // ranks_per_node]``.
    A single host serves every rank."""
    hosts = [h.strip() for h in str(seed_ip).split(",") if h.strip()]
    if not hosts:
        raise RuntimeError("[flashboot] seed_ip is empty — the clone needs the seed node IP(s)")
    if len(hosts) == 1:
        return hosts[0]
    ranks_per_host = max(1, int(world_size) // len(hosts))
    return hosts[min(int(shard_rank) // ranks_per_host, len(hosts) - 1)]
