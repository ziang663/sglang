"""How this instance is placed relative to the others, and where it meets them.

Every field mirrors one key of the launch interface — ``--model-loader-extra-config``
JSON, with an environment variable as the fallback. The field names say what the value
MEANS; the keys stay exactly what a deployment already writes, because changing those
would break every launch script in existence. The two vocabularies meet in one place,
:meth:`CoordinationSettings.from_load_config`, and nowhere else.
"""
from __future__ import annotations

import dataclasses
import os

from sglang.srt.configs.load_config import LoadConfig


@dataclasses.dataclass
class CoordinationSettings:
    """Where this instance meets the others, and what its place among them is."""

    # Comma-separated per-node addresses of the seed, in node-rank order (key
    # "seed_ip"). The instance holding global shard g picks the node that hosts g.
    seed_host_list: str
    # Base TCP port (key "seed_port"); each shard rank gets base + its own rank, so
    # every shard is a peer-to-peer pair with no rank-0 funnel.
    seed_base_port: int
    # How long a transfer may take before it is called stalled.
    transfer_timeout_seconds: float
    # Bounds only the CONNECT phase (FB_SEED_CONNECT_TIMEOUT_S): an absent or dead seed
    # then fails this instance in seconds — into the stock-loader fallback — instead of
    # spinning for the whole transfer budget. Raise it when clones are launched before
    # the seed is expected to be up, since a seed only serves after its own fill.
    connect_timeout_seconds: float = 30.0
    # How many clone INSTANCES start together (key "broadcast_world"). 1 is the plain
    # single-clone pull; K > 1 is a chain broadcast, where every hop streams
    # concurrently. It is needed on the CLONES always, and on an rdma seed too, because
    # its collective round cannot start collecting without knowing how many to expect.
    clone_instance_count: int = 1
    # This clone's fixed place in the chain, 1..clone_instance_count (key
    # "broadcast_rank"). Needed only where the transport's credentials are minted per
    # link before the registration that carries them, which is why it cannot be assigned by
    # arrival there; 0 means "unset, let the seed decide", which the handle transports
    # accept. Unused for a single clone.
    chain_position: int = 0
    # The DP-cooperative disk load (role "sharded_dp"): how many replica INSTANCES split
    # the read (key "dp_world"), and which one this is (key "dp_rank"). Index 0 is
    # meaningful, so -1 marks it unset — unlike the chain knobs, where 0 does.
    replica_count: int = 0
    replica_index: int = -1

    @classmethod
    def from_load_config(cls, load_config: LoadConfig | None) -> "CoordinationSettings":
        """Read the environment, then let the extra-config JSON overlay it.

        The one subtlety is ``dp_rank``: 0 is a valid replica index, so it is tested for
        presence rather than truth. Every other key is a count or an address where 0 and
        "" mean "not given".
        """
        settings = cls(
            seed_host_list=os.getenv("SGLANG_FABRIC_SEED_IP", "").strip(),
            seed_base_port=int(os.getenv("SGLANG_FABRIC_SEED_PORT", "0") or "0"),
            transfer_timeout_seconds=float(
                os.getenv("SGLANG_FABRIC_BROADCAST_TIMEOUT_S", "1800")),
            connect_timeout_seconds=float(os.getenv("FB_SEED_CONNECT_TIMEOUT_S", "30")),
            replica_count=int(os.getenv("FB_DP_WORLD", "0") or "0"),
            replica_index=int(os.getenv("FB_DP_RANK", "-1") or "-1"),
        )
        extra_config = getattr(load_config, "model_loader_extra_config", None) \
            if load_config is not None else None
        if not (isinstance(extra_config, dict) and extra_config):
            return settings
        if extra_config.get("seed_ip"):
            settings.seed_host_list = str(extra_config["seed_ip"]).strip()
        if extra_config.get("seed_port"):
            settings.seed_base_port = int(extra_config["seed_port"])
        if extra_config.get("connect_timeout_s"):
            settings.connect_timeout_seconds = float(extra_config["connect_timeout_s"])
        if extra_config.get("broadcast_world"):
            settings.clone_instance_count = int(extra_config["broadcast_world"])
        if extra_config.get("broadcast_rank"):
            settings.chain_position = int(extra_config["broadcast_rank"])
        if extra_config.get("dp_world"):
            settings.replica_count = int(extra_config["dp_world"])
        if extra_config.get("dp_rank") is not None:   # 0 is a valid replica index
            settings.replica_index = int(extra_config["dp_rank"])
        return settings
