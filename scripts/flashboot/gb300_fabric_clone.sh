#!/usr/bin/env bash
# Strict GB300 fabric deployment entry point.
#
# It reuses this branch's numbered scripts and only supplies the GB300 preset,
# topology normalization, and an IMEX check that fails closed on every local GPU.
#
# The default TP8 instance spans two four-GPU nodes. Run seed and clone on their
# respective node ranks with FB_NODE_RANK=0|1 and FB_MASTER equal to THAT instance's
# node-0 IP. SEED_IPS and CLONE_IPS are the two instances' comma-separated
# node-rank orders. Requiring both on the clone is the placement guard: the two
# lists must contain exactly 2*FB_NNODES distinct node addresses.
#
# K-clone chain: every clone instance sets BROADCAST_WORLD=K and a unique
# BROADCAST_RANK=1..K. All members must use the same FB_CHAIN_CHUNK_MB.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
ROLE=${1:-${FB_ROLE:-}}

log() { echo "[gb300-fabric][${ROLE:-unset} node ${FB_NODE_RANK:-?}] $*"; }
die() { echo "[gb300-fabric][${ROLE:-unset}] FAILED: $*" >&2; exit 1; }

case "$ROLE" in
  seed|clone|preflight|verify) ;;
  *) die "role must be seed | clone | preflight | verify" ;;
esac

# Correctness verification is an HTTP client operation. It must not require this
# shell to be on a GPU node, source the serving preset, or provide either instance's
# distributed topology.
if [ "$ROLE" = verify ]; then
  : "${SEED_URL:?set SEED_URL=http://<seed-node0>:30206}"
  : "${CLONE_URL:?set CLONE_URL=http://<clone-node0>:${CLONE_HTTP_PORT:-30306}}"
  exec bash "$ROOT/scripts/flashboot/04_verify_consistency.sh"
fi

source "$ROOT/scripts/flashboot/gb300_fabric_env.sh"
WORLD=$((TP_SIZE * PP_SIZE))

# A combined seed+clone scheduler gang exports PET ranks across BOTH instances;
# those are not SGLang node ranks. Single-node instances are always rank 0. For a
# multi-node instance, accept PET values only when that PET gang exactly matches it;
# otherwise require the unambiguous FB_ values.
if [ "$FB_NNODES" -eq 1 ]; then
  export FB_NODE_RANK=${FB_NODE_RANK:-0}
  export FB_MASTER=${FB_MASTER:-127.0.0.1}
else
  if [ -z "${FB_NODE_RANK:-}" ]; then
    [ "${PET_NNODES:-}" = "$FB_NNODES" ] && [ -n "${PET_NODE_RANK:-}" ] ||
      die "set FB_NODE_RANK=0..$((FB_NNODES - 1)); PET ranks are safe only when PET_NNODES=$FB_NNODES"
    export FB_NODE_RANK=$PET_NODE_RANK
  fi
  if [ -z "${FB_MASTER:-}" ]; then
    [ "${PET_NNODES:-}" = "$FB_NNODES" ] && [ -n "${PET_MASTER_ADDR:-}" ] ||
      die "set FB_MASTER to this instance's node-0 IP"
    export FB_MASTER=$PET_MASTER_ADDR
  fi
fi

strict_preflight() {
  [ $((WORLD % FB_NNODES)) -eq 0 ] ||
    die "tp$TP_SIZE x pp$PP_SIZE = $WORLD ranks does not divide over FB_NNODES=$FB_NNODES"
  [ "$FB_NODE_RANK" -ge 0 ] && [ "$FB_NODE_RANK" -lt "$FB_NNODES" ] ||
    die "FB_NODE_RANK=$FB_NODE_RANK is outside 0..$((FB_NNODES - 1))"
  if [ "$FB_NNODES" -gt 1 ] && [ "$FB_MASTER" = 127.0.0.1 ]; then
    die "FB_MASTER is 127.0.0.1 for a multi-node instance"
  fi

  local local_gpus seen rank_files rank
  local_gpus=$((WORLD / FB_NNODES))
  seen=$(nvidia-smi -L 2>/dev/null | wc -l || true)
  [ "$seen" -ge "$local_gpus" ] ||
    die "this node exposes $seen GPUs; this instance needs $local_gpus"

  [ -d "$SHARDS" ] ||
    die "SHARDS=$SHARDS is not a directory; export a GB300/$MOE_RUNNER_BACKEND checkpoint first"
  if [ ! -f "$SHARDS/.EXPORT_DONE" ]; then
    [ "${FB_ALLOW_UNMARKED_SHARDS:-0}" = 1 ] ||
      die "SHARDS=$SHARDS has no .EXPORT_DONE; use the current exporter, or explicitly set FB_ALLOW_UNMARKED_SHARDS=1 for a completed legacy export"
    log "WARNING: accepting an unmarked legacy export after full safetensors validation"
  fi
  rank_files=$(find "$SHARDS" -maxdepth 1 -type f -name 'model-rank-*-part-0.safetensors' | wc -l)
  [ "$rank_files" -eq "$WORLD" ] ||
    die "SHARDS has $rank_files/$WORLD rank files"
  for ((rank = 0; rank < WORLD; rank++)); do
    [ -f "$SHARDS/model-rank-$rank-part-0.safetensors" ] ||
      die "SHARDS is missing model-rank-$rank-part-0.safetensors"
  done

  PYTHONPATH="$FLASHBOOT_PYTHONPATH${PYTHONPATH:+:$PYTHONPATH}" \
    python3 - "$local_gpus" "${BROADCAST_WORLD:-1}" "$SHARDS" "$WORLD" <<'PY' || exit $?
import glob
import inspect
import json
import os
import re
import struct
import sys

needed = int(sys.argv[1])
chain_world = int(sys.argv[2])
shards = sys.argv[3]
world = int(sys.argv[4])

def require(condition, message):
    if not condition:
        raise RuntimeError(message)

# A legacy GB300-deploy export has no .EXPORT_DONE marker. Validate every part's
# complete safetensors extent and exact rank/part set so an in-progress export cannot
# pass merely because all part-0 names have appeared.
parts = {}
for path in glob.glob(os.path.join(shards, "model-rank-*-part-*.safetensors")):
    match = re.search(r"model-rank-(\d+)-part-(\d+)\.safetensors$", path)
    require(match is not None, f"unexpected sharded filename: {path}")
    rank, part = map(int, match.groups())
    require(rank < world, f"checkpoint has out-of-range rank {rank}: {path}")
    require(part not in parts.setdefault(rank, set()),
            f"checkpoint has duplicate rank {rank} part {part}")
    parts[rank].add(part)
    with open(path, "rb") as stream:
        raw = stream.read(8)
        require(len(raw) == 8, f"truncated safetensors length: {path}")
        header_bytes = struct.unpack("<Q", raw)[0]
        require(0 < header_bytes <= 100 * 1024 * 1024,
                f"invalid safetensors header size {header_bytes}: {path}")
        header_raw = stream.read(header_bytes)
        require(len(header_raw) == header_bytes, f"truncated safetensors header: {path}")
    header = json.loads(header_raw)
    ends = [entry["data_offsets"][1] for name, entry in header.items()
            if name != "__metadata__"]
    expected_size = 8 + header_bytes + max(ends, default=0)
    actual_size = os.path.getsize(path)
    require(actual_size == expected_size,
            f"incomplete safetensors part: {path} is {actual_size} bytes, expected {expected_size}")
require(set(parts) == set(range(world)),
        f"checkpoint ranks are {sorted(parts)}, expected 0..{world - 1}")
for rank, rank_parts in parts.items():
    require(rank_parts == set(range(max(rank_parts) + 1)),
            f"checkpoint rank {rank} has non-contiguous parts {sorted(rank_parts)}")

import torch  # load libtorch before flashboot._C
from flashboot import _C
from flashboot.transport import select_transport
from sglang.srt.configs.load_config import LoadFormat
from sglang.srt.model_loader import loader as loader_module
from sglang.srt.server_args import LOAD_FORMAT_CHOICES

require(hasattr(_C, "PeerArenaImporter"), "flashboot._C lacks PeerArenaImporter")
require(hasattr(_C, "check_imex"), "flashboot._C lacks check_imex")
require("FLASHLOAD" in LoadFormat.__members__,
        "sglang loader patch is missing LoadFormat.FLASHLOAD; apply "
        "built-in flashboot integration in the image")
require("FLASHCLONE" in LoadFormat.__members__,
        "sglang loader patch is missing LoadFormat.FLASHCLONE; apply "
        "built-in flashboot integration in the image")
require("flashload" in LOAD_FORMAT_CHOICES and "flashclone" in LOAD_FORMAT_CHOICES,
        "sglang loader patch is missing the public load-format choices")
dispatch = inspect.getsource(loader_module.get_model_loader)
require("get_flashboot_loader" in dispatch and "FLASHCLONE" in dispatch,
        "sglang loader patch is missing get_model_loader() dispatch")
if chain_world > 1:
    require(hasattr(_C, "ChainReceiver"), "flashboot._C lacks ChainReceiver")
    require(hasattr(_C, "chain_zero_flags"), "flashboot._C lacks chain_zero_flags")
for device in range(needed):
    status = _C.check_imex(device)
    print(f"[gb300-fabric][preflight] gpu{device}: "
          f"{'PASS' if status['ok'] else 'FAIL'} — {status['detail']}", flush=True)
    require(status["ok"],
            f"gpu{device} is not in a usable IMEX domain: {status['detail']}")
    require(select_transport(device) == "fabric",
            f"gpu{device}: FLASHBOOT_TRANSPORT did not resolve to fabric")
    if chain_world > 1:
        # Executes a kernel from this build on every local GPU. This catches a stale
        # sm90-only extension before a TP8 model spends minutes constructing itself.
        flags = _C.DeviceArena()
        flags.create(4, device, fabric_exportable=True)
        _C.chain_zero_flags(flags.ptr(0), 1)
        _C.ChainReceiver(device)
PY
  log "preflight PASS: tp=$TP_SIZE pp=$PP_SIZE nnodes=$FB_NNODES " \
      "local_gpus=$local_gpus transport=fabric runner=$MOE_RUNNER_BACKEND"
}

case "$ROLE" in
  preflight)
    strict_preflight
    ;;
  seed)
    strict_preflight
    log "starting fabric seed; stop it with SIGTERM, not SIGKILL"
    exec env FB_MASTER="$FB_MASTER" FB_NODE_RANK="$FB_NODE_RANK" \
      bash "$ROOT/scripts/flashboot/02_run_seed.sh"
    ;;
  clone)
    case "${BROADCAST_WORLD:-1}" in
      ''|*[!0-9]*) die "BROADCAST_WORLD must be an integer >= 1" ;;
      0) die "BROADCAST_WORLD must be >= 1" ;;
    esac
    case "${BROADCAST_RANK:-0}" in
      ''|*[!0-9]*) die "BROADCAST_RANK must be a non-negative integer" ;;
    esac
    if [ "${BROADCAST_WORLD:-1}" -gt 1 ]; then
      [ "${BROADCAST_RANK:-0}" -ge 1 ] &&
        [ "${BROADCAST_RANK:-0}" -le "$BROADCAST_WORLD" ] ||
        die "set a unique BROADCAST_RANK=1..$BROADCAST_WORLD on each clone instance"
    elif [ "${BROADCAST_RANK:-0}" -ne 0 ]; then
      die "BROADCAST_RANK must be 0 when BROADCAST_WORLD=1"
    fi
    : "${SEED_IPS:?set SEED_IPS=<seed node IPs in seed node-rank order>}"
    : "${CLONE_IPS:?set CLONE_IPS=<clone node IPs in clone node-rank order>; this proves seed/clone placement is disjoint>}"
    case "$SEED_IPS" in
      ,*|*,|*,,*) die "SEED_IPS contains an empty node address: $SEED_IPS" ;;
    esac
    case "$CLONE_IPS" in
      ,*|*,|*,,*) die "CLONE_IPS contains an empty node address: $CLONE_IPS" ;;
    esac
    IFS=, read -r -a seed_hosts <<<"$SEED_IPS"
    IFS=, read -r -a clone_hosts <<<"$CLONE_IPS"
    [ "${#seed_hosts[@]}" -eq "$FB_NNODES" ] ||
      die "SEED_IPS lists ${#seed_hosts[@]} node(s), expected $FB_NNODES"
    [ "${#clone_hosts[@]}" -eq "$FB_NNODES" ] ||
      die "CLONE_IPS lists ${#clone_hosts[@]} node(s), expected $FB_NNODES"
    declare -A seen_fabric_hosts=()
    normalized_seed_ips=()
    for seed_host in "${seed_hosts[@]}"; do
      seed_host="${seed_host#"${seed_host%%[![:space:]]*}"}"
      seed_host="${seed_host%"${seed_host##*[![:space:]]}"}"
      [ -n "$seed_host" ] || die "SEED_IPS contains a blank node address"
      [ -z "${seen_fabric_hosts[$seed_host]+x}" ] ||
        die "SEED_IPS repeats node address $seed_host; TP8 ranks must map to distinct nodes"
      seen_fabric_hosts[$seed_host]=seed
      normalized_seed_ips+=("$seed_host")
    done
    normalized_clone_ips=()
    for clone_host in "${clone_hosts[@]}"; do
      clone_host="${clone_host#"${clone_host%%[![:space:]]*}"}"
      clone_host="${clone_host%"${clone_host##*[![:space:]]}"}"
      [ -n "$clone_host" ] || die "CLONE_IPS contains a blank node address"
      [ -z "${seen_fabric_hosts[$clone_host]+x}" ] ||
        die "node address $clone_host appears in both/within placements; seed and clone require $((2 * FB_NNODES)) distinct nodes"
      seen_fabric_hosts[$clone_host]=clone
      normalized_clone_ips+=("$clone_host")
    done
    if [ "$FB_NNODES" -gt 1 ] && [ "${normalized_clone_ips[0]}" != "$FB_MASTER" ]; then
      die "CLONE_IPS node 0 (${normalized_clone_ips[0]}) must equal FB_MASTER ($FB_MASTER)"
    fi
    printf -v SEED_IPS '%s,' "${normalized_seed_ips[@]}"
    export SEED_IPS=${SEED_IPS%,}
    printf -v CLONE_IPS '%s,' "${normalized_clone_ips[@]}"
    export CLONE_IPS=${CLONE_IPS%,}
    strict_preflight
    log "starting strict fabric clone from $SEED_IPS; RDMA and disk fallbacks are disabled"
    exec env FB_MASTER="$FB_MASTER" FB_NODE_RANK="$FB_NODE_RANK" \
      bash "$ROOT/scripts/flashboot/03_run_clone.sh"
    ;;
esac
