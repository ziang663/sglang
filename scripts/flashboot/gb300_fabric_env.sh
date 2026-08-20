#!/usr/bin/env bash
# GB300/NVL72 preset for the fabric clone path on this branch.
#
# Source this file before driving scripts/01..04 by hand, or use
# scripts/flashboot/gb300_fabric_clone.sh. User-provided values always win.
#
# The checkpoint is architecture- and runner-specific: export it on GB300 with
# flashinfer_mxfp4, then use the same runner for the seed and every clone. An
# H100/marlin sharded_state checkpoint is not compatible even when every byte moves.

export MODEL_PATH=${MODEL_PATH:-/path/to/DeepSeek-V4-on-GB300}
export SHARDS=${SHARDS:-/path/to/DeepSeek-V4-GB300-tp8-flashinfer_mxfp4}

# One GB300 node has four GPUs. The requested TP8 deployment therefore spans two
# nodes per instance (four nodes for seed + clone). Override both values together
# for a TP4 single-node rehearsal.
export TP_SIZE=${TP_SIZE:-8}
export PP_SIZE=${PP_SIZE:-1}
export FB_NNODES=${FB_NNODES:-2}

export MOE_RUNNER_BACKEND=${MOE_RUNNER_BACKEND:-flashinfer_mxfp4}
_GB300_EXTRA_ARGS="--mem-fraction-static 0.9 \
  --enable-nccl-nvls --enable-symm-mem --cuda-graph-max-bs 32 \
  --max-running-requests 32 --log-level info --skip-server-warmup"
if [[ ${EXTRA_ARGS:-} =~ (^|[[:space:]])--moe-runner-backend($|[=[:space:]]) ]]; then
  echo "gb300_fabric_env.sh: put the runner in MOE_RUNNER_BACKEND, not EXTRA_ARGS; refusing duplicate --moe-runner-backend flags" >&2
  return 2 2>/dev/null || exit 2
fi
EXTRA_ARGS=${EXTRA_ARGS:-$_GB300_EXTRA_ARGS}
_GB300_EXTRA_ARGS_BASE=$EXTRA_ARGS
export EXTRA_ARGS

# Force the actual CUDA fabric-handle path. Strict mode makes the seed serve only
# its CUmemFabricHandle and makes the clone fail instead of switching to RDMA.
export FLASHBOOT_TRANSPORT=fabric
export FB_FABRIC_STRICT=1
export FB_SERVE_STRICT=1
export FB_LOAD_FALLBACK=0

# GB300: four GPUs over two Grace NUMA nodes.
export FB_FLASHSHARDED_GPUS_PER_NUMA=${FB_FLASHSHARDED_GPUS_PER_NUMA:-2}
export FLASHBOOT_CUDA_ARCH=${FLASHBOOT_CUDA_ARCH:-10.0}

export GLOO_SOCKET_IFNAME=${GLOO_SOCKET_IFNAME:-eth0}
export NCCL_SOCKET_IFNAME=${NCCL_SOCKET_IFNAME:-eth0}
export FB_SEED_CONNECT_TIMEOUT_S=${FB_SEED_CONNECT_TIMEOUT_S:-180}
export SGLANG_FABRIC_BROADCAST_TIMEOUT_S=${SGLANG_FABRIC_BROADCAST_TIMEOUT_S:-1800}
# The current implementation's shared fabric/RDMA chain chunk, intentionally 256 MiB.
export FB_CHAIN_CHUNK_MB=${FB_CHAIN_CHUNK_MB:-256}

# Fill the common ports and derived paths. env.sh appends the runner flag, but every
# numbered child sources it again, so hand the BASE argument list down to avoid a
# duplicate --moe-runner-backend option.
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/env.sh"
export EXTRA_ARGS=$_GB300_EXTRA_ARGS_BASE
unset _GB300_EXTRA_ARGS _GB300_EXTRA_ARGS_BASE
