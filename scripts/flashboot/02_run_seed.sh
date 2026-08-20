#!/usr/bin/env bash
# ============================================================================
# SEED: load the sharded_state checkpoint into ONE contiguous GPU arena per rank
# (--load-format flashload) and publish each arena so clones can pull it.
#
# Each seed rank listens on SEED_PORT + rank and serves an advertisement of its
# arena. That TCP control plane is the only cross-instance channel: nothing is
# written to another machine's filesystem, and the weight bytes themselves move
# GPU-to-GPU over the transport (rdma across nodes).
#
#   MODEL_PATH/SHARDS from env.sh, then:
#     bash scripts/flashboot/02_run_seed.sh                       # single node
#     FB_MASTER=<node0-ip> FB_NODE_RANK=0 bash scripts/flashboot/02_run_seed.sh   # multi node
#     FB_MASTER=<node0-ip> FB_NODE_RANK=K bash scripts/flashboot/02_run_seed.sh
#
# Ready when it logs "Application startup complete". Check:
#   curl localhost:$HTTP_PORT/health
#
# Leave this running -- the clone pulls from it. Stop it with SIGTERM (not -9) so
# the transport tears its endpoints down; a hard kill can leave the seed unable to
# establish a queue pair for the NEXT clone.
# ============================================================================
set -euo pipefail
cd "$(dirname "$0")/../.."
source scripts/flashboot/env.sh

MASTER=${FB_MASTER:-127.0.0.1}
NODE_RANK=${FB_NODE_RANK:-0}

# Publish the arenas for clones to pull.
export FB_FLASHSHARDED_SERVE=${FB_SERVE:-1}
export PYTHONPATH="$FLASHBOOT_PYTHONPATH${PYTHONPATH:+:$PYTHONPATH}"

HTTP_FLAGS=""
[ "$NODE_RANK" = "0" ] && HTTP_FLAGS="--host 0.0.0.0 --port $HTTP_PORT"

echo "[seed] shards=$SHARDS tp=$TP_SIZE transport=$FLASHBOOT_TRANSPORT backend=${FB_RDMA_BACKEND:-<default>}"
exec python3 -m sglang.launch_server \
  --model-path "$SHARDS" \
  --load-format flashload \
  --model-loader-extra-config "{\"role\":\"sharded\",\"seed_port\":$SEED_PORT}" \
  --tp-size "$TP_SIZE" --pp-size "$PP_SIZE" --trust-remote-code \
  $EXTRA_ARGS \
  --nnodes "$NNODES" --node-rank "$NODE_RANK" \
  --dist-init-addr "$MASTER:$DIST_PORT" --nccl-port "$NCCL_PORT" \
  $HTTP_FLAGS
