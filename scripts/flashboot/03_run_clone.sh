#!/usr/bin/env bash
# ============================================================================
# CLONE: bring up a second instance WITHOUT reading weight bytes from storage
# (--load-format flashclone).
#
# Each clone rank builds the same arena layout from its shard's safetensors
# HEADERS, connects to the matching seed rank (SEED_PORT + rank), receives the
# arena advertisement, and pulls the whole arena straight out of the seed's GPU
# memory over the transport: RDMA on H100/IB, or a CUmemFabricHandle plus one D2D
# copy inside a GB300 IMEX/NVLink domain.
#
#   SEED_IPS=<seed-node0-ip>[,<seed-node1-ip>...] bash scripts/flashboot/03_run_clone.sh
#
#   SEED_IPS  the SEED instance's node addresses in its node-rank order (one entry
#             for a single-node seed). The clone maps its own rank to the seed node
#             hosting the same rank.
#   FB_MASTER the CLONE's own node0 address (its torch.distributed rendezvous),
#             NOT the seed's. Defaults to 127.0.0.1 for a single-node clone.
#
# Start this only after the seed logs "Application startup complete". If you start
# both together, raise FB_SEED_CONNECT_TIMEOUT_S to cover the seed's startup.
#
# ---- fallback -----------------------------------------------------------------
# By default, a clone that cannot reach the seed falls back to a stock
# sharded_state disk load and still comes up healthy. That is the right behaviour
# in production and the WRONG behaviour in a benchmark: the run "succeeds" while
# reporting disk-read timings that have nothing to do with the path under test.
# These scripts therefore default FB_LOAD_FALLBACK to 0 -- fail loudly instead.
# Set FB_LOAD_FALLBACK=1 to get the production behaviour back.
# ============================================================================
set -euo pipefail
cd "$(dirname "$0")/../.."
source scripts/flashboot/env.sh

SEED_IPS=${SEED_IPS:?set SEED_IPS=<seed node addresses, comma separated>}
MASTER=${FB_MASTER:-127.0.0.1}
NODE_RANK=${FB_NODE_RANK:-0}
BROADCAST_WORLD=${BROADCAST_WORLD:-1}
BROADCAST_RANK=${BROADCAST_RANK:-0}

case "$BROADCAST_WORLD" in
  ''|*[!0-9]*)
    echo "BROADCAST_WORLD must be an integer >= 1, got $BROADCAST_WORLD" >&2
    exit 2
    ;;
esac
[ "$BROADCAST_WORLD" -ge 1 ] || {
  echo "BROADCAST_WORLD must be >= 1, got $BROADCAST_WORLD" >&2
  exit 2
}
case "$BROADCAST_RANK" in
  ''|*[!0-9]*)
    echo "BROADCAST_RANK must be a non-negative integer, got $BROADCAST_RANK" >&2
    exit 2
    ;;
esac

if [ "$BROADCAST_WORLD" -gt 1 ]; then
  [ "$BROADCAST_RANK" -ge 1 ] && [ "$BROADCAST_RANK" -le "$BROADCAST_WORLD" ] || {
    echo "BROADCAST_RANK must be in 1..BROADCAST_WORLD ($BROADCAST_WORLD), got $BROADCAST_RANK" >&2
    exit 2
  }
elif [ "$BROADCAST_RANK" -ne 0 ]; then
  echo "BROADCAST_RANK must be 0 when BROADCAST_WORLD=1, got $BROADCAST_RANK" >&2
  exit 2
fi

# A clone sharing a machine with the seed must not reuse its ports.
HTTP_PORT=${CLONE_HTTP_PORT:-30306}
DIST_PORT=${CLONE_DIST_PORT:-20306}
NCCL_PORT=${CLONE_NCCL_PORT:-22306}

export FB_SEED_CONNECT_TIMEOUT_S=${FB_SEED_CONNECT_TIMEOUT_S:-30}
export FB_LOAD_FALLBACK=${FB_LOAD_FALLBACK:-0}
export PYTHONPATH="$FLASHBOOT_PYTHONPATH${PYTHONPATH:+:$PYTHONPATH}"

HTTP_FLAGS=""
[ "$NODE_RANK" = "0" ] && HTTP_FLAGS="--host 0.0.0.0 --port $HTTP_PORT"

LOADER_CONFIG="{\"role\":\"sharded_clone\",\"seed_ip\":\"$SEED_IPS\",\"seed_port\":$SEED_PORT"
if [ "$BROADCAST_WORLD" -gt 1 ]; then
  LOADER_CONFIG="$LOADER_CONFIG,\"broadcast_world\":$BROADCAST_WORLD,\"broadcast_rank\":$BROADCAST_RANK"
fi
LOADER_CONFIG="$LOADER_CONFIG}"

echo "[clone] seed=$SEED_IPS:$SEED_PORT transport=$FLASHBOOT_TRANSPORT" \
     "backend=${FB_RDMA_BACKEND:-<default>} fallback=$FB_LOAD_FALLBACK" \
     "broadcast=$BROADCAST_RANK/$BROADCAST_WORLD"
exec python3 -m sglang.launch_server \
  --model-path "$SHARDS" \
  --load-format flashclone \
  --model-loader-extra-config "$LOADER_CONFIG" \
  --tp-size "$TP_SIZE" --pp-size "$PP_SIZE" --trust-remote-code \
  $EXTRA_ARGS \
  --nnodes "$NNODES" --node-rank "$NODE_RANK" \
  --dist-init-addr "$MASTER:$DIST_PORT" --nccl-port "$NCCL_PORT" \
  $HTTP_FLAGS
