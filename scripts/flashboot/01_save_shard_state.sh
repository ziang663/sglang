#!/usr/bin/env bash
# ============================================================================
# ONE-TIME: produce the sharded_state checkpoint the seed and clone both load.
#
#   single node, TP only:
#     MODEL_PATH=/hf SHARDS=/out bash scripts/flashboot/01_save_shard_state.sh
#
#   several nodes (any PP_SIZE > 1) -- run one copy PER NODE:
#     TP_SIZE=8 PP_SIZE=2 NNODES=2 NODE_RANK=$R MASTER=<node0> \
#       MODEL_PATH=/hf SHARDS=/out bash scripts/flashboot/01_save_shard_state.sh
#
# This runs the model normally (--load-format auto) with FB_SHARD_SAVE_DIR set, and
# built-in flashboot integration makes every rank write its own file at the
# end of load_model.
#
# It does NOT use sglang's save_sharded_model() RPC, and that is the whole point: that
# RPC reaches only PP stage 0's TP group. Under pipeline parallelism the other stages
# write nothing while the call returns success -- measured on tp8/pp2, ranks 0-7 wrote
# and ranks 8-15 did not, in 45s, leaving a checkpoint that looks complete and is
# missing half the model.
#
# The --moe-runner-backend used here is baked into the exported bytes. Every later load
# MUST use the same one -- see the note in env.sh.
# ============================================================================
set -uo pipefail
cd "$(dirname "$0")/../.."
source scripts/flashboot/env.sh

MASTER=${FB_MASTER:-127.0.0.1}
NODE_RANK=${FB_NODE_RANK:-0}
WORLD=$((TP_SIZE * PP_SIZE))
# env.sh does not define GPUS -- each script derives what it needs. Here it is this
# NODE's share of the world, not TP_SIZE: with PP the instance spans NNODES machines.
LOCAL_GPUS=$((WORLD / NNODES))
GPUS=$(python3 -c "print(','.join(str(i) for i in range($LOCAL_GPUS)))")
EXPORT_PORT=${EXPORT_PORT:-30500}
EXPORT_DIST_PORT=${EXPORT_DIST_PORT:-20500}
EXPORT_NCCL_PORT=${EXPORT_NCCL_PORT:-22500}

[ -f "$SHARDS/.EXPORT_DONE" ] && { echo "[export] already done: $SHARDS"; exit 0; }
mkdir -p "$SHARDS"
log() { echo "[export][node $NODE_RANK] $*"; }

# 256B-aligned layout (provided by built-in flashboot integration). Unset for a
# stock-identical export; see README step 2b.
export FB_SHARD_PAD_ALIGNMENT=${FB_SHARD_PAD_ALIGNMENT:-256}
export FB_SHARD_SAVE_DIR="$SHARDS"
export GLOO_SOCKET_IFNAME=${GLOO_SOCKET_IFNAME:-eth0} NCCL_SOCKET_IFNAME=${NCCL_SOCKET_IFNAME:-eth0}

python3 -c "
import sglang.srt.model_executor.model_runner as m
assert 'FB_SHARD_SAVE_DIR' in open(m.__file__).read(), 1" 2>/dev/null || {
  log "this sglang lacks the flashboot save hook -- install this branch"
  exit 1; }

HTTP=(); [ "$NODE_RANK" = 0 ] && HTTP=(--host 0.0.0.0 --port "$EXPORT_PORT")
log "tp=$TP_SIZE pp=$PP_SIZE nnodes=$NNODES world=$WORLD -> $SHARDS"
pkill -9 -f 'sglang.launch_server|managers.scheduler' 2>/dev/null || true; sleep 3
CUDA_VISIBLE_DEVICES=$GPUS python3 -m sglang.launch_server \
  --model-path "$MODEL_PATH" --load-format auto \
  --tp-size "$TP_SIZE" --pp-size "$PP_SIZE" \
  --nnodes "$NNODES" --node-rank "$NODE_RANK" \
  --dist-init-addr "$MASTER:$EXPORT_DIST_PORT" --nccl-port "$EXPORT_NCCL_PORT" \
  --trust-remote-code --mem-fraction-static "${EXPORT_MEM_FRACTION:-0.80}" \
  ${MOE_RUNNER_BACKEND:+--moe-runner-backend $MOE_RUNNER_BACKEND} \
  ${EXPORT_EXTRA:-} --skip-server-warmup --disable-cuda-graph \
  "${HTTP[@]}" > "$SHARDS/export_rank${NODE_RANK}.log" 2>&1 &
PID=$!
stop() { pkill -9 -f 'sglang.launch_server|managers.scheduler' 2>/dev/null || true
         wait "$PID" 2>/dev/null || true; }

if [ "$NODE_RANK" != 0 ]; then
  log "serving until node 0 writes .EXPORT_DONE"
  while [ ! -f "$SHARDS/.EXPORT_DONE" ]; do
    kill -0 "$PID" 2>/dev/null || { log "exited before the export completed"; exit 1; }
    sleep 5
  done
  stop; log "done"; exit 0
fi

# Completion is PRIMARILY /health: the server only becomes ready once every rank has
# finished its save hook, so health implies the last PP stage's layers are on disk too.
# The fallback counts DISTINCT ranks, never raw files -- each rank emits several parts
# and the nodes write at different speeds, so a file-count threshold lets node 0 declare
# done while node 1 is still writing. It also requires the count to hold still, so it
# cannot fire mid-write.
T0=$(date +%s); prev=-1; stable=0; OK=0
for _ in $(seq 1 1080); do
  if curl -sf --max-time 3 "localhost:$EXPORT_PORT/health" >/dev/null 2>&1; then
    OK=1; log "healthy after $(( $(date +%s)-T0 ))s -- every rank has saved"; break; fi
  nf=$(ls "$SHARDS"/model-rank-*.safetensors 2>/dev/null | wc -l)
  nr=$(ls "$SHARDS"/model-rank-*.safetensors 2>/dev/null | grep -aoE 'rank-[0-9]+' | sort -u | wc -l)
  if [ "$nr" -ge "$WORLD" ]; then
    [ "$nf" = "$prev" ] && stable=$((stable+1)) || stable=0
    [ "$stable" -ge 18 ] && { OK=1; log "fallback: $nr/$WORLD ranks, $nf files stable 90s"; break; }
  fi
  kill -0 "$PID" 2>/dev/null || { log "server exited: ranks=$nr files=$nf"; break; }
  prev=$nf; sleep 5
done

nr=$(ls "$SHARDS"/model-rank-*.safetensors 2>/dev/null | grep -aoE 'rank-[0-9]+' | sort -u | wc -l)
[ "$nr" -ge "$WORLD" ] || { log "FAILED: $nr/$WORLD ranks written"; tail -25 "$SHARDS/export_rank0.log"; stop; exit 1; }

find "$MODEL_PATH" -maxdepth 1 -type f ! -name '*.safetensors*' -exec cp -n {} "$SHARDS/" \; 2>/dev/null || true
log "$nr/$WORLD ranks, $(du -sh "$SHARDS" 2>/dev/null | cut -f1)"
[ "$OK" = 1 ] && touch "$SHARDS/.EXPORT_DONE"
sleep 7; stop
log "done (ok=$OK)"
[ "$OK" = 1 ]
