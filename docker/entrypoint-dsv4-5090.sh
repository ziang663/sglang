#!/usr/bin/env bash
set -euo pipefail

MODEL_PATH="${MODEL_PATH:-/models/preset/deepseek-ai/DeepSeek-V4-Flash/v1.0}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-DeepSeek-V4-Flash-local}"
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-30000}"
TP="${TP:-8}"
MEM_FRACTION_STATIC="${MEM_FRACTION_STATIC:-0.82}"
MAX_RUNNING_REQUESTS="${MAX_RUNNING_REQUESTS:-8}"
CHUNKED_PREFILL_SIZE="${CHUNKED_PREFILL_SIZE:-2048}"
CUDA_GRAPH_BS="${CUDA_GRAPH_BS:-1 2 4 8}"

export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"
export CUDA_PATH="${CUDA_PATH:-$CUDA_HOME}"
export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-12.0a}"
export SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK="${SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK:-1024}"
export PATH="$CUDA_HOME/bin:${PATH}"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"

exec sglang serve \
  --model-path "$MODEL_PATH" \
  --trust-remote-code \
  --tp "$TP" \
  --host "$HOST" \
  --port "$PORT" \
  --mem-fraction-static "$MEM_FRACTION_STATIC" \
  --max-running-requests "$MAX_RUNNING_REQUESTS" \
  --chunked-prefill-size "$CHUNKED_PREFILL_SIZE" \
  --disable-flashinfer-autotune \
  --reasoning-parser deepseek-v4 \
  --served-model-name "$SERVED_MODEL_NAME" \
  --cuda-graph-bs $CUDA_GRAPH_BS \
  "$@"
