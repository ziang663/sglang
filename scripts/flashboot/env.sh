#!/usr/bin/env bash
# ============================================================================
# Shared configuration. Sourced by every script here; override anything by
# exporting it before launch.
#
# Defaults describe the validated configuration: DeepSeek-V4-Flash, TP8 -- one
# whole 8-GPU node per instance, InfiniBand between nodes. A whole node per pod is
# also what guarantees the seed and the clone land on DIFFERENT machines; two 4-GPU
# pods can be packed onto one host, and then the "cross-node" numbers are not.
# ============================================================================

# ---- model / checkpoint -----------------------------------------------------
# MODEL_PATH: the original HF checkpoint (used only by 01_save_shard_state.sh).
# SHARDS:     the sharded_state checkpoint (model-rank-*-part-*.safetensors plus
#             config.json / tokenizer files). Seed AND clone point at the same
#             directory; the clone reads only the safetensors headers from it and
#             never the weight bytes.
MODEL_PATH=${MODEL_PATH:-/path/to/hf-checkpoint}
SHARDS=${SHARDS:-/path/to/sharded_state}

# ---- parallel topology (must be identical for export, seed and clone) -------
TP_SIZE=${TP_SIZE:-8}
PP_SIZE=${PP_SIZE:-1}
# Read from FB_NNODES, not NNODES. Cluster launchers commonly export NNODES into the
# pod environment (a 1+1 gang exports NNODES=2), and inheriting it is silent and fatal:
# a single-node instance is then launched with --nnodes 2 and blocks forever in
# "Init torch distributed begin", waiting for a peer that is a DIFFERENT instance.
NNODES=${FB_NNODES:-1}

# ---- ports ------------------------------------------------------------------
HTTP_PORT=${HTTP_PORT:-30206}   # seed HTTP entrypoint (node rank 0)
DIST_PORT=${DIST_PORT:-20206}   # torch.distributed rendezvous, instance-internal
NCCL_PORT=${NCCL_PORT:-22206}
SEED_PORT=${SEED_PORT:-21600}   # control plane BASE port: seed rank N listens on
                                # SEED_PORT + N. Plain TCP, used only to exchange
                                # the arena advertisement -- no files are copied
                                # between machines.

# ---- transport ---------------------------------------------------------------
#   rdma  cross-node over InfiniBand (what this repo's clone test measures)
#   ipc   same-node, CUDA IPC handle
#   auto  probe and pick
FLASHBOOT_TRANSPORT=${FLASHBOOT_TRANSPORT:-rdma}
export FLASHBOOT_TRANSPORT

# RDMA backend:
#   nixl  a NIXL agent (UCX underneath). Picks its own rail, and is the default
#         when the NIXL python bindings import.
#   raw   the in-package ibverbs engine, no NIXL agent. Its GPU->HCA map is
#         positional over name-sorted HCAs, which only matches topology when the
#         HCA naming happens to follow it -- measure before choosing it.
# Leave unset to take the default.
# Pinned to raw, and pinned ON PURPOSE. flashboot's own default is nixl whenever the
# NIXL python bindings import -- which they do in the official image -- so a run that
# leaves this unset takes a different path than the one these scripts were measured on,
# and pays a per-rank NIXL agent creation (8-12s measured) that the raw path does not.
# Anyone reproducing the numbers in the README needs the same backend, so the scripts
# choose it rather than inheriting whatever the environment happens to give.
# Set FB_RDMA_BACKEND=nixl to measure the other one; both ends must agree.
FB_RDMA_BACKEND=${FB_RDMA_BACKEND:-raw}
export FB_RDMA_BACKEND

# GPUs per NUMA node, for binding the pinned staging buffers used by the seed's
# disk fill. 8 GPUs across 2 sockets => 4.
export FB_FLASHSHARDED_GPUS_PER_NUMA=${FB_FLASHSHARDED_GPUS_PER_NUMA:-4}

# ---- server flags -------------------------------------------------------------
# IMPORTANT: a sharded_state checkpoint stores tensors in their FINAL
# post-process_weights_after_loading layout, and the MoE runner's expert repack is
# part of that layout. The export (01) and every load of it (02, 03) MUST use the
# same --moe-runner-backend. Mismatch is silent: every byte moves, the server comes
# up, and the weights are wrong.
#
# marlin is the FP4-experts path on Hopper. On hardware with native FP4 support,
# use that hardware's runner -- and re-export with it.
MOE_RUNNER_BACKEND=${MOE_RUNNER_BACKEND:-marlin}

EXTRA_ARGS=${EXTRA_ARGS:-"--context-length 8192 --mem-fraction-static 0.7 \
  --cuda-graph-max-bs 64 --max-running-requests 128 --log-level info"}
if [ -n "$MOE_RUNNER_BACKEND" ]; then
  EXTRA_ARGS="$EXTRA_ARGS --moe-runner-backend $MOE_RUNNER_BACKEND"
fi

# ---- python path --------------------------------------------------------------
# The flashboot package lives in this repo. Drop this if you `pip install -e .`
# from the python/ package instead.
FLASHBOOT_PYTHONPATH="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)/python"
