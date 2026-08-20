#!/usr/bin/env bash
# ============================================================================
# GLM-5.2-FP8 preset for this branch -- the native raw-flashboot path.
#
#   source scripts/flashboot/glm5.2_env.sh    # then drive 01..04 by hand, or
#   bash   scripts/flashboot/glm5.2_run.sh    # let the runner call them in order
#
# This is a PRESET, not a second env.sh. It exports the GLM-5.2 values and then
# sources scripts/flashboot/env.sh, which is written entirely as ${VAR:-default} -- so what
# is set here wins, and every other default and derived value in the branch still
# applies unchanged. The DeepSeek-V4 defaults in env.sh are not touched. For the
# same reason, anything YOU export before sourcing this file wins over everything
# below.
#
# Every value here comes from the tp8/pp2 GLM-5.2-FP8 run that went export -> seed
# -> clone -> identical token ids on FB_RDMA_BACKEND=raw; the one value that is not
# verbatim from it (EXPORT_EXTRA) says so where it is set. Where a flag is here
# because of a specific failure, that failure is written down: none of these are
# preferences.
# ============================================================================

# ---- model / checkpoint -------------------------------------------------------
# MODEL_PATH  the stock HF checkpoint (public id: zai-org/GLM-5.2-FP8). Read only
#             by 01_save_shard_state.sh.
# SHARDS      the exported sharded_state checkpoint. The seed, the clone AND the
#             post-export config fix-up below all point at this one directory.
# Both are placeholders on purpose -- export your own. The MoE runner is baked
# into the exported bytes, so it belongs in the directory name (README's ★ note).
export MODEL_PATH=${MODEL_PATH:-/path/to/GLM-5.2-FP8}
export SHARDS=${SHARDS:-/path/to/GLM-5.2-FP8-tp8pp2-triton}

# ---- topology: TP8 x PP2 = 16 ranks per instance, across 2 machines ------------
# An instance is two whole 8-GPU nodes. Seed and clone are one such instance each,
# so a cross-node clone test is 4 nodes / 32 GPUs -- not 2.
export TP_SIZE=${TP_SIZE:-8}
export PP_SIZE=${PP_SIZE:-2}
# FB_NNODES, never the bare NNODES: env.sh reads NNODES=${FB_NNODES:-1} precisely
# because cluster launchers export NNODES/NODE_RANK/MASTER into the pod, and
# inheriting them is silent and fatal (README, "Multi-node knobs are FB_-prefixed
# on purpose"). Same for FB_NODE_RANK and FB_MASTER, which the scripts read per
# invocation rather than from here.
export FB_NNODES=${FB_NNODES:-2}

# PP_SIZE=2 is also what makes built-in flashboot integration mandatory
# rather than optional: sglang's save_sharded_model() RPC reaches only PP stage 0's
# TP group, so with pp2 ranks 8-15 write nothing while the call returns success --
# a checkpoint that looks complete and is missing half the model. 01 uses the
# per-rank save hook instead and refuses to start without the patch.

# ---- MoE runner ---------------------------------------------------------------
# NOT marlin. The branch defaults to marlin because DeepSeek-V4-Flash has MXFP4
# experts; GLM-5.2 is block-fp8 (e4m3, 128x128) and takes a different path.
#
# An explicit runner is REQUIRED here, not a preference. Fp8MoEMethod assigns
# self.runner only for a known set of backends (deep_gemm / triton / aiter /
# flashinfer_*); anything else falls through a bare `else: pass` and the attribute
# is dereferenced anyway, giving
#     AttributeError: 'Fp8MoEMethod' object has no attribute 'runner'
# on the FIRST FORWARD -- i.e. at warmup, long after the model appears to have
# loaded fine, and after the whole checkpoint has been read.
#
# triton is pinned because it is exactly what the auto path resolves to when
# DeepGEMM is not enabled: it guarantees the attribute exists without choosing a
# compute path the auto logic would not have chosen.
#
# As always on this branch: the runner used at export must be the runner used at
# every load. env.sh appends it to EXTRA_ARGS, 01 passes it to the exporter.
export MOE_RUNNER_BACKEND=${MOE_RUNNER_BACKEND:-triton}

# ---- server flags -------------------------------------------------------------
# Passed to the seed (02) and the clone (03) by env.sh. Flag by flag:
#
#   --context-length 8192            the length this deployment serves.
#   --mem-fraction-static 0.85       46 GB of weights per rank needs more headroom
#                                    than the branch's DeepSeek-V4 default of 0.7.
#   --cuda-graph-max-bs 32           }  the batch shape this model is deployed
#   --max-running-requests 32        }  with. Graph capture and the static memory
#   --max-prefill-tokens 8192        }  above were sized together; keep them together.
#   --json-model-override-args
#     {"moe_router_dtype":"float32"} the tuned deployment config carries
#                                    moe_router_dtype, the raw preset config.json
#                                    does not. Without it the router runs in the
#                                    wrong dtype. See the POST-EXPORT note below:
#                                    this flag alone is not sufficient.
#   --kv-cache-dtype fp8_e4m3        what this model is deployed with.
#   --disable-piecewise-cuda-graph   the deployment disables it for this model.
#   --dist-timeout 1800              a 16-rank instance spanning two machines takes
#                                    a while to rendezvous, and the default window
#                                    is not sized for it.
#   --log-level info                 the branch default, kept so the per-phase
#                                    flashboot lines the README tells you to read
#                                    are actually emitted.
#
# --moe-runner-backend is deliberately NOT in this string: env.sh appends it from
# MOE_RUNNER_BACKEND, and each numbered script sources env.sh itself.
_GLM52_EXTRA_ARGS="--context-length 8192 --mem-fraction-static 0.85 \
  --cuda-graph-max-bs 32 --max-running-requests 32 --max-prefill-tokens 8192 \
  --json-model-override-args {\"moe_router_dtype\":\"float32\"} \
  --kv-cache-dtype fp8_e4m3 --disable-piecewise-cuda-graph \
  --dist-timeout 1800 --log-level info"
EXTRA_ARGS=${EXTRA_ARGS:-$_GLM52_EXTRA_ARGS}
# Keep the BASE list to re-export after sourcing env.sh -- see the note there.
_GLM52_EXTRA_ARGS_BASE=$EXTRA_ARGS
export EXTRA_ARGS

# The export (01) does not use EXTRA_ARGS; it builds its own command line and
# takes EXPORT_EXTRA. The only flag above that the exporter also needs is the
# rendezvous window, for the same reason: 16 ranks over two machines. Everything
# else there is about serving, and the exporter runs with --skip-server-warmup
# and --disable-cuda-graph anyway.
export EXPORT_EXTRA=${EXPORT_EXTRA:-"--dist-timeout 1800"}

# ---- transport ----------------------------------------------------------------
# env.sh already defaults to exactly these two, so this is not a fight -- it is
# the preset saying out loud which path the GLM-5.2 numbers were measured on:
# rdma (across machines there is no IPC path) over the in-package ibverbs engine,
# NOT NIXL. flashboot itself would pick nixl wherever the bindings import, which
# is a different code path and a per-rank agent creation the raw path never pays.
# Both ends of a transfer must run the same backend.
export FLASHBOOT_TRANSPORT=${FLASHBOOT_TRANSPORT:-rdma}
export FB_RDMA_BACKEND=${FB_RDMA_BACKEND:-raw}

# FB_FLASHSHARDED_GPUS_PER_NUMA is left to env.sh's 4, and 4 is correct here: it
# describes GPUs per NUMA node on ONE machine (8 GPUs across 2 sockets), which pp2
# does not change -- pp2 splits the instance across machines, not the node.

# ---- sockets / warmup ---------------------------------------------------------
# 01 sets these two itself for the exporter, but 02 and 03 inherit them from the
# environment, so the preset has to carry them. On a node with several interfaces
# gloo and NCCL will otherwise pick a link that does not route to the peer, and
# the instance hangs in "Init torch distributed begin" rather than failing.
export GLOO_SOCKET_IFNAME=${GLOO_SOCKET_IFNAME:-eth0}
export NCCL_SOCKET_IFNAME=${NCCL_SOCKET_IFNAME:-eth0}
# Shortens the DeepGEMM JIT warmup, which is otherwise paid on every start of
# every one of the four instances in this test.
export SGLANG_JIT_DEEPGEMM_FAST_WARMUP=${SGLANG_JIT_DEEPGEMM_FAST_WARMUP:-1}

# ---- POST-EXPORT, MANDATORY ---------------------------------------------------
# After 01 finishes, moe_router_dtype=float32 must be written into
# $SHARDS/config.json:
#
#   python3 -c "
#   import json, os
#   p = os.environ['SHARDS'] + '/config.json'
#   c = json.load(open(p)); c['moe_router_dtype'] = 'float32'
#   json.dump(c, open(p + '.tmp', 'w'), indent=2); os.replace(p + '.tmp', p)"
#
# 01 copies the raw preset's config.json into $SHARDS, and that copy has no
# moe_router_dtype. The seed and the clone -- and therefore both sides of the
# token-id comparison -- load from $SHARDS, so without this the router runs in the
# wrong dtype however the export itself was run. glm5.2_run.sh does it for you and
# refuses to start a seed or a clone against a checkpoint that is missing it.

# ---- and now the branch's own configuration -----------------------------------
# Sourced LAST: env.sh fills in ports, the transport pinning, FLASHBOOT_PYTHONPATH
# and everything else, and every one of its assignments is ${VAR:-default}, so
# nothing above is overwritten.
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/env.sh"

# env.sh appends "--moe-runner-backend $MOE_RUNNER_BACKEND" to EXTRA_ARGS. Every
# numbered script sources env.sh itself, so what we hand down must be the BASE
# list: letting env.sh's appended copy escape into the child environment gets the
# flag appended a second time in the child, and the server is launched with it
# twice.
export EXTRA_ARGS=$_GLM52_EXTRA_ARGS_BASE
unset _GLM52_EXTRA_ARGS _GLM52_EXTRA_ARGS_BASE
