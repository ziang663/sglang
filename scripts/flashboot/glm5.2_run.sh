#!/usr/bin/env bash
# ============================================================================
# GLM-5.2-FP8 end to end on this branch: export -> seed -> clone -> token ids.
#
# It calls the branch's own numbered scripts in order -- 01_save_shard_state.sh,
# 02_run_seed.sh, 03_run_clone.sh, 04_verify_consistency.sh -- with the preset in
# scripts/flashboot/glm5.2_env.sh. Nothing is reimplemented here; what this adds is the
# ordering, the one post-export step that lives outside those scripts, and a
# preflight that fails now instead of an hour in.
#
# TOPOLOGY: TP8 x PP2 = 16 ranks per instance = two whole 8-GPU nodes per
# instance. Seed and clone are one instance each, so the full test is 4 nodes:
#
#     node A0  node A1        node B0  node B1
#     \__ exports, then SEEDS __/      \__ CLONES over RDMA __/
#
# Run one copy per node, at the same time:
#
#   # the seed instance (exports first -- 01 must run on BOTH of its nodes)
#   FB_NODE_RANK=0 FB_MASTER=<A0> bash scripts/flashboot/glm5.2_run.sh seed
#   FB_NODE_RANK=1 FB_MASTER=<A0> bash scripts/flashboot/glm5.2_run.sh seed
#
#   # the clone instance (waits for the seed to be healthy, then pulls)
#   FB_NODE_RANK=0 FB_MASTER=<B0> SEED_IPS=<A0>,<A1> bash scripts/flashboot/glm5.2_run.sh clone
#   FB_NODE_RANK=1 FB_MASTER=<B0> SEED_IPS=<A0>,<A1> bash scripts/flashboot/glm5.2_run.sh clone
#
#   FB_MASTER is the rendezvous address of THIS instance's node 0 -- for the clone
#   that is the clone's own node 0, never the seed's. SEED_IPS lists the SEED's
#   nodes in ITS node-rank order, because each clone rank connects to the seed node
#   holding the same rank.
#
# Other roles: `export` (export only, then stop) and `verify` (run 04 by hand,
# SEED_URL/CLONE_URL). Set MODEL_PATH and SHARDS -- the preset ships placeholders.
#
# The four nodes coordinate through two things that already exist, so there is no
# extra control plane here: $SHARDS/.EXPORT_DONE, written by 01 on node 0 and
# already waited on by 01 on the other node, and the seed's own /health endpoint.
# ============================================================================
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
source "$ROOT/scripts/flashboot/glm5.2_env.sh"

ROLE=${1:-${FB_ROLE:-}}
FB_NODE_RANK=${FB_NODE_RANK:-0}
FB_MASTER=${FB_MASTER:-127.0.0.1}
WORLD=$((TP_SIZE * PP_SIZE))

# Node-local knobs of this script only (everything else is a branch name).
LOG_DIR=${GLM52_LOG_DIR:-$SHARDS}          # where 01 already puts export_rank*.log
SETTLE_S=${GLM52_EXPORT_SETTLE_S:-20}      # exporter teardown -> seed start
SEED_READY_S=${GLM52_SEED_READY_TIMEOUT_S:-5400}
CLONE_READY_S=${GLM52_CLONE_READY_TIMEOUT_S:-3600}
CONFIG_WAIT_S=${GLM52_CONFIG_WAIT_S:-900}
# 03 defaults its own HTTP port to 30306; export it so both sides agree even when
# it is overridden, since this script has to poll it.
export CLONE_HTTP_PORT=${CLONE_HTTP_PORT:-30306}

log()  { echo "[glm5.2][$ROLE node ${FB_NODE_RANK}] $*"; }
die()  { echo "[glm5.2][$ROLE node ${FB_NODE_RANK}] FAILED: $*" >&2; exit 1; }

# ---------------------------------------------------------------------------
# preflight. Each check below stands for a failure that presents as success at
# the moment it happens and only surfaces an hour later.
# ---------------------------------------------------------------------------
preflight_common() {
  case "$ROLE" in export|seed|clone|verify) ;; *)
    die "role must be one of: export | seed | clone | verify (got '${ROLE:-<unset>}') -- see the header" ;;
  esac
  if [ "$ROLE" = verify ]; then return 0; fi

  [ $((WORLD % NNODES)) -eq 0 ] ||
    die "tp$TP_SIZE x pp$PP_SIZE = $WORLD ranks does not divide over FB_NNODES=$NNODES"
  LOCAL_GPUS=$((WORLD / NNODES))
  [ "$FB_NODE_RANK" -lt "$NNODES" ] ||
    die "FB_NODE_RANK=$FB_NODE_RANK but this instance has only $NNODES nodes (0..$((NNODES-1)))"

  # A 16-rank instance packed onto fewer GPUs than it asks for does not fail at
  # launch -- it fails deep inside init, after the checkpoint has been read.
  local seen
  seen=$(nvidia-smi -L 2>/dev/null | wc -l || true)
  [ "$seen" -ge "$LOCAL_GPUS" ] ||
    die "this node offers $seen GPUs, a $WORLD-rank instance over $NNODES nodes needs $LOCAL_GPUS here"

  # The bare-NNODES trap in reverse: a multi-node instance left on 127.0.0.1 has
  # every node convinced it is its own rendezvous master, and all of them block in
  # "Init torch distributed begin" until --dist-timeout expires.
  if [ "$NNODES" -gt 1 ] && [ "$FB_MASTER" = "127.0.0.1" ]; then
    die "FB_MASTER is still 127.0.0.1 with FB_NNODES=$NNODES -- set it to this instance's node 0 address"
  fi

  [ -n "${MOE_RUNNER_BACKEND:-}" ] ||
    die "MOE_RUNNER_BACKEND is empty; GLM-5.2 needs an explicit one (triton) or Fp8MoEMethod
         raises AttributeError: no attribute 'runner' on the first forward, at warmup"

  if [ "$NNODES" -gt 1 ] && [ "$FLASHBOOT_TRANSPORT" != "rdma" ]; then
    die "FLASHBOOT_TRANSPORT=$FLASHBOOT_TRANSPORT across machines -- there is no IPC path between nodes"
  fi
  log "tp=$TP_SIZE pp=$PP_SIZE nnodes=$NNODES world=$WORLD gpus_here=$LOCAL_GPUS" \
      "runner=$MOE_RUNNER_BACKEND transport=$FLASHBOOT_TRANSPORT backend=${FB_RDMA_BACKEND:-<default>}"
}

# The export is the only step that reads MODEL_PATH.
preflight_export() {
  [ -d "$MODEL_PATH" ] ||
    die "MODEL_PATH=$MODEL_PATH is not a directory (the preset ships a placeholder -- export your own)"
  mkdir -p "$SHARDS"
}

# One file per GLOBAL rank -- 16 under tp8/pp2 -- plus 01's own completion marker.
# A partial export is the dangerous case: it loads cleanly and answers wrongly. The
# glob matches part-0 only, so this counts RANKS; counting every file would count
# the several parts each rank emits and pass on half a checkpoint.
require_complete_export() {
  local n
  n=$(ls "$SHARDS"/model-rank-*-part-0.safetensors 2>/dev/null | wc -l || true)
  [ "$n" -eq "$WORLD" ] ||
    die "export incomplete: $n/$WORLD rank files in $SHARDS -- run the export role first"
  [ -f "$SHARDS/.EXPORT_DONE" ] ||
    die "$SHARDS has $WORLD rank files but no .EXPORT_DONE -- the export did not finish cleanly"
  log "export: $n/$WORLD rank files in $SHARDS"
}

# The seed and the clone read only $SHARDS.
preflight_shards() {
  require_complete_export

  # The loader patch. Without it --load-format flashload is simply not a choice,
  # and sglang rejects the argument after every other check has passed.
  PYTHONPATH="$FLASHBOOT_PYTHONPATH${PYTHONPATH:+:$PYTHONPATH}" python3 -c "
from sglang.srt.configs.load_config import LoadFormat
assert LoadFormat.__members__.get('FLASHLOAD') is not None, 1" 2>/dev/null ||
    die "this sglang has no FLASHLOAD -- apply built-in flashboot integration"

  # An extension built without ibverbs imports perfectly well and has no transport.
  PYTHONPATH="$FLASHBOOT_PYTHONPATH${PYTHONPATH:+:$PYTHONPATH}" python3 -c "
import torch
from flashboot import _C
_C.DeviceArena()
assert hasattr(_C, 'Endpoint'), 1" 2>/dev/null ||
    die "flashboot._C has no rdma engine -- rebuild with FB_BUILD_RDMA=1"
}

# ---------------------------------------------------------------------------
# the post-export step that lives outside 01
# ---------------------------------------------------------------------------
# 01 copies the raw checkpoint's config.json into $SHARDS, and that copy has no
# moe_router_dtype. Seed, clone and both sides of the token-id comparison load
# from $SHARDS, so without this the router runs in the wrong dtype however the
# export itself was run. Written through a temp file and os.replace because the
# other three nodes poll this file.
inject_router_dtype() {
  local msg
  msg=$(SHARDS="$SHARDS" python3 -c "
import json, os
p = os.path.join(os.environ['SHARDS'], 'config.json')
c = json.load(open(p))
if c.get('moe_router_dtype') == 'float32':
    print('moe_router_dtype=float32 already in the exported config.json'); raise SystemExit
c['moe_router_dtype'] = 'float32'
json.dump(c, open(p + '.tmp', 'w'), indent=2)
os.replace(p + '.tmp', p)
print('moe_router_dtype=float32 written into the exported config.json')") ||
    die "could not inject moe_router_dtype into $SHARDS/config.json"
  log "$msg"
}

# Every other node waits for that write rather than assuming it happened: node 1
# of the seed instance leaves 01 the moment .EXPORT_DONE appears, which is before
# node 0 has fixed the config up.
require_router_dtype() {
  local waited=0 limit=${1:-0}
  while : ; do
    if SHARDS="$SHARDS" python3 -c "
import json, os, sys
c = json.load(open(os.path.join(os.environ['SHARDS'], 'config.json')))
sys.exit(0 if c.get('moe_router_dtype') == 'float32' else 1)" 2>/dev/null; then
      return 0
    fi
    if [ "$waited" -ge "$limit" ]; then break; fi
    if [ "$waited" = 0 ]; then log "waiting for moe_router_dtype=float32 in $SHARDS/config.json"; fi
    sleep 5; waited=$((waited + 5))
  done
  die "$SHARDS/config.json has no moe_router_dtype=float32 -- the post-export injection did not run"
}

# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
wait_health() {  # host port timeout_s
  local waited=0
  while [ "$waited" -lt "$3" ]; do
    if curl -sf -m 3 "http://$1:$2/health" >/dev/null 2>&1; then return 0; fi
    sleep 5; waited=$((waited + 5))
  done
  return 1
}

# The README's "Reading the result": the per-phase lines are the evidence, the
# server being up is not.
# Takes the logs of EVERY node of the instance, not just this one. LOG_DIR defaults to
# $SHARDS, which all nodes already share -- they all load the export from it.
# Reading node 0 alone is the trap this whole file exists to avoid: with PP2 half the
# ranks live on node 1, and the slowest rank of the instance is as likely to be there.
# Measured on a real GLM-5.2 pp2 run: node 0 said 14.48s, node 1 said 15.27s. The
# instance was ready at 15.27s. A node-0-only reading is not a rounding error, it is
# the wrong number.
phase_summary() {  # glob-prefix, e.g. "$LOG_DIR/clone_node"
  local lwe n
  local -a logs=( "$1"*.log )
  [ -e "${logs[0]}" ] || { echo "    no logs matched ${1}*.log"; return 0; }
  grep -aohE "\[flashboot\]\[(fill|seed|clone|rdma)\][^\"]{0,95}" "${logs[@]}" 2>/dev/null |
    sort -u | head -10 || true
  # Sorted, so the first and last numbers are the fastest and the slowest rank --
  # an instance is ready when its SLOWEST rank is, so read that one.
  lwe=$(grep -aohE "Load weight end\. elapsed=[0-9.]+ s" "${logs[@]}" 2>/dev/null |
        sed -E 's/.*elapsed=([0-9.]+) s/\1/' | sort -g | paste -sd' ' - || true)
  if [ -n "${lwe// /}" ]; then
    n=$(printf '%s\n' $lwe | grep -c .)
    echo "    load weight end, per rank (s): $lwe"
    echo "    ranks seen: $n of $WORLD (across ${#logs[@]} node log(s))"
    # Say it rather than let a half-sized sample look complete.
    [ "$n" -eq "$WORLD" ] ||
      echo "    WARNING: only $n of $WORLD ranks reported -- the slowest rank may be missing"
  fi
  return 0
}

# ---------------------------------------------------------------------------
# steps
# ---------------------------------------------------------------------------
EXPORT_RAN=0
run_export() {
  if [ -f "$SHARDS/.EXPORT_DONE" ]; then
    log "export already present in $SHARDS -- skipping 01"
    return 0
  fi
  preflight_export
  # 01 must run on BOTH nodes of the exporting instance: with pp2 the per-rank save
  # hook is what writes ranks 8-15, and they live on node 1.
  log "01_save_shard_state.sh -- the long step (run this role on both nodes)"
  FB_MASTER="$FB_MASTER" FB_NODE_RANK="$FB_NODE_RANK" \
    bash "$ROOT/scripts/flashboot/01_save_shard_state.sh" ||
    die "export failed on node $FB_NODE_RANK -- see $SHARDS/export_rank${FB_NODE_RANK}.log"
  EXPORT_RAN=1
}

run_seed() {
  preflight_shards
  require_router_dtype "$CONFIG_WAIT_S"
  mkdir -p "$LOG_DIR"
  local seed_log="$LOG_DIR/seed_node${FB_NODE_RANK}.log"

  if [ "$FB_NODE_RANK" != "0" ]; then
    # Only node 0 serves HTTP; the other nodes just join the instance. exec, so
    # this script gets out of the way of signal handling -- the README asks for
    # SIGTERM, not SIGKILL, and a hard-killed seed can leave the next clone
    # unable to establish a queue pair.
    # Redirect, do not exec bare: seed_node1.log has to exist for the slowest-rank
    # summary to cover all 16 ranks. With PP2 half the instance lives on this node.
    log "joining the seed instance (node 0 serves HTTP) -> $seed_log"
    exec env FB_MASTER="$FB_MASTER" FB_NODE_RANK="$FB_NODE_RANK" \
      bash "$ROOT/scripts/flashboot/02_run_seed.sh" > "$seed_log" 2>&1
  fi

  log "starting the seed -> $seed_log"
  FB_MASTER="$FB_MASTER" FB_NODE_RANK="$FB_NODE_RANK" \
    bash "$ROOT/scripts/flashboot/02_run_seed.sh" > "$seed_log" 2>&1 &
  local pid=$!
  if ! wait_health 127.0.0.1 "$HTTP_PORT" "$SEED_READY_S"; then
    tail -30 "$seed_log" || true
    kill "$pid" 2>/dev/null || true
    die "seed never became healthy on port $HTTP_PORT"
  fi
  log "SEED healthy on port $HTTP_PORT -- the clone nodes can start now"
  phase_summary "$LOG_DIR/seed_node"
  log "leave this running; stop it with SIGTERM (kill $pid), never -9"

  local rc=0
  wait "$pid" || rc=$?
  [ "$rc" = 143 ] && rc=0   # SIGTERM is the documented way to stop it
  log "seed exited (status $rc)"
  return "$rc"
}

run_clone() {
  local seed_ips=${SEED_IPS:?set SEED_IPS=<seed node addresses in seed node-rank order, comma separated>}
  local n_seed seed0
  n_seed=$(awk -F, '{print NF}' <<<"$seed_ips")
  [ "$n_seed" -eq "$NNODES" ] ||
    die "SEED_IPS lists $n_seed node(s) but the seed instance has $NNODES -- each clone rank connects to the seed node holding the same rank"
  seed0=${seed_ips%%,*}

  # Two containers on one host are not two machines, and the resulting number is
  # an intra-node measurement wearing a cross-node label.
  if hostname -i 2>/dev/null | tr ' ' '\n' | grep -qxF "$seed0"; then
    log "WARNING: this node's address is also in SEED_IPS -- seed and clone look co-located"
  fi

  mkdir -p "$LOG_DIR"
  local clone_log="$LOG_DIR/clone_node${FB_NODE_RANK}.log"

  # Wait for the seed BEFORE asserting anything about $SHARDS. The header tells the
  # operator to launch all four nodes at once, so at t=0 the export has not started and
  # $SHARDS is empty -- checking it first made both clone nodes die in ~50ms while the
  # seed arm went on to spend hours exporting for a peer that was already gone.
  # A healthy seed is proof the export finished AND the router dtype was injected, so
  # the two assertions below become the cheap confirmations they were meant to be.
  # 03 itself gives up on the seed after FB_SEED_CONNECT_TIMEOUT_S (30s by default),
  # which is why the wait lives here and not inside it.
  log "waiting for the seed at $seed0:$HTTP_PORT"
  wait_health "$seed0" "$HTTP_PORT" "$SEED_READY_S" ||
    die "seed at $seed0:$HTTP_PORT never became healthy"

  preflight_shards
  require_router_dtype "$CONFIG_WAIT_S"

  if [ "$FB_NODE_RANK" != "0" ]; then
    # Redirect, do not exec bare: without this, clone_node1.log is never created and
    # every later check -- the slowest-rank summary and the fallback grep -- silently
    # sees only ranks 0-7 of a 16-rank instance. With PP2 the slowest rank is as likely
    # to be on this node, and a rank that fell back to a disk load still serves correct
    # tokens, so the run would print PASS over half-RDMA, half-disk timings.
    log "joining the clone instance (node 0 serves HTTP and runs the check) -> $clone_log"
    exec env SEED_IPS="$seed_ips" FB_MASTER="$FB_MASTER" FB_NODE_RANK="$FB_NODE_RANK" \
      bash "$ROOT/scripts/flashboot/03_run_clone.sh" > "$clone_log" 2>&1
  fi

  log "starting the clone against $seed_ips -> $clone_log"
  local t0 pid rc=0
  t0=$(date +%s)
  SEED_IPS="$seed_ips" FB_MASTER="$FB_MASTER" FB_NODE_RANK="$FB_NODE_RANK" \
    bash "$ROOT/scripts/flashboot/03_run_clone.sh" > "$clone_log" 2>&1 &
  pid=$!
  if ! wait_health 127.0.0.1 "$CLONE_HTTP_PORT" "$CLONE_READY_S"; then
    grep -aE "Error|Traceback|fallback" "$clone_log" | tail -12 || true
    kill "$pid" 2>/dev/null || true
    die "clone never became healthy on port $CLONE_HTTP_PORT"
  fi
  log "CLONE healthy in $(( $(date +%s) - t0 ))s"
  phase_summary "$LOG_DIR/clone_node"

  # 03 sets FB_LOAD_FALLBACK=0, so this cannot normally happen -- but a clone that
  # read the weights off disk is the one failure that still reports success, so it
  # is worth one grep before believing any timing above.
  # Every node's log, not just this one: the fallback line is printed per rank, so a
  # rank on node 1 that read from disk would otherwise pass unnoticed -- and it serves
  # correct tokens, so the verify below would still say PASS over half-disk timings.
  if grep -aq "\[flashboot\]\[fallback\]" "$LOG_DIR"/clone_node*.log 2>/dev/null; then
    kill "$pid" 2>/dev/null || true
    die "a clone rank fell back to a disk load -- the numbers above are disk timings, not RDMA"
  fi

  log "04_verify_consistency.sh: seed and clone must return IDENTICAL token ids"
  SEED_URL="http://$seed0:$HTTP_PORT" CLONE_URL="http://127.0.0.1:$CLONE_HTTP_PORT" \
    bash "$ROOT/scripts/flashboot/04_verify_consistency.sh" || rc=$?

  kill "$pid" 2>/dev/null || true   # SIGTERM, so the transport tears its endpoints down
  wait "$pid" 2>/dev/null || true
  [ "$rc" = 0 ] || die "token ids diverged -- see $clone_log"
  log "PASS -- exported, seeded, cloned over ${FB_RDMA_BACKEND:-<default>} rdma, token ids identical"
  log "the seed is still running on $seed0; stop it with SIGTERM"
  return 0
}

# ---------------------------------------------------------------------------
preflight_common
case "$ROLE" in
  export)
    run_export
    require_complete_export
    # Node 0 is the only writer; the other node waits for the write to land so
    # that "export finished" means the same thing everywhere.
    if [ "$FB_NODE_RANK" = "0" ]; then
      inject_router_dtype
    else
      require_router_dtype "$CONFIG_WAIT_S"
    fi
    log "export complete -- now run the seed role on these same nodes"
    ;;
  seed)
    # The exporting instance is the seeding instance, so this role runs 01 first
    # and skips it when $SHARDS already holds a finished export.
    run_export
    require_complete_export
    if [ "$FB_NODE_RANK" = "0" ]; then inject_router_dtype; fi
    if [ "$EXPORT_RAN" = "1" ]; then
      # 01 kills the exporter and returns; the 16 processes still have to release
      # their GPU memory before an instance at --mem-fraction-static 0.85 asks for
      # it. The proven run paused here too.
      log "letting the exporter release its GPU memory (${SETTLE_S}s)"
      sleep "$SETTLE_S"
    fi
    run_seed
    ;;
  clone)
    run_clone
    ;;
  verify)
    : "${SEED_URL:?set SEED_URL=http://<seed-host>:$HTTP_PORT}"
    : "${CLONE_URL:?set CLONE_URL=http://<clone-host>:$CLONE_HTTP_PORT}"
    bash "$ROOT/scripts/flashboot/04_verify_consistency.sh"
    ;;
esac
