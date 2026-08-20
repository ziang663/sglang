#!/usr/bin/env bash
# ============================================================================
# Correctness check: greedy-decode the same prompts on the seed and on the clone
# and require IDENTICAL TOKEN IDS.
#
# This is the check that matters, and it has to be token ids rather than text.
# A clone whose arena pull, rebind or post-load derivation is wrong does not
# crash and does not produce gibberish -- every byte "moves" successfully, the
# server reports ready, and the model is subtly wrong. Comparing rendered text
# can hide a divergence that only shows up a few tokens in; comparing ids cannot.
#
#   SEED_URL=http://<seed-host>:30206 CLONE_URL=http://<clone-host>:30306 \
#     bash scripts/flashboot/04_verify_consistency.sh
#
# Exit status is 0 only if every prompt matched.
# ============================================================================
set -uo pipefail

SEED_URL=${SEED_URL:?set SEED_URL=http://<seed-host>:<port>}
CLONE_URL=${CLONE_URL:?set CLONE_URL=http://<clone-host>:<port>}
MAX_NEW_TOKENS=${MAX_NEW_TOKENS:-32}

# A factual prompt drifts visibly when a permute is wrong; a deterministic
# continuation snaps immediately if an expert repack is off by one element.
PROMPTS=(
  "The capital of France is"
  "1 2 3 4 5 6 7 8"
  "def quicksort(arr):"
  "In a distributed inference system, tensor parallelism means"
)

fail=0
for prompt in "${PROMPTS[@]}"; do
  body=$(python3 -c '
import json, sys
print(json.dumps({"text": sys.argv[1],
                  "sampling_params": {"max_new_tokens": int(sys.argv[2]),
                                      "temperature": 0}}))' "$prompt" "$MAX_NEW_TOKENS")

  for side in seed clone; do
    url=$([ "$side" = seed ] && echo "$SEED_URL" || echo "$CLONE_URL")
    if ! out=$(curl -sf -m 120 "$url/generate" -H 'content-type: application/json' -d "$body"); then
      echo "[verify] ERROR   $side did not answer (${prompt:0:32}...)"
      fail=1; continue 2
    fi
    printf '%s' "$out" > "/tmp/fbverify.$side.json"
  done

  if python3 -c '
import json, sys
a = json.load(open("/tmp/fbverify.seed.json"))
b = json.load(open("/tmp/fbverify.clone.json"))
sys.exit(0 if a["output_ids"] == b["output_ids"] else 1)'; then
    echo "[verify] MATCH    ${prompt:0:40}"
  else
    echo "[verify] MISMATCH ${prompt:0:40}"
    python3 -c '
import json
for name, path in (("seed ", "/tmp/fbverify.seed.json"), ("clone", "/tmp/fbverify.clone.json")):
    d = json.load(open(path))
    print(f"    {name}: {d[\"text\"]!r}")
    print(f"           ids={d[\"output_ids\"]}")'
    fail=1
  fi
done

rm -f /tmp/fbverify.seed.json /tmp/fbverify.clone.json
if [ "$fail" = "0" ]; then
  echo "[verify] PASS -- clone token ids are identical to the seed (${#PROMPTS[@]} prompts, greedy)"
else
  echo "[verify] FAIL -- clone diverged from the seed"
fi
exit $fail
