#!/usr/bin/env bash
# Day 1: vLLM's cold start for the model in config.env, at each TP in $TP_LIST,
# three run types x $REPEATS repeats, interleaved within each repeat so drift on
# shared storage spreads evenly over the types:
#   cold          weights evicted from the page cache, every compile cache empty
#   warm_weights  weights in the page cache, every compile cache empty
#   warm          weights in the page cache, compile caches kept from an earlier run
# Rows go to results/baseline.csv (date, model, tp, run, repeat, phase, seconds).
set -uo pipefail
cd "$(dirname "$0")"
source ./config.env
PY=${PYTHON:-python3}
CSV="$RESULTS/baseline.csv"
[[ -d "$MODEL_NET" ]] || { echo "no checkpoint at $MODEL_NET: run ./setup_vm.sh first" >&2; exit 1; }
read -r -a EXTRA <<< "$VLLM_EXTRA_ARGS"

fails=0
measure() {  # two failures in a row means something is broken, and the GPUs cost money
  "$PY" run_once.py "$@" -- ${EXTRA[@]+"${EXTRA[@]}"}
  local rc=$?
  if (( rc == 0 )); then fails=0; return 0; fi
  fails=$((fails + 1))
  echo "!! run failed (rc=$rc, $fails in a row): $*" >&2
  if (( fails >= 2 )); then
    echo "!! stopping after two failures in a row; see $RESULTS/runs.jsonl and $RESULTS/logs/" >&2
    exit 1
  fi
}

for TP in $TP_LIST; do
  WARM="$WARM_CACHE_ROOT/tp$TP"
  echo "== TP $TP: filling the persistent compile caches in $WARM (not recorded)"
  measure --no-record --run warm_prime --repeat 0 --tp "$TP" --model "$MODEL_NET" --prewarm "$MODEL_NET" --caches "persist:$WARM"
  for R in $(seq 1 "$REPEATS"); do
    measure --csv "$CSV" --run cold         --repeat "$R" --tp "$TP" --model "$MODEL_NET" --evict "$MODEL_NET" --caches fresh
    measure --csv "$CSV" --run warm_weights --repeat "$R" --tp "$TP" --model "$MODEL_NET" --prewarm "$MODEL_NET" --caches fresh
    measure --csv "$CSV" --run warm         --repeat "$R" --tp "$TP" --model "$MODEL_NET" --prewarm "$MODEL_NET" --caches "persist:$WARM"
  done
done
"$PY" summarize.py
