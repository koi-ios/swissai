#!/usr/bin/env bash
# Day 1 cross-check: vLLM's own startup benchmark, `vllm bench startup`, at each TP.
# It times LLM() construction (no API server) in a fresh subprocess per iteration.
# Its "cold" only swaps VLLM_CACHE_ROOT for an empty temp dir and never evicts the
# page cache, so the weights are read into memory first and it lines up with our
# warm_weights (cold) and warm runs, minus the API server's own startup.
set -euo pipefail
cd "$(dirname "$0")"
source ./config.env
PY=${PYTHON:-python3}
mkdir -p "$RESULTS/logs"
read -r -a EXTRA <<< "$VLLM_EXTRA_ARGS"
VLLM_BIN=${VLLM_BIN:-vllm}

for TP in $TP_LIST; do
  "$PY" evict.py --warm "$MODEL_NET"
  OUT="$RESULTS/bench_startup_tp$TP.json"
  # vLLM keeps Triton and Inductor caches under its (temporary) cache root unless told otherwise
  env -u TRITON_CACHE_DIR -u TORCHINDUCTOR_CACHE_DIR \
    HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 VLLM_NO_USAGE_STATS=1 DO_NOT_TRACK=1 \
    "$VLLM_BIN" bench startup --model "$MODEL_NET" --tensor-parallel-size "$TP" \
      --num-iters-cold "$REPEATS" --num-iters-warmup 1 --num-iters-warm "$REPEATS" \
      --output-json "$OUT" ${EXTRA[@]+"${EXTRA[@]}"} 2>&1 | tee "$RESULTS/logs/bench_startup_tp$TP.log"
done
"$PY" summarize.py
