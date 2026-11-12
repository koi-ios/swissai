#!/usr/bin/env bash
# Day 2, servekit's own presharding: `servekit launch --only-prepare` runs vLLM's
# sharded-state saver at TP $ARMS_TP through servekit and writes $SK_ARTIFACT with its
# manifest (servekit.json), which run_arms.sh's servekit_* arms launch from.
# Uses the GPUs for a few minutes; skip it if $SK_ARTIFACT already has a manifest.
set -euo pipefail
cd "$(dirname "$0")"
source ./config.env
mkdir -p "$RESULTS/logs"
if [[ -f "$SK_ARTIFACT/servekit.json" ]]; then
  echo "$SK_ARTIFACT is already prepared; delete it to re-shard"
  exit 0
fi
read -r -a EXTRA <<< "$VLLM_EXTRA_ARGS"
env HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 VLLM_NO_USAGE_STATS=1 DO_NOT_TRACK=1 \
  "${SERVEKIT_BIN:-servekit}" launch --servekit-artifact-path "$SK_ARTIFACT" --only-prepare -- \
    "${VLLM_BIN:-vllm}" serve "$MODEL_NET" --tensor-parallel-size "$ARMS_TP" \
    --served-model-name "$SERVED_NAME" ${EXTRA[@]+"${EXTRA[@]}"} 2>&1 | tee "$RESULTS/logs/servekit_prepare.log"
du -sh "$MODEL_NET" "$SK_ARTIFACT"
cat "$SK_ARTIFACT/servekit.json"
