#!/usr/bin/env bash
# Day 2, step 1: preshard the checkpoint at TP $ARMS_TP with vLLM's sharded-state saver,
# into $PRESHARD_NET on the network volume. Uses both GPUs for a few minutes.
set -euo pipefail
cd "$(dirname "$0")"
source ./config.env
PY=${PYTHON:-python3}
mkdir -p "$RESULTS/logs"
if [[ -d "$PRESHARD_NET" ]] && [[ -n "$(ls -A "$PRESHARD_NET")" ]]; then
  echo "$PRESHARD_NET already exists; delete it to re-shard"
  exit 0
fi
env HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 VLLM_NO_USAGE_STATS=1 DO_NOT_TRACK=1 \
  "$PY" preshard.py --model "$MODEL_NET" --tensor-parallel-size "$ARMS_TP" --enforce-eager \
    --output "$PRESHARD_NET" 2>&1 | tee "$RESULTS/logs/preshard.log"
du -sh "$MODEL_NET" "$PRESHARD_NET"
ls -l "$PRESHARD_NET" | head -20
