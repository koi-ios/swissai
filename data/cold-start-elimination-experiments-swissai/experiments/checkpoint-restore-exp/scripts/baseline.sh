#!/usr/bin/env bash
# Cold-launch baseline that checkpoint/restore is compared against. See NOTES.md.
# Usage: scripts/baseline.sh [--cold]   (--cold: evict weights + wipe compile caches)
set -u
source "$(dirname "$0")/utils/common.sh"

COLD=0
[ "${1:-}" = "--cold" ] && COLD=1

[ -x "$SERVEKIT" ] || { echo "servekit not found in $VENV -- run setup_env.sh install"; exit 2; }
ls "$MODEL_DIR"/*.safetensors >/dev/null 2>&1 || { echo "weights missing in $MODEL_DIR -- run setup_env.sh download"; exit 2; }

TS="$(date -u +%Y%m%dT%H%M%SZ)"
RESULTS_DIR="${RESULTS_DIR:-$HERE/results/baseline}"
mkdir -p "$RESULTS_DIR"
OUT="$RESULTS_DIR/baseline-${TS}-profile.json"

echo "==== baseline :: Apertus-8B TP=$TP_SIZE profile + bench (GPU $GPU_IDS) ===="
echo "model:   $MODEL_DIR"
echo "config:  ctx=$CONTEXT_LEN mem_frac=$MEM_FRACTION max_running=$MAX_RUNNING tp=$TP_SIZE"
echo "out:     $OUT"

if [ "$COLD" = 1 ]; then
  echo; echo "-- [cold 1/2] evicting weights from page cache --"
  "$PYBIN" "$HERE/scripts/cache_tools.py" evict --verify "$MODEL_DIR"

  echo; echo "-- [cold 2/2] wiping compile caches (flashinfer/triton/inductor/sglang) --"
  COLD_CACHE="$RESULTS_DIR/.cold-compile-cache"
  rm -rf "$COLD_CACHE"; mkdir -p "$COLD_CACHE"
  export FLASHINFER_WORKSPACE_BASE="$COLD_CACHE"
  export SGLANG_CACHE_DIR="$COLD_CACHE/sglang"
  export TRITON_CACHE_DIR="$COLD_CACHE/triton"
  export TORCHINDUCTOR_CACHE_DIR="$COLD_CACHE/inductor"
  export XDG_CACHE_HOME="$COLD_CACHE"
  echo "   compile caches -> $COLD_CACHE (wiped)"
fi

echo; echo "-- launch (servekit profile wraps sglang.launch_server) --"
CUDA_VISIBLE_DEVICES="$GPU_IDS" "$SERVEKIT" profile --out "$OUT" \
  -- \
  "$PYBIN" -m sglang.launch_server \
    --model-path "$MODEL_DIR" \
    --served-model-name "$SERVED_NAME" \
    --host 127.0.0.1 --port "$PORT" \
    --tensor-parallel-size "$TP_SIZE" \
    --trust-remote-code \
    --context-length "$CONTEXT_LEN" \
    --mem-fraction-static "$MEM_FRACTION" \
    --max-running-requests "$MAX_RUNNING" \
    $([ "${DISABLE_CUDA_GRAPH:-0}" = 1 ] && echo --disable-cuda-graph) &
PROF_PID=$!
trap 'kill $PROF_PID 2>/dev/null' EXIT

echo; echo "-- bench (waits for the profile report, then loads the server) --"
"$SERVEKIT" bench --url "http://127.0.0.1:$PORT" --into "$OUT" \
  --wait-ready 1800 --requests 64 --concurrency 16 --input-len 512 --output-len 128
RC=$?

kill $PROF_PID 2>/dev/null
wait $PROF_PID 2>/dev/null
trap - EXIT

echo; echo "==== baseline done (rc=$RC) -> $OUT ===="
exit $RC
