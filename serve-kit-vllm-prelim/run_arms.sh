#!/usr/bin/env bash
# Day 2: weight-loading arms at TP $ARMS_TP, $REPEATS cold repeats each (the arm's
# checkpoint evicted, every compile cache empty), interleaved by repeat.
#
# Correctness, with servekit verify against the default loader:
#   default_net, repeat 1   records the reference (results/gold.json)
#   default_net, repeat 2   checks itself against it: a control, since two launches of
#                           the same loader must agree, or the tolerance cannot judge the others
#   every other arm         is checked in its repeat 1; a failure is kept, not hidden
# Rows go to results/arms.csv; verify outcomes to results/verify.csv.
#
#   ARMS_ONLY="default_net staged_shm" ./run_arms.sh     # run a subset
set -uo pipefail
cd "$(dirname "$0")"
source ./config.env
PY=${PYTHON:-python3}
CSV="$RESULTS/arms.csv"
GOLD="$RESULTS/gold.json"
VCSV="$RESULTS/verify.csv"
read -r -a EXTRA_ALL <<< "$VLLM_EXTRA_ARGS"

# name | checkpoint the arm reads | staged into $SHM_DIR first? | needs (detect_options.py --has) | vllm serve args
ARMS=(
  "default_net|$MODEL_NET|no||"
  "default_local|$MODEL_LOCAL|no||"
  "lazy_net|$MODEL_NET|no|flag:--safetensors-load-strategy|--safetensors-load-strategy lazy"
  "eager_net|$MODEL_NET|no|flag:--safetensors-load-strategy|--safetensors-load-strategy eager"
  "prefetch8_net|$MODEL_NET|no|flag:--safetensors-prefetch-num-threads|--safetensors-load-strategy prefetch --safetensors-prefetch-num-threads 8"
  "prefetch32_net|$MODEL_NET|no|flag:--safetensors-prefetch-num-threads|--safetensors-load-strategy prefetch --safetensors-prefetch-num-threads 32"
  "instanttensor_net|$MODEL_NET|no|format:instanttensor pkg:instanttensor|--load-format instanttensor"
  "fastsafetensors_net|$MODEL_NET|no|format:fastsafetensors pkg:fastsafetensors|--load-format fastsafetensors"
  "runai_net|$MODEL_NET|no|format:runai_streamer pkg:runai_model_streamer|--load-format runai_streamer"
  "sharded_net|$PRESHARD_NET|no|format:sharded_state|--load-format sharded_state"
  "runai_sharded_net|$PRESHARD_NET|no|format:runai_streamer_sharded pkg:runai_model_streamer|--load-format runai_streamer_sharded"
  "staged_shm|$PRESHARD_NET|yes|format:sharded_state|--load-format sharded_state"
  # servekit itself: its own presharded artifact (run_servekit_prepare.sh), staged by its sliced
  # stager into /dev/shm/servekit, serially or overlapped with engine startup (its vLLM plugin)
  "servekit_serial|$SK_ARTIFACT|servekit|format:sharded_state|"
  "servekit_overlap|$SK_ARTIFACT|servekit-overlap|format:sharded_state|"
)

SELECTED=()
for spec in "${ARMS[@]}"; do
  IFS='|' read -r name src staged needs args <<< "$spec"
  if [[ -n "${ARMS_ONLY:-}" && " $ARMS_ONLY " != *" $name "* ]]; then continue; fi
  if [[ ! -d "$src" ]]; then echo "skip $name: no checkpoint at $src"; continue; fi
  if [[ "$staged" == servekit* && ! -f "$src/servekit.json" ]]; then
    echo "skip $name: no servekit artifact at $src (run ./run_servekit_prepare.sh)"; continue
  fi
  # shellcheck disable=SC2086
  if [[ -n "$needs" ]] && ! "$PY" detect_options.py --has $needs; then
    echo "skip $name: this install lacks $needs"; continue
  fi
  SELECTED+=("$spec")
done
(( ${#SELECTED[@]} )) || { echo "no arms to run" >&2; exit 1; }
echo "arms: $(for s in "${SELECTED[@]}"; do printf '%s ' "${s%%|*}"; done)"
echo "staged arm uses $STAGE_READERS readers x $STAGE_BLOCK_MIB MiB (set STAGE_READERS from the staging sweep)"

FAILED=" "
for R in $(seq 1 "$REPEATS"); do
  for spec in "${SELECTED[@]}"; do
    IFS='|' read -r name src staged needs args <<< "$spec"
    if [[ "$FAILED" == *" $name "* ]]; then echo "skip $name repeat $R: it failed earlier"; continue; fi
    opts=(--csv "$CSV" --arm "$name" --repeat "$R" --tp "$ARMS_TP" --caches fresh --evict "$src" --verify-csv "$VCSV")
    if [[ "$staged" == yes ]]; then
      opts+=(--stage-src "$src" --stage-dest "$SHM_DIR")
    elif [[ "$staged" == servekit* ]]; then
      # the command names the source checkpoint; servekit swaps in its artifact
      opts+=(--model "$MODEL_NET" --servekit-launch "$src")
      [[ "$staged" == servekit-overlap ]] && opts+=(--servekit-overlap)
    else
      opts+=(--model "$src")
    fi
    if [[ "$name" == default_net && "$R" == 1 ]]; then
      opts+=(--verify-record "$GOLD")
    elif [[ -f "$GOLD" ]] && { [[ "$name" == default_net && "$R" == 2 ]] || [[ "$name" != default_net && "$R" == 1 ]]; }; then
      opts+=(--verify-reference "$GOLD")
    fi
    read -r -a xa <<< "$args"
    "$PY" run_once.py "${opts[@]}" -- ${xa[@]+"${xa[@]}"} ${EXTRA_ALL[@]+"${EXTRA_ALL[@]}"}
    rc=$?
    if (( rc == 3 )); then
      echo "!! $name: servekit verify FAILED; timings kept, outcome in $VCSV"
    elif (( rc != 0 )); then
      echo "!! $name repeat $R failed (rc=$rc); skipping its other repeats"
      FAILED+="$name "
      if [[ "$name" == default_net ]]; then echo "!! the reference arm failed: stopping" >&2; exit 1; fi
    fi
  done
done
"$PY" summarize.py
