#!/usr/bin/env bash
# Day 2: how fast stage.py copies the presharded checkpoint from the network volume
# into /dev/shm, by number of reader processes ($STAGE_SWEEP), $REPEATS repeats each.
# Each staging starts with the source evicted and the destination empty.
# Then two checks at $STAGE_READERS readers: 64 MiB blocks, and O_DIRECT reads.
# Rows go to results/staging.csv (GB/s is decimal, as servekit's stage_GBps).
set -uo pipefail
cd "$(dirname "$0")"
source ./config.env
PY=${PYTHON:-python3}
CSV="$RESULTS/staging.csv"
[[ -d "$PRESHARD_NET" ]] || { echo "no presharded checkpoint at $PRESHARD_NET: run ./run_preshard.sh first" >&2; exit 1; }

"$PY" stage.py "$PRESHARD_NET" "$SHM_DIR" --sweep "$STAGE_SWEEP" --repeats "$REPEATS" \
  --block-mib "$STAGE_BLOCK_MIB" --csv "$CSV" || exit 1
"$PY" stage.py "$PRESHARD_NET" "$SHM_DIR" --sweep "$STAGE_READERS" --repeats "$REPEATS" \
  --block-mib 64 --csv "$CSV" || echo "!! 64 MiB blocks failed"
"$PY" stage.py "$PRESHARD_NET" "$SHM_DIR" --sweep "$STAGE_READERS" --repeats "$REPEATS" \
  --block-mib "$STAGE_BLOCK_MIB" --modes direct --csv "$CSV" \
  || echo "!! O_DIRECT reads failed on this file system: note it in NOTES.md"
rm -rf "$SHM_DIR"
"$PY" summarize.py
echo
echo "Set STAGE_READERS in config.env to the fastest reader count above, then run ./run_arms.sh"
