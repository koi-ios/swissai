#!/usr/bin/env bash
# TP=2 checkpoint: pins both GPUs and writes into results/tp2-exp/{naive,thin}_checkpoint
# instead of results/{naive,thin}_checkpoint. Thin wrapper around ../checkpoint.sh -- see
# NOTES.md for what checkpointing a TP process tree involves.
# Usage: scripts/tp2-exp/checkpoint.sh [--thin] [run-dir]
set -u
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EXP_ROOT="$(cd "$DIR/../.." && pwd)"

THIN=0; [ "${1:-}" = "--thin" ] && { THIN=1; shift; }
MODE_DIR="$([ $THIN = 1 ] && echo thin_checkpoint || echo naive_checkpoint)"
RUN="${1:-$EXP_ROOT/results/tp2-exp/$MODE_DIR/run-$(date -u +%Y%m%dT%H%M%SZ)}"

export GPU_IDS="0,1"
# NCCL's /dev/shm ring-buffer file is tens of MB under TP=2, over criu's 1M ghost-limit
# default -- see the FAILED dump in NOTES.md's TP2 section (once written).
export GHOST_LIMIT="${GHOST_LIMIT:-64M}"

if [ $THIN = 1 ]; then
  exec "$EXP_ROOT/scripts/checkpoint.sh" --thin "$RUN"
else
  exec "$EXP_ROOT/scripts/checkpoint.sh" "$RUN"
fi
