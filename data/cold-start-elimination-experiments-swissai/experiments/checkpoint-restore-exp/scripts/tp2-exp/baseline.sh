#!/usr/bin/env bash
# TP=2 cold-launch baseline: pins both GPUs (TP_SIZE derives from GPU_IDS count in
# config.sh) and writes into results/tp2-exp/baseline instead of results/baseline.
# Thin wrapper around ../baseline.sh -- see NOTES.md for what --cold controls.
# Usage: scripts/tp2-exp/baseline.sh [--cold]
set -u
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EXP_ROOT="$(cd "$DIR/../.." && pwd)"

export GPU_IDS="0,1"
export RESULTS_DIR="$EXP_ROOT/results/tp2-exp/baseline"

exec "$EXP_ROOT/scripts/baseline.sh" "$@"
