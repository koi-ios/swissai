#!/usr/bin/env bash
# TP=2 restore: pins both GPUs to match a snapshot taken by tp2-exp/checkpoint.sh. Thin
# wrapper around ../restore.sh -- naive/thin auto-detected from the run dir's meta.txt.
# Usage: scripts/tp2-exp/restore.sh [--cold] <run-dir>   (run-dir under results/tp2-exp/)
set -u
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EXP_ROOT="$(cd "$DIR/../.." && pwd)"

export GPU_IDS="0,1"

exec "$EXP_ROOT/scripts/restore.sh" "$@"
