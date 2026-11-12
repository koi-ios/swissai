#!/usr/bin/env bash
# make backup DEST=/path/to/folder: write a git bundle of this repository there.
# Run by the user. A bundle holds the tracked history only; clones, environments and model files are not in it.
# Restore with: git clone swissai-YYYYMMDD.bundle swissai
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
DEST="${1:-}"
if [[ -z "$DEST" || ! -d "$DEST" ]]; then
  echo "usage: make backup DEST=/an/existing/folder" >&2
  exit 2
fi
if [[ -n "$(git -C "$ROOT" status --porcelain)" ]]; then
  echo "backup: there are uncommitted changes; they will not be in the bundle" >&2
fi
OUT="$DEST/swissai-$(date +%Y%m%d-%H%M).bundle"
git -C "$ROOT" bundle create "$OUT" --all
git -C "$ROOT" bundle verify "$OUT" >/dev/null
echo "backup: wrote $OUT ($(du -h "$OUT" | cut -f1))"
