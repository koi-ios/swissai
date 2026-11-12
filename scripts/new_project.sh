#!/usr/bin/env bash
# make new-project SLUG=10-short-name: copy the template into opportunities/ and fill in the id.
# A new project starts parked (decision D2).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
SLUG="${1:-}"
if [[ ! "$SLUG" =~ ^[0-9]{2}-[a-z0-9][a-z0-9-]*$ ]]; then
  echo "usage: make new-project SLUG=10-short-name   (two digits, a hyphen, lower-case words)" >&2
  exit 2
fi
DEST="$ROOT/opportunities/$SLUG"
if [[ -e "$DEST" ]]; then
  echo "new-project: $DEST exists; nothing done" >&2
  exit 1
fi
if ls -d "$ROOT/opportunities/${SLUG:0:2}"-* >/dev/null 2>&1; then
  echo "new-project: the number ${SLUG:0:2} is taken" >&2
  exit 1
fi
cp -R "$ROOT/pipeline/template/project" "$DEST"
ID="${SLUG:0:2}"
for f in project.toml STATUS.md ASSUMPTIONS.md VERIFY.md MODAL.md; do
  sed -i '' -e "s/id = \"NN\"/id = \"$ID\"/" -e "s/NN-slug/$SLUG/g" "$DEST/$f"
done
echo "new-project: created opportunities/$SLUG (parked). Fill in project.toml, then run: make check"
