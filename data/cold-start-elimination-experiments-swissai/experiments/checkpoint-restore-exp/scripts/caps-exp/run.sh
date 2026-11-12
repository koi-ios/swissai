#!/usr/bin/env bash
# Find the minimal criu FILE capabilities checkpoint.sh/restore.sh actually need, when run
# the way criu is actually meant to be run unprivileged: as a genuinely non-root OS user
# (checkpoint.sh/restore.sh always pass --unprivileged), not root with capsh --drop -- see
# NOTES.md ("scripts/caps-exp") for why that earlier approach was replaced. We are always
# unprivileged now: every test runs as a genuinely non-root OS user, and the only thing
# that varies between tests is which Linux file capabilities are set on the copy of the
# `criu` binary that user executes.
#
# This script itself must run as root: only root can `setcap` the criu binary copy. It
# never runs checkpoint.sh/restore.sh itself -- every dump/restore happens as $RUN_USER via
# `sudo -u`, with CRIU pointed at that test's capability-tagged criu copy.
#
# $RUN_USER needs its own venv + model copy under its own $HOME already set up (checkpoint.
# sh/restore.sh default VENV/MODEL_DIR to $HOME) -- this script does not do that setup.
#
# Run dirs land under results/caps-exp/<test>/ (not the default results/naive_checkpoint/),
# so these exploratory runs don't mix with the real experiment's results.
#
# Usage (as root):
#   scripts/caps-exp/run.sh test1|test2|test3 checkpoint          # naive checkpoint only
#   scripts/caps-exp/run.sh test1|test2|test3 restore <run_dir>   # restore --cold
set -u
HERE="$(cd "$(dirname "$0")/../.." && pwd)"

# Named criu FILE-capability sets -- a `setcap` capability string ("" = strip all file
# caps).
CAPS_TEST1=""                                          # no file caps at all
CAPS_TEST2="cap_checkpoint_restore=ep"                 # just the capability whose name implies this use case
CAPS_TEST3="cap_checkpoint_restore,cap_sys_ptrace=ep"  # + ptrace, to seize a process tree the user doesn't own

RUN_USER="${RUN_USER:-user}"
VARIANT_DIR="$HERE/bin/criu-variants"

NAME="${1:-}"; ACTION="${2:-}"; RUN_DIR="${3:-}"
case "$NAME" in
  test1) CAPS="$CAPS_TEST1" ;;
  test2) CAPS="$CAPS_TEST2" ;;
  test3) CAPS="$CAPS_TEST3" ;;
  *) echo "usage: $0 test1|test2|test3 checkpoint|restore [run_dir]"; exit 2 ;;
esac
[ "$ACTION" = checkpoint ] || [ "$ACTION" = restore ] \
  || { echo "usage: $0 test1|test2|test3 checkpoint|restore [run_dir]"; exit 2; }
[ "$ACTION" = restore ] && [ -z "$RUN_DIR" ] && { echo "restore needs a run_dir"; exit 2; }
[ "$(id -u)" = 0 ] || { echo "must run as root (setcap needs it)"; exit 2; }

# Build this test's criu copy: start from a clean copy every time (cp -a would otherwise
# carry over whatever caps the system criu binary itself has), then apply just this test's
# set. "" (test1) means the binary ends up with no file caps at all.
SRC_CRIU="$(command -v criu || echo /usr/sbin/criu)"
VARIANT="$VARIANT_DIR/criu-$NAME"
mkdir -p "$VARIANT_DIR"
cp -f "$SRC_CRIU" "$VARIANT"
setcap -r "$VARIANT" 2>/dev/null
if [ -n "$CAPS" ]; then
  setcap "$CAPS" "$VARIANT" || { echo "setcap failed"; exit 1; }
fi
chmod 755 "$VARIANT"

echo "== $NAME =="
echo "criu: $VARIANT"
echo "caps: $(getcap "$VARIANT" | cut -d' ' -f2-)"
[ -n "$CAPS" ] || echo "caps: (none)"

pkill -9 -f sglang.launch_server 2>/dev/null
sleep 1

case "$ACTION" in
  checkpoint)
    OUT_DIR="results/caps-exp/$NAME/run-$(date -u +%Y%m%dT%H%M%SZ)"
    sudo -u "$RUN_USER" -H env CRIU="$VARIANT" \
      bash -c "cd '$HERE' && ./scripts/checkpoint.sh '$OUT_DIR'"
    ;;
  restore)
    sudo -u "$RUN_USER" -H env CRIU="$VARIANT" \
      bash -c "cd '$HERE' && ./scripts/restore.sh --cold '$RUN_DIR'"
    ;;
esac
