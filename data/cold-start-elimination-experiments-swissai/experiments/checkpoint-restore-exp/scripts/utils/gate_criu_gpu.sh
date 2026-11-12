#!/usr/bin/env bash
# Gate 2 -- criu-gpu round trip on a bare CUDA process. See NOTES.md.
# Usage: scripts/utils/gate_criu_gpu.sh [workdir]
set -u
HERE="$(cd "$(dirname "$0")/../.." && pwd)"
CRIU="$(command -v criu || echo /usr/sbin/criu)"
CCKPT="$HERE/bin/cuda-checkpoint"
COUNTER="$HERE/src/counter"
TICK_MS=1000
FREEZE_WINDOW=8

WORK="${1:-$HERE/results/gate2_criu_gpu/criu-gpu-run}"
LOG="$WORK/counter.log"
IMG="$WORK/img"
rm -rf "$WORK"; mkdir -p "$IMG"

for f in "$CRIU" "$CCKPT" "$COUNTER"; do
  [ -x "$f" ] || { echo "missing/executable: $f"; exit 2; }
done

cleanup() { kill "$PID" 2>/dev/null; }
trap cleanup EXIT

gpu_mem_for() { nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader 2>/dev/null | awk -F', ' -v p="$1" '$1==p {print $2}'; }
last_tick()  { grep -oE 'tick=[0-9]+' "$LOG" 2>/dev/null | tail -1 | cut -d= -f2; }

echo "==== Gate 2 :: criu-gpu (cuda-checkpoint + CRIU full round trip) ===="
echo "criu:  $CRIU ($($CRIU --version 2>&1 | head -1))"
echo "cckpt: $CCKPT ($("$CCKPT" 2>&1 | grep -i version | head -1))"
echo "tick_ms=$TICK_MS freeze_window=${FREEZE_WINDOW}s"
echo

"$COUNTER" "$LOG" "$TICK_MS" &
PID=$!
disown
# First CUDA call (context init) can take a few seconds cold; wait for the first tick
# rather than a fixed sleep, so a slow cudaFree(0) doesn't masquerade as a gate failure.
i=0; until [ -n "$(last_tick)" ]; do sleep 1; i=$((i+1)); [ $i -gt 30 ] && { echo "no tick after 30s"; exit 1; }; done
echo "counter pid=$PID  tick=$(last_tick)"
echo "GPU mem while running: $(gpu_mem_for "$PID")"

echo; echo "-- cuda-checkpoint lock + checkpoint --"
"$CCKPT" --action lock --pid "$PID";       echo "  lock rc=$?"
"$CCKPT" --action checkpoint --pid "$PID"; echo "  checkpoint rc=$?"
echo "  state: $("$CCKPT" --get-state --pid "$PID" 2>&1)"
FROZE=$(last_tick)
echo "  last tick while checkpointed: tick=$FROZE"
echo "  GPU mem while checkpointed: '$(gpu_mem_for "$PID")' (expect empty)"

echo; echo "-- criu dump (snapshot to disk; process is KILLED) --"
"$CRIU" dump -t "$PID" -D "$IMG" --shell-job -v4 -o dump.log
DRC=$?; echo "  dump rc=$DRC"; tail -2 "$IMG/dump.log" 2>/dev/null
if [ $DRC -ne 0 ]; then
  echo; echo "GATE criu-gpu : FAIL (dump rc=$DRC)"; grep -iE 'error|perm' "$IMG/dump.log" | tail -6; exit 1
fi

kill -0 "$PID" 2>/dev/null && echo "  UNEXPECTED: pid alive after dump" || echo "  process gone after dump: yes"
IMG_SZ=$(du -sh "$IMG" | cut -f1)
echo "  snapshot on disk: $IMG_SZ"
echo "  waiting ${FREEZE_WINDOW}s (process is dead; nothing should tick)"
sleep "$FREEZE_WINDOW"

echo; echo "-- criu restore (from disk) --"
"$CRIU" restore -D "$IMG" --shell-job -v4 -o restore.log -d
RRC=$?; echo "  restore rc=$RRC"; tail -2 "$IMG/restore.log" 2>/dev/null
if [ $RRC -ne 0 ]; then
  echo; echo "GATE criu-gpu : FAIL (restore rc=$RRC)"; grep -iE 'error|perm' "$IMG/restore.log" | tail -6; exit 1
fi
RPID=$(grep -oE 'Restoring on pid [0-9]+' "$IMG/restore.log" | grep -oE '[0-9]+' | tail -1)
RPID="${RPID:-$PID}"
echo "  restored pid=$RPID  state: $("$CCKPT" --get-state --pid "$RPID" 2>&1)"

echo; echo "-- cuda-checkpoint restore + unlock --"
"$CCKPT" --action restore --pid "$RPID"; echo "  restore rc=$?"
"$CCKPT" --action unlock --pid "$RPID";  echo "  unlock rc=$?"
echo "  state: $("$CCKPT" --get-state --pid "$RPID" 2>&1)"

sleep 3
AFTER=$(last_tick)
echo "  first-expected after restore: $((FROZE+1))   now at: tick=$AFTER"
echo "  GPU mem after restore: $(gpu_mem_for "$RPID")"

SEQ_OK=$(awk -F= '/tick=/{n=$2; if(n!=prev+1){bad=1}; prev=n} END{print (bad?"BROKEN":"contiguous")}' "$LOG")

echo
echo "  tick sequence: $SEQ_OK (froze at $FROZE)"
if [ "$SEQ_OK" = "contiguous" ] && [ "$AFTER" -gt "$FROZE" ] 2>/dev/null; then
  echo "GATE criu-gpu : PASS  (device memory survived kill+restore-from-disk: froze $FROZE, resumed $((FROZE+1)), now $AFTER)"
  RC=0
else
  echo "GATE criu-gpu : FAIL  (froze=$FROZE after=$AFTER seq=$SEQ_OK)"
  RC=1
fi
kill "$RPID" 2>/dev/null
echo "==== end Gate 2 ===="
exit $RC
