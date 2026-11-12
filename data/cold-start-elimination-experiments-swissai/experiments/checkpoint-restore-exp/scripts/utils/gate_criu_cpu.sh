#!/usr/bin/env bash
# Gate 1 — criu-cpu: dump/restore a plain (non-CUDA) process.
#
# Isolates "can CRIU run at all in this environment" from anything GPU-related. A
# trivial loop writes an incrementing tick to a file; we dump it to disk, confirm the
# process is gone and the tick is frozen across a window where several ticks should
# have elapsed, then restore and confirm the counter resumes and advances.
#
# PASS = process disappears on dump, tick stays frozen while dumped, and after restore
#        the FIRST new tick is exactly froze+1 (proving the in-memory counter
#        survived), with a contiguous sequence throughout.
#
# Usage: scripts/utils/gate_criu_cpu.sh [workdir]
set -u
HERE="$(cd "$(dirname "$0")/../.." && pwd)"
CRIU="$(command -v criu || echo /usr/sbin/criu)"
WORK="${1:-$HERE/results/gate1_criu_cpu/criu-cpu-run}"
TICKFILE="$WORK/tick.txt"
IMG="$WORK/img"
rm -rf "$WORK"; mkdir -p "$IMG"

cleanup() { pkill -f "$WORK/counter.sh" 2>/dev/null; }
trap cleanup EXIT

cat > "$WORK/counter.sh" <<EOF
i=0
while true; do i=\$((i+1)); echo "\$i" >> "$TICKFILE"; sleep 0.5; done
EOF

last_tick() { tail -1 "$TICKFILE" 2>/dev/null; }

echo "==== Gate 1 :: criu-cpu (plain process dump/restore) ===="
echo "criu: $CRIU  ($($CRIU --version 2>&1 | head -1))"
echo "getcap: $(getcap "$CRIU" 2>/dev/null || echo none)"
echo

setsid bash "$WORK/counter.sh" </dev/null >/dev/null 2>&1 &
PID=$!
disown
sleep 2
echo "counter pid=$PID  tick before dump=$(last_tick)"

echo; echo "-- criu dump --"
"$CRIU" dump -t "$PID" -D "$IMG" --shell-job -v4 -o dump.log
DRC=$?
echo "dump rc=$DRC"
[ -f "$IMG/dump.log" ] && tail -2 "$IMG/dump.log"
if [ $DRC -ne 0 ]; then
  echo; echo "GATE criu-cpu : FAIL (dump rc=$DRC)"
  echo "-- last errors --"; grep -iE 'error|perm' "$IMG/dump.log" 2>/dev/null | tail -5
  exit 1
fi
kill -0 "$PID" 2>/dev/null && { echo "UNEXPECTED: process still alive after dump"; ALIVE=YES; } || ALIVE=no
FROZE=$(last_tick)
echo "process alive after dump? $ALIVE"
echo "tick frozen at=$FROZE"

echo; echo "-- wait 3s (would be ~6 ticks if still running) --"
sleep 3
STILL=$(last_tick)
echo "tick file now=$STILL  (== $FROZE means truly dumped/frozen)"

echo; echo "-- criu restore --"
"$CRIU" restore -D "$IMG" --shell-job -v4 -o restore.log -d
RRC=$?
echo "restore rc=$RRC"
[ -f "$IMG/restore.log" ] && tail -2 "$IMG/restore.log"
if [ $RRC -ne 0 ]; then
  echo; echo "GATE criu-cpu : FAIL (restore rc=$RRC)"
  grep -iE 'error|perm' "$IMG/restore.log" 2>/dev/null | tail -5
  exit 1
fi

sleep 3
FIRST=$(awk -v f="$FROZE" '$1>f {print $1; exit}' "$TICKFILE")
AFTER=$(last_tick)
echo "first tick after restore=$FIRST  (must be $((FROZE+1)))"
echo "tick now=$AFTER            (just how far it got in the 3s sleep -- carries no correctness info)"

SEQ_OK=$(awk '{n=$1; if(NR>1 && n!=prev+1){bad=1}; prev=n} END{print (bad?"BROKEN":"contiguous")}' "$TICKFILE")
echo "tick sequence: $SEQ_OK"

echo
if [ "$STILL" = "$FROZE" ] && [ "$FIRST" = "$((FROZE+1))" ] && [ "$SEQ_OK" = "contiguous" ]; then
  echo "GATE criu-cpu : PASS  (frozen at $FROZE while dumped; resumed at exactly $FIRST = froze+1, sequence contiguous)"
  RC=0
else
  echo "GATE criu-cpu : FAIL  (froze=$FROZE still=$STILL first=$FIRST expected=$((FROZE+1)) seq=$SEQ_OK)"
  RC=1
fi
pkill -f "$WORK/counter.sh" 2>/dev/null
echo "==== end Gate 1 ===="
exit $RC
