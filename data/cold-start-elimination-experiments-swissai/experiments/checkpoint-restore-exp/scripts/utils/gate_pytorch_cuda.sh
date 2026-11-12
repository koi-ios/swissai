#!/usr/bin/env bash
# Gate 3a — Python+torch.cuda checkpoint/restore round trip.
#
# De-risks the piece between the tiny CUDA C program (gate 2) and the full SGLang
# server: a real CPython interpreter with the torch/CUDA runtime loaded. Same
# verification contract: the counter lives in device memory and appends every tick, so
# a restore that resumes at froze+1 with a contiguous sequence proves the CUDA context
# + device memory survived.
#
# Usage: scripts/utils/gate_pytorch_cuda.sh [workdir]
set -u
source "$(dirname "$0")/common.sh"

WORK="${1:-$HERE/results/gate3a_pytorch_cuda_roundtrip/gate3a-run}"
LOG="$WORK/py_counter.log"
IMG="$WORK/img"
rm -rf "$WORK"; mkdir -p "$IMG"

echo "==== Gate 3a :: Python+torch.cuda checkpoint/restore ===="
"$PYBIN" "$HERE/src/python_cuda_counter.py" "$LOG" 1000 &
PID=$!
disown
i=0
until grep -q "tick=" "$LOG" 2>/dev/null; do sleep 1; i=$((i+1)); [ $i -gt 60 ] && { echo "no tick"; kill $PID; exit 1; }; done
sleep 2
FROZE=$(grep -oE 'tick=[0-9]+' "$LOG" | tail -1 | cut -d= -f2)
"$CCKPT" --action lock --pid $PID >/dev/null
"$CCKPT" --action checkpoint --pid $PID >/dev/null
"$CRIU" dump -t $PID --shell-job -D "$IMG" -o dump.log || { echo "GATE 3a : FAIL (dump)"; grep -i error "$IMG/dump.log" | tail; exit 1; }
sleep 4
"$CRIU" restore -D "$IMG" --shell-job -d -o restore.log || { echo "GATE 3a : FAIL (restore)"; exit 1; }
"$CCKPT" --action restore --pid $PID >/dev/null
"$CCKPT" --action unlock --pid $PID >/dev/null
sleep 3
FIRST=$(awk -F= -v f="$FROZE" '/tick=/{if($2>f){print $2;exit}}' "$LOG")
kill $PID 2>/dev/null
if [ "$FIRST" = "$((FROZE+1))" ]; then
  echo "GATE 3a: PASS (froze $FROZE, resumed $FIRST)"
else
  echo "GATE 3a: FAIL (froze=$FROZE first=$FIRST expected=$((FROZE+1)))"
  exit 1
fi
echo "==== end Gate 3a ===="
