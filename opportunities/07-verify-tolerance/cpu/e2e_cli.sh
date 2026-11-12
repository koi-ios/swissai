#!/usr/bin/env bash
# `servekit verify` end to end, through its CLI, against the CPU engine's HTTP server on real
# Qwen3-0.6B weights and real corrupted copies of them. Writes results/e2e_cli/.
#
#   1. record a reference with --calibrate 8 --weights from a clean server
#   2. check a fresh clean server              -> auto picks exact: PASS
#   3. check a server whose kernel choices differ (--autotune-seed, a restart that does not
#      reproduce bitwise): auto/exact FAILs, says the drift is within noise and the weight
#      checksums are equal; --mode statistical PASSes
#   4. check a server whose checkpoint has 1/8 of one projection zeroed (a structural fault):
#      FAIL in every mode
#   5. check a server whose checkpoint has 16 mantissa-MSB flips (below the noise floor):
#      exact FAILs; statistical alone PASSes -- no tolerant check can see it -- but
#      statistical with --weights FAILs, because the checksum is exact
#
# STEPS="4 5" reruns only those steps against the reference already in results/e2e_cli/.
set -uo pipefail
cd "$(dirname "$0")/.."
ROOT=$PWD
PY="$ROOT/.venv/bin/python"
OUT=results/e2e_cli
WORK=${WORK:-$OUT/work}
STEPS=${STEPS:-"1 2 3 4 5"}
case " $STEPS " in *" 1 "*) rm -rf "$OUT" ;; esac
mkdir -p "$OUT" "$WORK"
want() { case " $STEPS " in *" $1 "*) return 0 ;; esac; return 1; }
export PYTHONPATH="$ROOT/vendor/servekit/src" HF_HUB_OFFLINE=1
CLEAN=models/Qwen3-0.6B

serve() {  # serve <model dir> <port> [extra args...]; leaves PID in $SRV
  local dir=$1 port=$2; shift 2
  "$PY" -I cpu/server.py --model-path "$dir" --port "$port" --threads 8 "$@" > "$OUT/server-$port.log" 2>&1 &
  SRV=$!
}
stop() { kill "$SRV" 2>/dev/null; wait "$SRV" 2>/dev/null; }
verify() { "$PY" -m servekit.cli verify "$@"; }

if want 1; then
echo "== 1. record + calibrate on a clean server"
serve "$CLEAN" 18101
verify --url http://127.0.0.1:18101 --record "$OUT/gold.json" --calibrate 8 --weights --wait-ready 300 \
  | tee "$OUT/1-record.txt"
stop

fi
if want 2; then
echo "== 2. a fresh clean server"
serve "$CLEAN" 18102
verify --url http://127.0.0.1:18102 --reference "$OUT/gold.json" --weights --wait-ready 300 --out "$OUT/2-clean.json" \
  > "$OUT/2-clean.txt"; echo "exit $?" >> "$OUT/2-clean.txt"
stop

fi
if want 3; then
echo "== 3. clean weights, other kernel choices (a restart that does not reproduce)"
serve "$CLEAN" 18103 --autotune-seed 7
verify --url http://127.0.0.1:18103 --reference "$OUT/gold.json" --weights --wait-ready 300 --out "$OUT/3-autotune-auto.json" \
  > "$OUT/3-autotune-auto.txt"; echo "exit $?" >> "$OUT/3-autotune-auto.txt"
verify --url http://127.0.0.1:18103 --reference "$OUT/gold.json" --mode statistical --out "$OUT/3-autotune-stat.json" \
  > "$OUT/3-autotune-stat.txt"; echo "exit $?" >> "$OUT/3-autotune-stat.txt"
stop

fi
if want 4; then
echo "== 4. 1/8 of one projection zeroed (structural)"
rm -rf "$WORK/zeroed"
"$PY" -m servekit.faults inject --src "$CLEAN" --dst "$WORK/zeroed" --kind zero_slice --tp 8 --shard 3 \
  --match 'layers\.\d+\.mlp\.down_proj\.weight$' --seed 3 | tee "$OUT/4-fault.txt"
serve "$WORK/zeroed" 18104
verify --url http://127.0.0.1:18104 --reference "$OUT/gold.json" --weights --wait-ready 300 --out "$OUT/4-zeroed-auto.json" \
  > "$OUT/4-zeroed-auto.txt"; echo "exit $?" >> "$OUT/4-zeroed-auto.txt"
verify --url http://127.0.0.1:18104 --reference "$OUT/gold.json" --mode statistical --out "$OUT/4-zeroed-stat.json" \
  > "$OUT/4-zeroed-stat.txt"; echo "exit $?" >> "$OUT/4-zeroed-stat.txt"
stop
fi
if want 5; then
echo "== 5. 16 mantissa-MSB flips: below the noise floor"
rm -rf "$WORK/mantflip"
"$PY" -m servekit.faults inject --src "$CLEAN" --dst "$WORK/mantflip" --kind bitflip --bit mant_msb --n 16 \
  --match 'layers\.\d+\.mlp\.down_proj\.weight$' --seed 4 | tee "$OUT/5-fault.txt"
serve "$WORK/mantflip" 18105
verify --url http://127.0.0.1:18105 --reference "$OUT/gold.json" --weights --wait-ready 300 --out "$OUT/5-mantflip-auto.json" \
  > "$OUT/5-mantflip-auto.txt"; echo "exit $?" >> "$OUT/5-mantflip-auto.txt"
verify --url http://127.0.0.1:18105 --reference "$OUT/gold.json" --mode statistical --out "$OUT/5-mantflip-stat.json" \
  > "$OUT/5-mantflip-stat.txt"; echo "exit $?" >> "$OUT/5-mantflip-stat.txt"
verify --url http://127.0.0.1:18105 --reference "$OUT/gold.json" --mode statistical --weights --out "$OUT/5-mantflip-stat-weights.json" \
  > "$OUT/5-mantflip-stat-weights.txt"; echo "exit $?" >> "$OUT/5-mantflip-stat-weights.txt"
stop
fi
rm -rf "$WORK"

for f in "$OUT"/[2-5]-*.txt; do
  case "$f" in *-fault.txt) continue ;; esac
  printf '%-30s %s\n' "$(basename "$f")" "$(grep -E '^(PASS|FAIL)$' "$f" | tail -1) $(tail -1 "$f")"
done | tee "$OUT/summary.txt"
