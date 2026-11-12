#!/usr/bin/env bash
# A one-minute check of every script against a mock engine: no GPU, no network.
# Run it on the VM before the first real run (it costs no GPU time), and after any edit.
#   ./tests/selftest.sh
set -uo pipefail
HERE="$(cd "$(dirname "$0")/.." && pwd)"
PY=${PYTHON:-python3}
command -v servekit >/dev/null || { echo "servekit is not installed: pip install git+https://github.com/eth-easl/servekit.git"; exit 1; }
T=$(mktemp -d "${TMPDIR:-/tmp}/sk-selftest.XXXXXX")
if [[ -n "${KEEP:-}" ]]; then echo "keeping $T"; else trap 'rm -rf "$T"' EXIT; fi
pass=0; fail=0
check() {  # check NAME COMMAND...
  local name="$1"; shift
  if "$@" >/dev/null 2>&1; then echo "  ok    $name"; pass=$((pass + 1)); else echo "  FAIL  $name"; fail=$((fail + 1)); fi
}

mkdir -p "$T/bin"
printf '#!/bin/sh\nexec "%s" "%s" "$@"\n' "$(command -v "$PY")" "$HERE/tests/mock_vllm.py" > "$T/bin/vllm"
chmod +x "$T/bin/vllm"
export PATH="$T/bin:$PATH"

echo "== scripts parse"
for f in "$HERE"/*.sh "$HERE"/tests/*.sh; do check "bash -n $(basename "$f")" bash -n "$f"; done
for f in "$HERE"/*.py "$HERE"/tests/*.py; do check "compiles: $(basename "$f")" "$PY" -m py_compile "$f"; done

echo "== stage.py copies byte for byte"
"$PY" - "$T/src" <<'EOF'
import os, sys
d = sys.argv[1]
os.makedirs(os.path.join(d, "sub"), exist_ok=True)
for name, n in {"config.json": 777, "empty.txt": 0, "model-rank-0-part-0.safetensors": 5 * 2**20 + 3,
                "model-rank-1-part-0.safetensors": 3 * 2**20 + 1, "sub/tokenizer.json": 300001}.items():
    with open(os.path.join(d, name), "wb") as f:
        f.write(os.urandom(n))
EOF
same() {
  "$PY" - "$1" "$2" <<'EOF'
import hashlib, os, sys
def digest(root):
    out = {}
    for dp, _, fs in os.walk(root):
        for f in fs:
            p = os.path.join(dp, f)
            with open(p, "rb") as fh:
                out[os.path.relpath(p, root)] = hashlib.sha256(fh.read()).hexdigest()
    return out
sys.exit(0 if digest(sys.argv[1]) == digest(sys.argv[2]) else 1)
EOF
}
for n in 1 4 32; do
  "$PY" "$HERE/stage.py" "$T/src" "$T/dst$n" --readers "$n" --block-mib 1 >/dev/null
  check "readers=$n: identical copy" same "$T/src" "$T/dst$n"
done
"$PY" "$HERE/stage.py" "$T/src" "$T/r1" --rank 1 >/dev/null
check "--rank 1 leaves out rank 0's shard" test ! -e "$T/r1/model-rank-0-part-0.safetensors"
check "--rank 1 keeps the config files" test -e "$T/r1/sub/tokenizer.json"
check "no temporary names left" bash -c "! find '$T' -name '.staging-*' | grep -q ."

echo "== evict.py"
if [[ "$(uname)" == Linux ]]; then
  "$PY" -c "import os; open('$T/big.bin', 'wb').write(os.urandom(256 * 2**20))"
  check "evict exits 0 (<1% left resident)" "$PY" "$HERE/evict.py" "$T/big.bin"
  check "residency ~0% after eviction" bash -c "'$PY' '$HERE/evict.py' --residency --json '$T/big.bin' | '$PY' -c 'import json,sys; sys.exit(json.load(sys.stdin)[\"resident_pct\"] >= 1)'"
  "$PY" "$HERE/evict.py" --warm "$T/big.bin" >/dev/null
  check "residency ~100% after --warm" bash -c "'$PY' '$HERE/evict.py' --residency --json '$T/big.bin' | '$PY' -c 'import json,sys; sys.exit(json.load(sys.stdin)[\"resident_pct\"] <= 99)'"
else
  echo "  skip  eviction (posix_fadvise needs Linux; residency and --warm still run)"
  check "residency works" "$PY" "$HERE/evict.py" --residency "$T/src"
fi

echo "== run_once.py against the mock engine"
export SK_TEST_MODE=1 MOCK_SCALE=0.05 RESULTS="$T/results" FRESH_CACHE_BASE="$T/caches" WARM_CACHE_ROOT="$T/warm" \
  PORT=18765 MODEL_NAME=Mock-8B SERVED_NAME=mock READY_TIMEOUT=60
mkdir -p "$T/model" && echo '{}' > "$T/model/config.json"
RO=("$PY" "$HERE/run_once.py" --csv "$T/results/one.csv" --verify-csv "$T/results/one_verify.csv" --tp 2)
check "launch, records a verify reference" "${RO[@]}" --arm default_net --repeat 1 --model "$T/model" --verify-record "$T/results/gold.json"
check "verify against itself passes" "${RO[@]}" --arm default_net --repeat 2 --model "$T/model" --verify-reference "$T/results/gold.json"
MOCK_PERTURB=1 "${RO[@]}" --arm perturbed --repeat 1 --model "$T/model" --verify-reference "$T/results/gold.json" >/dev/null 2>&1
check "different logprobs fail verify (exit 3)" test $? -eq 3
check "staged arm" "${RO[@]}" --arm staged_shm --repeat 1 --stage-src "$T/src" --stage-dest "$T/shm/x" --stage-readers 4 -- --load-format sharded_state
check "staged copy removed afterwards" test ! -e "$T/shm/x"
check "staging counted in the total" grep -q ",staged_shm,1,total_with_staging," "$T/results/one.csv"
MOCK_FAIL=1 "${RO[@]}" --arm crash --repeat 1 --model "$T/model" >/dev/null 2>&1
check "a crashing engine fails (exit 1)" test $? -eq 1
check "a failed run writes no CSV rows" bash -c "! grep -q ',crash,' '$T/results/one.csv'"
check "fresh cache dirs deleted" bash -c "[ -z \"\$(ls -A '$T/caches')\" ]"

echo "== the same harness on SGLang, and the hooks other experiments use"
SG=("$PY" "$HERE/run_once.py" --engine sglang --sglang-cmd "$PY $HERE/tests/mock_vllm.py sglang"
    --csv "$T/results/sgl.csv" --tp 2 --model "$T/model")
check "sglang: cold launch, caches kept, records a verify reference" \
  "${SG[@]}" --run cold --repeat 1 --caches "persist:$T/sglcache" --verify-record "$T/results/sgl_gold.json"
check "sglang: servekit's phases and SGLang's own lines" bash -c "for p in torch_distributed_init weight_loading cuda_graph_capture total sglang:load_weight sglang:dist_init sglang:decode_graphs; do grep -q \",cold,1,\$p,\" '$T/results/sgl.csv' || exit 1; done"
check "sglang: second launch on the kept cache, verify passes" \
  "${SG[@]}" --run warm --repeat 1 --caches "persist:$T/sglcache" --verify-reference "$T/results/sgl_gold.json"
check "sglang: the kept JIT cache shortens distributed init" "$PY" - "$T/results/sgl.csv" <<'EOF'
import csv, sys
v = {(r["run"], r["phase"]): float(r["seconds"]) for r in csv.DictReader(open(sys.argv[1]))}
sys.exit(0 if v[("warm", "sglang:dist_init")] < v[("cold", "sglang:dist_init")] else 1)
EOF
check "sglang: the ceiling flag reaches the engine" \
  "${SG[@]}" --run cap32 --repeat 1 --no-smoke -- --cuda-graph-max-bs 32
check "sglang: a smaller capture ceiling captures faster" "$PY" - "$T/results/sgl.csv" <<'EOF'
import csv, sys
v = {(r["run"], r["phase"]): float(r["seconds"]) for r in csv.DictReader(open(sys.argv[1]))}
sys.exit(0 if v[("cap32", "sglang:decode_graphs")] < v[("cold", "sglang:decode_graphs")] else 1)
EOF
check "--hold runs against the live server" "${SG[@]}" --run hold --repeat 1 --no-record \
  --hold "\"$PY\" -c \"import os, urllib.request; urllib.request.urlopen(os.environ['BASE_URL'] + '/v1/models'); open('$T/hold.ok', 'w').write(os.environ['SERVED_NAME'])\""
check "--hold saw the server and its name" grep -qx mock "$T/hold.ok"
"${SG[@]}" --run holdfail --repeat 1 --no-record --hold "exit 7" >/dev/null 2>&1
check "a failing --hold exits 4" test $? -eq 4
MOCK_ECHO_ENV=HOME,MOCK_X,MOCK_PREFIXED "${SG[@]}" --run envs --repeat 1 --no-record --fresh-home \
  --env MOCK_X=42 --engine-prefix "env MOCK_PREFIXED=yes" >/dev/null 2>&1
ENVLOG=$(ls -t "$T"/results/logs/*_sglang_envs_tp2_r1.log | head -1)
check "--env reaches the engine" grep -q "mock env MOCK_X=42" "$ENVLOG"
check "--engine-prefix wraps the engine" grep -q "mock env MOCK_PREFIXED=yes" "$ENVLOG"
check "--fresh-home points HOME into the fresh cache root" grep -Eq "mock env HOME=.*/caches/cold-[^/]+/home$" "$ENVLOG"

echo "== the run scripts, one repeat each"
export MODEL_NET="$T/model" MODEL_LOCAL="$T/model-local" PRESHARD_NET="$T/src" SHM_DIR="$T/shm/apertus" \
  TP_LIST=2 REPEATS=1 STAGE_SWEEP=1,4 STAGE_READERS=4
check "run_baseline.sh" "$HERE/run_baseline.sh"
check "baseline.csv has cold, warm_weights, warm" bash -c "grep -q ',cold,1,total,' '$T/results/baseline.csv' && grep -q ',warm_weights,1,' '$T/results/baseline.csv' && grep -q ',warm,1,vllm:compiled_graph_cache_load,' '$T/results/baseline.csv'"
check "run_bench_startup.sh" "$HERE/run_bench_startup.sh"
check "run_staging_sweep.sh" "$HERE/run_staging_sweep.sh"
check "staging.csv has rows" test "$(wc -l < "$T/results/staging.csv")" -gt 2
ARMS_ONLY="default_net lazy_net sharded_net staged_shm" check "run_arms.sh" "$HERE/run_arms.sh"
check "arms.csv has all four arms" bash -c "for a in default_net lazy_net sharded_net staged_shm; do grep -q \",\$a,1,total,\" '$T/results/arms.csv' || exit 1; done"
check "verify.csv: a reference and 3 checks" test "$(grep -c ',1,reference,' "$T/results/verify.csv")" -eq 3

echo "== servekit launch for vLLM (servekit's own stager and plugin)"
SK="$T/sk-artifact"
cp -R "$T/src" "$SK"
"$PY" - "$SK" "$T/model" <<'EOF2'
import json, sys
json.dump({"format": "sharded_state", "engine": "vllm", "engine_version": "0.33.0", "tp_size": 2, "pp_size": 1,
           "dp_size": 1, "dtype": "bfloat16", "quantization": None, "source": sys.argv[2]},
          open(sys.argv[1] + "/servekit.json", "w"))
EOF2
check "--servekit-launch builds servekit's command" bash -c "'$PY' '$HERE/run_once.py' --dry-run --arm x --repeat 1 --tp 2 --no-record --model '$T/model' --servekit-launch '$SK' --servekit-overlap | grep -q '\"launch\",'"
# The stager is the experiment's bash script (GNU find/stat, bash 4) and keeps 10 GiB of /dev/shm free
shm_free=$(df -Pk /dev/shm 2>/dev/null | awk 'NR==2 {print $4 * 1024}')
if [[ "$(uname)" == Linux ]] && (( ${shm_free:-0} > 11 * 1024**3 )); then
  SKO=("$PY" "$HERE/run_once.py" --csv "$T/results/sk.csv" --verify-csv "$T/results/sk_verify.csv" --tp 2 --model "$T/model")
  check "servekit launch, serial" "${SKO[@]}" --arm servekit_serial --repeat 1 --servekit-launch "$SK" --verify-reference "$T/results/gold.json"
  check "serial: the stage is a phase of the cold start" grep -q ",servekit_serial,1,stage," "$T/results/sk.csv"
  check "serial: every rank recorded its load" grep -q ",servekit_serial,1,vllm:sk_load_max_rank,1.3" "$T/results/sk.csv"
  check "servekit launch --overlap" "${SKO[@]}" --arm servekit_overlap --repeat 1 --servekit-launch "$SK" --servekit-overlap
  check "overlap: the stage is not counted, its time is logged" bash -c "! grep -q ',servekit_overlap,1,stage,' '$T/results/sk.csv' && grep -q ',servekit_overlap,1,vllm:sk_stage,' '$T/results/sk.csv'"
  check "servekit freed its /dev/shm copy" test ! -e "/dev/shm/servekit/$(basename "$SK")"
  check "servekit kept the compile caches it saw" test -d "$SK/servekit-caches/vllm"
  SK_ARTIFACT="$SK" ARMS_ONLY="servekit_serial servekit_overlap" check "run_arms.sh, servekit arms" "$HERE/run_arms.sh"
  check "arms.csv has both servekit arms" bash -c "for a in servekit_serial servekit_overlap; do grep -q \",\$a,1,total,\" '$T/results/arms.csv' || exit 1; done"
else
  echo "  skip  servekit launch (needs Linux and /dev/shm with > 11 GiB free; its stager keeps 10 GiB headroom)"
fi

check "summarize.py" "$PY" "$HERE/summarize.py"
check "summary has every section" bash -c "for s in 'Day 1' 'bench startup' 'Staging bandwidth' 'Day 2'; do grep -q \"\$s\" '$T/results/summary.md' || exit 1; done"
mkdir -p "$T/proj" && cp "$HERE/summarize.py" "$HERE/README.md" "$T/proj/"   # a copy: mock numbers never touch the real README
check "summarize.py --readme fills the README block" bash -c "'$PY' '$T/proj/summarize.py' --readme && grep -q 'staged_shm' '$T/proj/README.md' && grep -q 'results:end' '$T/proj/README.md'"
sleep 1
check "no engine processes left" bash -c "! pgrep -f '$HERE/tests/mock_vllm.py' >/dev/null"

echo
echo "$pass passed, $fail failed"
(( fail == 0 ))
