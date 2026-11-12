# Using the harness for other experiments

`run_once.py` measures one cold start of vLLM or SGLang under `servekit profile`. Every idea in this workspace uses it for launches, rather than writing its own launcher. Run `tests/selftest.sh` after any change: it checks both engines and every hook below against a mock engine, without a GPU.

## One launch

```bash
python run_once.py --engine {vllm|sglang} --tp N --model PATH \
    --run LABEL | --arm LABEL   --repeat K   --csv results/X.csv \
    [--caches fresh | persist:DIR | inherit] [--fresh-home] \
    [--evict PATH ...] [--prewarm PATH ...] [--stage-src DIR --stage-dest /dev/shm/DIR] \
    [--env KEY=VALUE ...] [--engine-prefix "CMD"] \
    [--verify-record gold.json | --verify-reference gold.json] [--hold "CMD"] \
    -- <extra engine flags>
```

| Option | What it does |
|---|---|
| `--engine` | `vllm` runs `vllm serve MODEL --tensor-parallel-size N`. `sglang` runs `$SGLANG_CMD --model-path MODEL --tp-size N` (default `python3 -m sglang.launch_server`). Both bind 127.0.0.1. |
| `--caches fresh` | A new empty directory for every compile/JIT cache variable (vLLM, Triton, Inductor, FlashInfer, TileLang, TVM-FFI, DeepGEMM, SGLang 0.5.20's two, XDG, CUDA), deleted afterwards unless `--keep-caches`. |
| `--caches persist:DIR` | The same variables, pointed at a directory kept between runs (warm runs, cache experiments). |
| `--fresh-home` | Also points `HOME` into the cache root, so caches without a variable start empty too. |
| `--env KEY=VALUE` | Extra environment for the engine, applied last (it can override a cache directory). |
| `--engine-prefix "CMD"` | Put a command in front of the engine, inside servekit: `strace -f -o trace.txt`, `env PYTHONPYCACHEPREFIX=...`, `python -X importtime`... |
| `--evict`, `--prewarm` | Make the weights cold (refuses to call a run cold if more than 1% stays resident) or warm. |
| `--stage-src/--stage-dest` | Copy weights into RAM first; the copy is timed and counted in `total_with_staging`. |
| `--verify-record/--verify-reference` | `servekit verify` against the live server. |
| `--servekit-launch ARTIFACT` | Run under `servekit launch --servekit-artifact-path ARTIFACT` instead of `servekit profile` (vLLM only): servekit stages its presharded artifact into `/dev/shm/servekit` itself. With `--caches fresh` the artifact's kept compile caches and servekit's node-local copy are cleared first, so the run is cold. `--servekit-overlap` adds `--overlap`. |
| `--hold "CMD"` | After ready and verify, run a shell command against the live server, then stop it. The command gets `BASE_URL`, `SERVED_NAME`, `RUN_ID` and `RUN_LOG_DIR`. Use it for load tests, latency sweeps or correctness checks. |
| `--no-record` | Measure without writing CSV rows (priming runs). |
| `--dry-run` | Print the command and settings, run nothing. |

## What it writes

- `--csv`: one row per phase: `date, model, tp, run|arm, repeat, phase, seconds`.
  - The phases are servekit's own table (for SGLang: `process_startup`, `tp_worker_spawn`, `torch_distributed_init`, `weight_loading`, `kv_cache_alloc`, the graph captures and `warmup_request(JIT)`), plus `total`.
  - Then the engine's own lines as `vllm:*` or `sglang:*`.
    - SGLang: `sglang:dist_init`, `load_weight`, `decode_graphs`, `prefill_graphs`, `piecewise_graphs`.
  - Then the harness's own wall clocks: `total_http`, `staging`, `total_with_staging`, `first_request`.
- `results/runs.jsonl`: one JSON line per run with every setting, the status, the cache env, log paths and hold/verify results, including failures.
- `results/logs/<run_id>.log` (engine output), `.profile.json` (servekit's report), `.verify.*` and `.hold.txt`.

Exit codes:

| Code | Meaning |
|---|---|
| 0 | ok |
| 1 | failed or not ready |
| 2 | port or GPUs busy, or a bad argument |
| 3 | verify failed |
| 4 | the `--hold` command failed |

## Rules it enforces

- A cold run with eviction that leaves more than 1% resident is refused.
- `SK_TEST_MODE=1`, used only for mocks, skips eviction and says so in every run's metadata.
- The server runs in its own process group. A dropped SSH session or a SIGTERM stops it cleanly, and stragglers are killed.
- A crashed engine is detected in about a second, not at the timeout.
