# servekit for vLLM: a preliminary result

Can [servekit](https://github.com/eth-easl/servekit)'s fast cold starts carry over to vLLM? This repository measures where vLLM's cold start goes for Apertus-8B-Instruct-2509. It then compares staging presharded weights in RAM against vLLM's default and built-in loaders, on one rented 2× H100 VM.

<!-- results:start -->
_No results yet: run the steps below, then `python summarize.py --readme`._
<!-- results:end -->

This ports the approach of servekit and of Youssef Boughizane's article [servekit: fast LLM weight loading from Lustre datastores](https://serving.swissai.svc.cscs.ch/articles/05-servekit) from SGLang to vLLM: presharding, parallel staging into `/dev/shm`, and a logprob check. The measurements use servekit's own `profile` and `verify`.

## Limits

- A rented GPU has no Lustre and no capstor. Its network volume shows the mechanism and a smaller version of the problem, not Alps speedups.
- One node, tensor parallelism of at most 2, one 8B model.
- *Cold* means the page cache is evicted on this machine and every compile cache is empty. The storage server's own cache cannot be cleared from a client; `evict.py --verify` measures how much it could hide.

## Reproduce

On a full VM (not a container: Docker's default `/dev/shm` is 64 MB) with 2 GPUs, a network volume linked at `/net`, and local NVMe at `/local`. Paths and settings live in [config.env](config.env). Run the scripts inside `tmux`, so a dropped SSH connection does not end a matrix halfway; if one does end, its current server is stopped cleanly.

```bash
./setup_vm.sh                                  # vLLM 0.33.0 + extras, servekit, checkpoint, local copy, ENV.md
python evict.py --verify /net/models/apertus-8b   # eviction really makes reads cold on this storage
./tests/selftest.sh                            # the whole harness against a mock engine (no GPU time)
```

Day 1, the baseline:

```bash
./run_baseline.sh        # TP 1 and 2 × {cold, warm_weights, warm} × 3 → results/baseline.csv
./run_bench_startup.sh   # cross-check with `vllm bench startup` → results/bench_startup.csv
```

Day 2, the arms:

```bash
./run_preshard.sh        # vLLM's sharded-state saver at TP 2 → /net/models/apertus-8b-tp2
./run_servekit_prepare.sh  # the same through `servekit launch --only-prepare` → /net/models/apertus-8b-sk-tp2 + servekit.json
./run_staging_sweep.sh   # stage.py bandwidth for 1–32 readers → results/staging.csv; then set STAGE_READERS
./run_arms.sh            # every loader at TP 2, 3 cold repeats, servekit verify → results/arms.csv, verify.csv
                         # (includes servekit_serial and servekit_overlap: the patched servekit's own vLLM path)
python summarize.py --readme
```

## On Modal

No VM at hand: `../opportunities/03-servekit-vllm/modal_app.py` runs this harness in `vllm/vllm-openai:v0.33.0` on Modal, with a Modal Volume as the slow store and the container's disk as the control; runbook and budget in its `MODAL.md`.

## What each file does

| File | Purpose |
|---|---|
| [run_once.py](run_once.py) | One measured launch of vLLM or SGLang: evict or warm the weights, fresh or kept compile caches, optional staging, `servekit profile -- <engine> …`, wait for `/health`, optional `servekit verify` and a `--hold` command against the live server, clean shutdown, rows to CSV. Interface for other experiments: [HARNESS.md](HARNESS.md) |
| [evict.py](evict.py) | Drop files from the page cache (`posix_fadvise` DONTNEED), measure residency (`mincore`), prove it with `--verify` |
| [stage.py](stage.py) | Copy a checkpoint into `/dev/shm` with N reader processes over disjoint byte ranges (`pread`/`pwrite`), temp names renamed when complete; `--sweep` for bandwidth |
| [run_servekit_prepare.sh](run_servekit_prepare.sh) | servekit's own presharding for the `servekit_*` arms (`servekit launch --only-prepare -- vllm serve …`) |
| [preshard.py](preshard.py) | vLLM's save-sharded-state example (v0.33.0, Apache-2.0), plus a manifest |
| [detect_options.py](detect_options.py) | Which vLLM loading options and loader packages this install has |
| [summarize.py](summarize.py) | CSVs → `results/summary.md` and the block at the top of this README |
| [collect_env.py](collect_env.py) | Versions, GPUs, storage → [ENV.md](ENV.md) and `results/env.json` |
| `run_*.sh`, [config.env](config.env) | The experiments, one command each |
| [tests/](tests/) | A mock `vllm` and a one-minute self-test |

## Definitions

- **cold**: the checkpoint evicted from the page cache and every compile/JIT cache empty. That covers `VLLM_CACHE_ROOT`, `TRITON_CACHE_DIR`, `TORCHINDUCTOR_CACHE_DIR`, `FLASHINFER_WORKSPACE_BASE`, `TILELANG_CACHE_DIR`, `TVM_FFI_CACHE_DIR`, `DG_JIT_CACHE_DIR`, `XDG_CACHE_HOME` and `CUDA_CACHE_PATH`, each a fresh directory per run.
- **warm_weights**: checkpoint in the page cache, compile caches empty.
- **warm**: both kept, from an earlier unrecorded run.
- **Phases**: servekit's own table, where `weight_loading` is vLLM's "Model loading took … seconds", which vLLM 0.33 logs on local rank 0 only (`info_once`, scope "local"), so not necessarily the slowest rank. The servekit arms add `vllm:sk_load_max_rank`, every rank's own load time from servekit's vLLM plugin; `torch_compile`; `cuda_graph_capture`; and the gaps between them. `vllm:*` rows add what vLLM logs beyond that table, and `total` is launch to "Application startup complete". For `staged_shm`, the staging copy is timed and counted in both weight loading and total; for `servekit_serial`, servekit's own `stage` phase is. Under `servekit_overlap` the stage runs behind engine startup and only the time a rank waits for it counts.
- Every number is a median of at least 3 runs, shown with min and max.

All runs bind to 127.0.0.1, run with `HF_HUB_OFFLINE=1`, and log to `results/logs/`; `results/runs.jsonl` keeps every run's settings, including failures.
