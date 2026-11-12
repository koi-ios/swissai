# servekit's JIT cache on SGLang 0.5.20: the fix, keys that mean something, and a guard against the next move

## The problem, with the lab's own numbers

servekit has kept the engine's JIT caches beside each checkpoint since #18 (Youssef Boughizane). His A/B ran on Bristen with Llama-3.1-70B, TP4, SGLang 0.5.16, one run per arm. Cold start went from 217.6 s to 146.2 s:

| phase | no cache | servekit cache |
|---|---|---|
| prefill capture | 73.4 s | 43.3 s |
| first prefill-capture step* | 28.49 s | 5.69 s |
| torch.distributed init | 19.54 s | 2.74 s |
| first request | 8.9 s | 1.5 s |

\* Read from the tqdm rate in the same logs by `tools/runlog.py`.

The dist-init drop is SGLang's custom all-reduce v2. It is JIT-built through `load_jit` (`kernels/ops/communication/all_reduce.py:47`) between "sglang is using nccl" and "All Reduce config".

Two things have broken since:

1. **SGLang 0.5.20, the platform's version, moved that cache.**
   - 0.5.16 built `load_jit` kernels in `TVM_FFI_CACHE_DIR` (`jit_kernel/utils/compile.py:244`).
   - 0.5.20 builds them in `SGLANG_JIT_CACHE_DIR or ~/.cache/sglang/jit` (`kernels/jit/utils/compile/cache.py:302`), which does not follow `SGLANG_CACHE_DIR` (`srt/environ.py:1171`).
   - servekit sets only `TRITON_CACHE_DIR`, `TVM_FFI_CACHE_DIR` and `FLASHINFER_WORKSPACE_BASE` (`jit_cache.py:16-22` at cd8f0b8). On 0.5.20 those kernels therefore land in the job's ephemeral `$HOME`, and every launch rebuilds them.
   - 0.5.20 also puts CuTe-DSL AOT objects, DeepGEMM and torch.compile output under `SGLANG_CACHE_DIR` (`environ.py:1154,1168`, `compilation/backend.py:403`), which servekit does not keep either.
   - Predicted cost: ~17 s per TP4 launch. **Not measured on 0.5.20 yet.**
2. **The cache is keyed by checkpoint, not by what makes a compiled kernel valid.**
   - Five Llama-70B TP4 launches (2 on Bristen, 3 on Clariden) staged caches from `llama70b-tp4-sharded` and still took 18.98–21.61 s in dist init.
   - That artifact was most likely prepared with another image; its prepare log is not in the repo.
   - Nothing in the log said why the cache missed.

## What exists, and what is new here

**What exists:**
- servekit #18 persists caches per artifact.
- SGLang 0.5.20 has its own content-addressed `load_jit` cache, with a fail-on-miss flag (`SGLANG_CRASH_ON_JIT_COMPILE`, `environ.py:1181`).
- `import sglang` redirects Triton, Inductor, CUDA and FlashInfer under `SGLANG_CACHE_DIR` with `setdefault` (`environ.py:1923-1946`, `__init__.py:11-14`).
- vLLM 0.32 has a JIT monitor that hooks compile events.
- #270 (Aryan Ahadinia) asks for per-image shared caches on the platform; the sibling `cache-270` work covers that side.

**New here:**
- the 0.5.20 regression, with a test that reproduces it and fails on main;
- validity keys that are cheap to compute inside the job, and that explain a miss in the log;
- an engine-agnostic audit that names every directory a run compiled into but servekit doesn't keep. Reading 0.5.20 for it turned up:
  - a `torch.utils.cpp_extension.load` build in `~/.cache/torch_extensions` (`kda_ptx_prefill/__init__.py:42`);
  - two opt-in builds under hard-coded temp-dir paths (`pynccl_allocator.py:199`, `cuda_vmm_utils.py:450`). No variable moves these, so only an audit can catch them.

## What was built

Three commits on cd8f0b8, in `patches/`. They apply cleanly to a fresh clone, merge cleanly with open PR #26, and pass 187 tests on Python 3.12 and 3.9.

1. **`fix: keep sglang 0.5.20's JIT caches`.**
   - `CACHES` now lists all 11 variables that move a compile cache in 0.5.16 and 0.5.20 and their libraries (TileLang, CuTe DSL, DeepGEMM, Inductor, the CUDA PTX cache, torch extensions).
   - All of them are set every time. Setting one that a version ignores is harmless, and a version gate would be one more list to rot (sglang cherry-picks across release branches).
   - A per-release table in the tests mirrors each cited source line.
   - `test_jit_cache_regression.py` fails on cd8f0b8 and passes with the patch (`results/regression-on-main-cd8f0b8.txt`).
2. **`feat: key JIT caches by environment, prune old ones, warm a new image's`.**
   - The key is `sglang-0.5.20-sm90-aarch64-<sha12>`. It covers:
     - package versions, read from metadata without importing sglang;
     - GPU compute capability (`nvidia-smi`, else libcuda in a child process);
     - CPU arch, Python version, the CUDA toolkit, and the image tag if the image sets one.
   - Launch stages only its own key, still to the fixed node-local path that ninja and Triton need (Triton's group files store absolute paths). It logs what separates the keys it skipped, for example: `not staged: sglang-0.5.16-… (sglang 0.5.16 -> 0.5.20; …)`.
   - Prepare publishes the namespace by rename and never persists the node-specific GPU P2P check. It keeps 3 namespaces.
   - `--only-prepare` on an already prepared artifact warms a new image's caches without re-sharding.
   - Artifacts written before keys still stage as before. Cache variables set by the environment (#270-style shared caches) are left alone.
3. **`feat: report JIT output the engine leaves outside the kept caches`.**
   - The engine's default cache directories are snapshotted before launch and diffed once it is ready. The diff gives size, directory, and either the variable that would move it or "unknown: add it".
   - The report JSON gets a `jit_cache` section.
   - `--jit-cache-strict` makes drift a failure: exit 3, and the server is stopped.

`results/demo-log.clean.txt` shows all three, with a fake engine.

## Results so far

| | status |
|---|---|
| 0.5.20 moved `load_jit`'s cache; servekit misses it | **Verified** in source (file:line above) and by the regression test |
| Lab A/B parsed again: dist init 19.54 → 2.74 s, first prefill step 28.49 → 5.69 s | **Measured** by the lab; re-read from their logs |
| On 0.5.20, main's launches pay the all-reduce JIT again; the patch saves it | **Predicted.** The Modal A/B below measures it |

## How to run

- **Locally:**
  - `PYTHONPATH=vendor/servekit/src python -m pytest vendor/servekit/tests -q` runs servekit's tests.
  - `.venv/bin/python -m pytest tests -q` runs the tooling tests.
  - `python modal_app.py --dry-run` prints the plan.
- **On Modal** (`MODAL.md`):
  - 2×H100 SXM, SGLang 0.5.20, Qwen2.5-1.5B at TP2.
  - Three arms: no caches, servekit-main caches, patched caches.
  - 3 runs each, interleaved; every run is a fresh job.
  - Per-phase table: dist init, first prefill-capture step, first request.
  - About 40 min and ~$6; the hard stop caps it at $7.90.

## What it would take on Alps, and the risks

**What it would take:**
- On Clariden, `tests/e2e/test_prepare_serve_e2e.py` (now asserting that the keyed namespace is staged), about one node-hour.
- Then a 3+3 Llama-70B TP4 A/B on the 0.5.20 image, which also settles the "other image" explanation for the five slow launches.
- model-launch doesn't call servekit today. The patch composes with #270: platform-set cache variables win.

**Risks:**
- **The key's precision.** An unrelated package bump means an extra cold start. Too loose a key is caught by each library's own hash.
- **`nvidia-smi` might not be present** in the platform's containers. There is a libcuda fallback, and the key degrades to `nogpu`.
- **False positives in strict mode.** The temp dir counts only known caches and compiled files.
- **Lustre rename semantics** for publishing a namespace (a same-directory rename) should be confirmed on capstor.

## Pitch (for Xiaozhe or Ana)

servekit's JIT cache silently stopped covering SGLang 0.5.20: `load_jit` moved to `SGLANG_JIT_CACHE_DIR`, so every TP4 launch rebuilds the custom all-reduce. That cost ~17 s in your own 0.5.16 A/B.

Three small patches fix it, key caches by what actually makes them valid, and turn the next cache move into a log line instead of a slowdown.

A regression test fails on main and passes with the fix. A $6 Modal A/B is ready to confirm the effect on GPUs.
