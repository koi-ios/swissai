# What an SGLang launch does before `torch.distributed` init, and how to cut it

*Opportunity 05, 12 Nov 2026. Everything **measured** comes from the lab's own logs or from local runs, and is labelled with its setup. **Predicted** means not yet measured: the Modal runs in MODAL.md, about $3.3, settle it.*

## 1. The problem, from the lab's own logs

- After servekit, "process start + TP worker spawn" is 25–30 s on Clariden, 21–29% of the cold start (EVALUATION.md). Nobody has looked at it.
- I re-timed 59 single-node launches from Youssef Boughizane's repository. Three clocks are lined up: servekit's sub-second Popen time, SGLang's log stamps, and where the compile-time `SyntaxWarning` appears (`analysis/log_timeline.py`, `results/log_timeline.md`).
- A parallel worker session reproduced the medians independently (`handoff-notes/python-startup/`).

| launch (median [min–max]) | main process: Popen → `server_args=` | TP workers → `Init torch distributed begin` |
|---|---|---|
| Clariden servekit-eval, torch already imported by a wrapper (n=18) | 11.3 s [9.9–60.8] | 15.0 s [14–16] |
| Bristen JIT A/B, nothing pre-imported (n=2) | **56.8 s** [49.1–64.4] | 14.5 s [14–15] |
| Bristen v0.5.10, nothing pre-imported (n=18) | 29.9 s [22.0–37.3] | 11.0 s [10–12] |

The phase has two parts with different causes.

1. **Workers: 11–16 s on every launch.**
   - SGLang 0.5.20 forces `spawn`.
   - Each TP scheduler and the detokenizer is a fresh interpreter. Each re-imports torch, transformers, triton and SGLang, which the main process already imported: `http_server.py` imports the scheduler module tree at module level.
2. **Main process: 11 s warm, 28–64 s when it is the first process on the node to touch the image.**
   - Youssef's servekit-eval runs put a `sitecustomize.py` on `PYTHONPATH` that imports SGLang in *every* Python process. So servekit's wrappers warmed the page cache and wrote the bytecode *before* SGLang's clock started.
   - The published 25–30 s is therefore a **lower bound** for a production launch.
   - servekit's own "time to first output line" is 0.4 s on Clariden with that pre-import, and 34–47 s on Bristen without it.

**Bytecode** (verified from logs and Dockerfiles):

- 44 of 72 logs show a compile-time `SyntaxWarning` for `torchao/quant_api.py`.
- It comes only from the first process(es) in each container, which shows three things:
  - the v0.5.16 image ships no valid `.pyc`;
  - the container root is writable (CSCS docs: `writable` defaults to true, and all tasks on a node share one container);
  - the compile happens once per job per node, in the critical path.
- Images, as their public Dockerfiles show:
  - upstream `lmsysorg/sglang:v0.5.20` deletes every `__pycache__`, and SGLang itself is an editable install;
  - **all of model-launch's vLLM images use `uv pip install`, which never compiles**;
  - model-launch's SGLang image uses pip, so it has pycs;
  - upstream `vllm/vllm-openai` sets `UV_COMPILE_BYTECODE=1`.

**vLLM:**

- vLLM forks its engine core and workers by default (fork unless CUDA is already initialised), so it avoids SGLang's full re-import per worker.
- It still pays post-fork lazy imports in each worker. On the platform's uv-built images every process compiles from source.
- servekit's NGC fixture: 14.8 s to vLLM's first line.

## 2. Related work, and what is new

- **Zygotes and import caching**: SOCK (ATC '18). **Snapshots**: Catalyzer (ASPLOS '20). **Working-set prefetch**: REAP (ASPLOS '21) and FaaSnap (EuroSys '22).
- **Prioritized image prefetch**: eStargz and Nydus.
- **Lazy imports**: Cinder / Instagram, PEP 690 (withdrawn), and PEP 810 (accepted Nov 2025 for Python 3.15; the engine images run 3.12).
- **Hash-based pycs**: PEP 552. **`multiprocessing.set_forkserver_preload`**; Python 3.14 makes forkserver the Linux default.
- **Fork and CUDA**: PyTorch's poison-fork note (`PYTORCH_NVML_BASED_CUDA_CHECK`). vLLM's fork-unless-CUDA-is-initialised rule.
- **CSCS** (CUG '25): a 22k-inode PyTorch venv squashed into SquashFS starts "up to 5×" faster.

**New here:**

1. The diagnosis above, from the lab's data: the measured phase is a lower bound, with a cold main-process cost an order of magnitude larger than its warm cost.
2. A **CUDA-safe zygote for GPU serving engines**. A plain forkserver preload is unsafe for SGLang:
   - 95 SGLang modules call `torch.cuda.is_available()` at import (`cuInit`);
   - `sgl_kernel` calls `current_device()` and `get_device_properties()` to pick its sm90 or sm100 build.

   The zygote guard handles this:
   - it serves exactly those facts, from an NVML probe run *in another process*;
   - any other route into the driver raises an exception that `except Exception` cannot swallow;
   - before forking it proves that torch is uninitialised, no `/dev/nvidia*` descriptor is open and only one thread runs, otherwise the launch falls back to spawn;
   - it gives forkserver children spawn's environment semantics;
   - each worker verifies the served facts after its own CUDA init.
3. A bytecode audit of the platform's images, with build-step patches.

## 3. What was built (`src/pystartup`, stdlib only, 41 tests passing locally)

- **Profiler** (`python -m pystartup.profile_startup`). It runs on the real images in CPU containers, with GPU import paths emulated by served facts and the CUDA stub library. Per role it records:
  - import time split into path search, reads, unmarshal, compile, extension loading and exec;
  - modules and files compiled, files and bytes read, mapped `.so` files;
  - pyc validity of the image and the import set;
  - spawn vs zygote vs fork;
  - process-tree tracing via `sitecustomize`.
- **(a) Image-build step**: `compileall --invalidation-mode unchecked-hash`, which survives mtime changes on the way to the `.sqsh`.
  - `patches/model-launch-python-startup.patch`: 6 Dockerfiles, plus `SML_SGLANG_ENTRYPOINT`.
  - `patches/sglang-v0.5.20-dockerfile-bytecode.patch`.
- **(b) Zygote**:
  - `python -m pystartup.sglang_launch -- <sglang args>` needs no SGLang change. It falls back to spawn for `--enable-memory-saver`, Ray, DP and numactl-bound workers.
  - Upstream-style opt-in version: `patches/sglang-v0.5.20-worker-zygote.patch` (`SGLANG_WORKER_ZYGOTE=1`; existing files change by 5–9 lines each).
  - Calibration keeps only modules that import cleanly; a calibrated list is replayed strictly.
- **(c) `PYTHONPYCACHEPREFIX`** on node-local disk: `--pycache-prefix` and the profiler's prefix modes.
- **(d) Prewarm**: a first-access-ordered manifest of the import working set, including every mapped `.so`, read by N threads from t=0. Includes `mincore` and `fadvise` tools.

## 4. Results so far

**Measured locally** (macOS M-series, CPython 3.12; torch 2.9.1 and transformers stand in for the engine):

| measurement | result |
|---|---|
| time to 4 workers ready (3 runs) | spawn **8.48 s** [7.58–11.10] · zygote **4.19 s** [3.83–4.31] · fork 4.15 s [3.77–4.45] |
| zygote workers | ready 0.024 s after launch; the preload fully overlapped the parent's own 4.2 s import |
| import, 1,291 modules / 22 MB of source | warm pycs 0.97 s; no pycs, read-only **2.13 s** (1.2 s compiling 1,189 files); prefix first process 2.44 s, later processes 0.94 s |
| after `uv pip install` | 1,003 of those modules had no pyc |
| SGLang 0.5.20 source | 3,356 files, 50 MB; `sglang.srt` alone compiles in 1.96 s on one core |
| shipping bytecode, whole venv (166 MB of source) | 5.7 s on 14 cores; +1.2× the source size |

**Predicted for Alps (to be measured):**

- **Zygote:** the worker phase drops from 11–16 s to about 2–5 s (unpickling, tokenizer, CUDA context), so about 9–13 s per launch is saved. The saved amount depends on how much of the scheduler's import set calibrates as preloadable; `cpu_study` and `gpu_e2e` measure it.
- **Bytecode:** on uv-built vLLM and upstream SGLang images, a few seconds per launch (first process), and multiples of that on a read-only root.
- **Prewarm:** the cold main-process gap (34–47 s to the first output line on Bristen) is the ceiling. capstor reads at 362 MB/s with one reader and 6.7 GB/s with 64 (the lab's job 81985). Whether the gap is read latency is the open question. Modal cannot reproduce Lustre or squashfs, so the decisive test is on Alps.

## 5. How to run it

- **Locally:**
  - `.venv/bin/python -m pytest -q tests`
  - `python -m pystartup.spawnbench --targets torch,transformers --workers 4`
  - `python -m pystartup.profile_startup imports --engine proxy`
- **On Modal:** `python modal_app.py --dry-run`, then `smoke`, `cpu_study --engine sglang|vllm`, `prewarm_study`, and `gpu_e2e` (TP2, which proves the invariants with real NVML facts and checks that greedy outputs are identical). See MODAL.md.

## 6. On Alps, and the risks

**Steps:**

1. Run `python -m pystartup.pycheck <site-packages>` inside `sglang_base` and `vllm_base` on Clariden. It confirms the bytecode state of the private images in a minute.
2. Run `python -m pystartup.e2e --tp 4` on one node, with the wrapper put on `PYTHONPATH` from capstor through `--pre-launch-cmds`.
3. Run the prewarm A/B on a node that hasn't touched the image.

**Risks:**

- **Preloaded modules that read the environment at import** see the zygote's start-time environment. Mitigations: SGLang's unconditional env defaults are applied first, `prometheus_client` is denied, and child environments are synced. Calibration plus output-equality runs are the check.
- **Untested combinations:** DP controllers, Ray and memory saver are excluded.
- **NUMA-bound workers** are spawned as before.
- **On GH200, check whether SGLang's auto NUMA binding triggers** (it needs `numactl` in the image). If it does, those workers keep spawning.

## 7. Pitch

> After servekit, an SGLang launch still spends 25–30 s before distributed init. In your own logs that is a lower bound: without the experiment's pre-import it is 33–79 s (medians 39–71 s), and the parts are cold image reads, missing bytecode in the uv-built vLLM and upstream SGLang images, and spawn re-importing everything per TP worker.
> I built a profiler for the real images and a CUDA-safe worker zygote that needs no SGLang change: locally it cut worker start from 4.3 s to 0.02 s, with proofs that CUDA is never initialised before fork. I also wrote one-line image fixes for the bytecode.
> A $3 Modal run and one Clariden job would turn the predicted 9–13 s per launch into a measured number for #270's SGLang half.
