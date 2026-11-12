# servekit: keep sglang 0.5.20's JIT caches, key them by environment, report what escapes

Draft for eth-easl/servekit. It's based on main `cd8f0b8`, as three commits in `patches/` (`git am patches/*.patch`). It does not conflict with #26: I merged it on top and all 196 tests pass.

Builds on #18 (Youssef's "persist JIT caches beside the checkpoint"). That PR's design is kept as it was:
- a read-only artifact;
- a per-job writable copy;
- the fixed node-local path that ninja and Triton need.

## Why

**1. sglang 0.5.20 moved its JIT cache, so the list servekit keeps no longer covers it.**
- In 0.5.16, `load_jit` built into `TVM_FFI_CACHE_DIR` (`python/sglang/jit_kernel/utils/compile.py:244`).
- In 0.5.20 it builds into `envs.SGLANG_JIT_CACHE_DIR.get() or "~/.cache/sglang/jit"` (`python/sglang/kernels/jit/utils/compile/cache.py:302`). That location does not follow `SGLANG_CACHE_DIR` (`srt/environ.py:1171`).
- The custom all-reduce is one of these kernels (`kernels/ops/communication/all_reduce.py:47`). It is built during `torch.distributed` init.
- In the 0.5.16 A/B (Llama-70B TP4, Bristen, one run per arm), that phase took 19.5 s without a cache and 2.7 s with one.
- On 0.5.20, servekit's three variables miss it, so it lands in the job's ephemeral `$HOME` and every launch rebuilds it. This is a **prediction; it has not been measured on a GPU yet** (see the Modal plan).

**2. The cache was keyed by artifact, not by what decides validity.**
- Five Llama-70B TP4 launches (Bristen jobs 80990/80991, Clariden 3184373/3184455/3184963) staged caches from `llama70b-tp4-sharded`. They still spent 18.98–21.61 s in distributed init.
- That artifact was most likely prepared with another image; its prepare log isn't in the repo.
- Nothing in the log said why the cache missed.

**3. A fixed list rots silently.** 0.5.20 also has:
- `SGLANG_CACHE_DIR/{cute_aot,deep_gemm}` (`srt/environ.py:1154,1168`);
- an `import sglang`-time `setdefault` of Triton/Inductor/CUDA/FlashInfer under `SGLANG_CACHE_DIR` (`srt/environ.py:1923-1946`, `__init__.py:11-14`);
- a `torch.utils.cpp_extension.load` build (`kernels/ops/attention/linear/kda_ptx_prefill/__init__.py:42`);
- two opt-in `load_inline` builds into hard-coded temp-dir paths (`pynccl_allocator.py:199`, `cuda_vmm_utils.py:450`).

## What

1. **`fix: keep sglang 0.5.20's JIT caches`**
   - `CACHES` lists every variable that moves a compile cache in 0.5.16 and 0.5.20 and their libraries:
     - Triton, TVM-FFI, FlashInfer;
     - `SGLANG_CACHE_DIR`, `SGLANG_JIT_CACHE_DIR`, `SGLANG_DG_CACHE_DIR`;
     - Inductor, `CUDA_CACHE_PATH`, `TORCH_EXTENSIONS_DIR`, TileLang, CuTe DSL.
   - All of them are set every time. A version that doesn't read one ignores it, and a version gate would be one more list to keep in step with sglang's cherry-picked release branches.
   - Tests:
     - A per-release table mirrors each cited source line and checks that every location lands under servekit's root.
     - `test_jit_cache_regression.py` reproduces the miss end to end: prepare, then a later job on another node. It **fails on cd8f0b8** (`results/regression-on-main-cd8f0b8.txt`).
2. **`feat: key JIT caches by environment, prune old ones, warm a new image's`**
   - The namespace is `<artifact>/servekit-caches/<engine>-<ver>-<sm>-<arch>-<sha12>`. It is keyed by:
     - package versions (from metadata, without importing sglang);
     - GPU compute capability (`nvidia-smi`, or libcuda in a child process);
     - CPU arch, Python, the CUDA toolkit, and the image tag when the image sets one.
   - Launch stages only its own key and logs what differs from the keys it skipped.
   - Prepare publishes by rename and never persists the node-specific P2P check. It prunes to `--jit-cache-keep` (3).
   - `--only-prepare` on an already prepared artifact warms the new key without re-sharding.
   - Pre-keying artifacts are still staged as before.
   - Cache variables the environment already sets are left alone.
3. **`feat: report JIT output the engine leaves outside the kept caches`**
   - The engine's default cache directories are snapshotted before it starts and checked again once it is ready.
   - Any new file outside the kept caches is reported with its size and the variable that would move it.
   - The report JSON gets a `jit_cache` section.
   - `--jit-cache-strict` makes drift a failure: exit 3, and the server is stopped (via `on_ready` returning True).

## What the log says (fake engine, `results/demo-log.clean.txt`)

```
[SERVEKIT] no JIT caches for this environment in .../llama70b-tp4; the engine will JIT from cold
[SERVEKIT]   not staged: sglang-0.5.16-sm90-aarch64-698ceae9bd08 (sglang 0.5.16 -> 0.5.20; apache-tvm-ffi 0.1.8 -> 0.1.11; ...)
[SERVEKIT] warning: the engine compiled into directories servekit does not keep:
[SERVEKIT]      4.5 MB in     3 files  ~/.cache/sglang/jit  (servekit set SGLANG_JIT_CACHE_DIR, and the engine wrote to its default anyway)
```

## Tests

- 51 new tests, no GPU. The full suite passes on Python 3.12 and 3.9: 187 passed.
  - The 2 stager tests error on macOS only (no `O_DIRECT`), as they do on main.
- The e2e prepare-serve job was updated for the keyed layout. **It has not been run on Clariden.**
  - It now also asserts that the serving launch staged the namespace that prepare kept.
- **The GPU measurement is still to come.** It is ready to run:
  - Setup: 2×H100, sglang 0.5.20, TP2.
  - Arms: no caches, main's caches, patched caches; 3 runs each, interleaved.
  - It reports dist init, the first prefill-capture step and the first request.
  - About $6.
  - I'll add the table to this PR once it has run.

🤖 Generated with [Claude Code](https://claude.com/claude-code)
