# #270: per-image JIT/compile caches for vllm_base and sglang_base, plus an audit that keeps them correct

## Problem

- **Startup time.** On `vllm_base`, DeepSeek-V4.1-Flash takes about 70 min to become ready, of which about 40 min is TileLang JIT. GLM-5.3 takes about 60 min, 8–10 min of it DeepGEMM warmup (model-launch #270, #265).
- **SGLang.** The lab's own A/B (Llama-70B TP4, 0.5.16, one run per arm) attributes about 54 s per launch to JIT caches. About 17 s of that is custom all-reduce JIT.
- **Nothing persists today.**
  - No env TOML sets a cache variable.
  - The container's `$HOME` is ephemeral: the EDFs don't mount `/users` (a strong inference from the CSCS CE docs).
  - vLLM puts Triton and Inductor under a compile-hash directory.
- **Checkpoint churn.** In the OpenTela trace, 68 of 93 deployed models lived 7 days or less, 43 of them Apertus checkpoints. A cache keyed per checkpoint, as servekit's is, is rebuilt for each of them.

## Related work, and what is new

**Prior work this builds on.**
- servekit #18 and Youssef Boughizane's write-up (with Xiaozhe):
  - three cache variables, keyed per artifact;
  - copied to a fixed `/tmp` path because ninja stores absolute paths;
  - measured −54 s on Llama-70B.
- The Cloudburst blog (fergusfinn.com/blog/fast-sglang-starts): 695 s → 88 s, but mostly from weights in the page cache.
- vLLM's compile cache (`VLLM_CACHE_ROOT`, AOT compile and mega artifacts by default since torch 2.10).
- PyTorch Mega-Cache (`torch.compiler.save/load_cache_artifacts`) and Inductor's Redis remote caches.
- Triton's `TRITON_CACHE_MANAGER` hook.
- FlashInfer's AOT / `flashinfer-jit-cache` / cubin wheels.
- The NVIDIA ComputeCache (`CUDA_CACHE_PATH`).
- DeepJIT's `rw:ro` cache list: a read-only layer inside one library.
- Content-addressed build caches (sccache, Bazel remote cache): immutable entries, atomic publish.

**New here.**
1. A **per-image × per-GPU-arch** cache, shared across checkpoints.
   - The key comes from a CI digest sidecar if present, else the 96-byte squashfs superblock (images are updated in place).
   - A fixed-path node-local overlay is promoted with **staging + rename/link**: first writer wins, no partial entries, and no cross-node locks (capstor's flock mode is unknown).
2. **`cache-audit`**:
   - runs a launch cold;
   - records every write unprivileged (strace, inotify, snapshots, and a PEP 578 hook that names the writing `file:line`);
   - maps each root to its variable from a knowledge base and verifies the mapping by re-running;
   - tests relocation and read-only use;
   - emits a table, an env snippet and a CI exit code.
   A GPU-free static mode catches renamed or new cache variables.
3. A source-verified key, write-safety and relocatability analysis for both engines: **1,137 file:line citations**, every quote checked by a test against the exact versions shipped.

## What I found (each item cited; ✓ = also measured locally)

- **SGLang 0.5.20.**
  - `import sglang` redirects Triton, Inductor, ComputeCache and FlashInfer under `SGLANG_CACHE_DIR`.
  - SGLang's own JIT (about 119 modules, incl. custom all-reduce v2) lives in `SGLANG_JIT_CACHE_DIR`, which does **not** follow it.
  - servekit's three variables therefore miss the 17 s all-reduce part on 0.5.20, and `TVM_FFI_CACHE_DIR` is dead for SGLang's kernels.
- **vLLM 0.32.0** (the probable `vllm_base` version; inferred from release dates and pins).
  - The torch.compile hash includes the **checkpoint path**, and vLLM forces Inductor into that directory.
  - Only Triton (if `TRITON_CACHE_DIR` is set), TileLang, DeepGEMM and FlashInfer are shared across checkpoints.
  - Nothing sets `TILELANG_CACHE_DIR`.
- **Relocation and read-only use.**
  - Triton group files store absolute paths: a moved copy misses, and a copy silently reads the original ✓ (mock).
  - Inductor relocates, but crashes on a read-only copy (lock files) ✓.
  - tvm-ffi rebuilds after a move ✓ and **hangs forever** on a read-only copy (549 lock retries in 60 s) ✓.
  - TileLang relocates; its hits work read-only ✓. Its Cython wrapper build leaves a `tmp*.cpp` + `tmp*.so` per kernel in `$TMPDIR` ✓. The audit reports these as leaked temporaries without gating them.
  - Inductor's precompiled headers ignore its variable (31 MB in `$TMPDIR`) ✓.
- **Hazards for a shared read-write cache.**
  - umask 022 directories lock out other infra01 users.
  - SGLang's autotune files are written 0600.
  - The P2P JSONs are written in place without a lock.
  - Inductor's temp names (`.<pid>.<tid>.tmp`) can collide across nodes.

  Hence the overlay design instead of a shared read-write directory.

## Results so far

| | measured (where) | result |
|---|---|---|
| Audit on real libraries | CPU, macOS arm64; torch 2.13.0, tvm-ffi 0.1.11, TileLang 0.1.12 | all three gates PASS; mappings verified by re-run; warm runs rebuilt 0 binaries; relocation and read-only behaviour as listed above (`results/audit-local/`) |
| Cold vs warm, whole-process wall s | same; median [min–max] of 3 | Inductor 19.2 [18.2–19.3] → 14.9 [14.7–16.6] (15 → 0 binaries); tvm-ffi 5.3 [5.3–5.7] → 2.4 [2.3–2.5] (3 → 0); TileLang 6.2 [5.7–7.4] → 3.2 [3.1–3.7] (3 → 0). These are CPU compiles, not GPU kernel times. |
| Write safety | 8 concurrent writers × 3 repeats into one dir | Inductor, tvm-ffi, TileLang: 0 writer failures, 0 load failures, 0 rebuilds, 0 leftovers (APFS, not Lustre) |
| Promotion protocol | 16 concurrent promoters; 8 jobs with disjoint kernels | each entry published exactly once; all 80 disjoint entries landed; 0 errors |
| model-launch patch | `pytest tests/unit`, ruff, mypy, shellcheck | 482 passed; the bash key is byte-identical to the helper's; a missing `nvidia-smi` does not abort master.sh |
| Saving on GPU | **predicted, not measured** | SGLang: about 54 s per 70B TP4 launch (the lab's A/B), about 17 s of which servekit misses on 0.5.20. vLLM: the TileLang and DeepGEMM warmups become loads after the first launch of a digest. |

## How to run

- **Locally.** Run `pytest`: 55 tests, 53 pass locally in about 4 min including the real-library audits, and 2 are Linux-only. Then:
  ```bash
  python -m cache_audit run --relocate --readonly -- <warmup cmd>
  python -m cache_audit static
  python modal_app.py --dry-run
  ```
- **On Modal.** `modal run modal_app.py::{smoke,plan_a,plan_b,plan_c}`, about $22 across three accounts. It covers:
  - cold vs servekit's three variables vs all variables, with servekit-profile phases and builds avoided;
  - GPU audits of both images, with a gate of the patched TOMLs;
  - a GPU stress test and a TileLang CUDA cold/warm benchmark.

  See MODAL.md.

## What it takes on Alps, and the risks

- **Admin setup.** Create the infra01 root once (setgid, group ACL) and run `gc` from a scrontab.
- **Images CI** (optional, best): write `<image>.sqsh.digest` and pre-warm the flagships after each release (`drafts/images-ci-prewarm.md`).
- **Validation.** One Clariden A/B: two launches of DeepSeek-V4.1-Flash and of an SGLang 70B on the same digest.
- **Risks.**
  - capstor's rename atomicity across MDTs (Lustre DNE) and `/tmp` (tmpfs) size are unverified.
  - The SGLang JIT key includes `nvcc --version`, so a toolchain change in the image is a new digest anyway.
  - The vllm_base version is inferred.
  - FA4/CuTe on GH200 is aarch64-specific JIT that Modal's x86 H100 cannot exercise.

## Pitch (for Xiaozhe / Ana)

> #270 for both engines, as a mergeable model-launch patch. It keeps one cache per image and GPU arch, shared across checkpoints, with no cross-node locks.
> It also ships a tracing audit that verified every cache variable against source. It finds that on SGLang 0.5.20 servekit's caches miss the 17 s all-reduce JIT, and that vLLM keys compile caches per checkpoint path.
> Next: one Clariden A/B, CI pre-warming, then traffic-aware lazy graph capture as the research step.
