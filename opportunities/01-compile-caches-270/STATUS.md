# STATUS: 01-compile-caches-270 (model-launch #270 for vllm_base and sglang_base)

12 Nov 2026. Nothing has been pushed, posted or run on Modal.

## Done

- **`cache_audit/`**: the audit tool. Stdlib only; runs inside the images.
  - Fresh HOME/TMPDIR sandbox.
  - Four unprivileged tracers: strace (`-f -y`, cwd tracking), inotify (ctypes, recursive), snapshots, and a PEP 578 audit hook that attributes writes to `file:line`.
  - Clusters files into roots and maps them through the KB. Changes under shared `/tmp` that no tracer attributes to the run are reported as noise, not gated.
  - Five runs: cold, redirect (verifies the mapping), warm (builds avoided), relocate (hides the original: a copied Triton cache would otherwise silently read it), read-only.
  - Classifies contents and detects baked-in absolute paths.
  - Reports randomly named temp files as leaked temporaries, not caches.
  - Outputs: markdown, JSON, env snippet, and a CI exit code (0, 2, 10 unredirected, 11 unknown, 12 drift). `static` mode detects drift without a GPU.
- **Tests**: 55 in `tests/`. 53 pass locally in 3 min 52 s, including the real-library audits; 2 are Linux-only (inotify, live strace).
- **`kb/caches.toml`**: 29 cache entries. Evidence comes from 1,137 citations:
  - `kb/engines.citations.json`, mechanical via `tools/make_citations.py`;
  - `research/*.citations.json`, from five source reviews in `research/*.md`.
  - `tests/test_kb.py` checks every quote at its exact line in `vendor/` (the shipped versions).
- **Measured locally** (`results/`; CPU, macOS arm64; real torch 2.13.0, tvm-ffi 0.1.11, TileLang 0.1.12):
  - `audit-local/{torch_compile_cpu,tvm_ffi_jit,tilelang_cpu}/`: all PASS, mappings verified by re-run, warm runs rebuilt 0 binaries.
    - Inductor: relocatable; a read-only copy crashes on lock files.
    - tvm-ffi: a moved copy rewrote 15 files; a read-only copy hangs (1,107 lock retries in 120 s).
    - TileLang: relocatable, and hits work read-only.
    - Inductor's precompiled headers ignore its variable.
  - `local_cpu_ab.md` (median [min–max] of 3, whole-process wall s): Inductor 19.2 → 14.9, tvm-ffi 5.3 → 2.4, TileLang 6.2 → 3.2. 0 binaries rebuilt warm.
  - `stress_local.json`: 8 concurrent writers × 3 repeats per library; 0 writer failures, 0 load failures, 0 rebuilds, 0 leftovers (APFS, not Lustre).
  - `ab_summary_MOCK_format_only.md` shows the report format only.
- **`integration/sml_jit_cache.py`**:
  - key: digest sidecar or squashfs superblock;
  - arch, seed, promote (staging + rename/link, first writer wins, 60 s settle window, retry passes), gc, du.
  - Tests: 16 concurrent promoters, 8 jobs with disjoint kernels, settle window, GC.
- **model-launch patches** (`patches/0001-0004`; branch `feat/shared-jit-caches` in `vendor/model-launch-jit-cache`):
  - launcher (`{jit_cache}` like `{arch}`, key and arch on the batch host, embedded helper, seed and promote hooks, `--jit-cache overlay|shared|off`, `--jit-cache-root`);
  - env TOMLs for vllm_base and sglang_base;
  - docs;
  - a fix so a missing `nvidia-smi` cannot abort master.sh.
  - 482 unit tests pass; ruff, format, mypy and shellcheck are clean.
  - `patches/after-272/` replays the env commit onto PR #272's `*_nvidia.toml`.
  - The author is a placeholder: run `git am`, then `git commit --amend --reset-author`.
- **`modal_app.py` + `MODAL.md`**: smoke, plan_a (SGLang), plan_b (vLLM), plan_c (image audits + gate of the patched TOMLs, GPU stress, TileLang CUDA benchmark). Estimated about $22.
  - `python modal_app.py --dry-run` passes; the module imports with Modal (8 functions, 4 entrypoints).
  - `harness/ab.py` passed end to end against a faithful mock vLLM, using servekit's own profiler.
- **Drafts**: `drafts/270-comment.md`, `drafts/env-toml-diffs.md`, `drafts/images-ci-prewarm.md`.
- **WRITEUP.md**, README.md, and a findings note in the shared notes folder.

## Not done or blocked

- **GPU numbers.** None measured: the GPU savings are predictions until the Modal plans run. A Clariden A/B needs CSCS access.
- **vllm_base's vLLM version** is inferred as 0.32.0 (the image repo is private). Check with `pip show vllm`.
- **CuTe DSL 4.6.2 / 4.8.0** cache code ships compiled; it was verified on the 4.5.0 source as a proxy. HF `kernels` and cuda-tile internals are not vendored.
- **Lustre behaviour is untested:** flock mode, rename across MDTs, and seed time for many small files on capstor.
- **Not exercised locally:** the inotify and live-strace tracer tests are Linux-only and run in Modal's `selftest`.

## Next 3 actions

1. Run `modal run modal_app.py::smoke`, then plans A, B and C on three accounts. Put `results/ab_summary.md` into WRITEUP.md and `drafts/270-comment.md`, replacing the "predicted" lines.
2. Send the #270 comment, then the patches as a PR from a fork once Aryan answers the three questions: root permissions and flock, digest sidecar, default mode.
3. With CSCS access, do one Clariden A/B: DeepSeek-V4.1-Flash and an SGLang 70B, two launches on the same digest. Run `cache-audit run --env-toml` inside `vllm_base` / `sglang_base` to confirm the KB on the real images.
