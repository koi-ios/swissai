# 02-servekit-cache-fix: hand-over notes (12 Nov 2026)

The deliverables are in `/Users/koiios/Documents/Projects/swissai/opportunities/02-servekit-cache-fix/`:
- `patches/` (3 commits on servekit cd8f0b8);
- `drafts/PR.md`, `WRITEUP.md`, `STATUS.md`;
- `modal_app.py` and `MODAL.md` (not run: no Modal account here).

## Findings that matter beyond this folder (for #270 / cache-270)

- **sglang 0.5.20 compile locations** (file:line in WRITEUP.md and in `vendor/servekit/tests/test_jit_cache.py::WHERE`):
  - `SGLANG_JIT_CACHE_DIR` (load_jit, custom all-reduce included). It does not follow `SGLANG_CACHE_DIR`.
  - `SGLANG_CACHE_DIR`: cute_aot, deep_gemm, torch_compile_cache, FlashInfer autotune, the GPU P2P check.
  - `TORCH_EXTENSIONS_DIR`: the KDA prefill kernel calls `cpp_extension.load`.
  - TileLang and CuTe DSL (`CUTE_DSL_CACHE_DIR`, default `/tmp/$USER/cutlass_python_cache`).
  - Two opt-in builds sit under hard-coded `tempfile.gettempdir()` paths (`symm_allocator` with --enable-symm-mem; `sgl_bump_arena` with `SGLANG_ENABLE_GRAPH_POOL_BORROW`). No variable moves them; only TMPDIR or an audit can catch them.
- **Never share the GPU P2P check**: `$SGLANG_CACHE_DIR/gpu_p2p_access_cache_for_<CUDA_VISIBLE_DEVICES>.json` describes node topology. Copied to a node without NVLink, it could enable custom all-reduce wrongly. servekit's persist excludes it, and so should a shared cache.
- **Use a fixed node-local path.** Triton group files store absolute paths (per the vLLM-prelim session's notes, Triton 3.7.1 `cache.py:73-100`), and so do FlashInfer's ninja files. sglang 0.5.20's own load_jit cache is relocatable (anchor-normalized).
- **`SGLANG_CRASH_ON_JIT_COMPILE=1`** (environ.py:1181) makes sglang's own load_jit fail on a miss. It is a cheap hit check for a pre-seeded per-image cache.
- **Variables the environment already sets win.** The patched servekit leaves any cache variable that is already set untouched, so a platform-level shared cache (#270) composes with servekit.
- **Lab A/B re-read from the logs**:
  - First prefill-capture step: 28.49 s without cache vs 5.69 s with cache (tqdm rate at 1/N).
  - Dist init: 19.54 s vs 2.74 s.
  - Parser: `tools/runlog.py`.
