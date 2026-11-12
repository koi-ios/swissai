# 01-compile-caches-270: cross-cutting findings (12 Nov 2026)

From the #270 session (folder `opportunities/01-compile-caches-270/`). Each item below has a file:line citation in that
folder: `kb/*.citations.json` and `research/*.citations.json`, 1,137 citations, all mechanically checked against the
vendored sources.

- **vllm_base is most likely vLLM 0.32.0** (inference):
  - v0.32.0 was tagged 2026-10-17 and the image was promoted 2026-10-18 (#265, #272);
  - KISTI pins v0.32.0;
  - pins: torch 2.13.0, Triton 3.7.1, FlashInfer 0.7.0.post1, TileLang 0.1.12, tvm-ffi 0.1.11.
- **SGLang 0.5.20 redirects caches at import.**
  - `import sglang` setdefaults `TRITON_CACHE_DIR`, `TORCHINDUCTOR_CACHE_DIR`, `CUDA_CACHE_PATH` and
    `FLASHINFER_WORKSPACE_BASE` under `SGLANG_CACHE_DIR` (`python/sglang/__init__.py:14`, `srt/environ.py:1923-1947`).
  - So, without servekit, Triton and FlashInfer are no longer "where they were": they live under `~/.cache/sglang`.
  - `SGLANG_JIT_CACHE_DIR` (`~/.cache/sglang/jit`) does NOT follow `SGLANG_CACHE_DIR`.
  - `TVM_FFI_CACHE_DIR` is dead for SGLang 0.5.20's own kernels (relevant to servekit-cache-fix).
- **vLLM's torch.compile and Inductor cache is per checkpoint PATH.**
  - `ModelConfig.compute_hash` ignores `served_model_name` but not `model`.
  - vLLM overwrites `TORCHINDUCTOR_CACHE_DIR` per compile hash.
  - Triton is shared across checkpoints only if `TRITON_CACHE_DIR` is set explicitly (relevant to servekit-vllm).
- **vLLM forces `TORCHINDUCTOR_COMPILE_THREADS=1` at import** (`env_override.py`), so Inductor compiles serially.
- **Triton's disk cache is not relocatable.**
  - Group files store absolute child paths, so a moved cache misses.
  - A *copied* cache silently reads the original while the original still exists.
  - Keep the cache path fixed (as servekit does).
- **tvm-ffi 0.1.11 hangs forever on a read-only cache.** Its lock retries on any OSError: measured 549 attempts in 60 s.
- **Inductor's precompiled headers ignore `TORCHINDUCTOR_CACHE_DIR`.** They always go to `$TMPDIR`; measured 31 MB.
- **Nothing sets `TILELANG_CACHE_DIR`.** `import tilelang` creates empty `~/.tilelang/cache/...` dirs, which explains
  #270's "~/.tilelang exists but is empty". TileLang kernels are not compiled per batch size (`num_tokens` is
  `T.dynamic`).
- **Local stress, 8 concurrent writers × 3 repeats:** Inductor, tvm-ffi and TileLang had 0 failures and 0 rebuilds on
  load. This is APFS, not Lustre.
- **Patch for model-launch:** `patches/0001-0004`. It adds `{jit_cache}` resolved like `{arch}`, a per-image × GPU-arch
  shared cache, a node-local overlay, and atomic first-writer-wins promotion with no flock. 482 unit tests pass.
