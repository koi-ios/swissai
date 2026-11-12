# SGLang v0.5.20: how every on-disk cache is wired

Scope: `vendor/sglang-v0.5.20/python/sglang` (the `sglang_base` image's SGLang), plus the few lines of
FlashInfer / tvm-ffi / DeepGEMM / torch / Triton / servekit / model-launch needed to show how SGLang's
wiring takes effect. All paths are relative to `$OPP`. Every cited line is also in
`research/sglang_0520.citations.json` (line ranges here are for reading; the JSON pins one line each).

Labels: **inferred** = follows from the cited code plus stdlib/POSIX semantics, not executed.
**UNVERIFIED** = cannot be settled from the vendored sources (reason given).

Platform context. `sglang_base.toml` points at a CI-built `sglang_base-{arch}.sqsh`
(`vendor/model-launch/src/swiss_ai_model_launch/assets/envs/sglang_base.toml:2`). The closest recipe in the
repo is `vendor/model-launch/images/sglang_0.5.20/Dockerfile`, which is not the upstream SGLang
Dockerfile. It is a single stage on `nvidia/cuda:13.0.0-cudnn-devel`, so nvcc is present
(`vendor/model-launch/images/sglang_0.5.20/Dockerfile:8`, `:11`). It wheel-installs `sglang[all]==0.5.20`
(`vendor/model-launch/images/sglang_0.5.20/Dockerfile:71`) and sets no cache variables. That this
Dockerfile is the one that built `sglang_base` is **UNVERIFIED**: the CI mapping is not in the repo.
The env file sets no cache variables either; it sets only `SGL_ENABLE_JIT_DEEPGEMM = "0"`
(`vendor/model-launch/src/swiss_ai_model_launch/assets/envs/sglang_base.toml:31`).

---

## 0. The redirect: `SGLANG_CACHE_DIR` and `redirect_third_party_caches()`

1. **Variables and defaults.**
   - `SGLANG_CACHE_DIR` defaults to `~/.cache/sglang`. `expanduser` runs in the `Envs` class body, so it
     uses `$HOME` at the moment `sglang.srt.environ` is imported
     (`vendor/sglang-v0.5.20/python/sglang/srt/environ.py:1164`).
   - Derived defaults are computed lazily as `expanduser(SGLANG_CACHE_DIR)/<name>`
     (`vendor/sglang-v0.5.20/python/sglang/srt/environ.py:37`).
   - `import sglang` runs `_redirect_third_party_caches()` unconditionally, with no flag or check, at
     package import (`vendor/sglang-v0.5.20/python/sglang/__init__.py:14`). It deliberately runs before
     torch or FlashInfer are imported, "the first value set is the one that sticks"
     (`vendor/sglang-v0.5.20/python/sglang/__init__.py:8`).
   - `environ.py` imports only the stdlib (`vendor/sglang-v0.5.20/python/sglang/srt/environ.py:1-8`).
     `sglang/srt` has no `__init__.py` (checked by listing the directory), so nothing heavy runs first.
   - The redirect only **setdefaults** four variables
     (`vendor/sglang-v0.5.20/python/sglang/srt/environ.py:1947`):

     | variable | value | cite |
     |---|---|---|
     | `TRITON_CACHE_DIR` | `<base>/triton` | `vendor/sglang-v0.5.20/python/sglang/srt/environ.py:1926` |
     | `TORCHINDUCTOR_CACHE_DIR` | `<base>/inductor` | `vendor/sglang-v0.5.20/python/sglang/srt/environ.py:1927` |
     | `CUDA_CACHE_PATH` | `<base>/nv` | `vendor/sglang-v0.5.20/python/sglang/srt/environ.py:1928` |
     | `FLASHINFER_WORKSPACE_BASE` | `<base>` | `vendor/sglang-v0.5.20/python/sglang/srt/environ.py:1931` |

     A value already exported by the platform therefore wins.
   - FlashInfer appends `.cache/flashinfer` to its base
     (`vendor/src/flashinfer-python-0.6.18/flashinfer/jit/env.py:62`). Its JIT ops therefore land in
     `<SGLANG_CACHE_DIR>/.cache/flashinfer/<ver>/<archs>/cached_ops`, a hidden `.cache` inside the sglang
     directory (`vendor/src/flashinfer-python-0.6.18/flashinfer/jit/env.py:162`).
   - The SGLang docs describe `SGLANG_CACHE_DIR` as "the default root for compiled-kernel caches"
     (`vendor/sglang-v0.5.20/docs/docs/references/environment_variables.mdx:83`). A grep of `docs/` finds no
     entry for `SGLANG_JIT_CACHE_DIR`.
2. **When the redirect has no effect.**
   - FlashInfer resolves its workspace into a module constant at import
     (`vendor/src/flashinfer-python-0.6.18/flashinfer/jit/env.py:59`). If it was imported before `sglang`,
     the redirect does nothing for it (`vendor/sglang-v0.5.20/python/sglang/srt/environ.py:1942`).
   - Inductor's `cache_dir()` writes `TORCHINDUCTOR_CACHE_DIR` into `os.environ` itself
     (`vendor/src/pytorch-v2.13.0/torch/_inductor/runtime/cache_dir_utils.py:19`). If it already ran, the
     redirect does nothing for Inductor either (`vendor/sglang-v0.5.20/python/sglang/srt/environ.py:1943`).
   - Triton reads `TRITON_CACHE_DIR` through a knob that re-reads the environment on each access, so a
     late value still takes effect (`vendor/src/triton-3.7.1-5d6048a/python/triton/knobs.py:347`).
3. **Worker subprocesses.** The engine forces the `spawn` start method
   (`vendor/sglang-v0.5.20/python/sglang/srt/entrypoints/engine.py:1766`). Schedulers and TP workers are
   `mp.Process` children (`vendor/sglang-v0.5.20/python/sglang/srt/entrypoints/engine.py:896`).
   - They inherit the parent's already-redirected `os.environ`, and re-importing `sglang` in the child
     re-runs the setdefault as a no-op. Both are **inferred** from stdlib spawn semantics.
   - Edge case (**inferred**): if a wrapper imports FlashInfer before `sglang`, the parent uses
     `~/.cache/flashinfer` while the children use `<base>/.cache/flashinfer`.
4. **Legacy `SGL_*` names.** At `environ` import, every `SGL_X` variable is copied over `SGLANG_X`,
   overwriting it (`vendor/sglang-v0.5.20/python/sglang/srt/environ.py:1920`). This runs in every
   interpreter (`vendor/sglang-v0.5.20/python/sglang/srt/environ.py:1950`).
5. **Legacy directories.** `~/.triton`, `~/.cache/flashinfer` and `~/.cache/deep_gemm` are only logged,
   never deleted (`vendor/sglang-v0.5.20/python/sglang/srt/entrypoints/engine.py:1776`).
6. **What ends up under `SGLANG_CACHE_DIR`.**
   - `triton/`, `inductor/`, `nv/` (CUDA driver ComputeCache), `.cache/flashinfer/`.
   - `deep_gemm/` (§3), `cute_aot/` (§2), `flashinfer/autotune/` (§5).
   - `gpu_p2p_access_cache_for_*.json` (§6), `torch_compile_cache/` (§7), `rust_extensions/` (§8).
   - `model_streamer/` and `ple/` (§9).
   - `kernels.lock`, an input file (§4).
   - **Not** `jit/`: it is under the same default prefix only by coincidence (§1).

---

## 1. JIT kernel build cache: `SGLANG_JIT_CACHE_DIR` (the 0.5.20 replacement for the tvm-ffi cache)

**Who uses it.** `load_jit()` is called from 119 kernel modules under `python/sglang/kernels`, or 102
if diffusion and `kda_kernels` are excluded. This is a grep count that leaves out tests and the
`jit/utils` package itself. Two users are on default CUDA paths:

- **MHA KV-cache store.** It runs through the JIT `kvcache` module
  (`vendor/sglang-v0.5.20/python/sglang/srt/mem_cache/memory_pool.py:160`,
  `vendor/sglang-v0.5.20/python/sglang/kernels/ops/kvcache/kvcache.py:26`).
- **Custom all-reduce v2.** It is on by default
  (`vendor/sglang-v0.5.20/python/sglang/srt/environ.py:1284`) and selected on CUDA
  (`vendor/sglang-v0.5.20/python/sglang/srt/distributed/device_communicators/custom_all_reduce.py:358`).
  It is a JIT module (`vendor/sglang-v0.5.20/python/sglang/kernels/ops/communication/all_reduce.py:212`).

**tvm-ffi relationship.**

- SGLang now writes its own `build.ninja` instead of calling `tvm_ffi.cpp.load_inline`
  (`vendor/sglang-v0.5.20/python/sglang/kernels/jit/utils/compile/toolchain.py:3`).
- It still compiles against tvm-ffi headers and links `-l<tvm_ffi>`
  (`vendor/sglang-v0.5.20/python/sglang/kernels/jit/utils/compile/toolchain.py:158`).
- It loads the result with `tvm_ffi.load_module`
  (`vendor/sglang-v0.5.20/python/sglang/kernels/jit/utils/compile/loader.py:211`).
- A grep finds no `tvm_ffi.cpp` builder call anywhere in SGLang, so `TVM_FFI_CACHE_DIR` plays no part
  in these kernels.

1. **Variable and default.**
   - `SGLANG_JIT_CACHE_DIR` has no default value (`vendor/sglang-v0.5.20/python/sglang/srt/environ.py:1171`).
   - The root is the variable, or the literal `~/.cache/sglang/jit`
     (`vendor/sglang-v0.5.20/python/sglang/kernels/jit/utils/compile/cache.py:302`). An empty string
     also selects the literal.
   - This default does **not** follow `SGLANG_CACHE_DIR`. Relocating only `SGLANG_CACHE_DIR` leaves JIT
     builds in `$HOME/.cache/sglang/jit`.
   - The engine never overrides this variable.
   - Layout (`vendor/sglang-v0.5.20/python/sglang/kernels/jit/utils/compile/cache.py:17`, `:308`):

     ```
     <root>/<target>/<module_name>/build-<build_key16>/
         .lock
         .staging-<uuid4hex>/       build in progress
         deps-<deps_key16>/         one leaf per dependency-content state
     ```

   - `<target>` is `sm<maj><min><suffix>`
     (`vendor/sglang-v0.5.20/python/sglang/kernels/jit/utils/compile/cache.py:221`). It is `sm90a` on GH200,
     because the "a" suffix applies to 9.x and newer
     (`vendor/sglang-v0.5.20/python/sglang/kernels/jit/utils/arch.py:73`) and the capability comes from
     the local GPU (`vendor/sglang-v0.5.20/python/sglang/kernels/jit/utils/arch.py:99`).
   - Module names are `sgl_kernel_jit_<args joined by _>`
     (`vendor/sglang-v0.5.20/python/sglang/kernels/jit/utils/compile/spec.py:20`).
2. **Key.**
   - `build_key` is a 16-hex sha256 over five inputs
     (`vendor/sglang-v0.5.20/python/sglang/kernels/jit/utils/compile/cache.py:269-288`):
     - the module args (dtype, world size, PDL flag, ...);
     - path and content of each direct source;
     - the anchor-normalized `build.ninja` text
       (`vendor/sglang-v0.5.20/python/sglang/kernels/jit/utils/compile/cache.py:284`). That text includes
       the compilers, all flags, `-gencode=arch=compute_90a,code=sm_90a`
       (`vendor/sglang-v0.5.20/python/sglang/kernels/jit/utils/compile/toolchain.py:124`), include
       directories and link flags;
     - the generated wrapper sources;
     - an environment fingerprint
       (`vendor/sglang-v0.5.20/python/sglang/kernels/jit/utils/compile/cache.py:286`).
   - The environment fingerprint covers:
     - the full target (`cuda:9.0a`);
     - the `--version` output of nvcc and of the host `c++`
       (`vendor/sglang-v0.5.20/python/sglang/kernels/jit/utils/compile/cache.py:244`). nvcc is found via
       `CUDA_HOME`, then `CUDA_PATH`, then `PATH`, then `/usr/local/cuda`
       (`vendor/sglang-v0.5.20/python/sglang/kernels/jit/utils/compile/toolchain.py:38`); the host
       compiler is `$CXX` or `c++` (`vendor/sglang-v0.5.20/python/sglang/kernels/jit/utils/compile/toolchain.py:72`);
     - the installed dist versions of apache-tvm-ffi, flashinfer-python, deep_gemm, nvidia-mathdx and
       torch (`vendor/sglang-v0.5.20/python/sglang/kernels/jit/utils/compile/cache.py:252`).
   - `deps_key` is a 16-hex sha256 over the sorted `(anchor, relpath, sha256)` list of every file the
     compiler read, taken from the `.o.d` depfiles
     (`vendor/sglang-v0.5.20/python/sglang/kernels/jit/utils/compile/cache.py:317-318`,
     `vendor/sglang-v0.5.20/python/sglang/kernels/jit/utils/compile/ninja.py:175-194`).
   - **Not in the key:** driver version, GPU model beyond capability, Python version, hostname, cache-root
     location, and absolute install paths (these are normalized; see point 4).
   - Note: the dist name `deep_gemm` does not match the installed `sgl_deep_gemm` dist-info, so it is
     probably recorded as "absent" (**inferred**). The DeepGEMM headers are still covered by
     content digests.
   - Gotcha: if `nvcc` or `c++` cannot be run, the fingerprint records "unknown"
     (`vendor/sglang-v0.5.20/python/sglang/kernels/jit/utils/compile/cache.py:247`). The key then
     changes, so even a warm cache misses.
   - Consequence: the runtime image must carry the same nvcc and host compiler as the image that
     filled the cache. Which `c++` wins on the platform image's PATH, conda's
     (`/opt/conda/bin` comes first) or Ubuntu's, is **UNVERIFIED**.
3. **Write pattern.**
   - **Lookup without a lock**
     (`vendor/sglang-v0.5.20/python/sglang/kernels/jit/utils/compile/loader.py:116`).
   - **On a miss:**
     - An optional crash check runs first
       (`vendor/sglang-v0.5.20/python/sglang/kernels/jit/utils/compile/loader.py:131`).
     - It then takes an exclusive `fcntl.flock`
       (`vendor/sglang-v0.5.20/python/sglang/kernels/jit/utils/compile/loader.py:202`) on `<scope>/.lock`,
       opened `O_CREAT|O_RDWR` with mode `0o644`
       (`vendor/sglang-v0.5.20/python/sglang/kernels/jit/utils/compile/loader.py:200`).
     - It re-checks the cache.
     - It builds in `.staging-<uuid4>` (`vendor/sglang-v0.5.20/python/sglang/kernels/jit/utils/compile/loader.py:168`).
       The uuid is used instead of the pid because the cache is "meant to be a shared mount".
     - ninja runs with `-j $MAX_JOBS` if set
       (`vendor/sglang-v0.5.20/python/sglang/kernels/jit/utils/compile/ninja.py:156`) and a 1800 s timeout.
   - **Publication:**
     - The staged `.so` is dlopened before publishing
       (`vendor/sglang-v0.5.20/python/sglang/kernels/jit/utils/compile/loader.py:173`).
     - `sgl_deps.json` is written into staging
       (`vendor/sglang-v0.5.20/python/sglang/kernels/jit/utils/compile/cache.py:433`).
     - The directory is renamed atomically
       (`vendor/sglang-v0.5.20/python/sglang/kernels/jit/utils/compile/cache.py:522`).
     - Losing that race to an existing leaf counts as success
       (`vendor/sglang-v0.5.20/python/sglang/kernels/jit/utils/compile/cache.py:524`).
   - The lock only deduplicates the N TP ranks' compiles; the rename is what makes publication safe
     (`vendor/sglang-v0.5.20/python/sglang/kernels/jit/utils/compile/loader.py:189-197`).
   - **Half-written entries.**
     - A leaf appears complete or not at all.
     - Under the lock, a leaf whose `.so` fails to load is deleted and rebuilt
       (`vendor/sglang-v0.5.20/python/sglang/kernels/jit/utils/compile/loader.py:159`).
     - A build killed mid-way (SIGKILL) leaves a `.staging-*` directory. Lookups ignore it, since they
       only scan `deps-*`. No code removes these leftovers (**inferred**).
   - **Pruning.**
     - `SGLANG_JIT_CACHE_KEEP` defaults to keeping everything
       (`vendor/sglang-v0.5.20/python/sglang/srt/environ.py:1177`,
       `vendor/sglang-v0.5.20/python/sglang/kernels/jit/utils/compile/cache.py:451`).
     - When set, it keeps the newest N leaves per build-key directory and removes the rest with
       `rmtree` (`vendor/sglang-v0.5.20/python/sglang/kernels/jit/utils/compile/cache.py:465`).
     - Old build-key directories are never pruned (**inferred**).
     - With KEEP set, `_prune` calls `stat()` on sibling leaves without a try
       (`vendor/sglang-v0.5.20/python/sglang/kernels/jit/utils/compile/cache.py:454-462`). A
       concurrent prune can therefore raise `FileNotFoundError` out of `load_jit` (**inferred** race).
4. **Relocatability: yes, by design.**
   - Paths are rewritten to anchor tokens: `kernels`, `tvm_ffi`, `pkg:flashinfer`, `pkg:deep_gemm`,
     `pkg:nvidia`, `toolkit`, `sitepkgs`, and `sys` (= `/usr`), resolved and longest first
     (`vendor/sglang-v0.5.20/python/sglang/kernels/jit/utils/compile/cache.py:117`,
     `vendor/sglang-v0.5.20/python/sglang/kernels/jit/utils/compile/cache.py:297`).
   - This applies both in the key and in `sgl_deps.json`
     (`vendor/sglang-v0.5.20/python/sglang/kernels/jit/utils/compile/cache.py:81`). Paths outside every
     anchor are stored as `abs` (`vendor/sglang-v0.5.20/python/sglang/kernels/jit/utils/compile/cache.py:145`).
   - `build.ninja` and the `.o.d` files inside a leaf do contain absolute paths, but a lookup never reads
     them (it reads only `sgl_deps.json` and the `.so`).
   - So moving the cache root costs nothing, and moving the install keeps hits if every input sits under
     an anchor.
   - No rpath is emitted (`vendor/sglang-v0.5.20/python/sglang/kernels/jit/utils/compile/toolchain.py:158`).
     How the `.so`'s `libtvm_ffi` and `libcudart` dependencies resolve at dlopen time is **UNVERIFIED**
     (dynamic-loader behaviour, not in source).
   - servekit's premise ("ninja records absolute paths") does not hold for this cache (§11).
5. **Hit/miss check.**
   - Leaves are scanned newest-mtime first (`vendor/sglang-v0.5.20/python/sglang/kernels/jit/utils/compile/cache.py:361`),
     at most 100 (`vendor/sglang-v0.5.20/python/sglang/kernels/jit/utils/compile/cache.py:75`).
   - For each leaf, every recorded dependency is re-hashed with SHA-256
     (`vendor/sglang-v0.5.20/python/sglang/kernels/jit/utils/compile/cache.py:176`).
   - It is a hit only if the recomputed `deps_key` equals the leaf's own name
     (`vendor/sglang-v0.5.20/python/sglang/kernels/jit/utils/compile/cache.py:383`) and the `.so` exists.
     Hits are decided by content hash; mtime only orders the scan.
   - **Read-only directory:**
     - A hit takes no lock and makes no directory. Its only write is an `os.utime` that is explicitly
       best-effort, because "cache root can be a read-only mount"
       (`vendor/sglang-v0.5.20/python/sglang/kernels/jit/utils/compile/cache.py:391`). Hits therefore work
       on a read-only cache. Whether `tvm_ffi.load_module` itself writes anywhere is **UNVERIFIED**.
     - A miss on a read-only root fails when it opens `.lock` with `O_CREAT|O_RDWR`
       (`vendor/sglang-v0.5.20/python/sglang/kernels/jit/utils/compile/loader.py:200`). The OSError is not
       caught (**inferred**).
     - `SGLANG_CRASH_ON_JIT_COMPILE=1` turns a miss into a clear RuntimeError raised before the lock
       (`vendor/sglang-v0.5.20/python/sglang/srt/environ.py:1181`,
       `vendor/sglang-v0.5.20/python/sglang/kernels/jit/utils/compile/loader.py:131`).
   - **Several users sharing one cache:** `.lock` is created `0o644`, so another UID cannot open it
     `O_RDWR`. A second user's misses therefore fail while hits work (**inferred**; depends on umask and
     ACLs).
   - Debugging: `SGLANG_JIT_CACHE_DEBUG` logs the changed dependency at INFO
     (`vendor/sglang-v0.5.20/python/sglang/srt/environ.py:1173`,
     `vendor/sglang-v0.5.20/python/sglang/kernels/jit/utils/compile/cache.py:400`).
6. **Content.**
   - The `<module>.so`.
   - `.o` objects named `<stem>_<i>.o` (`vendor/sglang-v0.5.20/python/sglang/kernels/jit/utils/compile/ninja.py:122`).
   - `.o.d` depfiles, kept because `deps = gcc` is not emitted
     (`vendor/sglang-v0.5.20/python/sglang/kernels/jit/utils/compile/ninja.py:10`).
   - `build.ninja` (`vendor/sglang-v0.5.20/python/sglang/kernels/jit/utils/compile/ninja.py:153`).
   - The generated `main.cpp` / `cuda.cu` wrappers.
   - `sgl_deps.json` (`vendor/sglang-v0.5.20/python/sglang/kernels/jit/utils/compile/cache.py:67`).
   - Per scope: `.lock` and transient `.staging-*` directories.
   - ninja's own `.ninja_log` is **UNVERIFIED** (ninja behaviour, not SGLang source).
   - No PTX or IR is kept: the build emits `code=sm_90a` SASS only.

---

## 2. CuTe DSL AOT object cache: `SGLANG_CUTE_AOT_CACHE_DIR`

**Who uses it.** `--attention-backend fa4` imports SGLang's vendored CuTe FA4 unless
`SGLANG_INKLING_FA4_USE_PIP=1` (`vendor/sglang-v0.5.20/python/sglang/kernels/ops/attention/flash_attention_v4.py:17`).
FA4 keeps persistent caches named `fwd`, `fwd_shear_bias`, `fwd_prepare_shear_bias` and `fwd_combine`
(`vendor/sglang-v0.5.20/python/sglang/kernels/ops/attention/flash_attn/cute/interface.py:1862`). The Kimi-K3
KDA MTP kernel also uses it. The platform image names fa4 as the GH200 backend
(`vendor/model-launch/images/sglang_0.5.20/Dockerfile:68`), so this cache is on the platform's hot path.

1. **Variable.** Defaults lazily to `<SGLANG_CACHE_DIR>/cute_aot`
   (`vendor/sglang-v0.5.20/python/sglang/srt/environ.py:1168`). An empty value gives an in-memory,
   process-local cache (`vendor/sglang-v0.5.20/python/sglang/kernels/jit/cute_aot_cache.py:412`). The
   directory is `expanduser(root)/<fingerprint>/<name>`
   (`vendor/sglang-v0.5.20/python/sglang/kernels/jit/cute_aot_cache.py:421`).
2. **Key.**
   - The fingerprint directory is a sha256 over:
     - Python `major.minor` (`vendor/sglang-v0.5.20/python/sglang/kernels/jit/cute_aot_cache.py:60`);
     - host platform and machine (`vendor/sglang-v0.5.20/python/sglang/kernels/jit/cute_aot_cache.py:62`);
     - `cutlass.__version__` (`vendor/sglang-v0.5.20/python/sglang/kernels/jit/cute_aot_cache.py:63`);
     - `cutlass.CUDA_VERSION` (`vendor/sglang-v0.5.20/python/sglang/kernels/jit/cute_aot_cache.py:64`);
     - the tvm-ffi flag and the arch (`vendor/sglang-v0.5.20/python/sglang/kernels/jit/cute_aot_cache.py:66`).
       The arch comes from `CUTE_DSL_ARCH`, else the torch
       capability as `sm_XY` with no "a" (`vendor/sglang-v0.5.20/python/sglang/kernels/jit/cute_aot_cache.py:94`);
     - `tvm_ffi.__version__` (`vendor/sglang-v0.5.20/python/sglang/kernels/jit/cute_aot_cache.py:70`);
     - the relative path and content of every `.py` source
       (`vendor/sglang-v0.5.20/python/sglang/kernels/jit/cute_aot_cache.py:85`). For FA4 this is the
       whole `flash_attn/cute` directory
       (`vendor/sglang-v0.5.20/python/sglang/kernels/ops/attention/flash_attn/cute/interface.py:1857`).
   - The entry name is `sha256(pickle((enable_tvm_ffi, compile_key)))`
     (`vendor/sglang-v0.5.20/python/sglang/kernels/jit/cute_aot_cache.py:376`). FA4's `compile_key` is a
     tuple of dtype, head dims, causal flag, mask and score mods, tile sizes, arch and similar
     (`vendor/sglang-v0.5.20/python/sglang/kernels/ops/attention/flash_attn/cute/interface.py:1238-1301`).
   - Not in the key: driver, nvcc, absolute paths.
3. **Write pattern.**
   - Takes an exclusive `fcntl.flock` on `<hash>.lock`, polled with `LOCK_NB`
     (`vendor/sglang-v0.5.20/python/sglang/kernels/jit/cute_aot_cache.py:182`) and timing out after 15 s
     (`vendor/sglang-v0.5.20/python/sglang/kernels/jit/cute_aot_cache.py:238`).
   - Skips the export if `<hash>.o` already exists
     (`vendor/sglang-v0.5.20/python/sglang/kernels/jit/cute_aot_cache.py:349`).
   - Otherwise exports to `.<hash>.tmp.o` (`vendor/sglang-v0.5.20/python/sglang/kernels/jit/cute_aot_cache.py:352`)
     and publishes with `os.replace` (`vendor/sglang-v0.5.20/python/sglang/kernels/jit/cute_aot_cache.py:367`).
   - Readers hold a shared flock while loading. An object that fails to load is unlinked, under the
     exclusive lock, only if its inode is unchanged
     (`vendor/sglang-v0.5.20/python/sglang/kernels/jit/cute_aot_cache.py:330`).
4. **Relocatability.** The fingerprint uses relative source paths plus content
   (`vendor/sglang-v0.5.20/python/sglang/kernels/jit/cute_aot_cache.py:85`), so moving the cache root or
   the install does not change keys. Whether the exported `.o` embeds absolute paths is **UNVERIFIED**:
   CuTe DSL's `export_to_c` and `load_module` are not vendored in readable form.
5. **Hit/miss check.**
   - A key is checked in memory first, then on disk under a shared lock (`<hash>.o` exists → load).
   - Both lock modes open the lock file with `O_CREAT`
     (`vendor/sglang-v0.5.20/python/sglang/kernels/jit/cute_aot_cache.py:172`).
   - **Read-only directory:** a hit works only if `<hash>.lock` already exists. It does when the cache
     was seeded by an exporting run, because lock files are never deleted
     (`vendor/sglang-v0.5.20/python/sglang/kernels/jit/cute_aot_cache.py:379`). This relies on
     `O_RDONLY|O_CREAT` on an existing file succeeding on a read-only mount, which is Linux `open(2)`
     behaviour (**inferred**).
   - A miss on a read-only or foreign-UID directory raises from `os.open`, either when creating the
     lock or when exporting (**inferred**).
   - On a filesystem without flock support, `flock` errors are retried until the 15 s timeout and then
     raise RuntimeError (**inferred** from the polling loop; Lustre's flock mount option is
     **UNVERIFIED**).
6. **Content.**
   - `<sha256>.o`, a host object with the device code embedded. These are loaded into the process, so the
     directory "must be trusted" (`vendor/sglang-v0.5.20/python/sglang/kernels/jit/cute_aot_cache.py:7`).
   - A persistent `<sha256>.lock` per entry (`vendor/sglang-v0.5.20/python/sglang/kernels/jit/cute_aot_cache.py:283`, `:379`).
   - Transient `.<sha256>.tmp.o` files.
   - SGLang does not redirect CuTe DSL's own compile machinery (other topic).

---

## 3. DeepGEMM: how SGLang sets `DG_JIT_CACHE_DIR`

1. **Variable.**
   - `SGLANG_DG_CACHE_DIR` defaults lazily to `<SGLANG_CACHE_DIR>/deep_gemm`
     (`vendor/sglang-v0.5.20/python/sglang/srt/environ.py:1154`).
   - Importing `compile_utils` **hard-assigns**, rather than setdefaults, `DG_JIT_CACHE_DIR` from it
     (`vendor/sglang-v0.5.20/python/sglang/srt/layers/deep_gemm_wrapper/compile_utils.py:46`). Any
     `DG_JIT_CACHE_DIR` the platform exports is therefore overwritten. Use `SGLANG_DG_CACHE_DIR`
     instead.
   - `compile_utils` is imported by `entrypoint.py`
     (`vendor/sglang-v0.5.20/python/sglang/srt/layers/deep_gemm_wrapper/entrypoint.py:8`), which
     `model_runner` imports unconditionally
     (`vendor/sglang-v0.5.20/python/sglang/srt/model_executor/model_runner.py:75`). So every
     scheduler/TP process sets the variable, whether or not DeepGEMM JIT is enabled.
   - It also sets `DG_JIT_USE_NVRTC` from the legacy name `SGL_DG_USE_NVRTC`, default 0
     (`vendor/sglang-v0.5.20/python/sglang/srt/layers/deep_gemm_wrapper/compile_utils.py:51`).
   - On the DeepGEMM side (summary only; see the DeepGEMM topic):
     - `<prefix>_JIT_CACHE_DIR` is read
       (`vendor/src/sgl-deep-gemm-0.2.0/deep_gemm/include/deep_jit/cache/disk.hpp:91`).
     - It is parsed as one writable root followed by read-only lookup roots, `PATH_1:PATH_2:...`
       (`vendor/src/sgl-deep-gemm-0.2.0/deep_gemm/include/deep_jit/cache/disk.hpp:88`).
     - The runtime is built lazily
       (`vendor/src/sgl-deep-gemm-0.2.0/deep_gemm/include/deep_jit/runtime/runtime.hpp:102`), so
       SGLang's import-time assignment comes before the first JIT. That `_C.so` uses this lazy path with
       prefix `DG` is **UNVERIFIED**: the wheel ships no csrc.
   - **Interaction (inferred):** SGLang's lock code treats the variable as one path
     (`vendor/sglang-v0.5.20/python/sglang/srt/layers/deep_gemm_wrapper/compile_utils.py:132`). A
     colon-separated read-only layering would create a directory literally named `"<p1>:<p2>/locks"`.
2. **Key (SGLang side).** Only the lock name `{KERNEL}_n{N}_k{K}_g{G}.lock`
   (`vendor/sglang-v0.5.20/python/sglang/srt/layers/deep_gemm_wrapper/compile_utils.py:134`). The kernel
   key belongs to DeepGEMM. Its entries are `<root>/cache/<tag>.<digest>/` with a commit marker
   (`vendor/src/sgl-deep-gemm-0.2.0/deep_gemm/include/deep_jit/cache/disk.hpp:77`).
3. **Write pattern and precompile flow.**
   - Locking: `open(lock, "w")`
     (`vendor/sglang-v0.5.20/python/sglang/srt/layers/deep_gemm_wrapper/compile_utils.py:135`), then
     exclusive `fcntl.flock` (`vendor/sglang-v0.5.20/python/sglang/srt/layers/deep_gemm_wrapper/compile_utils.py:136`).
     The holder compiles; the others then hit.
   - Precompile is on by default (`SGLANG_JIT_DEEPGEMM_PRECOMPILE=True`,
     `vendor/sglang-v0.5.20/python/sglang/srt/environ.py:1149`).
   - When a GEMM shape group is first executed, only the first rank on the node
     (`vendor/sglang-v0.5.20/python/sglang/srt/layers/deep_gemm_wrapper/compile_utils.py:105`) runs every M
     from 1 to `m_max` (`vendor/sglang-v0.5.20/python/sglang/srt/layers/deep_gemm_wrapper/compile_utils.py:97`;
     `m_max` is 16K-128K) through the kernel
     (`vendor/sglang-v0.5.20/python/sglang/srt/layers/deep_gemm_wrapper/compile_utils.py:249`).
   - The offline tool `python -m sglang.compile_deep_gemm` forces
     `SGLANG_ENABLE_JIT_DEEPGEMM=True` in its own process
     (`vendor/sglang-v0.5.20/python/sglang/compile_deep_gemm.py:41`).
4. **Relocatability.** DeepGEMM's concern (other topic).
5. **Read-only directory.** The lock is `open(..., "w")`, which fails on a read-only filesystem or for
   another UID (**inferred**). This path is only reached when the wrapper is enabled.
6. **Content (SGLang side).** `locks/*.lock`, never removed.

**What `SGL_ENABLE_JIT_DEEPGEMM=0` does.** The platform sets it
(`vendor/model-launch/src/swiss_ai_model_launch/assets/envs/sglang_base.toml:31`) and also exports
`SGLANG_ENABLE_JIT_DEEPGEMM="false"` (`vendor/model-launch/src/swiss_ai_model_launch/launchers/framework.py:64`).

- The legacy name is forwarded by the generic `SGL_` → `SGLANG_` rewrite
  (`vendor/sglang-v0.5.20/python/sglang/srt/environ.py:1916`). `ENABLE_JIT_DEEPGEMM` then becomes False
  (`vendor/sglang-v0.5.20/python/sglang/srt/layers/deep_gemm_wrapper/configurer.py:36`; the default is
  True at `vendor/sglang-v0.5.20/python/sglang/srt/environ.py:1128`).
- **It disables:**
  - the wrapper's dense FP8 and BF16 GEMMs and grouped MoE GEMMs;
  - `update_deep_gemm_config` (`vendor/sglang-v0.5.20/python/sglang/srt/model_executor/model_runner.py:457`);
  - the precompile/warmup and the `locks/` directory.
- **It does not disable:**
  - the `DG_JIT_CACHE_DIR` assignment;
  - direct `import deep_gemm` users on CUDA, such as the DSA backend
    (`vendor/sglang-v0.5.20/python/sglang/srt/layers/attention/dsa_backend.py:98`), the DSA and DSv4
    indexers, and mega-MoE. These still JIT into `DG_JIT_CACHE_DIR` when those models run
    (**inferred**).
- **Side effect (inferred):** because `SGL_` is re-applied at every interpreter's `environ` import, the
  `set(True)` in `compile_deep_gemm` is undone in its spawned server children while
  `SGL_ENABLE_JIT_DEEPGEMM` is exported.

---

## 4. FA3 kernel lockfile and the HF kernels hub

1. **Variable.**
   - By default `SGLANG_USE_SGL_FA3_KERNEL=True`
     (`vendor/sglang-v0.5.20/python/sglang/srt/environ.py:1244`). FA3 then comes from the `sglang-kernel`
     wheel and no cache is involved
     (`vendor/sglang-v0.5.20/python/sglang/kernels/ops/attention/flash_attention_v3.py:34`).
   - Only with the flag set to 0 is `<SGLANG_CACHE_DIR>/kernels.lock` used
     (`vendor/sglang-v0.5.20/python/sglang/kernels/ops/attention/flash_attention_v3.py:16`, `:42`). That
     path is joined with no `expanduser`.
2. **Key.** The repo is `kernels-community/sgl-flash-attn3`
   (`vendor/sglang-v0.5.20/python/sglang/kernels/ops/attention/flash_attention_v3.py:14`), pinned by the
   lockfile or by revision `v1`.
3. **Write pattern.** Done by the `kernels` package, not by SGLang. Its storage location (HF hub cache?)
   is **UNVERIFIED**: the package is not vendored.
4. **Relocatability.** Not applicable to SGLang.
5. **Hit/miss check.** The lockfile exists → `load_kernel(repo, lockfile)`
   (`vendor/sglang-v0.5.20/python/sglang/kernels/ops/attention/flash_attention_v3.py:51`, `:52`).
   Otherwise `get_kernel(repo, revision="v1")` attempts a Hub fetch
   (`vendor/sglang-v0.5.20/python/sglang/kernels/ops/attention/flash_attention_v3.py:54`). Any exception
   falls back to sglang-kernel FA3
   (`vendor/sglang-v0.5.20/python/sglang/kernels/ops/attention/flash_attention_v3.py:68`).
6. **Content.** One input file, `kernels.lock`.
   - Upstream's Dockerfile places it in `/root/.cache/sglang`
     (`vendor/sglang-v0.5.20/docker/Dockerfile:609`) and copies that directory into the runtime image
     (`vendor/sglang-v0.5.20/docker/Dockerfile:767`). It skips the cubin download on aarch64
     (`vendor/sglang-v0.5.20/docker/Dockerfile:587`).
   - Relocating `SGLANG_CACHE_DIR` hides that lockfile.
   - The platform image runs no `kernels lock` step. It says FA3 is available only where the
     sglang-kernel wheel carries it
     (`vendor/model-launch/images/sglang_0.5.20/Dockerfile:66`), and recommends fa4 on GH200.
   - Net effect on the platform: no cache, unless someone sets `SGLANG_USE_SGL_FA3_KERNEL=0`, which
     triggers a network fetch.

---

## 5. FlashInfer autotune results

This cache is used only when autotune applies: a flashinfer MoE runner backend, or FP4/FP8 GEMMs that
need tuning, on sm ≥ 90
(`vendor/sglang-v0.5.20/python/sglang/srt/model_executor/runner/flashinfer_autotune.py:124`).

1. **Variable.**
   - Path: `<SGLANG_CACHE_DIR>/flashinfer/autotune/<flashinfer_version>/sm<maj><min>/<key16>/rank_tp{r}_pp{p}_dp{d}.json`.
     The root gets no `expanduser`
     (`vendor/sglang-v0.5.20/python/sglang/srt/model_executor/runner/flashinfer_autotune.py:163`, `:141`, `:173`).
   - Reuse is controlled by `SGLANG_FLASHINFER_AUTOTUNE_CACHE`, default True
     (`vendor/sglang-v0.5.20/python/sglang/srt/environ.py:1060`). Setting it False writes timestamped
     files under `runs/` instead
     (`vendor/sglang-v0.5.20/python/sglang/srt/model_executor/runner/flashinfer_autotune.py:264`).
2. **Key.**
   - `sha256[:16]` of a model key
     (`vendor/sglang-v0.5.20/python/sglang/srt/model_executor/runner/flashinfer_autotune.py:161`). It
     includes:
     - the model path string, an HF id or absolute path
       (`vendor/sglang-v0.5.20/python/sglang/srt/model_executor/runner/flashinfer_autotune.py:145`);
     - dtype, quantization and MoE backend;
     - tp, pp, attn-dp and ep sizes
       (`vendor/sglang-v0.5.20/python/sglang/srt/model_executor/runner/flashinfer_autotune.py:149`);
     - the HF config class, skip_ops and draft quantization.
   - FlashInfer also stamps `_metadata` (CUDA, cuBLAS and cuDNN versions, GPU name) and ignores
     mismatching files (`vendor/src/flashinfer-python-0.6.18/flashinfer/autotuner/autotuner.py:325`).
3. **Write pattern.**
   - FlashInfer saves on exit only when new tactics were tuned
     (`vendor/src/flashinfer-python-0.6.18/flashinfer/autotuner/autotuner.py:1002`).
   - The save writes a same-directory `mkstemp` file
     (`vendor/src/flashinfer-python-0.6.18/flashinfer/autotuner/autotuner.py:2619`) and publishes it with
     `os.replace` (`vendor/src/flashinfer-python-0.6.18/flashinfer/autotuner/autotuner.py:2639`).
   - SGLang deletes the file when TP ranks' copies disagree
     (`vendor/sglang-v0.5.20/python/sglang/srt/model_executor/runner/flashinfer_autotune.py:246`).
4. **Relocatability.** The model path is in the key, so the same model at another path tunes again.
5. **Hit/miss check.**
   - An existing file is loaded at context entry
     (`vendor/src/flashinfer-python-0.6.18/flashinfer/autotuner/autotuner.py:934`) with an unguarded
     `open` (`vendor/src/flashinfer-python-0.6.18/flashinfer/autotuner/autotuner.py:2718`).
   - **Read-only directory:** works if every shape is already covered. In that case nothing is saved, and
     `mkdir(exist_ok)` is harmless
     (`vendor/sglang-v0.5.20/python/sglang/srt/model_executor/runner/flashinfer_autotune.py:170`).
   - **Several users:** `mkstemp` creates files with mode 0600, a stdlib behaviour, and `os.replace`
     keeps that mode. Another UID's `open` therefore raises PermissionError at startup (**inferred**).
6. **Content.** Per-rank JSON tactic tables, `runs/` timestamped copies, and transient `.autotuner_*.tmp`
   files.

---

## 6. GPU P2P access cache

1. **Variable.** `<expanduser(SGLANG_CACHE_DIR)>/gpu_p2p_access_cache_for_<CUDA_VISIBLE_DEVICES>.json`
   (`vendor/sglang-v0.5.20/python/sglang/srt/distributed/device_communicators/custom_all_reduce_utils.py:267`, `:269`).
2. **Key.** Only the `CUDA_VISIBLE_DEVICES` string
   (`vendor/sglang-v0.5.20/python/sglang/srt/distributed/device_communicators/custom_all_reduce_utils.py:261`).
   It has no hostname, driver or GPU model, so all nodes sharing one directory share the file.
3. **Write pattern.**
   - Local rank 0 writes the file only if it is missing
     (`vendor/sglang-v0.5.20/python/sglang/srt/distributed/device_communicators/custom_all_reduce_utils.py:284`).
   - It writes in place with `open(path, "w")` and `json.dump`. There is no temp file, rename or lock
     (`vendor/sglang-v0.5.20/python/sglang/srt/distributed/device_communicators/custom_all_reduce_utils.py:323`, `:324`).
   - Other ranks wait on a barrier
     (`vendor/sglang-v0.5.20/python/sglang/srt/distributed/device_communicators/custom_all_reduce_utils.py:326`).
   - **Race (inferred):** a job on another node can `json.load`
     (`vendor/sglang-v0.5.20/python/sglang/srt/distributed/device_communicators/custom_all_reduce_utils.py:329`)
     a half-written file and crash.
4. **Relocatability.** The file contains no paths.
5. **Hit/miss check.** The file exists → read it. This works read-only once the file is present.
   - It is reached on CUDA through the custom all-reduce check
     (`vendor/sglang-v0.5.20/python/sglang/srt/distributed/device_communicators/custom_all_reduce_utils.py:504`),
     which applies to v1 and v2.
   - `SGLANG_SKIP_P2P_CHECK=1` skips the file
     (`vendor/sglang-v0.5.20/python/sglang/srt/distributed/device_communicators/custom_all_reduce_utils.py:430`).
6. **Content.** One small JSON. The probe also uses a temp file in `$TMPDIR`
   (`vendor/sglang-v0.5.20/python/sglang/srt/distributed/device_communicators/custom_all_reduce_utils.py:304`).

---

## 7. torch.compile piecewise backend (`tc_piecewise`) and Inductor/Triton overrides

The default prefill CUDA-graph backend on CUDA is `breakable`, which does not use torch.compile
(`vendor/sglang-v0.5.20/python/sglang/srt/model_executor/runner/prefill_cuda_graph_runner.py:17`). This
section applies only to `tc_piecewise`
(`vendor/sglang-v0.5.20/python/sglang/srt/model_executor/runner/prefill_cuda_graph_runner.py:31`), which
installs `SGLangBackend` (`vendor/sglang-v0.5.20/python/sglang/srt/compilation/compile.py:170`).

1. **Variable.**
   - Path: `<SGLANG_CACHE_DIR>/torch_compile_cache/<hash10>/rank_0_0/<model_tag>/`, with no
     `expanduser` (`vendor/sglang-v0.5.20/python/sglang/srt/compilation/backend.py:403`, `:408`).
   - `rank` and `dp_rank` are hard-coded to 0, so all ranks share one directory
     (`vendor/sglang-v0.5.20/python/sglang/srt/compilation/backend.py:415`).
   - **Engine override:** `InductorAdaptor.initialize_cache` hard-sets `TORCHINDUCTOR_CACHE_DIR=<dir>/inductor_cache`
     (`vendor/sglang-v0.5.20/python/sglang/srt/compilation/compiler_interface.py:198`) and
     `TRITON_CACHE_DIR=<dir>/triton_cache`
     (`vendor/sglang-v0.5.20/python/sglang/srt/compilation/compiler_interface.py:201`) for the rest of
     the process. This overrides the redirect and any platform or servekit value.
2. **Key.**
   - `md5[:10]` (`vendor/sglang-v0.5.20/python/sglang/srt/compilation/compiler_interface.py:179`) of
     Inductor's system factors
     (`vendor/sglang-v0.5.20/python/sglang/srt/compilation/compiler_interface.py:117`) and `torch_key()`
     (`vendor/sglang-v0.5.20/python/sglang/srt/compilation/compiler_interface.py:123`). The contents of
     those factors are torch internals (torch topic).
   - Model identity is not part of the directory. Reuse comes from Inductor's local FX graph cache,
     which is forced on (`vendor/sglang-v0.5.20/python/sglang/srt/compilation/compiler_interface.py:219`).
3. **Write pattern.**
   - The index `sglang_compile_cache.py`
     (`vendor/sglang-v0.5.20/python/sglang/srt/compilation/backend.py:93`) is rewritten in place
     (`vendor/sglang-v0.5.20/python/sglang/srt/compilation/backend.py:108`).
   - It is parsed at start-up (`vendor/sglang-v0.5.20/python/sglang/srt/compilation/backend.py:97`), but
     loading graphs from it is not implemented
     (`vendor/sglang-v0.5.20/python/sglang/srt/compilation/backend.py:157`). A grep finds no caller of
     `CompilerManager.load`.
   - Rank 0 writes `computation_graph_<time.time()>.py` on every compile
     (`vendor/sglang-v0.5.20/python/sglang/srt/compilation/backend.py:459`), so the directory grows each run.
4. **Relocatability.** Inductor and Triton concern (other topics).
5. **Read-only directory.** Not safe: `makedirs`, the graph dump and the index write all write
   (**inferred**).
6. **Content.** The index `.py`, graph dumps, `inductor_cache/` and `triton_cache/`.

Separately, plain `@torch.compile` helpers sit on ordinary paths, for example in the sampler
(`vendor/sglang-v0.5.20/python/sglang/srt/layers/sampler.py:849`). When those helpers execute, a normal
run writes Inductor and Triton artefacts to the redirected `<base>/inductor` and `<base>/triton`
(**inferred**).

---

## 8. Rust extension cache

1. **Variable.**
   - Root: `<SGLANG_CACHE_DIR>/rust_extensions`
     (`vendor/sglang-v0.5.20/python/sglang/srt/rust_extensions/loader.py:359`).
   - Mode: `SGLANG_RUST_BUILD_MODE`, default `auto`
     (`vendor/sglang-v0.5.20/python/sglang/srt/environ.py:1740`). Upstream's image sets `never`
     (`vendor/sglang-v0.5.20/docker/Dockerfile:612`).
   - A bundled extension is trusted in mode `never`, or when there is no source checkout
     (`vendor/sglang-v0.5.20/python/sglang/srt/rust_extensions/loader.py:117`). A checkout means a
     `rust/` directory four levels up
     (`vendor/sglang-v0.5.20/python/sglang/srt/rust_extensions/loader.py:45`).
   - The platform image is a wheel install with no checkout, so the bundled module is used. This cache
     is untouched unless the bundled import fails (**inferred**).
2. **Key.** A JSON fingerprint of crate, features and source digest, the cargo and rustc versions
   (`vendor/sglang-v0.5.20/python/sglang/srt/rust_extensions/loader.py:262`), the Python ABI
   (`vendor/sglang-v0.5.20/python/sglang/srt/rust_extensions/loader.py:265`) and `RUSTFLAGS`/`CARGO_*`.
3. **Write pattern.**
   - An exclusive flock on `locks/<package>-<targetfp>.lock`
     (`vendor/sglang-v0.5.20/python/sglang/srt/rust_extensions/loader.py:136`, `:385`) is taken even
     before a cache hit (`vendor/sglang-v0.5.20/python/sglang/srt/rust_extensions/loader.py:139`).
   - cargo builds into `targets/<fp>`. The artifact is staged via `mkstemp`, fsync and `os.replace`
     (`vendor/sglang-v0.5.20/python/sglang/srt/rust_extensions/loader.py:456`).
4. **Relocatability.** Not analysed; this cache is unused on the platform.
5. **Read-only directory.** No: even a hit opens the lock (**inferred**). Mode `never` raises instead of
   building (`vendor/sglang-v0.5.20/python/sglang/srt/rust_extensions/loader.py:143`).
6. **Content.** `artifacts/<pkg>/<fp>/<module><EXT_SUFFIX>`
   (`vendor/sglang-v0.5.20/python/sglang/srt/rust_extensions/loader.py:377`), cargo `targets/` and `locks/`.

---

## 9. Other writers (not compile caches, or opt-in)

- **Run:ai streamer metadata.** `<SGLANG_CACHE_DIR>/model_streamer/<sha256(model_path)[:16]>`
  (`vendor/sglang-v0.5.20/python/sglang/srt/utils/runai_utils.py:122`, `:126`).
- **Qwen4 PLE file backend.** Tables live in `<SGLANG_CACHE_DIR>/ple`
  (`vendor/sglang-v0.5.20/python/sglang/srt/environ.py:311`) and are rewritten on every boot
  (`vendor/sglang-v0.5.20/python/sglang/srt/models/qwen4_exp_ple_table.py:236`). If `SGLANG_CACHE_DIR` is
  pointed at shared or read-only storage, this large per-boot write moves there too.
- **Kimi-K3 GGUF expert packs.** Stored in `$XDG_CACHE_HOME`, or `~/.cache`, under
  `sglang-expert-pack/kimi-k3/<fp>`. This does **not** follow `SGLANG_CACHE_DIR`
  (`vendor/sglang-v0.5.20/python/sglang/srt/model_loader/expert_pack_runtime.py:29`, `:38`). The key includes
  the absolute source directory, size and mtime
  (`vendor/sglang-v0.5.20/python/sglang/srt/model_loader/expert_pack_runtime.py:35`).
- **`torch.utils.cpp_extension.load` builds (opt-in).** These land in `TORCH_EXTENSIONS_DIR` or torch's
  default, which SGLang does not redirect. Users:
  - the experimental C++ radix tree
    (`vendor/sglang-v0.5.20/python/sglang/srt/mem_cache/cpp_radix_tree/radix_tree.py:10`, gated at
    `vendor/sglang-v0.5.20/python/sglang/srt/mem_cache/registry.py:104`);
  - HiCache native page hashing, used with storage or KV events
    (`vendor/sglang-v0.5.20/python/sglang/srt/mem_cache/cpp_utils/native_hash.py:34`);
  - the KDA PTX prefill, which is sm_103a only
    (`vendor/sglang-v0.5.20/python/sglang/kernels/ops/attention/linear/kda_ptx_prefill/__init__.py:51`).
- **Per-process scratch builds in `$TMPDIR` (opt-in).**
  - The symm-mem NCCL allocator builds in `/tmp/symm_allocator`
    (`vendor/sglang-v0.5.20/python/sglang/srt/distributed/device_communicators/pynccl_allocator.py:199`)
    and deletes a stale torch build lock first
    (`vendor/sglang-v0.5.20/python/sglang/srt/distributed/device_communicators/pynccl_allocator.py:205`).
  - Bump-arena stubs build in `/tmp/sgl_bump_arena/<pid>_<n>`
    (`vendor/sglang-v0.5.20/python/sglang/srt/utils/cuda_vmm_utils.py:450`, `:430`). The pid in the path
    means they are never reused. Graph-pool borrow, one of their users, is off by default
    (`vendor/sglang-v0.5.20/python/sglang/srt/environ.py:1385`).
- **Weight-download file locks.** Kept in the system temp directory
  (`vendor/sglang-v0.5.20/python/sglang/srt/model_loader/weight_utils.py:158`).
- multimodal_gen/diffusion: skipped, per the brief. Its own JIT loaders exist under `kernels/ops/diffusion`.

## 10. Caches SGLang does not redirect

The redirect covers exactly the four variables in §0. A grep finds no SGLang code that sets
`CUTE_DSL_*`, `TILELANG_*`, `TVM_FFI_CACHE_DIR`, `TORCH_EXTENSIONS_DIR`, `HF_*` cache variables,
`NUMBA_CACHE_DIR` or `XDG_CACHE_HOME`. The following libraries therefore keep their own defaults, mostly
under `$HOME`; see their topics:

- nvidia-cutlass-dsl and quack;
- tilelang, whose DSv4 MHC kernels are on by default;
- xgrammar and numba;
- pip flash-attn-4, used only with `SGLANG_INKLING_FA4_USE_PIP=1`;
- tvm-ffi's addon (§11);
- torch cpp extensions.

## 11. servekit's three variables on 0.5.20

- **What servekit sets.** It exports exactly `TRITON_CACHE_DIR`, `TVM_FFI_CACHE_DIR` and
  `FLASHINFER_WORKSPACE_BASE` (`vendor/servekit/src/servekit/jit_cache.py:30`). The values are under the
  node-local `/tmp/servekit/caches/<name>/{triton,tvm-ffi,flashinfer}`
  (`vendor/servekit/src/servekit/jit_cache.py:14`, `:17`, `:18`, `:21`).
- **Still honoured.**
  - `TRITON_CACHE_DIR` and `FLASHINFER_WORKSPACE_BASE`: SGLang only setdefaults them, so servekit's
    values win (`vendor/sglang-v0.5.20/python/sglang/srt/environ.py:1947`).
  - Exception: with `tc_piecewise`, SGLang hard-overrides `TRITON_CACHE_DIR`
    (`vendor/sglang-v0.5.20/python/sglang/srt/compilation/compiler_interface.py:201`).
- **`TVM_FFI_CACHE_DIR` is effectively dead for SGLang kernels.** It is still read in two places, but
  neither is SGLang's kernel path (§1):
  - tvm-ffi's cpp builders, only when no `build_directory` is passed
    (`vendor/src/apache-tvm-ffi-0.1.11/tvm_ffi/cpp/extension.py:681`, `:682`);
  - the optional torch-DLPack addon JIT
    (`vendor/src/apache-tvm-ffi-0.1.11/tvm_ffi/_optional_torch_c_dlpack.py:122`, `:137`), which runs at
    `import tvm_ffi` unless `TVM_FFI_DISABLE_TORCH_C_DLPACK` is set
    (`vendor/src/apache-tvm-ffi-0.1.11/tvm_ffi/_optional_torch_c_dlpack.py:206`).
  - DeepGEMM 0.2.0 ships `_C.so` and builds only if it is missing
    (`vendor/src/sgl-deep-gemm-0.2.0/deep_gemm/__init__.py:101`). That fallback build uses an explicit
    `build_directory` (`vendor/src/sgl-deep-gemm-0.2.0/deep_gemm/__init__.py:80`).
  - Whether the DLPack addon builds at all on torch 2.13 is **UNVERIFIED**. It depends on whether
    `torch.Tensor` already exposes `__dlpack_c_exchange_api__`, and the vendored torch is only a subset
    (`_inductor`, `_dynamo`, ...).
- **Missed on 0.5.20.** These stay under `$HOME/.cache/sglang/...` unless relocated:
  - `SGLANG_JIT_CACHE_DIR`, the direct successor of the tvm-ffi cache;
  - everything derived from `SGLANG_CACHE_DIR`: `inductor/`, `nv/`, `deep_gemm/` (via
    `SGLANG_DG_CACHE_DIR`), `cute_aot/` (fa4 on GH200), `flashinfer/autotune/`, the P2P JSON,
    `torch_compile_cache/` and `rust_extensions/`;
  - the §10 libraries.
- **Stale premise.** servekit assumes "ninja records absolute paths", so that a moved cache rebuilds
  (`vendor/servekit/src/servekit/jit_cache.py:12`). That is false for SGLang 0.5.20's own JIT cache,
  which is anchor-normalized and content-verified (§1.4).

## 12. UNVERIFIED items (why)

- **CI image mapping.** The mapping `sglang_base` → `images/sglang_0.5.20/Dockerfile` is not in the repo.
- **Host compiler.** Which `c++` resolves first in the platform image (conda or Ubuntu) is a runtime
  PATH question. It matters because its `--version` output is part of the JIT key.
- **Library internals not vendored in readable form:** the `kernels` (HF) package's storage location;
  CuTe DSL `load_module` and `export_to_c`; whether DeepGEMM's `_C.so` uses the lazy DeepJIT runtime
  with prefix `DG`; tvm-ffi `load_module` side effects; ninja's `.ninja_log`.
- **Missing torch sources.** The torch 2.13 `__dlpack_c_exchange_api__` attribute (tvm-ffi addon trigger)
  is not in the vendored torch subset.
- **Filesystem behaviour on capstor/Lustre:** flock support and its mount options, cross-node rename
  atomicity, and how umask and ACLs affect a shared multi-user directory.

## Summary table

| component | env var | default | key includes | write pattern | relocatable? | read-only-safe? |
|---|---|---|---|---|---|---|
| redirect | `SGLANG_CACHE_DIR` → TRITON/TORCHINDUCTOR/CUDA_CACHE_PATH/FLASHINFER_WORKSPACE_BASE | `~/.cache/sglang` (HOME at import) | n/a | `setdefault` at `import sglang`; inherited by spawn children | n/a | n/a |
| JIT kernels | `SGLANG_JIT_CACHE_DIR` | literal `~/.cache/sglang/jit` (not under SGLANG_CACHE_DIR) | module args, source + transitive dep sha256, normalized build.ninja (flags, sm_90a), nvcc and c++ `--version`, tvm-ffi/flashinfer/torch/mathdx versions | flock `.lock` (0644) + uuid staging + atomic dir rename; leaves immutable | yes (anchor-normalized) | hits yes (utime best-effort); misses raise |
| CuTe AOT (FA4) | `SGLANG_CUTE_AOT_CACHE_DIR` | `<SGLANG_CACHE_DIR>/cute_aot` | py ver, host, cutlass ver + CUDA_VERSION, arch, tvm-ffi ver, source content; pickled compile key | flock (sh/ex, 15 s poll) + tmp + `os.replace` | yes (relative source paths) | hits only if `.lock` files pre-exist; misses raise |
| DeepGEMM (SGLang side) | `SGLANG_DG_CACHE_DIR` → hard-sets `DG_JIT_CACHE_DIR` | `<SGLANG_CACHE_DIR>/deep_gemm` | (DeepGEMM's) | flock `locks/*.lock` around precompile | DeepGEMM topic | lock `open("w")` fails; off on platform (SGL_ENABLE_JIT_DEEPGEMM=0) except direct users |
| FA3 hub | `SGLANG_CACHE_DIR/kernels.lock` (only if SGLANG_USE_SGL_FA3_KERNEL=0) | none | repo + lock/revision | by `kernels` pkg (UNVERIFIED) | n/a | n/a (network fetch on miss) |
| FlashInfer autotune | `SGLANG_FLASHINFER_AUTOTUNE_CACHE` toggle | `<SGLANG_CACHE_DIR>/flashinfer/autotune/<ver>/smXY/<key>/rank_*.json` | model path, dtype, quant, backend, tp/pp/dp/ep, flashinfer ver, sm (+ FI metadata) | FI mkstemp + `os.replace`, save only if new tactics | no (model path in key) | yes if fully tuned; files 0600 (other UIDs fail) |
| P2P | (SGLANG_CACHE_DIR) | `gpu_p2p_access_cache_for_<CVD>.json` | CUDA_VISIBLE_DEVICES only | direct `open("w")`, no lock | n/a | yes once present; racy first write |
| torch.compile (tc_piecewise only) | hard-sets TORCHINDUCTOR/TRITON | `<SGLANG_CACHE_DIR>/torch_compile_cache/<md5>/rank_0_0/` | Inductor system + torch_key | in-place index + timestamped graph dumps | torch topic | no |
| Rust ext | `SGLANG_RUST_BUILD_MODE` | `<SGLANG_CACHE_DIR>/rust_extensions` | sources, features, rustc/cargo, py ABI | flock + mkstemp + `os.replace` | n/a | no (lock even on hit); unused with wheel |
