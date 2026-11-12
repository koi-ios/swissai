# FlashInfer, apache-tvm-ffi and TileLang: on-disk caches

Scope: the unpacked wheels under `vendor/src/` that the images ship: FlashInfer **0.6.18** (SGLang 0.5.20),
FlashInfer **0.7.0.post1** (vLLM 0.32.0), apache-tvm-ffi **0.1.11** and TileLang **0.1.12** (both engines).
Engine and launcher code that changes these caches comes from `vendor/sglang-v0.5.20`, `vendor/vllm-v0.32.0`,
`vendor/servekit` and `vendor/model-launch`. All paths are relative to the opportunity folder. When I cite only
0.6.18, the 0.7.0.post1 code is identical (checked with `diff`). Version differences are listed separately.

Short answer for #270:

* **FlashInfer**'s tree is already keyed by version and arch (`<base>/.cache/flashinfer/<version>/<arch>/`).
  But every process takes a file lock and runs `ninja` for every JIT module. A read-only cache is not supported.
* **tvm-ffi** does **not** put the GPU arch in its key. Its lock retries forever on any `OSError`, so a
  read-only directory, or a filesystem without `flock`, hangs the process instead of failing.
* **TileLang** has the best write pattern of the three: a staging directory, then an atomic `rename`. But
  nothing in either engine or in servekit sets `TILELANG_CACHE_DIR`.

---

## 1. FlashInfer JIT workspace (`cached_ops/`, `generated/`, `tmp/`, `flashinfer_jit.log`)

**1. Variable.**
- `FLASHINFER_WORKSPACE_BASE` replaces `$HOME` as the base. The default is `Path.home()`, with no XDG
  fallback. It is read **once at import** of `flashinfer.jit.env`:
  vendor/src/flashinfer-python-0.6.18/flashinfer/jit/env.py:59
  (0.7.0.post1: vendor/src/flashinfer-python-0.7.0.post1/flashinfer/jit/env.py:67).
- The cache root `FLASHINFER_CACHE_DIR = <base>/.cache/flashinfer` is a derived constant, not an env var:
  vendor/src/flashinfer-python-0.6.18/flashinfer/jit/env.py:62. The CLI's env table lists the name anyway
  (vendor/src/flashinfer-python-0.6.18/flashinfer/__main__.py:311). There is also no `FLASHINFER_JIT_DIR`
  env var.
- Sub-directories:
  - `cached_ops`: vendor/src/flashinfer-python-0.6.18/flashinfer/jit/env.py:162
  - `generated`: vendor/src/flashinfer-python-0.6.18/flashinfer/jit/env.py:163
  - the lock dir `cached_ops/tmp`: vendor/src/flashinfer-python-0.6.18/flashinfer/jit/core.py:601
- Default trees:

  | Engine | Default tree | Why |
  |---|---|---|
  | vLLM 0.32.0 | `$HOME/.cache/flashinfer/0.7.0.post1/<arch>/` | vLLM does not set the variable |
  | SGLang 0.5.20 | `~/.cache/sglang/.cache/flashinfer/0.6.18/<arch>/` | SGLang **setdefaults** the base to `SGLANG_CACHE_DIR` on `import sglang` (see §12) |

**2. Key (directory name).**
- The workspace path is `<cache>/<flashinfer version>/<arch>`:
  vendor/src/flashinfer-python-0.6.18/flashinfer/jit/env.py:157 and
  vendor/src/flashinfer-python-0.7.0.post1/flashinfer/jit/env.py:364.
- The version strings differ (`0.6.18` vs `0.7.0.post1`), so the two images never share a JIT tree.
- `<arch>` is the **sorted** set of target archs joined by `_`:
  vendor/src/flashinfer-python-0.6.18/flashinfer/jit/env.py:153-157.
  - The set comes from `FLASHINFER_CUDA_ARCH_LIST` when it is set
    (vendor/src/flashinfer-python-0.6.18/flashinfer/compilation_context.py:90). Otherwise it comes from
    **every visible device** (vendor/src/flashinfer-python-0.6.18/flashinfer/compilation_context.py:104).
  - SM 9.x gets an `a` suffix (vendor/src/flashinfer-python-0.6.18/flashinfer/compilation_context.py:73),
    so GH200 gives `90a`.
  - If no GPU is visible and the variable is unset, the error is swallowed
    (vendor/src/flashinfer-python-0.6.18/flashinfer/compilation_context.py:107-108). The arch component is
    then empty, so a CPU-only pre-warm writes `<cache>/<version>/` itself (derived from the code above).
- Inside `cached_ops/`, each module directory is the JitSpec `name`, which encodes dtypes, head dims and so
  on (vendor/src/flashinfer-python-0.6.18/flashinfer/jit/core.py:337-345).
- **Not** in any directory name: CUDA/nvcc version, `CUDA_HOME`, compiler, flags, Python version. Those are
  only text inside `build.ninja`: `cuda_home` (vendor/src/flashinfer-python-0.6.18/flashinfer/jit/cpp_ext.py:278)
  and the nvcc path (vendor/src/flashinfer-python-0.6.18/flashinfer/jit/cpp_ext.py:270).

**3. Write pattern and lock.**
- **Import side effects:**
  - `os.makedirs(WORKSPACE_DIR)`: vendor/src/flashinfer-python-0.6.18/flashinfer/jit/core.py:19.
  - A logging `FileHandler` on `<workspace>/flashinfer_jit.log`, created if missing:
    vendor/src/flashinfer-python-0.6.18/flashinfer/jit/core.py:48-53. The stdlib default `delay=False`
    opens the file immediately.
- **Generated sources** are written when the spec is created (for example
  vendor/src/flashinfer-python-0.6.18/flashinfer/jit/attention/modules.py:136-165). That is before
  `build_and_load()` takes any lock.
  - They are written with `write_if_different`, which compares and then truncate-writes **in place**:
    vendor/src/flashinfer-python-0.6.18/flashinfer/jit/utils.py:22-30, with `open(path, "w")` at line 29.
  - This is a cold-start race. Two processes can rewrite the same file while a third process's ninja reads
    it. Identical content is not rewritten, so a warm cache is unaffected.
- **Build:** `build_and_load()` takes a py-filelock `FileLock` on `cached_ops/tmp/<name>.lock`:
  - the lock: vendor/src/flashinfer-python-0.6.18/flashinfer/jit/core.py:305
  - the lock path: vendor/src/flashinfer-python-0.6.18/flashinfer/jit/core.py:376
  - the import: vendor/src/flashinfer-python-0.6.18/flashinfer/jit/core.py:12

  Under that lock it:
  1. rewrites `build.ninja` if different (vendor/src/flashinfer-python-0.6.18/flashinfer/jit/core.py:426);
  2. runs `ninja -C <module dir> -f <build.ninja>`
     (vendor/src/flashinfer-python-0.6.18/flashinfer/jit/cpp_ext.py:355-362 and :374);
  3. loads the `.so` **still under the lock**
     (vendor/src/flashinfer-python-0.6.18/flashinfer/jit/core.py:320-321 and :430).
- There is no temp-file-plus-rename step. nvcc, the linker and ninja write `.o`, `.so`, `.ninja_log` and
  `.ninja_deps` in place, in each module's own directory
  (comment at vendor/src/flashinfer-python-0.6.18/flashinfer/jit/cpp_ext.py:321-323).
- The lock files live **inside** the cache tree. The code itself flags NFS concerns:
  vendor/src/flashinfer-python-0.6.18/flashinfer/jit/core.py:600.
- A reader never validates the `.so`. Repair after a killed build depends on ninja's log and mtime logic
  (**UNVERIFIED**, see §13).

**4. Relocatability.**
- `build.ninja` contains absolute paths:
  - outputs: vendor/src/flashinfer-python-0.6.18/flashinfer/jit/cpp_ext.py:332 and :338
  - sources, including `generated/` in the cache: vendor/src/flashinfer-python-0.6.18/flashinfer/jit/cpp_ext.py:334
  - include dirs in site-packages and the tvm_ffi and Python headers:
    vendor/src/flashinfer-python-0.6.18/flashinfer/jit/cpp_ext.py:126-135
  - `cuda_home`: vendor/src/flashinfer-python-0.6.18/flashinfer/jit/cpp_ext.py:278
- In 0.7.0.post1 it also contains the absolute Python interpreter (embed_cubin rule,
  vendor/src/flashinfer-python-0.7.0.post1/flashinfer/jit/cpp_ext.py:310) and resolved cubin paths
  (vendor/src/flashinfer-python-0.7.0.post1/flashinfer/jit/cpp_ext.py:358).
- So moving the tree, or changing the site-packages or CUDA path, rewrites `build.ninja`, and ninja then
  rebuilds. The rebuild is ninja behaviour (**UNVERIFIED** here). servekit's comment agrees
  (vendor/servekit/src/servekit/jit_cache.py:10-13).
- The `.so` itself is loaded by absolute path and linked with only `-L` flags, no rpath
  (vendor/src/flashinfer-python-0.6.18/flashinfer/jit/cpp_ext.py:254-260).

**5. Hit/miss and read-only.**
- `try_load()` only trusts **AOT** artifacts. JIT freshness "is owned by ninja"
  (vendor/src/flashinfer-python-0.6.18/flashinfer/jit/core.py:396-410, comment at :398).
- So for every JIT module, each process does, on first use: lock, write-if-different `build.ninja`, run
  `ninja`, then `dlopen` (vendor/src/flashinfer-python-0.6.18/flashinfer/jit/core.py:300-321 and :412-427).
  Module getters are memoised per process (for example
  vendor/src/flashinfer-python-0.6.18/flashinfer/norm/__init__.py:77).
- **Read-only does not work:**
  - The import opens the log file for append (vendor/src/flashinfer-python-0.6.18/flashinfer/jit/core.py:53).
  - Every load needs to create or open `tmp/<name>.lock` (vendor/src/flashinfer-python-0.6.18/flashinfer/jit/core.py:305).
- `FLASHINFER_DISABLE_JIT=1` is **not** a read-only mode. Because `try_load()` ignores JIT outputs, the flag
  raises `MissingJITCacheError` for every non-AOT module, even when `cached_ops/<name>/<name>.so` exists
  (vendor/src/flashinfer-python-0.6.18/flashinfer/jit/core.py:301-318, flag at :311).

**6. Content classes.**
- `cached_ops/<name>/`: `build.ninja`, `*.o` / `*.cuda.o` with `.d` depfiles, `<name>.so`, `.ninja_log`,
  `.ninja_deps`.
- `cached_ops/tmp/`: `<name>.lock`, plus `flashinfer_jit.lock` and `flashinfer_jit.ninja` for batch builds
  (vendor/src/flashinfer-python-0.6.18/flashinfer/jit/core.py:629-633).
- `generated/<uri>/`: `*.cu`, `*.inc`.
- `flashinfer_jit.log`.
- The CuTe-DSL objects from §2.
- In 0.7.0.post1, tvm-ffi build directories for the `cake_*` modules (§6).

**Version differences (0.6.18 → 0.7.0.post1).**
- **AOT discovery:** a single `flashinfer_jit_cache.get_jit_cache_dir()`
  (vendor/src/flashinfer-python-0.6.18/flashinfer/jit/env.py:139) becomes multiple per-arch providers
  (vendor/src/flashinfer-python-0.7.0.post1/flashinfer/jit/env.py:184, :211-215, :269-301). There is
  heterogeneous dispatch with a lock-protected JIT fallback
  (vendor/src/flashinfer-python-0.7.0.post1/flashinfer/jit/core.py:339 and :542-545).
- **Link line:** nvcc link now gets the arch flags (vendor/src/flashinfer-python-0.7.0.post1/flashinfer/jit/cpp_ext.py:254).
- **New tvm-ffi builds** inside `cached_ops` (§6).
- **New managed autotune store** (§5).
- The workspace path and the lock scheme are unchanged.

## 2. FlashInfer CuTe-DSL kernel cache (inside `cached_ops/`)

1. **Variable:** follows `FLASHINFER_JIT_DIR`. `FLASHINFER_CUTE_DSL_DISABLE_CACHE=1` disables it
   (vendor/src/flashinfer-python-0.6.18/flashinfer/jit/cute_dsl_core.py:34).
2. **Key:**
   - The directory is `<module>_<arch>_cute_dsl`. The arch comes from `CUTE_DSL_ARCH`, or from the device
     (vendor/src/flashinfer-python-0.6.18/flashinfer/jit/cute_dsl_core.py:76-83 and :109-114).
   - `meta.json` holds the arch, every installed `nvidia-cutlass-dsl*` name and version, and the source hash
     (vendor/src/flashinfer-python-0.6.18/flashinfer/jit/cute_dsl_core.py:116-120).
   - The source hash covers each key file's **path string** and its bytes
     (vendor/src/flashinfer-python-0.6.18/flashinfer/jit/cute_dsl_core.py:59-65), so the install path is part
     of the key.
3. **Write:**
   - Takes a module-level FileLock on `tmp/<module_dir>.lock`
     (vendor/src/flashinfer-python-0.6.18/flashinfer/jit/cute_dsl_core.py:129).
   - Wipes a stale module (vendor/src/flashinfer-python-0.6.18/flashinfer/jit/cute_dsl_core.py:175).
   - Writes `.o.tmp.<pid>` then `os.replace` (vendor/src/flashinfer-python-0.6.18/flashinfer/jit/cute_dsl_core.py:231 and :237).
   - Writes `meta.json` last, also through tmp and replace, as the commit marker
     (vendor/src/flashinfer-python-0.6.18/flashinfer/jit/cute_dsl_core.py:238-245).
   - If persisting fails, it only logs (vendor/src/flashinfer-python-0.6.18/flashinfer/jit/cute_dsl_core.py:179-185).
4. **Relocatability:** the `.o` is loaded by path and the key has no cache path in it, but it does have the
   source path. Relocatable.
5. **Hit/miss:**
   - A hit needs the `.o` to exist and `meta.json` to match, and does not lock
     (vendor/src/flashinfer-python-0.6.18/flashinfer/jit/cute_dsl_core.py:145-160).
   - A load error gives a recompile.
   - On a read-only cache, hits work and misses fail at the lock.
   - 0.7.0.post1 reloads the exported artifact even right after a build
     (vendor/src/flashinfer-python-0.7.0.post1/flashinfer/jit/cute_dsl_core.py:218-238).
6. **Content:** `<kernel>.o`, `meta.json`, `tmp/*.lock`.

## 3. FlashInfer AOT and prebuilt kernels (`flashinfer-jit-cache`, `flashinfer-cubin`, package `data/aot`)

- **0.6.18:**
  - If `flashinfer_jit_cache` is importable, `FLASHINFER_AOT_DIR` is its directory, after a
    version-prefix check (vendor/src/flashinfer-python-0.6.18/flashinfer/jit/env.py:120-139). Otherwise it is
    the package's `data/aot` (vendor/src/flashinfer-python-0.6.18/flashinfer/jit/env.py:142).
  - A module is AOT if `<AOT_DIR>/<name>/<name>.so` exists
    (vendor/src/flashinfer-python-0.6.18/flashinfer/jit/core.py:364-368), and it is loaded first
    (vendor/src/flashinfer-python-0.6.18/flashinfer/jit/core.py:401-403).
- **0.7.0.post1:**
  - The package `data/aot` copy wins (vendor/src/flashinfer-python-0.7.0.post1/flashinfer/jit/env.py:271-273).
  - Otherwise the best arch-compatible provider is picked
    (vendor/src/flashinfer-python-0.7.0.post1/flashinfer/jit/env.py:275-301), with a version check per
    provider (vendor/src/flashinfer-python-0.7.0.post1/flashinfer/jit/env.py:142-172).
- These are read-only package directories, not caches. They are the only artifacts FlashInfer loads without
  ninja.
- **What the images ship:**
  - Upstream vLLM's Dockerfile runs `flashinfer download-kernels`, which installs `flashinfer-cubin` and
    `flashinfer-jit-cache` (vendor/vllm-v0.32.0/docker/Dockerfile:939-943).
  - The vLLM wheel itself does not depend on them (vendor/vllm-v0.32.0/setup.py:1499-1502).
  - Upstream SGLang's Dockerfile defaults to `INSTALL_FLASHINFER_JIT_CACHE=0`
    (vendor/sglang-v0.5.20/docker/Dockerfile:15) and always installs the cubin wheel
    (vendor/sglang-v0.5.20/docker/Dockerfile:284-291).
  - The platform's SGLang 0.5.20 image is a plain `pip install "sglang[all]==0.5.20"`, with the comment
    "flashinfer JIT-compiles into its cache" (vendor/model-launch/images/sglang_0.5.20/Dockerfile:66-71).
  - **UNVERIFIED:** whether the platform's vLLM 0.32.0 image contains the two wheels. No vLLM 0.32.0 image
    is defined in `vendor/model-launch/images`.

## 4. FlashInfer cubin and artifact download cache (`FLASHINFER_CUBIN_DIR`)

1. **Variable:**
   - Order of precedence:
     1. `FLASHINFER_CUBIN_DIR` (vendor/src/flashinfer-python-0.6.18/flashinfer/jit/env.py:74);
     2. the installed `flashinfer-cubin` package directory, which must match the version exactly
        (vendor/src/flashinfer-python-0.6.18/flashinfer/jit/env.py:85-104);
     3. `<FLASHINFER_CACHE_DIR>/cubins` (vendor/src/flashinfer-python-0.6.18/flashinfer/jit/env.py:107).
   - It is resolved **at import** (vendor/src/flashinfer-python-0.6.18/flashinfer/jit/env.py:110).
   - The source is `FLASHINFER_CUBINS_REPOSITORY`, default `edge.urm.nvidia.com` Artifactory
     (vendor/src/flashinfer-python-0.6.18/flashinfer/jit/cubin_loader.py:35-38). It can also be a local
     directory, which is copied instead of downloaded
     (vendor/src/flashinfer-python-0.6.18/flashinfer/jit/cubin_loader.py:89-94).
2. **Key:**
   - The directory is **not** versioned or arch-keyed. Files sit at their Artifactory relative path, which
     starts with a pinned commit hash (vendor/src/flashinfer-python-0.6.18/flashinfer/artifacts.py:140-150).
     Each file is checked against a sha256.
   - The pins differ between versions (for example
     vendor/src/flashinfer-python-0.7.0.post1/flashinfer/artifacts.py:142), but `TRTLLM_GEN_FMHA` is the
     same in both. Sharing one cubin directory across both images is therefore safe.
3. **Write:**
   - A per-file `<dest>.lock` (py-filelock, 30 s timeout):
     vendor/src/flashinfer-python-0.6.18/flashinfer/jit/cubin_loader.py:79-80.
   - A temp file `<dest>.<uuid>.tmp`: vendor/src/flashinfer-python-0.6.18/flashinfer/jit/cubin_loader.py:86.
   - `os.replace`: vendor/src/flashinfer-python-0.6.18/flashinfer/jit/cubin_loader.py:112.
   - A lock timeout returns False (vendor/src/flashinfer-python-0.6.18/flashinfer/jit/cubin_loader.py:141-145),
     and the caller then gets empty bytes.
4. **Relocatable:** yes. Two notes:
   - `ensure_symlink` creates absolute symlinks from generated include dirs into this directory, and recreates
     them when the target changes (vendor/src/flashinfer-python-0.6.18/flashinfer/jit/cubin_loader.py:240-264).
   - The bulk `get_checksums` always re-downloads `checksums.txt`
     (vendor/src/flashinfer-python-0.6.18/flashinfer/artifacts.py:214).
5. **Hit/miss:**
   - A hit is a local read plus a sha256 check (vendor/src/flashinfer-python-0.6.18/flashinfer/jit/cubin_loader.py:188-204
     and :216-219). `FLASHINFER_CUBIN_CHECKSUM_DISABLED` skips the check
     (vendor/src/flashinfer-python-0.6.18/flashinfer/jit/cubin_loader.py:192).
   - The hit path takes no lock and does no write, so it is **read-only safe**.
   - A miss needs a writable directory and network access, unless `FLASHINFER_NO_DOWNLOAD` is set, which
     raises instead (vendor/src/flashinfer-python-0.6.18/flashinfer/jit/cubin_loader.py:221-232).
   - A half-written file is never visible, and a corrupt file fails sha256 and is downloaded again.
   - `flashinfer download-cubin` fills the directory with checksum verification
     (vendor/src/flashinfer-python-0.6.18/flashinfer/__main__.py:445-448;
     vendor/src/flashinfer-python-0.6.18/flashinfer/artifacts.py:297-335).
6. **Content:** `*.cubin`, `include/*.h`, `checksums.txt`, DSL-FMHA `.so` per sm arch, `*.lock`.

## 5. FlashInfer autotuner persistence

**v1 JSON (`autotune(cache=path)`, both versions).**
- **Variable:** none. There is no default file; the engine passes the path. The file is loaded on entry if
  it exists (vendor/src/flashinfer-python-0.6.18/flashinfer/autotuner/autotuner.py:927-936) and saved on
  exit if tuning added entries (vendor/src/flashinfer-python-0.6.18/flashinfer/autotuner/autotuner.py:1002-1003).
- **Key:**
  - Each entry is keyed by `str((custom_op, runner_class_name, nearest_profile, extras))`
    (vendor/src/flashinfer-python-0.6.18/flashinfer/autotuner/autotuner.py:1111-1115).
  - The whole file carries `_metadata`: flashinfer version, torch CUDA version, cuBLAS, cuDNN backend and
    frontend versions, GPU name (vendor/src/flashinfer-python-0.6.18/flashinfer/autotuner/autotuner.py:323-342).
  - A mismatch makes the whole file be ignored
    (vendor/src/flashinfer-python-0.6.18/flashinfer/autotuner/autotuner.py:2729-2762). A definite conflict
    is never overwritten (vendor/src/flashinfer-python-0.6.18/flashinfer/autotuner/autotuner.py:2561-2577).
- **Write:**
  - Re-reads the file and merges (vendor/src/flashinfer-python-0.6.18/flashinfer/autotuner/autotuner.py:2540-2586).
  - Writes with `mkstemp` in the same directory, then `os.replace`
    (vendor/src/flashinfer-python-0.6.18/flashinfer/autotuner/autotuner.py:2619-2639).
  - **No lock.** The docstring calls the feature experimental: "concurrent readers" are supported, and
    ordinary records keep merge semantics
    (vendor/src/flashinfer-python-0.6.18/flashinfer/autotuner/autotuner.py:797-800). The last writer wins
    after the merge.
- **Relocatable:** yes; it is plain JSON.
- **Read-only:** loading works. A save would raise from `mkstemp`, but it only runs when something new was
  tuned.
- **Where the engines put it:**
  - SGLang: `$SGLANG_CACHE_DIR/flashinfer/autotune/<fi version>/sm<maj><min>/<sha16>/rank_tp*_pp*_dp*.json`.
    The hash covers the **model path**, dtype, quantisation, MoE backend, parallel sizes, HF class and
    skip_ops (vendor/sglang-v0.5.20/python/sglang/srt/model_executor/runner/flashinfer_autotune.py:136-174,
    passed as `cache=` at :284).
  - vLLM: `$VLLM_CACHE_ROOT/flashinfer_autotune_cache/<fi version>/<arch>/<sha256(config hash)>/autotune_configs_dp{dp}_rank{r}.json`,
    or under `VLLM_FLASHINFER_AUTOTUNE_CACHE_DIR`
    (vendor/vllm-v0.32.0/vllm/model_executor/warmup/flashinfer_autotune_cache.py:38-55;
    vendor/vllm-v0.32.0/vllm/model_executor/warmup/kernel_warmup.py:453-458, load at :468, save at :503-504).

**v2 managed store (0.7.0.post1 only).**
- **Variable:** `FLASHINFER_AUTOTUNE_CACHE_DIR`. The default is `<FLASHINFER_CACHE_DIR>/autotune`
  (vendor/src/flashinfer-python-0.7.0.post1/flashinfer/autotune_cache.py:160-167).
- **Layout:** `<root>/v2/<env_hash16>/{manifest.json, entries/<sha256(file_key)[:24]>.json}`
  (vendor/src/flashinfer-python-0.7.0.post1/flashinfer/autotune_cache.py:15-17 and :269-286).
- **Key:** the environment hash is the sha256 of the v1 metadata plus the measurement policy
  (vendor/src/flashinfer-python-0.7.0.post1/flashinfer/autotune_cache.py:682-689).
- **Write:** `mkstemp` then `os.replace`, "No locks"
  (vendor/src/flashinfer-python-0.7.0.post1/flashinfer/autotune_cache.py:24-26 and :242-252).
- **Read:** a missing, malformed or key-mismatched entry counts as a miss
  (vendor/src/flashinfer-python-0.7.0.post1/flashinfer/autotune_cache.py:295-326). A failed publish is only
  logged (vendor/src/flashinfer-python-0.7.0.post1/flashinfer/autotune_cache.py:359-364).
- **Use:** `autotune_v2` is called by **neither** engine (grep over both trees), so in practice nothing
  writes this store.

**Side files.**
- `FLASHINFER_AUTOTUNE_DIR` defaults to `<WORKSPACE_DIR>/autotune/*.json`. It is used for SM120 sparse-MLA
  (vendor/src/flashinfer-python-0.6.18/flashinfer/mla/_sparse_mla_sm120.py:1036-1047) and the 0.7.0.post1
  PCIe IPC allreduce (vendor/src/flashinfer-python-0.7.0.post1/flashinfer/comm/pcie_ipc_tuning.py:256-271).
- The MoE-EP knob cache defaults to `~/.cache/flashinfer/moe_ep_knob_cache.json`, which **ignores**
  `FLASHINFER_WORKSPACE_BASE`. It can be moved with `FLASHINFER_MOE_EP_KNOB_CACHE`
  (vendor/src/flashinfer-python-0.6.18/flashinfer/moe_ep/kernel_src/cutedsl_megamoe/shim/knob_cache.py:60-66).
- The block-sparse CuTe-DSL cache is opt-in with `FLASH_ATTENTION_CUTE_DSL_CACHE_ENABLED=1`. Its default
  is `$TMPDIR/<user>/flash_attention_cute_dsl_cache`
  (vendor/src/flashinfer-python-0.6.18/flashinfer/cute_dsl/sparse/bsa_utils/cache_utils.py:44-63).

## 6. apache-tvm-ffi 0.1.11 C++/CUDA JIT build cache (`tvm_ffi.cpp.build` / `build_inline` / `load` / `load_inline`)

1. **Variable:**
   - `TVM_FFI_CACHE_DIR`, default `~/.cache/tvm-ffi` (expanded from `$HOME`; no XDG).
   - It is read **at call time**, and only when `build_directory` is None
     (vendor/src/apache-tvm-ffi-0.1.11/tvm_ffi/cpp/extension.py:681-695 and :932-946).
   - Neither engine sets it. servekit does (vendor/servekit/src/servekit/jit_cache.py:18).
2. **Key:**
   - The directory is `<cache>/<name>_<sha256[:16]>`
     (vendor/src/apache-tvm-ffi-0.1.11/tvm_ffi/cpp/extension.py:946 and :117).
   - For `build_inline`, the hash covers the decorated C++/CUDA source text, the exported-function map,
     the extra cflags, CUDA cflags and ldflags, the extra include-path strings and the `embed_cubin` bytes
     (vendor/src/apache-tvm-ffi-0.1.11/tvm_ffi/cpp/extension.py:934-945 and :102-115).
   - For `build`, it covers the **resolved source paths, not their contents**, plus the flags
     (vendor/src/apache-tvm-ffi-0.1.11/tvm_ffi/cpp/extension.py:670 and :683-694).
   - **Not hashed:**
     - the GPU arch: `-gencode` comes from `TVM_FFI_CUDA_ARCH_LIST`, or from the first GPU through
       `nvidia-smi` (vendor/src/apache-tvm-ffi-0.1.11/tvm_ffi/cpp/extension.py:161-181), and is added only
       inside the ninja text (vendor/src/apache-tvm-ffi-0.1.11/tvm_ffi/cpp/extension.py:409);
     - nvcc location and version (vendor/src/apache-tvm-ffi-0.1.11/tvm_ffi/cpp/extension.py:444);
     - `CXX` (vendor/src/apache-tvm-ffi-0.1.11/tvm_ffi/cpp/extension.py:435);
     - tvm-ffi version and paths (vendor/src/apache-tvm-ffi-0.1.11/tvm_ffi/cpp/extension.py:365-367 and :396);
     - the Python version.
3. **Write and lock:**
   - Takes `FileLock(<build_dir>/lock)` (vendor/src/apache-tvm-ffi-0.1.11/tvm_ffi/cpp/extension.py:954 and :725).
     This is tvm-ffi's own lock: `os.open(O_WRONLY|O_CREAT)` plus a non-blocking `fcntl.flock`
     (vendor/src/apache-tvm-ffi-0.1.11/tvm_ffi/utils/lockfile.py:96-97).
   - **Any `OSError` is treated as "busy"** (vendor/src/apache-tvm-ffi-0.1.11/tvm_ffi/utils/lockfile.py:99-103).
     The `with` form polls every 0.1 s with no timeout
     (vendor/src/apache-tvm-ffi-0.1.11/tvm_ffi/utils/lockfile.py:60-66 and :136-146).
   - Under the lock, `_maybe_write` truncate-writes `main.cpp`, `cuda.cu` and `build.ninja` in place when the
     content differs (vendor/src/apache-tvm-ffi-0.1.11/tvm_ffi/cpp/extension.py:120-129, :956-958 and :727).
     Then it runs `ninja -v` with cwd set to the build dir
     (vendor/src/apache-tvm-ffi-0.1.11/tvm_ffi/cpp/extension.py:585-592).
   - Outputs have relative names (`cpp_0.o`, `cuda_0.o`, `<name>.so`) and are written in place, with no
     rename (vendor/src/apache-tvm-ffi-0.1.11/tvm_ffi/cpp/extension.py:504-528 and :354).
4. **Relocatability:** absolute paths in `build.ninja`:
   - the source file (vendor/src/apache-tvm-ffi-0.1.11/tvm_ffi/cpp/extension.py:951-952);
   - include paths (vendor/src/apache-tvm-ffi-0.1.11/tvm_ffi/cpp/extension.py:365 and :420-422);
   - `-L<tvm_ffi lib dir>` (vendor/src/apache-tvm-ffi-0.1.11/tvm_ffi/cpp/extension.py:396);
   - nvcc (vendor/src/apache-tvm-ffi-0.1.11/tvm_ffi/cpp/extension.py:444);
   - `sys.executable` for `embed_cubin` (vendor/src/apache-tvm-ffi-0.1.11/tvm_ffi/cpp/extension.py:492).

   After a move the directory names stay the same for `build_inline`, but the command lines change, so ninja
   rebuilds (**UNVERIFIED**, ninja).
5. **Hit/miss:**
   - There is no fast path. Every call locks, writes if different, runs ninja and returns the path
     (vendor/src/apache-tvm-ffi-0.1.11/tvm_ffi/cpp/extension.py:725-749 and :954-977).
   - `load_inline` and `load` call `load_module` **after** the lock is released
     (vendor/src/apache-tvm-ffi-0.1.11/tvm_ffi/cpp/extension.py:1122-1137).
   - A read-only directory makes the lock fail forever, so the process **hangs**; it does not raise.
   - Because the key has no arch, two GPU archs sharing a directory flip `build.ninja` back and forth and
     rebuild each time. A process can `dlopen` a `.so` that another process is relinking.
6. **Content:** `<name>_<hash>/{main.cpp, cuda.cu, build.ninja, *.o, *.o.d, <name>.so, lock, <cubin>.cubin, unified*.o, .ninja_log, .ninja_deps}`.

**Who uses it in these images:**
- FlashInfer 0.7.0.post1's `cake_*` modules pass `build_directory=FLASHINFER_JIT_DIR/<module>`, whose name
  includes source hashes and the arch
  (vendor/src/flashinfer-python-0.7.0.post1/flashinfer/cake_vsa.py:200-204 and :239). They get the tvm-ffi
  lock semantics inside the FlashInfer tree.
- sgl-deep-gemm builds into its own package directory
  (vendor/src/sgl-deep-gemm-0.2.0/deep_gemm/__init__.py:80-90).
- SGLang 0.5.20's own JIT writes its own `build.ninja` instead of calling `load_inline`
  (vendor/sglang-v0.5.20/python/sglang/kernels/jit/utils/compile/toolchain.py:3-4).
- No vLLM code imports `tvm_ffi.cpp` (grep).
- So the **default** `TVM_FFI_CACHE_DIR` is used in practice only by §7.

## 7. tvm-ffi torch-c-dlpack add-on (also `TVM_FFI_CACHE_DIR`, at import)

1. **When it runs:**
   - On `import tvm_ffi` (vendor/src/apache-tvm-ffi-0.1.11/tvm_ffi/__init__.py:91), unless
     `TVM_FFI_DISABLE_TORCH_C_DLPACK` is not `"0"`
     (vendor/src/apache-tvm-ffi-0.1.11/tvm_ffi/_optional_torch_c_dlpack.py:206-207).
   - It is skipped when torch already exposes `__dlpack_c_exchange_api__`
     (vendor/src/apache-tvm-ffi-0.1.11/tvm_ffi/_optional_torch_c_dlpack.py:98-102), or when
     `torch_c_dlpack_ext` imports (vendor/src/apache-tvm-ffi-0.1.11/tvm_ffi/_optional_torch_c_dlpack.py:107-111).
   - **UNVERIFIED** for torch 2.13: those torch files are not in the sparse checkout.
2. **Key:** the file `<cache>/libtorch_c_dlpack_addon_torch<maj><min>-<cuda|rocm|cpu>.so`
   (vendor/src/apache-tvm-ffi-0.1.11/tvm_ffi/_optional_torch_c_dlpack.py:122-138). Only torch
   `major.minor` and the device kind are in the name.
3. **Write:** a build subprocess (vendor/src/apache-tvm-ffi-0.1.11/tvm_ffi/_optional_torch_c_dlpack.py:139-165):
   - takes `FileLock(<cache>/<lib>.lock)` (vendor/src/apache-tvm-ffi-0.1.11/tvm_ffi/utils/_build_optional_torch_c_dlpack.py:801);
   - builds in a `mkdtemp` directory (vendor/src/apache-tvm-ffi-0.1.11/tvm_ffi/utils/_build_optional_torch_c_dlpack.py:774);
   - then does `shutil.move` into the cache (vendor/src/apache-tvm-ffi-0.1.11/tvm_ffi/utils/_build_optional_torch_c_dlpack.py:908).
     Across filesystems that is a copy, so it is not atomic.
4. **Relocatable:** yes (a single `.so`).
5. **Hit/miss:**
   - A hit is an existence check with no lock, then `ctypes.CDLL`
     (vendor/src/apache-tvm-ffi-0.1.11/tvm_ffi/_optional_torch_c_dlpack.py:139 and :169).
   - Build errors become a warning (vendor/src/apache-tvm-ffi-0.1.11/tvm_ffi/_optional_torch_c_dlpack.py:180-184).
   - A read-only cache that already exists hangs the build subprocess on the lock (§6). `subprocess.run`
     has no timeout, so this only matters if the add-on is needed at all.
6. **Content:** `libtorch_c_dlpack_addon_*.so` and `.lock`.

## 8. TileLang 0.1.12 kernel cache (`<cache>/<version>/<platform>-<machine>/kernels/<key>/`)

1. **Variable:**
   - `TILELANG_CACHE_DIR`, default `~/.tilelang/cache`. The default is expanded at import; no XDG
     (vendor/src/tilelang-0.1.12/tilelang/env.py:337).
   - `TILELANG_TMP_DIR`, default `<TILELANG_CACHE_DIR>/tmp` (vendor/src/tilelang-0.1.12/tilelang/env.py:338).
   - Both are read **live from `os.environ`** on each access (vendor/src/tilelang-0.1.12/tilelang/env.py:267-272).
   - Disable switches:
     - `TILELANG_DISABLE_CACHE` (vendor/src/tilelang-0.1.12/tilelang/env.py:342-344 and :389-399);
     - `TILELANG_KERNEL_CACHE_USE_LIB_STAMP=1` adds a hash of the native libraries to the key
       (vendor/src/tilelang-0.1.12/tilelang/env.py:345-347;
       vendor/src/tilelang-0.1.12/tilelang/cache/kernel_cache.py:139-142).
   - **Nobody sets `TILELANG_CACHE_DIR`:** not vLLM, not SGLang (grep), and not servekit
     (vendor/servekit/src/servekit/jit_cache.py:16-22).
2. **Key:**
   - The namespace is `<version>/<sys.platform>-<machine>`, for example `0.1.12/linux-aarch64`
     (vendor/src/tilelang-0.1.12/tilelang/cache/kernel_cache.py:164-171).
   - The entry name is the sha256 of a sorted JSON of:
     - the sha256 of `func.script(show_meta=True)`, the TVMScript of the elaborated PrimFunc with its static
       shapes and constants (vendor/src/tilelang-0.1.12/tilelang/cache/kernel_cache.py:267-269);
     - `out_idx` and `args_repr`, which is empty when called from `tilelang.compile`
       (vendor/src/tilelang-0.1.12/tilelang/jit/__init__.py:161-170);
     - `str(target)` (vendor/src/tilelang-0.1.12/tilelang/cache/kernel_cache.py:272);
     - `target_host`, the backend, `pass_configs` and `compile_flags`;
     - the tilelang version, plus the torch version on macOS only
       (vendor/src/tilelang-0.1.12/tilelang/cache/kernel_cache.py:136-147 and :241-282).
   - The "auto" target is normalised to `Target({"kind":"cuda","arch":"sm_XX[a]"})` from the current torch
     device (vendor/src/tilelang-0.1.12/tilelang/cuda/target.py:29-45;
     vendor/src/tilelang-0.1.12/tilelang/contrib/nvcc.py:523-531), so the arch is in the key.
   - **Not in the key:** nvcc/CUDA version, driver, host compiler, Python version.
3. **Write:**
   - The cache singletons are built at `import tilelang`
     (vendor/src/tilelang-0.1.12/tilelang/__init__.py:186 → vendor/src/tilelang-0.1.12/tilelang/jit/__init__.py:27
     → vendor/src/tilelang-0.1.12/tilelang/cache/__init__.py:23-29;
     vendor/src/tilelang-0.1.12/tilelang/cache/kernel_cache.py:180-188).
     - `_create_dirs` makes the cache, tmp, namespace, `kernels/` and `.staging/` directories.
     - It also deletes `.staging` entries older than 1 h
       (vendor/src/tilelang-0.1.12/tilelang/cache/kernel_cache.py:191-203 and :217-239).
   - On a miss, after compiling, still under an **in-process `threading.Lock` only**
     (vendor/src/tilelang-0.1.12/tilelang/cache/kernel_cache.py:44 and :404-409):
     1. Skip if a complete entry already exists (vendor/src/tilelang-0.1.12/tilelang/cache/kernel_cache.py:496-498).
     2. `os.makedirs(<ns>/.staging/<key>_<pid>_<rand8>)`, **outside the try**
        (vendor/src/tilelang-0.1.12/tilelang/cache/kernel_cache.py:502-506).
     3. Write each file to `TILELANG_TMP_DIR/<pid>_<uuid>`, then `os.replace` it into staging
        (vendor/src/tilelang-0.1.12/tilelang/cache/kernel_cache.py:460-473). The `executable.so` comes from
        `export_library` (vendor/src/tilelang-0.1.12/tilelang/jit/adapter/kernel_cache.py:8 and :25-30).
     4. Delete an incomplete final directory, then `os.rename(staging, final)`
        (vendor/src/tilelang-0.1.12/tilelang/cache/kernel_cache.py:535-539).
   - Losing a rename race (EEXIST/ENOTEMPTY) discards the staging directory
     (vendor/src/tilelang-0.1.12/tilelang/cache/kernel_cache.py:540-544 and :699-701). Any other error is
     logged and the entry is just not saved (vendor/src/tilelang-0.1.12/tilelang/cache/kernel_cache.py:545-547).
   - There is no cross-process lock. Concurrent misses compile redundantly, and the first rename wins.
   - `TILELANG_TMP_DIR` must be on the same filesystem as the cache: a cross-device `os.replace` fails.
4. **Relocatability:** entries are content-hashed, and the `.so` is loaded by the entry's path.
   **UNVERIFIED:** whether `export_library` bakes rpaths into `executable.so` (TVM C++, not vendored).
5. **Hit/miss:**
   - Lookup is the in-memory dict, then disk, with the disk read outside the lock
     (vendor/src/tilelang-0.1.12/tilelang/cache/kernel_cache.py:352-370).
   - A hit needs `device_kernel.cu`, `host_kernel.cu`, `executable.so` and `params.pkl`; otherwise it is a
     miss (vendor/src/tilelang-0.1.12/tilelang/cache/kernel_cache.py:583-587 and :671-691).
   - Loading is `runtime.load_module(executable.so)` (vendor/src/tilelang-0.1.12/tilelang/jit/adapter/tvm_ffi.py:334).
   - A load exception is treated as a miss and the entry is removed with `rmtree(ignore_errors)`
     (vendor/src/tilelang-0.1.12/tilelang/cache/kernel_cache.py:613-621).
   - **A corrupt `params.pkl` is never repaired.** It is only logged
     (vendor/src/tilelang-0.1.12/tilelang/cache/kernel_cache.py:589-597), and `_build_kernel` returns None
     (vendor/src/tilelang-0.1.12/tilelang/cache/kernel_cache.py:739-746). The save path then sees a
     "complete" directory and returns (vendor/src/tilelang-0.1.12/tilelang/cache/kernel_cache.py:496-498),
     so every process recompiles that kernel.
   - **Read-only:** import and hits are pure reads, provided the directories already exist. A **miss
     raises** at `os.makedirs(staging_path)` (vendor/src/tilelang-0.1.12/tilelang/cache/kernel_cache.py:506),
     and earlier at the cubin cache (§9).
6. **Content:** `device_kernel.cu`, `host_kernel.cu`, `executable.so` (`kernel_lib.so` for other backends),
   `params.pkl` (cloudpickle), `resource_usage.json` (HIP only); plus `.staging/` and `tmp/`.

## 9. TileLang CUDA binary cache (`<cache>/<version>/cuda-binaries/<sha256>.cubin|fatbin`)

1. **Variable:** same as §8. There is no platform or machine component; the file calls itself a
   "Cross-host" cache (vendor/src/tilelang-0.1.12/tilelang/cache/cuda_binary_cache.py:1, :20 and :37-44).
2. **Key:** tilelang version, sha256 of the CUDA source, target kind, arch, code list, format and the nvcc
   options (vendor/src/tilelang-0.1.12/tilelang/cache/cuda_binary_cache.py:104-112).
   - The options include **absolute** `-I<TileLang src>` and `-I<cutlass>` paths
     (vendor/src/tilelang-0.1.12/tilelang/engine/lower.py:120-126), plus fast-math and ptxas flags
     (vendor/src/tilelang-0.1.12/tilelang/engine/lower.py:144-147).
   - The nvcc version is not in the key.
3. **Write:** called directly from the nvcc compile callback, with load at
   vendor/src/tilelang-0.1.12/tilelang/engine/lower.py:162 and save at
   vendor/src/tilelang-0.1.12/tilelang/engine/lower.py:173. The save is `makedirs`, a temp file in
   `TILELANG_TMP_DIR`, then `os.replace`, with no lock and **no try**
   (vendor/src/tilelang-0.1.12/tilelang/cache/cuda_binary_cache.py:137-148). So an unwritable cache, or a
   cross-filesystem `TILELANG_TMP_DIR`, fails the compile.
4. **Relocatable:** the file name is a hash. A different TileLang install path changes the key, but not the
   directory.
5. **Hit/miss:** a hit opens and reads the file; `FileNotFoundError` is a miss, and nothing is validated
   (vendor/src/tilelang-0.1.12/tilelang/cache/cuda_binary_cache.py:126-134). It is disabled by
   `TILELANG_DISABLE_CACHE`.
6. **Content:** `.cubin` or `.fatbin`.

## 10. TileLang autotuner cache (`<cache>/<version>/<platform>-<machine>/autotuner/<key>/`)

1. **Variable:** the namespace root plus `autotuner` (vendor/src/tilelang-0.1.12/tilelang/autotuner/tuner.py:254-262).
   It is turned off by `TILELANG_AUTO_TUNING_DISABLE_CACHE` or `TILELANG_DISABLE_CACHE`
   (vendor/src/tilelang-0.1.12/tilelang/env.py:360; vendor/src/tilelang-0.1.12/tilelang/autotuner/tuner.py:940).
2. **Key:** the sha256 of:
   - the version;
   - parameter defaults and kernel parameters;
   - closure variables;
   - `inspect.getsource(fn)`;
   - the configs;
   - stable sha256-based hashes of the compile args (normalised target with arch, backend, pass configs)
     and the profile args
     (vendor/src/tilelang-0.1.12/tilelang/autotuner/tuner.py:318 and :444-468;
     vendor/src/tilelang-0.1.12/tilelang/autotuner/param.py:76-87 and :128-141).
3. **Write:** a staging directory under `.staging`, created **outside the try**, filled with tmp-and-replace
   files, then `os.rename` (vendor/src/tilelang-0.1.12/tilelang/autotuner/param.py:171-181, :400-401 and :459-465).
4. **Relocatable:** yes. `function.pkl` is a cloudpickled PrimFunc.
5. **Hit/miss:** memory, then disk, under a thread lock
   (vendor/src/tilelang-0.1.12/tilelang/autotuner/tuner.py:939-958). The reads of `best_config.json`,
   `function.pkl` and `latency.json` are not guarded, so a damaged entry raises
   (vendor/src/tilelang-0.1.12/tilelang/autotuner/param.py:487-509).
6. **Content:** `best_config.json`, `function.pkl`, `out_idx.json`, `latency.json`, plus the kernel files
   from §8 (vendor/src/tilelang-0.1.12/tilelang/autotuner/param.py:28-40).

Not used by either engine: grep finds no `tilelang.autotune`.

## 11. TileLang temp dirs, and shape specialisation (the vLLM DeepSeek-V4.1 question)

**Temp dirs.**
- nvcc runs in a TVM `tempdir`. With `TILELANG_CLEANUP_TEMP_FILES=1` (the 0.1.12 default,
  vendor/src/tilelang-0.1.12/tilelang/env.py:348-350), that is `tempfile.mkdtemp()`, which follows
  `$TMPDIR` (vendor/src/tilelang-0.1.12/tilelang/contrib/nvcc.py:137;
  vendor/src/tilelang-0.1.12/tilelang/3rdparty/tvm/python/tvm/contrib/utils.py:121).
- With `0`, it is the shared `/tmp/tvm-debug-mode-tempdirs`
  (vendor/src/tilelang-0.1.12/tilelang/3rdparty/tvm/python/tvm/contrib/utils.py:69).
- vLLM's `setdefault("TILELANG_CLEANUP_TEMP_FILES","1")` (vendor/vllm-v0.32.0/vllm/env_override.py:178-183)
  therefore changes nothing for 0.1.12.

**Shapes in the key.**
- The in-memory key is `parse_args`, which is checked before compiling
  (vendor/src/tilelang-0.1.12/tilelang/jit/__init__.py:526-530). It has two parts:
  - **phase 1:** all non-tensor (compile-time) arguments; tensors are excluded
    (vendor/src/tilelang-0.1.12/tilelang/language/eager/builder.py:1242-1248);
  - **phase 2:** the values of `T.const` dims or strides taken from the tensors
    (vendor/src/tilelang-0.1.12/tilelang/language/eager/builder.py:1074-1095).
- `T.dynamic` dims are plain symbolic `tirx.Var`s and are in neither key
  (vendor/src/tilelang-0.1.12/tilelang/language/symbolics.py:13-30).
- The disk key hashes the specialised PrimFunc script (§8).
- Result: one compile per distinct Python-int argument or `T.const` value. A `T.dynamic` batch dimension
  is **not** recompiled per batch size.
- vLLM's mHC kernels declare `num_tokens = T.dynamic(...)`
  (vendor/vllm-v0.32.0/vllm/model_executor/kernels/mhc/tilelang_kernels.py:151). They do pick tile and
  split parameters from `num_tokens`, but in coarse buckets
  (vendor/vllm-v0.32.0/vllm/model_executor/kernels/mhc/tilelang_kernels.py:85-88 and :1147-1163). So it is
  a handful of variants, not one per captured graph.
- Even with a warm disk cache, every process still rebuilds the TIR and prints the script just to compute
  the key (vendor/src/tilelang-0.1.12/tilelang/jit/__init__.py:356-368;
  vendor/src/tilelang-0.1.12/tilelang/cache/kernel_cache.py:267).
- **UNVERIFIED:** how the ~40 min splits between nvcc and host link (which a warm cache removes) and
  lowering (which it does not). That needs a profile.

## 12. Engine and launcher overrides

**SGLang 0.5.20.**
- On `import sglang` (vendor/sglang-v0.5.20/python/sglang/__init__.py:10-14) it setdefaults
  `FLASHINFER_WORKSPACE_BASE` to `SGLANG_CACHE_DIR`, default `~/.cache/sglang`
  (vendor/sglang-v0.5.20/python/sglang/srt/environ.py:1923-1947 and :1164).
- It warns that this only works before FlashInfer is imported
  (vendor/sglang-v0.5.20/python/sglang/srt/environ.py:1939-1944).
- The model runner sets `FLASHINFER_CUDA_ARCH_LIST` to the current device's arch, for example `9.0a`
  (vendor/sglang-v0.5.20/python/sglang/srt/utils/common.py:1042-1048). Since a lettered minor is kept as-is
  (vendor/src/flashinfer-python-0.6.18/flashinfer/compilation_context.py:96-97), GH200 still lands in `90a`.
- It does not set `TVM_FFI_CACHE_DIR` or `TILELANG_CACHE_DIR`.

**vLLM 0.32.0.**
- Sets only `TILELANG_CLEANUP_TEMP_FILES` (above) and the autotune path (§5).
- FlashInfer stays under `$HOME`.

**servekit.**
- Maps `TRITON_CACHE_DIR`, `TVM_FFI_CACHE_DIR` and `FLASHINFER_WORKSPACE_BASE` under a fixed node-local
  root `/tmp/servekit/caches` (vendor/servekit/src/servekit/jit_cache.py:14-22).
- The cache is copied in and out (vendor/servekit/src/servekit/jit_cache.py:39-43).
- There is **no TileLang entry**. FlashInfer's autotune files live under `SGLANG_CACHE_DIR` /
  `VLLM_CACHE_ROOT`, outside `FLASHINFER_WORKSPACE_BASE`, so they are not covered either.

## 13. UNVERIFIED items (and why)

- **py-filelock semantics** (FlashInfer's lock): whether a lock open on a read-only directory raises, and
  whether `flock` returns ENOSYS on Lustre without `-o flock`. The `filelock` package is not vendored.
- **ninja behaviour:**
  - rebuilding after relocation because of the command and `.ninja_log` mismatch;
  - whether log files open lazily on a no-op build;
  - repair after a killed link.

  ninja's source is not vendored.
- **capstor mount options:** whether `flock` is coherent across nodes (`flock` vs `localflock`). This is not
  in any source here. It decides whether the FlashInfer and tvm-ffi locks give cross-node exclusion, or, for
  tvm-ffi, an infinite hang.
- **What the images contain:**
  - whether torch 2.13 provides `__dlpack_c_exchange_api__`, which decides whether §7 runs at all;
  - whether the platform's vLLM 0.32.0 image has `flashinfer-jit-cache` or `flashinfer-cubin`.
- **TVM internals:** whether `func.script(show_meta=True)` is byte-identical across processes (the TVM
  printer is C++), and the exact `str(Target)` text.
- **The cause of the vLLM DSV4.1 40 min** (needs a profile).
