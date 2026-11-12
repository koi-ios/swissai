# torch_triton: on-disk compile caches of PyTorch 2.13.0 and Triton 3.7.1

Scope: the caches owned by PyTorch (Inductor, AOTAutograd, `torch.utils.cpp_extension`, Mega-Cache, remote
caches) and by Triton (kernel cache, `cuda_utils` build, autotuner disk cache, `TRITON_CACHE_MANAGER` hook).
Engines (vLLM, SGLang) are out of scope here, except where torch or Triton themselves override a variable.

All paths are relative to `$OPP`. Each claim carries a `path:line` citation. Machine-checked citations are in
`research/torch_triton.citations.json`.

Version check: the Triton checkout is 3.7.1 (`vendor/src/triton-3.7.1-5d6048a/python/triton/__init__.py:2`).
It is the exact commit that torch 2.13.0 pins (`vendor/src/pytorch-v2.13.0/.ci/docker/ci_commit_pins/triton.txt:1`).

**Missing from the sparse checkouts.** These files are not on disk, so any claim that depends on them is marked
UNVERIFIED:

- `torch/_appdirs.py`
- `torch/utils/_filelock.py`
- `torch/_inductor/triton_bundler.py`
- `torch/_inductor/runtime/runtime_utils.py`
- `torch/_inductor/runtime/compile_tasks.py`
- `torch/utils/_cpp_extension_versioner.py`
- `torch/compiler/config.py`
- `torch/_dynamo/__init__.py`
- `torch/_inductor/codegen/*`
- Triton's C++ (`libtriton`, which includes `get_cache_invalidating_env_vars`)

Also not vendored: ninja and the `filelock` PyPI package.

## Default layout at a glance (nothing set, Linux, user `u`)

```
$TMPDIR|/tmp/torchinductor_u/           <- Inductor root (TORCHINDUCTOR_CACHE_DIR)
  <xx>/<hash>.py                           PyCodeCache: graph wrappers + Inductor Triton kernel sources
  <xx>/<hash>.best_config                  Inductor autotune result (JSON)
  <xx>/<hash>.main.cpp / .so               Inductor C++ codecache (CPU kernels / cpp_wrapper)
  fxgraph/<xx>/<key>/<sha256>              FX graph cache entries (pickles)
  aotautograd/<key>/<sha256>               AOTAutograd cache entries (pickles)
  triton/<device-index>/<base32>/...       Triton kernel cache, ONLY if TRITON_CACHE_DIR was unset
  locks/<key>.lock                         FileLocks (C++/CUDA/AOTI paths)
  cache/<system-hash>                      max-autotune GEMM timing JSON
  dynamo/<source-id>/entry                 only with TORCH_CACHING_PRECOMPILE=1
  precompiled_headers/...                  always under the TMPDIR default, even if TORCHINDUCTOR_CACHE_DIR is set
~/.triton/cache/<base32>/...               Triton kernel cache when Triton runs outside Inductor
~/.cache/torch_extensions/py<XY>_cu<V>/<name>/   cpp_extension JIT builds (XDG; UNVERIFIED)
```

---

## 1. Inductor cache root, shared write primitive, PyCodeCache, locks

**1. Variable.**

- `TORCHINDUCTOR_CACHE_DIR` is re-read from `os.environ` on every `cache_dir()` call
  (`vendor/src/pytorch-v2.13.0/torch/_inductor/runtime/cache_dir_utils.py:15`).
- If unset, the default is `os.path.join(tempfile.gettempdir(), "torchinductor_" + <user>)`
  (`vendor/src/pytorch-v2.13.0/torch/_inductor/runtime/cache_dir_utils.py:32`,
  `vendor/src/pytorch-v2.13.0/torch/_inductor/runtime/cache_dir_utils.py:33`).
  - `gettempdir()` honours TMPDIR/TEMP/TMP (Python stdlib behaviour).
  - fbcode uses `/var/tmp` instead.
  - The user name falls back to `uid_<uid>` (`vendor/src/pytorch-v2.13.0/torch/_inductor/runtime/cache_dir_utils.py:29`).
- **Engine override.** `cache_dir()` absolutizes the value and assigns it back into `os.environ`. It does not
  use `setdefault` (`vendor/src/pytorch-v2.13.0/torch/_inductor/runtime/cache_dir_utils.py:19`). It also calls
  `os.makedirs(exist_ok=True)` on every call
  (`vendor/src/pytorch-v2.13.0/torch/_inductor/runtime/cache_dir_utils.py:20`).
  - After the first call, the value is pinned for the process and inherited by child processes.
  - Any later `os.environ.setdefault("TORCHINDUCTOR_CACHE_DIR", ...)` is therefore a no-op.
- `torch/_dynamo/package.py` calls `cache_dir()` at module import
  (`vendor/src/pytorch-v2.13.0/torch/_dynamo/package.py:1267`). Importing that module creates and pins the
  default root. Whether a plain `import torch` imports it is UNVERIFIED (`torch/_dynamo/__init__.py` not checked out).
- `codecache.py` imports `cache_dir` from `runtime_utils`
  (`vendor/src/pytorch-v2.13.0/torch/_inductor/codecache.py:100`). That this module re-exports
  `cache_dir_utils` is UNVERIFIED (file not checked out). Dynamo imports `cache_dir_utils` directly
  (`vendor/src/pytorch-v2.13.0/torch/_dynamo/package.py:1262`).
- **Engine override of Triton.** If `TRITON_CACHE_DIR` is unset, every `CachingAutotuner.__init__` sets it
  (`vendor/src/pytorch-v2.13.0/torch/_inductor/runtime/triton_heuristics.py:492`,
  `vendor/src/pytorch-v2.13.0/torch/_inductor/runtime/triton_heuristics.py:493`).
  - The value is `triton_cache_dir(device)` = `<inductor root>/triton/<device index>`
    (`vendor/src/pytorch-v2.13.0/torch/_inductor/runtime/cache_dir_utils.py:37`). The device is the kernel's
    CUDA device index (`vendor/src/pytorch-v2.13.0/torch/_inductor/runtime/triton_heuristics.py:454`).
  - An explicit `TRITON_CACHE_DIR` wins (`vendor/src/pytorch-v2.13.0/torch/_inductor/runtime/cache_dir_utils.py:38`).
  - Consequence: tensor-parallel ranks on devices 0..N each fill their own `triton/<i>` copy.
  - Consequence: Triton kernels JIT-compiled in the same process *before* the first Inductor kernel go to
    `~/.triton/cache`, and later ones go under the Inductor root. This is inferred from the env being set lazily
    and process-wide.

**2. Key.**

- Generic `write()` / PyCodeCache files are `<root>/<key[1:3]>/<key>.<ext>`
  (`vendor/src/pytorch-v2.13.0/torch/_inductor/codecache.py:448`).
- The key is a hash of the stripped content plus `extra` (`vendor/src/pytorch-v2.13.0/torch/_inductor/codecache.py:510`).
- Content-addressed: no version or path enters the file name except through the content.

**3. Write pattern.**

- `write_atomic` writes a same-directory temp file named `.<pid>.<thread ident>.tmp`
  (`vendor/src/pytorch-v2.13.0/torch/_inductor/codecache.py:538`).
- It then publishes with `Path.rename`, an atomic replace on POSIX
  (`vendor/src/pytorch-v2.13.0/torch/_inductor/codecache.py:543`).
- There is no lock and no fsync.
- **Race (low probability).** The temp name has no host or uuid component. Two writers on different nodes with
  the same pid *and* the same thread ident, writing into the same directory, would share one temp file. Per-container
  PID namespaces make equal pids likely. Equal thread idents (pthread addresses) are unlikely under ASLR.
- `write()` only writes when the target does not exist (`vendor/src/pytorch-v2.13.0/torch/_inductor/codecache.py:512`).
- **Locks.**
  - Lock files live in `<root>/locks` (`vendor/src/pytorch-v2.13.0/torch/_inductor/codecache.py:423`); the
    directory is created if missing (`vendor/src/pytorch-v2.13.0/torch/_inductor/codecache.py:424`).
  - The timeout is `TORCHINDUCTOR_FILE_LOCK_TIMEOUT`, default 600 s
    (`vendor/src/pytorch-v2.13.0/torch/_inductor/config.py:1616`).
  - The lock class is `torch.utils._filelock.FileLock` (`vendor/src/pytorch-v2.13.0/torch/_inductor/codecache.py:3777`).
    That it wraps the `filelock` package (`fcntl.flock` on Linux) is **UNVERIFIED** (file not checked out).
  - Whether capstor's Lustre mount supports `flock` is **UNVERIFIED**.

**4. Relocatability.** Derived paths are recomputed from the current root. Example: on an FX hit the wrapper
path is `get_path(self.cache_key, "py")` (`vendor/src/pytorch-v2.13.0/torch/_inductor/output_code.py:952`).
Exceptions are listed in §2.

**5. Hit/miss and read-only.**

- PyCodeCache hit = the file exists (`vendor/src/pytorch-v2.13.0/torch/_inductor/codecache.py:512`).
- An FX hit re-writes the wrapper `.py` only if it is missing
  (`vendor/src/pytorch-v2.13.0/torch/_inductor/output_code.py:954`,
  `vendor/src/pytorch-v2.13.0/torch/_inductor/output_code.py:956`).
- `makedirs(exist_ok=True)` on an existing read-only directory does not raise (stdlib behaviour).
- Lock acquisition and any missing file need write access.

**6. Content classes (besides FX, AOT and autotune, covered below).**

- `cache/<system-hash>`: max-autotune choice timings, one JSON file
  (`vendor/src/pytorch-v2.13.0/torch/_inductor/codecache.py:320`).
  - It is read-modify-written whole (`vendor/src/pytorch-v2.13.0/torch/_inductor/codecache.py:407`,
    `vendor/src/pytorch-v2.13.0/torch/_inductor/codecache.py:417`) via `write_atomic`
    (`vendor/src/pytorch-v2.13.0/torch/_inductor/codecache.py:335`) with no lock.
  - Concurrent jobs can lose each other's updates; readers never see a torn file.
- `dynamo/`: only with `TORCH_CACHING_PRECOMPILE=1` (`vendor/src/pytorch-v2.13.0/torch/_dynamo/config.py:752`,
  `vendor/src/pytorch-v2.13.0/torch/_dynamo/package.py:1258`).
- `precompiled_headers/` sits under `default_cache_dir()`, which is TMPDIR-based and **ignores
  `TORCHINDUCTOR_CACHE_DIR`** (`vendor/src/pytorch-v2.13.0/torch/_inductor/codecache.py:3542`). It is on by
  default in OSS (`vendor/src/pytorch-v2.13.0/torch/_inductor/config.py:214`).
- `TORCHINDUCTOR_KEEP_STATIC_CUBIN_RAW=1` (default off) keeps cubin bytes inside cached autotuners
  (`vendor/src/pytorch-v2.13.0/torch/_inductor/config.py:1421`). See §2.

## 2. FX graph cache

**1. Variable.**

- No dedicated path variable. Entries live at `<inductor root>/fxgraph/<key[1:3]>/<key>/<sha256(entry)>`
  (`vendor/src/pytorch-v2.13.0/torch/_inductor/codecache.py:1884`,
  `vendor/src/pytorch-v2.13.0/torch/_inductor/codecache.py:1891`,
  `vendor/src/pytorch-v2.13.0/torch/_inductor/codecache.py:2119`).
- On by default (`vendor/src/pytorch-v2.13.0/torch/_inductor/config.py:109`).
  - Force with `TORCHINDUCTOR_FX_GRAPH_CACHE` (`vendor/src/pytorch-v2.13.0/torch/_inductor/config.py:112`).
  - Change the default with `TORCHINDUCTOR_FX_GRAPH_CACHE_DEFAULT` (`vendor/src/pytorch-v2.13.0/torch/_inductor/config.py:111`).
- `force_disable_caches` is an alias of `torch.compiler.config.force_disable_caches`
  (`vendor/src/pytorch-v2.13.0/torch/_inductor/config.py:166`). Its env-var name is UNVERIFIED (file not checked out).

**2. Key.** The key is the hash of a pickled `FxGraphHashDetails`
(`vendor/src/pytorch-v2.13.0/torch/_inductor/codecache.py:1637`). It includes:

- The graph module, fx kwargs, user Triton kernel sources, deterministic flags, default dtype and matmul precision
  (`vendor/src/pytorch-v2.13.0/torch/_inductor/codecache.py:1451`,
  `vendor/src/pytorch-v2.13.0/torch/_inductor/codecache.py:1476`).
- Input tensor metadata, *including the CUDA device index*: the pickler is created without `device_id_agnostic`
  (`vendor/src/pytorch-v2.13.0/torch/_inductor/codecache.py:1639`). Normalisation to index 0 exists only when that
  flag is set (`vendor/src/pytorch-v2.13.0/torch/_inductor/codecache.py:656`). So `cuda:0` and `cuda:1` graphs
  get different keys.
- `torch_key()` (`vendor/src/pytorch-v2.13.0/torch/_inductor/codecache.py:1502`), which is sha256 of:
  - `torch.__version__` (`vendor/src/pytorch-v2.13.0/torch/_inductor/codecache.py:952`);
  - every torch module's name and file bytes (`vendor/src/pytorch-v2.13.0/torch/_inductor/codecache.py:926`,
    `vendor/src/pytorch-v2.13.0/torch/_inductor/codecache.py:927`).

  This is content, not the install path.
- `CacheBase.get_system()` (`vendor/src/pytorch-v2.13.0/torch/_inductor/codecache.py:1503`):
  - the GPU **name** (`vendor/src/pytorch-v2.13.0/torch/_inductor/codecache.py:298`), not the compute capability;
  - `torch.version.cuda`, the CUDA version torch was built with, not the driver
    (`vendor/src/pytorch-v2.13.0/torch/_inductor/codecache.py:299`);
  - Triton's `triton_key()` (`vendor/src/pytorch-v2.13.0/torch/_inductor/codecache.py:289`).
- The portable Inductor config (`vendor/src/pytorch-v2.13.0/torch/_inductor/codecache.py:1504`). The ignore list
  `_cache_config_ignore_prefix` excludes `cutlass_dir` options because each "uses absolute path"
  (`vendor/src/pytorch-v2.13.0/torch/_inductor/config.py:2848`). That `save_config_portable` applies this list is
  UNVERIFIED (`torch/utils/_config_module.py` not checked out).
- `cache_key_tag` (`vendor/src/pytorch-v2.13.0/torch/_inductor/codecache.py:1378`). Its env-var name is UNVERIFIED
  (`torch/compiler/config.py` not checked out).

Not in the key: driver version, Python version, nvcc/ptxas (except through `triton_key`), absolute paths.

**3. Write pattern.**

- `makedirs` then `write_atomic` (`vendor/src/pytorch-v2.13.0/torch/_inductor/codecache.py:2113`,
  `vendor/src/pytorch-v2.13.0/torch/_inductor/codecache.py:2120`). No lock.
- Write errors are logged and swallowed (`vendor/src/pytorch-v2.13.0/torch/_inductor/codecache.py:2193`).
- Concurrent identical compiles each write their own sha256-named file. Duplicates are harmless.

**4. Relocatability.** Mostly relocatable: the entry carries the wrapper source and is re-materialized under the
current root (`vendor/src/pytorch-v2.13.0/torch/_inductor/output_code.py:952`). Exceptions:

- **Pickled autotuners keep an absolute path.** Statically launchable autotuners are pickled with their
  compile-time absolute `filename` (`vendor/src/pytorch-v2.13.0/torch/_inductor/runtime/triton_heuristics.py:508`).
  On load, `recheck_autotune_cache` derives the `.best_config` location from it
  (`vendor/src/pytorch-v2.13.0/torch/_inductor/runtime/triton_heuristics.py:613`). After moving the root that lookup
  points at the old path. This is inferred; the end-to-end effect is UNVERIFIED.
- **Cubins are expected under `triton_cache_dir()`.** They are re-located through the current `triton_cache_dir()`
  (`vendor/src/pytorch-v2.13.0/torch/_inductor/runtime/triton_heuristics.py:2583`). By default the raw cubin bytes are
  dropped from the entry (`vendor/src/pytorch-v2.13.0/torch/_inductor/runtime/triton_heuristics.py:980`). If the cubin
  is not at that path, load raises "Cubin file saved by TritonBundler not found"
  (`vendor/src/pytorch-v2.13.0/torch/_inductor/runtime/triton_heuristics.py:2593`).

**5. Hit/miss and read-only.**

- **Lookup.** All files in the key directory are listed, dot-files (in-flight temps) are skipped
  (`vendor/src/pytorch-v2.13.0/torch/_inductor/codecache.py:1710`), and each is unpickled.
- **Corrupt entries.** A corrupt or unloadable entry is logged and skipped, but never deleted
  (`vendor/src/pytorch-v2.13.0/torch/_inductor/codecache.py:1717`).
- **Guards.** The first entry whose guards pass is the hit (`vendor/src/pytorch-v2.13.0/torch/_inductor/codecache.py:1781`).
- **libdevice check.** A hit is rejected if the extern-libs (libdevice) key differs
  (`vendor/src/pytorch-v2.13.0/torch/_inductor/codecache.py:2092`).
- **On a hit:**
  - The Triton bundle is re-emitted (`vendor/src/pytorch-v2.13.0/torch/_inductor/codecache.py:1953`). Bundling is on
    by default outside fbcode (`vendor/src/pytorch-v2.13.0/torch/_inductor/config.py:48`). The bundler's write pattern
    is UNVERIFIED (`triton_bundler.py` not checked out).
  - The wrapper `.py` is re-written only if missing (`vendor/src/pytorch-v2.13.0/torch/_inductor/output_code.py:956`).
  - An `OSError` during this step is treated as a miss (`vendor/src/pytorch-v2.13.0/torch/_inductor/codecache.py:1978`).
- **Read-only.** Safe only when every derived file (PyCodeCache `.py`, Triton kernel directories, `.best_config`) is
  already present. A miss recompiles, and the compile writes PyCodeCache and Triton files, which fails on a
  read-only mount.

**6. Content classes.** Pickled `CompiledFxGraph` entries, one per guard set, which also carry the Triton bundle and
the pickled static autotuners.

## 3. AOTAutograd cache

**1. Variable.**

- On by default; force with `TORCHINDUCTOR_AUTOGRAD_CACHE` (`vendor/src/pytorch-v2.13.0/torch/_functorch/config.py:73`).
- Remote with `TORCHINDUCTOR_AUTOGRAD_REMOTE_CACHE=1` (`vendor/src/pytorch-v2.13.0/torch/_functorch/config.py:116`).
- Path: `<inductor root>/aotautograd/<key>/<sha256>`, with no `[1:3]` shard
  (`vendor/src/pytorch-v2.13.0/torch/_functorch/_aot_autograd/autograd_cache.py:1159`,
  `vendor/src/pytorch-v2.13.0/torch/_functorch/_aot_autograd/autograd_cache.py:1166`,
  `vendor/src/pytorch-v2.13.0/torch/_functorch/_aot_autograd/autograd_cache.py:1283`).

**2. Key.**

- `"a" + hash(AOTAutogradCacheDetails)` (`vendor/src/pytorch-v2.13.0/torch/_functorch/_aot_autograd/autograd_cache.py:883`).
- The details class subclasses `FxGraphHashDetails`, so it inherits `torch_key`, system info and Inductor config
  (`vendor/src/pytorch-v2.13.0/torch/_functorch/_aot_autograd/autograd_cache.py:448`).
- It adds grad mode, autocast and deterministic state (`vendor/src/pytorch-v2.13.0/torch/_functorch/_aot_autograd/autograd_cache.py:535`)
  and the functorch config (`vendor/src/pytorch-v2.13.0/torch/_functorch/_aot_autograd/autograd_cache.py:544`).

**3. Write pattern.** `write_atomic`, no lock
(`vendor/src/pytorch-v2.13.0/torch/_functorch/_aot_autograd/autograd_cache.py:1285`).

**4. Relocatability.** Same as FX. Entries reference the forward and backward graphs "via keys to the FXGraphCache"
(`vendor/src/pytorch-v2.13.0/torch/_functorch/_aot_autograd/autograd_cache.py:971`).

**5. Hit/miss and read-only.**

- Uses the same guarded iteration as FX (`vendor/src/pytorch-v2.13.0/torch/_functorch/_aot_autograd/autograd_cache.py:1236`).
- A hit also needs the FX entries; an FX miss there counts as a miss
  (`vendor/src/pytorch-v2.13.0/torch/_functorch/_aot_autograd/autograd_cache.py:109`).
- Load errors are logged and treated as a miss (`vendor/src/pytorch-v2.13.0/torch/_functorch/_aot_autograd/autograd_cache.py:1261`).
- Read-only behaviour is the same as FX.

**6. Content classes.** Pickled AOTAutograd results: metadata plus FX keys.

## 4. Inductor autotune results (`.best_config`)

**1. Variable.**

- On by default (`vendor/src/pytorch-v2.13.0/torch/_inductor/config.py:139`).
- The file sits next to the kernel's PyCodeCache `.py`
  (`vendor/src/pytorch-v2.13.0/torch/_inductor/runtime/autotune_cache.py:132`) as `<hash>.best_config`
  (`vendor/src/pytorch-v2.13.0/torch/_inductor/runtime/autotune_cache.py:166`).
- It is skipped under `force_disable_caches` (`vendor/src/pytorch-v2.13.0/torch/_inductor/runtime/triton_heuristics.py:294`).

**2. Key.**

- The kernel file basename (= source hash) plus `cache_key_tag`
  (`vendor/src/pytorch-v2.13.0/torch/_inductor/runtime/autotune_cache.py:151`), re-hashed with `torch_key()`
  (`vendor/src/pytorch-v2.13.0/torch/_inductor/runtime/autotune_cache.py:165`).
- Inside the file, `configs_hash` covers the candidate configs
  (`vendor/src/pytorch-v2.13.0/torch/_inductor/runtime/triton_heuristics.py:3033`). A mismatch makes the entry ignored
  (`vendor/src/pytorch-v2.13.0/torch/_inductor/runtime/autotune_cache.py:664`).
- The GPU architecture is not hashed directly. It reaches the key only through the kernel source and the FX key.
  That Inductor kernel sources embed device properties is UNVERIFIED (codegen not checked out).

**3. Write pattern.**

- `makedirs` then `write_atomic` (`vendor/src/pytorch-v2.13.0/torch/_inductor/remote_cache.py:149`,
  `vendor/src/pytorch-v2.13.0/torch/_inductor/remote_cache.py:152`), after benchmarking
  (`vendor/src/pytorch-v2.13.0/torch/_inductor/runtime/autotune_cache.py:339`).
- No lock: the last writer wins.

**4. Relocatability.** The path is derived from the kernel file under the current root. The only exception is the
pickled static-autotuner case in §2.

**5. Hit/miss and read-only.**

- A hit is a plain read (`vendor/src/pytorch-v2.13.0/torch/_inductor/runtime/autotune_cache.py:182`) and writes nothing.
- A missing file returns `None` (`vendor/src/pytorch-v2.13.0/torch/_inductor/remote_cache.py:144`).
- Corrupt JSON is logged and treated as a miss (`vendor/src/pytorch-v2.13.0/torch/_inductor/remote_cache.py:370`).
- Read-only safe on a hit. A miss followed by tuning tries to write, which raises inside the autotune save path
  (exception handling there UNVERIFIED).

**6. Content classes.** JSON with config kwargs, `num_warps`/`num_stages`, `configs_hash`, `found_by_coordesc`,
`time_taken_ms` and `triton_cache_hash` (`vendor/src/pytorch-v2.13.0/torch/_inductor/runtime/autotune_cache.py:321`).

## 5. Inductor C++ codecache (`CppCodeCache`, `cpp_builder.py`), precompiled headers, CUTLASS

This cache is used only when Inductor emits C++ (CPU kernels, `cpp_wrapper`, AOTI). Triton kernels do not go
through it.

**1. Variable.**

- No own variable. Files are `<inductor root>/<xx>/<key>.main.cpp`, with the `.so` next to it
  (`vendor/src/pytorch-v2.13.0/torch/_inductor/codecache.py:3764`,
  `vendor/src/pytorch-v2.13.0/torch/_inductor/cpp_builder.py:2323`).
- Precompiled headers ignore the root (§1); their locks live under that TMPDIR directory
  (`vendor/src/pytorch-v2.13.0/torch/_inductor/codecache.py:3543`).

**2. Key.**

- Source plus the full compile command line (`vendor/src/pytorch-v2.13.0/torch/_inductor/codecache.py:3764`). The line
  is built with dummy file names but real flags (`vendor/src/pytorch-v2.13.0/torch/_inductor/codecache.py:3757`).
- That includes `-I<absolute include dirs>` (`vendor/src/pytorch-v2.13.0/torch/_inductor/cpp_builder.py:2393`), taken
  from `cpp_extension.include_paths()` (`vendor/src/pytorch-v2.13.0/torch/_inductor/cpp_builder.py:1315`).
- Relative paths are used only in fbcode (`vendor/src/pytorch-v2.13.0/torch/_inductor/codecache.py:3719`).
- So the key changes with the torch or Python install prefix: stable within one image, different across images
  with different prefixes.
- The precompiled-header key adds the compiler version (`vendor/src/pytorch-v2.13.0/torch/_inductor/codecache.py:3597`).

**3. Write pattern.**

- A per-key `FileLock` at `<root>/locks/<key>.lock` (`vendor/src/pytorch-v2.13.0/torch/_inductor/codecache.py:3779`)
  is held around the existence check and build (`vendor/src/pytorch-v2.13.0/torch/_inductor/codecache.py:3879`,
  `vendor/src/pytorch-v2.13.0/torch/_inductor/codecache.py:3881`).
- The compiler writes the target **in place** (`-o <final path>`), running in a `<name>_CxxBuild` working directory
  (`vendor/src/pytorch-v2.13.0/torch/_inductor/cpp_builder.py:2571`,
  `vendor/src/pytorch-v2.13.0/torch/_inductor/cpp_builder.py:2566`).
- There is no temp-and-rename.
- **Crash hazard (inferred).** A compiler killed mid-write leaves a partial `.so`. It passes the existence check
  later and then fails in `cdll.LoadLibrary` (`vendor/src/pytorch-v2.13.0/torch/_inductor/codecache.py:3658`).
- CUTLASS/CUDA templates (max-autotune only) also compile in place under a `FileLock`
  (`vendor/src/pytorch-v2.13.0/torch/_inductor/codecache.py:4940`).

**4. Relocatability.**

- The key embeds absolute include paths (see 2).
- No rpath is added: `cpp_builder.py` has no `-Wl,-rpath`, checked by grep, not citable.
- The `.so` must live on an exec-mounted filesystem (`vendor/src/pytorch-v2.13.0/torch/_inductor/codecache.py:3679`).

**5. Hit/miss and read-only.**

- Hit = the target exists, checked under the lock (`vendor/src/pytorch-v2.13.0/torch/_inductor/codecache.py:3881`).
- Every load acquires the lock, so this cache is **not read-only safe**.
- Correctness across nodes depends on `flock` working on the filesystem (UNVERIFIED, see §1).

**6. Content classes.** `.main.cpp`, `.optimized.cpp`, `.o`, `.so`, `_CxxBuild/` (transient), `locks/*.lock`, and
the `precompiled_headers` `.h`/`.gch`.

## 6. `torch.utils.cpp_extension` (JIT extensions)

**1. Variable.**

- `TORCH_EXTENSIONS_DIR` (`vendor/src/pytorch-v2.13.0/torch/utils/cpp_extension.py:2800`).
- The default is `realpath(torch._appdirs.user_cache_dir('torch_extensions'))`
  (`vendor/src/pytorch-v2.13.0/torch/utils/cpp_extension.py:520`). That this resolves to
  `$XDG_CACHE_HOME` or `~/.cache` is UNVERIFIED (`torch/_appdirs.py` not checked out).
- A `py<maj><min><abiflags>_cu<ver>` subfolder (`vendor/src/pytorch-v2.13.0/torch/utils/cpp_extension.py:2814`) is
  added **only when the env var is unset** (`vendor/src/pytorch-v2.13.0/torch/utils/cpp_extension.py:2801`).
  - The accelerator tag is `cu` plus `torch.version.cuda` without dots
    (`vendor/src/pytorch-v2.13.0/torch/utils/cpp_extension.py:2810`).
  - The Python tag is `py<major><minor><abiflags>` (`vendor/src/pytorch-v2.13.0/torch/utils/cpp_extension.py:2813`).

**2. Key.**

- Only the extension name: the build directory is `<root>/<name>`
  (`vendor/src/pytorch-v2.13.0/torch/utils/cpp_extension.py:2822`).
- `_v<N>` is appended only when sources or flags change *within one process*
  (`vendor/src/pytorch-v2.13.0/torch/utils/cpp_extension.py:2343`). The versioner is a module-level object
  (`vendor/src/pytorch-v2.13.0/torch/utils/cpp_extension.py:468`); that it is purely in-memory is UNVERIFIED.
- No torch version, GPU architecture or source hash is in the path.
- The architecture comes from `TORCH_CUDA_ARCH_LIST` (`vendor/src/pytorch-v2.13.0/torch/utils/cpp_extension.py:2693`)
  or the visible GPUs (`vendor/src/pytorch-v2.13.0/torch/utils/cpp_extension.py:2701`). It is baked into `build.ninja`,
  not into the directory name.

**3. Write pattern.**

- **Lock (FileBaton).** A FileBaton at `<build_dir>/lock` (`vendor/src/pytorch-v2.13.0/torch/utils/cpp_extension.py:2345`)
  is taken with `O_CREAT|O_EXCL` (`vendor/src/pytorch-v2.13.0/torch/utils/file_baton.py:34`).
- **Build.** The holder (`vendor/src/pytorch-v2.13.0/torch/utils/cpp_extension.py:2346`) rewrites `build.ninja`
  in place (plain `open('w')`) only when its content changed
  (`vendor/src/pytorch-v2.13.0/torch/utils/cpp_extension.py:2515`,
  `vendor/src/pytorch-v2.13.0/torch/utils/cpp_extension.py:505`). It then runs `ninja -v` in the build dir
  (`vendor/src/pytorch-v2.13.0/torch/utils/cpp_extension.py:2860`).
- **Waiters.** Other processes poll for the lock file to disappear with **no timeout and no stale-lock detection**
  (`vendor/src/pytorch-v2.13.0/torch/utils/file_baton.py:49`). Release deletes the file
  (`vendor/src/pytorch-v2.13.0/torch/utils/file_baton.py:63`).
- **Hang hazard.** A job killed while holding the baton leaves `lock` behind. Every later load of that extension
  then waits forever.
- Whether ninja writes atomically is UNVERIFIED (ninja not vendored).

**4. Relocatability.** Not relocatable without a rebuild.

- `build.ninja` holds absolute source paths (`vendor/src/pytorch-v2.13.0/torch/utils/cpp_extension.py:3177`) and
  absolute include paths (`vendor/src/pytorch-v2.13.0/torch/utils/cpp_extension.py:2960`).
- `deps = gcc` records header dependencies in `.ninja_deps`
  (`vendor/src/pytorch-v2.13.0/torch/utils/cpp_extension.py:3193`).
- `load_inline` writes its sources into the build dir itself
  (`vendor/src/pytorch-v2.13.0/torch/utils/cpp_extension.py:2259`), so the build dir's own absolute path is in the
  rules.
- Moving or remounting the directory changes the commands, and ninja rebuilds. This is ninja behaviour, UNVERIFIED.
- Standalone executables get an absolute rpath to `torch/lib`
  (`vendor/src/pytorch-v2.13.0/torch/utils/cpp_extension.py:2605`).
- **Stale-binary risk across images (inferred).** The key is the name only and staleness is decided by ninja mtimes.
  The default root is shared by every image with the same Python/CUDA tag, so a newer image with *older* header
  mtimes can reuse a stale `.so`.

**5. Hit/miss and read-only.**

- There is no Python-level hit check: the baton holder runs ninja, and ninja's mtime and command logs decide.
- Waiters import `<build_dir>/<name>.so` directly (`vendor/src/pytorch-v2.13.0/torch/utils/cpp_extension.py:2925`).
- **Never read-only safe**: the lock file must be created.

**6. Content classes.** `build.ninja`, `.ninja_log`, `.ninja_deps`, `*.o`, `*.cuda.o`, `*.o.d`, `<name>.so`, `lock`,
and the inline `main.cpp` / `cuda.cu`.

## 7. Mega-Cache (`torch.compiler.save_cache_artifacts` / `load_cache_artifacts`)

- **Variable.** None: it is an API. Save returns `CacheArtifactManager.serialize()`
  (`vendor/src/pytorch-v2.13.0/torch/compiler/__init__.py:664`). Load deserializes and populates the local caches
  (`vendor/src/pytorch-v2.13.0/torch/compiler/__init__.py:683`).
- **Captured caches.**
  - Inductor FX entries (`vendor/src/pytorch-v2.13.0/torch/compiler/_cache.py:339`), which include their Triton bundle.
  - AOTAutograd (`vendor/src/pytorch-v2.13.0/torch/compiler/_cache.py:336`).
  - Autotune `.best_config` (`vendor/src/pytorch-v2.13.0/torch/compiler/_cache.py:340`).
  - Dynamo PGO (`vendor/src/pytorch-v2.13.0/torch/compiler/_cache.py:335`).
  - Precompile entries.

  Not captured: PyCodeCache files, C++ `.so`, cpp_extension builds, and stand-alone Triton kernels (only kernels
  inside FX bundles are included).
- **Format.** Custom binary: per type, a type string, a uint64 count, then key strings and content bytes
  (`vendor/src/pytorch-v2.13.0/torch/compiler/_cache.py:154`, `vendor/src/pytorch-v2.13.0/torch/compiler/_cache.py:34`).
  - There is no FB/FC (backward/forward) compatibility guarantee
    (`vendor/src/pytorch-v2.13.0/torch/compiler/_cache.py:204`).
  - Deserialization failure is logged and returns `None` (`vendor/src/pytorch-v2.13.0/torch/compiler/_cache.py:313`).
- **Write on load.** The normal writers run:
  - FX `_write_to_local_cache` plus Triton bundle emission (`vendor/src/pytorch-v2.13.0/torch/_inductor/codecache.py:1840`,
    `vendor/src/pytorch-v2.13.0/torch/_inductor/codecache.py:1841`);
  - autotune under the current root (`vendor/src/pytorch-v2.13.0/torch/_inductor/runtime/autotune_cache.py:97`);
  - AOT (`vendor/src/pytorch-v2.13.0/torch/_functorch/_aot_autograd/autograd_cache.py:940`).
- **Relocatable.** Autotune keys are the last two path components
  (`vendor/src/pytorch-v2.13.0/torch/_inductor/runtime/autotune_cache.py:170`), and FX/AOT keys are hashes.
  A blob per image digest and GPU architecture, loaded into a node-local root at job start, avoids shared-directory
  races entirely.

## 8. Remote caches (PyTorch)

- **Variables.** All tri-state, and unset means off in OSS
  (`vendor/src/pytorch-v2.13.0/torch/_inductor/config.py:121`):
  - `TORCHINDUCTOR_FX_GRAPH_REMOTE_CACHE` (`vendor/src/pytorch-v2.13.0/torch/_inductor/config.py:26`);
  - `TORCHINDUCTOR_AUTOTUNE_REMOTE_CACHE` (`vendor/src/pytorch-v2.13.0/torch/_inductor/config.py:38`);
  - `TORCHINDUCTOR_BUNDLED_AUTOTUNE_REMOTE_CACHE` (`vendor/src/pytorch-v2.13.0/torch/_inductor/config.py:42`);
  - `TORCHINDUCTOR_AUTOGRAD_REMOTE_CACHE` (§3);
  - `TORCH_DYNAMO_AUTOMATIC_DYNAMIC_REMOTE_PGO` (`vendor/src/pytorch-v2.13.0/torch/_dynamo/config.py:865`).
- **Backend.** The only OSS backend is Redis.
  - `TORCHINDUCTOR_REDIS_URL` takes precedence (`vendor/src/pytorch-v2.13.0/torch/_inductor/remote_cache.py:283`).
  - Otherwise `TORCHINDUCTOR_REDIS_HOST` and `TORCHINDUCTOR_REDIS_PORT` are used
    (`vendor/src/pytorch-v2.13.0/torch/_inductor/remote_cache.py:287`,
    `vendor/src/pytorch-v2.13.0/torch/_inductor/remote_cache.py:288`).
  - Classes: `RemoteFxGraphCache` and its siblings (`vendor/src/pytorch-v2.13.0/torch/_inductor/remote_cache.py:391`).
  - Key format `pt2:<cache_id>::<key>:c1` (`vendor/src/pytorch-v2.13.0/torch/_inductor/remote_cache.py:339`).
- **FX payload.** Base64 of the pickle inside JSON (`vendor/src/pytorch-v2.13.0/torch/_inductor/codecache.py:2188`),
  with cache id `fx-graph-v1` (`vendor/src/pytorch-v2.13.0/torch/_inductor/codecache.py:2264`).
- **Remote autotune key.** `torch_key`, backend hash, configs hash and a salt
  (`vendor/src/pytorch-v2.13.0/torch/_inductor/runtime/autotune_cache.py:250`).
- **Failure mode.** A connection error silently disables the backend
  (`vendor/src/pytorch-v2.13.0/torch/_inductor/remote_cache.py:301`).
- There is no shared-filesystem remote backend in OSS.

## 9. Triton kernel cache (`FileCacheManager`)

**1. Variable.**

- `TRITON_CACHE_DIR`, default `<TRITON_HOME>/.triton/cache` (`vendor/src/triton-3.7.1-5d6048a/python/triton/knobs.py:347`,
  `vendor/src/triton-3.7.1-5d6048a/python/triton/knobs.py:353`).
- `TRITON_HOME` defaults to `os.path.expanduser("~/")`, evaluated at import
  (`vendor/src/triton-3.7.1-5d6048a/python/triton/knobs.py:343`). If HOME is a shared filesystem, as is typical on
  clusters (an environment assumption, not from source), plain Triton already shares `~/.triton/cache` across jobs
  and images.
- Also `TRITON_DUMP_DIR` and `TRITON_OVERRIDE_DIR` (`vendor/src/triton-3.7.1-5d6048a/python/triton/knobs.py:345`,
  `vendor/src/triton-3.7.1-5d6048a/python/triton/knobs.py:346`).
- Values are read from the environment at every access, not frozen at import
  (`vendor/src/triton-3.7.1-5d6048a/python/triton/knobs.py:118`).
- Under Inductor, see the override in §1.

**2. Key.** The directory is `<dir>/<base32(sha256(key))>`
(`vendor/src/triton-3.7.1-5d6048a/python/triton/runtime/cache.py:256`,
`vendor/src/triton-3.7.1-5d6048a/python/triton/runtime/cache.py:53`). The key is
`triton_key-src.hash-backend.hash-options.hash-env_vars`
(`vendor/src/triton-3.7.1-5d6048a/python/triton/runtime/cache.py:316`), where:

- **`triton_key`** is `__version__` plus content hashes (`vendor/src/triton-3.7.1-5d6048a/python/triton/runtime/cache.py:312`)
  of the compiler and backend modules (`vendor/src/triton-3.7.1-5d6048a/python/triton/runtime/cache.py:290`) and the
  `libtriton` binary (`vendor/src/triton-3.7.1-5d6048a/python/triton/runtime/cache.py:300`). No paths.
- **Source hash** is `fn.cache_key`, attrs, signature and constants
  (`vendor/src/triton-3.7.1-5d6048a/python/triton/compiler/compiler.py:75`).
  - `fn.cache_key` hashes the function source *without decorators*
    (`vendor/src/triton-3.7.1-5d6048a/python/triton/runtime/jit.py:468`), its callees and its starting line number
    (`vendor/src/triton-3.7.1-5d6048a/python/triton/runtime/jit.py:512`).
- **Backend hash** is the `ptxas --version` output plus the target arch (compute capability)
  (`vendor/src/triton-3.7.1-5d6048a/third_party/nvidia/backend/compiler.py:561`,
  `vendor/src/triton-3.7.1-5d6048a/third_party/nvidia/backend/compiler.py:43`).
- **Options hash** covers every `CUDAOptions` field (warps, stages, ctas, `ptx_version`, `PTXAS_OPTIONS`, and so on),
  the arch `sm<cc>` or `TRITON_OVERRIDE_ARCH` (`vendor/src/triton-3.7.1-5d6048a/third_party/nvidia/backend/compiler.py:182`),
  and libdevice by *content* hash (`vendor/src/triton-3.7.1-5d6048a/third_party/nvidia/backend/compiler.py:145`).
- **Env vars** come from `get_cache_invalidating_env_vars()`
  (`vendor/src/triton-3.7.1-5d6048a/python/triton/compiler/compiler.py:245`). The list itself is in C++ and
  **UNVERIFIED**.

Not in the key: driver version, Python version, absolute paths.

**3. Write pattern.**

- `put` writes into a unique temp dir `tmp.pid_<pid>_<uuid4>` inside the key dir
  (`vendor/src/triton-3.7.1-5d6048a/python/triton/runtime/cache.py:115`).
- It then calls `os.replace` (`vendor/src/triton-3.7.1-5d6048a/python/triton/runtime/cache.py:124`) and
  `os.removedirs(temp_dir)` (`vendor/src/triton-3.7.1-5d6048a/python/triton/runtime/cache.py:125`).
- **No lock is ever taken.** `lock_path` is computed but unused
  (`vendor/src/triton-3.7.1-5d6048a/python/triton/runtime/cache.py:54`).
- Each IR stage is stored as soon as it is produced
  (`vendor/src/triton-3.7.1-5d6048a/python/triton/compiler/compiler.py:339`). The group index
  `__grp__<name>.json` is written **last** (`vendor/src/triton-3.7.1-5d6048a/python/triton/compiler/compiler.py:356`).
- Concurrent compiles of the same kernel both do the work and both replace the files; the last group writer wins.
- Readers see either no group (a miss) or a complete one.
- `ptxas` works on NamedTemporaryFiles in TMPDIR (`vendor/src/triton-3.7.1-5d6048a/third_party/nvidia/backend/compiler.py:468`).

**4. Relocatability.** Not relocatable by moving the files.

- `put` returns an absolute joined path (`vendor/src/triton-3.7.1-5d6048a/python/triton/runtime/cache.py:109`), and the
  group JSON stores those paths (`vendor/src/triton-3.7.1-5d6048a/python/triton/runtime/cache.py:98`).
- `get_group` silently drops children whose path does not exist
  (`vendor/src/triton-3.7.1-5d6048a/python/triton/runtime/cache.py:90`). After a move or a different mount point
  the `.json` child disappears, so the lookup misses, the kernel recompiles and the group is rewritten.
- **Ping-pong hazard.** Jobs that mount the shared cache at different absolute paths keep invalidating each other.
  Mount it at an identical path everywhere.

**5. Hit/miss and read-only.**

- Every compile constructs the manager, which runs `os.makedirs(<dir>/<key>)` *before* the lookup
  (`vendor/src/triton-3.7.1-5d6048a/python/triton/runtime/cache.py:55`).
- Hit = the group lists an existing `.json` (`vendor/src/triton-3.7.1-5d6048a/python/triton/compiler/compiler.py:265`,
  `vendor/src/triton-3.7.1-5d6048a/python/triton/compiler/compiler.py:268`).
  - A corrupt group is treated as a miss (`vendor/src/triton-3.7.1-5d6048a/python/triton/runtime/cache.py:82`).
  - `TRITON_ALWAYS_COMPILE` forces a miss (`vendor/src/triton-3.7.1-5d6048a/python/triton/knobs.py:361`).
- On a hit the binary and IR files are read (`vendor/src/triton-3.7.1-5d6048a/python/triton/compiler/compiler.py:431`).
  If the json is present but the cubin was deleted, this is a KeyError (inferred).
- **Read-only.** A hit is safe: `makedirs` on an existing directory is tolerated and nothing is written. A miss
  **raises** at `makedirs` on a read-only filesystem, so the compile fails.

**6. Content classes.**

- `<name>.source`, `.ttir`, `.ttgir`, `.llir`, `.ptx`, `.cubin`
  (`vendor/src/triton-3.7.1-5d6048a/python/triton/compiler/compiler.py:316`).
- `<name>.json` metadata (`vendor/src/triton-3.7.1-5d6048a/python/triton/compiler/compiler.py:264`).
- `__grp__<name>.json` (`vendor/src/triton-3.7.1-5d6048a/python/triton/runtime/cache.py:99`).
- Transient `tmp.pid_*` directories.
- `TRITON_STORE_BINARY_ONLY=1` keeps only cubin/json
  (`vendor/src/triton-3.7.1-5d6048a/python/triton/compiler/compiler.py:338`,
  `vendor/src/triton-3.7.1-5d6048a/python/triton/knobs.py:360`).
- The same tree also holds `cuda_utils*.so` (§10) and `*.autotune.json` (§11).

## 10. Triton launcher stub build (`cuda_utils`)

- **What is built.** In 3.7.1 the NVIDIA backend builds a single C extension, `cuda_utils`
  (`vendor/src/triton-3.7.1-5d6048a/third_party/nvidia/backend/driver.py:68`), from `driver.c`
  (`vendor/src/triton-3.7.1-5d6048a/third_party/nvidia/backend/driver.py:67`). Kernels launch through the generic
  `utils.launch` (`vendor/src/triton-3.7.1-5d6048a/third_party/nvidia/backend/driver.py:300`). There is no per-kernel
  launcher `.so`.
- **Key.** sha256 of the C source plus a platform key
  (`vendor/src/triton-3.7.1-5d6048a/python/triton/runtime/build.py:77`). The platform key is
  machine/system/architecture (`vendor/src/triton-3.7.1-5d6048a/python/triton/runtime/build.py:62`).
  - The file name carries `EXT_SUFFIX`, the CPython ABI tag (`vendor/src/triton-3.7.1-5d6048a/python/triton/runtime/build.py:80`).
  - The compiler (`CC`, `vendor/src/triton-3.7.1-5d6048a/python/triton/runtime/build.py:26`) and the libcuda
    directories (`TRITON_LIBCUDA_PATH` or `ldconfig`, `vendor/src/triton-3.7.1-5d6048a/third_party/nvidia/backend/driver.py:26`)
    are flags, not part of the key (`vendor/src/triton-3.7.1-5d6048a/third_party/nvidia/backend/driver.py:69`).
- **Write.** Built in a `TemporaryDirectory` under TMPDIR
  (`vendor/src/triton-3.7.1-5d6048a/python/triton/runtime/build.py:89`), then published with the manager's atomic
  `put` (`vendor/src/triton-3.7.1-5d6048a/python/triton/runtime/build.py:95`).
- **Relocatable.** No rpath in the compile command (`vendor/src/triton-3.7.1-5d6048a/python/triton/runtime/build.py:43`).
- **Hit/miss.** The file exists, then it is imported. A load failure logs a warning and rebuilds
  (`vendor/src/triton-3.7.1-5d6048a/python/triton/runtime/build.py:87`). A hit is read-only safe; a rebuild needs
  write access.

## 11. Triton autotuner disk cache

- **Variable.** Opt-in through `TRITON_CACHE_AUTOTUNING=1` (`vendor/src/triton-3.7.1-5d6048a/python/triton/knobs.py:375`)
  or `cache_results=True` (`vendor/src/triton-3.7.1-5d6048a/python/triton/runtime/autotuner.py:39`). It is **off by
  default**.
- **Key.** sha256 of:
  - `triton_key`;
  - the backend hash (`vendor/src/triton-3.7.1-5d6048a/python/triton/runtime/autotuner.py:185`);
  - `fn.cache_key`;
  - env vars (`vendor/src/triton-3.7.1-5d6048a/python/triton/runtime/autotuner.py:187`);
  - the tuning key: values of the declared key args
    (`vendor/src/triton-3.7.1-5d6048a/python/triton/runtime/autotuner.py:218`) plus dtypes of the tensor args
    (`vendor/src/triton-3.7.1-5d6048a/python/triton/runtime/autotuner.py:221`);
  - all candidate configs (`vendor/src/triton-3.7.1-5d6048a/python/triton/runtime/autotuner.py:189`).
- **Write.** `<TRITON_CACHE_DIR>/<base32>/<fn>.autotune.json`, through the same manager
  (`vendor/src/triton-3.7.1-5d6048a/python/triton/runtime/autotuner.py:191`,
  `vendor/src/triton-3.7.1-5d6048a/python/triton/runtime/autotuner.py:192`,
  `vendor/src/triton-3.7.1-5d6048a/python/triton/runtime/autotuner.py:209`). Atomic, no lock.
- **Read.** A bare `json.load` with no error handling
  (`vendor/src/triton-3.7.1-5d6048a/python/triton/runtime/autotuner.py:196`).
- **Bypass.** Configs with pre-hooks skip the disk cache
  (`vendor/src/triton-3.7.1-5d6048a/python/triton/runtime/autotuner.py:172`).
- **Read-only.** Same as §9: a hit is safe, a miss fails at `makedirs`.

## 12. `TRITON_CACHE_MANAGER` hook (and the remote backend)

- **Loading.** `TRITON_CACHE_MANAGER=module:Class` (`vendor/src/triton-3.7.1-5d6048a/python/triton/knobs.py:349`) is
  re-read on each access (`vendor/src/triton-3.7.1-5d6048a/python/triton/knobs.py:161`).
  - The value must have the form MODULE:CLASS (`vendor/src/triton-3.7.1-5d6048a/python/triton/knobs.py:166`).
  - The module is imported with `importlib.import_module` (`vendor/src/triton-3.7.1-5d6048a/python/triton/knobs.py:167`).
  - The class must have a base named `CacheManager` in its MRO
    (`vendor/src/triton-3.7.1-5d6048a/python/triton/knobs.py:169`).
  - It is instantiated as `cls(base32_key)` (`vendor/src/triton-3.7.1-5d6048a/python/triton/runtime/cache.py:255`),
    with `override=True` (`vendor/src/triton-3.7.1-5d6048a/python/triton/runtime/cache.py:261`) or `dump=True`
    (`vendor/src/triton-3.7.1-5d6048a/python/triton/runtime/cache.py:266`) for those modes.
- **Interface.**
  - `__init__(key, override, dump)` (`vendor/src/triton-3.7.1-5d6048a/python/triton/runtime/cache.py:16`)
  - `get_file` (`vendor/src/triton-3.7.1-5d6048a/python/triton/runtime/cache.py:20`)
  - `put` returns a path (`vendor/src/triton-3.7.1-5d6048a/python/triton/runtime/cache.py:24`)
  - `get_group` (`vendor/src/triton-3.7.1-5d6048a/python/triton/runtime/cache.py:28`)
  - `put_group` (`vendor/src/triton-3.7.1-5d6048a/python/triton/runtime/cache.py:32`)

  It serves kernels, `cuda_utils` and autotune JSON alike.
- **Read-only shared layer + per-job overlay manager.** Subclass `CacheManager`, not `FileCacheManager`, so that the
  constructor never runs `makedirs` in the shared tree. Then:
  - `get_file` / `get_group`: check the overlay first, then shared.
  - `put` / `put_group`: write to the overlay.
  - Returned paths only need to exist; absolute paths are fine.
- **Caveat: Inductor bypasses the manager.** Inductor builds static-launch cubin paths itself, as
  `triton_cache_dir(device)/<hash>/<name>.cubin`
  (`vendor/src/pytorch-v2.13.0/torch/_inductor/runtime/triton_heuristics.py:2542`).
  - A cubin missing there makes Inductor skip static launch, which costs performance only
    (`vendor/src/pytorch-v2.13.0/torch/_inductor/runtime/triton_heuristics.py:2549`).
  - On an FX hit, a missing cubin raises (§2) unless the Triton bundle re-emits it into `triton_cache_dir()`. That
    the bundler does so is UNVERIFIED.
  - So point `TRITON_CACHE_DIR` at the layer Inductor should see: the writable overlay, with the manager reading
    through to shared.
- **Remote manager.** `RemoteCacheManager` needs `TRITON_REMOTE_CACHE_BACKEND`
  (`vendor/src/triton-3.7.1-5d6048a/python/triton/knobs.py:350`). It always queries the backend
  (`vendor/src/triton-3.7.1-5d6048a/python/triton/runtime/cache.py:196`) and materializes results through a local
  FileCacheManager (`vendor/src/triton-3.7.1-5d6048a/python/triton/runtime/cache.py:182`).
- **Probable bug (inferred, not executed).** The built-in Redis backend reads `knobs.cache.redis.*`
  (`vendor/src/triton-3.7.1-5d6048a/python/triton/runtime/cache.py:151`). `cache_knobs` defines no `redis`; the Redis
  knobs live at module level (`vendor/src/triton-3.7.1-5d6048a/python/triton/knobs.py:554`). Constructing it would
  raise AttributeError.

## Cross-cutting findings for #270

1. **Set `TRITON_CACHE_DIR` explicitly.** Otherwise Inductor redirects Triton to `$TORCHINDUCTOR_CACHE_DIR/triton/<device index>`
   once the first Inductor kernel exists, which means per-rank duplicates. Earlier non-Inductor Triton JITs in the same
   process go to `~/.triton/cache` (§1).
2. **Set `TORCHINDUCTOR_CACHE_DIR` before anything calls `cache_dir()`.** That function pins the variable with an
   assignment, so a later `setdefault` is a no-op. Importing `torch._dynamo.package` already calls it (§1).
3. **Triton group files hold absolute paths.** Bind the shared cache at the same absolute path in every job.
   Otherwise every job recompiles and rewrites the groups (§9).
4. **Locking:**
   - Triton, FX, AOT and autotune use atomic rename with no locks: duplicate work, but safe.
   - The C++ codecache uses a `FileLock` and writes `.so` in place; it needs working `flock` (UNVERIFIED on capstor).
   - cpp_extension uses FileBaton, which hangs forever on a stale lock after a killed job.
   - Inductor's temp names `.<pid>.<tid>.tmp` carry no host or uuid; Triton's carry a uuid4.
5. **Read-only shared directories:**
   - Triton hits work; any Triton miss crashes at `makedirs`.
   - Inductor works only when fully populated.
   - The C++ codecache and cpp_extension never work read-only.

   A per-job writable overlay (Triton manager hook) or Mega-Cache preloading is required.
6. **What the keys miss.**
   - The FX key uses the GPU *name*, not the compute capability, plus `torch.version.cuda`, and also contains the
     input device index.
   - Triton keys use ptxas version and arch.
   - No key contains the driver version.
   - cpp_extension keys on name only. With `TORCH_EXTENSIONS_DIR` set it drops even the py/cu subfolder, so it must be
     per image.
7. **Caches that ignore `TORCHINDUCTOR_CACHE_DIR`.** Precompiled headers always use the TMPDIR default (§1/§5), and so
   do the TMPDIR scratch spaces (ptxas, the `cuda_utils` build).
