# On-disk JIT caches: DeepGEMM/DeepJIT, CuTe DSL, FA4, QuACK and smaller libraries

Topic `deepgemm_cutedsl_misc`. All paths are relative to `$OPP` (`opportunities/01-compile-caches-270`).
Machine-checkable citations are in `research/deepgemm_cutedsl_misc.citations.json`.

## 0. Provenance and what could not be checked

- vLLM 0.32.0 builds DeepGEMM from `vllm-project/DeepGEMM@1e1842a` (`vendor/vllm-v0.32.0/cmake/external_projects/deepgemm.cmake:37`).
  `git ls-tree HEAD third-party/` in the sparse checkout shows the `third-party/deep_jit` gitlink is `3732a3b9a0a2…`,
  the same commit as `vendor/src/DeepJIT-3732a3b`. The kernel code generators (`csrc/jit_kernels/`) are not in the sparse
  checkout, so the exact template parameters baked into generated sources are **UNVERIFIED**.
- SGLang 0.5.20 pins `sgl-deep-gemm==0.2.0` (`vendor/sglang-v0.5.20/python/pyproject.toml:79`). Its host JIT code is compiled
  into `deep_gemm/_C.so`. The wheel also ships DeepJIT headers; `diff -r` against DeepJIT 3732a3b shows the cache code
  (`cache/disk.hpp`, `cache/memory.hpp`, `utils/filesystem.hpp`, `utils/uuid.hpp`, `utils/hash.hpp`, `utils/parser.hpp`,
  `runtime/config.hpp`, `backend/cuda/options.hpp`, `backend/cuda/device.hpp`) is byte-identical. The other differences
  are only GIL attributes, stable-ABI stream lookup, extra driver symbols and enum env parsing. `strings` on
  `vendor/src/sgl-deep-gemm-0.2.0/deep_gemm/_C.so` finds `JIT_CACHE_DIR`, `DJ`, `DG`, `HOME`, `.dj`, `cache`, `tmp`,
  `.committed`, `kernel.cu`, `kernel.cubin`, `meta.json`, `cutlass-`, `deep_gemm/`, `include`, `release (\d+)\.(\d+)`.
  `objdump -T` shows imports of `std::filesystem::rename` and `popen` and no `nvrtc*` symbol. No `USE_NVRTC` string
  exists. I therefore treat it as the same DeepJIT cache; the exact compiled source remains **UNVERIFIED**.
- CuTe DSL: the vendored `nvidia-cutlass-dsl-libs-base` wheels contain only `_mlir`, `_mlir_helpers` and `iket` (their
  RECORD files list no `base_dsl` file). The DSL runtime, including the file cache, ships in
  `nvidia-cutlass-dsl-libs-core==4.6.2` / `==4.8.0`, which are not vendored
  (`vendor/src/nvidia-cutlass-dsl-libs-base-4.6.2/nvidia_cutlass_dsl_libs_base-4.6.2.dist-info/METADATA:29`,
  `vendor/src/nvidia-cutlass-dsl-libs-base-4.8.0/nvidia_cutlass_dsl_libs_base-4.8.0.dist-info/METADATA:28`).
  The only local CuTe DSL Python source is the CUTLASS **4.5.0** `python/CuTeDSL` tree that FlashInfer bundles as data
  (`vendor/src/flashinfer-python-0.6.18/flashinfer/data/cutlass/include/cutlass/version.h:37-39`). It is identical in
  flashinfer 0.6.18 and 0.7.0.post1. Section 4 is based on that proxy and is **UNVERIFIED for 4.6.2/4.8.0**.
- Not available locally, so these are **UNVERIFIED**: the HF `kernels` library (the fetch script requested a manylinux
  wheel and none was unpacked); `tokenspeed-triton`; `tokenspeed_mla==0.1.8` (SGLang's pin; only 0.2.15 is vendored);
  the real `cuda-tile` wheel (the sdist is a wheel-stub); `xgrammar==0.2.8` and `numba==0.65.0` (vLLM's pins; 0.2.1 and
  0.65.1 are vendored).
- Version drift between the images: vLLM pins cutlass-dsl 4.8.0, quack 0.6.5, tokenspeed-mla 0.2.15 and humming 0.1.17
  (`vendor/vllm-v0.32.0/requirements/cuda.txt:24-31`). SGLang pins cutlass-dsl 4.6.2, quack 0.6.4, tokenspeed_mla 0.1.8
  and humming 0.1.12 (`vendor/sglang-v0.5.20/python/pyproject.toml:44,55,73,87`).

---

## 1. DeepJIT disk cache (the JIT engine behind DeepGEMM in both images)

**Variable.**
- Every setting is read as `<PREFIX>_<SUFFIX>` and, if unset, as `DJ_<SUFFIX>`
  (`vendor/src/DeepJIT-3732a3b/include/deep_jit/utils/env.hpp:66-69,79`). DeepGEMM uses the prefix `DG`
  (`vendor/src/vllm-DeepGEMM-1e1842a/csrc/runtime/jit.hpp:19`). The cache suffix is `JIT_CACHE_DIR`
  (`vendor/src/DeepJIT-3732a3b/include/deep_jit/cache/disk.hpp:91`), so the precedence is
  `DG_JIT_CACHE_DIR` > `DJ_JIT_CACHE_DIR` > `$HOME/.dj` (`vendor/src/DeepJIT-3732a3b/README.md:347`).
- The built-in default is `$HOME/.dj`, not `~/.deep_gemm`. An empty `HOME` is an assertion failure
  (`vendor/src/DeepJIT-3732a3b/include/deep_jit/cache/disk.hpp:103-105`).
- The value may be a colon list. The first root is writable; the other roots are read-only lookup roots. Empty elements
  are rejected (`vendor/src/DeepJIT-3732a3b/include/deep_jit/cache/disk.hpp:88-101`,
  `vendor/src/DeepJIT-3732a3b/README.md:49-52`).
- The value is read once, when the lazy `Runtime` is constructed on first `jit->` use, not at import
  (`vendor/src/DeepJIT-3732a3b/include/deep_jit/runtime/runtime.hpp:43`,
  `vendor/src/DeepJIT-3732a3b/include/deep_jit/utils/lazy.hpp:26-29`, `vendor/src/DeepJIT-3732a3b/README.md:354`).

**Key.** The digest is a 128-bit two-lane FNV-1a. Every input is length-prefixed
(`vendor/src/DeepJIT-3732a3b/include/deep_jit/utils/hash.hpp:10-37`). The inputs, in order:
1. `extra_signature`, which for DeepGEMM is `cutlass-<CUTLASS_VERSION>`
   (`vendor/src/DeepJIT-3732a3b/include/deep_jit/runtime/runtime.hpp:52`,
   `vendor/src/vllm-DeepGEMM-1e1842a/csrc/runtime/jit.hpp:20`).
2. A hash of the complete `nvcc --version` text: toolkit release plus build string
   (`vendor/src/DeepJIT-3732a3b/include/deep_jit/runtime/runtime.hpp:53`,
   `vendor/src/DeepJIT-3732a3b/include/deep_jit/backend/cuda/backend.hpp:40,64`).
3. The effective NVCC flag string (`vendor/src/DeepJIT-3732a3b/include/deep_jit/backend/cuda/options.hpp:136-139`). It contains:
   - `--gpu-architecture=sm_<arch>` (`options.hpp:98`). The arch is `90a` on sm_90 and the family target (`100f`,
     `103f`) on newer GPUs (`vendor/src/DeepJIT-3732a3b/include/deep_jit/backend/cuda/device.hpp:68-80`).
   - `-O3`, `--register-usage-level=10` and `-std=c++<DG_JIT_CPP_STANDARD|20>` (`options.hpp:39-62`).
   - DeepGEMM's extra `--diag-suppress` / `-Wno-*` flags (`vendor/src/vllm-DeepGEMM-1e1842a/csrc/runtime/jit.hpp:36-39`).
   - Debug env vars change the flags and therefore the key: `DG_JIT_DEBUG`, `DG_JIT_WITH_LINEINFO` and
     `DG_JIT_PTXAS_VERBOSE` (`options.hpp:111,127`). `JIT_DUMP_PTX/SASS` do not (`vendor/src/DeepJIT-3732a3b/README.md:304`).
4. The post-hook hash. DeepGEMM sets none.
5. The generated source plus the recursive content hash of every `#include <deep_gemm/...>` header
   (`vendor/src/DeepJIT-3732a3b/include/deep_jit/runtime/runtime.hpp:95-96`,
   `vendor/src/DeepJIT-3732a3b/include/deep_jit/utils/parser.hpp:61-65,116`).

Not in the key:
- Absolute include-dir paths. They are passed as `--include-path` after `get_flags()`
  (`vendor/src/DeepJIT-3732a3b/README.md:288`, `vendor/src/DeepJIT-3732a3b/include/deep_jit/backend/cuda/backend.hpp:105-108`).
- The kernel tag, which only names the directory (`vendor/src/DeepJIT-3732a3b/README.md:294`).
- Driver version, host compiler, `NVCC_PREPEND_FLAGS`/`NVCC_APPEND_FLAGS`/`CPATH`, and CUTLASS header contents
  (only `CUTLASS_VERSION` is used) (`vendor/src/DeepJIT-3732a3b/README.md:306,332`).
- The GPU SKU and SM count, except where the generated source embeds them. **UNVERIFIED**, because
  `csrc/jit_kernels/` is not vendored. The `_C.so` templates show many `{}` template slots per kernel.

The entry directory is `<root>/cache/<tag>.<digest>/` (`vendor/src/DeepJIT-3732a3b/include/deep_jit/cache/disk.hpp:120-122`).

**Write pattern.** No lock.
1. A miss creates `<root0>/tmp/<pid>-<8hex>-<8hex>-<8hex>`
   (`vendor/src/DeepJIT-3732a3b/include/deep_jit/cache/disk.hpp:130-131`,
   `vendor/src/DeepJIT-3732a3b/include/deep_jit/utils/uuid.hpp:22`).
2. It writes `kernel.cu` with fsync and runs `cd <tmp> && nvcc kernel.cu --cubin --output-file kernel.cubin …` through
   `popen` (`vendor/src/DeepJIT-3732a3b/include/deep_jit/backend/cuda/backend.hpp:86-115`).
3. It requires a non-empty cubin (`backend.hpp:128-130`) and writes `meta.json` (`backend.hpp:161-167`).
4. `commit()` writes an empty `.committed` marker and fsyncs the whole tree bottom-up
   (`vendor/src/DeepJIT-3732a3b/include/deep_jit/cache/disk.hpp:47-48`,
   `vendor/src/DeepJIT-3732a3b/include/deep_jit/utils/filesystem.hpp:41-51`). It then creates `<root0>/cache` and
   renames the whole directory into place (`disk.hpp:52-54`).
5. If the rename fails because another process already published the entry, the loser deletes its temp dir and uses the
   existing entry (`disk.hpp:55-66`). Concurrent identical compiles waste CPU but are safe on filesystems with atomic
   directory rename and fsync (`vendor/src/DeepJIT-3732a3b/README.md:36`).
6. A reader never sees a half-written entry, because a hit requires `.committed` inside the renamed directory
   (`disk.hpp:123`). A killed build leaves an orphan `tmp/<pid>-…` directory. Cleanup runs only from the destructor
   (`disk.hpp:69-73`), and there is no garbage collection.

Edge case: any rename error, not only "target exists", is treated as "another rank won" (`disk.hpp:55-66`). After an
EXDEV or EACCES the later load panics with `missing CUDA CUBIN`
(`vendor/src/DeepJIT-3732a3b/include/deep_jit/backend/cuda/kernel.hpp:54-55`).

**Relocatability.** Yes. Loading opens only `<entry>/kernel.cubin` (`kernel.hpp:53`), and the key excludes paths. So
moving the cache root, or installing the same image at another prefix, keeps the hits. `meta.json` records the NVCC
command with temp-dir paths and the absolute `python_library_root`/`include_dirs` (`backend.hpp:161-166`,
`vendor/src/DeepJIT-3732a3b/include/deep_jit/runtime/config.hpp:52-63`). It is informational only
(`vendor/src/DeepJIT-3732a3b/README.md:302`). A different CUDA toolkit build, which changes the `nvcc --version` text,
is a miss.

**Hit/miss and read-only.**
- A hit means `exists(<root>/cache/<tag>.<digest>/.committed)`, searched over the roots in order
  (`vendor/src/DeepJIT-3732a3b/include/deep_jit/cache/disk.hpp:119-127`). Kernels are then memoised in process
  (`vendor/src/DeepJIT-3732a3b/include/deep_jit/cache/memory.hpp:15-20`).
- A hit also touches the marker's mtime. This is best effort, and a failure never turns a hit into a miss
  (`disk.hpp:124`, `vendor/src/DeepJIT-3732a3b/include/deep_jit/utils/filesystem.hpp:61-69`). Read-only lookup roots are
  supported (`vendor/src/DeepJIT-3732a3b/README.md:52`).
- A miss needs a writable first root: `create_directories` throws (`disk.hpp:131`).
- For #270 that means `DG_JIT_CACHE_DIR=<node-local or per-job rw>:<capstor ro>` works natively in DeepJIT, but see the
  SGLang caveat in §3.

**Content classes.** `cache/<tag>.<digest>/{kernel.cu, kernel.cubin, meta.json, .committed}`, plus `kernel.ptx` and
`kernel.sass` when `JIT_DUMP_*` is set, and `tmp/` build dirs (`vendor/src/DeepJIT-3732a3b/README.md:302`). Only NVCC
is used; there is no NVRTC path. Runtime NVCC must be at least 12.9
(`vendor/src/DeepJIT-3732a3b/include/deep_jit/backend/cuda/backend.hpp:72`). NVCC is found through
`CUDA_HOME` / `CUDA_PATH` / `which nvcc` / `/usr/local/cuda`, or `DG_JIT_NVCC_COMPILER` (`backend.hpp:174-216`).

## 2. DeepGEMM in vLLM 0.32.0 (vendored `vllm.third_party.deep_gemm`)

- **Engine default (setdefault-like, lazy).** `_lazy_init()` sets `DG_JIT_CACHE_DIR=$VLLM_CACHE_ROOT/deep_gemm` only
  when the variable is unset or empty (`vendor/vllm-v0.32.0/vllm/utils/deep_gemm.py:234,278-283`). `VLLM_CACHE_ROOT`
  defaults to `${XDG_CACHE_HOME:-~/.cache}/vllm` (`vendor/vllm-v0.32.0/vllm/envs.py:338-341,712-716`). The effective
  default is therefore `~/.cache/vllm/deep_gemm`, not DeepJIT's `~/.dj`.
- **Surprise.** vLLM writes its default into `DG_JIT_CACHE_DIR`, which DeepJIT ranks above `DJ_JIT_CACHE_DIR`
  (`vendor/src/DeepJIT-3732a3b/include/deep_jit/utils/env.hpp:66-69`). A launcher that sets only `DJ_JIT_CACHE_DIR` is
  silently overridden in vLLM, so the launcher must set `DG_JIT_CACHE_DIR`.
- **Ordering caveat (UNVERIFIED in practice).** The variable is set only in `_lazy_init()`. Some call sites call
  `_import_deep_gemm()` directly (`vendor/vllm-v0.32.0/vllm/model_executor/layers/fused_moe/deep_gemm_mega_moe.py:153`).
  If such a path made the first `jit->` call before any `_lazy_init()`, DeepJIT would snapshot `$HOME/.dj`
  (`vendor/src/DeepJIT-3732a3b/README.md:354`). Setting `DG_JIT_CACHE_DIR` explicitly removes the dependency.
- **Which package.** vLLM prefers a pip-installed `deep_gemm`, then the vendored copy
  (`vendor/vllm-v0.32.0/vllm/utils/deep_gemm.py:194,205`).
- **What the build installs.** The build installs `deep_gemm/{__init__.py,utils,testing,legacy,mega}`, `_C.py` and
  `_C_extension*.so` (`vendor/vllm-v0.32.0/cmake/external_projects/deepgemm.cmake:97,137-170`). It also installs DeepGEMM
  and CUTLASS/CuTe headers into `vllm/third_party/deep_gemm/include` (`deepgemm.cmake:186-192`).
- **No env injection at import.** The generated `envs.py` is empty (`deepgemm.cmake:173-178`), so
  `deep_gemm/__init__.py:5-12` injects nothing.
- **JIT configuration.** The JIT root is the package dir (`vendor/src/vllm-DeepGEMM-1e1842a/deep_gemm/__init__.py:113`):
  - Include dir is `<pkg>/include` and the tracked include prefix is `deep_gemm/`
    (`vendor/src/vllm-DeepGEMM-1e1842a/csrc/runtime/jit.hpp:16-22`). CUTLASS headers are in the same include dir but
    are keyed only by `CUTLASS_VERSION`.
  - On SM107 the arch is forced to `100f` (`jit.hpp:31-34`).
- **Warm-up.** `VLLM_DEEP_GEMM_WARMUP` defaults to `relax` (`vendor/vllm-v0.32.0/vllm/envs.py:1645-1647`), so kernels
  are compiled at startup. vLLM adds no lock of its own and relies on DeepJIT's rename.
- **TileLang: no.** `third-party/tilelang_ops` contains `@tilelang.jit` kernels
  (`vendor/src/vllm-DeepGEMM-1e1842a/third-party/tilelang_ops/swiglu_apply_weight_to_fp8.py:13`,
  `vendor/src/vllm-DeepGEMM-1e1842a/third-party/tilelang_ops/norm.py:19`). That directory is not in vLLM's install set
  (`deepgemm.cmake:137-192`), and no vendored `deep_gemm/*.py` file references `tilelang`. DeepGEMM therefore does not
  trigger TileLang JIT in vLLM. `deep_gemm/mega/` is not in the sparse checkout (**UNVERIFIED**), but it could not import
  a module that is not installed.
- **Triton.** The A100 legacy path is Triton (`vendor/src/vllm-DeepGEMM-1e1842a/deep_gemm/__init__.py:106-110`), so it
  uses Triton's cache, which another topic covers.

## 3. DeepGEMM in SGLang 0.5.20 (`sgl-deep-gemm` 0.2.0)

- **Engine override (hard, at import).**
  - `deep_gemm_wrapper/compile_utils.py` executes
    `os.environ["DG_JIT_CACHE_DIR"] = envs.SGLANG_DG_CACHE_DIR.get()` unconditionally
    (`vendor/sglang-v0.5.20/python/sglang/srt/layers/deep_gemm_wrapper/compile_utils.py:44-46`), so a user's
    `DG_JIT_CACHE_DIR` is ignored.
  - `SGLANG_DG_CACHE_DIR` defaults to `$SGLANG_CACHE_DIR/deep_gemm`, and `SGLANG_CACHE_DIR` defaults to
    `~/.cache/sglang` (`vendor/sglang-v0.5.20/python/sglang/srt/environ.py:31-37,1154,1164`). The knob to set is
    `SGLANG_DG_CACHE_DIR` (or `SGLANG_CACHE_DIR`).
- **Dead knob.** SGLang also sets `DG_JIT_USE_NVRTC` (`compile_utils.py:48-51`). It has no effect: DeepJIT only invokes
  NVCC, and `_C.so` contains no `USE_NVRTC` string and imports no `nvrtc*` symbol. `_C.so` does list
  `libnvrtc.so.13` as NEEDED, from the `-lnvrtc` link flag (`vendor/src/sgl-deep-gemm-0.2.0/deep_gemm/__init__.py:70`).
- **Same disk cache as §1.** The cache semantics come from the shipped headers
  (`vendor/src/sgl-deep-gemm-0.2.0/deep_gemm/include/deep_jit/cache/disk.hpp:54,91,105`,
  `vendor/src/sgl-deep-gemm-0.2.0/deep_gemm/include/deep_jit/runtime/runtime.hpp:96`). The JIT root is the package dir
  and CUDA home is passed in (`vendor/src/sgl-deep-gemm-0.2.0/deep_gemm/__init__.py:559-562`).
- **Prebuilt `_C.so`.** It is shipped prebuilt. Only if it is missing does import rebuild it with tvm-ffi into
  `<site-packages>/deep_gemm/_C_build` (`vendor/src/sgl-deep-gemm-0.2.0/deep_gemm/__init__.py:80,96-109`). No envs.py
  ships (RECORD), so `__init__.py:15-22` injects nothing.
- **Extra lock layer (SGLang only).** The first rank on each node wraps every precompile group in a blocking
  `fcntl.flock(LOCK_EX)` on `<DG_JIT_CACHE_DIR>/locks/<KIND>_n<N>_k<K>_g<G>.lock`, opened with mode `"w"`
  (`compile_utils.py:99-105,120-140,179-186`). Precompile is on by default
  (`vendor/sglang-v0.5.20/python/sglang/srt/environ.py:1149`). SGLang warns that it can take "typically 10-20 mins"
  (`compile_utils.py:165`) and walks up to 16k-128k M values (`compile_utils.py:89-97`). Consequences:
  - The SGLang DG dir is **not read-only-safe**: the lock is `open(…, "w")`.
  - Cross-node exclusion needs coherent `flock` on the filesystem. Whether capstor is mounted with the Lustre `flock`
    option is **UNVERIFIED**. If `flock` is unsupported, the `OSError` is not caught here.
  - A DeepJIT colon list (`rw:ro`) breaks here: the whole string is used as one path (`compile_utils.py:132`) and would
    create e.g. `/rw:/ro/locks`.
- **TileLang: no.** `tilelang` is only a `dev` extra
  (`vendor/src/sgl-deep-gemm-0.2.0/sgl_deep_gemm-0.2.0.dist-info/METADATA:18`), and the wheel RECORD contains no
  `tilelang_ops`.

## 4. CuTe DSL file cache (`nvidia-cutlass-dsl` 4.6.2 in SGLang, 4.8.0 in vLLM). Proxy: CUTLASS 4.5.0 source, **UNVERIFIED** for both

Paths in this section are under
`vendor/src/flashinfer-python-0.6.18/flashinfer/data/cutlass/python/CuTeDSL/cutlass/`, written in full each time.

**Variable.**
- `CUTE_DSL_CACHE_DIR` is read once, at import of `cutlass.base_dsl.cache_helpers`, into a module-level default that the
  load and dump helpers use as their default `path`
  (`vendor/src/flashinfer-python-0.6.18/flashinfer/data/cutlass/python/CuTeDSL/cutlass/base_dsl/cache_helpers.py:70,235-236,270-276`).
- Default is `$TMPDIR` (else `tempfile.gettempdir()`) `/<USER>/cutlass_python_cache`. The directory is created at import
  (`…/base_dsl/cache_helpers.py:71-79`). The user comes from `$USER`, `$USERNAME` or pwd (`…/base_dsl/cache_helpers.py:43-56`).
  If that fails, the code falls back to `<tmp>/cutlass_python_cache` (`…/base_dsl/cache_helpers.py:80-82`).
- The env prefix is `CUTE_DSL` (`vendor/src/flashinfer-python-0.6.18/flashinfer/data/cutlass/python/CuTeDSL/cutlass/cutlass_dsl/cutlass.py:1099`).
  `CUTE_DSL_DISABLE_FILE_CACHING` stops writes, and `CUTE_DSL_NO_CACHE` disables the in-memory and file caches
  (`vendor/src/flashinfer-python-0.6.18/flashinfer/data/cutlass/python/CuTeDSL/cutlass/base_dsl/env_manager.py:351-356,375-377`).
- The docstring's default (`/tmp/{dsl_name}_python_cache_{tmpfile_suffix}`) is stale (`…/base_dsl/env_manager.py:320`).
- **Engines.** Neither engine sets any `CUTE_DSL_*CACHE*` variable (grep over both trees finds nothing). FlashInfer
  0.7.0.post1 (vLLM image) sets `CUTE_DSL_CACHE_DIR=${FLASHINFER_WORKSPACE_BASE:-~}/.cache/flashinfer/cute_dsl` when it
  is unset (`vendor/src/flashinfer-python-0.7.0.post1/flashinfer/prims_ts/utils.py:24-35`). That runs when
  `flashinfer.prims_ts` is imported (`vendor/src/flashinfer-python-0.7.0.post1/flashinfer/prims_ts/__init__.py:26`), and
  takes effect only if `cutlass` was not imported earlier. The import order is **UNVERIFIED**.

**Key.** `module_hash` is the sha256 of the following
(`vendor/src/flashinfer-python-0.6.18/flashinfer/data/cutlass/python/CuTeDSL/cutlass/base_dsl/dsl.py:1037-1060`):
- an empty version seed (`…/base_dsl/dsl.py:1039-1041`). In the open-source copy no DSL version is hashed; the shipped
  libs-core may differ (**UNVERIFIED**);
- the traced MLIR module bytecode (`…/base_dsl/dsl.py:1045`);
- `str()` of every non-None env-manager attribute (`…/base_dsl/dsl.py:1046-1048`);
- the compile options (`…/base_dsl/dsl.py:1050`).

The env-manager attributes include:
- `dump_dir`, which defaults to `Path.cwd()` (`…/base_dsl/env_manager.py:357-359`, `…/base_dsl/cache_helpers.py:85-91`);
- `arch`, which is `CUTE_DSL_ARCH` or the detected `sm_<maj><min>[a]`, falling back to `sm_100a` when no GPU is seen
  (`…/base_dsl/env_manager.py:106-129,367`);
- the `cache_dir` string itself (`…/base_dsl/env_manager.py:364`);
- the CUDA toolkit path, the absolute runtime-library paths, and logging flags (`…/base_dsl/env_manager.py:379,382`).

Consequence, if 4.6.2/4.8.0 behave like the proxy: entries are reusable only with the same CWD, the same
`CUTE_DSL_CACHE_DIR` string, the same install paths and the same arch. Setting `CUTE_DSL_DUMP_DIR` to a constant would
pin the CWD component.

**Write pattern.** No lock.
1. Dump writes into a fresh `tmp.pid_<pid>_<uuid4>` dir inside the cache dir. It writes the bytecode plus a 4-byte CRC32
   (`…/base_dsl/cache_helpers.py:94-108,204-226`).
2. `os.replace` moves the file to `cute_dsl_<module_hash>.mlir`, then `os.removedirs` removes the temp dir
   (`…/base_dsl/cache_helpers.py:227-230`). Last writer wins with identical content.

**Hit/miss and read-only.**
- Before compiling, the DSL looks for the file (`…/base_dsl/dsl.py:1198-1206`).
- A wrong CRC or a parse error only logs a warning and counts as a miss (`…/base_dsl/cache_helpers.py:110-134,251-267`).
- A file hit still re-JITs host code through the MLIR execution engine (`…/base_dsl/dsl.py:1253-1258`).
- Writes happen after a miss unless disabled (`…/base_dsl/dsl.py:1282-1294`).
- Read-only directories work: write errors are caught and logged (`…/base_dsl/cache_helpers.py:294-306`). The
  `os.makedirs(path, exist_ok=True)` before that is outside the `try` and fails only if the directory is missing and
  cannot be created (`…/base_dsl/cache_helpers.py:293`).

**Relocatable / content.** The files are self-contained bytecode, but the key embeds the cache-dir string, the CWD and
absolute paths (above). The directory holds flat `cute_dsl_<sha256>.mlir` files and transient `tmp.pid_*` dirs.

## 5. FlashAttention-4 persistent cache (`flash-attn-4` 4.0.0b35 and vLLM's vendored 9cd61de)

**Who uses which copy.**
- **vLLM** installs vllm-flash-attention 9cd61de `flash_attn/cute` as `vllm/vllm_flash_attn/cute`, rewriting imports
  (`vendor/vllm-v0.32.0/cmake/external_projects/vllm_flash_attn.cmake:85-115`). vLLM sets no `FLASH_ATTENTION_*`
  variable (grep), so the persistent cache is off.
- **SGLang** uses its own vendored FA4 by default. The pip package is used only with `SGLANG_INKLING_FA4_USE_PIP=1`
  (`vendor/sglang-v0.5.20/python/sglang/kernels/ops/attention/flash_attention_v4.py:11-19`) and by the qwen sparse
  backend (`vendor/sglang-v0.5.20/python/sglang/srt/layers/attention/qwen_sparse_attn_backend.py:82`). The SGLang
  vendored copy uses the cache in §6, not this one.
- **FlashInfer.** FlashInfer's BSA kernels read the same two variables with the same default dir
  (`vendor/src/flashinfer-python-0.7.0.post1/flashinfer/cute_dsl/sparse/bsa_utils/cache_utils.py:44-52`).

**Variable.**
- `FLASH_ATTENTION_CUTE_DSL_CACHE_ENABLED=1` enables the cache. It is off by default, read at import
  (`vendor/src/flash-attn-4-4.0.0b35/flash_attn/cute/cache_utils.py:34`; vLLM copy:
  `vendor/src/vllm-flash-attention-9cd61de/flash_attn/cute/cache_utils.py:34`).
- `FLASH_ATTENTION_CUTE_DSL_CACHE_DIR` sets the directory (`cache_utils.py:39`). The default is
  `tempfile.gettempdir()/<getuser()>/flash_attention_cute_dsl_cache` (`cache_utils.py:42-48`). `gettempdir` honours
  `TMPDIR`.

**Key.** The path is `<dir>/<fingerprint>/<name>/<sha256(pickle(compile_key))>.o`
(`vendor/src/flash-attn-4-4.0.0b35/flash_attn/cute/cache_utils.py:277-278,302-307`).
- The fingerprint is sha256 over: Python major.minor, `cutlass.__version__`, `tvm_ffi.__version__`, and the relative
  path and bytes of every `flash_attn/cute/**/*.py` (`cache_utils.py:51-78`).
- `name` is `fwd`, `bwd`, etc. (`vendor/src/flash-attn-4-4.0.0b35/flash_attn/cute/interface.py:1969`).
- The forward key includes dtype, head_dim/head_dim_v, the GQA ratio, causal, mask/score-mod hashes, feature flags, tile
  sizes, arch and the FA log level (`interface.py:1452-1521`). Arch is `major*10+minor`, or `FLASH_ATTENTION_ARCH`
  (`interface.py:100-117`).
- It is compiled once per dtype, head-dim, feature combination and arch, not per sequence length. This is inferred
  from the key contents, **UNVERIFIED** for every path.

**Write pattern** (both copies take an exclusive per-key `fcntl.flock` on `<sha>.lock`;
`vendor/src/flash-attn-4-4.0.0b35/flash_attn/cute/cache_utils.py:238-246`):
- **pip 4.0.0b35.** Export is skipped if a non-empty, loadable `.o` exists. Otherwise it writes
  `.<sha>.<pid>.tmp.o` and `os.replace`s it (`cache_utils.py:248-275`). An unloadable entry counts as a miss and is
  replaced (`cache_utils.py:221-231`).
- **vLLM 9cd61de.** Export is skipped whenever the `.o` exists, even if empty, and `export_to_c` writes **directly to
  the final path** (`vendor/src/vllm-flash-attention-9cd61de/flash_attn/cute/cache_utils.py:236-245`). The loader has
  no `try` (`…/9cd61de/flash_attn/cute/cache_utils.py:217-221`). A process killed mid-export leaves a truncated `.o`
  that makes every later load raise until it is deleted.

**Lock details.**
- Locks are polled with `LOCK_NB` every 0.1 s and raise `RuntimeError` after 15 s
  (`vendor/src/flash-attn-4-4.0.0b35/flash_attn/cute/cache_utils.py:117-139,180`).
- Readers take a shared lock but open the lock file with `O_CREAT` (`cache_utils.py:118,211-218`).
- On a filesystem without `flock` support every attempt fails, so loads and exports raise after 15 s. This is
  **UNVERIFIED** for capstor.

**Read-only.** Not read-only-safe. Looking up a key creates `<sha>.lock`, and enabling the cache mkdirs the directory
(`cache_utils.py:47,184`).

**Relocatable.** Yes for the directory: no cache path is in the key. The fingerprint is image-specific.

**Content.** `<sha>.o` (object from `export_to_c`, loaded with `cute.runtime.load_module` after preloading the DSL
runtime libraries `RTLD_GLOBAL`; `cache_utils.py:22-28,222`), `<sha>.lock` (never removed), and transient
`.<sha>.<pid>.tmp.o`.

## 6. SGLang 0.5.20 CuTe AOT cache (`SGLANG_CUTE_AOT_CACHE_DIR`). SGLang's default FA4 path, on by default

**Variable.**
- Default is `$SGLANG_CACHE_DIR/cute_aot`, which is `~/.cache/sglang/cute_aot`
  (`vendor/sglang-v0.5.20/python/sglang/srt/environ.py:1168`). An empty string disables the cache
  (`vendor/sglang-v0.5.20/python/sglang/kernels/jit/cute_aot_cache.py:409-415`).
- It is used by SGLang's vendored FA4 (`vendor/sglang-v0.5.20/python/sglang/kernels/ops/attention/flash_attn/cute/interface.py:14,1854-1866`)
  and by the Kimi-K3 KDA decode kernel (`vendor/sglang-v0.5.20/python/sglang/kernels/ops/kimi_k3/kda_decode_mtp.py:895`).

**Key.** `<dir>/<fingerprint>/<name>/<sha256(pickle((enable_tvm_ffi, key)))>.o`
(`vendor/sglang-v0.5.20/python/sglang/kernels/jit/cute_aot_cache.py:374-376,417-427`).
- The fingerprint covers Python major.minor, `sys.platform`, `machine`, the cutlass version, the CUDA version, the
  tvm-ffi flag and version, the arch, and the bytes of this module plus the caller's source dir
  (`cute_aot_cache.py:44-90`; FA4 passes its own directory, `…/flash_attn/cute/interface.py:1857`).
- Arch is `CUTE_DSL_ARCH`, else `sm_<major><minor>` (`cute_aot_cache.py:93-100`).

**Write pattern.**
- An exclusive `flock` on `<sha>.lock` (15 s polled timeout; `cute_aot_cache.py:170-193,238`) protects the write.
- If `<sha>.o` exists the export is skipped. Otherwise it exports to `.<sha>.tmp.o`, then `os.replace` publishes it
  (`cute_aot_cache.py:337-372`).
- Readers hold a shared lock while loading (`cute_aot_cache.py:276-310`).
- An unloadable object is unlinked under the exclusive lock, with an inode check (`cute_aot_cache.py:312-335`).

**Read-only, relocatable, content.**
- Not read-only-safe: the lock file is opened `O_CREAT` and the directory is mkdir'ed on first use
  (`cute_aot_cache.py:170-176,251-258`).
- Relocatable: yes, no cache path is in the key.
- Content: `<sha>.o`, `<sha>.lock` and transient `.<sha>.tmp.o`.

## 7. QuACK 0.6.4 (SGLang) / 0.6.5 (vLLM)

`quack/cache/` and `autotuner.py` are identical between the two versions (`diff -rq`).

**`.o` cache.**
- **Variable.** On by default with `QUACK_CACHE_ENABLED=1`, relocated by `QUACK_CACHE_DIR`
  (`vendor/src/quack-kernels-0.6.5/quack/cache/__init__.py:38-39`,
  `vendor/src/quack-kernels-0.6.4/quack/cache/__init__.py:38-39`). Default is
  `tempfile.gettempdir()/<getuser()>/quack_cache` (`vendor/src/quack-kernels-0.6.5/quack/cache/jit.py:47-53`).
- **Key.**
  - The fingerprint is sha256 over Python major.minor, the cutlass version, the tvm_ffi version, all `quack/**/*.py`
    files and any `EXTRA_SOURCE_DIRS` (`jit.py:56-82`).
  - The entry name is `sha256(pickle((qualname,)+args+sorted kwargs))` (`jit.py:85-86,162,180`).
  - **No arch in the fingerprint**, and e.g. `_compile_rmsnorm_fwd` keys on dtypes, N, flags and config
    (`vendor/src/quack-kernels-0.6.5/quack/rmsnorm.py:426-440`). Whether `config` carries the arch is **UNVERIFIED**, so
    a directory shared across GPU archs may be unsafe. Per-arch dirs avoid this.
- **Write pattern.**
  - An unlocked `exists()` check is followed by a shared-lock load (`jit.py:205-224`).
  - A miss takes the exclusive lock and compiles under it (`jit.py:275-313`). It exports to `<sha>.o.tmp.<pid>` and
    `os.replace`s the result (`jit.py:320-326`).
  - Corrupt objects are unlinked and recompiled (`jit.py:191-203`).
  - The lock is `fcntl.flock`, polled with a 60 s timeout. On timeout QuACK compiles in process and skips the disk write
    (`jit.py:43,94-122,279-300`).
- **Read-only:** no. A miss needs the exclusive lock (`O_WRONLY|O_CREAT`), and the fingerprint dir is mkdir'ed
  (`jit.py:104,182`).
- **Relocatable:** yes.
- **Content:** `<fingerprint>/<sha>.o` and `<sha>.lock` (`jit.py:183-184`).

**Autotune cache.**
- Enabled by default by the `autotune(...)` decorator (`cache_results=True`) or by `QUACK_CACHE_AUTOTUNING=1`
  (`vendor/src/quack-kernels-0.6.5/quack/autotuner.py:126-128,526`).
- The directory is `$QUACK_CACHE_DIR/<base32 key>/`, else `${QUACK_HOME:-~}/.quack/cache/<key>/` (`autotuner.py:34-53`).
- The file is `<fn>.autotune.json` and the key is sha256 of the quack version, the tuning key (shapes, stride classes,
  dtypes) and the config strings (`autotuner.py:256-294`). The GPU is not keyed, so per-arch dirs are needed.
- Writes inherit Triton's `put` (temp dir + `os.replace`; `vendor/src/triton-3.7.1-5d6048a/python/triton/runtime/cache.py:115,124`).
- Side effect: Triton's base `__init__` first creates an empty `<TRITON_CACHE_DIR>/<key>/` before QuACK overrides the
  directory (`vendor/src/triton-3.7.1-5d6048a/python/triton/runtime/cache.py:51-55`, `autotuner.py:42-51`).

**Engines.** Both engines import only QuACK utilities (`make_fake_tensor`, `copy_utils`, `layout_utils`, `ParamsBase`;
e.g. `vendor/vllm-v0.32.0/vllm/model_executor/layers/mamba/ops/gdn_chunk_cutedsl/__init__.py:10`). No engine call site
of a `@jit_cache` QuACK kernel was found, so these caches are expected to stay dormant in serving. **UNVERIFIED** for
every model path.

## 8. tokenspeed-mla 0.2.15 (vLLM): no own disk cache

- CuTe kernels are compiled with `cute.compile` and memoised in process only
  (`vendor/src/tokenspeed-mla-0.2.15/tokenspeed_mla/mla_decode.py:186-187,404`,
  `vendor/src/tokenspeed-mla-0.2.15/tokenspeed_mla/mla_prefill.py:212`). They are still subject to the CuTe DSL file
  cache in §4.
- The Triton kernels (`vendor/src/tokenspeed-mla-0.2.15/tokenspeed_mla/mla_kv_pack_quantize_fp8.py:40`) use the
  separate `tokenspeed_triton` fork (`vendor/src/tokenspeed-mla-0.2.15/tokenspeed_mla/_triton.py:23`;
  `vendor/src/tokenspeed-mla-0.2.15/tokenspeed_mla-0.2.15.dist-info/METADATA:35`). Its cache variable and default are
  **UNVERIFIED** because it is not vendored. It may or may not share `TRITON_CACHE_DIR` with upstream Triton.
- The only env var is a numerics switch (`vendor/src/tokenspeed-mla-0.2.15/tokenspeed_mla/mla_decode.py:53`).
- SGLang's `tokenspeed_mla==0.1.8` is not vendored (**UNVERIFIED**).

## 9. humming-kernels 0.1.12 (SGLang) / 0.1.17 (vLLM): real disk cache, NVRTC by default

**Variable.**
- The cache is `HUMMING_CACHE_DIR`, default `~/.humming/cache/`
  (`vendor/src/humming-kernels-0.1.17/humming/utils/jit.py:117-122`).
- A separate `HUMMING_TMP_DIR`, default `~/.humming/tmp/`, holds `lock/` and generated Python `module/`s (`jit.py:107-138`).
- Neither engine sets any `HUMMING_*` path variable (grep).

**Key.** md5 of `"<CompilerClass>$$<nvrtc+maj.min | nvcc+ver>$$<flags>$$<kernel_expr>$$<source>$$<json{abs .cuh path: st_mtime}>"`
(`vendor/src/humming-kernels-0.1.17/humming/jit/compiler.py:48-53,64-70`; md5 `jit.py:63-66`).
- The flags include `--gpu-architecture=sm_<sm>[a]` and absolute `-I` include dirs
  (`compiler.py:186-207`; `vendor/src/humming-kernels-0.1.17/humming/jit/runtime.py:57-61`).
- 0.1.12 truncates the md5 to 16 hex characters (`vendor/src/humming-kernels-0.1.12/humming/utils/jit.py:66`), so the
  two images never share entry names.
- Because header **mtimes and absolute install paths** are keyed, any reinstall or a different image is a full miss. A
  per-image cache dir is required anyway.

**Compiler.** NVRTC is used unless `HUMMING_COMPILER=nvcc`, or `cuda.bindings` is missing
(`vendor/src/humming-kernels-0.1.17/humming/jit/runtime.py:63-78`).

**Write pattern.**
- `filelock.FileLock` on `<TMP>/lock/<hash>.lock` is held for both hit and miss (`compiler.py:76-78`). The lock lives
  in the TMP dir, not the cache dir, so cross-node exclusion requires a shared `HUMMING_TMP_DIR`.
- A miss writes `kernel.cu`, `signature.txt`, `cmdline.json`, `shims/` and logs directly. The NVRTC helper writes
  `kernel_tmp.cubin`, which is published with `os.replace(… "kernel.cubin")` (`compiler.py:81-109,215-247`).
- A partial `kernel.cu` or logs are harmless, because a hit requires `kernel.cubin`.

**Read-only.** Hits work from a read-only cache dir as long as `HUMMING_TMP_DIR` is writable: `mkdir(exist_ok)` on an
existing dir, then the lock in TMP, then `exists` (`compiler.py:72-79`). Misses need write access.

**Side builds.**
- Importing `humming.utils.nvrtc` and `humming.ops.utils` spawns background subprocesses that may build helpers
  (`vendor/src/humming-kernels-0.1.17/humming/utils/nvrtc.py:126-140`,
  `vendor/src/humming-kernels-0.1.17/humming/ops/utils.py:248-262`).
- Prebuilt aarch64 `nvrtc_compile`, `libhumming_launcher.so` and `libcubinpatch.so` are used when the manifest's source
  hashes match (`vendor/src/humming-kernels-0.1.17/humming/utils/jit.py:28-42`;
  `vendor/src/humming-kernels-0.1.17/humming/_native/aarch64/manifest.json:1`). I recomputed the manifest hashes and they
  match the shipped sources in both versions.
- Otherwise the helpers are built under `HUMMING_CACHE_DIR/{nvrtc_compile,cubin_patcher,launcher}/…`. The launcher uses
  a `torch.utils.cpp_extension` ninja tree with absolute paths
  (`vendor/src/humming-kernels-0.1.17/humming/ops/utils.py:170-178,201-245`). The prebuilt launcher needs torch ≥ 2.11,
  which both images have (torch 2.13), so on aarch64 these fallbacks should not trigger.

**Content.** `<md5>/{kernel.cu, signature.txt, cmdline.json (absolute paths, informational), stdout.log, stderr.log, shims/, kernel.cubin}`.
The TMP dir holds `lock/*.lock` and `module/humming_module_<md5>.py`.

## 10. cuda-tile 1.6.0rc5: **UNVERIFIED** (no code vendored)

- The sdist is a `wheel-stub` that downloads the real wheel from `pypi.nvidia.com` at install time
  (`vendor/src/cuda-tile-1.6.0rc5/cuda_tile-1.6.0rc5/pyproject.toml:4-10`). The compiler comes from the optional
  `cuda-toolkit[tileiras]` extra (`vendor/src/cuda-tile-1.6.0rc5/cuda_tile-1.6.0rc5/PKG-INFO:24-26`).
- It enters the images via FlashInfer (`vendor/src/flashinfer-python-0.7.0.post1/flashinfer_python-0.7.0.post1.dist-info/METADATA:14`;
  SGLang pin `vendor/sglang-v0.5.20/python/pyproject.toml:34`). Its users are FlashInfer's opt-in `cutile` backends
  (`vendor/src/flashinfer-python-0.6.18/flashinfer/gemm/kernels/cutile/mm_bf16_cutile.py:35`).
- Whether cuTile keeps an on-disk cache, and under which variable, cannot be determined from local sources.

## 11. HF `kernels` hub and FA3 in SGLang 0.5.20

- **Default path.** `SGLANG_USE_SGL_FA3_KERNEL=True` by default
  (`vendor/sglang-v0.5.20/python/sglang/srt/environ.py:1244`), so FA3 comes from the AOT-built `sgl-kernel`. No hub
  download and no JIT happen (`vendor/sglang-v0.5.20/python/sglang/kernels/ops/attention/flash_attention_v3.py:30-38,72-80`).
- **Hub path (only with `SGLANG_USE_SGL_FA3_KERNEL=0`).**
  - It looks for `$SGLANG_CACHE_DIR/kernels.lock` (`flash_attention_v3.py:16,41-43`).
  - If the file exists, `load_kernel(repo, lockfile)` loads it; otherwise `get_kernel("kernels-community/sgl-flash-attn3", revision="v1")`
    fetches from the Hub (`flash_attention_v3.py:14-15,51-54`).
  - Any failure falls back to sgl-kernel (`flash_attention_v3.py:60-68`).
  - The library pin is `kernels>=0.14.1,<0.15` (`vendor/sglang-v0.5.20/python/pyproject.toml:47,265-266`). Its storage
    (presumably the `huggingface_hub` cache under `HF_HOME`/`HF_HUB_CACHE`) is **UNVERIFIED**, because the library is
    not vendored.
- **Image build.**
  - `kernels lock python` runs at build time (`vendor/sglang-v0.5.20/docker/Dockerfile:585`).
  - On aarch64 the cubin download is skipped. The comment says the kernels "will be JIT-compiled at runtime"
    (`Dockerfile:587-591`). That claim is about the `kernels` library and is **UNVERIFIED**.
  - The lockfile is moved to `/root/.cache/sglang/kernels.lock` (`Dockerfile:608-609`).
- **Gotcha.** Pointing `SGLANG_CACHE_DIR` at capstor (to relocate DeepGEMM and cute_aot) also moves the lookup away
  from the baked `/root/.cache/sglang/kernels.lock`. That matters only if `SGLANG_USE_SGL_FA3_KERNEL=0`, in which case
  it turns a locked offline load into a network `get_kernel`.

## 12. numba 0.65.1 (SGLang pin; vLLM pins 0.65.0, not vendored)

**Variable.**
- `NUMBA_CACHE_DIR` (`vendor/src/numba-0.65.1/numba-0.65.1/numba/core/config.py:279`).
- Locator order is user-provided, in-tree, user-wide, IPython, zip (`vendor/src/numba-0.65.1/numba-0.65.1/numba/core/caching.py:376-382`):
  1. `$NUMBA_CACHE_DIR/<parentdir>_<sha1(abs source dir)>` (`caching.py:142-157,192-211`).
  2. `__pycache__` next to the source file (`caching.py:214-226`).
  3. `${XDG_CACHE_HOME:-~/.cache}/numba/<parentdir>_<sha1>` (`caching.py:241-270`,
     `vendor/src/numba-0.65.1/numba-0.65.1/numba/misc/appdirs.py:303`).
- A locator is used only if `makedirs` and a test `TemporaryFile` succeed (`caching.py:108-112,184-188`). A read-only
  `NUMBA_CACHE_DIR` is **silently skipped**, and numba falls through to `__pycache__` or `~/.cache/numba`.
- No engine sets `NUMBA_CACHE_DIR` (grep).

**Key.**
- Index file `<module>.<qualname>-<lineno>.py<maj><min><abiflags>.nbi` (`caching.py:434-440,539`). It holds
  `numba.__version__` and the source stamp `(mtime, size)` of the .py file (`caching.py:166-173,543,581-604`).
- Per-overload key: signature, codegen magic tuple (target triple, host CPU name, CPU features), and sha256 of the
  bytecode and closure vars (`caching.py:764-782`; `vendor/src/numba-0.65.1/numba-0.65.1/numba/core/codegen.py:1298-1303`).

**Write pattern.** `<file>.tmp.<uuid16>` + `os.replace` for both the index and the data, with no lock
(`caching.py:638-656`). Concurrent writers race on the index read-modify-write and on data-file numbering
(`caching.py:548-565`); the last writer wins. Whether a mismatched data file is detected on load is **UNVERIFIED**.
A stale stamp or version mismatch means a miss.

**Relocatable.** No. The sha1 of the absolute source directory is in the subdir name, and the source mtime is in the index.

**Content.** `.nbi` and `.nbc` files.

**Engines.** Only a few multimodal processors use `cache=True`:
- vLLM: `vendor/vllm-v0.32.0/vllm/transformers_utils/processors/inkling.py:123` and
  `vendor/vllm-v0.32.0/vllm/transformers_utils/processors/kimi_k25_vision_fused.py:26`.
- SGLang: `vendor/sglang-v0.5.20/python/sglang/srt/multimodal/inkling/image_processing.py:106`.

The n-gram proposer does not use `cache=True` (`vendor/vllm-v0.32.0/vllm/v1/spec_decode/ngram_proposer.py:180`).

## 13. xgrammar 0.2.1 (SGLang pin; vLLM pins 0.2.8, not vendored): no disk cache

- The grammar compiler cache is in memory only, with `cache_enabled` and `cache_limit_bytes`
  (`vendor/src/xgrammar-0.2.1/xgrammar-0.2.1/python/xgrammar/compiler.py:110-141`).
- The default CUDA bitmask backend is Triton (`vendor/src/xgrammar-0.2.1/xgrammar-0.2.1/python/xgrammar/matcher.py:149-155`).
- The CUDA backend is opt-in and documented as unused
  (`vendor/src/xgrammar-0.2.1/xgrammar-0.2.1/python/xgrammar/kernels/__init__.py:4`). It would JIT a
  `torch.utils.cpp_extension.load_inline(name="xgrammar")` module into the torch-extensions dir
  (`vendor/src/xgrammar-0.2.1/xgrammar-0.2.1/python/xgrammar/kernels/apply_token_bitmask_inplace_cuda.py:72-96`;
  matcher dispatch `matcher.py:180-183`).
- SGLang uses its own bitmask kernels (`vendor/sglang-v0.5.20/python/sglang/kernels/ops/grammar/__init__.py:13`).

---

## Summary

| component | env var | default | key includes | write pattern | relocatable? | read-only-safe? |
|---|---|---|---|---|---|---|
| DeepJIT / DeepGEMM (vLLM) | `DG_JIT_CACHE_DIR` (> `DJ_JIT_CACHE_DIR`; `rw:ro…` list) | vLLM sets `~/.cache/vllm/deep_gemm` if unset; lib default `~/.dj` | `cutlass-<ver>`, full `nvcc --version`, NVCC flags incl. `sm_90a`/`100f`, source + `deep_gemm/` header contents | `tmp/<pid-rand>` → fsync → `.committed` → dir `rename`; no lock | yes | hits yes (RO lookup roots); misses need rw first root |
| DeepGEMM (SGLang, sgl-deep-gemm 0.2.0) | `SGLANG_DG_CACHE_DIR` (overwrites `DG_JIT_CACHE_DIR`) | `~/.cache/sglang/deep_gemm` | same as DeepJIT | same + SGLang `locks/*.lock` `flock` (first rank per node) | yes | no (lock opened `"w"`); colon list breaks SGLang lock path |
| CuTe DSL file cache (proxy 4.5.0; 4.6.2/4.8.0 UNVERIFIED) | `CUTE_DSL_CACHE_DIR`, `CUTE_DSL_DISABLE_FILE_CACHING` | `$TMPDIR/$USER/cutlass_python_cache` (mkdir at import) | MLIR bytecode + all `CUTE_DSL_*` values incl. CWD (`dump_dir`), arch, cache-dir string, abs lib paths + options | per-pid tmp dir + `os.replace`, CRC32; no lock | files yes, key no | yes (write errors logged) |
| FA4 (pip b35 / vLLM 9cd61de) | `FLASH_ATTENTION_CUTE_DSL_CACHE_ENABLED=1` (off), `…_DIR` | `/tmp/$USER/flash_attention_cute_dsl_cache` | fingerprint(py, cutlass, tvm_ffi, all cute/*.py) / sha256(pickle(key: dtype, hdim, flags, arch, log level)) | flock 15 s + tmp→replace (b35); flock + in-place write, no recovery (vLLM) | yes | no (lock `O_CREAT`) |
| SGLang cute_aot (SGLang's FA4) | `SGLANG_CUTE_AOT_CACHE_DIR` (`""` disables) | `~/.cache/sglang/cute_aot` (on) | fingerprint(py, host, cutlass, CUDA, arch, tvm_ffi, sources) / sha256(pickle) | flock 15 s + tmp→replace; bad objects evicted | yes | no |
| QuACK 0.6.4/0.6.5 | `QUACK_CACHE_ENABLED` (on), `QUACK_CACHE_DIR`, `QUACK_HOME` | `/tmp/$USER/quack_cache`; autotune `~/.quack/cache` | fingerprint(py, cutlass, tvm_ffi, quack src) / sha256(pickle(qualname, args)); no arch | flock 60 s + tmp→replace; autotune via Triton put | yes | no (dormant in engines) |
| humming 0.1.12/0.1.17 | `HUMMING_CACHE_DIR`, `HUMMING_TMP_DIR` | `~/.humming/cache`, `~/.humming/tmp` | md5(compiler+NVRTC ver, flags incl. abs `-I`, sm, source, `.cuh` mtimes) | `filelock` in TMP + `kernel_tmp.cubin`→replace | dir yes; reinstall = miss | hits yes if TMP writable |
| numba 0.65.1 | `NUMBA_CACHE_DIR` | `__pycache__` next to source, else `~/.cache/numba` | numba ver, src mtime+size, sig, triple/CPU/features, bytecode | uuid tmp + replace; no lock | no (sha1 of abs src dir) | no (RO dir silently skipped) |
| tokenspeed-mla 0.2.15 / xgrammar 0.2.1 | none | none (in-memory) | – | – | – | – |
| cuda-tile 1.6.0rc5 / HF kernels | UNVERIFIED (not vendored) | FA3 default = AOT sgl-kernel | – | – | – | – |

Surprises:
1. SGLang hard-overwrites `DG_JIT_CACHE_DIR`. vLLM fills `DG_JIT_CACHE_DIR` lazily, which shadows `DJ_JIT_CACHE_DIR`.
2. DeepJIT's default is `~/.dj`, and it supports `rw:ro` root lists natively. SGLang's own lock code breaks on such a list.
3. `DG_JIT_USE_NVRTC` (set by SGLang) is dead.
4. vLLM's vendored FA4 export is non-atomic, with no recovery from a truncated `.o`.
5. SGLang's default FA4 path uses `~/.cache/sglang/cute_aot` (on by default), not `FLASH_ATTENTION_CUTE_DSL_*`.
6. The CuTe DSL key (proxy) includes the CWD and the cache-dir string.
7. QuACK and humming key on no arch, or on mtimes and absolute paths respectively.
8. The CuTe DSL runtime is not vendored at all.
9. All `flock`-based caches (FA4, cute_aot, QuACK, the SGLang DG lock, humming `filelock`) depend on `flock` support on
   capstor (UNVERIFIED).
