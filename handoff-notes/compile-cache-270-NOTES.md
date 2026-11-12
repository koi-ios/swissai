# Working notes: idea 2 (#270 for both engines + servekit 0.5.20 fix)

Scratch: /private/tmp/claude-501/-Users-koiios--claude-spares-spare-qHWvhZ/85cbb828-b415-43aa-adc6-b088d62505ae/scratchpad/compile-cache-270/
Idea folders: /Users/koiios/Documents/Projects/swissai/compile-cache-270/ and /Users/koiios/Documents/Projects/swissai/servekit-sglang-0520/
SHARED=/private/tmp/claude-501/-Users-koiios--claude-spares-spare-qHWvhZ/85cbb828-b415-43aa-adc6-b088d62505ae/scratchpad/shared
Previous (aborted) session left pinned sources (read-only for me):
  /Users/koiios/Documents/Projects/swissai/opportunities/01-compile-caches-270/vendor/src/{tilelang-0.1.12, flashinfer-python-0.6.18, flashinfer-python-0.7.0.post1, apache-tvm-ffi-0.1.11, sgl-deep-gemm-0.2.0, vllm-DeepGEMM-1e1842a, DeepJIT-3732a3b, triton-3.7.1-5d6048a, pytorch-v2.13.0 (sparse), nvidia-cutlass-dsl-libs-base-4.6.2/4.8.0, flash-attn-4-4.0.0b35, quack-kernels, humming-kernels, tokenspeed-mla, xgrammar, numba, cuda-tile}
  .../vendor/vllm-v0.32.0 (full-ish), .../vendor/sglang-v0.5.20
  fetch script: .../tools/fetch_sources.sh (pins from sglang v0.5.20 pyproject + vLLM v0.32.0 requirements/cuda.txt)

## #270 (fetched 11 Nov via API; saved gh/issue270.json, gh/issue270_comments.json)
- Opened 2026-10-18 by AryanAhadinia, unassigned, no labels.
- Title: Persist JIT/compile caches (TileLang, DeepGEMM/FlashInfer, vLLM) on capstor to cut 40-70 min startups
- Clariden 2026-10-18 normal partition: DeepSeek-V4.1-Flash (mfa) 2 nodes, 60-70 min to ready, ~40 min after weight load mostly TileLang JIT during CUDA graph capture (51 graphs), "GPUs at 0% while the CPU compiles".
  GLM-5.3 (mfa) 4 nodes, ~60 min, 8-10 min DeepGEMM warmup (VLLM_DEEP_GEMM_WARMUP=relax default).
- One DSV4.1 run was ready with ~20 min of a 1:30 job left; one GLM-5.3 run didn't finish in time. Also slows restarts/handovers (--consecutive).
- "We deliberately keep the warmup" -> pay once per model.
- Table (checked in vllm_base image): TILELANG_CACHE_DIR ~/.tilelang/cache; VLLM_CACHE_ROOT ~/.cache/vllm; FLASHINFER_WORKSPACE_BASE $HOME -> ~/.cache/flashinfer ("DeepGEMM goes through it in this image"); TRITON_CACHE_DIR ~/.triton/cache.
- "In practice nothing persists: after these runs ~/.tilelang exists but is empty, and ~/.cache/vllm doesn't exist. Either the container's $HOME isn't the shared home, or the writes go elsewhere. This needs confirming."
- Proposal: mount capstor dir -> /jit-cache, per-image subdirs (TILELANG_CACHE_DIR=/jit-cache/vllm_base/tilelang etc.), check concurrent-write safety (atomic writes or one dir per GPU arch), infra01 permissions, second launch skips compile; clear cache when vLLM/TileLang/FlashInfer version changes (image digest or version in path).
- Comments (Aryan, 18 Oct): path should be /capstor/store/cscs/swissai/infra01/sml/shared-caches ; "The equivalent should be also implemented for SGL."

## model-launch @02b74ce facts
- Images built in private swiss-ai/model-launch-images; published as ghcr.io/swiss-ai/<name>:<channel> (private: anonymous token DENIED, checked 11 Nov) and /capstor/.../container-images/ci/<name>-<arch>.sqsh. docs/building-images.md:30 "Images are updated in place (a vLLM upgrade is a change to vllm_base, not a new image) ... with no rollback." => the sqsh PATH is not a version key. Need a content key.
- vllm_base.toml was renamed from vllm_0.30.0.toml in #265 (74da8ab, 19 Oct). vllm_0.30.0 Dockerfile: images/vllm_0.30.0/Dockerfile pins vllm==0.30.0 (cu130). Current vllm_base vLLM version: NOT visible publicly. docs/kisti.md uses vllm v0.32.0 cu129.
- sglang_base: SGLang 0.5.20 (images/sglang_0.5.20/Dockerfile; conda py3.12, torch 2.13.0 cu130, sglang-kernel 0.4.7, sglang-router 0.3.2). #271 (521da02).
- EDF [env]: only NCCL/FI vars + SGL_ENABLE_JIT_DEEPGEMM=0 (sglang) + VLLM_ALLREDUCE_USE_SYMM_MEM=0. vllm_cuda13_v2.toml adds VLLM_USE_DEEP_GEMM=0, TORCHDYNAMO_DISABLE=1 (only used by 1 catalog entry: qwen3 tts omni).
- models.json: 33 entries; 15 sglang default env(sglang_base), 11 vllm default(vllm_base), 4 vllm_apertus_1.5_release, 2 sglang.toml (GLM-5/5.1), 1 vllm_cuda13_v2.
- Launch: master.sh on batch host -> per node `srun --container-writable --container-mounts=$RANKS_DIR:$RANKS_DIR --environment=$SML_ENV_FILE bash $RANKS_DIR/head.sh` (framework.py _render_container_flags ~L533-560). Rank scripts: `set -ex`, framework env_exports, user pre_launch_cmds, then (opentela wrap) engine. RANKS_DIR=$PWD/job-$SLURM_JOB_ID on shared FS (working dir default ~/.sml, #261).
- vLLM multi-node: head.sh writes /tmp/sml-ray-head-$JOBID.sh: ray start --head; wait for GPUs; `vllm serve --distributed-executor-backend ray`. Followers: `ray start --address ... --block`. So env for Ray workers on followers comes from follower.sh/EDF env, not from head.
- SGLang sets SGL_ENABLE_JIT_DEEPGEMM=false and SGLANG_ENABLE_JIT_DEEPGEMM=false in rank scripts (framework.py Sglang.env_exports).
- DSV4.1-Flash mfa: vllm_base, 2 nodes, TP8, highprio, 3h. GLM-5.3 mfa: vllm_base, 4 nodes TP16.
- pyxis runtime path (KISTI): --container-writable, --container-env=<keys of [env]>.

## CSCS Container Engine facts (docs.cscs.ch, fetched 11 Nov)
- EDF ref (docs.cscs.ch/software/container-engine/edf/): writable default true; mounts default []; env "Inherited from the host and the image by default"; workdir inherited from image.
- known-issue page: "It is generally NOT recommended to mount home folders inside containers"; "Defining a mount related to /users in the EDF should only be done when there is a specific reason".
- => platform EDFs mount /capstor, /iopsstor, opentela-share, 4 libs; NOT /users. HOME inherited from host = /users/<u> (not mounted) -> writes go to container writable layer, lost at job end. (Strong inference, not verified on Alps.)
- Youssef's EDFs (coldstart-exp edf/llama-3.1-70b-sglang.toml, servekit-eval/scripts/sglang.toml) set HOME="/root" with comment "Container-local HOME: caches (HF/Triton/pip/XDG) fall back under here and live in the container's ephemeral overlay, so every srun is a genuine cold start" => lab already relies on overlay being ephemeral.
- run page: "all Slurm tasks on an individual compute node share the same container, including its filesystem"; image cache at .edf_imagestore in scratch.

## servekit jit_cache (cd8f0b8, #18 d866545)
- jit_cache.py:16-22 CACHES = TRITON_CACHE_DIR->triton, TVM_FFI_CACHE_DIR->tvm-ffi, FLASHINFER_WORKSPACE_BASE->flashinfer. NODE_LOCAL_ROOT=/tmp/servekit/caches (L14) "same path every time -- ninja records absolute paths"; not /dev/shm because container mounts it noexec (L10-13).
- prepare.py:84-88 builds at /tmp/servekit/caches/<artifact-name>, copytree to <artifact>/servekit-caches (L138-141, "kept N JIT cache files"). launch.py:313-323 copies back to node-local, env_for(); "concurrent jobs share nothing"; new entries from serving never published back.
- Keyed per artifact (checkpoint+TP layout), populated only by `prepare` warmup (one "Hello" generate).
- coldstart-exp prepare log: "kept 137 JIT cache files" (Llama-70B TP4, 0.5.16).

## SGLang 0.5.20 (tag 94602c9) cache code
- python/sglang/__init__.py:11-14 calls redirect_third_party_caches() at import.
- srt/environ.py:1923-1947 third_party_cache_defaults(): setdefault TRITON_CACHE_DIR=$SGLANG_CACHE_DIR/triton, TORCHINDUCTOR_CACHE_DIR=.../inductor, CUDA_CACHE_PATH=.../nv, FLASHINFER_WORKSPACE_BASE=$SGLANG_CACHE_DIR (flashinfer appends .cache/flashinfer). Silently no-op if flashinfer imported first / inductor cache_dir() already called.
- environ.py:1164 SGLANG_CACHE_DIR default ~/.cache/sglang; 1154 SGLANG_DG_CACHE_DIR=$SGLANG_CACHE_DIR/deep_gemm (compile_utils.py:46 forces os.environ["DG_JIT_CACHE_DIR"]); 1168 SGLANG_CUTE_AOT_CACHE_DIR=$SGLANG_CACHE_DIR/cute_aot; 1171 SGLANG_JIT_CACHE_DIR=None -> cache.py:302 "~/.cache/sglang/jit" (does NOT follow SGLANG_CACHE_DIR); 1173 SGLANG_JIT_CACHE_DEBUG; 1177 SGLANG_JIT_CACHE_KEEP; 1181 SGLANG_CRASH_ON_JIT_COMPILE (raise on miss: "deployment that expects a pre-seeded cache fails loudly").
- Other SGLANG_CACHE_DIR users: custom_all_reduce_utils.py:266 (p2p cache), flashinfer_autotune.py:163, compilation/backend.py:403, runai_utils.py:123, rust_extensions/loader.py:358, flash_attention_v3.py:42 (lockfile), entrypoints/engine.py:1776-1796 (warns about pre-SGLANG_CACHE_DIR dirs), multimodal_gen gpu_worker.py:288.
- JIT cache (kernels/jit/utils/compile/cache.py): content-addressed; build_key = module args + direct source digests + normalized ninja text + TUs + env fingerprint (target arch incl. suffix, nvcc+host cc --version, versions of apache-tvm-ffi, flashinfer-python, deep_gemm, nvidia-mathdx, torch) [cache.py:196-276]; deps_key = transitive dep digests, self-verifying leaf [cache.py:286-360]; layout $DIR/<sm90a>/<module>/build-<key>/deps-<key>/{sgl_deps.json,.so}; anchor-normalized paths => RELOCATABLE (cache.py:91-145); publish = staging dir .staging-<uuid4> + os.rename dir (cache.py:497-513, loader.py:161-182) "random rather than pid-based: the cache root is meant to be a shared mount"; build lock = fcntl.flock on <scope>/.lock (loader.py:186-205), only dedups work; read path takes no lock; utime failure on read-only mount ignored (cache.py:340-348).
  => flock ENOSYS on a noflock Lustre would raise on a miss (loader.py:202, no fallback).

## vLLM facts (v0.33.0 shared clone 4ba787e; v0.32.0 in opportunities vendor 5df63a7)
- envs.py:35 VLLM_CACHE_ROOT default ~/.cache/vllm; L340-343 get_default_cache_root honours XDG_CACHE_HOME; L714-717.
- envs.py:375-398 VLLM_DISABLE_COMPILE_CACHE; VLLM_USE_AOT_COMPILE default 1 if torch>=2.10 and cache enabled; VLLM_USE_MEGA_AOT_ARTIFACT default 1 if torch>=2.12.0.dev and AOT; L2145 VLLM_COMPILE_CACHE_SAVE_FORMAT binary|unpacked (default binary); L1626 VLLM_DEEP_GEMM_WARMUP skip|full|relax (default relax); L1795 VLLM_FLASHINFER_AUTOTUNE_CACHE_DIR (None); L281/1986 VLLM_ENABLE_STARTUP_PLAN (0).
- compilation/backends.py:1056-1107 compile cache dir = VLLM_CACHE_ROOT/torch_compile_cache/<sha256(env_hash, config_hash, code_hash, compiler_hash)[:10]>/rank_<r>_<dp>/<prefix>; config/model.py:457-525 ModelConfig.compute_hash ignores tokenizer, served_model_name, seed... but INCLUDES `model` (path), `model_weights`, `revision`, `max_model_len` => vLLM compile cache is PER CHECKPOINT PATH.
- compiler_interface.py:29-42 initialize_inductor_cache(base): TORCHINDUCTOR_CACHE_DIR=<base>/inductor_cache always; TRITON_CACHE_DIR=<base>/triton_cache unless user set TRITON_CACHE_DIR. decorators.py:554-593 AOT: VLLM_CACHE_ROOT/torch_compile_cache/torch_aot_compile/<hash>/... and TORCHINDUCTOR_CACHE_DIR=<that>/inductor_cache.
  => Inductor cache per vLLM hash (per checkpoint); Triton shared across checkpoints only if TRITON_CACHE_DIR set explicitly.
- v1/worker/startup_plan.py (new in 0.33): VLLM_ENABLE_STARTUP_PLAN=1 persists KV-cache memory profiling result in VLLM_CACHE_ROOT/startup_plan/startup_plan_<fp>.json, fingerprint = vllm version, vllm_config hash, device name/mem/capability, torch, cuda, rank, world_size; tmp.<pid> + os.replace (L176-189). Skips memory profiling + CUDA-graph memory estimation.
- v0.32 env_override.py:163-183 on import vllm: PYTORCH_NVML_BASED_CUDA_CHECK=1, TORCHINDUCTOR_COMPILE_THREADS=1 (!), setdefault TRITON_CACHE_AUTOTUNING=1 (autotune results to TRITON_CACHE_DIR), setdefault TILELANG_CLEANUP_TEMP_FILES=1.
- v0.32 utils/jit_monitor.py: --jit-monitor-mode warn|error after warmup; hooks CuTeDSL cute.compile, Triton autotune print + jit_post_compile_hook, TileLang @tilelang.jit first compile. Reusable for a hit report.
- TileLang users in vLLM 0.32: model_executor/kernels/mhc/tilelang_kernels.py (mHC = DeepSeek-V4 hyper-connections), model_executor/warmup/jit_warmup_tilelang_helper.py, tilelang_utils/__init__.py.
- Pins (raw requirements/cuda.txt): v0.30.0: torch 2.13.0, flashinfer-python 0.6.18.post1 (+flashinfer-cubin), apache-tvm-ffi 0.1.11, tilelang 0.1.12, cutlass-dsl 4.7.1, quack 0.6.5, tokenspeed-mla 0.1.8, humming 0.1.12, numba 0.65.0. v0.33.0: torch 2.14.0, flashinfer 0.7.2.post1, tvm-ffi 0.1.11, tilelang 0.1.12, cutlass-dsl 4.8.0, tokenspeed-mla 2.0.3, humming 0.1.19, instanttensor>=0.2.2. v0.32.0: torch 2.13.0, flashinfer 0.7.0.post1, tvm-ffi 0.1.11, tilelang 0.1.12, cutlass-dsl 4.8.0.
  => TileLang 0.1.12 + tvm-ffi 0.1.11 constant across 0.30-0.33, so TileLang analysis holds whatever vllm_base pins in that range.

## TileLang 0.1.12
- env.py:337 TILELANG_CACHE_DIR ~/.tilelang/cache; 338 TILELANG_TMP_DIR = $CACHE/tmp; 342 TILELANG_DISABLE_CACHE; 345 TILELANG_KERNEL_CACHE_USE_LIB_STAMP (0); 347 TILELANG_CLEANUP_TEMP_FILES; 360 TILELANG_AUTO_TUNING_DISABLE_CACHE; TILELANG_PRINT_ON_COMPILATION=1 prints kernel name on compile (miss signal).
- cache/kernel_cache.py: namespace $CACHE/<version>/<platform-machine>/kernels/<key>; key = sha256{func: sha256(TIR script), out_idx, args, target str, target_host, execution_backend, pass_configs, compile_flags, version} (L241-282, L137-147). No nvcc/driver in key (lib stamp optional).
- write: staging dir under <ns>/.staging + os.rename (L475-547); files via tmp in TILELANG_TMP_DIR + os.replace (L460-473) => TMP must be same FS as cache. Stale staging cleanup >1h at singleton init (L218-238). No flock. Lustre-safe pattern.
- _create_dirs at KernelCache init creates ~/.tilelang/cache/{tmp,<ns>/kernels,<ns>/.staging} (L192-197) -> if container HOME were the real home, ~/.tilelang/cache/... would exist; #270 says ~/.tilelang exists but is EMPTY -> consistent with HOME not shared (dir likely from an interactive/login-node import), needs probe.

## Triton 3.7.1 (5d6048a)
- knobs.py:343-353 TRITON_HOME default ~/ ; TRITON_CACHE_DIR default $TRITON_HOME/.triton/cache; TRITON_CACHE_MANAGER, TRITON_REMOTE_CACHE_BACKEND (Redis backend built in, cache.py:146-166), TRITON_ALWAYS_COMPILE, TRITON_CACHE_AUTOTUNING (L375).
- cache.py:102-126 put(): tmp dir "tmp.pid_<pid>_<uuid4>" inside the key dir + os.replace (atomic file), no lock.
- cache.py:73-100 get_group/put_group: __grp__<name>.json stores ABSOLUTE child_paths; get_group keeps only paths that exist => compiler.py:265-270 metadata_path None -> recompile. => Triton disk cache is NOT relocatable: must be read at the same absolute TRITON_CACHE_DIR it was written at.
- key: compiler.py:245-251 get_cache_key(src, backend, options, env_vars=get_cache_invalidating_env_vars()); triton_key() hashes frontend/compiler/backend sources + libtriton (cache.py:280+).

## Inductor (torch 2.13.0)
- runtime/cache_dir_utils.py:14-33 TORCHINDUCTOR_CACHE_DIR default <tempfile.gettempdir()>/torchinductor_<user> (i.e. /tmp/torchinductor_<user>); triton_cache_dir(): TRITON_CACHE_DIR or <inductor>/triton/<device>.
- codecache.py:524-552 write_atomic: tmp ".{pid}.{thread_ident}.tmp" in same dir + rename => pid/tid collision possible across nodes on a shared dir (SGLang code comment calls this out for its own design).
- codecache.py:422-426 get_lock_dir=<cache>/locks; FileLock (torch/utils/_filelock.py wraps `filelock` pkg -> fcntl.flock) at L2850, 3779-3879, 4486, 4525, 4940, 5237. filelock: old versions raise NotImplementedError on ENOSYS; current main falls back to soft lock (filelock _unix.py). Lustre noflock -> ENOSYS.
- FX graph cache key includes torch_key, system info (triton key, device name, cuda version), inductor config (L1490-1515); counters fxgraph_cache_hit/miss/bypass (L2246-2317) => hit report source.

## Lustre flock
- mount.lustre: flock (coherent, sync MDS RPC per op), localflock (node-local only), noflock (ENOSYS). Default flock since Lustre 2.13 (LU-10885, per wiki); older default noflock. capstor setting unknown.
