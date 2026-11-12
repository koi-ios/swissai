# Env TOML diffs for #270 (draft; not posted)

Commit 2 of `patches/` (against model-launch main @02b74ce). For PR #272 (files renamed to `*_nvidia.toml`) the same lines are in `patches/after-272/0001-*.patch`, replayed with rename detection onto the PR head (3d57ae2).

`{jit_cache}` is resolved by sml on the batch host, like `{arch}` (commit 1): the fixed node-local overlay `/tmp/sml-jit-cache` in the default `overlay` mode, or `<shared-caches>/<image key>/<gpu arch>` in `shared` mode.

```diff
diff --git a/src/swiss_ai_model_launch/assets/envs/sglang_base.toml b/src/swiss_ai_model_launch/assets/envs/sglang_base.toml
index fa1873b..fd64188 100644
--- a/src/swiss_ai_model_launch/assets/envs/sglang_base.toml
+++ b/src/swiss_ai_model_launch/assets/envs/sglang_base.toml
@@ -30,6 +30,17 @@ FI_CXI_DISABLE_HOST_REGISTER = "1"
 OFI_NCCL_DISABLE_DMABUF = "1"
 SGL_ENABLE_JIT_DEEPGEMM = "0"
 VLLM_ALLREDUCE_USE_SYMM_MEM = "0"
+# Persistent JIT/compile caches (#270). sml resolves {jit_cache} to a node-local copy of
+# <shared-caches>/<image key>/<gpu arch>/ and publishes new entries back (docs/jit-cache.md).
+# Generated and checked by cache-audit against SGLang 0.5.20 (torch 2.13.0, Triton 3.7.1).
+SGLANG_CACHE_DIR = "{jit_cache}/sglang"               # import sglang puts Triton, Inductor, CUDA ComputeCache, FlashInfer,
+                                                      # DeepGEMM, cute_aot (FA4) and autotune results under it
+SGLANG_JIT_CACHE_DIR = "{jit_cache}/sglang-jit"       # SGLang's own JIT kernels (custom all-reduce v2, KV store):
+                                                      # does NOT follow SGLANG_CACHE_DIR
+TRITON_CACHE_DIR = "{jit_cache}/triton"               # explicit (SGLang only setdefaults it)
+TILELANG_CACHE_DIR = "{jit_cache}/tilelang"           # DeepSeek-V4 mHC kernels; SGLang does not redirect it
+TVM_FFI_CACHE_DIR = "{jit_cache}/tvm-ffi"             # tvm-ffi's torch DLPack add-on (SGLang's kernels no longer use it)
+CUTE_DSL_CACHE_DIR = "{jit_cache}/cute-dsl"
 
 [annotations]
 com.hooks.aws_ofi_nccl.enabled = "true"
diff --git a/src/swiss_ai_model_launch/assets/envs/vllm_base.toml b/src/swiss_ai_model_launch/assets/envs/vllm_base.toml
index cb88d0b..cf56ea5 100644
--- a/src/swiss_ai_model_launch/assets/envs/vllm_base.toml
+++ b/src/swiss_ai_model_launch/assets/envs/vllm_base.toml
@@ -29,6 +29,17 @@ FI_CXI_DEFAULT_TX_SIZE = "32768"
 FI_CXI_DISABLE_HOST_REGISTER = "1"
 OFI_NCCL_DISABLE_DMABUF = "1"
 VLLM_ALLREDUCE_USE_SYMM_MEM = "0"
+# Persistent JIT/compile caches (#270). sml resolves {jit_cache} to a node-local copy of
+# <shared-caches>/<image key>/<gpu arch>/ and publishes new entries back (docs/jit-cache.md).
+# Generated and checked by cache-audit against vLLM 0.32.0 (torch 2.13.0, Triton 3.7.1).
+VLLM_CACHE_ROOT = "{jit_cache}/vllm"                  # torch.compile/AOT artifacts (per checkpoint), FlashInfer autotune
+TRITON_CACHE_DIR = "{jit_cache}/triton"               # explicit: vLLM otherwise keeps Triton per compile hash, i.e. per checkpoint
+DG_JIT_CACHE_DIR = "{jit_cache}/deep_gemm"            # explicit: paths that bypass vLLM's lazy init fall back to ~/.dj
+TILELANG_CACHE_DIR = "{jit_cache}/tilelang"           # DeepSeek-V4.x mHC kernels; nothing sets it today
+FLASHINFER_WORKSPACE_BASE = "{jit_cache}/flashinfer"  # FlashInfer appends .cache/flashinfer/<version>/<arch>
+TVM_FFI_CACHE_DIR = "{jit_cache}/tvm-ffi"
+CUTE_DSL_CACHE_DIR = "{jit_cache}/cute-dsl"
+CUDA_CACHE_PATH = "{jit_cache}/nv"
 
 [annotations]
 com.hooks.aws_ofi_nccl.enabled = "true"
```
