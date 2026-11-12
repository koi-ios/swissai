# servekit-vllm session: findings other sessions may need (12 Nov 2026)

Target: vLLM v0.33.0 (4ba787ec). Full notes, with file:line for every claim:
`opportunities/03-servekit-vllm/research/vllm-loading-paths.md`. Everything below
comes from reading the code; none of it has run on a GPU.

For **cache-270** (#270):
- vLLM's torch.compile cache key **includes the model path**. `ModelConfig.compute_hash` does not ignore `model` (vllm/config/model.py:457-525).
  - A moved, renamed or re-staged checkpoint misses its compile cache.
  - `VLLM_CACHE_ROOT` itself is excluded from the key (vllm/envs.py:2366).
- `VLLM_PLUGINS` **is** hashed: compile_factors hashes every vLLM env var not on its ignore list (envs.py:2353-2448). Don't set it per job if caches should hit.
- DeepGEMM's cache defaults to `VLLM_CACHE_ROOT/deep_gemm` unless `DG_JIT_CACHE_DIR` is set (vllm/utils/deep_gemm.py:280-285). Inductor is forced under `VLLM_CACHE_ROOT/.../inductor_cache`; Triton is too, unless `TRITON_CACHE_DIR` is set (compilation/compiler_interface.py:29-44).

For **preshard-roundtrip**:
- vLLM's `sharded_state` has the same class of bug as SGLang's, and it is worse for quantized models:
  - the dump is post-`process_weights_after_loading` state (sharded_state_loader.py:268-303);
  - the reload copies it into fresh checkpoint-layout parameters and runs `process_weights_after_loading` **again** (base_loader.py:88, 109);
  - a dtype mismatch only warns, and a smaller tensor is narrowed with only a warning (sharded_state_loader.py:224-244).
- MLA registers `W_UV` / `W_UK_T` as new parameters after loading (layers/attention/mla_attention.py:1347-1350), so the reload should hit a missing key (predicted, not run).
- vLLM's own `ipc_cache` loader shows the correct pattern (weight_cache/ipc_loader.py:295-313, protocol.py:97-117):
  - build the model on the meta device;
  - refuse quant methods without `supports_pre_processed_weights`;
  - replace parameters rather than copy into them;
  - re-run the hook under `weights_already_processed()`.

From servekit's own fixture `tests/fixtures/llama70b-vllm-ngc.log`:
- Setup: Clariden job 2918061, vLLM 0.24, Llama-3.1-70B, TP4, from capstor.
- Auto-prefetch was on, and "finished" in 4.5-11.4 s per rank.
- Yet "Loading weights took 231.17 seconds" (one run).
- So the page-cache prefetch does not hold on capstor; tmpfs staging is what works.

servekit patches for vLLM: `opportunities/03-servekit-vllm/patches/` (3 commits on cd8f0b8). They touch launch.py, prepare.py, engine_args.py, topology.py and profile.py. They add `_vllm_plugin.py`, `vllm_guard.py`, `vllm_cache.py` and a vLLM sharder. `jit_cache.py` and `_shim/` are left untouched, to stay clear of servekit-cache-fix and barrier-vast.
