# Working notes (idea 6: servekit correctness)

Running log for context compaction. Facts carry a source (file:line or URL). Started 11 Nov 2026.

## Where things are

- Scratch: `/private/tmp/claude-501/-Users-koiios--claude-spares-spare-qHWvhZ/85cbb828-b415-43aa-adc6-b088d62505ae/scratchpad/servekit-correctness/`
  - `.venv/` (Python 3.12, torch 2.15 CPU, safetensors 0.8.0, transformers 5.22.2, scipy)
  - `src/sglang-v0.5.16`, `src/sglang-v0.5.20`, `src/sglang-v0.5.23` (sparse: model_loader, models/deepseek*, kimi*, glm*, plugins)
  - `src/vllm-v0.33.0` (sparse: model_loader, layers/mla.py, layers/attention, models/deepseek_v2.py, model_executor/utils.py, tests/model_executor/model_loader)
  - `gh/` raw GitHub API JSON (servekit #14, #19 + comments; sglang PR #35715 + comments)
- Shared read-only servekit: `$SHARED/repos/servekit` @cd8f0b8.
- Sibling sessions: `opportunities/07-verify-tolerance` and `opportunities/08-preshard-roundtrip` (another coordinator, started 20:05) overlap with this idea. Read-only for me. Qwen3-0.6B and Qwen3-0.6B-FP8 are already in `~/.cache/huggingface/hub` (downloaded by that session; I do not download models).

## Part (a): #19, what is already known (do not re-pitch)

- servekit #19 (open, youssef62, 15 Aug 2026): Kimi-K3 TP4/PP8/EP4 `KeyError: '...self_attn.attn_mha.kv_b_proj.weight'`. Youssef already wrote the root cause in the issue: lazy alias on first forward + `_filter_subtensors` keeps the lexicographically smaller key. He frames it as "the same family" as #14: "the checkpoint is written from a model whose module graph is not the one that reads it back".
- Comment 4 Sep: upstream fix proposed as sgl-project/sglang#35715 (`object.__setattr__` so the alias is never a submodule); backported in servekit #24 as `servekit._patches.sglang_35715`; "keeping until it's solved upstream".
- sglang#35715: open, unmerged since 20 Aug 2026, all three CI runs red, no unit test (box unchecked), two pings to maintainers without reply (gh/repos_sgl-project_sglang_pulls_35715.json). Upstream issue sgl-project/sglang#35702.
- servekit commit 0d54a36 (#24) message already states the lexicographic tie-break; the e2e gate checks "no self_attn.kv_b_proj.weight in any shard" (model-specific).
- servekit #25 (cd8f0b8): `_patches/__init__.py apply_all()` runs every `sglang_*.py`; a patch that does not fit is dropped with one log line and the engine runs unpatched.
- #14 (closed): mxfp4 `process_weights_after_loading` moves expert weights off the module (`del layer.w13_weight`), so they are absent from `state_dict()`; 69% of gpt-oss-20b weights silently zero; caught by verify (16/16 fail). Fix: `quant_guard.py` refuses 6 rule families (#16). Upstream sgl-project/sglang#34448.

## Part (a): new findings (11 Nov)

1. **The #19 backport does not apply to the platform's SGLang 0.5.20 (nor 0.5.23).** 0.5.20 changed the forward fill-in to
   `if self.attn_mha.kv_b_proj is None and hasattr(self, "kv_b_proj"):` (src/sglang-v0.5.20/python/sglang/srt/models/deepseek_v2.py:2070), so `OLD_FORWARD` occurs 0 times; `main()` returns 1; `apply_all` drops it. The lazy alias is still there (deepseek_v2.py:1920 init `= None`, :2070-2071 fill). Checked by counting the patch's own anchors against each tag (v0.5.16 applies; v0.5.20 and v0.5.23 dropped).
   - kimi_k3.py exists in 0.5.20 (`KimiK3MLAAttention(DeepseekV2AttentionMLA)`, kimi_k3.py:2058), so Kimi-K3 inherits the lazy alias.
   - kimi_k3.py:2097 `del self.kv_b_proj` happens in `__init__` (GGUF split K/V only), so both sides agree; that is why 0.5.20 added the `hasattr` guard.
   - Same lesson as the 0.5.20 cache directories: text-anchored backports rot across releases.
2. **SGLang 0.5.20 `_filter_subtensors`** (loader.py:1687-1730) still has "Same tensors, keep the one with the smaller key" (:1725-1727). New in 0.5.20: non-contiguous tensors are kept as-is before `get_end_ptr` (:1707-1715). `ShardedStateLoader.load_model` (:1748-1809): init -> quant `process_weights_after_loading` on the fresh model -> `_filter_subtensors(model.state_dict())` -> copy by checkpoint key (`state_dict[key]` KeyError) -> "Missing keys" check -> `_post_load_weights(model)` (new vs 0.5.16?). `save_model` (:1811-1845) dumps `_filter_subtensors(model.state_dict())` with `save_file` and no metadata.
3. **SGLang 0.5.20 ships a new `PreshardedModelLoader`** (loader.py:1848+): "Dump/reload post-process weights under <model_path>/presharded/<subdir>/", with `checksum.json`, a `READY` file, content hashes, `verify_on_load`. Upstream now does part of what servekit prepare does. Must read fully (does it dedup by `_filter_subtensors`? does it hash?).
4. **vLLM 0.33** (src/vllm-v0.33.0):
   - `sharded_state_loader.py:137-172` `_filter_subtensors` is the original: lexicographic tie-break, and `get_end_ptr(t)` (`t.view(-1)`) is computed for every tensor before any contiguity check (:158), so any non-contiguous tensor in `state_dict()` raises.
   - Load order is `create_model -> load_weights (sharded copy) -> process_weights_after_loading` (base_loader.py:63-115). The save dumps `model_runner.model.state_dict()` after processing (v1/worker/gpu_worker.py:1490-1503). So the loaded state is post-processed and then processed again.
   - MLA: `kv_b_proj` is registered under three names at construction (deepseek_v2.py `self.kv_b_proj`, layers/mla.py:97 wrapper, mla_attention.py:477 `MLAAttention.kv_b_proj`). Construction-time aliases are consistent on both sides; lexicographic min is `self_attn.kv_b_proj.weight` ('k' < 'm').
   - But on CUDA, `MLAAttention.process_weights_after_loading` registers NEW parameters `W_UV = W_UV.transpose(0, 1)` and `W_UK_T = W_UK.permute(1, 2, 0)` via `replace_parameter` (mla_attention.py:1345-1349; utils.py:71-148 registers a new Parameter when none exists). For an unquantized kv_b_proj with weight dtype == act dtype, `split_kv_b_proj` returns views of `kv_b_proj.weight` (`get_and_maybe_dequant_weights` returns `weight.to(out_dtype)`, a no-op; then `.T.view(...).split(...)`, mla_attention.py:408-423). So W_UV/W_UK_T are non-contiguous views sharing kv_b_proj's storage.
   - Prediction: vLLM 0.33 `save_sharded_state` on a BF16 MLA model raises `RuntimeError: view size is not compatible...` inside `_filter_subtensors`; and even if it saved, `load_weights` would `KeyError` on `...W_UV` because the fresh model has not processed yet. To confirm on CPU with the real functions.
   - ROCm aiter paths register `W_K`, `W_K_scale`, `W_V`, `W_V_scale` (mla_attention.py:1286-1311): same family. CPU AMX path empties kv_b_proj.weight to numel 0 (:1248-1252): dropped by `if tensor.numel()`, so "Missing keys" at load.
   - vLLM 0.33 has `weight_checksum.py` (sha256 per named parameter, `compute_tensor_digests`) "for verifying weight updates": prior art for a white-box check.
