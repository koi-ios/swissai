# Draft new issue for sgl-project/sglang (HOLD until confirmed on a GPU)

Confirm first: a Qwen3.5/Qwen3-Next or MiniMax-M3 checkpoint at TP≥2 with FlashInfer all-reduce fusion. Save with `sharded_state`, reload, and compare logprobs (or `servekit deepstate`) against the default loader.

---

**Title:** [Bug] sharded_state leaves `GemmaRMSNorm.gemma_weight` at its initial value

**Version:** v0.5.20; the same code is on main at 95999790a4.

**What happens.** `GemmaRMSNorm` keeps `gemma_weight = weight + 1` as a non-persistent buffer and fills it inside its weight loader (`layernorm.py:1076-1089`):

```python
self.register_buffer("gemma_weight", torch.ones_like(self.weight), persistent=False)
...
def _weight_loader(self, param, loaded_weight):
    param.data.copy_(loaded_weight)
    torch.add(param.data, 1.0, out=self.gemma_weight)
```

`ShardedStateLoader.load_model` copies saved tensors straight into `state_dict()` (`loader.py:1781-1803`). It never calls weight loaders, and non-persistent buffers are not in the checkpoint. So after a `sharded_state` load, `norm.weight` is correct but `norm.gemma_weight` is still `ones`. That equals what a checkpoint whose norm weights are all zero would give.

**Who reads `gemma_weight`:**
- `forward_with_allreduce_fusion` (FlashInfer fusion for the architectures in `_FLASHINFER_ALLREDUCE_FUSION_ARCHS`);
- the ROCm `vllm` path and the large-hidden NPU path;
- `qwen35_flashinfer_fusion.py`;
- MiniMax-M3's attention (`minimax_m3.py:1102`, `q_weight=self.q_norm.gemma_weight`).

**Reproduction (CPU, no GPU needed).** The `__init__` and `_weight_loader` above, with v0.5.20's `ShardedStateLoader.save_model`/`load_model` run unmodified on a one-norm model:
- the parameter round-trips;
- `gemma_weight` comes back as `ones(16)` instead of `weight + 1`.

**Fix options:**
- Recompute in a post-load hook that every loader calls: `ShardedStateLoader` already calls `model.post_load_weights()` through `_post_load_weights`.
- Derive the buffer lazily.
- Reuse `PreshardedModelLoader._rebind_parameter_aliases` in `ShardedStateLoader.load_model`; it already handles exactly this buffer (`loader.py:2181-2188`).
