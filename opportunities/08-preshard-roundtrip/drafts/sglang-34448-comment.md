# Draft comment on sgl-project/sglang#34448

---

Status as of 11 Nov, for anyone landing here.

- **Released versions are still affected, with default flags.** For gpt-oss with `--moe-runner-backend auto`, v0.5.20 resolves to `triton_kernel` on anything below SM100, H100 and GH200 included (`arg_groups/model_overrides/gpt_oss.py:106-125`; `server_args.py` did the same since at least v0.5.10). So a plain `--load-format sharded_state` round trip of gpt-oss-20b on Hopper takes the path that drops the expert weights. There is no flag that would warn the user. A guard that only checks for an explicit `triton_kernel` flag misses this.
- **main no longer has this path.** #49660 (10 Nov) removed the `triton_kernel` runner and #49661 replaced its dequantization, so the instance goes away from the next release. v0.5.20 to v0.5.23 keep it.
- **The class remains.** Any state that post-processing or a weight loader moves off `state_dict()` is lost the same way. Example: `GemmaRMSNorm.gemma_weight` is a non-persistent buffer set inside its weight loader (v0.5.20 `layernorm.py:1084-1089`). `ShardedStateLoader` copies the parameter directly and never calls the loader, so the buffer stays 1.0. This is predicted from the code and reproduced on CPU, but not yet seen on a GPU. `PreshardedModelLoader._rebind_parameter_aliases` already special-cases exactly this buffer.

**Two generic fixes worth considering, both prototyped in servekit:**
1. A **round-trip check** that compares every tensor reachable from the model, including quant-method attributes and dataclasses, not just `state_dict()`. On this bug it flags `quant_method.w13_weight_triton_tensor`, while a `state_dict()` comparison passes.
2. A **raw sharded format**. It stores, per rank, the checkpoint bytes that rank's normal load read, and replays them through the normal loader, so post-processing always runs on real weights.

Either could back #41616's fail-loud direction without a per-quant-method denylist.
