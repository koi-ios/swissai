# Draft PR for eth-easl/servekit (send after the backport fix, ideally with Modal results)

**Branch:** `preshard-roundtrip`, commit 2 of 2 (`patches/servekit/0002-*.patch`).

**Title:** feat: raw presharding, a deep-state round-trip check, and the gpt-oss default-backend guard

---

## Why

`sharded_state` saves `state_dict()` *after* SGLang post-processed the model. It then reloads that into a model whose post-processing already ran on placeholder values: v0.5.20 `loader.py:1765-1768` runs post-processing on the placeholders, and `:1781-1807` copies the saved state in. Every tensor the engine keeps outside `state_dict()`, or creates lazily, is lost silently or breaks the load. Three instances on 0.5.20:

| instance | where | outcome |
|---|---|---|
| MXFP4 experts on the `triton_kernel` runner (#9, #14, sglang#34448) | `mxfp4.py:1015-1043` parks them on the quant method | loads; 69% of gpt-oss-20b's bytes missing (Youssef's measurement) |
| MLA `attn_mha.kv_b_proj` (#19, sglang#35702) | `deepseek_v2.py:1920`, `:2070` | `KeyError` on load |
| `GemmaRMSNorm.gemma_weight`, a non-persistent buffer derived in the weight loader | `layernorm.py:1068-1089` | loads; stays 1.0 (predicted from code, CPU-reproduced) |

`quant_guard` keys on the literal `--moe-runner-backend triton_kernel` flag. SGLang picks that backend itself for gpt-oss below SM100 (`model_overrides/gpt_oss.py:106-125` in 0.5.20, and `server_args.py` since at least 0.5.10). So a GH200 launch with no flag passes the guard and serves zeros.

## What

- **Raw preshard** (`servekit/rawshard/`, `servekit launch --preshard-format raw`).
  - During one ordinary load, each checkpoint tensor the weights iterator yields is wrapped in a `__torch_dispatch__` subclass. Every non-view op records the strided byte box of the view it reads.
  - Each rank stores exactly those bytes (`servekit-raw-rank-{r}-part-{p}.safetensors` plus a JSON index), so a rank's files hold its shard and its replicated tensors.
  - Serving replays them through the **ordinary** loader. Plugin hooks on `DefaultModelLoader._get_weights_iterator`, `load_model` and `commit_model_weights` handle this, and no SGLang patch is needed.
  - Every weight loader, post-processing step and lazy alias then runs exactly as in a default load.
- **Fail-closed checks** on the replay:
  - the recorded parallel layout and engine must match the current ones;
  - every read is checked against the stored boxes (`SERVEKIT_RAW_VERIFY=1`, the default).
- **Bounded memory while writing:** a tensor's bytes are copied out as soon as no wrapper of it is alive, and parts are flushed at 5 GB.
- **Deep-state check** (`servekit/deepstate.py`, `SERVEKIT_DEEPSTATE_DIR`, `servekit deepstate REF CAND`).
  - It fingerprints every tensor a loaded model holds: parameters, buffers, and tensors on quant-method objects, plain attributes and dataclasses.
  - It compares two loads, with an optional second reference to ignore scratch buffers.
  - A state_dict-only comparison passes on the MXFP4 case; this one flags `quant_method.w13_weight_triton_tensor.storage.data`.
- **Guard.** It refuses gpt-oss MXFP4 on the default backend for `sharded_state`, and never refuses a raw preshard.

## Tests

- 176 pass, including:
  - replay exact at TP 1/2/4 and EP;
  - another rank's preshard caught;
  - a changed layout refused;
  - threaded loaders;
  - a loader that keeps the checkpoint tensor;
  - hooks driven through a stand-in with SGLang's signatures.
- The 2 pre-existing stager errors need Linux `O_DIRECT`.
- **Reproductions:** a separate suite runs SGLang 0.5.20's own `ShardedStateLoader`, MLA alias lines, MXFP4 triton branch and `GemmaRMSNorm` (AST-extracted, nothing re-implemented). It shows all three failures under `sharded_state` and exact round trips under raw.
- **CPU overhead** (median of 5 runs):
  - +66 µs per tensor to record;
  - +50 µs per tensor to replay with verification, +5 µs without;
  - at 8 MB per tensor, replay costs about the same as a plain load.

## Not yet done

- **GPU validation.** It is ready as a Modal app: DeepSeek-V2-Lite and gpt-oss-20b at TP=2, `servekit verify` against the default loader, deep state, and load time. It costs about $10–15.
- **Open costs:**
  - raw pays post-processing at every load;
  - prepare with DP > 1 is not covered;
  - draft models that share the checkpoint path are keyed `target/draft` but not GPU-tested.

🤖 Generated with [Claude Code](https://claude.com/claude-code)
