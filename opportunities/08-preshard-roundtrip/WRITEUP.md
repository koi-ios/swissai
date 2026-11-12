# Presharding that cannot silently change the model

## Problem

servekit's fast load relies on SGLang's `sharded_state` format. That format saves `state_dict()` *after* the engine has post-processed the weights, then reloads it into a model whose post-processing already ran on placeholder values (loader.py:1765-1768 and :1781-1807 in v0.5.20). Anything a model keeps outside `state_dict()`, or creates lazily, is either lost silently or breaks the load:

- **MXFP4, gpt-oss** (sglang #34448, servekit #14). The triton path parks the swizzled experts on the quant-method object and deletes them from the layer (mxfp4.py:1015-1043).
  - Youssef measured 69.4% of the bytes dropped and 16/16 prompts wrong.
  - **New:** servekit's guard keys on the literal `--moe-runner-backend triton_kernel` flag. But SGLang 0.5.20 picks `triton_kernel` itself for gpt-oss below SM100 (model_overrides/gpt_oss.py:106-125), so a GH200 launch with no flag is not refused.
- **MLA** (sglang #35702, servekit #19). `attn_mha.kv_b_proj` is registered on the first forward (deepseek_v2.py:1920, :2070), so the canonical name is deduplicated away and the reload raises `KeyError`.
  - Kimi-K3 was prepared at 1.5 TB and then failed on every stage.
  - **New:** servekit's backport stopped applying at v0.5.19. SGLang commit 2d8484740d (#35314) added a `hasattr` guard, so on the platform's 0.5.20 the backport is dropped with one log line and MLA models fail again.
- **New, predicted:** `GemmaRMSNorm.gemma_weight` is a non-persistent buffer derived inside the weight loader (layernorm.py:1068-1089). After a `sharded_state` load it stays at 1.0, which acts as if the checkpoint norm were 0. It is read by the all-reduce-fusion, ROCm and NPU paths, Qwen3.5 fusion, and MiniMax-M3's `q_norm`.

**Upstream status (11 Nov 2026).** #35715 is open with no review. #35702 was auto-closed as inactive while the bug is still present on main. #34558 was closed unmerged on 15 Sep after a 4×H100 CI failure; #41616 (fail loudly) is open. On 10 Nov, main removed the `triton_kernel` runner (#49660, unreleased), which removes this instance but not the class.

SGLang 0.5.20 also added a second post-processed dumper, `PreshardedModelLoader` (loader.py:1848). It rebinds `gemma_weight`, `A_log`, `dt_bias` and `conv_weights` by hand, one alias at a time. A sibling session found that vLLM 0.33 has the same design and predicts it fails on MLA: `_filter_subtensors` calls `view(-1)` before checking contiguity, and the transposed `W_UV` is registered as a parameter (read, not run).

## Idea and what was built (branch `preshard-roundtrip` in vendor/servekit, 2 commits)

1. **Raw preshard** (`servekit/rawshard/`, `--preshard-format raw`).
   - During one ordinary load, every checkpoint tensor is wrapped in a `__torch_dispatch__` subclass. Any non-view op records the strided byte box of the view it reads.
   - Each rank stores exactly those bytes, so it holds its share and nothing else.
   - Serving replays them through the *ordinary* loader, via servekit's SGLang plugin hooks on `_get_weights_iterator`, `load_model` and `commit_model_weights`. So every weight loader, post-processing step and alias runs as in a default load.
   - Two checks make a mismatch fatal rather than silent: the parallel layout and engine are checked against the recorded ones, and every read is checked against the stored boxes (`SERVEKIT_RAW_VERIFY`, on by default).
2. **Generic round-trip checker** (`servekit/deepstate.py`, `servekit deepstate REF CAND`). It fingerprints every tensor a loaded model holds: parameters, buffers, and tensors reachable through plain attributes and dataclasses (`quant_method....storage.data`). A state_dict-only check passes on the MXFP4 case; this one does not.
3. **Fixes.** The MLA backport now matches both spellings, and the guard refuses gpt-oss on the default backend for `sharded_state` while allowing raw.

| approach | covers unknown models | load-time cost | upkeep |
|---|---|---|---|
| per-model fixes (#35715, #34558, rebinds) | no | none | one patch per alias; breaks on refactors (as at 0.5.19) |
| deep-state checker | detects, does not fix | none (prepare: one extra load) | none |
| raw preshard | fixes by construction | post-processing runs at load; about 5 µs per tensor, 50 µs with verification (CPU) | none |

Recommendation: prepare raw by default, and keep `sharded_state` only where the deep-state check passes and post-processing is expensive.

## Results so far: measured on CPU, predicted on GPU

- **Reproductions** (`repro/`, 9 pass). They run SGLang v0.5.20's own `ShardedStateLoader`, the MLA alias lines, the MXFP4 triton branch and `GemmaRMSNorm` (AST-extracted, listed in `results/cpu_repro.json`) on tiny models:
  - MLA raises `KeyError`;
  - MXFP4 loads silently and wrong (71% of the mini model's bytes dropped), state_dict check passes, deep-state check flags 3 hidden tensors;
  - `gemma_weight` goes stale;
  - servekit cd8f0b8's backport refuses 0.5.20, and its guard passes a bare gpt-oss launch.
  - Raw preshard round-trips all three exactly.
- **servekit tests:** 176 pass (TP 1/2/4, EP, threaded loads, wrong-rank and wrong-layout replays caught). The 2 errors are pre-existing: the stager tests need Linux `O_DIRECT`.
- **CPU overhead** (`bench/`, Apple M-series, median of 5 runs):
  - recording: +66 µs per tensor;
  - replay with verification: +50 µs per tensor;
  - replay without verification: +5 µs per tensor;
  - for 64 × 8 MB tensors at TP 4: 40 ms against 38 ms for a plain load.
- **GPU:** the Modal experiment is ready but has not been run (MODAL.md). It covers DeepSeek-V2-Lite on 2×L40S and gpt-oss-20b on 2×H100: verify against the default loader, deep state, and load time over 3 runs. It costs about $10–15. The expected outcomes are predictions.

## Alps and risks

- Raw preshard needs no SGLang change, only servekit's plugin, which is already installed. Multi-node staging slices by world rank (`raw_glob`).
- Risks:
  - post-processing cost at every load for heavily repacked models (measure on Kimi-K3);
  - dispatch overhead on very large expert counts (about 180k tensors means seconds with verification);
  - prepare with DP > 1;
  - draft models that share the checkpoint path (keyed `target/draft`, not GPU-tested).

## Pitch

servekit's presharding saves the model *after* SGLang has transformed it, and three transforms on the platform's 0.5.20 are lost: MLA, MXFP4, and Gemma norms. The backport for one of them has silently stopped applying. I built a preshard that stores only the checkpoint bytes each rank reads and replays them through SGLang's own loader, so it is correct by construction and checked on every load. I also built a checker that catches the remaining cases generically. Both have tests and CPU reproductions against 0.5.20's own code, and a $15 Modal run is ready to confirm them on DeepSeek-V2-Lite and gpt-oss-20b.
