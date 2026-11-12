# 08 preshard round trip: hand-over (12 Nov 2026)

Everything is in `opportunities/08-preshard-roundtrip/`: WRITEUP.md, STATUS.md, MODAL.md, `patches/`, `drafts/`, `results/`.

## Findings other sessions may need
- **servekit's MLA backport (`_patches/sglang_35715.py`) does not apply to SGLang ≥ 0.5.19,** including the platform's 0.5.20.
  - Cause: SGLang 2d8484740d (#35314) added `and hasattr(self, "kv_b_proj")`. `apply_all` drops the patch with one log line, so MLA models (GLM-5.x, DeepSeek, Kimi) fail presharded loads again.
  - Fix: commit a8ec9d2 on vendor/servekit branch `preshard-roundtrip`.
- **servekit's quant guard misses gpt-oss on the default backend.** SGLang 0.5.20 auto-selects `triton_kernel` below SM100 (`model_overrides/gpt_oss.py:106-125`).
- **`GemmaRMSNorm.gemma_weight` goes stale after `sharded_state` loads.** Predicted from code and reproduced on CPU, not on a GPU.
- **Upstream (11 Nov):**
  - #35715 is open with no review;
  - #34558 closed unmerged;
  - #34448 is open;
  - #35702 was auto-closed as inactive, though still present;
  - main removed the `triton_kernel` MoE runner (#49660, unreleased).
- **The verify-tolerance session may care:** `servekit/deepstate.py` compares every tensor a model holds byte for byte (`servekit deepstate`). It is a tolerance-free correctness check for presharding. Logprob tolerance only matters where two loads legitimately differ.

## Built
- **Raw preshard:** stores the checkpoint bytes each rank reads and replays them through SGLang's ordinary loader via servekit's plugin. Engine-agnostic core, so it would also serve a vLLM port.
- **Deep-state checker.**
- **Tests:** 176 servekit tests and 9 CPU reproductions against SGLang 0.5.20's own code.
- **Modal run:** ready but not run (about $10–15).
