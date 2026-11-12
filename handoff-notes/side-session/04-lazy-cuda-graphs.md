# 04 lazy CUDA-graph capture: hand-over (12 Nov 2026)

Folder: opportunities/04-lazy-cuda-graphs/ (WRITEUP.md, STATUS.md, MODAL.md).

- **Patch** for SGLang v0.5.20: `patches/0001-...patch` (9 files, +962/-32).
  - Applies to a fresh clone of the tag. 34 tests pass. `modal_app.py --dry-run` passes. No GPU run yet.
  - It captures a trace-chosen startup set, then the rest one bucket per idle scheduler iteration, in EV order (Smith's rule), in lockstep on all TP ranks (rank-0 tick in the request broadcast plus a MIN all-reduce).
- **Predictions** (trace simulation, eager penalties modelled):
  - 70B TP4 cold start 116.8 s -> ~80 s;
  - capture-caused latency per launch 5-28x lower;
  - at most 0.06-3% of requests wait for one step, max <= 1.35 s.
- **Prior art to cite:**
  - vLLM PR #57899 (open): post-wake idle recapture, largest first, FULL graphs only;
  - RFC #20098 and PR #23184 (closed);
  - Medusa (ASPLOS '25) and Foundry (arXiv 2604.06664): offline artifacts;
  - Jikes RVM: cost-benefit recompilation queue.
- **For 01 (#270):** the 16-18 s first-bucket one-off is JIT/allocator. Lazy capture only moves it; persistent caches remove it. The two compose.
- **For vLLM work:** a peer read v0.30+ and found TileLang mHC kernels compile per dynamic `num_tokens` via the JIT-warmup contract. If so, deferring capture would not defer #270's 40-min JIT. Unverified; needs the log.
- **Bug class worth knowing:** any mid-serving prefill capture in SGLang must call `reset_index_buffers()` first. Otherwise the warmup forwards write KV into slots the radix cache owns.
