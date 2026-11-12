<!-- DRAFT design note for a vLLM port. Not posted. Hook locations are at vLLM v0.33.0 (commit 4ba787e), taken from research/related_work_engines.md §1, which read the files at that tag; re-check line numbers before writing code. -->

# Lazy CUDA graph capture in vLLM (Model Runner V2): design note

The design is the same as the SGLang patch. Capture a traffic-chosen startup set, dispatch bucket-exact, then capture the rest one descriptor per step while the engine is idle, in expected-value order, in lockstep on every worker. vLLM's process structure makes the lockstep part easier than in SGLang. The engine core is a single leader, and `collective_rpc` already delivers the same call to every worker in order. Credit to vLLM PR #57899 (open), which built idle-boundary recapture for the sleep/wake path with exactly this RPC pattern.

## Why it matters there

- **The default grid is 51 sizes** at max 512: [1, 2, 4] + range(8, 256, 8) + range(256, 513, 16) (`config/vllm.py` `_set_cudagraph_sizes`). Under `FULL_AND_PIECEWISE`, PIECEWISE and FULL graphs are both captured.
- **Startup captures twice.** `profile_cudagraph_memory()` does a throwaway pass before the real one.
- **Measured in model-launch #270:** DeepSeek-V4.1-Flash spends about 40 min of a 70-min start in capture-time JIT (51 graphs).
  - Whether deferring capture also defers that JIT is open.
  - Per a peer session's reading of v0.30+, TileLang mHC kernels use `T.dynamic("num_tokens")`, and the explicit JIT-warmup contract compiles them before capture. If so, lazy capture saves only the per-graph capture time, not the 40 min.
  - The #270 log is needed to tell.

## Hooks (MRV2, the default since v0.29)

| what | where (v0.33.0) | change |
|---|---|---|
| startup capture | `v1/worker/gpu_worker.py` `_compile_or_warm_up_model()` → `self.model_runner.capture_model()` (L993-994) | pass the startup subset |
| capture loop | `v1/worker/gpu/model_runner.py` `capture_model()` L1086-1162 → `cudagraph_utils.py` `ModelCudaGraphManager.capture()` L613-788 → `CudaGraphManager.capture()` L424-515 (sets `_graphs_captured = True` at L515) | split into `capture(descs)` and `capture_one(desc)`; keep the process-wide pool |
| dispatch | `execute_model()` L1764 → `dp_utils.py` `dispatch_cg_and_sync_dp()` L211 → `CudaGraphManager.dispatch()` L523-553 | replace the all-or-nothing `_graphs_captured` flag with a per-descriptor captured set; a batch whose default (capture-all) descriptor is not captured yet runs `NONE` (eager) |
| FULL replay | `run_fullgraph()` L555-568, `assert desc in self.graphs` at L560 | unchanged; dispatch guarantees it |
| PIECEWISE replay | inner `CUDAGraphWrapper.__call__` (`compilation/cuda_graph.py`) captures on a miss when not guarded | must never see an uncaptured key from dispatch, or must be armed with `set_cudagraph_capturing_enabled(False)` outside deferred steps (MRV2 never arms it today) |
| memory | `profile_cudagraph_memory()` (MRV2 `cudagraph_utils.py` L922-1028) reserves for all graphs before the KV cache exists | keep it, so deferred graphs fit the reserve; #57475 shows the estimate already under-reserves at high utilization, so add a free-memory check before each step |
| idle detection | `EngineCore` busy loop: when the scheduler has no requests it blocks on the input queue | poll with a timeout while deferred work remains; after idle ≥ hysteresis, `collective_rpc("capture_deferred", args=(mode, num_tokens))` |
| DP | `sync_cudagraph_and_dp_padding()` (`dp_utils.py` L45-208) already all-reduces `cg_mode` over the DP group | DP ranks must step the same descriptor in the same idle window; first version TP/PP only |

## Rank consistency

- The engine core owns the queue and sends the descriptor, so workers do not need replicated scheduling state (unlike SGLang).
- A worker that fails a capture while its peers block inside a collective must not be left hanging. #57899 terminates the executor in that case; the SGLang patch avoids it by agreeing on headroom before the step.

## Open questions

1. Is the TileLang/DeepGEMM JIT in #270 per captured size? This decides whether lazy capture shortens the 70-min start or only the capture part.
2. Can piecewise capture under torch.compile be deferred safely? #57899 excludes torch.compile.
3. Does the `FULL_AND_PIECEWISE` throwaway memory-profiling capture leave enough headroom to grow the pool lazily?
