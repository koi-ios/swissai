# Related work (engines): CUDA-graph capture sets and lazy / deferred / idle-time capture

Compiled 2026-11-11/12 for the write-up on **lazy, traffic-driven CUDA-graph capture**. The idea: serve after capturing a small set of sizes, capture the rest one at a time during scheduler idle time in lockstep across TP ranks, and promote hot sizes that run eagerly.

Companion file: `related_work_papers.md` covers papers (Medusa, Foundry, λScale, ServerlessLLM, …). This file covers **engines and their code, issues, PRs and RFCs**. SGLang's own capture code (v0.5.20) is being read separately, so it is not described here.

**Versions checked**
- **vLLM v0.33.0.** Latest release on PyPI (uploaded 2026-11-02). Tag commit `4ba787ec9959e566d3b587a796dff07673324da5`. v0.34.0rc0 was tagged 2026-11-08; differences are noted where relevant.
- **PyTorch v2.15.0.**
- **TensorRT-LLM v1.2.1** (latest stable on PyPI/GitHub) and **v1.4.0rc3** (2026-11-03).
- **SGLang:** latest is v0.5.23 (2026-10-30). Only issues, PRs and the LMSYS blog were read.

**How things were read**
- Code was read from `raw.githubusercontent.com` at the tags above.
- Issues and PRs were read from plain github.com pages, including the embedded JSON of issue-search pages. The GitHub REST API was not used.

**Legend.**
- *Verified* = read in the primary source in this session.
- **UNVERIFIED** = from a secondary source or a summarizer, not checked against the primary source.
- *Derived* = my arithmetic.

---

## 0. Bottom line

1. **No engine ships traffic-driven, idle-time capture at cold start in its default path.** Every engine checked captures its whole size set before serving, or captures on first use on the request's critical path. The engines checked are vLLM, SGLang (issues, PRs and blog only), TensorRT-LLM, LMDeploy, TGI, MLC/TVM and PyTorch `reduce-overhead`.
2. **Closest existing precedent: vLLM PR #57899** (open, unmerged, opened 2026-09-21), "[Experimental] Add CUDA graph discard and lazy recapture to sleep/wake".
   - After a sleep/wake cycle the engine serves **eagerly**. It then recaptures "at most one configured shape, largest first" at **each idle boundary**. Multi-rank capture runs through the executor RPCs. TP2, TP4 and TP2/PP2 were validated.
   - Differences from our idea:
     - It runs only after sleep/wake, not at cold start.
     - It is FULL graphs only, Model Runner V2 only, without torch.compile.
     - It is opt-in through `VLLM_SLEEP_DISCARD_GRAPHS=1`, with a Qwen3 model allowlist.
     - Order is largest-first, not traffic-driven.
     - There is no hot-size promotion.
3. **The idea was proposed and rejected in vLLM in 2025.**
   - **RFC #20098** "Lazy CUDA Graph capture" was opened 2025-06-25 and closed as not planned by the stale bot on 2025-12-26.
   - **PR #23184** "Lazy/Delayed CUDA graph" was opened 2025-08-19, put on hold 2025-08-26 and closed 2026-03-02 without merging.
   - A maintainer explicitly suggested capturing "while the server sits idle" (PR #23184) and capturing when "there are no requests to be scheduled" (RFC #20098).
   - The author deferred idle capture to "a separate PR", which never appeared.
   - Rejection reasons: reliability ("capturing cuda graphs at runtime affects reliability a lot", WoosukKwon), maintenance burden ("strong no", youkaichao), unpredictable TTFT, and OOM after the server is up.
4. **Lazy capture on first use, synchronous and on the critical path, exists in several places.**
   - **PyTorch CUDA Graph Trees:** warm up on the first call, record on the second, separately per distinct dynamic-shape key.
   - **TVM/MLC-LLM:** `run_or_capture`.
   - **LMDeploy:** decode graphs are captured on a cache miss. Since about v0.10 they are pre-captured at startup unless `LMDEPLOY_SKIP_WARMUP=1`; prefill piecewise graphs are never captured at runtime.
   - **TensorRT-LLM ≤ v1.2.1:** an uncaptured key could be captured inside a live forward. It was gated off in 2026 by PR #12306; see §4.
   - **SGLang:** only the opt-in EAGLE post-verify graph is captured lazily, after warmup rounds. Issue #49206 (open) reports it crashes when JIT happens inside the capture.
   - **vLLM's own `CUDAGraphWrapper` and `BreakableCUDAGraphWrapper`:** both capture any new key they are dispatched. vLLM prevents runtime capture only by never dispatching an uncaptured key and, in the V1 runner, by a global guard.
   - **None of these schedules capture into idle time, and none orders capture by traffic.**
   - TensorRT-LLM lists offline traffic-driven sizing ("parse serving logs …") as future work.
5. **vLLM code anticipates lazy capture but does not implement it.**
   - Two comments added with PR #20059 (in v0.10.1, Aug 2025) say "lazy capturing in future". They are still present in v0.33.0.
   - In the default Model Runner V2, the runtime guard `validate_cudagraph_capturing_enabled()` is **never armed**. Runtime capture is prevented only by `CudaGraphManager.dispatch()`, which returns graph descriptors only after `capture()` has finished, plus an `assert` in `run_fullgraph()`. See §1.6.
6. **TP lockstep is a hard requirement in vLLM.**
   - Each capture session ends with a cross-rank collective: custom all-reduce `register_graph_buffers()` broadcasts IPC handles across the TP group.
   - Graphs embed NCCL or custom all-reduce kernels.
   - DP ranks already agree per step on the graph mode by CPU all-reduce. If any DP rank wants eager, all DP ranks run eager.
   - Any idle-time capture must therefore be a collective, agreed-upon step.

---

## 1. vLLM (v0.33.0; Model Runner V2 is the default)

### 1.1 Two runners, and which one is default

- **Model Runner V2 (MRV2)** lives in `vllm/v1/worker/gpu/*`. The V1 runner is `vllm/v1/worker/gpu_model_runner.py`.
- `VllmConfig.use_v2_model_runner` returns True unless `VLLM_USE_V2_MODEL_RUNNER=0`, Triton is missing, or an unsupported feature is configured. See [vllm/config/vllm.py](https://github.com/vllm-project/vllm/blob/v0.33.0/vllm/config/vllm.py), property `use_v2_model_runner`.
- **History** (verified by reading `vllm/config/vllm.py` at each tag):
  - v0.12.0 (2025-12-03): the env var appeared with default `"0"`.
  - v0.22.0 (2026-05-29): MRV2 default for an allowlist of architectures (`DEFAULT_V2_MODEL_RUNNER_ARCHITECTURES = {"Qwen3ForCausalLM"}` at first).
  - v0.29.0 (2026-09-09): MRV2 default for all models. The allowlist is removed; see PR #53183 "[Model Runner V2] Use MRV2 for all models by default" (njhill, merged 2026-08-27).

### 1.2 `CompilationConfig.cudagraph_mode`

Defined in [vllm/config/compilation.py L53–111](https://github.com/vllm-project/vllm/blob/v0.33.0/vllm/config/compilation.py#L53-L111), class `CUDAGraphMode`. The field and its docstring are at L610–645.

| Value | Meaning (from the docstring) |
|---|---|
| `NONE` | No capture |
| `PIECEWISE` | Piecewise graphs only; cudagraph-incompatible ops (attention) run outside the graph |
| `FULL` | Full graphs for all batches. "not supported by many backends" |
| `FULL_DECODE_ONLY` = `(FULL, NONE)` | Full graphs for uniform-decode batches; mixed prefill-decode batches run with no graph |
| `FULL_AND_PIECEWISE` = `(FULL, PIECEWISE)` | Full graphs for decode batches and piecewise graphs for prefill and mixed batches. "the most performant mode for most models and is the default" |

Only `NONE`, `PIECEWISE` and `FULL` are concrete *runtime* modes; `valid_runtime_modes()` is at L100.

**Defaults in v0.33.0**
- The default comes from the optimization level. `optimization_level` defaults to `O2` (`vllm/config/vllm.py` L435).
  - `-O0`: `NONE`
  - `-O1`: `PIECEWISE`
  - `-O2` and `-O3`: `FULL_AND_PIECEWISE`

  These are the `OPTIMIZATION_LEVEL_0x` dicts, L255–345; `class OptimizationLevel` is at L132.
- Enforce-eager sets `NONE`.
- Backends lacking support are downgraded later, in `_check_and_update_cudagraph_mode`.
- The design doc says pooling models default to `PIECEWISE`: https://docs.vllm.ai/en/latest/design/cuda_graphs/

**History** (verified by reading `compilation.py` at each tag)
- **v0.10.1** (2025-08-19): `CUDAGraphMode` introduced with default `PIECEWISE` ("PIECEWISE. (v1 default)"). This came from PR #20059, "[Core] Allow full cudagraph with separate attention routines and orthogonal to compilation, add support for FA2 and FlashInfer" (fhl2000, created 2025-06-25, merged 2025-08-15, commit 74f441f).
- **v0.11.0** (2025-10-04): default became `FULL_AND_PIECEWISE`.
- **v0.12.0** (2025-12-03): optimization levels `-O0…-O3` appeared.
- **v0.17.0** (2026-03-07): `performance_mode` (`balanced` / `interactivity` / `throughput`) appeared.

### 1.3 Default capture sizes (`cudagraph_capture_sizes`, `max_cudagraph_capture_size`)

The code is `VllmConfig._set_cudagraph_sizes()`, [vllm/config/vllm.py L2398–2656](https://github.com/vllm-project/vllm/blob/v0.33.0/vllm/config/vllm.py#L2398-L2656). Verified at v0.33.0:

- **Ceiling.** If unset:
  - `max_cudagraph_capture_size = min(max_num_seqs * uniform_decode_query_len * 2, default_max)`.
  - `default_max` = **1024 on SM100-family GPUs** (`is_device_capability_family(100)`, i.e. data-centre Blackwell) and **512 otherwise**.
  - It is then clipped to `max_num_batched_tokens`.
  - `uniform_decode_query_len = 1 + num_speculative_tokens` (property at L694–714).
- **Grid** (`performance_mode="balanced"`): `[1, 2, 4] + range(8, 256, 8) + range(256, max+1, 16)`.
  - `performance_mode="interactivity"` replaces `[1, 2, 4]` by `range(1, min(max, 32)+1)`.
- **Extra sizes:**
  - `max_num_batched_tokens` is appended if it is ≤ the ceiling.
  - Without speculation, `max_num_seqs` is appended if it is ≤ the ceiling.
  - With speculative decoding (`decode_query_len > 1`), exact uniform-decode sizes `n * query_len` are appended, using a request-count grid and one tier per dynamic-speculation query length.
- **User overrides.** A user-given list is de-duplicated, filtered to ≤ `max_num_batched_tokens` and sorted. Giving both `cudagraph_capture_sizes` and an inconsistent `max_cudagraph_capture_size` raises an error.
- **Count (derived).** The peer report is confirmed: with ceiling 512 the grid has 3 + 31 + 17 = **51 sizes**. With ceiling 1024 (Blackwell) it has **83 sizes**. The pre-v0.11.1 step-8 grid to 512 had **67 sizes**, the "67 graphs" quoted in RFC #20098.
- **Which ceiling the defaults reach.** `vllm serve` defaults (`EngineArgs.get_batch_defaults`, [vllm/engine/arg_utils.py L2940–3039](https://github.com/vllm-project/vllm/blob/v0.33.0/vllm/engine/arg_utils.py#L2940-L3039)):
  - `max_num_seqs = 1024` and `max_num_batched_tokens = 8192` on H100/H200-class GPUs (≥ 70 GiB, not A100).
  - `max_num_seqs = 1024` and `max_num_batched_tokens = 16384` on ≥ 160 GiB GPUs.
  - `max_num_seqs = 256` and `max_num_batched_tokens = 2048` otherwise.
  - So H100 gets ceiling 512 (51 sizes) and B200 gets 1024 (83 sizes) (derived).
- **Number of captures, not sizes (derived from §1.4 code).**
  - In `FULL_AND_PIECEWISE`, each size ≤ `max_num_seqs × decode_query_len` gets one FULL decode graph *and* one PIECEWISE capture. That is 51 + 51 descriptors on H100 defaults.
  - Each PIECEWISE "capture" is one CUDA graph per piece between attention ops, about `num_layers + 1` graphs.
  - BoyuanFeng estimated "~5000 CUDAGraphs for LLAMA-3.1-70B" in RFC #20098.
- **History** (verified at tags):

  | Version | Change |
  |---|---|
  | v0.11.0 | `max_graph_size = min(max_num_seqs * 2, 512)`, sizes `[1,2,4] + multiples of 8` (the old docstring) |
  | v0.11.1 (2025-11-19) | `max_cudagraph_capture_size` field and step-16 above 256 |
  | v0.13.0 | Speculative `decode_query_len` factor in the default |
  | v0.21.0 | Append `max_num_batched_tokens` |
  | v0.28.0 (2026-08-26) | Blackwell 1024 ceiling |
  | v0.29.0 / v0.31.0 | Uniform-decode size tiers |

- **v0.34.0rc0 (2026-11-08)** includes PR #63406 "[Core][CUDA Graph] Separate prefill (token) and decode (request) CUDA graph capture sizes" (merged 2026-11-04 by WoosukKwon).
  - New fields `prefill_cudagraph_capture_token_sizes`, `max_prefill_cudagraph_capture_tokens`, `decode_cudagraph_capture_request_sizes`, `max_decode_cudagraph_capture_requests` and `max_decode_cudagraph_capture_tokens`.
  - On SM100 the prefill grid reaches 8192 tokens: 54 sizes at 8192, per `prefill_cudagraph_capture_grid()` in v0.34.0rc0 `compilation.py`.
  - The PR reports CUDA-graph memory per GPU rising, e.g. DeepSeek-V4-Flash K7 "4.73 → 12.52" GiB with KV "115.80 → 108.01" GiB. Throughput: "DeepSeek-V4-Flash DSpark K7 3.67× at C256 and 2.31× at C512" output tok/s. For dense models: "Wider graphs cost a dense 27B on one GPU 1.8% of KV cache for no throughput gain, and off-grid sizes run 4–11% slower" (verified in the PR text).
  - Issue #66192 (open, 2026-11-07): since #63406, Qwen3.8-2.4T-A95B-FP8 at TP16 on 4×4 GB200 with `--max-model-len 256000` fails at startup. Weights and the profiled KV budget are the same on both sides; "Only CUDA-graph memory differs", and at the #63406 commit "only 2.5 GiB of KV remains" (verified).
  - `cudagraph_capture_sizes` and `max_cudagraph_capture_size` are "Deprecated with model runner V2".
- **Known complaint about the default ceiling.** Issue #60264 (open, 2026-10-06), "Default CUDA graph capture size (2 * max_num_seqs) runs short prefills eagerly at small max_num_seqs".
  - At `--max-num-seqs 8` the ceiling is 16 tokens. A 17-token prefill takes 34.0–34.4 ms (eager) vs 11.1–11.2 ms with ceiling 512 (1× B200, DeepSeek-V2-Lite-Chat, vLLM 0.31.0).
  - Raising the ceiling to 512 raises "Graph memory … from 0.68 to 1.15 GiB and capture time from 2 s to 10 s".
  - The issue says "#20628 chose `2 * max_num_seqs` to avoid OOM with small `max_num_seqs` in tight-memory setups". (Verified.)

### 1.4 Where capture happens (v0.33.0, exact files and functions)

**Call path at startup**

`Worker.compile_or_warm_up_model()` → `_compile_or_warm_up_model()`, in [vllm/v1/worker/gpu_worker.py L933–1110](https://github.com/vllm-project/vllm/blob/v0.33.0/vllm/v1/worker/gpu_worker.py#L933-L1110):
1. Compile/warm-up dummy runs for `compile_sizes` not in the capture list, largest first.
2. `kernel_warmup()`.
3. MRV2 JIT warmup.
4. `self.model_runner.capture_model()` at L993–994.

Before this, `determine_available_memory()` (L641) calls `model_runner.profile_cudagraph_memory()` (L707–717) **whenever `cudagraph_mode != NONE`**. That is a throwaway capture pass, so default startup captures graphs twice (§1.7).

**MRV2 (default)**
- [`GPUModelRunner.capture_model()`](https://github.com/vllm-project/vllm/blob/v0.33.0/vllm/v1/worker/gpu/model_runner.py#L1086-L1162), in `vllm/v1/worker/gpu/model_runner.py` L1086–1162.
  - It runs under `freeze_gc_for_cudagraph_capture()`.
  - It calls `self.cudagraph_manager.capture(...)` (L1116), then `model_state.capture_inner_cudagraphs(...)` and `speculator.capture()`.
  - Then `lock_workspace()`, and it logs "Graph capturing finished in %.0f secs, took %.2f GiB".
- `ModelCudaGraphManager.capture()` builds the per-descriptor `create_forward_fn`: dummy inputs, attention metadata, LoRA hook. It is at [`vllm/v1/worker/gpu/cudagraph_utils.py` L613–788](https://github.com/vllm-project/vllm/blob/v0.33.0/vllm/v1/worker/gpu/cudagraph_utils.py#L613-L788).
- `CudaGraphManager.capture()` is the loop, at [L424–515](https://github.com/vllm-project/vllm/blob/v0.33.0/vllm/v1/worker/gpu/cudagraph_utils.py#L424-L515):
  - It runs inside `graph_capture(device=...)` (`vllm/distributed/parallel_state.py` L1695).
  - It captures modes in the order **PIECEWISE, then FULL**. The code comment: "PIECEWISE has larger activations so FULL activations should fit in already allocated buffers in the graph pool".
  - Within a mode, descriptors are sorted by `num_tokens` **descending** (largest first, L376–378 in `_init_candidates`).
  - Per descriptor:
    1. Build inputs.
    2. One **eager warmup** with `forward_fn(CUDAGraphMode.NONE)`.
    3. Capture.
  - PIECEWISE is captured by calling the compiled model with `cudagraph_runtime_mode=PIECEWISE`; the piecewise backend's `CUDAGraphWrapper`s then capture.
  - FULL is captured by `torch.cuda.graph(graph, pool, stream=...)` around a plain forward.
  - At the end it sets `_graphs_captured = True` (L515).
- **Descriptor set.** `CudaGraphManager._init_candidates()` (L256–395) builds, per capture size and LoRA case:
  - a FULL uniform-decode descriptor (or a varlen-decode one), and
  - a PIECEWISE (or FULL) mixed descriptor.

  It also precomputes `_candidates[(num_tokens, loras)]`, which maps **every** token count up to each captured size to that size's descriptors (the padding table).

**V1 runner (fallback)**
- [`GPUModelRunner.capture_model()`](https://github.com/vllm-project/vllm/blob/v0.33.0/vllm/v1/worker/gpu_model_runner.py#L6417-L6521), in `vllm/v1/worker/gpu_model_runner.py` L6417–6521.
  - It calls `set_cudagraph_capturing_enabled(True)` and runs under `_freeze_gc()` and `graph_capture()`.
  - It loops over `self.cudagraph_dispatcher.get_capture_descs()` and calls `_capture_cudagraphs()` (L6571).
- `_capture_cudagraphs()` calls `_warmup_and_capture()` (L6523). That runs `cudagraph_num_of_warmups` eager `_dummy_run`s, set to 1 in `VllmConfig.__post_init__` L2011, then one `_dummy_run(..., cudagraph_runtime_mode=mode, is_graph_capturing=True)`.
- The capture itself happens inside `CUDAGraphWrapper.__call__` ([vllm/compilation/cuda_graph.py L235–345](https://github.com/vllm-project/vllm/blob/v0.33.0/vllm/compilation/cuda_graph.py#L235-L345)) the first time it sees a `BatchDescriptor`.
- Afterwards `set_cudagraph_capturing_enabled(False)` (L6503). The comment above it:
  > "We don't put it into graph_capture context manager because we may do lazy capturing in future that still allows capturing after here."
- **Order** comes from `CudagraphDispatcher.get_capture_descs()` ([vllm/v1/cudagraph_dispatcher.py L324–348](https://github.com/vllm-project/vllm/blob/v0.33.0/vllm/v1/cudagraph_dispatcher.py#L324-L348)): PIECEWISE first, then FULL, each "sorted largest-first for memory efficiency".
  - The older comment in `capture_model` dates from PR #10528, Nov 2024: "Capture the large shapes first so that the smaller shapes can reuse the memory pool allocated for the large shapes."

**Memory pool**
- One process-wide pool: `current_platform.get_global_graph_pool()` ([vllm/platforms/interface.py L1324–1329](https://github.com/vllm-project/vllm/blob/v0.33.0/vllm/platforms/interface.py#L1324-L1329)) is a lazily created singleton `graph_pool_handle()`.
- It is used by `CudaGraphManager` (L181), by every `CUDAGraphWrapper` (the comment says "it might not be safe to share a global pool" with multiple streams) and by the breakable wrapper.
- FULL and PIECEWISE graphs of all sizes therefore share one pool, captured largest-first.
- With sleep mode (`sleep_mode_offload_cudagraph`), `capture_pool()` routes capture into a cuMem pool instead: [vllm/compilation/cudagraph_pool.py](https://github.com/vllm-project/vllm/blob/v0.33.0/vllm/compilation/cudagraph_pool.py).

**TP/collective coupling of capture**
- `graph_capture()` enters `ca_comm.capture()` for custom all-reduce.
- On exit it calls `register_graph_buffers()`, which does a `dist.broadcast_object_list` per rank over the CPU group ([vllm/distributed/device_communicators/custom_all_reduce.py L464–494](https://github.com/vllm-project/vllm/blob/v0.33.0/vllm/distributed/device_communicators/custom_all_reduce.py#L464-L494)).
- So **every capture session is a TP-group collective**, and all ranks must capture the same descriptors in the same order.
- TP ranks receive the same scheduler output, so runtime dispatch is deterministic across them without an extra agreement.

### 1.5 How a batch is mapped to a graph (padding vs eager)

**MRV2 (default)**
- `GPUModelRunner.execute_model()` ([model_runner.py L1764](https://github.com/vllm-project/vllm/blob/v0.33.0/vllm/v1/worker/gpu/model_runner.py#L1764)) calls `dispatch_cg_and_sync_dp(...)` (L1832).
- `dispatch_cg_and_sync_dp(...)` is in [vllm/v1/worker/gpu/dp_utils.py L211](https://github.com/vllm-project/vllm/blob/v0.33.0/vllm/v1/worker/gpu/dp_utils.py#L211). It calls `CudaGraphManager.dispatch()` ([cudagraph_utils.py L523–553](https://github.com/vllm-project/vllm/blob/v0.33.0/vllm/v1/worker/gpu/cudagraph_utils.py#L523-L553)).
- `dispatch()` looks up `_candidates[(num_tokens, loras)]`. That table already maps a token count to the next captured size up. It returns the first compatible descriptor, so the batch is **padded up to the next captured size**.
- If no graph fits, or `_graphs_captured` is False, it returns `CUDAGraphMode.NONE` for **eager** execution. "No graph fits" covers `num_tokens > max`, an incompatible uniform/varlen/LoRA/ubatch state, and a FULL-only mode on a mixed batch.
- **Replay.**
  - FULL: `CudaGraphManager.run_fullgraph()` (L555–568). It **asserts `desc in self.graphs`** at L560, then calls `self.graphs[desc].replay()`. It is called from `execute_model` at L2059.
  - PIECEWISE: `run_pw_graph()` (L578–584) calls the compiled model, whose `CUDAGraphWrapper`s replay, or the breakable runner.
- **DP coupling.** `sync_cudagraph_and_dp_padding()` (`dp_utils.py` L45–208) all-reduces `[num_tokens, cg_mode, uniform_len, max_query_len, ubatch, num_reqs, skip_drafts]` over the DP CPU group.
  - It takes `min(cg_mode)`: "If any rank wants to run eager, all ranks run eager".
  - It pads all ranks to the max token count, then dispatches again.

**V1 runner**
- `_determine_batch_execution_and_padding()` ([gpu_model_runner.py L3684–3786](https://github.com/vllm-project/vllm/blob/v0.33.0/vllm/v1/worker/gpu_model_runner.py#L3684-L3786)) calls `CudagraphDispatcher.dispatch()` ([cudagraph_dispatcher.py L233–322](https://github.com/vllm-project/vllm/blob/v0.33.0/vllm/v1/cudagraph_dispatcher.py#L233-L322)).
- `_bs_to_padded_graph_size` (L71–108) pads to the next captured size.
- The dispatcher tries a FULL key, then a relaxed PIECEWISE key (`num_reqs=None`, `uniform=False`), else NONE.
- `num_tokens > max_cudagraph_capture_size` gives **NONE (eager)**.
- DP: `coordinate_batch_across_dp(...)` re-dispatches with `valid_modes={synced_mode}`.
- The docstring of `_set_cudagraph_sizes` states both rules:
  > "If batch size <= one of the `cudagraph_capture_sizes`, the closest padded CUDA graph will be used. If batch size > largest `cudagraph_capture_sizes`, cudagraph will not be used."

### 1.6 Lazy-capture hooks that exist but are not used (peer report checked)

| Peer claim (v0.33.0) | Status | Evidence |
|---|---|---|
| (1) Default list with max 512 = `[1,2,4]+range(8,256,8)+range(256,513,16)` = 51 sizes | **Confirmed** | `_set_cudagraph_sizes`, `vllm/config/vllm.py` L2572–2605. This holds for `balanced` mode, `decode_query_len=1` and `max_num_batched_tokens ≥ 512`. It is 83 sizes on SM100 (ceiling 1024) |
| (2) Comments anticipate lazy capture but it is not implemented | **Confirmed** | `cudagraph_dispatcher.py` L184–186: "if we allow lazy capturing in future PR, some keys may never be triggered." `gpu_model_runner.py` L6498–6503 (quoted in §1.4). Both comments already exist in v0.10.1 (PR #20059); the dispatcher comment was last edited by #23046 (2025-09-26). No lazy, delayed or idle option exists in `CompilationConfig` or `envs.py` at v0.33.0 |
| (3) `validate_cudagraph_capturing_enabled()` raises once startup capture is done | **Partly correct** | It raises only if `set_cudagraph_capturing_enabled(False)` was called ([compilation/monitor.py L98–115](https://github.com/vllm-project/vllm/blob/v0.33.0/vllm/compilation/monitor.py#L98-L115); module default `True`). Only the **V1** runner calls it (`gpu_model_runner.py` L6324/6382 in `profile_cudagraph_memory`, L6435/6503 in `capture_model`). None of the 96 Python files under `vllm/v1/worker/gpu/` (MRV2) calls it, so under the default MRV2 the guard is never armed. The check sits in `CUDAGraphWrapper.__call__` (L279) and `BreakableCUDAGraphWrapper` (`breakable_cudagraph.py` L373) |
| (4) MRV2 is the default and asserts the descriptor was captured | **Confirmed, with nuance** | `run_fullgraph()` asserts `desc in self.graphs` (`cudagraph_utils.py` L560). The *selection* path never returns an uncaptured descriptor: `dispatch()` (L523–553) only consults `_candidates` when `_graphs_captured` is True, else it returns NONE. PIECEWISE replay has no assert. A PIECEWISE descriptor that was not captured would be captured silently by the inner `CUDAGraphWrapper`, because the MRV2 guard is not armed. The same holds for `BreakableCUDAGraphWrapper.__call__` (`breakable_cudagraph.py` L317–345), which captures on a miss of `(batch_descriptor, key_fn(...))`. Dispatch makes that unreachable today for the main model |

**Practical consequence for a vLLM port of lazy capture (derived)**
1. Remove or bypass the `_graphs_captured` all-or-nothing flag.
2. Per descriptor, keep a "captured" bit, and let `dispatch()` fall back to eager (or to a *larger* captured size) for uncaptured descriptors.
3. Run capture of one descriptor as a collective step on all TP/PP/DP ranks, inside `graph_capture()`, because of custom all-reduce registration.
4. Keep the global pool, and capture the largest size first so later captures reuse it. Reviewers asked for this in PR #23184.

### 1.7 Startup-time machinery relevant to capture

**Double capture at startup (memory estimate)**
- `profile_cudagraph_memory()` (MRV2: [cudagraph_utils.py L922–1028](https://github.com/vllm-project/vllm/blob/v0.33.0/vllm/v1/worker/gpu/cudagraph_utils.py#L922-L1028)) runs **before** the KV cache exists. It:
  - bootstraps a minimal KV cache;
  - captures **all PIECEWISE**, encoder and speculator graphs, plus the **2 largest FULL graphs** (`_FULL_GRAPH_PROFILING_SAMPLES = 2`), into a throwaway pool;
  - extrapolates the FULL total as `first + (n−1) × second` (`_extrapolate_full_graph_memory`, L1031–1039);
  - discards everything.
- The real capture follows.
- In v0.33.0 the pass runs whenever `cudagraph_mode != NONE`. `VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS` (default `1`) only decides whether the estimate is subtracted from the KV budget.
- History:
  - PR #30515 "[UX][Startup] Account for CUDA graphs during memory profiling" (MatthewBonanni) was merged 2026-03-07; the env var appears in v0.18.0 with default `"0"`.
  - PR #38284 "[Startup][UX] Enable CUDAGraph memory profiling by default" was merged 2026-04-21. The env default is `"1"` in the v0.20.0 tag; the code comment and log say "default since v0.21.0".
  - PR #64622 "[Startup] Skip the CUDA graph memory profile when its estimate is disabled" was merged 2026-11-06. It is in v0.34.0rc0, not in v0.33.0.
  - The GeometricAGI fork issue #66 makes the same "captures graphs twice" observation (search result only, **UNVERIFIED**).
- `VLLM_ENABLE_STARTUP_PLAN=1` (opt-in, present since v0.26.0, [vllm/v1/worker/startup_plan.py](https://github.com/vllm-project/vllm/blob/v0.33.0/vllm/v1/worker/startup_plan.py)) persists the profiled `--kv-cache-memory`. Later boots then skip "the memory-profiling measurement and the CUDA-graph memory estimation pass" when the fingerprint matches.

**Capture-speed work**

| PR | Change | Result |
|---|---|---|
| #10059 (merged 2024-11-06) | Patch gc in piecewise capture | "previously about 5 seconds for every shape, now it is less than 0.5 second" |
| #21146 (merged 2025-07-24, v0.10.0) | gc collect+freeze during capture (`VLLM_ENABLE_CUDAGRAPH_GC=0` default ⇒ freeze) | "Qwen3-0.6B goes from 35s to 2s". A later comment shows 67 shapes in 00:10 (6.66 it/s) vs 00:01 (46.41 it/s); the preceding benchmark comment says it ran "on B200" (verified) |
| #29275 (merged 2025-11-23) | MRV2 warmup with `seq_lens=1` instead of `max_model_len` | Capture "takes up to a minute" before |

**Breakable CUDA graphs in vLLM** (PIECEWISE without torch.compile)
- Module `vllm/compilation/breakable_cudagraph.py`, "inspired by sgl-project/sglang#19102".
- PRs: #42304 "[Experimental] Breakable CUDA graph" (ZJY0516, merged 2026-05-16); #44050 "[MRV2] Support breakable CUDA graph" (WoosukKwon, merged 2026-05-30).
- Env var `VLLM_USE_BREAKABLE_CUDAGRAPH`, present from v0.22.0, default off. It is auto-enabled for `DEFAULT_BREAKABLE_CUDAGRAPH_ARCHITECTURES` and under `VLLM_BATCH_INVARIANT` (`vllm/config/vllm.py` L113–128, L798–870).

### 1.8 vLLM issues / PRs / RFCs on lazy, deferred, idle or background capture and startup time

*Verified* means read on the GitHub page in this session; states and dates come from the GitHub page or the issue-search JSON.

| # | Title | Author | State / dates | What matters for us |
|---|---|---|---|---|
| [#8888](https://github.com/vllm-project/vllm/pull/8888) | "[MISC] add a flag --lazy-capture-cuda-graph to allow cuda graph capture only happens on-demand" | sfc-gh-zhwang | PR, closed unmerged by stale bot 2025-03-27 (opened 2024-09-27) | First lazy-capture PR (V0 era). "Graph capture usually takes from 10s - 40+s (for large model such as 405b)". Test: `vllm serve /models/llama3.1-70b --tensor_parallel_size 8 --lazy-capture-cuda-graph`; log "batch size 1 finished in 1.1 secs". youkaichao: "if you are running a large model, I think you can just disable cudagraph." (Verified quotes.) |
| [#19824](https://github.com/vllm-project/vllm/issues/19824) | "[Feature]: Improve startup time UX" | ProExpertProg | Issue, 2025-06-19; closed not planned by stale bot 2026-06-16 | Llama-70B-FP8 TP8 (robertgshaw2-redhat): "~60s to check for P2P access", "~10s to load weights", "~15s to convert dynamo bytecode", "~70s to run torch.compile", "~60s to capture the cudagraphs". Also: serializing `cudaGraph_t` "would be amazing. But it's not clear th[is is possible]". Lists lazy capture. (Verified.) |
| [#20098](https://github.com/vllm-project/vllm/issues/20098) | "[RFC]: Lazy CUDA Graph capture" | lionelvillard | RFC, 2025-06-25; closed not planned (stale) 2025-12-26 | **Verified text**, quoted in the row below |
| [#22897](https://github.com/vllm-project/vllm/pull/22897) | "Lazy cuda graph" | diegocastanibm | PR, 2025-08-14 → closed by author 2025-08-19 (sign-off problems), replaced by #23184 | Flag `use_cudagraph_delayed_capture`. Same benchmark as #23184. A Gemini bot review: "By using set.pop(), an arbitrary graph size is captured. This breaks the 'large-to-small' capture order, which is crucial for efficient memory pool allocation". Closed by the author: "Clean PR here: #23184 … I forgot to sign-off every commit". (Verified.) |
| [#23184](https://github.com/vllm-project/vllm/pull/23184) | "ON HOLD - [Core] Lazy/Delayed CUDA graph" | diegocastanibm (IBM / llm-d) | PR, 2025-08-19; "ON HOLD" 2025-08-26; closed by ProExpertProg 2026-03-02, unmerged | **Verified text**, see the bullets below |
| [#57899](https://github.com/vllm-project/vllm/pull/57899) | "[Experimental] Add CUDA graph discard and lazy recapture to sleep/wake" | AlanFokCo | **Open** PR (not draft), 2026-09-21; `needs-rebase` since 2026-11-02; no approvals | **Verified text, closest precedent**, see the bullets below |
| [#16716](https://github.com/vllm-project/vllm/issues/16716) | "[Bug]: vllm 0.8.3 v1 engine CUDA Graph Capturing time is too long" | spacecat2002 | Issue, 2025-04-16 → closed 2025-04-17 | "the capture time of cuda graph is always 65s" (Llama-2-7B, `--enable-lora --max-loras 4`, RTX 4090, vLLM 0.8.3). Reply: "Yes … Then use `--enforce-eager`". (Verified.) |
| [#60264](https://github.com/vllm-project/vllm/issues/60264) | Default capture ceiling runs short prefills eagerly | zzw4257 | Open, 2026-10-06; fix PR #60265 open | See §1.3 |
| [#63406](https://github.com/vllm-project/vllm/pull/63406) | Separate prefill/decode capture sizes | Pb314314 | Merged 2026-11-04 | See §1.3 |
| [#66192](https://github.com/vllm-project/vllm/issues/66192) | "SM100 default prefill CUDA-graph grid (up to 8192 tokens, #63406) adds ~5 GiB/GPU graph memory and breaks max_model_len=256000 startup …" | — | Open, 2026-11-07 | See §1.3 (verified). **Graph memory competes directly with the KV cache**, a cost that lazy capture would make dynamic |
| [#57475](https://github.com/vllm-project/vllm/issues/57475) | "[Bug] CUDA graph memory estimate under-reserves → OOM during graph capture at high `--gpu-memory-utilization`" | — | Issue, created and closed 2026-09-18 | At 0.98 the real capture OOMs, while 0.97 boots. "The KV pool is sized using `cudagraph_memory_estimate` from `profile_cudagraph_memory()`, but that estimate does not leave enough headroom for the *real* capture" (vLLM 0.29.1rc1.dev339). (Verified.) **Shows the estimate-then-capture protocol is fragile even at startup** |
| [#48193](https://github.com/vllm-project/vllm/issues/48193) | "[Roadmap] Weight-loading-bound startup" (was "Cold Start Q3 2026") | matteso1 | Open, 2026-07-10 | Cold-start roadmap centred on initialized-engine snapshots: #51360 TP1 snapshot restore merged; #48996 CRIU restore. "Warm restart today is 29-35s vs 71-78s cold" (A10, Qwen3-0.6B). A restore-path comment: "What remains after the restore is mostly process start, profiling, KV allocation and CUDA graph capture". Live init after restore (~18 s median) includes "graph capture". Lazy capture is not mentioned. (Verified quotes.) |
| [#34303](https://github.com/vllm-project/vllm/issues/34303) | "[RFC]: CUDA Checkpoint/Restore for Near-Zero Cold Starts" | — | — | Snapshot route; see the companion file. (Search result only here.) |
| [#66472](https://github.com/vllm-project/vllm/pull/66472) | "[Fast Start] Persist CUDA graphs with Foundry and prepare them during preload" | UNIDY2002 | Open PR, 2026-11-09 | Foundry SAVE/LOAD of FULL decode graphs, opt-in. Qwen3-0.6B on B300 with 129 sizes: startup −4.0%; graph construction 214/288 ms vs about 3 s native capture. See §5.5 (verified text) |
| [#57227](https://github.com/vllm-project/vllm/issues/57227) | "GPU memory-access fault on gfx950 (MI350X) when triton kernels JIT during lazy CUDA-graph capture of a 16K-token chunked prefill (GLM-5.3-Flash …)" | mustafayildirim | Closed, 2026-09-16 | Verified text. "Lazy capture" is the reporter's wording. The thread found the faulting kernel runs in an eager segment of a *breakable* graph (`BreakableCUDAGraphCapture.current() is None`), and the fix was #57317. **Not evidence of runtime capture in vLLM.** It does show JIT of unseen shapes at request time. "a deterministic prewarm cannot cover it" (chunked-prefill specializations) |

**RFC #20098 (verified text)**
- Motivation: "By default, vLLM captures 67 graphs, which depending on model size and GPU type, can take more than 10s".
- ProExpertProg suggested "a feature to the scheduler where it lazily captures cudagraphs if there are no requests to be scheduled". This is **idle-time capture, June 2025**.
- charlesfrye (Modal) gave the cost: boots "~45s without it, so ~10s would add 20%". He wants "one or two graphs eagerly (say, batch size one and max)".
- Objections:
  - zou3519: less predictable.
  - BoyuanFeng: "abrupt performance drop during runtime". He suggested fewer sizes instead and estimated "~5000 CUDAGraphs for LLAMA-3.1-70B".
  - yeqcharlotte and LucasWilkinson: OOM after the server is up, so the feature should be opt-in only.
  - angkywilliam: `max_loras=36` gives 7 min startup for an 8B model; asked to prebuild a subset and capture the rest lazily.
  - lionelvillard (Jul 2025): about 50 ms per graph for an 8B model after the gc fix; proposed to "interleave capturing graphs and doing a real forward passes".
  - youkaichao (Aug 2025): "I have a strong no for this feature. capturing cudagraph during runtime is very complicated, with huge maintenance burden, and it can break many other features."
  - diegocastanibm: batch-size histograms over 4 runs of 1,000 prompts are unpredictable, so cutting the size list hurts.

**PR #23184 (verified text)**
- Design: "The highest priority of CUDAGraph caching is given by the current runtime shape if not cached already. Otherwise, it is capturing the highest shape that has not been captured yet". One capture per step. After review it captures the largest size at init.
- Flag: `use_cudagraph_delayed_capture`, later `use_delay_cudagraph_capture`.
- Benchmark: Llama-3.1-8B, ShareGPT 1,000 prompts, GPU not stated. Throughput 13.73 → 13.50 req/s. Mean TTFT 20493.77 → 20431.48 ms. Mean TPOT 79.60 → 79.64 ms. Init capture 67 graphs, "4 secs, took 0.54 GiB", vs 0 at init.
- Per-graph runtime capture, from the PR's logs (verified):
  - 76 "finished in" lines; the per-size captures are 0.036–0.095 s, median about 0.053 s, for Llama-3.1-8B piecewise.
  - In the `"cudagraph_mode": "FULL"` run, which FlashAttention downgraded to `FULL_AND_PIECEWISE`, some sizes took 1.101–1.249 s.
  - Some "took" memory deltas are negative (−342 MiB at 512 tokens).
- Review:
  - ProExpertProg: "Could we build in a mechanism to capture cudagraphs while the server sits idle?" Author: "Good idea. I propose to do it in a separate PR".
  - ProExpertProg proposed an enum `STARTUP` / `DELAYED` / `LAZY`, and a memory warning after the first (largest) capture.
  - WoosukKwon (changes requested): "I think capturing cuda graphs at runtime affects reliability a lot. Also for simplicity, I think we should strictly limit it to init time only."
  - A user reported garbled first-request output with multi-TP (2025-12-01); the author had not seen it.

**PR #57899 (verified text, closest precedent)**
- "recaptures one graph shape at each idle boundary".
- "A completed request after wake enables idle recapture. Each idle boundary captures at most one configured shape, largest first. Continuous traffic can postpone completion. A request arriving during capture waits for that shape".
- "Multi-worker capture failures terminate the executor: a peer may already be blocked inside a collective".
- Scope: MRV2, FULL/FULL_DECODE_ONLY, no torch.compile, no async scheduling, Qwen3 allowlist. TP and PP validated.
- Numbers (30B-A3B): 0.07–0.14 s per capture. A request during recapture waited 0.628 s vs 0.093 s settled. The eager window costs throughput: −66.0% (0.6B), −29.3% (8B), −77.4% (30B-A3B).

**Also relevant (sleep-mode graph handling, all 2026)**
- #59160 "[Feature] Release the CUDA graph pool on sleep".
- #59909 "[Core] Release the CUDA graph pool on sleep instead of backing it up" (wake replays without recapture).
- #56373 "[V2] Offload decoder CUDA graph pools in sleep mode" / "without graph recapture" (closed).

These show the mainstream direction is **keeping** graphs across sleep, not recapturing them. (Titles from the search JSON.)

### 1.9 vLLM docs

- **"CUDA Graphs" design doc**: https://docs.vllm.ai/en/latest/design/cuda_graphs/ (latest is the dev docs, footer 2026-11-07, "mostly based on the last commit of Pull Request #20059").
  - It says the wrapper captures a key "if key does not exist, create a new entry and cache it".
  - "If no graph fits, it returns a descriptor with `CUDAGraphMode.NONE` for eager execution".
  - `FULL_AND_PIECEWISE` "requires the most memory and takes the longest to capture".
  - It does not discuss lazy capture.
- **torch.compile design doc**: https://docs.vllm.ai/en/latest/design/torch_compile/
  - "Larger caps and longer lists capture more and wider graphs, which take memory from the KV cache and lengthen startup."
  - "Decode graphs no wider than the widest prefill graph mostly reuse its memory in the shared graph pool."
- **Optimization docs** (`--enforce-eager`, `--kv-cache-memory` skip profiling): https://docs.vllm.ai/en/stable/configuration/optimization/ (search snippet).

---

## 2. SGLang (issues, PRs, LMSYS blog; capture code not read here)

### 2.1 Issues and PRs on lazy, deferred, background capture and capture time

| # | Title | State / dates | Notes |
|---|---|---|---|
| [#3902](https://github.com/sgl-project/sglang/issues/3902) | "[Feature] Allow Serving Requests During CUDA Graph Capture" | Issue by junliu-mde, 2025-02-27; closed as completed by the author 2025-05-05; **no linked PR** | Restart "requires about 10-minute CUDA Graph capture process" (longer with torch.compile). Baizhou (relayed): "Theoretically feasible to let model runner choose whether to replay existing CUDA Graph". Fridge003 (assignee): "`init_cuda_graphs` can become asynchronous so requests can be served at the same time". A reply: "there might be many synchronous operations at now, making it difficult to implement asynchronous". **Not implemented** as far as found. (Verified quotes.) |
| [#33483](https://github.com/sgl-project/sglang/issues/33483) | "`max_running_requests` (4096) and `cuda_graph_max_bs` (32) defaults are derived independently; crossing the graph ceiling is an absorbing state" | Issue by KeMaSF, 2026-08-04; closed by bot 2026-11-08 (inactive) | L40, Qwen2.5-0.5B: decode above bs 32 runs eager. ITL "3.0 → 12.6 ms (4.2x)". `--cuda-graph-max-bs 32` = 8 graphs `[1,2,4,8,12,16,24,32]`, "0.89 s" capture, 0.15 GB. `128` = 20 graphs, "1.28 s", 0.37 GB. "leaving it requires N below 32, but at 12.6 ms the self-consistent N is 77". Suggestion 4: "Capture a larger shape lazily on first overflow — raising as a question; I have no idea what it costs." (Verified.) **Evidence that eager fallback above the ceiling can trap the system.** |
| [#43172](https://github.com/sgl-project/sglang/pull/43172) | "Clamp default decode CUDA graph max_bs to max_running_requests" | Merged 2026-10-10 (cctry) | Default-sizing fix |
| [#8577](https://github.com/sgl-project/sglang/pull/8577) | "[Optimization][Perf] Disable the GC during CUDA graph capture to speed up by up to 3x" | Merged 2025-07-31 | "2.3x–3.7x speedup in capture time". Progress bars: Llama-4-Maverick-FP8 23 sizes 00:25 → 00:10. Qwen3-0.6B 23 sizes 00:06 → 00:01. (Verified.) |
| [#9616](https://github.com/sgl-project/sglang/pull/9616) | "Speed up CUDA graph capture on Flashinfer" | Closed unmerged 2025-11-14 | — |
| [#32017](https://github.com/sgl-project/sglang/pull/32017) | "[Model Loading] Overlap checkpoint staging with CUDA graph capture during startup" | Merged 2026-08-13 | Opt-in `--startup-weight-load-mode overlap` ("This PR does not enable overlap by default"). Page-cache prefetch overlaps capture; capture itself stays foreground and there is no serving during capture. Qwen3-32B BF16 on H100, local NVMe: TP1 84.83 → 35.60 s and TP2 78.60 → 45.16 s, mostly from prefetch. Overlap vs serial+prefetch alone: −8.6% (TP1) and −11.7% (TP2). (Verified numbers.) |
| [#49206](https://github.com/sgl-project/sglang/issues/49206) | "[Bug] SGLANG_ENABLE_NV_MEGA_RUNTIME_OPTIMIZATION: post-verify CUDA graph capture crashes on first sampled request (torch.compile JIT inside capture via flashinfer top_k)" | Open, 2026-11-08 | Verified text: "After its eager warmup rounds, `eagle_post_verify_graphs` captures the post-verify graph lazily. During that capture, Dynamo compiles a new frame, and torch raises an error". **SGLang has at least one opt-in runtime lazy-capture path**: EAGLE post-verify graphs behind `SGLANG_ENABLE_EAGLE_POST_VERIFY_CUDA_GRAPH`, added Oct–Nov 2026 around #45474 and #47200, i.e. after v0.5.20. It fails when JIT happens inside the lazy capture |
| [#29942](https://github.com/sgl-project/sglang/issues/29942) | "DeepEP low_latency buffer lazy init fails during CUDA graph capture with PP=2, TP=8 …" | Closed 2026-07-02 | Lazy *buffer* init inside capture fails; motivates warmup before capture. (Snippet.) |
| [#37411](https://github.com/sgl-project/sglang/pull/37411) | "fix: lazily instantiate CUDA graph compatibility executables" | Closed unmerged 2026-09-16 (HanHan009527) | Lazy instantiation of the *dedup* compat executable (see §5.5). GLM-5.3 FP8 TP16 on H20, decode breakable sizes [1,4,10,16,24,32]: instantiations per rank 12 → 6; 220.0 → 89.1 ms per rank; +234 MiB free HBM. (Verified.) |
| [#44627](https://github.com/sgl-project/sglang/pull/44627) | "[Perf] DSv4.1: Make the breakable prefill CUDA graph usable above 16K contexts, and add a serving-shape startup warm-up" | Closed 2026-10-16 (YAMY1234) | Verified text: "graph capture and the server's own warm-up run only short decode-shaped batches. Every other shape compiles its Triton and JIT kernel variants inside serving traffic the first time it appears … On a cold cache, the first 8K-token request on main takes 7.8 s instead of 0.19 s". The proposed fix was an opt-in `--warmups deepseek_v4_serving_shapes`. **JIT-on-first-use is already a lazy cost that any lazy capture must order before capture** |
| [#49263](https://github.com/sgl-project/sglang/pull/49263) | "[Fix] cuda-graph: Pause automatic GC during CUDA graph capture" | (search result) | Collections during capture cause intermittent failures |

Searches of SGLang issue/PR titles found **no** general lazy, on-demand, deferred, delayed, idle or background CUDA-graph capture feature for the main model. The title queries were `lazy`, `on-demand`, `deferred`, `delayed`, `idle`, `background`, `async … capture`, `"capture time"` and `startup "cuda graph"`.
- The only runtime lazy capture found is the opt-in EAGLE post-verify graph (#49206).
- "Idle" hits are all about DP-attention idle batches, e.g. #7269 "Enable cuda graph for dp idle batch" (merged 2025-06-17).
- Worth checking in the v0.5.20 code you are reading: `runner_backend/cuda_graph_dedup_mixin.py`, which is executable dedup via `cudaGraphExecUpdate` (#29625, merged 2026-06-29, off by default).

### 2.2 LMSYS blog: "Advanced CUDA Graph Techniques in SGLang" (2026-08-17)

URL: https://www.lmsys.org/blog/2026-08-17-advanced-cuda-graph/ (verified full text and the SVG chart labels). By the SGLang Team: Yuwei An\*, Cheng Wan, Xiaoyu Zhang, Mick Qian, Baizhou Zhang, Yusheng Su, Ke Bao; with Meta: Shiyang Chen\*, Lianmin Zheng.

**What BCG is**
- Breakable CUDA Graph "captures graph-safe regions while allowing selected operations to run eagerly between graph segments".
- An incompatible operation is marked with `@eager_on_graph`. "capture stops before the marked function and resumes afterward, producing a sequence of CUDA Graph segments separated by eager regions."
- Replay runs the recorded segments and eager functions in order.
- During capture, each eager function's returned tensor is kept as a **persistent boundary buffer** with a fixed address. On replay, the fresh result is copied into that buffer so the next segment reads it.
- BCG "never inspects or traces the operations inside the eager region".

**How BCG differs from TC piecewise** (torch.compile `fullgraph=True` → FX split at registered split points → compile and capture each piece)
- No compiler: breaks are placed during capture, not discovered by tracing.
- Segmented execution "in roughly a quarter of the code (521 versus 1,771 lines)".
- No `torch.library` wrappers or fake implementations are needed for custom kernels. Graph boundaries follow serving logic rather than compiler type constraints.
- Replay calls recorded segments directly. TC piecewise "calls back into the compiled callable every time, paying Torch Dynamo's guard checks and dispatch".
- Debuggable: eager regions run normal Python on every replay. `--debug-cuda-graph` (#19102) wraps the whole forward in one eager break.
- Coverage: it captures GLM-5.2, which TC piecewise cannot trace.

**Other techniques in the post**
- Runner/backend refactor (#23906). Runners own shapes, static buffers, metadata and padding. Backends are full, breakable or TC piecewise.
- Full CUDA Graph for prefill (#27988, experimental). Token buckets plus fixed request slots with zero-length sentinels; more requests than slots means eager.
- One shared pool per captured shape across all its segments.
- Weak references at breaks, a technique taken from vLLM #9724.
- One max-size output buffer shared across capture sizes.
- Capture through `chunked_prefill_size`, which removes the largest eager activation peak.
- Standalone library `meta-pytorch/breakable-cuda-graphs` (`breakable_graph(...)`, `@no_graph`, `CUDAGraphSequence`).
- **Not covered:** capture order, lazy or on-demand capture, idle capture.

**Every number in the post, with its setup**

| Quantity | Value | Setup (as stated) |
|---|---|---|
| Share of prefill-graph preparation spent in torch.compile (TC piecewise) | 78–86% | Prefill graphs |
| Cold-start time to build prefill graphs, Qwen3-235B-A22B (94-layer MoE) | **BCG 27.7 s** (capture only) vs **TC piecewise 106.6 s** = 90.4 s compile + 16.2 s capture ("3.8× slower") | TP4 on 4×GB300, **42 captured shapes**, weight loading and kernel JIT excluded (SVG chart) |
| Same, GLM-5.2 (78-layer MoE + DSA) | **BCG 35.2 s** vs **TC piecewise 183.1 s** = 158.2 s compile + 24.9 s capture ("5.2× slower") | Same |
| Chart note | "BCG removes the compiler phase; its graph capture alone is slightly slower than TC piecewise capture" | So BCG *capture* is 27.7 vs 16.2 s and 35.2 vs 24.9 s (derived: about 1.4–1.7× slower capture) |
| Prefill speedup vs eager | Full 1.93×, BCG 1.70×, TC piecewise 1.45×; BCG 17% faster than TC piecewise at replay | gpt-oss-120b, TP4, 4×GB300; median TTFT, one request at a time, fixed input length, one output token, decode graphs off; input 64–2048 tokens ("flat across a 32× range") |
| GLM-5.2 prefill | BCG 1.60× over eager (only BCG can capture) | — |
| Graph memory | 42 shapes "add 2.4 GB of graph memory on GLM-5.2" | 78-layer MoE |
| Prefill memory above the no-graph baseline, gpt-oss-120b | none 0.560 GB; ceiling 1024: 0.575; 2048: 0.580; 4096: 0.591; **8192 (= chunk): 0.055 GB** | TP2, chunk 8192, after one prefill at exactly 8192 tokens (SVG chart) |
| Same, GLM-5.2-FP8 | none 1.552; 1024: 1.572; 2048: 1.584; 4096: 1.607; **8192: 0.456 GB** | TP4, chunk 8192 |
| Text versions | Activation peak 0.56 GB → 0.001 GB (gpt-oss-120b) and 1.55 → 0.35 GB (GLM-5.2). Total 0.51 GB and 1.10 GB below the no-graph baseline | — |
| Diffusion (BCG in DiT, #27436) | Qwen-Image 512×512: 6.48 s → 2.45 s; Z-Image: 1.231 s → 0.662 s end-to-end after warmup | Qwen-Image on a single B200 |

**Note for our write-up.** The memory chart implies that capture sizes *below* the chunk size cost resident memory without lowering the peak. A lazy scheme that leaves the largest prefill size uncaptured keeps the eager activation peak (derived from the chart).

### 2.3 BCG lineage (verified via GitHub issue-search JSON: state, created, merged)

| PR | Title | Created → merged |
|---|---|---|
| #19102 | "Introduce CUDA graph debug mode with breakable CUDA graph" (cctry) | 2026-02-21 → 2026-04-11 |
| #22218 | "[Experimental] Breakable Piecewise Cuda Graph" (Oasis-Git) | 2026-04-07 → 2026-04-24 |
| #24662 | "Breakable Cuda Graph Support for bs > 1" | → 2026-05-11 |
| #23906 | "[Refactor] Cuda Graph Runner/Backend Refactor" | → 2026-06-10 |
| #27988 | "[Experimental] Full Cuda Graph Support for Prefill" | → 2026-07-07 |
| #27436 | "[diffusion] Enable breakable CUDA graph (BCG) for diffusion DiTs" | → 2026-07-08 |
| **#29458** | **"Enable Breakable Cuda Graph as Default"** (Oasis-Git) | 2026-06-26 → **2026-07-02** |
| #31682 | "Turn on breakable prefill cuda graph for dp attention by default" | → 2026-07-21 |

- The **SGLang v0.5.15 release notes** (2026-07-10) say: "Breakable CUDA Graph on by default: Breakable CUDA Graph is now the default capture path, reducing per-step kernel-launch overhead (#29458); full CUDA Graph support for the prefill phase lands as experimental (#27988)." Verified on the release page.
- vLLM adopted the idea: `vllm/compilation/breakable_cudagraph.py` docstring says "inspired by sgl-project/sglang#19102".
- TensorRT-LLM has a prototype `prefill_cuda_graph_backend="breakable"` (§4).

---

## 3. PyTorch CUDA Graph Trees (`torch.compile(mode="reduce-overhead")`), v2.15.0

**Sources (verified)**
- Doc source [docs/source/user_guide/torch_compiler/torch.compiler_cudagraph_trees.md @ v2.15.0](https://github.com/pytorch/pytorch/blob/v2.15.0/docs/source/user_guide/torch_compiler/torch.compiler_cudagraph_trees.md). Rendered: https://docs.pytorch.org/docs/2.15/torch.compiler_cudagraph_trees.html (the stable URL redirects there).
- Code [torch/_inductor/cudagraph_trees.py @ v2.15.0](https://github.com/pytorch/pytorch/blob/v2.15.0/torch/_inductor/cudagraph_trees.py) and [torch/_inductor/config.py @ v2.15.0](https://github.com/pytorch/pytorch/blob/v2.15.0/torch/_inductor/config.py) (class `triton`; the cudagraph knobs are at L2144–2242).

**When a graph is recorded: warmup, then record, on the caller's critical path**
- The doc example: "the first run warms up each graph, which does things like CuBlas or Triton benchmarking"; "The second run does a CUDA Graph recording, and replays it"; then "we hit the optimized, CUDA Graph replay path".
- In code, `CUDAGraphTreeManager._run()` (L2680+) does the following:
  - If `function_id not in self.warmed_up_functions` (or a warmup is in progress), it runs `run_eager()`. That creates a `CUDAWarmupNode` and executes **in the cudagraph memory pool**.
  - The next invocation goes to `record_function()` (L2885). That calls `torch.cuda.synchronize()`, builds a `CUDAGraphNode` (whose constructor records with `torch.cuda.graph(..., pool=self.cuda_graphs_pool, capture_error_mode="thread_local")` in `_record`, L1566–1575), synchronizes again, and returns the recording outputs.
  - Later calls `execute_node()` → replay.
- So **the request that hits a new shape pays both the eager warmup (first hit) and the recording (second hit)**, synchronously.
- Capture is serialized process-wide: "CUDA graph capture and private-pool warmup are process-wide CUDA states. Keep them serialized across Python threads …" (`graph_capture_lock`, L335–338).
- Warmup and recording interact through the allocator. "any warm up of a node can only be followed by warm up runs" (comment at L2716–2720), because a later recording must see the same allocator state.

**Per-shape lazy recording**
- `cudagraphify_impl()` (L441–520) docstring: "Wrap `model` so it is cudagraphified lazily on first call, recording a separate cudagraph tree function per distinct set of int (dynamic shape) inputs. Shapes excluded by `cudagraph_capture_sizes` or beyond `cudagraph_dynamic_shape_rerecord_limit` run eagerly."
- The inner function is literally named `deferred_cudagraphify`. Each new symint key calls `cudagraphify(...)` and caches the result in `fn_cache`.

**Memory pools**
- "CUDA Graph Trees use a single memory pool across all graph captures."
- One `CUDAGraphTreeManager` per device. It is kept in thread-local `tree_manager_containers` (L320–331), with `self.cuda_graphs_thread_pool = torch.cuda.graph_pool_handle()` (L2469).
- Different control-flow paths form a *tree*. Before recording a new branch after replays, the allocator state is checkpointed back ("checkpoint pool state").
- Outputs of a prior invocation are overwritten. Use `torch.compiler.cudagraph_mark_step_begin()` or clone; `cudagraph_trees_generation_cloning="user_visible"` is opt-in.
- Optional `cudagraph_initial_mempool_allocation_gb` pre-reserves one segment.

**Limits and knobs** (`torch._inductor.config.triton`, v2.15.0 defaults)

| Knob | Default | Effect |
|---|---|---|
| `cudagraph_trees` | `True` | Use trees and a single shared pool |
| `cudagraph_skip_dynamic_graphs` | `False` | "Should we skip cudagraphing graphs with dynamic shape inputs. If False, we will re-record a graph for each unique set of shape inputs" |
| `cudagraph_capture_sizes` | `None` | Only these dynamic-shape keys are captured; others run eager. `None` = all |
| `cudagraph_dynamic_shape_warn_limit` | `8` | Warn when the number of per-shape graphs exceeds this |
| `cudagraph_dynamic_shape_rerecord_limit` | `None` | "Stop re-recording new cudagraphs after this many distinct dynamic shapes … further new shapes run eager" |
| `cudagraph_unexpected_rerecord_limit` | `128` | Max re-records of a function/node due to static-input address changes |
| `cudagraph_managed_input_rerecord_limit` / `_action` | `5` / `"copy"` | After N pointer changes, copy inputs into stable buffers (or `"skip"` → eager) |
| `skip_cudagraph_warmup` / `force_cudagraphs_warmup` | `False` / `False` | Skip the warmup run / always stay in warmup |

The doc also notes "Nvidia uses 64 KB of device memory per kernel launch in CUDAGraph, up until CUDA 12.4 and Driver Version 550+" and recommends padding to a few fixed shapes.

**Relevance**
- This is the canonical **lazy, capture-on-first-use** design. Its costs are exactly those the vLLM reviewers raised:
  - the triggering call pays warmup and record synchronously;
  - capture is serialized and includes device-wide `synchronize()`;
  - there is no notion of idle-time or background capture, and no traffic-driven ordering.
- It caps graph count by *number of distinct shapes seen*, not by expected benefit.

---

## 4. TensorRT-LLM (PyTorch backend)

**Config** ([tensorrt_llm/llmapi/llm_args.py @ v1.4.0rc3](https://github.com/NVIDIA/TensorRT-LLM/blob/v1.4.0rc3/tensorrt_llm/llmapi/llm_args.py), `BaseCudaGraphConfig` L175–258; `CudaGraphConfig = DecodeCudaGraphConfig`; same semantics in v1.2.1 `CudaGraphConfig` L107)
- `batch_sizes: Optional[List[int]]`, `max_batch_size` (default 0, meaning 128) and **`enable_padding: bool = False`**: "If true, batches are rounded up to the nearest cuda_graph_batch_size. This is usually a net win for performance."
- Generated lists when `batch_sizes` is not given:
  - padding on: `[1,2,4] + [8,16,…,128]`, then +64 steps up to max.
  - padding off: `1..31, 32, 64, 128` + powers of two up to max.
  - `max_batch_size` is always appended.
- `cuda_graph_config` is on by default (`default_factory=CudaGraphConfig`). It is decode-only: "CUDA graphs are only created for the batch sizes in cuda_graph_config.batch_sizes, and are enabled for batches that consist of decoding requests *only*". "each CUDA graph can use up to 200 MB of extra memory."
- Prefill graphs: `prefill_cuda_graph_backend` ∈ {`disabled` (default), `piecewise`, `breakable`} (prototype). `prefill_capture_num_tokens` defaults to `_DEFAULT_PREFILL_CAPTURE_NUM_TOKENS = [2**i for i in range(8)] + list(range(256, 3073, 256))`, i.e. 1–128 in powers of two, then 256–3072 in steps of 256.

**Capture**
- At warmup, `_capture_generation_cuda_graphs()` ([model_engine.py @ v1.2.1 L700–780](https://github.com/NVIDIA/TensorRT-LLM/blob/v1.2.1/tensorrt_llm/_torch/pyexecutor/model_engine.py#L700-L780)) does the following:
  - batch sizes are sorted descending: "Reverse order so smaller graphs can reuse memory from larger ones";
  - draft lengths are also descending;
  - each `CUDAGraphRunner.capture()` does `WARMUP_STEPS` eager runs, then `torch.cuda.graph(graph, pool=self.memory_pool)`. `WARMUP_STEPS` is 2 in v1.2.1 and 1 in v1.4.0rc3 (`cuda_graph_runner.py` L252). The v1.4.0rc3 warm-up driver was not located in `model_engine.py`;
  - one pool is shared, `self.memory_pool = graph.pool()`.

**Runtime**
- `maybe_get_cuda_graph()` returns no graph (eager) if `batch_size not in self.supported_batch_sizes`.
- With `enable_padding`, `_get_padded_batch()` first appends dummy generation requests up to the next captured size, provided this does not exceed `max_batch_size`.

**On-the-fly capture existed and was removed**
- **v1.2.1:** the forward path did `if self.cuda_graph_runner.needs_capture(key): self.cuda_graph_runner.capture(...)`, then replayed (model_engine.py L3320–3340). An uncaptured key was captured **inside a live forward**, e.g. when warmup skipped a size for lack of KV space.
- **v1.4.0rc3:** capture is gated by `allow_capture()` (cuda_graph_runner.py L787–803). Its comment: "Capture is disabled by default. On-the-fly captures outside this context are prevented because they can resize the shared cuda_graph_workspace tensor, invalidating addresses baked into previously captured graphs".
- `maybe_get_cuda_graph()` adds: "Capturing a mixed graph on a live batch would execute its KV-cache writes during graph warmup/capture … Only shapes captured by the two-pass startup warmup may replay".
- The gating lines were introduced by commit 371e38d, PR **#12306** "[None][fix] KVCacheManagerV2 bug fixes (V2 remains default OFF)", 2026-04-24 (GitHub blame). It is present from v1.3.0rc25 at least.
- **Lesson for us:** a lazy capture must not let a capture resize shared workspaces after earlier graphs baked their addresses. vLLM has the same hazard and handles it with `lock_workspace()` after capture.

**TRT-LLM tech blog 20, "Tuning CUDA Graph Batch Sizes for Higher Output Throughput"** ([link](https://nvidia.github.io/TensorRT-LLM/blogs/tech_blog/blog20_Tuning_CUDA_Graph_Batch_Sizes_for_Higher_Output_Throughput.html); "Last updated on November 03, 2026"; experiments on TRT-LLM v1.3.0rc8, GB200, ISL 500 / OSL 2000; verified full text)
- **Size sets.**
  - Old default "x2": 23 sizes `1, 2, 4, 8, 16, 24, …, 128, 256, 512, 1024, 2048`.
  - "+64": 49 sizes, 64-step from 128 to 2048.
  - "+8": 259 sizes.
- **Capture happens twice at startup:** "once as a dry run to measure runtime GPU memory consumption … and once as the live launch with the full KV cache allocated". DeepSeek-R1 TP=4, max batch 2048:

  | Config | Graphs | 1st capture | 2nd capture | Total LLM init |
  |---|---|---|---|---|
  | x2 (default) | 23 | 10.77 s | 10.38 s | 183.62 s |
  | +64 | 49 | 25.90 s | 25.39 s | 215.13 s |

  That is about 0.45–0.53 s per graph per pass (derived).
- **Memory.** The pool is shared via `graph.pool()`, so "the memory pool size is determined by the memory footprint of the largest graph". Each extra graph adds "approximately 10 MB of dedicated data" outside the pool (DeepSeek-R1 TP4).
  - +8 over +64 (210 graphs) is reported as both "2.6 GB" and "1.99 GB" for DeepSeek-R1; the blog is internally inconsistent. It is 4.86 GB for DeepSeek-V3.2, which cut KV capacity by 17.2%.
- **Throughput.** +64 gives up to 1.3× TPS aggregated and up to 1.5× disaggregated (DeepSeek-R1). It **became the default when padding is enabled**, consistent with the v1.4.0rc3 generator above.
- **Future work** (verbatim headings):
  - Capture "only two to four graphs during the dry-run phase".
  - "Automatic CUDA graph batch size tuning: We could parse serving logs to compute the best set of CUDA graph batch sizes to capture based on the recorded concurrency distribution".
  - That is *offline* traffic-driven sizing, listed as not yet done.

**Open startup PRs** (titles from the search JSON)
- #21886 "[None][perf] Capture the final CUDA graphs without the warm-up-only pass (opt-in)" (2026-11-08, open).
- #21628 "KernelBuilder … build kernels on worker threads while the weights load" (open).

**No lazy, idle or background capture option exists in TensorRT-LLM as of v1.4.0rc3** (searched titles: `lazy`, `on-the-fly`, `capture … warmup`).

---

## 5. Other engines and systems (capture off the critical path or lazily)

### 5.1 LMDeploy (PyTorch engine), v0.18.0 (2026-09-28), verified in code

- **Decode: full graphs, captured on a cache miss, synchronously.**
  - `CUDAGraphRunner._forward_full_graph()` ([lmdeploy/pytorch/backends/cuda/graph_runner/runner.py @ v0.18.0](https://github.com/InternLM/lmdeploy/blob/v0.18.0/lmdeploy/pytorch/backends/cuda/graph_runner/runner.py)) looks up `graph_key` in `_full_graph_runners`. On a miss it calls `_capture_full_graph()`, docstring "Capture and publish a full graph for one cache miss". That capture does one eager warmup, then capture, in `full_graph.py`.
  - Key: `(padded batch_size, is_decoding, enable_microbatch, query_len) + model extra key`.
  - Padding: `_get_capture_tokens()` returns the smallest capture size ≥ the batch. It asserts if the batch is larger ("Unsupported batch_size").
  - Default sizes: powers of two up to `min(256, max_batches)`, then steps of 256, plus `max_batches` (`_get_capture_batch_size_impl`). They can be overridden with `cudagraph_capture_batch_sizes`.
  - One shared pool: `self._full_graph_pool_handle = torch.cuda.graph_pool_handle()`.
  - Capture uses `capture_error_mode='thread_local'` (`full_graph.py` L211), which is consistent with capture that may happen during serving.
- **Startup pre-capture by default.**
  - `ModelAgent.warmup()` ([engine/model_agent/agent.py @ v0.18.0 L518–566](https://github.com/InternLM/lmdeploy/blob/v0.18.0/lmdeploy/pytorch/engine/model_agent/agent.py#L518-L566)) runs a dummy decode for every capture batch size, **largest first** (`sorted(..., reverse=True)`). Each run triggers the miss-capture.
  - With DP > 1 it first does a CPU-group barrier: "make sure warmup started together".
  - Warmup is skipped with `LMDEPLOY_SKIP_WARMUP=1` (default `False`, `lmdeploy/pytorch/envs.py` L191). Then **all decode graphs are captured lazily on first use while serving**.
  - Lazy miss-capture also happens for keys warmup did not cover, and after `reset_graph_runner()` (e.g. after weight updates), where warmup is re-run only when prefill piecewise graphs are active.
- **Prefill: piecewise graphs (BCG-like, no FX), startup only.** From the README of the graph_runner package:
  > "All candidate buckets are captured at startup and retained; there is no runtime capture or eviction."
  > "Serving never captures a piecewise plan. If startup warmup is disabled or no matching plan exists, eager execution is selected before piecewise execution can mutate KV or state caches."
  - The fixed 512-token bucket stride is "provisional".
- **History.** The companion file reports lazy-only capture in v0.7.0 (no startup pre-capture) and a `warmup()` by v0.10.0. That is consistent with the above.
- **TP lockstep.** It is implicit: every rank sees the same batch, so it misses on the same step. This is SPMD by construction, not an explicit agreement.

### 5.2 Hugging Face TGI v3.3.7 (2025-12-19; repo now "in maintenance mode" per README)

- Launcher default `--cuda-graphs` = `[1, 2, 4, 8, 16, 32]` ([launcher/src/main.rs @ v3.3.7](https://github.com/huggingface/text-generation-inference/blob/v3.3.7/launcher/src/main.rs), L2169–2185). Graphs are disabled for bitsandbytes and exl2.
- Sizes are sorted descending: "sorting the cuda graphs in descending order helps reduce the memory impact" (`server/text_generation_server/models/globals.py` L58–61).
- They are all captured in `FlashCausalLM.warmup()` via `cuda_graph_warmup(bs, …)`, into one global pool `MEM_POOL = torch.cuda.graph_pool_handle()` (globals.py L30; `flash_causal_lm.py` L1483).
- **Runtime.** The smallest captured bs ≥ batch is used (padding). Above the largest, or any prefill (`cu_seqlen_prefill is not None`), runs eager (`flash_causal_lm.py` L1793–1800).
- **No lazy capture.**

### 5.3 MLC-LLM / Apache TVM: lazy capture on first call per shape (verified in code)

- TVM `vm.builtin.cuda_graph.run_or_capture` is `CUDAGraphExtensionNode::RunOrCapture` in [src/runtime/vm/cuda/cuda_graph_builtin.cc @ v0.27.0.post1](https://github.com/apache/tvm/blob/v0.27.0.post1/src/runtime/vm/cuda/cuda_graph_builtin.cc). Its doc comment: "Launch the cuda graph if it has been cached, otherwise execute it in capture mode".
  - The key is `(entry_index, symbolic shape values)`.
  - On a miss it runs the function once uncaptured, as "a warm up step to do necessary initialization of the CUDA module", then captures with `cudaStreamCaptureModeGlobal`, instantiates and caches.
- MLC-LLM's default compile preset is `--opt O2`, which sets `cudagraph=True` (`python/mlc_llm/interface/compiler_flags.py`, `OPT_FLAG_PRESET`; default in `python/mlc_llm/cli/compile.py`; read on `main`).
- So MLC engines capture **lazily, per shape, synchronously** on first use.
- Whether MLC's serving engine pre-warms shapes at startup: **UNVERIFIED** (not checked).

### 5.4 llama.cpp

Per the companion file (NVIDIA blog, 2024), graphs are captured at runtime, reused, and updated with `cudaGraphExecUpdate` instead of being recaptured. That covers batch size 1 only. Not re-checked in this session.

### 5.5 Graph persistence / templating (avoid capture instead of deferring it)

- **Foundry** (arXiv 2604.06664, v1 2026-04-08; Liu, Wu, Yao, Zhuo, Stoica, Mao). Details are in the companion file. Abstract and HTML read here:
  - It persists graph topology plus execution context offline. It builds one template per unique topology and specializes others **on demand at serving time** via `cuGraphExecUpdate`, at 0.98–2.89 ms per update vs 59.7–198.6 ms per stream capture.
  - Qwen3-235B-A22B EP8 vLLM init: 650 s → 3.9 s.
  - Qwen3-30B-A3B EP2–EP8: 112–154 s → 2.7–2.8 s.
  - The prototype was on vLLM v0.11.2.
  - Engine PRs:
    - vLLM **#66472** "[Fast Start] Persist CUDA graphs with Foundry and prepare them during preload" (open, 2026-11-09, UNIDY2002). FULL decode graphs only, opt-in. Qwen3-0.6B on B300 with 129 sizes: startup −0.994 s (−4.0%). Archived graph construction 214/288 ms vs about 3 s native capture. One SAVE preparation 30.889 s. "Disk TP2/TP4 restoration is validated with NCCL; FlashInfer fused all-reduce and custom all-reduce buffer restoration need further support."
    - SGLang **#42254** "[Perf] Foundry: adding the adapter of foundry to enable fast CUDA graph save/load" (open, 2026-10-02). Earlier attempts #41032 and #41791 were closed.
- **SGLang CUDA-graph executable dedup.**
  - **#29625** "CUDA graph executable dedup via cudaGraphExecUpdate" (merged 2026-06-29, cctry). Structurally identical breakable-graph segments across batch sizes share one `cudaGraphExec`, switched with `cudaGraphExecUpdate` at replay. `SGLANG_ENABLE_CUDA_GRAPH_DEDUP`, default off.
  - Follow-ups: #37657, #39176, #47295 (all merged).
  - **#37411** "fix: lazily instantiate CUDA graph compatibility executables" (closed unmerged 2026-09-16). GLM-5.3 FP8 TP16 on H20: executable instantiations per rank 12 → 6; instantiation time 220.0 → 89.1 ms per rank; +234 MiB free HBM.
- **Snapshot / restore.** vLLM RFC #34303, vLLM roadmap #48193 (#51360 TP1 snapshot restore merged), SGLang RFC #43965 "SGLang Engine Initialization Snapshots: Reusing Initialized State for Faster Startup" (open, 2026-10-13). See the companion file.

### 5.6 Background / overlap work at startup (capture itself stays foreground)

- SGLang #32017 overlaps page-cache weight staging with capture (§2.1).
- TensorRT-LLM:
  - #21628 "KernelBuilder … build kernels on worker threads while the weights load" (open, 2026-11-04).
  - #18542 "Overlap TRTLLM-Gen FMHA JIT warmup compilation with engine warmup (opt-in)" (open, 2026-09-01).
  - #21886 "Capture the final CUDA graphs without the warm-up-only pass (opt-in)" (open, 2026-11-08).
- **sglang-omni (SGLang's omni-model companion repo) does live, lazy capture during serving, and hit the threading hazard.** Issues verified on the GitHub pages:
  - [#825](https://github.com/sgl-project/sglang-omni/issues/825), "MOSS streaming vocoder lazy CUDA-graph recapture can race colocated AR .tolist() under global capture mode" (opened and closed 2026-06-19). A vocoder graph is "re-captured … synchronously" for a new streaming session after the startup session was closed. Under the default `capture_error_mode="global"`, a sibling AR thread's `.tolist()` fails with "operation not permitted when stream is capturing". Fix: `capture_error_mode="thread_local"`.
  - [#829](https://github.com/sgl-project/sglang-omni/issues/829), "Fix Ming Omni live/lazy CUDA graph capture to use thread-local capture error mode" (2026-06-19 → closed 2026-06-20) states the general rule: "CUDA graph capture that may happen after serving starts should not use PyTorch's default global capture error mode". It lists lazy capture sites `CFMGraphExecutor._initialize_graph()` and `MingOmniTalker.generate()`.
  - These are the only *live* capture-during-serving sites found in any SGLang-family repo. They are model-local (vocoder or talker), not the LLM runner.
- **No engine was found that runs CUDA-graph capture in a background thread concurrently with serving.**
  - `torch.cuda.graph`'s default `capture_error_mode="global"` "will error on actions in other threads". See the `capture_begin` docstring in [torch/cuda/graphs.py @ v2.15.0](https://github.com/pytorch/pytorch/blob/v2.15.0/torch/cuda/graphs.py), L610–614.
  - **Default `"global"`:**
    - vLLM v0.33.0: `CudaGraphManager.capture` and `CUDAGraphWrapper` via `torch.cuda.graph(...)`, and `BreakableCUDAGraphCapture` via `g.capture_begin(pool=...)`.
    - TGI v3.3.7.
    - TRT-LLM's decode runner, `nccl_window_graph_capture` with no mode (`cuda_graph_runner.py` L913). TRT-LLM passes `"thread_local"` at one other site (L2791–2794).
  - **`"thread_local"`:**
    - **LMDeploy v0.18.0**, for both full and piecewise captures (`full_graph.py` L211, `piecewise.py` L643).
    - **CUDA Graph Trees**, which also serializes capture with a process-wide lock.
  - TVM uses `cudaStreamCaptureModeGlobal`.
  - Background capture concurrent with serving would therefore need relaxed or thread-local capture and an allocator-pool discipline. It would also need collectives that do not interleave with serving collectives. Idle-time capture on the scheduler thread avoids all three (derived).

---

## 6. How close is prior art to "lazy, traffic-driven capture" (synthesis)

| Feature of our idea | Closest prior art | Gap |
|---|---|---|
| Serve after capturing a **small** set | vLLM PR #23184 (largest size at init, then runtime capture); vLLM #57899 (serve eager after wake) | #23184 closed. #57899 is post-wake only |
| Capture remaining sizes **one at a time in scheduler idle time** | vLLM #57899 ("at most one configured shape" per idle boundary); RFC #20098 and PR #23184 suggestions | #57899: FULL-only, MRV2, no torch.compile, opt-in, sleep/wake only |
| **Lockstep across TP ranks** | #57899 (executor RPCs; multi-worker failure kills the executor); vLLM DP all-reduce of `cg_mode` (`sync_cudagraph_and_dp_padding`) | Nobody publishes a design for agreeing *which* size to capture next across ranks beyond "same deterministic order" |
| **Traffic-driven order / promotion of hot eager sizes** | None found. TRT-LLM blog 20 lists "parse serving logs to compute the best set of CUDA graph batch sizes … based on the recorded concurrency distribution" as future work (offline; verified). PR #23184 captured the *current* shape first | Novel |
| Lazy capture on first use | PyTorch CUDA Graph Trees; TVM/MLC `run_or_capture`; LMDeploy decode miss-capture (with `LMDEPLOY_SKIP_WARMUP=1`, or for keys not warmed); TRT-LLM ≤ v1.2.1 implicit; SGLang EAGLE post-verify graph (opt-in); sglang-omni vocoder/talker graphs; vLLM `CUDAGraphWrapper` mechanics | All synchronous on the critical path; no idle scheduling. sglang-omni shows that live capture needs `capture_error_mode="thread_local"` when other threads touch CUDA |
| Avoiding capture altogether | Foundry (templates and on-demand `cuGraphExecUpdate`, vLLM PR #66472 open); snapshot/restore (vLLM RFC #34303, #48193 roadmap) | Needs an offline step or driver C/R (see companion file) |
| Reserving memory for graphs not yet captured | vLLM `profile_cudagraph_memory()` (captures the 2 largest FULL graphs and extrapolates; PIECEWISE in full); TRT-LLM dry-run capture of all graphs; RFC #20098 ideas ("size of 1 cudagraph … with a safety factor", LucasWilkinson; memory warning after the first, largest capture, ProExpertProg on #23184) | vLLM's estimate already under-reserves at high utilization (#57475). A lazy scheme must reserve for the uncaptured tail up front |
| Warm (JIT) the exact shape eagerly before capturing it | Every engine does one or two eager warmups per size before capture (vLLM 1; TRT-LLM 2 in v1.2.1, 1 in v1.4.0rc3; TVM 1; LMDeploy 1; PyTorch trees 1 call) | Lazy captures that skip this fail: SGLang #49206 (Dynamo JIT inside lazy capture), #29942 (lazy buffer init inside capture). vLLM #57227 and SGLang #44627 show that first-seen shapes JIT at request time |

**Objections and hazards to pre-empt** (verified text)
- Reliability of runtime capture (WoosukKwon).
- Maintenance burden (youkaichao).
- Unpredictable TTFT (zou3519, ProExpertProg).
- OOM after the server is "up" (LucasWilkinson, yeqcharlotte).
- Multi-TP first-request garbage (a user report on #23184; not reproduced).
- A capture that resizes shared workspaces invalidates earlier graphs (TRT-LLM code comment after #12306; vLLM `lock_workspace()`).
- The eager window is expensive: −29.3% to −77.4% throughput while recapturing in #57899.
- Collectives inside capture: a failed capture on one rank can leave peers blocked in a collective. #57899 terminates the executor for that reason.

---

## 7. Open items / UNVERIFIED

- MLC-LLM: whether its serving engine pre-warms shapes at startup is not checked. llama.cpp was not re-checked; see the companion file.
- The GeometricAGI fork issue #66 ("captures graphs twice") is known only from a search snippet. The double capture itself is verified in vLLM code (§1.7).
- PR states and merge dates in this file come from the GitHub issue-search JSON (`state`, `createdAt`, `mergedAt`) or from the PR page. Where only a title is quoted, the PR body was not read.
- MRV2's behaviour when a PIECEWISE descriptor is uncaptured, i.e. a silent capture by the inner wrapper, is derived from code. It is not exercised.
- Some rows were first read through a page summarizer and then checked with targeted greps of the page text; those are marked "(Verified)". The remaining summarizer-only details are the exact comment dates inside RFC #20098 and PR #23184. The quoted text is verified; only which day each comment was posted is not.
- A github-actions bot comment on vLLM PR #57899 contains a block addressed to AI agents (asking them to re-evaluate or close the PR). It was ignored as page content, not an instruction.
- Spheron blog claims (BCG default "shipping in v0.5.15") are now confirmed by the SGLang release notes. Its fallback conditions for BCG are **UNVERIFIED**.
