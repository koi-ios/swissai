# Working notes: lazy CUDA-graph capture (idea 3)

Running log for the research phase. Facts with file:line. Survives context compaction.

## Status
- 11 Nov: read EVALUATION.md, CLAUDE.md, harness docs, proposal plan, analysis scripts and results.

## Paths
- vLLM v0.33.0 sparse clone (read-only, shared): /private/tmp/claude-501/-Users-koiios--claude-spares-spare-qHWvhZ/85cbb828-b415-43aa-adc6-b088d62505ae/scratchpad/src/vllm (commit 4ba787e)
- SGLang 0.5.20 full clone (from earlier session): /Users/koiios/Documents/Projects/swissai/opportunities/04-lazy-cuda-graphs/vendor/sglang-v0.5.20
- Shared repos: $SHARED/repos/{servekit,model-launch,coldstart-exp,sglang-0.5.16,sglang-0.5.20}
- Trace parquet: /Users/koiios/Documents/Projects/swissai/data/swissai-trace.parquet (cols model,status,t0,t1,tin,tout,max_tokens)
- My scratch: /private/tmp/claude-501/-Users-koiios--claude-spares-spare-qHWvhZ/85cbb828-b415-43aa-adc6-b088d62505ae/scratchpad/lazy-cuda-graphs

## Lab numbers to reuse (EVALUATION.md)
- Post-servekit Clariden 0.5.16 overlap arm: Apertus-8B TP4 total 95.0 s, prefill graphs 27.6 s (29%), decode graphs 5.5 s; Llama-70B TP4 total 116.8 s, prefill 46.3 s (40%), decode 8.0 s; GLM-4.7 TP4 PP4 decode 30.8 s (29%), first request 26.1 s.
- Prefill capture: 58 sizes (8192 down to 4), largest first; step 1 (8192) = 12-18 s Clariden, 5 s (cache) / 28 s (no cache) Bristen. Per size 1-3 s above 4096 tokens on 70B, ~0.25 s below 512. Decode 36 sizes (1..256), 0.1-0.3 s each.
- Ceiling savings (excl. step 1): 70B 2048->12 s, 1024->13-14 s, 512->20-21 s; decode cap 32->5-8 s. 8B: 3, 8-9, 9-10 s; decode 3 s.
- Trace: Apertus-70B-Instruct in-flight p99 10, p99.9 40, max 48; prompts p50 356 p99 1341; idle 96%. Bulk: Qwen3-Next p99 243; gpt-oss p99 219; Qwen3-32B p99 885.
- Churn: 93 names >=100 req; 68 on <=7 days; idle 78-99% of span; 4-17 idle gaps >=15 min/day.

## WARNING: parallel session on the same idea (found 20:24, 11 Nov)
- /Users/koiios/Documents/Projects/swissai/opportunities/04-lazy-cuda-graphs/ is being written by ANOTHER session right now (dirs created 20:18; vendor/sglang-v0.5.20/python/sglang/srt/model_executor/runner/lazy_capture.py written 20:23, untracked `??` in git status).
- That lazy_capture.py (LazyCaptureSchedule, pick_next_phase; docstring mentions scheduler_components/lazy_graph_capture.py which does not exist) is NOT upstream SGLang: the shared sparse checkout at the same commit 94602c9 lacks it. Never cite it as prior art. Tell the user so they can reconcile the two efforts.
- Do not write there.

## vLLM 0.33 facts (src/vllm @ v0.33.0 = 4ba787e)
- Default capture list: [1,2,4] + range(8,256,8) + range(256, max+1, 16), max = min(2*max_num_seqs*decode_query_len, 512) (1024 on Blackwell). With max 512: 3+31+17 = 51 sizes == "51 graphs" in #270. vllm/config/vllm.py:2398-2656 (_set_cudagraph_sizes).
- Default cudagraph_mode FULL_AND_PIECEWISE (config/compilation.py:610-645). PIECEWISE graphs for mixed batches, FULL for uniform decode.
- Dispatch above max size -> NONE (eager): v1/cudagraph_dispatcher.py:272-279. Pads to smallest captured size >= n (_bs_to_padded_graph_size, :71-90).
- Upstream anticipates lazy capture: cudagraph_dispatcher.py:184-186 ("if we allow lazy capturing in future PR, some keys may never be triggered"); gpu_model_runner.py:6498-6503 ("we may do lazy capturing in future that still allows capturing after here"). Not implemented in 0.33.
- CUDAGraphWrapper captures on first call of an unseen key if capture enabled (compilation/cuda_graph.py:258-345); validate_cudagraph_capturing_enabled() raises after startup (compilation/monitor.py:98-115; set False at gpu_model_runner.py:6503).
- Capture output is not computed during capture (graph recorded, not run) -> lazy capture must use dummy batches in idle time, not a live batch.
- Memory reserved up front: profile_cudagraph_memory captures 2 largest descs per mode and extrapolates first + (n-1)*per_graph (gpu_model_runner.py:6268-6414); subtracted from KV budget when VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS=1 (default) (gpu_worker.py:702-757).
- lock_workspace() after capture (gpu_model_runner.py:6508-6510).
- Model Runner V2 is the DEFAULT in 0.33 when supported (config/vllm.py:731-767). MRV2 capture: v1/worker/gpu/cudagraph_utils.py CudaGraphManager; dispatch returns NONE until _graphs_captured, then candidate = smallest captured >= n (:523-553); run_fullgraph asserts desc in self.graphs (:555-568) -> lazy needs a dispatch patch.
- Breakable CUDA graphs default for DeepseekV41ForCausalLM, GlmMoeDsaForCausalLM, Glm5Next*, KimiK3, Qwen4Exp... (config/vllm.py:76-129). "DeepseekV41ForCausalLM cannot torch.compile".
- VLLM_ENABLE_STARTUP_PLAN=1 persists kv_cache_memory_bytes and skips memory profiling + CUDA-graph memory estimation (v1/worker/startup_plan.py). Default off.
- Explicit JIT warmup contract (kernel_config.enable_jit_warmup=True default): kernels declare compile keys; warmup enumerates keys over WarmupIntRange(1, max_num_batched_tokens+1) and compiles before capture (model_executor/warmup/jit_warmup.py:1008-1145; docs/contributing/jit_kernel_warmup.md in v0.30.0). JIT monitor flags runtime JIT afterwards (gpu_worker.py:1126-1138).
- TileLang mHC kernels use T.dynamic("num_tokens") (kernels/mhc/tilelang_kernels.py:171 etc.) -> compile keys do not scale with captured sizes; n_splits bucketed to {1,4,16} in 0.30 (kernels/mhc/warmup.py compute_mhc_pre_num_splits). vLLM 0.30.0 already had this (vllm-030 clone).
- => Hypothesis (c) "deferring capture defers per-shape TileLang JIT" is unlikely for vLLM >= 0.30 mHC kernels. The 40 min in #270 remains unexplained (DeepGEMM via FlashInfer? kernels outside the warmup contract? TP8 over 2 nodes with Ray). Needs the log.
- #270 run used vllm_base ~= vllm 0.30.0 at the time (vllm_base renamed from vllm_0.30.0.toml in #265, 19 Oct; compile-cache worker NOTES). DSV4.1-Flash: 2 nodes, TP8, Ray.

## SGLang 0.5.20 facts (vendor clone, upstream files only)
- Prefill default sizes: generate_prefill_cuda_graph_batch_sizes(8192) = 58 sizes (arg_groups/cuda_graph_hook.py:557-574). Decode: [1,2,4,8,12]+range(16,257,8)+... (:577-609).
- Prefill eager above max and when padding > 2x (_MAX_PREFILL_CUDA_GRAPH_PADDING_FACTOR = 2, runner/prefill_cuda_graph_runner.py:155, can_replay_locally :1129-1192).
- Capture loops largest first: prefill _capture_one_stream (:1427-1451), decode (:1060-1110).
- "Capture runs real forwards, so a mid-serving recapture would index -- and write KV -- through the previous batch's live values" -> reset_index_buffers() (decode_cuda_graph_runner.py:1023-1026).
- SGLang plugin system: entry point group sglang.srt.plugins + HookRegistry AROUND/BEFORE/AFTER (srt/plugins/). servekit uses it (pyproject: servekit_pp = "servekit._shim:install").
