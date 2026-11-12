# Engine determinism and logprob-tolerance precedents

Checked 11–12 Nov 2026 against primary sources. Pinned versions: **SGLang v0.5.20** (PyPI 18 Sep 2026; newest is 0.5.23, 30 Oct 2026) and **vLLM v0.33.0** (PyPI 2 Nov 2026). "src@tag" means the source was read on raw.githubusercontent.com at that tag.

## 1. Thinking Machines Lab, "Defeating Nondeterminism in LLM Inference"
- https://thinkingmachines.ai/blog/defeating-nondeterminism-in-llm-inference/ (Horace He and TML, 10 Sep 2025). Code: https://github.com/thinking-machines-lab/batch_invariant_ops
- **Argument:** forward kernels are run-to-run deterministic (no atomics), but not *batch-invariant*: a request's reduction order depends on batch size, which depends on server load. TP and collectives are excluded ("Reductions related to parallelism are out of the scope").
- **Ops:**
  - RMSNorm: data-parallel, with one row per core and no split reduction.
  - Matmul: one kernel config for every shape and no Split-K ("we only lose about 20% performance compared to cuBLAS").
  - Attention: the KV cache and page table are updated before the kernel, so the layout does not depend on chunked prefill or prefix caching. Split-KV uses a "fixed split-size" ("instead of fixing the # of splits, we fix the size of each split").
- **Experiment:** Qwen3-235B-A22B-Instruct-2507, T=0, 1000 completions × 1000 tokens. The result was **80 unique completions** (the most common occurred 78 times). The first divergence was at token 103 (992 "Queens, New York" vs 8 "New York City"). With batch-invariant kernels, "all of our 1000 completions are identical".
- **Performance** (one GPU, type not stated; Qwen-3-8B; 1000 sequences; 90–110 output tokens): vLLM default **26 s**, unoptimized deterministic **55 s**, with improved attention **42 s** (the "61.5%" slowdown that SGLang cites).
- **Repo:** uses `torch.Library` to override `mm`, `addmm`, `log_softmax` and `mean` via `set_batch_invariant_mode()`. The PoC `deterministic_vllm_inference.py` goes from 18 to 1 unique outputs over 1000 completions. The post never claims batch-invariant output equals default-kernel output.

## 2. SGLang deterministic inference
- **Sources:**
  - Blog: https://lmsys.org/blog/2025-09-22-sglang-deterministic/ (22 Sep 2025; requires `sglang>=0.5.3`).
  - Docs: https://docs.sglang.io/advanced_features/deterministic_inference.html
  - Roadmap: https://github.com/sgl-project/sglang/issues/10278 (closed Feb 2026).
- **Flag:** `--enable-deterministic-inference`. **Present in v0.5.20** as `ExecDeterministic.enable_deterministic_inference` (default False) in `python/sglang/srt/arg_groups/fields/exec_.py`; src@v0.5.20 moved the server args into `arg_groups/`. `--rl-on-policy-target fsdp` force-enables it. Logic lives in `arg_groups/attention_hook.py::handle_deterministic_inference` and `arg_groups/overrides.py`.
- **What v0.5.20 does when the flag is on (src@v0.5.20):**
  - **Attention:** `DETERMINISTIC_ATTENTION_BACKEND_CHOICES` = ascend, fa3, fa4, flashinfer, intel_xpu, triton; any other backend raises ValueError. If unset, it uses fa3 on Hopper and older, and flashinfer on SM100/SM120 (triton for DeepSeek-family MLA).
  - **Radix cache:** kept only for `RADIX_SUPPORTED_DETERMINISTIC_ATTENTION_BACKEND` (ascend, fa3, fa4, triton). Otherwise, including FlashInfer, it sets `disable_radix_cache=True`.
  - **Chunked prefill:** the truncation point is aligned to 4096 for flashinfer (`SGLANG_FLASHINFER_PREFILL_SPLIT_TILE_SIZE`) and triton (`SGLANG_TRITON_PREFILL_TRUNCATION_ALIGN_SIZE`). The Triton decode split is set by `--triton-attention-split-tile-size`.
  - **Batch-invariant aten overrides** (`srt/batch_invariant_ops/`): mm, addmm, _log_softmax, mean.dim, rms_norm, mm.dtype, bmm, using a Triton persistent matmul or a DeepGEMM path. BF16 GEMM backend `auto` selects `torch`, not the cutedsl split-K kernel, on SM10x.
  - **Sampling:** the backend is forced to `pytorch`. Temperature > 0 is reproducible through per-request `sampling_seed` (default 42), which uses `multinomial_with_seed` (Gumbel noise from a seeded hash).
  - **TP>1 (CUDA):** sets `NCCL_ALGO=allreduce:tree`, pins `NCCL_MIN/MAX_NCHANNELS` to `SGLANG_DETERMINISTIC_NCCL_NCHANNELS` (default 8), and disables custom, symm-mem, FlashInfer and AITER all-reduce paths (ROCm uses a 1-stage all-reduce). This gives batch invariance **at a fixed TP degree**, not across TP degrees.
  - **CUDA graphs:** decode graphs are kept; the torch.compile piecewise prefill graph is disabled.
  - **MoE / quantization:** sets `SGLANG_FLASHINFER_MOE_FUSED_FINALIZE=0`, and the v0.5.20 release note adds "keeping router GEMM in fp32 for deterministic inference (DeepSeek V3/V4)". The flag hard-errors with DeepEP v2, speculative rejection sampling and UNO. The roadmap has blockwise FP8 (#11491); per-channel FP8, FP8 fused MoE and NVFP4 had no PRs.
- **Measured cost** (blog; Qwen3-8B, TP1, H200, 256 requests; in/out 1024 / 4096 / 8192):
  - FlashInfer +42.6 / +46.0 / +24.4%; FA3 +27.2 / +30.2 / +35.7%; Triton +55.1 / +44.64 / +44.80%.
  - "average slowdown of only 34.35% with the FlashInfer and FlashAttention 3 backends".
  - LLM-42 (item 4) measured SGLang-Deterministic **24–36% slower** (Llama-3.1-8B, 4×H100, v0.5.3rc0).
- **Harness** `python3 -m sglang.test.test_deterministic` (`python/sglang/test/test_deterministic.py`):
  - **Modes by version:** v0.5.3 has `single|mixed|prefix`. **v0.5.20 has `single|prefix|radix_cache|p_vs_d`** ("mixed" was removed).
  - `single`: one prompt at batch sizes 1..n; counts unique texts.
  - `prefix`: random batches drawn from prefix lengths 1/511/2048/4097, cache flushed per batch. With `--return-logprob` it compares logprobs at tolerance 0, but only prints the result.
  - `radix_cache`: cached vs uncached prefill. Passes only if token ids and logprobs are exactly equal.
  - `p_vs_d`: decode output vs re-prefill of prompt+output. Passes only on exact token ids and logprobs; on a mismatch it prints a K3 KL estimate.
  - Defaults: `--n-trials 50 --temperature 0.0 --sampling-seed 42 --max-new-tokens 100`; expected "Unique samples: 1" (blog, FlashInfer: prefix 5/8/18/2 → all 1).

## 3. vLLM batch invariance
- **Enable:** `VLLM_BATCH_INVARIANT=1`. Status is beta.
  - Docs: https://docs.vllm.ai/en/latest/features/batch_invariance/ (= `docs/features/batch_invariance.md` src@v0.33.0).
  - Tracker: https://github.com/vllm-project/vllm/issues/27433 (open; activity up to Nov 2026).
- **Hardware** (v0.33.0 doc): NVIDIA CC ≥ 8.0; AMD gfx90a/gfx942/gfx950; Intel XPU with Triton (requires `TRITON_ATTN`). The v0.12.0 doc said CC ≥ 9.0, and the comment in `envs.py` still says 9.0.
- **Attention:** backends where `supports_batch_invariance()` is True (src@v0.33.0):
  - FLASH_ATTN (num_splits=1).
  - TRITON_ATTN.
  - FLEX_ATTENTION (16×16 blocks).
  - TRITON_MLA (num_kv_splits=1).
  - FLASH_ATTN_MLA.
  - **FLASHINFER is not supported.** The test utils say "temporarily disabled due to invariant CTA sizes" (FlashInfer #2424). Unsupported backends are rejected at selection.
- **What `init_batch_invariance()` does** (`vllm/model_executor/determinism/batch_invariant.py`):
  - **Matmul:** on SM80, ROCm and XPU it overrides aten mm/addmm/matmul/linear with a Triton persistent matmul. On Hopper/Blackwell it keeps cuBLAS with `CUBLAS_WORKSPACE_CONFIG=:16:8` and `CUBLASLT_WORKSPACE_SIZE=1` (no split-K).
  - **Other ops:** overrides mean.dim, softmax/_log_softmax and bmm; turns off reduced-precision reductions and TF32.
  - **Collectives:** `VLLM_CUSTOM_ALLREDUCE_ALGO=1stage`; NCCL pinned (`NCCL_ALGO=ring,tree;allreduce:tree`, `NCCL_PROTO=Simple`, 1 channel, NVLS off); FlashInfer, AITER and QuickReduce all-reduce and the AR+RMSNorm fusion disabled. The module also defines fixed-order all-gather+sum collectives. Doc warning: all-reduces whose size is not a multiple of 16 bytes are not covered.
  - **Compilation:** bf16 unquantized models with tuned tables (Ada/Hopper/Blackwell) run without torch.compile using breakable CUDA graphs (`VLLM_USE_BREAKABLE_CUDAGRAPH=0` opts out). Sets `VLLM_USE_AOT_COMPILE=0`. LoRA shrink is summed in 8 fixed K chunks.
- **Models tested (doc):** DeepSeek V3/R1/V3.1; Qwen3 dense (incl. AWQ) and MoE (incl. Qwen3-30B-A3B-Thinking-2507-**FP8**); Qwen2.5; Llama 3.1/3.2; **gpt-oss-20b/120b**; Granite, Phi, GLM-4, EXAONE and others.
- **Quantization and caching:** the tracker lists CUTLASS FP8 (#40408), NVFP4 linear (#39912) and MoE (#40372), deterministic MoE combine (#45683, merged), and AWQ/GPTQ (#60479). Known gaps are Marlin GPTQ/AWQ MoE, FLASHINFER_MLA and GLM-5.3 DSA. **Prefix caching is listed as "not done"**, and many invariance tests set `enable_prefix_caching=False`.
- **Cost:** the docs give no number. Issue #65569 (H100, main at 0.33.0rc1.dev454; decode at batch 32 vs mode off):
  - **+63.1%** Qwen2.5-7B.
  - +59.2% Gemma-2-9B.
  - +58.2% Qwen2.5-32B TP2.
  - +61.0% Qwen2.5-7B TP2.
  - Prefill +9.5–25.5%.
- **CI** (`tests/v1/determinism/test_batch_invariance.py`): bitwise (`torch.equal`) logprob checks for BS1 vs BSN, ragged chunked prefill, decode vs prefill and FP8-MoE, plus an inverse test that expects mismatches with the mode off.
- **Blogs:**
  - https://vllm.ai/blog/2025-11-10-bitwise-consistent-train-inference (10 Nov 2025; TorchTitan + vLLM, Qwen3-1.7B; `kl_div` 0.0; "2.4x slower than the non-bitwise case"; eager mode only).
  - Later posts (2026-10-14 train-inference-consistency; 2026-08-21 isoexec) report 9–31% rollout overheads with custom kernels.
- **Gotcha:** `VLLM_SYSTEM_START_DATE` (envs.py src@v0.33.0). The gpt-oss Harmony template injects the current date, which "introduces non-determinism … at temperature=0", so pin it.

## 4. TP-size invariance and verified speculation
- **TBIK**, arXiv:**2511.17826**, "Deterministic Inference across Tensor Parallel Sizes That Eliminates Training-Inference Mismatch":
  - Zhang et al.; v2 29 May 2026; ICML 2026 poster https://icml.cc/virtual/2026/poster/66224 (matched by title and authors); code https://github.com/nanomaoli/llm_reproducibility.
  - Claim: tree-ordered matmul for row-parallel layers plus a custom tree all-reduce give "bit-wise identical results regardless of TP size" in vLLM and FSDP (chunked prefill disabled).
  - On L40S (TP 1/2/4/8 × BS 8/16/32), Qwen3-8B AIME24 averages 12.00 unique outputs in BF16, 7.87 with batch-invariant kernels only, and 1 with TBIK. TP alone moves accuracy by >4%.
  - Overhead: **22–63%** end-to-end (Qwen3-8B, 4×H20, TP4).
- **LLM-42**, arXiv:**2601.17768** (Gond, Kamath, Ramjee, Panwar; v2 30 Jan 2026; code "will be" at https://github.com/microsoft/llm-42):
  - Mechanism: a fast nondeterministic decode path, plus a verifier that replays a fixed window under a fixed-shape reduction schedule and rolls back on mismatch. Built on SGLang 0.5.3rc0.
  - Within 8/2/1/1% of nondeterministic SGLang at 5/10/20/50% deterministic traffic.
  - Recomputation 6.81% (window 32) to 46.41% (window 256).
- **Related:** arXiv:2609.25624 uses fixed-order FP32-accumulate GEMMs; the linear layers are bitwise identical across Ampere/Ada/Hopper.

## 5. Reproducibility studies
- **Yuan et al., arXiv:2506.09501.** v1 (11 Jun 2025) was titled "Give Me FP32 or Give Me Death? Challenges and Solutions for Reproducible Reasoning"; v2 (24 Oct 2025) is "Understanding and Mitigating Numerical Sources of Nondeterminism in LLM Inference".
  - Setup: vLLM; {L40S, A100} × {2, 4 GPUs} × BS {8, 16, 32}.
  - Greedy, R1-Distill-Qwen-7B, AIME'24: Std@Acc **BF16 9.15%, FP16 5.74%, FP32 0**; output-length std 9189.53 / 5990.32 / 0.
  - MATH500 BF16: ">90%" of examples diverge vs FP32 "2.2%" (the table says 5.80%).
  - Mean first-divergence index 65.33 (BF16) vs 1825.79 (FP32).
  - Divergence happens at near-ties ("know" 49.75% vs "have" 46.65%).
  - LayerCast (16-bit weights, FP32 compute) matches FP32 stability with 34% less memory than FP32.
- **Atil et al., arXiv:2408.04667** (v5 Apr 2025): five LLMs (GPT-3.5/4o, Llama-3-8B/70B, Mixtral-8x7B) at T=0 with a fixed seed, 8 tasks × 10 runs. "accuracy variations up to 15%" and a best-to-worst gap "up to 70%" (metrics TARr@N and TARa@N).
- **Fu et al., arXiv:2601.06118** (Jan 2026; BF16; A100/A6000/H200/Ascend; 50 runs): about 90% of top-10 token probabilities vary by less than 0.01% in range, and about 1% reach 5%. The variation is concentrated at p ∈ 0.1–0.9.
- **Du et al., arXiv:2609.26621** (TMLR 2026; BF16 vs FP16): 49–100% of prompts diverge. Whether a token flips depends on the **top-2 logit margin at the LM head**. FP32 LM-head recompute stops helping at batch ≥ 8 and under end-to-end FP8.
- **FP8/FP4 run-to-run variance:** no dedicated study found. Closest: DiFR (item 6) needs about 1–5k tokens to separate an FP8 KV cache from benign noise. FlashInfer PRs #8537/#8565 (batch-invariant NVFP4/MXFP4 GEMM and quantizers) opened 3 Nov 2026. arXiv:2609.00363 (INT8): injected faults that move outputs ≤ 1 BF16 ulp evade tolerance checks ("four of the five faults are detected by no check in the suite").

## 6. Tolerance precedents in engine tests
- **(a) vLLM `tests/models/utils.py::check_logprobs_close` (src@v0.33.0):**
  - No numeric tolerance.
  - It walks the generated tokens. At the **first** index where token ids differ, it asserts that each side's token is in the other side's top-N logprob dict, then **breaks**; the rest of the sequence is not checked.
  - Prompt logprobs: the top-k key sets must be equal at every position.
  - With `always_check_logprobs=True`, the docstring says every offset is checked. As written, the `break` is inside the same branch, so only index 0 is checked.
  - `check_outputs_equal` requires exact token ids and text.
- **(b) SGLang `test_generation_models.py`** (moved to `test/registered/e2e/models/` at v0.5.20; formerly `test/srt/models/`):
  - `ModelCase` defaults: `prefill_tolerance=5e-2`, `decode_tolerance=6e-2` ("Increased to fix numerical error in issue #8614"), `rouge_l_tolerance=1` (exact string match).
  - Overrides: Qwen2-1.5B decode 7e-2; OLMo-1B-0724-hf decode 8e-2.
  - Compares HF and SRT in fp16 with max_new_tokens=32 (`runners.py::check_close_model_outputs`). It uses the max abs diff over **sorted top-5 logprob values**, not matched by token id. Logprob assertions run only when input_len ≤ 100.
  - `swiss-ai/Apertus-8B` is in ALL_MODELS.
- **(c) Recorded references:**
  - **SGLang `test/manual/test_logprobs.py`:**
    - Builds a local baseline: 1000 ShareGPT prompts, top-20 logprobs, deterministic mode, flashinfer.
    - Re-runs across chunk sizes 1–256 and max_running_requests 8–128.
    - Passes if max and mean |Δ| ≤ **1e-5** and the top-20 sets match.
    - Its stated reason for using a local baseline: logprobs are "highly sensitive to GPU hardware, kernels, torch versions … we cannot maintain a unified logprobs baseline across different machines."
  - **vLLM accuracy gates:** lm-eval requires measured ≥ ground_truth·(1−0.08) (`DEFAULT_RTOL=0.08`); the GSM8K gate requires measured ≥ expected − 0.08. Neither is statistically calibrated.
  - **Calibrated precedents outside the engines:**
    - KVV (item 7).
    - **DiFR**, arXiv:2511.20621 (Karvonen et al., Nov 2025; https://github.com/adamkarvonen/difr). Token-DiFR is a seed-synchronized Gumbel-max margin against a trusted reference. Thresholds come from honest reruns pooled over acceptable configs (H200 TP4, A100, A100 TP4, HF) and are reported at 1% FPR.
    - DiFR results: more than 98% of tokens match exactly when the provider is honest. 4-bit quantization reaches AUC > 0.999 within 300 tokens.

## 7. Serving-corruption incidents
- **Anthropic postmortem, 17 Sep 2025:** https://www.anthropic.com/engineering/a-postmortem-of-three-recent-issues
  - **Routing:** Sonnet 4 requests misrouted to long-context servers; 0.8% at first, peaking at 16% (31 Aug).
  - **TPU misconfiguration** (25 Aug–2 Sep): it "occasionally assigned a high probability to tokens that should rarely be produced" (Thai/Chinese text in English answers, code syntax errors).
  - **Approximate top-k on XLA:TPU:** bf16/fp32 disagreement under `xla_allow_excess_precision` (default true) sometimes dropped the top token, and exposed a latent approximate-top-k miscompile. Fixed with exact top-k and fp32 standardization.
  - **Why evals missed it:** "The evaluations we ran simply didn't capture the degradation users were reporting, in part because Claude often recovers well from isolated mistakes". Also: privacy limits, symptoms varying by platform, and "noisy evaluations". Fixes: more sensitive evals, continuous production evals, an unexpected-character test.
- **Moonshot K2 Vendor Verifier:** https://github.com/MoonshotAI/K2-Vendor-Verifier
  - Setup: about 4,000 tool-call requests per provider vs the official API. Metrics: `tool_call_f1` on `finish_reason=="tool_calls"`, and JSON `schema_accuracy`.
  - **Threshold calibrated from official-API reruns:** min 75.81%, average 76% → accept >73% (K2-thinking). Min 82.71%, average 84% → accept >80% (K2-0905).
  - Results (2025-11-15): Nebius f1 50.60%, Groq 69.52%, Chutes 68.10%. Self-hosted vLLM schema accuracy 87.22% / 76.00%, SGLang 95.52% / 73.13% (thinking / 0905). Causes cited: wrong engine versions, malformed tool-call ids, no guided decoding.
  - The newer KVV (https://www.kimi.com/blog/kimi-vendor-verifier.html) adds sampling-parameter enforcement checks and AIME2025 long outputs to catch "KV cache bugs and quantization degradation".
- **gpt-oss:** Artificial Analysis AIME25 (gpt-oss-120b, high reasoning, 32 runs), via https://simonwillison.net/2025/Aug/15/inconsistent-performance/: 93.3% for six providers, Amazon 83.3%, Azure 80.0% (an old vLLM ignored `reasoning_effort`; fixed to 93.3%). OpenAI's https://developers.openai.com/cookbook/articles/gpt-oss/verifying-implementations blames the harmony format, CoT passing and MXFP4-only MoE weights.

## 8. Weight checksums in the loaders
- **SGLang v0.5.20, `PreshardedModelLoader`** (`python/sglang/srt/model_loader/loader.py`, `--load-format presharded`; separate from `ShardedStateLoader`/`sharded_state`):
  - **First load:** runs the normal load and `process_weights_after_loading`. It then dumps each rank's **post-processed** `state_dict()` plus non-Parameter tensor attributes to safetensors under `<model_path>/presharded/<subdir>/`, with `checksum.json` (plan v1) and `READY`.
  - **Per-tensor hash:** xxHash `xxh3_128` over `str(shape)+str(dtype)` plus the raw bytes of a contiguous CPU copy.
  - **Per-rank `rank_checksums`:** the sum mod 2^64 of the first 8 bytes of xxh3_128(name:hash), so it does not depend on order.
  - **`verify_on_load`** (default False; set with `--model-loader-extra-config '{"verify_on_load": true}'`) re-hashes the tensors as read from the files (on CPU, before the copy to GPU). So it checks **on-disk tensor bytes** against dump-time hashes of post-processed parameters, not the live GPU parameters.
  - **Also in v0.5.20:**
    - `--model-checksum` (`utils/model_file_verifier.py`): SHA-256 of whole on-disk HF files against a manifest or HF LFS metadata.
    - The `/weights_checker` endpoint (snapshot / compare / checksum / reset_tensors). Its checksum is a per-parameter 64-bit `tensor_hash` of **live, dequantized** weights plus a per-GPU SHA-256, with tp/dp/pp ranks.
- **vLLM 0.33.0, `vllm/model_executor/model_loader/weight_checksum.py`** ("Per-parameter weight digests and reset, for verifying weight updates"):
  - `compute_tensor_digests(model)` takes the **SHA-256 of raw bytes of every `named_parameters()` entry**, after `.cpu().contiguous().view(uint8)`. It recurses into `SharedWeightParameter` partitions and tensor-subclass inner tensors (TorchAO qdata/scale). Buffers are excluded.
  - Exposed as `Worker.compute_weight_checksums()` (keys prefixed `dp{dp}:pp{pp}:pcp{pcp}:tp{tp}:`) and `AsyncLLM/EngineCore.compute_weight_checksums()` via collective_rpc.
  - It hashes **live post-load parameters**, not on-disk bytes.

## UNVERIFIED
TML's GPU; vLLM 0.33.0 release-note text (did not render; facts above are from the tag source); SGLang FP8/NVFP4/MXFP4 batch invariance beyond the items cited; vLLM MXFP4 coverage (gpt-oss is listed as tested); vLLM prefix-cache-hit bitwise equality; merge status of FlashInfer #8537/#8565 and vLLM #65931; an HTTP route for vLLM `compute_weight_checksums`; SGLang blog latency units; any dedicated FP8/FP4 reproducibility study (none found).
