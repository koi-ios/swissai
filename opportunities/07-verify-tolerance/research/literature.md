# Prior work: calibrated logprob verification of served weights, and checkpoint fault injection

Compiled 11–12 Nov 2026. Sources were checked against arXiv abstract/HTML pages (or the arXiv API), official blogs and docs, and raw GitHub files. Anything not checked against a primary source is marked **UNVERIFIED**. "Ours" means the planned setting: OpenAI completions API with `echo` + `logprobs` (teacher-forced per-token logprobs plus top-k); a trusted reference run; benign nondeterminism; a checkpoint fault injector.

## 1. TOPLOC and verifiable inference

**TOPLOC.** Ong et al., Prime Intellect. arXiv [2501.16007](https://arxiv.org/abs/2501.16007), **ICML 2025** ([PMLR 267](https://proceedings.mlr.press/v267/ong25a.html)).
- **Mechanism.** The provider records the top-k values of the **last hidden state** (k=128) "every 32 new tokens", plus the prompt. The indices and values are encoded as an interpolated polynomial. A validator recomputes the hidden states by prefill and compares them.
- **Proof size.** "258 bytes of storage per 32 new tokens" (abstract; 256 B in §5.8), about 8 B/token. That is "1000×" less than storing the activations.
- **Acceptance rule.** Exponent-mismatch count, then the mean and median mantissa difference where the exponents match. Thresholds:
  - bf16: **T_exp=38, T_mean=10, T_median=8**;
  - fp32: 8/256/128.
- **How the thresholds were set.** They were "chosen based on our analysis of the error statistics in Table 2 and Table 5": hand-picked, with no stated false-positive rate. §6 leaves "threshold tuning to future work".
- **Benign variation tested** (Table 2; vLLM; 400 prompts × 512 tokens).
  - Hardware: 1×A100, 1×4090 and 2×4090 (TP=2).
  - Attention: FlashAttention-2, SDPA and FlexAttention.
  - Worst-case exponent mismatch: 28/128.
  - **Batch size and CUDA version were not varied.**
- **Changes detected.**
  - A different model gives at least 125/128 mismatches; altered prompts give at least 67–95.
  - A bf16 generation checked by an fp32 validator gives a mantissa mean of at least 27,892.
  - Claimed "no false positives or negatives in our empirical evaluations".
- **Limitations stated by the authors.** Speculative decoding cannot be detected. The fp8 vs bf16 margin is "small".
- **vs ours.** It needs hidden states and has hand-set thresholds.

**Uses and follow-ups.**
- **INTELLECT-2** ([2505.07291](https://arxiv.org/abs/2505.07291)). TOPLOC validators re-prefill the rollouts. Extra checks: EOS probability must be > 0.1, and the sampled-logit distribution should be "resembling an exponential with a mode at 1". Proof generation costs "∼1%" of throughput. No false-positive or false-negative rates are reported.
- **TOPLOC v2** ([SYNTHETIC-2 blog](https://primeintellect.ai/blog/synthetic-2-release), 10 Jul 2025).
  - Adds group-level accept/reject, stage replay, and "Reproducible Gumbel sampling proofs" (paper "coming soon"; **not found**).
  - "False positive rate of TOPLOC was just 0.000925% (37 slashes)" over 4M samples. Verification was "25x cheaper" than re-inference.

**Other schemes** (all need white-box internals, seeds or determinism):
- **DiFR.** Karvonen et al., [2511.20621](https://arxiv.org/abs/2511.20621), Nov 2025. This is the closest nondeterminism-aware verifier.
  - **Token-DiFR** shares the sampling seed. The verifier regenerates the Gumbel noise and scores each claimed token by its clipped post-Gumbel logit gap.
  - **Activation-DiFR** sends random projections of activations.
  - Honest configurations: H200 TP4, A100 TP1, A100 TP4, and HF. Thresholds are empirical; results are reported as AUC.
  - Results:
    - 4-bit quantization: "AUC > 0.999 within 300 output tokens" (Token-DiFR) and "using just 2 output tokens" (Activation-DiFR);
    - FP8 KV cache: about 1,000–5,000 tokens;
    - sampling bugs and wrong seed/temperature are also detected.
  - The methods "require access to model weights". Top-k-logprob-only verification is not evaluated.
- **LOGIC** ([inference.net blog](https://inference.net/blog/logic/), 5 Nov 2025; paper "Coming Soon").
  - The provider returns top-k logprobs per generated token (k=20 in the payload description; k=10 in the experiments). The verifier recomputes 10 random decode positions and runs a two-sample **KS test**, accepting if p_KS ≥ 0.95.
  - Honest H100 and 4090 runs: p > 0.999. A 1B substitute: p≈0.006. FP8: p≈0.150.
  - This is the closest top-k-logprob decision rule, but the cutoff is ad hoc and the evaluation is small.
- **TAO** ([2510.16028](https://arxiv.org/abs/2510.16028), **EuroSys 2026**). Per-operator IEEE-754 bounds plus "empirical percentile profiles calibrated across hardware", 10²–10³× tighter than the bounds. White-box and operator-level, but the nearest instance of calibrating tolerances across hardware.
- **Rinberg et al.** ([2511.02620](https://arxiv.org/abs/2511.02620)): re-runs logged (input, output, seed) and models "valid sources of non-determinism". FPR "<0.01%".
- **Brief.**
  - SVIP ([2410.22307](https://arxiv.org/abs/2410.22307)): hidden-state proxy task; FNR <5%, FPR <3%.
  - Proof of Sampling ([2405.00295](https://arxiv.org/abs/2405.00295)): game-theoretic.
  - VeriLLM ([2509.24257](https://arxiv.org/abs/2509.24257)): rerun at about 1% of cost.
  - EigenAI ([2602.00182](https://arxiv.org/abs/2602.00182)): bit-exact, so a byte-equality check.
  - Cankaya ([2606.00279](https://arxiv.org/abs/2606.00279)): bit-exact recomputation across NVIDIA GPU variants.
  - IMMACULATE ([2602.22700](https://arxiv.org/abs/2602.22700)): verifiable computation on sampled requests.

**Benign nondeterminism background.**
- **Thinking Machines blog** ([He, Sep 2025](https://thinkingmachines.ai/blog/defeating-nondeterminism-in-llm-inference/)): 1000 temperature-0 Qwen3-235B completions gave **80 unique** outputs. The cause is lack of **batch invariance** (split-K, split-KV, chunked prefill), not atomics. Batch-invariant vLLM took 26 s → 42–55 s.
- **Yuan et al.** ([2506.09501](https://arxiv.org/abs/2506.09501)): "up to 9% variation in accuracy and 9,000 tokens difference in response length" (bf16 greedy; GPU count, type and batch size).
- **Deterministic modes.** SGLang `--enable-deterministic-inference` ([blog](https://lmsys.org/blog/2025-09-22-sglang-deterministic/), 34.35% overhead) and vLLM `VLLM_BATCH_INVARIANT=1` ([docs](https://docs.vllm.ai/en/latest/features/batch_invariance/)).

## 2. Model Equality Testing

Gao, Liang, Guestrin. [2410.20247](https://arxiv.org/abs/2410.20247), **ICLR 2025**.
- **Test.** MMD on **sampled completions**, with Hamming-type kernel k=1{x=x′}·Σᵢ1{yᵢ=y′ᵢ}.
  - m=25 prompts, L=50 tokens, "an average of just 10 samples per prompt".
  - The p-value comes from a **null simulated by resampling the reference 1,000 times** (a Monte Carlo p-value). α=0.05 or 0.01.
  - Precision is handled with a composite null (fp32 *or* fp16 reference).
- **Power.** "Median of 77.4% power against a range of distortions".
  - int8: 0.08 (Llama-3.1-8B) to 1.00 (70B). nf4: 0.93–1.00.
  - Watermark: 0.54–1.00.
  - Fine-tuning: above 50% power after one epoch (42 steps).
  - Within-Llama swaps: about 0.75.
  - Empirical FPR: 0.04–0.07 at α=0.05.
- **Audit (Summer 2024).** "11 out of 31 endpoints serve different distributions than reference weights released by Meta" (9 providers). Some were further from the reference than a different model.
- **vs ours.** No logprobs; the only benign variation modelled is fp32/fp16.

## 3. Cai et al., and closer black-box audits

**Cai, Shi, Zhao, Song.** [2504.04715](https://arxiv.org/abs/2504.04715), v2 Sep 2025, no venue listed.
- **Logprob methods.** "Methods using log probabilities are defeated by inherent inference nondeterminism in production environments."
  - Greedy logprob traces diverge across vLLM vs HF, H100 vs A100, and software versions (figures only).
  - "Even at T=0, greedy decoding can diverge after only tens of tokens."
  - Gemma-2-9B agreement across providers was below 5%.
  - Logprob checks are "vulnerable to false alarms unless requests are isolated".
- **Other methods.**
  - Text classifiers are at chance (49–52%).
  - MMD power collapses to α for substitution rate ≤ 0.2.
  - Benchmarks cannot separate FP8 (78.05 vs 77.88 MMLU).
- **Proposed fix.** TEEs, with 2.9–15.4% throughput overhead.
- **vs ours.** This is exactly the failure that calibration targets.

**Log Probability Tracking.** Chauvin et al. [2512.03816](https://arxiv.org/abs/2512.03816), **ICLR 2026**. The closest black-box logprob work.
- **Probe.** Prompt "x", one output token, top-k logprobs (20, or 5–8 for some providers), N=10 samples per side.
- **Statistic.** Mean absolute difference of per-token mean logprobs, with a **permutation** p-value.
- **TinyChange benchmark.** 290 variants: fine-tuning for 1–512 steps, LoRA, pruning, Gaussian noise.
  - AUC: 0.915, vs 0.878 (MMLU) and 0.670 (MET).
  - Cost: $0.14 per year hourly.
- **In the wild.** 23% of OpenRouter endpoints return logprobs. 37 suspected changes across 29 endpoints.
- **Limitation.** It "does not distinguish between infrastructure changes and model updates". Its null is the same deployment resampled.

**RUT.** Zhu et al. [2506.06975](https://arxiv.org/abs/2506.06975), **ICLR 2026**.
- One API completion per prompt is ranked among 100 local samples by log-rank score; a Cramér–von Mises test checks the ranks for uniformity. 100 prompts.
- "None of the methods succeed in reliably detecting substitution for the 8-bit variants of Gemma and Mistral."
- The same model on an A100 tested against an A6000 reference was rejected at rates 0.058–0.094 (RUT) and 0.084–0.142 (MMD) at α=0.05. Hardware change therefore inflates the FPR (our reading of Table 4).

## 4. Accuracy is Not All You Need

Dutta et al. [2407.09141](https://arxiv.org/abs/2407.09141), **NeurIPS 2024**.
- **Flips.** "The % of answers that changed from correct → incorrect and incorrect → correct".
- **KL divergence.** Mean per-token full-vocabulary KL over answer-option tokens.
- **Key numbers.**
  - Spearman correlation between flips and KLD: **0.981** (MMLU).
  - Accuracy usually changes by less than 1% while flips reach **13.6%** (Llama2-7B, GPTQ W4A16).
  - GPTQ W8A16 is the exception (0.26% flips).
  - MT-Bench drops about 5% (Llama2-70B).
- **Recommendation.** Report distance metrics (KLD, flips).
- **vs ours.** It motivates a distance statistic, but gives no decision rule.

## 5. llama.cpp KL-divergence mode, and engine CI checks

**llama.cpp.** [README](https://raw.githubusercontent.com/ggml-org/llama.cpp/master/tools/perplexity/README.md), [perplexity.cpp](https://raw.githubusercontent.com/ggml-org/llama.cpp/master/tools/perplexity/perplexity.cpp).
- **Two-pass workflow.**
  1. Run `--kl-divergence-base FILE`. It stores full-vocabulary log-probs as uint16 with per-token scale and offset, about 37 GiB for LLaMA-3 on WikiText-2.
  2. Run `--kl-divergence-base FILE --kl-divergence` on the candidate.
- **Printed statistics** (source):
  - Mean PPL(Q), Mean PPL(base), Cor(ln PPL(Q), ln PPL(base)), Mean ln(PPL(Q)/PPL(base)), Mean PPL(Q)/PPL(base), Mean PPL(Q)−PPL(base);
  - **Mean KLD ± unc.**, Maximum, 99.9/99/95/90%, Median, 10/5/1/0.1%, Minimum KLD. KLD is KL(base‖Q) per token;
  - **Mean Δp ± unc.**, Maximum, 99.9/99/95/90/75%, Median, 25/10/5/1/0.1%, Minimum Δp, **RMS Δp**, and **Same top p** (argmax agreement ± s.e.). Δp = p_Q − p_base of the actual next token.
- **Example.** LLaMA-3-8B KLD: q8_0 0.001355, q4_K_M 0.031273. BF16 vs FP16: Mean KLD 0.00002515, RMS Δp 0.150%, Same top p 99.739%.
- **Gap.** White-box and descriptive; there is no accept/reject rule.

**Engine CI checks** (fixed heuristics, the baseline):
- vLLM `check_logprobs_close` ([utils.py](https://raw.githubusercontent.com/vllm-project/vllm/main/tests/models/utils.py)): at the first greedy divergence, "each predicted token must be in top N logprobs of the other".
- SGLang `check_close_model_outputs` ([runners.py](https://raw.githubusercontent.com/sgl-project/sglang/main/python/sglang/test/runners.py)): `max|HF−SRT|` logprob < fixed prefill/decode tolerance, plus a ROUGE-L threshold.
- servekit `verify`: 1e-6.

## 6. Quantization formats and fault models

- **DeepSeek-V3 FP8** ([2412.19437](https://arxiv.org/abs/2412.19437); [README_WEIGHTS.md](https://raw.githubusercontent.com/deepseek-ai/DeepSeek-V3/main/README_WEIGHTS.md)).
  - Activations use "1x128 tile basis (i.e., per token per 128 channels)"; weights use a "128x128 block basis". E4M3.
  - `weight_block_size: [128,128]`. `weight_scale_inv` is float32 with one value per block; dequantization is "(128x128 weight block) * weight_scale_inv". Edge blocks are zero-padded.
  - *Implication:* one bad scale coherently rescales 16,384 weights.
- **OCP MX v1.0** ([spec](https://www.opencompute.org/documents/ocp-microscaling-formats-mx-v1-0-spec-final-pdf), Sep 2023).
  - **MXFP4 = E2M1 elements, k=32, E8M0 scale.** The value is vᵢ = X·Pᵢ.
  - E8M0: bias 127, no Inf, no zero, NaN=0xFF. "When the scale is NaN, all values in the MX block are NaNs."
  - E2M1 max is ±6.
- **NVFP4** ([NVIDIA blog](https://developer.nvidia.com/blog/introducing-nvfp4-for-efficient-and-accurate-low-precision-inference/), Jun 2025).
  - E2M1 with an "E4M3 scaling factor to each 16-value micro-block", plus an FP32 per-tensor scale (4.5 bits/value).
  - DeepSeek-R1 FP8→NVFP4: "1% or less" accuracy loss.
- **gpt-oss MXFP4.**
  - [index.json](https://huggingface.co/openai/gpt-oss-20b/raw/main/model.safetensors.index.json): only MoE experts are quantized, as `model.layers.N.mlp.experts.{gate_up_proj,down_proj}_{blocks,scales}`. Attention, router and `sinks` are unquantized.
  - The [model card](https://arxiv.org/abs/2508.10925) gives "4.25 bits per parameter".
  - Transformers [mxfp4.py](https://raw.githubusercontent.com/huggingface/transformers/main/src/transformers/integrations/mxfp4.py):
    - `_blocks` is uint8 with shape (E, rows, K/32, 16), low nibble first;
    - dequantization is `ldexp(lut[nibble], scale−127)`.
  - *Implication:* flipping bit b of a scale multiplies 32 weights by 2^(±2^b), and 0xFF makes the block NaN.
- **Egiazarian et al.** [2509.23202](https://arxiv.org/abs/2509.23202).
  - W4A4 average recovery on Llama-3.1-8B-Instruct:

    | Format | RTN | GPTQ | MR-GPTQ |
    |---|---|---|---|
    | MXFP4 | 87.83% | 89.47% | 93.31% |
    | NVFP4 | 94.67% | 95.92% | 96.08% |

  - Power-of-two scales "severely degrade accuracy". "FP4 is not an automatic upgrade over INT4."
- **FP8 across accelerators.** [2502.01070](https://arxiv.org/abs/2502.01070), v4 retitled "An Inquiry into Datacenter TCO for LLM Inference with FP8".
  - "Hopper GPUs use a 14-bit accumulator for FP8 GEMMs"; "Gaudi HPUs always accumulate in FP32".
  - Gaudi 2 E4M3 max is 240 vs 448.
  - No cross-device accuracy comparison.
  - *Implication:* calibration must cover the target hardware.

## 7. Weight corruption, SDC and fault injection

**Bit flips.**
- **BFA.** Rakin et al., [ICCV 2019](https://openaccess.thecvf.com/content_ICCV_2019/html/Rakin_Bit-Flip_Attack_Crushing_Neural_Network_With_Progressive_Bit_Search_ICCV_2019_paper.html).
  - It targets **8-bit quantized** weights, not exponents.
  - "13 bit-flips out of 93 million bits" took ResNet-18 from 69.8% to 0.1%. "Randomly flipping 100 bits merely degrades the accuracy by less than 1%."
- **Exponent bits: Hong et al.**, USENIX Security 2019 ([1906.01017](https://arxiv.org/abs/1906.01017)).
  - float32 weights: "40-50% of the parameters" can cause a >10% drop with a single flip.
  - The damage comes from **0→1 exponent flips, "especially the 31st-bit"** (the exponent MSB). Mantissa, sign and 1→0 flips are mostly harmless.
- **LLMs.**
  - PrisonBreak ([2412.07192](https://arxiv.org/abs/2412.07192)): "5 to 25 bit-flips" jailbreak 10 LLMs, demonstrated on a GDDR6 GPU.
  - GenBFA ([2411.13757](https://arxiv.org/abs/2411.13757)): "as few as three bit-flips" on W8 LLaMA3-8B; MMLU 67.3% → 0%.
  - SBFA ([2509.21843](https://arxiv.org/abs/2509.21843)): **one** in-distribution flip (BF16 or INT8) gives below-random MMLU.
  - GPUHammer ([2507.08166](https://arxiv.org/abs/2507.08166), USENIX Security 2025): Rowhammer on an A6000, accuracy drops "up to 80%".
  - Shared KV-cache flips ([2604.17249](https://arxiv.org/abs/2604.17249)): "13 of 16 BF16 bit positions produce coherent but altered outputs".

**SDC at fleet scale.**
- Dixit et al., Meta ([2102.11245](https://arxiv.org/abs/2102.11245)): "SDCs are a systemic issue across generations".
- Hochschild et al. ([HotOS 2021](https://dl.acm.org/doi/10.1145/3458336.3465297)): "on the order of a few mercurial cores per several thousand machines".

**SDC in LLM work, 2024–26.**
- Llama 3 ([2407.21783](https://arxiv.org/abs/2407.21783)): 6 of 419 unexpected interruptions (1.4%) were SDC, over 54 days.
- Gemini ([2312.11805](https://arxiv.org/abs/2312.11805)): "SDC events to impact training every week or two". Found by deterministic replay.
- Ma et al. ([2502.12340](https://arxiv.org/abs/2502.12340)): usually small, but can reach "different optima" and cause loss spikes.
- Altenbernd et al. ([2604.00726](https://arxiv.org/abs/2604.00726), CCGrid 2026) and LLM-PRISM ([2604.10390](https://arxiv.org/abs/2604.10390), 7,664 runs): training only.
- Inference-side SDC studies are rare. Agarwal et al. (ISSRE 2023) is **UNVERIFIED**.

**Fault injection for loaders and weight verification.** Confirmed rare; **we found no prior work** that injects checkpoint- or loader-level faults to measure a verifier's detection power. The nearest:
- **LM-Fix** ([2511.02866](https://arxiv.org/abs/2511.02866), ICCD 2025). Random in-memory weight bit flips are detected by an exact comparison of a test-vector output with a stored reference. Detection is 91.1–99.8% at TVL=200. It is white-box and assumes determinism.
- **DeepMutation** ([1805.05206](https://arxiv.org/abs/1805.05206), ISSRE 2018). Weight mutation operators (GF, WS, NEB, NAI, NS, LD, LAm, AFRm) measure test adequacy.
- **GoldenTransformer** ([2509.10790](https://arxiv.org/abs/2509.10790)): a robustness fault injector.
- **TinyChange** (§3).
- **Motivation evidence.**
  - [2506.09713](https://arxiv.org/abs/2506.09713) (TOSEM): model loaders are vLLM's largest bug category (**21%**); "over 35% of bugs show other symptoms" than crashes.
  - [Anthropic postmortem](https://www.anthropic.com/engineering/a-postmortem-of-three-recent-issues) (Sep 2025): TPU misconfiguration and a top-k miscompile; "the evaluations we ran simply didn't capture the degradation".
  - [K2-Vendor-Verifier](https://raw.githubusercontent.com/MoonshotAI/K2-Vendor-Verifier/main/README.md): 4,000 requests, with a `tool_call_f1` threshold set from the official API's run-to-run fluctuation, an informal calibration.
- **Model signing** (OpenSSF, 2025; search snippets only): it hashes files and cannot see loader misinterpretation.

## 8. Model fingerprinting (one line each)

- **HuRef** ([2312.04828](https://arxiv.org/abs/2312.04828), NeurIPS 2024): weight-invariant terms → StyleGAN2 image plus ZKP. Needs parameter access by the owner.
- **Instructional Fingerprinting** ([2401.12255](https://arxiv.org/abs/2401.12255), NAACL 2024): a backdoor key implanted by instruction tuning; 11 LLMs.
- **Idiosyncrasies** ([2502.12150](https://arxiv.org/abs/2502.12150), ICML 2025): "97.1% accuracy" on 5-way attribution of model outputs.
- **LLMmap** ([2407.15847](https://arxiv.org/abs/2407.15847), USENIX Security 2025): "8 interactions" identify 42 versions with >95% accuracy.
- **Finlayson et al.** ([2403.09539](https://arxiv.org/abs/2403.09539), COLM 2024): softmax-bottleneck signatures from logprobs audit model updates.
- **2025–26.**
  - Log Probability Tracking and RUT (§3).
  - "One Token Is Enough" ([2607.10252](https://arxiv.org/abs/2607.10252)): sampled single-token distributions; 7.3% EER over 165 models.
  - FLIPS ([2606.03330](https://arxiv.org/abs/2606.03330), ICML 2026): instance-configuration fingerprints, 96% closed-set.
- Fingerprinting identifies a model; ours must resolve far smaller, loader-induced differences.

## 9. Statistical methodology

- **Monte Carlo / permutation p-values.** p = (1+#{T_k ≥ T})/(K+1) is exactly valid under exchangeability.
  - Phipson & Smyth, SAGMB 2010 ([1603.05766](https://arxiv.org/abs/1603.05766)): omitting the +1 understates p "by about 1/m".
  - Classical sources: Barnard 1963; Besag & Clifford 1989. **UNVERIFIED**, not fetched.
  - MET (§2) uses a simulated-null Monte Carlo p-value, and Chauvin et al. (§3) a permutation p-value.
- **Conformal p-values.** Bates, Candès, Lei, Romano, Sesia, Ann. Statist. 51(1):149–178, 2023 ([2104.08279](https://arxiv.org/abs/2104.08279)).
  - The p-values are "marginally valid but mutually dependent". BH is valid via positive dependence.
  - Calibration-conditional variants give guarantees conditional on the calibration set.
  - *Implications:*
    - the minimum attainable p is 1/(K+1), so α=0.05 needs K ≥ 19 honest runs and α=0.01 needs K ≥ 99;
    - an unseen GPU or TP layout breaks exchangeability;
    - with a small K, the realized FPR given one calibration set varies.
- **Calibration in ML-systems testing.**
  - CRADLE ([ICSE 2019](https://www.cs.purdue.edu/homes/lintan/publications/cradle-icse19.pdf)): cross-backend thresholds (T_C=8, T_M=0.2) tuned on known cases.
  - FLASH (ISSTA 2020; search-level only): convergence tests for flaky ML tests.
  - TAO and DiFR (§1): empirical honest-config thresholds.
  - **No prior use of conformal or rank p-values for numerical-reproducibility or regression testing of ML systems was found.**

## Closest prior work, and what is new

**Closest.**
- Log Probability Tracking (black-box logprobs, permutation p-values, but a temporal null that cannot separate infrastructure from weight changes).
- DiFR (honest-config calibration across TP and GPU, but needs seeds plus white-box logits or activations, reports AUC, and tests quantization and sampling faults).
- LOGIC (top-k KS test, ad hoc cutoff).
- TOPLOC (hidden states, hand-set thresholds).

**New.** A verifier that:
1. **Needs only an OpenAI completions endpoint with `echo`+`logprobs`.** It scores fixed sequences by prefill, with no sampling, seeds, hidden states or hooks.
2. **Turns a logprob/top-k distance into a finite-sample-valid rank p-value** against K honest runs. The runs deliberately span batch size and concurrency, chunked prefill, CUDA graphs, TP degree and GPU type, replacing fixed tolerances (servekit 1e-6, TOPLOC, CRADLE, vLLM/SGLang CI, LOGIC).
3. **Is evaluated for power vs. token budget against loader-realistic faults:** shard zeroing and duplication, exponent flips, missing tensors, corrupted FP8 `weight_scale_inv` and MXFP4 E8M0 `_scales`, MoE expert swaps, and TP permutations or transposes.

Prior work tests quantization, fine-tuning, pruning, noise, substitution, sampling bugs or adversarial flips, never loader corruption.

## Could not verify

- The TOPLOC v2 Gumbel-sampling paper.
- The LOGIC paper.
- Barnard / Besag–Clifford.
- Agarwal et al. (ISSRE 2023).
- OpenSSF model-signing internals.
- FLASH details.
- The affiliations for Ma et al.
- DiFR's public-API appendix.
- The RUT Table 4 α-level reading.
- The FP8-accelerator paper's venue.
