# Draft PR for eth-easl/servekit (not opened; for the user to send)

**Title:** verify: calibrate the noise floor (exact where the engine reproduces, statistical where it cannot) + a checkpoint fault injector

**Branch:** `verify-statistical`, three commits on top of `main` @ cd8f0b8 (the fault injector, the calibrated verify, then a fix to the `max_z` threshold that the study forced). Patches are in `../patches/`; they apply with `git am`.

---

## Why

`servekit verify` guards `sharded_state` against silent corruption, which it has caught in practice:
- #14 / sgl-project/sglang#34448: 69% of gpt-oss-20b's weights loaded as zeros, 16/16 prompts failed;
- #19 / sglang#35702: MLA's canonical `kv_b_proj` missing from every shard.

Its rule is |Δ logprob| ≤ 1e-6 per token plus exact greedy continuations. That rule is equality, and equality only holds when the candidate runs the same kernels on the same shapes as the reference. When it does not, a correct load fails:
- **Cross-hardware.** A fixture recorded on other hardware drifts by about 1e-1 and fails 16/16 on a correct load (commit message of #22). `llama70b.sbatch` still fails this way.
- **TP degree.** The repo's own Llama-3.1-70B fixtures at TP4 and TP8 differ by up to **0.155**: median 4e-5, p99 0.11 over 1,140 tokens.
- **Greedy near-ties.** Continuations flipped warm vs cold at three exact top-2 ties, with every logprob within 1e-6 (#22).
- **FP4.** Models such as Kimi-K3 fail bitwise (Boughizane's write-up, §VI).

Raising the tolerance by hand is unsafe: any fixed number either alarms on benign drift or hides real faults. On the CPU study below, 1e-6, 0.05 and 0.1 each alarm on **100%** of benign runs.

## What

**1. `--record --calibrate N`** (32 recommended) measures the trusted server's own noise floor and stores it in the reference:
- one repeat of the reference schedule, which answers whether it reproduces bitwise;
- N captures under perturbed load: concurrency 1–16, background filler requests and a shuffled order. These move what a serving engine moves: split-K and split-KV choices, chunked-prefill boundaries and CUDA-graph buckets.

**2. `--reference --mode auto|exact|statistical|legacy`**:
- **`exact`**: every logprob and greedy token must be bitwise equal. On a mismatch it also reports whether the drift is inside the calibrated noise, which separates a likely process-level difference from a corrupted load. It never turns that into a pass.
- **`statistical`** (`verify_stats.py`, stdlib only):
  - **Normalization.** Per-token drift is divided by s·max(1−p, 1e-4). A token's logprob moves by (1−p_y)·dz_y − Σ_{j≠y} p_j·dz_j, so its noise scales with 1−p. On the TP4-vs-TP8 fixtures, |Δ|/(1−p) has the same spread over six decades of 1−p. s is the 90th percentile of the calibration's normalized drift.
  - **Continuous statistics:**
    - `max_z`: a few tokens hit hard, as by one expert, one vocab shard or an exponent flip.
    - `rms_z`: drift spread over many tokens.
    - `shift`: one-sided; the candidate finds the reference text less likely.
  - **Thresholds,** with the false-alarm budget split α/3 per statistic (default α = 0.01):
    - `rms_z` and `shift`: leave-one-out prediction bounds over the calibration runs (t-distribution; log scale for `rms_z`). Each token's |z| is capped at the calibration's 99.9th percentile, so one fragile token cannot fire them.
    - `max_z`: a generalized Pareto tail fitted to all K×N calibration tokens' |z|, extrapolated to per-token probability α/(3N). This is a union bound over tokens.
    - Every threshold is at least the largest null sample.
  - **Rules:**
    - a top-1 flip where the reference's winner led by more than `margin_tau` (2 × 1.5 × the largest drift seen in calibration);
    - a greedy divergence at such a step. Near-ties explain a divergence and are reported; they do not fail the check.
- **`auto`** (default):
  - calibrated reference: `exact` if the repeat reproduced bitwise, `statistical` if not;
  - uncalibrated reference: `legacy`, i.e. today's `--token-tol` / `--nll-tol` with identical semantics.
- **`--calibration-from FILE…`**: add benign captures recorded on another node, restart or TP degree, when the check must accept those differences too.

**3. `--weights`** records and compares SGLang ≥ 0.5.20's `POST /weights_checker {"action": "checksum"}`. These are per-tensor hashes of the live, dequantized parameters and buffers. They are exact and noise-free, so the check uses them when present. They cover only what `named_parameters()` / `named_buffers()` expose, which is the same view `sharded_state` dumps. If the MXFP4 triton path still moves expert weights off the module, the checksum is blind to them; that is a source-level prediction the Modal study tests.

**4. `python -m servekit.faults`** is a checkpoint fault injector for negative-control tests. Each fault takes a magnitude knob and a seed, and records ground truth in `servekit-fault.json`. The faults:
- zeroed, duplicated, permuted or transposed TP shards;
- exponent and mantissa bit flips in any dtype;
- missing tensors;
- FP8 `weight_scale_inv` rolled, set to 1, jittered, transposed or taken from the other TP shard;
- MXFP4 E8M0 scales shifted, rolled or zeroed;
- swapped MoE experts (per-expert or fused gpt-oss tensors);
- noise and scaling.

It works on raw bytes through numpy, so no torch is needed, and it rewrites only the shards it touches (the rest are hard-linked). `inject`, `list`, `diff` and a shape-only `validate` are included. `numpy` is an optional extra: `pip install servekit[faults]`.

**Capture format 2.** It keeps full-precision logprobs instead of 6-decimal rounding, so exact mode really is bitwise. It adds token strings, top-2 alternatives per position and per greedy step, and the schedule. Format-1 references keep working: they are compared in `legacy` mode, or in `exact` mode at their rounding.

## Evidence

**CPU study.** Qwen3-0.6B in BF16, its official FP8 release and an MXFP4-MLP variant, on a transformers-based engine with BF16 rounding at kernel boundaries. Its knobs are split-K, split-KV, TP all-reduce, chunked prefill and prefix hits. With rounding off it is bitwise equal to transformers' fp32 forward. Full tables are in `results/cpu/summary.md`:
- **False alarms** on 180 held-out benign runs (perturbed load plus TP 2 and 4): **3/180** at `--calibrate 16` and **2/132** at 32, against a nominal 1%. Every fixed tolerance (1e-6, 0.05, 0.1) alarms on **180/180**.
- **Detection,** of 147 faults injected into the real safetensors files:
  - **63/63** of those whose KL exceeds 3× the benign noise, including every structural fault (TP-shard zeroing ≥ 1/32 of a tensor, vocab shard, duplication, permutation, transposition, exponent-MSB flips, a missing tensor, MXFP4 zeroed blocks and rolled scales).
  - Faults below the noise floor are not caught. No check that tolerates serving noise can catch them, and they move the model less than serving it under load does.
  - Exact mode caught **147/147** on an engine that reproduces bitwise. That is why `auto` keeps exact wherever the engine reproduces.
- **FP8 is the cautionary case.** Dynamic FP8 activation quantization makes the benign noise 17× BF16's in KL. A statistical tolerance then hides FP8 scale faults: "5% of blocks ×2" went 0/3. FP8 should be verified exactly, which this engine allows.
- **The first `max_z` rule was miscalibrated, and the study caught it.** It raised 4/60 MXFP4 false alarms: one fragile token moved 10σ under a setting calibration never sampled. That is what the third commit fixes. See its message.

**End to end.** `cpu/e2e_cli.sh` drives this CLI against an OpenAI-compatible server over the same engine, on real corrupted checkpoints:
- **Fresh clean server.** It passes (`auto` picks exact).
- **Clean weights, other kernel choices.** It fails exact mode, with *"the drift is within the calibrated noise … weight checksums: equal"*, and passes `--mode statistical`.
- **1/8 of one projection zeroed.** It fails every mode (`max_z` 11,660 against a threshold of 8.6).
- **16 mantissa flips.** They pass `statistical` alone but fail with `--weights`, which names the tensor. Checksums and logprobs are complementary: checksums are exact for any parameter change, and logprobs cover what the module tree hides.

**GPU.** `modal_app.py` runs the same study on SGLang 0.5.20 and vLLM 0.33.0 on an H100 (Qwen3-1.7B BF16 and FP8, gpt-oss-20b MXFP4; estimated ~$14.6, budget-capped at $24.6). It has not run yet.

## Tests

`PYTHONPATH=src pytest tests -q`: **183 passed**. 47 of them are new: 22 for the fault injector on real safetensors files and 25 for the statistics (including GPD tail recovery and the fragile-token case), the modes and the CLI against the stub server. The 2 pre-existing `test_launch.py` stager errors are macOS-only (no `O_DIRECT`).

## Not in this PR

- The e2e scripts are unchanged. Suggested follow-up: `--calibrate 32 --weights` on the `MODE=baseline` record step, keeping `--mode auto`.
- A two-sample mode (several candidate captures against the calibration) would raise power for faults smaller than the noise floor.

🤖 Generated with [Claude Code](https://claude.com/claude-code)
