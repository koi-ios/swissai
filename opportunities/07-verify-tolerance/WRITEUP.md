# Calibrated `servekit verify`: exact where engines reproduce, statistical where they cannot

*Opportunity 07. Answers §VI of Youssef Boughizane's cold-start write-up: study the nondeterminism, advise whether to relax the 1e-6 check, and do not relax it lightly.*

## Problem

`servekit verify` makes presharded loading safe.
- It caught servekit #14 / sglang#34448, where **69% of gpt-oss-20b's weights loaded as zeros**, and #35702 (MLA).
- Its rule, |Δ logprob| ≤ 1e-6 plus exact greedy text, is bitwise equality. It holds only when the candidate runs the reference's kernels on the reference's shapes.

Otherwise correct loads fail:
- servekit's own Llama-3.1-70B fixtures differ between **TP4 and TP8 by up to 0.155** (median 4×10⁻⁵; `results/fixtures_tp4_vs_tp8.txt`);
- other hardware drifts by ~10⁻¹ and fails 16/16 (#22);
- greedy text flips at top-2 ties (#22);
- FP4 Kimi-K3 fails bitwise (§VI).

Hand-picked tolerances do not help. In our study, 1e-6, 0.05 (SGLang CI) and 0.1 each alarm on **180/180** benign runs.

## Related work, and what is new

The full survey is in `research/`.
- **Batch-invariant modes** give exact equality only within one configuration and one TP degree:
  - SGLang `--enable-deterministic-inference`: in 0.5.20, ~34% average slowdown;
  - vLLM `VLLM_BATCH_INVARIANT=1`: +58–63% decode cost;
  - TP-size invariance (TBIK): 22–63% overhead.
- **Weight checksums** are exact but see only what the module tree exposes: SGLang 0.5.20 `/weights_checker`, vLLM 0.33 `compute_weight_checksums`.
- **Verifiers and audits:**
  - TOPLOC needs hidden states and uses hand-set thresholds.
  - DiFR (arXiv 2511.20621) is the closest work: thresholds from honest reruns at 1% false-positive rate, but it needs seeds and logits.
  - Log Probability Tracking (ICLR 2026) cannot separate infrastructure from model changes.
  - Model Equality Testing (ICLR 2025) works from samples.
- **Engine CI** uses hand-set logprob tolerances.
- **Mutation testing of weights** exists (DeepMutation), but not for loaders.

**New here:**
1. **A per-token noise model**, Δ / (s·(1−p)). It follows from d log p_y = (1−p_y)dz_y − Σp_j dz_j. On the lab's own fixtures, |Δ|/(1−p) has the same median (0.013–0.016) across six decades of 1−p.
2. **Calibration through the same OpenAI endpoint verify already uses:** perturbed load on the trusted server, with no seeds or hooks.
   - Thresholds hold at a stated α. The bulk statistics use leave-one-out prediction bounds. The extremes use a generalized Pareto tail fitted to all calibration tokens, with a union bound over tokens.
   - The greedy and top-1 checks are margin-aware.
3. **A policy that never relaxes silently.** `auto` stays bitwise wherever calibration shows the reference schedule reproduces.
4. **A loader-realistic fault injector** used as a negative control to measure power against effect size. We found no prior work doing this.

## Built

- **servekit** (branch `verify-statistical`; 3 patches in `patches/`, applying onto `main`):
  - `verify_stats.py` (stdlib only);
  - `verify` flags: `--calibrate N`, `--mode auto|exact|statistical|legacy`, `--alpha`, `--calibration-from`, `--weights` (`/weights_checker`);
  - capture format 2: full precision, top-2 logprobs;
  - `faults.py`: 11 fault kinds on raw safetensors bytes, with `inject`, `diff` and `validate`;
  - 47 new tests; 183 pass.
- **This folder:**
  - `cpu/engine.py`: HF Qwen3 weights in fp32 with BF16 rounding at kernel boundaries, plus split-K, split-KV, TP all-reduce, chunked prefill and prefix hits. With rounding off it is bitwise equal to transformers' forward.
  - `cpu/server.py`: an OpenAI and `/weights_checker` server whose numerics follow load.
  - The study (`cpu/study.py`, `cpu/report.py`) and the GPU runbook (`study_driver.py`, `modal_app.py`, `MODAL.md`).

## Results (CPU: measured; GPU: not yet run)

Qwen3-0.6B in BF16, its official FP8 release, and an MXFP4-MLP variant. 147 faults injected into the real safetensors files, 180 held-out benign runs, α = 0.01. Tables are in `results/cpu/summary.md`.

| | BF16 | FP8 | MXFP4 |
|---|---|---|---|
| benign noise, KL(P_ref‖P_run) | 5.4×10⁻⁴ | **9.2×10⁻³ (17×)** | 5.0×10⁻⁴ |
| false alarms, 16 / 32 calibration runs | 0/60 / 0/44 | 0/60 / 2/44 | 3/60 / 0/44 |
| structural faults detected (shard zeroed/duplicated/permuted/transposed, vocab shard, exponent-MSB flip, missing tensor, zeroed MXFP4 blocks) | **39/39** | — | **6/6** |
| faults detected with KL ≥ 3× benign | **49/49** | **3/3** | **11/11** |
| faults detected below benign KL | 0/32 | 2/23 | 0/1 |
| exact mode (reference schedule reproduces) | 93/93 | 33/33 | 21/21 |

- **Detection follows effect size.** Every fault that moves the output distribution by at least 3× serving noise is caught (63/63). No noise-tolerant check can catch faults smaller than the noise, and they change the model less than serving it under load does.
- **FP8: verify exactly.** Dynamic FP8 activation quantization amplifies benign drift 17×; one run moved a logprob by 3.3 nats. A tolerance that absorbs that hides FP8 scale faults: "5% of blocks ×2" and "scales from the other TP shard" were each caught 0/3. Both the reference schedule and the batch-invariant mode reproduce bitwise here.
- **The study caught our own first rule.** A log-normal bound over 16 run maxima gave 4/60 MXFP4 false alarms. One token (p = 0.988) moved 10σ under a per-request setting that calibration never sampled. Run maxima are heavy-tailed (fitted GPD shape ξ = 0.14–0.47), hence the pooled-tail rule. False alarms are now 3/180 at K=16 and 2/132 at K=32, ~1.6% against a nominal 1%. **Use `--calibrate 32`** (~2–5 min on a GPU server).
- **The CLI end to end** (`results/e2e_cli/`):
  - a clean server passes;
  - clean weights with other kernel choices fail exact with *"within the calibrated noise; weight checksums equal"*, and pass `statistical`;
  - zeroing 1/8 of one projection fails every mode (logprob −0.0006 → −0.91);
  - 16 mantissa flips pass `statistical` but fail `--weights`, which names the tensor.

  **Checksums and logprobs are complementary.** The study driver also ran end to end (`results/e2e_cpu_final/`).

## How to run

```bash
cd vendor/servekit && PYTHONPATH=src python -m pytest tests -q
```

```bash
python -I cpu/study.py --variant bf16 && python -I cpu/report.py results/cpu
```

```bash
bash cpu/e2e_cli.sh
```

```bash
python modal_app.py plan
```

`MODAL.md` covers Qwen3-1.7B BF16 and FP8 and gpt-oss-20b MXFP4 on SGLang 0.5.20, plus a vLLM 0.33 arm, on one H100. **~$14.6 estimated, $24.6 worst case** (budget-capped), split over up to three accounts. Every fault spec and engine flag has been checked against the released packages.

## On Alps, and the risks

- **On Alps.** Record baselines with `--calibrate 32 --weights` and keep `--mode auto`.
  - BF16 and FP8 stay exact.
  - FP4 MoE falls back to the statistical test only if its repeat does not reproduce.
  - Cross-node or cross-TP comparisons go through `--calibration-from` a second baseline.
- **Risks.**
  - The false-alarm guarantee covers only benign variation that calibration samples; the Modal study measures restarts, settings and batch-invariant mode.
  - Faults below the noise rely on exact mode or checksums.
  - **Prediction:** `/weights_checker` cannot see MXFP4 experts moved off the module (the #14 mechanism). This is read from the source; the gpt-oss arm tests it.

## Pitch

> servekit's 1e-6 verify fails correct loads across TP degree (0.155 in your Llama-70B fixtures) and hardware, and any hand-set tolerance also hides real faults. I made verify measure its own noise floor on the trusted server: it stays bitwise where the engine reproduces and becomes a calibrated test where it cannot, next to weight checksums. On CPU it caught 63/63 faults larger than serving noise, at ~1.6% false alarms where fixed tolerances alarmed on 180/180. It also showed FP8 must be verified exactly, because its drift is 17× BF16's. A ~$14 Modal study measures the same on SGLang 0.5.20, including gpt-oss MXFP4.
