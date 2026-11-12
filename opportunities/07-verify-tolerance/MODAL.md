# Modal runbook: the verify-tolerance study on real engines

`modal_app.py` runs `study_driver.py` on one H100 per model. The same driver was run end to end on this laptop against the CPU engine's HTTP server (`results/e2e_cpu/`), so what is left untested is only what needs a GPU: the engine images, their flags and their timings.

## What one run measures

For one model on one engine, in one container:

1. **Reference.** A clean server gives the reference capture, the `/weights_checker` checksums (SGLang), `calibrate()` (one repeat of the reference schedule plus 32 captures under perturbed concurrency and filler traffic; 16 were too few in the CPU study) and 24 held-out perturbed captures. This is the request-level noise of one process.
2. **Restarts.** The clean model on 2 fresh processes, each captured on the reference schedule and once under load. Does a restart reproduce bitwise, and if not, does its drift stay inside the calibrated noise?
3. **Other engine settings,** without changing the weights:
   - SGLang: `--disable-cuda-graph`, `--chunked-prefill-size 512` and `--attention-backend triton`.
   - vLLM: `--enforce-eager` and `--max-num-batched-tokens 512`.
   - These are the drifts a fixed tolerance cannot tell from a fault.
4. **Faults.** Each corruption in the model's grid is injected into a copy of the real checkpoint (`servekit.faults`; only the touched shard is rewritten). For each, a fresh server is captured on the reference schedule, which is the candidate as `servekit verify` sees it, and once under load, together with checksums and held-out NLL. A server that will not start counts as a detection.
5. **Batch-invariant mode.** SGLang `--enable-deterministic-inference`, or vLLM with `VLLM_BATCH_INVARIANT=1`, gets its own reference and calibration (8 runs) and 4 held-out captures under concurrency 8 with 6 filler streams. If those reproduce bitwise, exact mode works on a loaded server.

| arm | model | weights | faults in the grid |
|---|---|---|---|
| `qwen3-1.7b`, SGLang 0.5.20 | Qwen/Qwen3-1.7B | BF16, 4.1 GB | 17: shard zeroing, duplication, permutation, transposed square projection, exponent and mantissa bit flips, missing norm, layer noise, norm scaling |
| `qwen3-1.7b-fp8`, SGLang 0.5.20 | Qwen/Qwen3-1.7B-FP8 | FP8 128×128 blocks, 2.7 GB | 8: `weight_scale_inv` rolled, jittered, ×2 on 5% of blocks, set to 1, taken from the other TP shard; an FP8 exponent flip; a zeroed FP8 shard |
| `gpt-oss-20b`, SGLang 0.5.20 | openai/gpt-oss-20b | MXFP4 MoE experts, 13.8 GB | 9: experts swapped (all tensors or `down_proj` only), E8M0 scales +1 on 1% / 10% of blocks, scales rolled, one layer's experts zeroed, **all experts zeroed (the sglang#34448 failure)**, an attention exponent flip, attention noise |
| `qwen3-1.7b`, vLLM 0.33.0 | Qwen/Qwen3-1.7B | BF16 | the first 8 of the BF16 grid |

`python modal_app.py plan` checks every fault spec against the real checkpoints' safetensors headers, read over HTTP range requests without downloading weights. All 42 apply (12 Nov 2026).

## Cost

The container is an H100 with 8 physical cores and 64 GiB. At modal.com/pricing on 12 Nov 2026 it costs $0.001097 (GPU) + 8 × $0.0000131 (CPU) + 64 × $0.00000222 (memory) = **$0.001344/s, or $4.84/h**. The download function runs on CPU only and costs about $0.01 per model.

The expected minutes are **estimates**. They come from `study_driver.estimate()`: assumed server start times (60 s for Qwen3-1.7B, 70 s for FP8, 150 s for gpt-oss-20b, plus 15 s to stop and restart) and capture times (6 s, or 12 s for gpt-oss), and nothing here has been timed on a GPU. Each arm also has a **budget**: the driver starts no new server once the budget has passed. The worst case is therefore the budget plus one step (about 15 min) at $4.84/h. The function's hard timeout (3 h) applies only if the driver itself hangs.

| step | expected | expected cost | budget | worst case |
|---|---|---|---|---|
| smoke, any model | ~5–8 min | ~$0.5 | 30 min timeout | $2.42 |
| study `qwen3-1.7b` SGLang | ~46 min | ~$3.74 | 60 min | $6.05 |
| study `qwen3-1.7b-fp8` SGLang | ~35 min | ~$2.82 | 50 min | $5.24 |
| study `gpt-oss-20b` SGLang | ~69 min | ~$5.56 | 90 min | $8.47 |
| study `qwen3-1.7b` vLLM | ~31 min | ~$2.50 | 45 min | $4.84 |
| **all four studies** | ~3.0 h | **~$14.6** | | **$24.6** |

### Split across accounts

Each account has $30 of credit, and Volumes are per account, so each one downloads its own weights.

| account | runs | expected | worst case |
|---|---|---|---|
| A | download + smoke + study `qwen3-1.7b` and `qwen3-1.7b-fp8` (SGLang) | ~$7.1 | $13.7 |
| B | download + smoke + study `gpt-oss-20b` (SGLang) | ~$6.1 | $10.9 |
| C | download + study `qwen3-1.7b` on vLLM | ~$2.5 | $4.9 |

If only one account is available, run A first: it answers the main question (BF16 and FP8 on the platform's engine) for about $7.

## Commands

```bash
pip install modal==1.6.3   # the version modal_app.py was checked with
modal setup                # once per account
```

Run every command from `opportunities/07-verify-tolerance/`, so that `vendor/servekit/src` and `study_driver.py` are uploaded with the image.

Plan, cost and fault-grid check. This runs locally and needs no account:

```bash
python modal_app.py plan
```

Weights into the account's `hf-cache` Volume. The models are ungated, so no Secret is needed:

```bash
modal run modal_app.py --mode download --model qwen3-1.7b
```

```bash
modal run modal_app.py --mode download --model qwen3-1.7b-fp8
```

```bash
modal run modal_app.py --mode download --model gpt-oss-20b
```

Smoke test, about 5–8 min. It brings up the clean server, captures, loads one corrupted checkpoint and checks it:

```bash
modal run modal_app.py --mode smoke --model qwen3-1.7b
```

The studies (`--detach` keeps a study running if your laptop sleeps):

```bash
modal run --detach modal_app.py --mode study --model qwen3-1.7b --engine sglang
```

```bash
modal run --detach modal_app.py --mode study --model qwen3-1.7b-fp8 --engine sglang
```

```bash
modal run --detach modal_app.py --mode study --model gpt-oss-20b --engine sglang
```

```bash
modal run --detach modal_app.py --mode study --model qwen3-1.7b --engine vllm
```

Optional flags:
- `--budget-min N` overrides an arm's budget.
- `--max-faults N` runs only the first N faults.
- `--quick` runs a 4-calibration-run, 1-restart plan as a cheaper second smoke test.

Every capture is committed to the Volume as it is taken. A study that is stopped, or runs out of budget, keeps what it has. Rerunning the same command resumes it and skips the records already written.

## Results

```bash
modal volume get verify-tolerance-results / results/modal/
```

```bash
python -I cpu/report.py results/modal/qwen3-1.7b
```

- Each model has its own directory holding `<model>-<engine>.jsonl.gz`, `<model>-<engine>-det.jsonl.gz` and `logs/`, with one engine log per server start.
- `cpu/report.py` writes `summary.md`, `runs.csv` and `roc.csv` next to them. It covers:
  - false alarms per benign family (held-out load, restarts, each config arm, the deterministic arm);
  - detection per fault class for statistical mode, exact mode, a fixed 0.05 tolerance and weight checksums;
  - which statistic fired;
  - the same faults under load;
  - ROC per fault class.
- Real engines give no full-vocab distributions, so the GPU tables report the held-out ΔNLL as the effect size. KL is measured only in the CPU study.

## What to look at first

1. Do restarts and config arms reproduce bitwise (exact mode viable)? If they do not, is their drift inside the calibrated noise (`statistical` false alarms)?
2. Does `--enable-deterministic-inference` make perturbed captures bitwise identical on BF16, FP8 and gpt-oss's MXFP4 MoE? Its own `-det` file answers this.
3. For gpt-oss-20b, how does the #34448-style "all experts zeroed" fault compare with the mild faults, and does `/weights_checker` notice it? The checksum hashes only `named_parameters()` and `named_buffers()`. servekit #14 found that SGLang's MXFP4 triton-kernel path moves expert weights off the module in `process_weights_after_loading`. If that still holds in 0.5.20, the checksum cannot see the experts at all and stays equal while the logprobs do not. This is a prediction from the source, and the study confirms or refutes it.

## Limits

- **Hardware.** An H100 SXM has the same SM 9.0 kernels as Clariden's GH200, but not the Grace CPU or NVLink-C2C. The same capture on a GH200 can differ from an H100's, which is exactly the cross-hardware drift a fixed tolerance trips on.
- **TP=1 only.** No TP-degree arm fits this budget. The CPU study emulates TP 2 and 4, and servekit's own Llama-3.1-70B TP4 and TP8 fixtures give one real TP pair (max |Δ| 0.155).
- **Storage and isolation.** The weights sit on a Volume, and the study copies them to the container's local disk so that corrupted copies can hard-link. There is no Lustre, which does not matter here. Containers are not VMs.
- **Start times** in the table are estimates until the smoke test has run once.
