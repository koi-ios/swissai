# Modal runbook: lazy CUDA-graph capture (SGLang 0.5.20)

Everything here is ready to run but **has not been run**: no Modal account is configured on this machine. `python modal_app.py --dry-run` exercises every mode against a CPU mock server (`bench/mock_server.py`); its output in `results/dryrun/` is mock data.

## What it measures

| mode | runs | what | answers |
|---|---|---|---|
| `smoke` | 1 lazy cold start, short replay | patch applies in the image, server starts, deferred steps appear in the log | is the setup sound before spending more |
| `arms` | 3 arms × `--repeats` cold starts, interleaved | per run: time-to-ready; TTFT of a request sent at launch; p50/p99 TTFT and TPOT under a trace replay that starts at launch; time until every deferred bucket is captured; closed-loop throughput afterwards (512 req, 512 in / 128 out, 64 concurrent); GPU memory after full capture | the main comparison: default vs small fixed set vs lazy |
| `eager` | 2 servers: default graphs vs `--disable-cuda-graph` | prefill TTFT at 512 / 1536 / 4096 / 8192 tokens and decode TPOT at batch 1 / 8 / 32 / 64, median of 5 | eager-fallback penalty per size → `eager_penalty.json` for the policy tool |
| `correctness` | default vs lazy (all prefill buckets deferred) | greedy output ids for 6 prompts; a 3000-token prefix is cached, deferred prefill captures run, then prefix+suffix hits the radix cache | deferred capture changes no output and does not corrupt cached KV |

**Arms.** All use the same image (SGLang 0.5.20 + `patches/0001-...patch`); only `--cuda-graph-config` differs.
- `default`: no lazy keys. Captures all 58 prefill and 36 decode buckets at startup, same as unpatched 0.5.20.
- `fixed`: prefill buckets ≤ 2048 tokens, decode ≤ 32.
- `lazy`: the policy's startup set from `results/policy/modal_apertus70b_traffic_8b_tp1.json` (decode {1, 2, 4, 8, 32, 40, 48}; no prefill buckets). The rest are deferred in expected-value order. Hysteresis is 1 s and promotion triggers after 4 eager hits.

**Replay.** The 8 post-cold-start windows of `swissai/Apertus-70B-Instruct` nearest the median size, taken from `results/replay/apertus70b_windows.jsonl` (cold start = first request after ≥ 15 min idle).
- The windows are superposed, which is about 8× one launch's rate, and time is compressed 2×. That gives roughly 120 requests in 5 minutes.
- Prompts and output lengths come from the trace. Prompts are capped at 8192 tokens, outputs at 256, and the prompt token ids are synthetic.

**Model.** `swiss-ai/Apertus-8B-Instruct-2509` at TP1 (and TP2 optionally) on H100. That is the smallest GPU that runs the platform's own FA3 code path, since FA3 is Hopper-only.

## Commands

Run from this folder (`opportunities/04-lazy-cuda-graphs/`), with `pip install modal` and `modal setup` done:

```bash
python modal_app.py --dry-run
```

```bash
modal run modal_app.py --mode smoke
```

```bash
modal run modal_app.py --mode arms --repeats 3
```

```bash
modal run modal_app.py --mode eager
```

```bash
modal run modal_app.py --mode correctness
```

Optional, on a second account:

```bash
modal run modal_app.py --mode arms --tp 2 --repeats 2
```

Fetch and summarize (the run id is printed at the end of each `modal run`):

```bash
modal volume get lazy-cuda-graphs-results / ./results/modal
```

```bash
python modal_app.py --summarize results/modal
```

Feed the measured eager penalties back into the policy and re-plan:

```bash
python -m lazygraph.cli policy --model swissai/Apertus-70B-Instruct --engine llama8b_tp1_h100 --penalty results/modal/<run>/eager_penalty.json --out results/policy/modal_measured.json
```

Environment overrides:
- `LAZY_MODEL=meta-llama/Llama-3.1-8B-Instruct` switches the model.
- `LAZY_HF_SECRET=huggingface` uses a Modal secret for gated models (create it with `modal secret create huggingface HF_TOKEN=...`).
- `LAZY_SGLANG_IMAGE=...` picks another image. `lmsysorg/sglang:v0.5.20` exists on Docker Hub: 15.1 GB amd64, published 2026-09-18.

## Time and cost (estimates, H100 ≈ $3.95/h; check modal.com/pricing)

| step | GPU time | cost |
|---|---|---|
| image build (pull 15 GB, apply patch) | CPU only | < $0.2 |
| weight download to the `hf-cache` volume | CPU only | < $0.1 |
| smoke | ~10 min | ~$0.7 |
| arms, TP1, 3 repeats (9 cold starts × ~14 min) | ~2.1 h | ~$8.3 |
| eager | ~24 min | ~$1.6 |
| correctness | ~24 min | ~$1.6 |
| **TP1 total** | ~3.1 h | **~$12** |
| arms, TP2, 2 repeats (2× H100) | ~1.4 h × 2 GPUs | ~$11 |

**Account split:**
- Account A: smoke, then arms TP1 (~$9).
- Account B: eager and correctness (~$3.5), plus TP2 arms if wanted (~$11).
- That stays within $30 on two accounts and leaves a third account as a reserve for reruns.

Stop early if `smoke` shows no `Lazy CUDA graph capture:` lines in the log (`results/.../lazy-tp1-r0.log`). That means the patch did not apply or lazy mode fell back. The fallback reason is logged as `Lazy CUDA graph capture disabled for <phase> (...)`.

## What would confirm or refute the predictions (WRITEUP.md §4)

- **Time-to-ready:** lazy minus default ≈ default's startup capture minus about 1 s. Predicted 15–25 s on 8B TP1 H100; the lab measured 33 s on Apertus-8B TP4 GH200 including the one-off.
- **Launch TTFT:** the same saving.
- **Interference:**
  - The maximum extra wait of any replayed request is one capture step: 0.1–0.9 s on 8B.
  - Under 3% of requests wait at all.
- **Full capture:** every deferred bucket captured within a few minutes of ready, in every lazy run.
- **Steady state:** lazy throughput equals default throughput within run-to-run noise. Graph memory after full capture is within ~10% of default; capture order differs, so pool fragmentation may differ.
- **Correctness:** all outputs identical between default and lazy, including prefix+suffix.

## Limits to keep in mind

- **Not Lustre.** Weights load from a Modal Volume, not capstor. Lazy capture changes only the capture phase, so this matters little here.
- **Not the platform's shape.** It runs 1–2 GPUs, not Clariden's GH200 TP4, and no 70B. The per-size cost model and the trace simulation carry the 70B story; Modal checks the mechanism and the 8B constants.
- **JIT caches start empty** in every container. Both arms pay the one-off JIT; lazy moves part of it from startup capture to the first eager forward or the first deferred step. Persistent caches (#270, opportunity 01) remove it in both.
- **Replay is open-loop with synthetic token ids.** Same prompt lengths as the trace, different content.
