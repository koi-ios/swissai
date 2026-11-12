# SGLang cold-start idea: evaluation

11 Nov 2026. Checks the original plan ([PLAN-2026-11-11.md](PLAN-2026-11-11.md), then named `CLAUDE.md`) against the lab's code, issues, logs and production trace. Every number below comes from the lab's own runs or data and is reproducible with the scripts in [analysis/](analysis/). Single runs are marked as such.

## Verdict

- **Part A (persistent caches) is not new for SGLang.** servekit has had JIT caching since 17 Aug (eth-easl/servekit #18). Youssef Boughizane measured it during his summer internship with Xiaozhe: Llama-3.1-70B TP4 went from 217.6 s to 146.2 s. Pitching Part A as written would show you hadn't read their work.
- **But three gaps are open, and nobody has written them down:**
  1. **Not deployed.** The platform doesn't use it: model-launch has no reference to servekit or to any cache variable, and #270 is open and unassigned.
  2. **No longer covers the current image.** In the platform's SGLang 0.5.20, SGLang moved its own JIT cache to directories servekit doesn't persist.
  3. **Keyed by checkpoint, not image.** servekit stores the cache next to each checkpoint. The trace shows the platform launches dozens of short-lived checkpoints, which a per-image cache would serve better.
- **Part B (trace-driven CUDA-graph capture) is real but modest.** On the lab's Clariden logs, trimming the capture set saves about 12–22 s of a 90–160 s post-servekit cold start. The lever is the prefill capture ceiling, not the decode batch-size list. It is good as a research extension, not as the lead.
- **Most effective idea:** lead with #270 for both engines plus an automated cache audit per image. Back it with the post-servekit breakdown from their own data. Offer traffic-aware, lazy graph capture as the research part. Keep servekit-for-vLLM as milestone 2.
- **Plausibility:** a specific, data-backed email is likely to get a meeting. Paid hours from December are plausible but not the default outcome (see the last section).

## Claims in the plan, checked

| Plan says | What is true | Source |
|---|---|---|
| The user-facing platform runs on SGLang | **Partly.** The big flagships run on SGLang: GLM-5.1 (8 nodes, TP32), GLM-5.1-FP8, Qwen3.5-397B, MiniMax-M2.5, Qwen3-235B, gpt-oss-120b, and Apertus-70B and 8B. **Every model added since September runs on vLLM:** DeepSeek-V4.1-Flash, GLM-5.3, MiMo-V2.6 and the Apertus 1.5 release. | model-launch `assets/models.json`, `mfa_examples/` |
| SGLang version: whatever `sglang_base` pins | **SGLang 0.5.20 with FA3.** The image is built in the private repo swiss-ai/model-launch-images. `sglang_base` sets `SGL_ENABLE_JIT_DEEPGEMM=0` and no cache variables. | model-launch #271, `envs/sglang_base.toml` |
| Does servekit JIT caching run on the platform? | **No.** It runs only under `servekit launch` with a prepared artifact, and model-launch never calls servekit. | grep of model-launch; servekit `jit_cache.py` |
| Cloudburst cut 695 s to 88 s "by persisting caches" | **Mostly weights in the page cache** (weight loading 531 s → 31 s). The blog gives no number for the compile caches alone. Don't cite it as evidence for caching. | fergusfinn.com blog |
| The trace has no durations | **It has `created_at` and `finished_at`.** But only 4.08M of 13.88M successful requests have `finished_at > created_at`. Coverage per model: Llama-3.3-70B 6%, Qwen3-32B 73%, Apertus-70B / Qwen3-Next / gpt-oss 100%. | HF dataset card; [analysis](analysis/results/trace/meta.json) |
| Over 13M requests, Jul 2024 – Oct 2025 | 16,329,237 rows (13,878,069 successful), 2024-07-17 to 2025-10-23, 214 model names. The trace ends a year before today's flagships, so it doesn't cover them. | same |
| Newer SGLang may split `--cuda-graph-max-bs` by phase | **Yes, in 0.5.20.** The flags are `--cuda-graph-{max-bs,bs}-{decode,prefill}`, `--cuda-graph-backend-{decode,prefill}` and a JSON `--cuda-graph-config`. Default backends: prefill `breakable`, decode `full`. So a traffic-derived capture set needs no code change. | sglang v0.5.20 `cuda_graph_config.py`, `arg_groups/` |
| Part B "works within CSCS limits" | **Yes, and that matters.** Youssef tried CRIU and graph snapshotting (Medusa, Foundry). CRIU needs capabilities CSCS won't grant and is blocked by enroot's seccomp filter. Medusa and Foundry aren't production-ready. His write-up asks for "an alternative that works in HPC environments". | experiment repo README §IV–VI |
| (implicit) Apertus-8B traffic is real | **No.** 99.2% of its trace requests are one fixed 85-token prompt every 10.0 s: a health probe. I excluded it. | [analysis](analysis/trace_concurrency.py) |

## The numbers

### 1. Where the cold start goes after servekit (the lab's runs, Clariden GH200, SGLang 0.5.16)

servekit overlap arm, head node, **one run per cell** (the serial arm agrees within about 10%). Source: [residual_breakdown.md](analysis/results/residual_breakdown.md).

| phase (s) | Apertus-8B TP4 | Llama-3.1-70B TP4 | GLM-4.7 TP4·PP4, 4 nodes |
|---|---|---|---|
| **total** | **95.0** | **116.8** | **105.2** |
| process start + TP worker spawn | 26.1 (27%) | 24.7 (21%) | 26.2 (25%) |
| torch.distributed init | 20.8 (22%) | 19.0 (16%) | 7.8 (7%) |
| weight loading (servekit) | 0.9 (1%) | 6.0 (5%) | 6.7 (6%) |
| KV cache allocation | 4.1 | 3.2 | 4.1 |
| prefill CUDA graphs (breakable) | 27.6 (29%) | 46.3 (40%) | n/a with PP |
| decode CUDA graphs | 5.5 (6%) | 8.0 (7%) | 30.8 (29%) |
| first request (JIT) | 6.5 (7%) | 6.3 (5%) | 26.1 (25%) |
| other gaps | 3.5 | 3.2 | 3.6 |

Weight loading is solved. What's left is four roughly equal pieces: Python start, distributed init, graph capture and JIT.

### 2. What caching buys, and where (Bristen, Llama-70B TP4, 0.5.16, one run per arm)

| phase (s) | no cache | servekit cache | change |
|---|---|---|---|
| total | 217.6 | 146.2 | −71.4 |
| prefill graph capture | 73.4 | 43.3 | −30.1 (the first, 8192-token size: 28 s → 5 s) |
| torch.distributed init | 19.5 | 2.7 | −16.8 |
| first request (JIT) | 8.9 | 1.5 | −7.4 |
| process start + worker spawn | 79.8 | 63.4 | −16.4 |

- **The write-up's 54.6 s total stands, but its attribution is reversed.** It called the distributed-init drop "network-fabric noise", removed it, and kept the process-start drop. The logs say otherwise:
  - The 17 s sits between "sglang is using nccl" and "All Reduce config" (+10 s) and just after it (+7 s). That is SGLang's custom all-reduce v2 setup, which JIT-compiles kernels through `load_jit` (`sglang/kernels/ops/communication/all_reduce.py`).
  - The `prepare` run paid the same 18.6 s and "kept 137 JIT cache files". The cached launch then took 2.7 s.
  - Process start has no JIT I can find and swings 9–53 s across runs.
  - So the attributable saving is 30.1 + 16.8 + 7.4 ≈ 54 s (25%). SGLang's own JIT kernels are about 17 s of it.
- **Caveat:** five earlier launches staged caches from a different artifact (`llama70b-tp4-sharded`) and still took 19–22 s in distributed init. Its `prepare` log isn't in the repo. The likely cause is that it was built with another image, so the hashed build keys didn't match. A 3+3 controlled repeat would settle it.

### 3. servekit's cache list misses the platform's SGLang 0.5.20

| What gets compiled | 0.5.16 (measured by Youssef) | 0.5.20 (`sglang_base`) | Persisted by servekit? |
|---|---|---|---|
| SGLang `load_jit` kernels (incl. all-reduce) | `$TVM_FFI_CACHE_DIR` | `$SGLANG_JIT_CACHE_DIR`, default `~/.cache/sglang/jit`. Does **not** follow `SGLANG_CACHE_DIR`. | 0.5.16 yes, **0.5.20 no** |
| CuTe DSL AOT kernels | n/a | `$SGLANG_CUTE_AOT_CACHE_DIR` = `$SGLANG_CACHE_DIR/cute_aot` | no |
| DeepGEMM | `~/.cache/deep_gemm` | `$SGLANG_DG_CACHE_DIR` = `$SGLANG_CACHE_DIR/deep_gemm` | no (DeepGEMM JIT is off on the platform) |
| Triton, FlashInfer, TVM-FFI | as before | as before | yes |

- **Prediction (not yet measured):** on 0.5.20, a servekit launch still pays the all-reduce JIT (about 17 s at TP4).
- **The fix:** set `SGLANG_CACHE_DIR` **and** `SGLANG_JIT_CACHE_DIR`.
- **The general lesson:** cache locations move between releases, so a fixed list rots. #270 needs an audit that runs per image, not a list.

### 4. What a smaller capture set saves (Part B)

- SGLang captures the largest size first, so the first size also carries one-off costs: JIT, the first forward pass, allocator warmup. That step took 5 s with a cache and 28 s without (Bristen A/B), and 12–18 s on the Clariden runs. A lower ceiling doesn't remove it; it only moves it to the new first size.
- Savings below exclude that first step. Source: [capture_timeline.csv](analysis/results/capture_timeline.csv).

| Clariden, servekit runs | prefill ceiling 2048 | 1024 | 512 | decode cap 32 | decode cap 16 |
|---|---|---|---|---|---|
| Llama-70B TP4 (3 runs; prefill phase 45–51 s) | 12 s | 13–14 s | 20–21 s | 5–8 s | 6–8 s |
| Apertus-8B TP4 (2 runs; prefill phase 25–26 s) | 3 s | 8–9 s | 9–10 s | 3 s | 3–4 s |

- **Per-size cost grows with the token count on 70B:** 1–3 s per size above 4096 tokens, about 0.25 s per size below 512. Decode sizes cost 0.1–0.3 s each.
- **Not yet measured:** the latency cost (prefills above the ceiling run eagerly; so do decode batches above the cap) and the graph memory saved. Both are needed before recommending it.

### 5. What traffic needs (trace estimates)

"In flight" counts a model's requests between creation and finish, summed over all its replicas and including queueing. So it is an upper bound on any one replica's batch size.

| Model (trace) | ok requests | in flight p99 / p99.9 / max | busy time with ≤16 / ≤32 / ≤64 in flight | prompt tokens p50 / p99 | idle share of its deployment |
|---|---|---|---|---|---|
| swiss-ai Apertus-70B-Instruct (real users) | 68,718 | 10 / 40 / 48 | 99.1% / 99.3% / 100% | 356 / 1,341 | 96% |
| Qwen3-Next-80B-A3B-Instruct (bulk) | 196,766 | 243 / 476 / 542 | 28% / 46% / 78% | 1,572 / 5,058 | 88% |
| gpt-oss-120b (bulk) | 34,433 | 219 / 283 / 313 | 27% / 36% / 62% | 4,288 / 4,603 | 99% |
| Qwen3-32B (bulk, 73% duration coverage) | 2,899,634 | 885 / 1,157 / 1,650 | 46% / 52% / 58% | 979 / 7,326 | 91% |

- **Two traffic classes.** Bulk endpoints need the full capture set. The Apertus chat model and research checkpoints do not: most checkpoints sit at p99 ≤ 10–45 in flight with prompts p99 ≤ about 1.8K tokens.
- **Checkpoint churn.**
  - Of 93 model names with ≥ 100 requests, 68 were served on ≤ 7 distinct days (43 of them Apertus), and 54 on ≤ 3 days.
  - August 2025 alone added 33 new names, 30 of them Apertus.
  - They sit idle 78–99% of their span, with 4–17 idle gaps ≥ 15 min per day ([model_churn.txt](analysis/results/trace/model_churn.txt)).
- **Why this matters:**
  - A per-checkpoint artifact (servekit's presharding plus the cache beside it) is rebuilt for every checkpoint.
  - A compile cache keyed by image and GPU, which is #270's design, is shared across every checkpoint of the same architecture.
  - The idle gaps are where lazy graph capture would run.

## Opportunity map

| # | Opportunity | Who asked for it | Value to the lab | Evidence you hold now | Effort | As a hiring hook |
|---|---|---|---|---|---|---|
| 1 | **#270 for both engines: per-image + per-arch shared compile caches, plus a cache audit per image** | Aryan in #270 (vLLM first, then "the equivalent for SGL"); unassigned since 18 Oct | High. vLLM: DeepSeek-V4.1-Flash spends ~40 min in TileLang JIT, GLM-5.3 takes ~60 min to start. SGLang: ~54 s per 70B launch. | §2, §3, §5 (churn), plus your vLLM harness | 2–3 weeks at 10–15 h/week; final validation needs Clariden access | **Strongest:** an explicit ask, measurable, and you found a regression they don't know about |
| 2 | servekit: cover 0.5.20's cache directories (`SGLANG_CACHE_DIR`, `SGLANG_JIT_CACHE_DIR`) with a test | Not asked; follows from #18 | Medium; restores about 17 s per TP4 launch under servekit | §3 | Hours | A good first PR |
| 3 | servekit for vLLM (presharding, `/dev/shm` staging, overlap) | servekit roadmap; Youssef's future work ("CSCS folks use vLLM") | Medium-high, but vLLM ≥ 0.33 auto-prefetches on Lustre, which may shrink the gap; measure on capstor first | Your vLLM harness (no GPU run yet) | 4–6 weeks | Good; already your milestone |
| 4 | **Traffic-aware, lazy CUDA-graph capture**: serve after a small hot set, capture the rest in idle time, with a policy per traffic class | Xiaozhe in #125: cold start "can be turned into a research project"; Youssef: "an alternative that works in HPC environments" | Medium now (12–22 s per launch). Higher where JIT runs per captured shape (vLLM DeepSeek-V4.1: 51 graphs, ~40 min) | §4, §5 | Research; 1–2 h of GPU for a first trade-off curve | **Best research angle** |
| 5 | Process start + worker spawn (21–29%) | Nobody | Possibly every launch | Weak: a compile-time `SyntaxWarning` in 44 of 56 experiment logs, so some modules recompile at launch | A 10-minute import-time test | A nice finding if it holds |
| 6 | OpenTela #73: a restarted worker stays unroutable for minutes | Xiaozhe, 6 Nov | High for time-to-serve after a restart | None (Go, needs a mesh) | Unknown | Off your core skills |
| 7 | Others: correctness tolerance for low-precision models (Youssef's future work); VAST datastore for the new platform; servekit #19 (MLA round-trip); model-launch-images #22 (GLM-5 TP32 barrier, private repo) | Various | Varies | None | Varies | Weaker fits |

## Recommendation

Lead with #1. It is what the lab asked for, it is measurable, and you have something new to say about it.

1. **Email, Fri 13 Nov.** Frame it as building on servekit, not redoing it.
   - Show the §1 table: "after servekit, here's what's left, from your runs".
   - Give three observations:
     - Caching saves about 54 s, a third of it from the all-reduce JIT.
     - On 0.5.20 that JIT lives where servekit doesn't look.
     - The capture ceiling, not the batch list, is what costs time.
   - Ask for a 20-minute meeting and propose 10–15 h/week:
     - M1: #270 for SGLang and vLLM, with the audit.
     - M2: servekit for vLLM.
     - M3: traffic-aware graph capture.
   - Phrase the attribution point as an observation with log lines, not as a correction.
2. **GPU add-on.** Run it in the same VM session as the vLLM prelim: about 1.5 h on 2× H100, roughly USD 5–10.
   - (a) SGLang 0.5.20 (`lmsysorg/sglang:v0.5.20`), Apertus-8B TP2, three arms with 3 runs each: all caches empty; servekit's three variables kept; all 0.5.20 variables kept. This turns §3 into a measured number.
   - (b) Capture sweep: prefill ceiling {default, 2048, 1024} × decode cap {default, 32, 16}. Record startup time, graph memory, TTFT at 512 / 1.5K / 4K / 8K tokens, and TPOT at batch 1 / 8 / 32 / 64.
   - (c) `python -X importtime` and a `.pyc` count in the image.
   - Each needs a small SGLang version of `run_once.py`.
3. **First contribution, once you choose to.** Either a #270 comment with the per-cache table for both images (variable, key, write pattern, saving), or the servekit PR from #2. Both are public actions, so they're yours to send.

**Questions for Xiaozhe:**

1. Is servekit (or its cache) meant to move into model-launch, or should #270 live in the images and env TOMLs?
2. Which launches hurt most today: SGLang flagships after image upgrades, or the vLLM MoEs?
3. Does TileLang on DeepSeek-V4.1 compile per captured batch size? If so, the capture set and the cache are one problem.

## Plausibility of an internship through this path

- **For:**
  - **Precedent:** the lab ran a summer internship on exactly this topic, supervised by Xiaozhe.
  - **A stated research interest:** Xiaozhe's #125 comment.
  - **An unassigned issue the maintainers wrote themselves.**
  - **An open-PR culture:** an AGENTS.md, and the "Opening PRs by Community" issue.
  - **Fit:** your systems, HPC and kernel background.
- **Against or unknown:**
  - Whether there is budget for paid hours.
  - Xiaozhe's bandwidth.
  - The lab may prefer a semester project or thesis.
  - CSCS access has to be granted.
  - A model-launch team (Aryan, Rob, Faruk, Mel) already ships this kind of change.
- **My rough judgment:**
  - A specific, data-backed email is more likely than not to get a reply or meeting.
  - Paid 10–15 h/week from December is about one in three.
  - The most likely first offer is "start on #270" or a project, which you can convert later.
- **What moves the odds:**
  - Something they can use already in hand (the 0.5.20 cache table, or a measured A/B).
  - Flexibility on the format.
  - Asking for a meeting, not a job.

## Reproduce

Data sources, cloned or downloaded outside this folder:
- trace: `https://huggingface.co/datasets/eth-easl/swissai-serving-trace` (`trace.jsonl`, 6.98 GB)
- lab runs: `git clone https://github.com/youssef62/cold-start-elimination-experiments-swissai`

```bash
python analysis/trace_to_parquet.py trace.jsonl trace.parquet          # ~20 s on 12 cores
python analysis/trace_concurrency.py trace.parquet analysis/results/trace
python analysis/residual_breakdown.py <experiments-repo> --out analysis/results/residual_breakdown.md
python analysis/capture_timeline.py <logs...> --out analysis/results/capture_timeline.csv
```

Needs `pandas pyarrow orjson numpy`.

## Sources

- eth-easl/servekit: README, `jit_cache.py`, `launch.py`, `prepare.py`, `profile.py`, test fixtures (commit cd8f0b8)
- Youssef Boughizane, "Cold Start Elimination for SwissAI Model Launch", github.com/youssef62/cold-start-elimination-experiments-swissai (§III JIT, §IV CUDA graphs, §V CRIU), and its raw `servekit profile` JSONs and logs
- swiss-ai/model-launch: #270, #265, #271, #125 (comments by schuups and xzyaoi), `envs/*.toml`, `assets/models.json` (commit 02b74ce)
- sgl-project/sglang v0.5.16 and v0.5.20: `environ.py`, `jit_kernel/utils/compile.py`, `kernels/jit/utils/compile/cache.py`, `custom_all_reduce_v2.py`, `cuda_graph_config.py`
- eth-easl/swissai-serving-trace (OpenTela, OSDI '26)
- fergusfinn.com/blog/fast-sglang-starts; lmsys.org/blog/2026-08-17-advanced-cuda-graph; ROCm Infera DSv4 example; serving.swissai.svc.cscs.ch/articles/05-servekit
