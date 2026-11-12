# Lazy, traffic-driven CUDA-graph capture for SGLang

*12 Nov 2026. **M** = measured (the lab's logs or the trace). **P** = predicted (model or simulation). Nothing has run on a GPU yet.*

## 1. Problem

- **After servekit, graph capture is the biggest remaining slice of a cold start.** Clariden GH200, SGLang 0.5.16, lab runs:
  - Llama-3.1-70B TP4: prefill 46.3 s + decode 8.0 s = **46% of 116.8 s** (M);
  - Apertus-8B TP4: **35% of 95.0 s** (M).
- **Cost per bucket**, fitted to the lab's capture timelines (3 runs):
  - 70B: prefill **0.37 s + 121 µs per token**, decode **0.29 s** (M);
  - SGLang captures 58 prefill and 36 decode buckets, largest first;
  - the first bucket also carries a 16–18 s one-off (JIT, allocator), which persistent caches shrink and lazy capture only moves.
- **Most buckets go unused at launch** (OpenTela trace, M):
  - Apertus-70B users reach p99 10 in flight with p99 1.3K-token prompts;
  - research checkpoints are idle 78–99% of the time.
- **CRIU and GPU snapshots are not allowed on CSCS** (Youssef's write-up §IV–V). A fix must need no privileges and no offline artifact.

## 2. Related work, and what is new

- **Capture on first use, paid by the request:** PyTorch CUDA Graph Trees, LMDeploy, TVM/MLC. vLLM PR #23184 captured at runtime, current shape first. It was closed unmerged in Mar 2026; a reviewer reported garbled output under multi-TP.
- **Idle-time capture:**
  - suggested in vLLM RFC #20098 ("capture if there are no requests to be scheduled") but never built for cold start;
  - the closest built work is **vLLM PR #57899** (open). After sleep/wake it recaptures one shape per idle boundary, largest first, via executor RPCs. FULL graphs only; 0.07–0.14 s per graph.
- **Removing capture with offline artifacts:**
  - **Medusa** (ASPLOS '25). Its §2.4 argues deferral "merely delays and disperses" capture.
  - **Foundry** (arXiv 2604.06664): templates plus `cuGraphExecUpdate`. Needs an offline step and an `LD_PRELOAD` hook.
- **Conceptual precedent:** the Jalapeño/Jikes RVM adaptive optimizer orders background recompilation by expected improvement net of compile cost.
- **New here:**
  1. **A cold-start policy from traffic.**
     - EV = request-weighted frequency × eager penalty ÷ capture cost.
     - Deferred buckets go in decreasing EV order (Smith's rule, optimal for Σ w·C).
     - A bucket is captured at startup iff EV > λ/H (requests waiting at launch ÷ deferral time), both estimated per model from the trace.
  2. **Idle-only, value-ordered capture in SGLang,** including breakable prefill graphs.
     - A request waits at most one step.
     - Dispatch is bucket-exact, so replayed batches run exactly today's kernels.
  3. **A TP agreement protocol**, tested with simulated ranks:
     - rank 0 decides *when*, through the existing request broadcast;
     - the next bucket is replicated state;
     - one MIN all-reduce checks match, idle and headroom;
     - lazy capture disables itself everywhere on divergence.
  4. **Hazards found in the code and fixed:**
     - deferred prefill capture would write dummy KV into slots the radix cache owns; SGLang resets index buffers only for decode capture;
     - eager all-reduces return zeros during a capture session, so background-thread capture is unsafe;
     - the breakable-graph output buffer would be reallocated per session;
     - four draft runners skip the base `__init__`.
  5. **Evaluation on a production trace,** with sensitivity analysis.
  - **On Medusa's objection:** no capture runs on a request's path, and the stall is bounded and measured.

## 3. What was built

- **`patches/0001-…patch`** for SGLang **v0.5.20** (94602c9): 9 files, +962/−32, applies to a fresh clone of the tag. Every hook is listed with `file:line` in `patches/HOOKS.md`. The main ones:
  - **`scheduler.py`:** leader tick in `ingest_requests` (2066), step in `on_idle` (4721).
  - **Decode runner:** check in `can_run_graph` (680), capture split (993/1060).
  - **Prefill runner:** check after `can_replay_locally` (1209), split capture (1405) with an index reset.
  - **Breakable backend:** shared output buffer (95/133).
  - **Config:** 3 keys in `--cuda-graph-config` (`cuda_graph_config.py:76`).
  - The frozen `model_runner.py` is untouched.
- **`lazygraph/`:** trace profiles, cost models, the EV policy (emits the config JSON), and a trace-overlay simulator. CLI: `policy|simulate|sweep|replay`.
- **`tests/`:** 34 passing tests, also against the clean patched tree.
  - 4 simulated ranks with a barrier all-reduce cover lockstep, a tick dropped when a request arrives, the one-step wait bound, memory failure, divergence and promotion.
  - Smith's rule is checked by brute force.
  - The patched parser accepts the policy's JSON.
- **`modal_app.py` + `MODAL.md`:** default vs small fixed set vs lazy, an eager-penalty sweep, and a correctness test (radix-cache reuse across deferred prefill captures).
  - It passes `--dry-run` against a CPU mock.
  - About $12 at TP1, plus $11 for TP2.
- **`drafts/`:** an SGLang RFC, and a vLLM design note (Model Runner V2 hooks at v0.33.0).

## 4. Results so far

| claim | value | type |
|---|---|---|
| 70B TP4 capture, default → lazy (startup set for Apertus-70B traffic: decode {1,2,4,8,40}) | 54.3 s → ~17 s (one-off + 1.4 s); cold start 116.8 → **~80 s (−32%)** | P from M costs |
| Same with persistent JIT caches (#270, one-off ~5 s) | ~67 s | P |
| Extra latency per launch caused by capture (startup wait + eager penalties + stalls), default / fixed / lazy | Apertus-70B (105 launches): **163 / 65 / 5.9 s**; bulk Qwen3-Next (28): **1734 / 2880 / 357 s** | P |
| Requests delayed by a capture step (lazy, 1 s hysteresis) | 0.06–3%; max **≤ 1.35 s** (one 70B prefill step) | P |
| All buckets captured (lazy) | p50 52–82 s after ready; 93–100% of launches within 30 min | P |
| Robustness | lazy lowest for eager penalties ×0.5–×5 (prefill ×30) and every hysteresis tried | P |
| Where it does not pay | if the startup set already covers ~100% of traffic, fixed beats lazy by ≤ 0.5 s per launch | P |

**Simulation caveats:**
- A launch is the first request after ≥ 15 min idle.
- Busy periods come from the trace, with no feedback from the arms.
- Eager penalties are modelled (`lazygraph/costs.py`). The Modal `eager` sweep measures them, and `--penalty` re-plans with the measured values.

**Experiment that confirms them** (`MODAL.md`, Apertus-8B, TP1 on H100, 3 runs per arm). The prediction is:
- lazy ready = default − default's capture + ~1 s, and launch TTFT falls by the same;
- replayed requests wait at most one step (≤ 0.9 s);
- every bucket is captured within minutes;
- throughput is equal and graph memory is within ~10%;
- greedy outputs are identical, including after prefix-cache reuse.

## 5. How to run

- **Locally:** `uv pip install pandas pyarrow numpy pytest msgspec modal`, then `pytest tests` and `python modal_app.py --dry-run`. For a policy: `python -m lazygraph.cli policy --model swissai/Apertus-70B-Instruct --engine llama70b_tp4_gh200`.
- **On Modal:** see `MODAL.md`.

## 6. On Alps, and risks

- **On Alps:** put the patch in `sglang_base`, or ship it as an SGLang plugin as servekit does. Generate the config per model from the platform's logs and pass it through model-launch's env TOML. Validating on Clariden takes about 1 node-hour: Llama-70B TP4, 3 cold starts per arm.
- **Risks:**
  - **OOM in a deferred step.** The reserve already covers the full set, and a headroom check runs on every rank before each step. A capture failure is still fatal, as at startup.
  - **Fragmentation** from capturing in value order rather than largest first.
  - **JIT moves to the first eager forward** without persistent caches; pair with #270.
  - **Untested model types:** MoE, MLA and PP are excluded for now.
  - **Maintainer resistance** to runtime capture (vLLM pushed back).

## 7. Pitch

After servekit, graph capture is a third to half of a cold start, yet the platform's own trace shows most buckets go unused for minutes. I built a no-privilege SGLang patch that captures a trace-chosen set at startup and the rest during idle time, in lockstep across TP ranks. On the trace it predicts ~30 s off a 70B launch and 5–28× less capture-caused latency, with no request delayed by more than one step. A $12 Modal run tests it; Clariden is next.
