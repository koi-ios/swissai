<!-- DRAFT for the user to review and post. Not posted. Target: sgl-project/sglang issues, "[RFC]" label. -->

# [RFC] Lazy CUDA graph capture: serve after a traffic-chosen startup set, capture the rest while idle

## Motivation

After weights load quickly, CUDA graph capture becomes a large share of cold start.

On a Llama-3.1-70B TP4 GH200 deployment of SGLang 0.5.16 with fast weight loading (servekit), the measured cold start was 116.8 s:
- prefill (breakable) capture: 46.3 s (58 buckets, 8192 down to 4);
- decode capture: 8.0 s (36 buckets).

Per-bucket cost grows with size: about 0.37 s + 121 µs per token for prefill, and about 0.29 s per decode bucket (fit over 3 runs). The first bucket also carries 16–18 s of one-off JIT and allocator warmup.

Many deployments never use most of these buckets in the first minutes, or ever:
- In a production trace of a research-serving platform (eth-easl/swissai-serving-trace), the chat model sees p99 10 concurrent requests and p99 1.3K-token prompts.
- Research checkpoints sit idle 78–99% of their lifetime.

Yet every cold start pays for all 94 buckets before the first token.

## Proposal

Opt-in, per phase, via the existing `--cuda-graph-config`:

```json
{"decode":  {"bs": [...], "initial_bs": [1, 2, 4, 8], "deferred_bs": [12, 16, 24, ...], "deferred_priority": [...]},
 "prefill": {"bs": [...], "initial_bs": [], "deferred_bs": [288, 576, 320, ...]}}
```

- **Startup.** Only `initial_bs` is captured. Static buffers and attention-backend graph state stay sized for the full `bs` list, so nothing is re-allocated later.
- **Dispatch.** Dispatch is bucket-exact: a batch replays a graph only if the bucket the default capture-all runner would use for it is already captured; otherwise it runs eagerly.
  - Lazy mode never pads more than today.
  - Replayed batches run exactly the kernels they run today.
- **Deferred capture.** The remaining buckets are captured one per scheduler iteration, and only while the scheduler is fully idle (`is_fully_idle()`, after a configurable hysteresis).
  - A request that arrives during a step waits for at most that one step.
  - The queue is ordered by `deferred_priority`, merging decode and prefill. Pending buckets that keep running eagerly are promoted to the head.
- **Lockstep across TP ranks** (graphs contain collectives; replay and eager take different all-reduce paths, so ranks must never disagree on whether a bucket is captured):
  1. TP rank 0 decides *when*. It appends a `LazyGraphCaptureTick(seq, phase, size)` to the existing per-iteration request broadcast (`ingest_requests` → `recv_requests`). *Which* bucket is next is replicated state on every rank.
  2. Every rank reaches `on_idle` in the same iteration. One MIN all-reduce on the TP CPU group agrees on three things: the tick matches the local queue head, the scheduler is fully idle, and there is memory headroom.
  3. Then every rank captures that bucket inside one `graph_capture` session.
  4. On disagreement, lazy capture turns itself off everywhere and serving continues with eager fallback.
- **Correctness detail: prefill capture during serving.**
  - Capture runs real warmup forwards, which write KV through `out_cache_loc`. After serving, that buffer still holds the last prefill's KV slots, which the radix cache may now own.
  - Decode capture already zeroes index buffers (`reset_index_buffers`, decode_cuda_graph_runner.py:1023-1026); prefill capture does not, because it never ran mid-serving before.
  - The patch resets the prefill index buffers before a deferred capture, so dummy writes land in the reserved slot 0.
- **Memory.**
  - The default mem-fraction reserve already accounts for the full bucket list (`reserve_for_graph_mb`), and deferred captures fill the same pool.
  - Before each step every rank calls `empty_cache()`, to return eager activation blocks to the driver, and checks a free-memory floor. A bucket that does not fit stays eager.
  - The BCG eager-tail output buffer is kept across sessions (one buffer sized for the largest planned bucket), not re-allocated per session.
- **Scope of the first version.** CUDA, TP only, decode `full` plus prefill `breakable`, no speculative decoding, LoRA, PP, DP attention, CP, PD, torch.compile, MLA, hybrid state pools or post-capture KV sizing. Anything else falls back to capture-all with a logged reason.

A policy tool, kept outside SGLang, picks `initial_bs` and the order from a request trace:
- Expected value per bucket = request-weighted frequency × eager penalty ÷ capture cost. Smith's rule says decreasing EV order minimises accumulated eager penalty.
- A bucket goes into `initial_bs` iff EV > (requests waiting at startup) ÷ (expected deferral time). Both quantities are estimated from the trace.

## Prior art (and how this differs)

- **vLLM RFC #20098 (2025):** "lazily capture cudagraphs if there are no requests to be scheduled" was suggested but not built.
- **vLLM PR #23184 (closed 2026-03):** captured at runtime on the request path, current shape first, otherwise largest.
- **vLLM PR #57899 (open):** after sleep/wake, recaptures one shape per idle boundary, largest first, FULL graphs only, via executor RPCs.
- **PyTorch CUDA graph trees, LMDeploy, TVM/MLC:** capture on first use, paid by the request.
- **Medusa (ASPLOS '25) and Foundry (arXiv 2604.06664):** remove capture with offline artifacts.

This proposal targets cold start, orders deferral by traffic value rather than size, covers breakable prefill graphs, and needs no offline step or privileges.

## Status

A patch against v0.5.20 (9 files, about +960/−30 lines) applies cleanly to the tag.
- Unit tests of the GPU-free parts pass: the schedule, and the controller under a simulated 4-rank all-reduce, covering lockstep, divergence, memory failure and promotion.
- It has **not** run on a GPU yet. A Modal experiment (default vs small fixed set vs lazy; Apertus-8B TP1/TP2 on H100) is prepared.

Questions for maintainers:
1. Is `--cuda-graph-config` the right surface, or should this be a server arg?
2. Is `ingest_requests` plus `on_idle` the right place for the tick?
3. Would you accept the prefill `reset_index_buffers()` change on its own, as a guard for any future mid-serving prefill recapture?
