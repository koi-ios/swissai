# SGLang cold starts after servekit

Handoff from a Claude chat, 11 Nov 2026. An additional project, separate from the vLLM one (`servekit-vllm-prelim`). Read this first, then start at the first unchecked task.

> **Checked on 11 Nov: read [EVALUATION.md](EVALUATION.md) before any task.** Several facts below are out of date:
> - servekit already persists JIT caches (#18, measured: about 54 s saved on Llama-70B), but it misses SGLang 0.5.20's `SGLANG_JIT_CACHE_DIR` and `SGLANG_CACHE_DIR`.
> - The platform runs SGLang 0.5.20 without servekit or any cache variables.
> - The trace does have durations, for 4.1M requests.
> - Cloudburst's 695 s → 88 s is mostly weights in the page cache, not compile caches.
> - Apertus-8B traffic in the trace is a probe.
>
> Part A is now "#270 for both engines, plus a cache audit per image". Part B's lever is the prefill capture ceiling. Scripts and numbers are in [analysis/](analysis/).

## Why this project exists

The person who suggested the original contact list said: the team uses vLLM for local hosting and inference for other tasks, but for deployment to people they use SGLang. So the user-facing Swiss AI serving platform runs on SGLang.

servekit (https://github.com/eth-easl/servekit) already fixed SGLang's weight loading. This project targets what is left of an SGLang cold start after that, in two parts:

- **Part A: persistent caches.** Keep compiled kernels, autotune results and JIT outputs between SLURM jobs. This is the SGLang half of swiss-ai/model-launch #270, which Aryan Ahadinia asked for on the issue.
- **Part B: trace-driven CUDA-graph capture sets.** Capture fewer graphs per model, chosen from the platform's real traffic in the OpenTela production trace.

Both parts can be measured faithfully on a rented GPU; Lustre weight loading cannot.

## Who it is for

- **Ana Klimovic** (ETH, head of EASL): can hire; asks students to email a CV.
- **Xiaozhe Yao** (her PhD student, serving-platform project lead): would supervise; supervised servekit.
- **Imanol Schlag** (Apertus co-lead, serving-platform project lead): can hire.
- **Aryan Ahadinia** opened #270. **Robert Smith** is the platform's "System Expert".

Results can go into the Fri 13 Nov email as an alternative or second milestone, or into a follow-up.

## Ask first (the user's contact or Xiaozhe)

1. Which models take longest to come up on the platform today?
2. Does servekit's JIT kernel caching already run on the platform? The servekit article says servekit "implements fast weight loading and JIT kernel caching", but the README documents only weight loading. If caching already runs, Part B leads.

## Context and numbers

- **servekit article** (https://serving.swissai.svc.cscs.ch/articles/05-servekit), Llama-3.1-70B on SGLang, Bristen:
  - before servekit, weight loading was 72% of a 629.8 s cold start, CUDA-graph capture (piecewise + regular) 17%, and the rest mostly JIT compilation and Python imports;
  - after servekit, total cold start was 179.0 s with 9.7 s of weight loading, and "the cuda graph capture is" the bottleneck.
- **servekit README**, Clariden GH200, SGLang v0.5.10: total cold start with servekit still 95.53 s for Apertus-8B (TP4) and 127.06 s for Llama-3.1-70B (TP4). That remainder is this project's target.
- **#270** (https://github.com/swiss-ai/model-launch/issues/270):
  - caches named: TileLang (`TILELANG_CACHE_DIR`), vLLM (`VLLM_CACHE_ROOT`), FlashInfer (`FLASHINFER_WORKSPACE_BASE`), Triton (`TRITON_CACHE_DIR`);
  - nothing persists today; shared path proposed by Aryan: `/capstor/store/cscs/swissai/infra01/sml/shared-caches`;
  - wanted: one directory per image digest and GPU architecture, concurrent-write safety, a check that a second launch skips the compile, and "the equivalent ... for SGL".
- **Outside evidence:**
  - Cloudburst (https://fergusfinn.com/blog/fast-sglang-starts/): an SGLang cold start on 8x B200 fell from 695 s to an 88 s warm start by persisting artifacts SGLang already knows how to cache (compiled kernels, autotuned configs, JIT outputs).
  - AMD (https://rocm.docs.amd.com/projects/infera/en/latest/examples/sglang_mix_dsv4.html): after weights load, SGLang captures one CUDA graph per batch size up to `--cuda-graph-max-bs`. A smaller max or an explicit `--cuda-graph-bs` list starts faster but replays graphs only up to that size.
  - LMSYS (https://www.lmsys.org/blog/2026-08-17-advanced-cuda-graph): Breakable CUDA Graph is now SGLang's default for prefill. With compiler-based piecewise graphs, torch.compile took 78-86% of prefill graph setup.
  - Newer SGLang versions may split capture limits into `--cuda-graph-max-bs-decode` and `--cuda-graph-max-bs-prefill`; check `--help` in the installed version.
- **OpenTela trace** (paper: https://www.cl.cam.ac.uk/~ey204/pubs/2026_OSDI_OpenTela.pdf):
  - dataset `eth-easl/swissai-serving-trace` on Hugging Face, July 2024 to October 2025, over 13 million requests;
  - per request: timestamp, model identifier, input and output token counts, sampling parameters, plus hashed 16-token prefix buckets;
  - no request durations, so concurrency must be estimated.

## Machine

- One H100 80 GB is enough for 8B models. Use a full VM (Docker's default `/dev/shm` is 64 MB).
- Add a second GPU only if you need TP 2.
- SGLang version: the one `sglang_base` in swiss-ai/model-launch pins if you can find it, otherwise the latest stable `lmsysorg/sglang` image. Record everything in `ENV.md`: SGLang, torch, CUDA, driver, GPU, servekit commit.
- Models: `swiss-ai/Apertus-8B-Instruct-2509` (in the servekit article's table), plus one FP8 mixture-of-experts model that fits one GPU, to exercise DeepGEMM and FlashInfer JIT paths.

## Tasks

### Part A: persistent caches

- [ ] Baseline with `servekit profile -- python -m sglang.launch_server --model-path <model> ...`. Three cold runs (fresh, empty cache directories) and three warm runs (same directories kept). Save `results/sglang_baseline.csv` (date, model, run, repeat, phase, seconds).
- [ ] Find every cache a cold start writes: run under `strace -f -e trace=openat,rename,mkdir`, or diff `find $HOME /tmp -newer <marker>` before and after. Map each directory to the environment variable that moves it, checking SGLang, FlashInfer, DeepGEMM, Triton, TorchInductor and TileLang in the installed versions.
- [ ] "New node" test: copy only the cache directories to a fresh location or machine, point the variables there, and start again. Measure what is skipped and what still recompiles; note anything keyed to the GPU, driver or path.
- [ ] Write-safety audit: for each cache, does it write via temp file plus atomic rename, a file lock, or neither? Stress test with eight processes compiling the same kernels into one directory at once, then load every entry and count failures.
- [ ] Deliverable: one table per cache (variable, write pattern, key, safe to share, saving), plus a proposed `sglang_base` environment change keyed by image digest and GPU architecture. Draft it as a comment for #270.

### Part B: trace-driven CUDA-graph capture sets

- [ ] Download the trace and inspect its schema. Per model: request rate over time, input and output token distributions.
- [ ] Measure on the GPU for one model: prefill time against input length, and time per output token against batch size, with and without CUDA graphs.
- [ ] Estimate the decode batch-size distribution per model by simulating the trace: each request holds a slot from arrival for prefill plus output tokens times the time per token at the current batch size. Report p50, p90, p99 and max. Label it clearly as an estimate.
- [ ] Measure the capture cost: startup time and graph memory for `--cuda-graph-max-bs` 8, 16, 32, 64, 128, 256 and a few explicit `--cuda-graph-bs` lists. Also measure the decode slowdown for batches above the largest captured size, which run without graphs.
- [ ] Combine the two: per model, the smallest capture set that keeps a chosen share (for example 99%) of estimated decode steps on graphs, with its startup saving and expected latency cost. Record which graph mode (breakable or piecewise) the version uses.
- [ ] Deliverable: a script that reads the trace and prints a recommended capture set per model, plus a one-page write-up with the trade-off plot.

## Rules

- Every number has units, its setup and a median of at least 3 runs with min and max.
- "Cold" means empty cache directories and an evicted page cache.
- Anything derived from the trace is an estimate; say so.
- Credit servekit, the OpenTela trace and the outside sources.
- GPU budget about USD 60. Stop idle instances.
