# Cold-start and serving work for the Swiss AI platform: what was built

Nine projects. Each takes one open problem from the EASL / Swiss AI serving team's own issues, logs and trace, and each has:
- related-work research;
- an implementation with tests;
- patches and drafted texts ready to send;
- a Modal runbook;
- a `WRITEUP.md` (at most 2 pages) and a `STATUS.md`.

Built 11–12 Nov 2026, one session per folder. Facts and rules shared by all of them are in [SHARED_CONTEXT.md](SHARED_CONTEXT.md); the evaluation they came from is [../sglang-cold-start/EVALUATION.md](../sglang-cold-start/EVALUATION.md).

**Status of the numbers.** Everything below ran on a laptop (CPU, macOS) or comes from the lab's own logs. No GPU and no Alps job has run yet. Any GPU saving is a **prediction** until its Modal plan (below) or a Clariden job measures it. Nothing has been pushed, posted or sent.

<!-- verification:start -->
## Checked independently (12 Nov, coordinating session)

| Check | Result |
|---|---|
| Every test suite, rerun from scratch on this Mac | 01: 53 pass (2 skipped, Linux only) · 02: 187 servekit + 8 tooling · 03: 195 servekit (3 skipped) + vLLM harness self-test 57/57 · 04: 34 · 05: 41 · 06: `go vet` clean, `go test -race ./...` 17/17 packages · 07: 183 · 08: 176 servekit + 9 reproductions against SGLang 0.5.20 · 09: 37 |
| The 2 errors in the 02, 07 and 08 servekit runs | Existing in unpatched servekit main too: two stager tests need Linux `os.O_DIRECT`, which macOS lacks. 03's first patch turns them into skips. |
| Patches on fresh upstream clones | servekit main (cd8f0b8): 02, 03, 07 and 08 each apply. OpenTela main (4b6e90f): 06 applies, with or without the optional patch. model-launch main (02b74ce): 01 and 05 apply, alone and together. SGLang v0.5.20: 04, 05 (zygote and Dockerfile), 08 and 09 apply, alone and all five together. |
| Patches that conflict | servekit 02, 03 and 08 conflict pairwise in `launch.py` / `prepare.py` (02 vs 08 also in `cli.py` and `save_sharded_state.py`); 07 stacks with each. **Submit 02 → 08 → 03, rebasing each on the last; 07 can go any time.** |
<!-- verification:end -->

## The nine, in one line each

| # | Folder | Problem (whose) | What was built | Headline |
|---|---|---|---|---|
| 01 | [01-compile-caches-270](01-compile-caches-270/) | model-launch #270: no compile cache survives a job (Aryan) | A `cache-audit` tool and knowledge base, 4 model-launch patches, a CI pre-warm design | One cache per image + GPU arch, shared by all checkpoints; every cache variable verified against source |
| 02 | [02-servekit-cache-fix](02-servekit-cache-fix/) | servekit's JIT cache no longer covers SGLang 0.5.20 | 3 servekit patches + regression test | Restores the ~17 s all-reduce JIT per TP4 launch (predicted); future cache moves become a warning |
| 03 | [03-servekit-vllm](03-servekit-vllm/) | servekit roadmap: "Support vllm fast weight loading" | servekit for `vllm serve` as a plugin (no vLLM fork) + harness arms | vLLM's own Lustre prefetch still took 231 s on capstor (lab log); servekit's path should take seconds (predicted) |
| 04 | [04-lazy-cuda-graphs](04-lazy-cuda-graphs/) | Graph capture is 35–47% of a post-servekit cold start | SGLang 0.5.20 patch: traffic-chosen startup set, the rest captured in idle time, in lockstep across TP | ~−32% cold start on 70B TP4 (117 → ~80 s), no request delayed by more than one step (predicted) |
| 05 | [05-python-startup](05-python-startup/) | 25–30 s before `torch.distributed` init, which nobody has looked at | Import profiler, a CUDA-safe worker zygote, bytecode fixes for the images | Zygote cut 4-worker start 8.5 → 4.2 s locally; 9–13 s per launch on Alps (predicted) |
| 06 | [06-opentela-73](06-opentela-73/) | OpenTela #73: a restarted worker is unroutable for minutes (Xiaozhe) | Root cause and 2 Go patches (+1 optional) with a deterministic regression test and an e2e harness | Baseline never routable within 180 s → routable in 15–20 s (6–9 s with the optional patch), measured locally |
| 07 | [07-verify-tolerance](07-verify-tolerance/) | `servekit verify`'s 1e-6 tolerance fails correct low-precision loads (Youssef's open question) | Calibrated verify (exact where it reproduces, statistical where it can't) + a checkpoint fault injector | CPU: 63/63 faults larger than serving noise caught at ~1.6% false alarms; FP8 must be verified exactly |
| 08 | [08-preshard-roundtrip](08-preshard-roundtrip/) | Presharding silently changes MLA, MXFP4 and Gemma models (servekit #19 and #9) | "Raw preshard" (replays each rank's checkpoint bytes through SGLang's own loader) + a deep-state checker | Correct by construction; 9 reproductions against 0.5.20's own code; servekit's MLA backport had silently stopped applying |
| 09 | [09-barrier-vast](09-barrier-vast/) | model-launch-images #22: 8-node TP32 launches fail SGLang's post-load barrier; VAST readiness | Barrier analysis, gloo reproduction, a timeout fix that needs no rebuild, a storage benchmark | The failing launches die at a hard-coded 480 s barrier; ranked hypotheses and the logs that decide them |

## What each one did

### 01: per-image compile caches for both engines (#270)

**Problem.** On vLLM, DeepSeek-V4.1-Flash takes about 70 min to become ready, about 40 of them in TileLang JIT, and GLM-5.3 takes about 60 min. On SGLang, a 70B TP4 launch spends about 54 s compiling. Nothing persists between jobs.

**Built.**
- `cache_audit/`: runs a cold launch with fresh directories, traces every file the engine writes, maps each cache root to the variable that moves it (verified by re-running), and gates CI on any root left unredirected.
- A knowledge base for SGLang 0.5.20 and vLLM, cited to file:line.
- Write-safety stress tests and an atomic promotion protocol.
- 4 model-launch patches: one cache per image digest and GPU arch with a node-local overlay, and the env TOML changes, plus a variant that applies after #272.
- Drafts: a #270 comment, the env diffs, and a design for pre-warming the cache in the images CI.

**Findings.**
- On SGLang 0.5.20, servekit's cache list misses the 17 s all-reduce JIT.
- vLLM keys compile caches by checkpoint path, so each new checkpoint recompiles.
- Measured on CPU with real libraries: warm runs rebuilt 0 binaries, and 8 concurrent writers produced 0 failures.

**Next.** Modal plans A–C (~$25), then one Clariden A/B.

### 02: servekit's JIT cache on SGLang 0.5.20

**Problem.** SGLang 0.5.20 moved `load_jit`'s build cache to `SGLANG_JIT_CACHE_DIR`. servekit doesn't persist it, so every TP4 launch rebuilds the custom all-reduce, about 17 s in the lab's own 0.5.16 A/B.

**Built.** Three servekit patches:
- persist `SGLANG_JIT_CACHE_DIR` and `SGLANG_CACHE_DIR`;
- key each cache by the environment that makes it valid, and prune old keys;
- report any JIT output the engine leaves outside the persisted roots, with an optional strict mode.

They come with tests, including a regression test that fails on main, and a PR draft that credits #18. They merge cleanly with open PR #26.

**Next.** A $6 Modal A/B (TP2, 3 runs per arm).

### 03: servekit for vLLM

**Problem.** This is servekit's roadmap item and the user's original plan. Every model added since September runs on vLLM. The lab's capstor log (vLLM 0.24, Llama-70B TP4) shows auto-prefetch still spent 231 s loading weights.

**Built.**
- `servekit prepare` / `launch` for `vllm serve`, with overlap through a vLLM loader plugin (no fork).
- A guard for models presharding can't handle, per-rank load records, and cache write-back.
- 59 new tests, and new arms in the `serve-kit-vllm-prelim` harness (57/57).
- A decision table for which vLLM-native loader to use when. For example, `vllm preload` + `ipc_cache` for restarts on the same GPUs, and vLLM's own loader for quantized and MLA models.

**Next.** A ~$14 Modal run, but Modal has no Lustre, so the decisive measurement is a Clariden job.

### 04: traffic-aware, lazy CUDA-graph capture (the research piece)

**Problem.** After servekit, graph capture is 35–47% of a Clariden cold start. The trace shows most capture sizes go unused for minutes, and models sit idle 78–99% of the time.

**Built.**
- An SGLang 0.5.20 patch. It captures a set chosen from traffic at startup, captures the rest one size at a time while the scheduler is idle, keeps TP ranks in lockstep, and pauses whenever a request waits.
- A policy tool that writes `--cuda-graph-config` per model from the trace.
- A trace-driven simulator.
- 34 tests.
- Drafts: an SGLang RFC and a design note for vLLM.

**Predicted from measured per-size costs.**
- 70B TP4 cold start 116.8 → ~80 s.
- Capture-caused latency 5–28× lower than default capture across the trace's launches.
- At most one capture step (≤ 1.35 s) of delay per request.

**Next.** A ~$12 Modal run (Apertus-8B: default vs a fixed set vs lazy, eager-fallback costs, greedy-output equality).

### 05: Python start and TP worker spawn

**Problem.** Before `torch.distributed` init, a launch spends 25–30 s on Clariden and 33–79 s on Bristen, in the lab's logs. The causes are cold image reads, missing bytecode in uv-built images, and `spawn` re-importing everything for each TP worker.

**Built.**
- A profiler for the real images.
- A worker zygote that preloads CUDA-free modules and forks the workers. It comes with checks that CUDA is never initialised before fork, and as a wrapper that needs no SGLang change.
- Patches that add bytecode compilation to the image builds.
- 41 tests.

**Measured locally.**
- 4 workers ready: spawn 8.48 s vs zygote 4.19 s.
- Imports on a read-only root without bytecode: 2.13 s vs 0.97 s with it.

**Predicted on Alps.** 9–13 s saved per launch.

**Next.** A ~$3 Modal run, then one Clariden job.

### 06: OpenTela #73, a restarted worker unroutable for minutes

**Root cause.** A restart keeps the peer ID but wipes the CRDT store. The worker's new rows land at DAG heights 1, 2, … while every replica already holds its old LEFT row at height H, so `setValue` drops them silently. It is a regression: #51 removed the periodic re-announce that used to hide it.

**Fix.**
- Patch 1: a Lamport rule in `addDAGNode`.
- Patch 2: a self-row guard based on incarnation.
- Optional patch 3: list a rejoined peer at once when the head holds a direct connection to it.

It comes with a deterministic regression test (fails 10/10 before the fix, passes 10/10 after) and stays compatible with PR #72.

**Measured on localhost.** Baseline: never routable within 180 s. Fixed: 15–20 s; with patch 3: 6.5–8.6 s.

**Next.** Send the PR (draft ready) and confirm on the private mesh. No GPU needed.

### 07: calibrated correctness checks for low precision

**Problem.** `servekit verify`'s fixed 1e-6 tolerance fails correct loads across TP degrees and hardware. Any hand-set tolerance would hide real faults. Youssef's write-up leaves this open.

**Built.**
- Verify now measures its own noise floor on the trusted server. It stays exact where the engine reproduces bitwise and uses a calibrated test where it can't, next to weight checksums.
- A fault injector for safetensors.
- 183 tests.

**Measured on CPU** (Qwen3-0.6B in BF16, FP8 and MXFP4, 147 injected faults):
- 63/63 faults larger than serving noise caught, at ~1.6% false alarms.
- FP8 noise is 17× BF16's, which hides FP8 scale faults, so FP8 must be verified exactly.

**Next.** A ~$15 Modal study on real engines.

### 08: presharding that can't silently change the model

**Problem.** servekit saves the model after SGLang has transformed it. Three transforms are lost on 0.5.20: MLA aliasing, MXFP4 repacking, and Gemma norms. servekit's MLA backport had silently stopped applying.

**Built.**
- **Raw preshard.** Records the exact bytes each rank reads during one normal load, then replays them through SGLang's own loader, checked on every load.
- **A deep-state checker.** Fingerprints every tensor the model holds, which catches what a `state_dict` check misses.
- Guard fixes.
- 176 servekit tests, and 9 reproductions against 0.5.20's own code.

**Measured on CPU.** Raw preshard costs ~5–50 µs per tensor at load.

**Next.** A ~$10–15 Modal run on DeepSeek-V2-Lite (MLA) and gpt-oss-20b (MXFP4).

### 09: the TP32 post-load barrier (#22), and VAST

**Barrier (#22).** The failing 8-node launches die at `monitored_barrier(timeout=480 s)`, which no flag reaches. Ranked hypotheses:
1. The node's checkpoint footprint exceeds DRAM: 1.30× for the failing configuration vs 0.78× for those that pass.
2. A slowdown introduced in 0.5.20.
3. Plain spread between nodes (least likely).

The failing logs already decide between them.

Fixes:
- now, without a rebuild: a `sitecustomize` hook;
- upstream: an env-variable patch;
- shrink the footprint: `--ep-size 32` (predicted 3.2× fewer bytes per node) or servekit staging.

**VAST.**
- Researched the client (NFS readahead limits vs Lustre's own readahead engine).
- Predicted which servekit techniques still pay off there.
- Built `bench_storage.py` with a runbook: 15 minutes per filesystem, tested on this Mac.

37 tests. No GPU needed.

## Modal plan for three $30 accounts (≈ $88 if everything runs)

Run the cheapest experiments with a decisive answer first, and check each result before spending more. Volumes are per account, so an account that runs a later step must also have run that plan's download step. Each folder's `MODAL.md` has the exact commands.

| Order | Account | Experiment | Cost (est.) | What it decides |
|---|---|---|---|---|
| 1 | A | 02 `gpu_check` + `experiment` | ~$6 | The 0.5.20 cache drift: seconds saved by the servekit fix |
| 2 | A | 05 steps 1–5 | ~$3.3 | Zygote and bytecode savings in the real SGLang image |
| 3 | A | 04 `smoke`, `arms`, `eager`, `correctness` | ~$12 | Lazy capture: startup, request delay, output equality |
| 4 | B | 01 `smoke`, `plan_a` (SGLang), `plan_c` (audits, write safety) | ~$15 | #270 for SGLang, measured cold vs warm |
| 5 | B | 08 `smoke` + `dsv2-lite` | ~$8 | MLA raw preshard on a real GPU |
| 6 | C | 07 study (smoke + Qwen3-1.7B BF16/FP8 + gpt-oss-20b) | ~$15 | Calibrated verify on real engines |
| 7 | C | 08 `gpt-oss-20b` | ~$7 | MXFP4 raw preshard |
| 8 | C (if left) | 01 `plan_b` (vLLM) or 03 `experiment_tp1` | ~$7–10 | vLLM halves; 03's decisive test is on Lustre, not Modal |

06 and 09 need no GPU.

## For the email

Lead with three, because they are concrete, small, and on the team's own open issues:
- **06:** a root-caused fix for Xiaozhe's own OpenTela #73, with a regression test.
- **02:** servekit's cache silently stopped working on the platform's SGLang.
- **01:** #270 for both engines.

Then offer **04** (lazy, traffic-aware graph capture) as the research direction, using **05**, **07** and **08** as evidence of depth. Each folder's `WRITEUP.md` ends with a three-line pitch.

## Housekeeping

- `vendor/` folders hold the clones and build trees.
- `06-opentela-73/toolchain/` holds a folder-local Go toolchain.
- `06-opentela-73/e2e/runs/` holds about 500 MB of trial logs and can be deleted.
- `07-verify-tolerance/models/` holds small CPU test models (about 3 GB with vendor).
- Total is about 11 GB.
- Nothing outside `opportunities/`, `data/` and the vLLM harness was changed.
