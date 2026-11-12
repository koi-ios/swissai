# STATUS: 04 lazy CUDA-graph capture

12 Nov 2026.

## Done

- **SGLang v0.5.20 patch** (`patches/0001-sglang-v0.5.20-lazy-cuda-graph-capture.patch`, hooks in `patches/HOOKS.md`).
  - It applies with `git apply` to a fresh `--depth 1 --branch v0.5.20` clone (94602c9).
  - All changed files compile; `ruff` (F, E9), `ruff format` and `isort` are clean.
  - The default behaviour is unchanged unless `initial_bs` is set.
- **Bugs caught by review before any GPU run:**
  - the radix-cache hazard in deferred prefill capture (fixed with an index reset);
  - the BCG output buffer reallocated per session (fixed);
  - draft runners that skip the base `__init__` would have hit an `AttributeError` (fixed with a class default);
  - a tqdm `set_description` crash on the no-progress path (fixed).
- **Policy tool and simulator** (`lazygraph/`):
  - cost models fitted to the lab's Clariden capture logs;
  - EV ranking with a λ/H startup threshold estimated per model;
  - three-arm trace simulation;
  - sweeps over eager-penalty scale (×0.5–×5, prefill ×30) and idle rules.
  - Results are in `results/sim/` and `results/policy/`.
- **Tests:** 34 pass (`pytest tests`), also against the clean patched tree (`LAZYGRAPH_SGLANG_ROOT=<tree> pytest tests`).
- **Modal app:**
  - `modal_app.py` passes `python modal_app.py --dry-run` (every mode against `bench/mock_server.py`; mock output in `results/dryrun/`).
  - Runbook and costs are in `MODAL.md` (~$12 at TP1, +$11 at TP2).
  - The `lmsysorg/sglang:v0.5.20` image exists on Docker Hub.
- **Related work**, read from primary sources by two research agents: `research/related_work_papers.md` and `research/related_work_engines.md`. The closest prior work is vLLM PR #57899 (open; post-wake idle recapture, largest first).
- **Write-ups:**
  - `WRITEUP.md`;
  - `drafts/sglang_rfc_issue.md` (not posted);
  - `drafts/vllm_design_note.md` (Model Runner V2 hooks at v0.33.0, taken from the research digest; line numbers not re-checked by hand).

## Not done or blocked

- **No GPU run.** No Modal account is configured here.
  - Every latency or speedup number in WRITEUP §4 is a prediction.
  - The eager penalties in particular are modelled, not measured.
- **The patch has never been imported with real SGLang.** It needs CUDA wheels. Static checks pass, but runtime behaviour is untested, in particular:
  - the attention-backend per-bucket state for buckets added after serving starts;
  - capture inside the scheduler's stream and inference-mode context;
  - the gc pause from `freeze_gc` in each step.
- **vLLM:** a design note only, no patch.
- **Capture under load** (stalling a busy server to capture a hot bucket) is designed but not wired. Promotion only reorders the idle queue.
- **Simulation simplifications:** no feedback from arm latency to the timeline, and in-flight counts used as batch sizes.

## Next 3 actions

1. **Run the Modal smoke test**, then the arms.
   - `modal run modal_app.py --mode smoke`, then `--mode arms --repeats 3`.
   - Confirm `Lazy CUDA graph capture:` lines, the one-step wait bound and equal steady-state throughput.
   - Fix whatever the first real run breaks.
2. **Run the `eager` and `correctness` modes.**
   - Feed `eager_penalty.json` into `python -m lazygraph.cli policy --penalty ...` and re-run the sweep with measured penalties.
   - Check that the outputs are identical, including the prefix-cache probe.
3. **Ask Xiaozhe for a Clariden slot** (about 1 node-hour) to replicate the 70B TP4 cold-start breakdown with and without lazy capture. Combine with opportunity 01 (#270 persistent caches), since the 16–18 s one-off is what remains.
