# Runbook: servekit for vLLM on Modal

Nothing here has been run yet. It needs a Modal account, which this machine does not have.

## What it measures

The question: on a slow network store, how long does vLLM 0.33.0 take to load Apertus-8B-Instruct-2509 with each loader?

- **Setup.** TP 1 and TP 2, cold runs: page cache evicted, compile caches empty.
- **Arms.**
  - vLLM's own options: default/auto-prefetch, lazy, eager, prefetch with 8 or 32 threads, `instanttensor`, `fastsafetensors`, `runai_streamer`, `sharded_state`, `runai_streamer_sharded`;
  - the harness's own `/dev/shm` staging (`staged_shm`);
  - the patched servekit, serial (`servekit_serial`) and with `--overlap` (`servekit_overlap`).
- **Correctness.** Every arm runs `servekit verify` against the default loader's logprobs. The default loader also checks itself against its own reference first.

The arms come from `serve-kit-vllm-prelim/run_arms.sh`. The image is `vllm/vllm-openai:v0.33.0`, plus the patched servekit from `vendor/servekit` (installed with pip, so vLLM finds its plugin) and the harness.

## Limits, stated up front

- **No Lustre on Modal.** The slow store is a Modal Volume and the control is the container's local disk. Speeds and the Volume's own caching are Modal's. Read the results as the mechanism and the code working, not as Alps or capstor speedups.
- **`/dev/shm` may be small.** The container's `/dev/shm` size is not documented. If it cannot hold the staged copy plus the stager's 10 GiB headroom, staging goes to local disk instead. `modal_run.json` then says `shm_fallback: true`, and the staged arms are disk-staged, not RAM-staged.
- **Cold is not fully cold.** `evict.py` drops this container's page cache, but no one can clear the Volume's server-side cache. `evict_verify.json` records how much slower a read after eviction is, and whether a second cold read gets faster, which would mean the store is caching.
- **Small scale.** One node, at most 2 GPUs, one 8B model. Multi-node staging, the main point for large models, cannot be shown on Modal.

## Commands

```bash
cd opportunities/03-servekit-vllm
python modal_app.py --dry-run          # local check: inputs, plan, cost estimate; no Modal account needed
modal token new                        # once, in the browser
```

The model is gated: accept its licence on Hugging Face first. Then store a token as a Modal secret named `huggingface` (with `HF_TOKEN`), or run with `SK_HF_SECRET=0` if the model is public to you.

| step | command | GPU | expected time | est. USD |
|---|---|---|---|---|
| weights into the `hf-cache` Volume | `modal run modal_app.py::download` | none | 10–15 min | 0.06 |
| harness selftest on Linux | `modal run modal_app.py::selftest` | none | 2–5 min | 0.03 |
| smoke: TP 1, 1 repeat of `default_net`, `servekit_serial`, `servekit_overlap` | `modal run modal_app.py::smoke` | 1× A10G | 15–25 min | 0.65 |
| every arm, TP 1, 3 cold repeats, plus staging sweep and Day-1 baseline | `modal run modal_app.py::experiment_tp1` | 1× A10G | 2.5–3.5 h | 5.6 |
| every arm, TP 2, 3 cold repeats | `modal run modal_app.py::experiment_tp2` | 2× A10G | 2–3 h | 7.4 |

- The total estimate is about **$14**: GPU at about $1.10/h per A10G, plus about $0.76/h for the 8 CPUs and 48 GiB each GPU container reserves. Check modal.com/pricing first.
- `SK_GPU=L40S` raises the estimate to about $22. The GPU type barely matters here: loading is bound by storage, CPU and RAM.
- Run the smoke first, and continue only if it passes. A passing smoke has all three arms in `arms.csv`, `verify.csv` showing a pass for both servekit arms, and both of these lines in the servekit logs:
  - `Registered model loader ... with load format sharded_state`;
  - `[SERVEKIT] weight load per rank`.

## Split across accounts

Volumes are per account, so each account downloads the model once.

- **Account A** (about $6.5): `download`, `selftest`, `smoke`, `experiment_tp1`.
- **Account B** (about $7.5): `download`, `experiment_tp2`.
- **Account C** is a spare: re-run a repeat that failed, or run `SK_GPU=L40S` for a second GPU type.

## Results

Each run writes `/<stamp>_<label>/` to the `sk-vllm-results` Volume, containing:

- `arms.csv`, `verify.csv`, `staging.csv`, `baseline.csv`;
- `runs.jsonl`, `env.json`, `modal_run.json`, `evict_verify.json`;
- `summary.md` and `logs/`.

Fetch and summarize them:

```bash
modal volume get sk-vllm-results / results/modal
RESULTS=results/modal/<stamp>_arms-tp2 python ../../serve-kit-vllm-prelim/summarize.py
```

Then copy the headline rows into `WRITEUP.md`'s results table, labelled "Modal Volume, A10G".

## If something fails

- **Image build fails on an extra.** It only skips that arm; `detect_options.py` records which loaders exist.
- **`servekit_*` arms skip with "no servekit artifact".** `run_servekit_prepare.sh` failed; see `logs/servekit_prepare.log`. A guard refusal does not apply here, since Apertus-8B is bf16 and not MLA.
- **A servekit arm fails verify.** Keep it: that is a result. Look for `[SERVEKIT] refusing to serve`, which means a narrowing or cast load, and send the log.
- **Out of GPU memory at TP 1.** Lower `--gpu-memory-utilization` or `--max-model-len` in `VLLM_EXTRA_ARGS` at the top of `modal_app.py`.
