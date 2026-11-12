# Modal runbook: JIT-cache A/B on SGLang 0.5.20

This experiment turns the claim behind the patch into a measurement. The claim: on SGLang 0.5.20, servekit at cd8f0b8 keeps Triton and FlashInfer but loses `load_jit`, so every launch rebuilds the custom all-reduce during `torch.distributed` init, and the patch fixes that.

Nothing here has been run yet. No Modal account is configured on this machine. Everything that works without a GPU was validated locally (see "Validated locally").

## What runs

- **Setup:** `modal_app.py`, one app, one 2×H100 container.
- **Image:** the stock `lmsysorg/sglang:v0.5.20`. Two servekit trees are added to it:
  - `/opt/servekit-main`: upstream at cd8f0b8;
  - `/opt/servekit-patched`: cd8f0b8 plus `patches/*.patch`, applied with `git am` at image build. So the run tests the exact deliverable.
- **Model:** `Qwen/Qwen2.5-1.5B-Instruct`, TP2.
  - It is small, ungated and TP2-divisible. The all-reduce JIT does not depend on model size, so a small model keeps cost low.
  - Override with `MODEL=...`.
- **Why H100 SXM:** sglang's custom all-reduce v2 only comes up on a full-NVLink group (`custom_all_reduce_v2.py:482-497` → `custom_all_reduce_utils.py:442`). The all-reduce is the kernel whose JIT is in question.

| arm | servekit | caches in the artifact |
|---|---|---|
| `none` | patched | none (the patched artifact's shards, no `servekit-caches/`), so cold every run |
| `main` | cd8f0b8 | what main's `prepare` keeps (`triton`, `tvm-ffi`, `flashinfer`) |
| `patched` | patched | its keyed namespace (all 11 variables) |

**Job order** (from `python modal_app.py --dry-run`):
- 2 prepares, one per artifact.
- 1 unrecorded priming launch.
- 9 measured launches, the three arms interleaved and rotated across 3 rounds.

**Each job is a new SLURM job on a new node** (`tools/run_arm.py`):
- a fresh `$HOME` and `$TMPDIR`, and an empty `/tmp/servekit/caches`;
- nothing carried over except the artifact;
- the server is stopped once ready, as the lab's sbatch scripts did.

**What is recorded per job:**
- servekit's report and the engine log;
- `torch.distributed` init;
- the first prefill-capture step (the tqdm rate at 1/N, to 0.01 s);
- the first request (servekit's `warmup_request(JIT)`);
- whether custom all-reduce v2 came up;
- an inventory of the compiled output the job left in its own `$HOME` and temp dir. That output is exactly what the next job loses.

## Steps

```bash
pip install modal && modal setup
```

```bash
python modal_app.py --dry-run
```

```bash
modal run modal_app.py::download
```

```bash
modal run modal_app.py::smoke
```

```bash
modal run modal_app.py::gpu_check
```

```bash
modal run modal_app.py::experiment
```

```bash
modal volume get servekit-jit-results <stamp> results/modal/
```

```bash
python tools/summarize_modal.py results/modal/<stamp>/runs.jsonl --out results/modal/<stamp>/summary.md
```

| step | hardware | time | cost (estimate) |
|---|---|---|---|
| `--dry-run` | local | seconds | $0 |
| `download` | CPU | ~2 min | < $0.05 |
| `smoke` (first call also builds the image: ~20 GB pull + 2 clones) | CPU | 10–20 min | < $0.20 |
| `gpu_check` | 2×H100 | ~2 min | ~$0.30 |
| `experiment` | 2×H100 | ~40 min (hard stop 60 min) | ~$5.40, at most $7.90 |
| **total** | | | **~$6, at most ~$8.50** |

Prices assume $3.95 per H100-hour; check modal.com/pricing before running. One account is enough. If a run dies midway, `runs.jsonl` keeps every finished job (it is committed after each one). Rerun on a second account; that account needs its own `download`.

## Check before trusting the numbers

- **`gpu_check`:**
  - The driver must run the image's CUDA 13.0 (driver ≥ 580). If it doesn't, set `SGLANG_IMAGE` to a CUDA 12 build of 0.5.20 and say so in the write-up.
  - `nvidia-smi topo -m` must show `NV#` between GPU0 and GPU1.
  - The p2p check must print `True`.
- **The summary's line "Custom all-reduce v2 enabled in every run" must say True.** If not, the dist-init comparison measures nothing; stop and report that.
- **`smoke`** runs servekit's unit tests on Linux. These include the two stager tests that cannot run on macOS.

## What to expect (predictions, not measurements)

- **`torch.distributed` init:**
  - `none` ≈ `main`, both paying the all-reduce JIT. At TP4 on 0.5.16 the JIT took ~17 s; TP2 on H100 may differ.
  - `patched` should be ~2–3 s.
- **Only `main`'s jobs should leave `~/.cache/sglang/jit` behind.** That directory is the kernels main loses on 0.5.20.
- **First prefill step and first request:** `main` should sit between `none` and `patched`, because it still keeps Triton and FlashInfer. How far between is what the run tells us.

## Validated locally

- **`python modal_app.py --dry-run`** prints the 12 jobs and the cost (`results/modal-dry-run.txt`).
- **`import modal_app`** works with modal 1.6.4, and all four functions register.
- **`tests/test_tools.py`** (8 tests, `.venv/bin/python -m pytest tests -q`):
  - The parser reproduces the lab's own 0.5.16 A/B from its logs:
    - `torch.distributed` init 19.54 s vs 2.74 s;
    - first prefill-capture step 28.49 s vs 5.69 s.
  - `tools/run_arm.py` runs end to end with a fake `sglang.launch_server` against both servekit trees. Under main, the fake 0.5.20 kernel lands in `$HOME` and is listed as left behind. Under the patch, it lands in the node-local namespace.

## Limits

- **Modal is not Clariden:**
  - H100 SXM on x86 instead of GH200 on aarch64;
  - TP2, not TP4;
  - no Lustre: the artifact sits on local disk, and the weight stage is a plain copy in every arm. Weight loading is therefore not compared, and the total excludes the stage.
- **All jobs share one container.** The OS page cache stays warm across jobs, which helps process start equally in every arm. Each job's caches still start empty, apart from the arm's artifact.
- **3 runs per arm,** reported as median [min–max]. That is enough to separate a ~15 s effect from noise, not to resolve a 1 s one.
