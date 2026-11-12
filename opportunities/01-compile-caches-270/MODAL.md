# MODAL.md: runbook for the #270 validation

Everything runs from this folder. Nothing has been run on Modal yet: every time and cost below is an
**estimate**. Approximate prices: H100 about $3.95/h (see modal.com/pricing).

## 0. Local checks (free)

```bash
uv venv .venv --python 3.12 && uv pip install --python .venv/bin/python torch==2.13.0 apache-tvm-ffi==0.1.11 tilelang==0.1.12 ninja pytest modal
uv pip install --python .venv/bin/python -e vendor/servekit --no-deps   # servekit profile, used by harness/ab.py
.venv/bin/python modal_app.py --dry-run            # imports, harness files, plan and cost
.venv/bin/python -m pytest -q                      # 55 tests (53 run on macOS, ~4 min), incl. the harness against a mock vLLM
```

## 1. Setup (once per account)

```bash
pip install modal && modal setup
modal secret create huggingface HF_TOKEN=<token>    # optional; Apertus and Qwen are public
```

Each account has its own Volumes, so each one downloads its own weights. That is CPU time and costs
cents: Apertus-8B is 16 GB, Qwen3-30B-A3B-FP8 31 GB, Qwen2.5-0.5B 1 GB.

## 2. What to run, split across the three accounts

| account | command | what | GPU | est. time | est. cost |
|---|---|---|---|---|---|
| any (first) | `modal run modal_app.py::smoke` | Linux self-test (strace, inotify) + vLLM image audit on Qwen2.5-0.5B | H100 | ~30 min | ~$2 |
| A | `modal run modal_app.py::plan_a` | SGLang 0.5.20: Apertus-8B TP1 (cold / servekit3 / full), TP2 (cold / full), Qwen3-30B-A3B-FP8 TP1 | H100, H100:2 | ~1.8 h | ~$9 |
| B | `modal run modal_app.py::plan_b` | vLLM 0.32.0: same matrix | H100, H100:2 | ~2 h | ~$10 |
| C | `modal run modal_app.py::plan_c` | audits of both images (+ gate of the patched env TOMLs), GPU write-safety stress (Triton, FlashInfer), TileLang CUDA JIT cold vs warm | H100 | ~1.1 h | ~$4.3 |

The total is about $22 (`python modal_app.py --dry-run` prints the breakdown). If an account runs low,
use `--repeats 3` and drop `moe_fp8` first: edit `plan_a` / `plan_b`.

## 3. What each run does

- **`ab_*`** (`harness/ab.py`)
  - Every run is a new "job": a fresh HOME and TMPDIR, and caches in a node-local overlay at
    `/tmp/sml-jit-cache`, which is sml's fixed path.
  - Persisting arms seed from and promote to the `cache-270-shared` Volume, which stands in for
    capstor. They use the same `integration/sml_jit_cache.py` that the model-launch patch embeds.
  - Each run executes `servekit profile` (eth-easl/servekit @cd8f0b8, vendored) and records:
    - its per-phase JSON;
    - time to ready and the first-request latency;
    - the cache files and binaries written per component (KB-matched).
  - Builds avoided = binaries built by the populate run − binaries rebuilt by a warm run.
  - Weights are read into the page cache before each arm. The experiment measures compile caches,
    not weight I/O.
- **`audit_*`**: `python -m cache_audit run --relocate --readonly` on a one-request warmup
  (`harness/warmup_once.py`). The tracer is strace when ptrace is allowed, else snapshots.
  - It also runs a gate pass with `--env-toml harness/envs/<engine>_base.toml` (the patched TOMLs).
    Exit 0 there means the patch redirects every cache the image writes.
  - Plus `cache_audit static`.
- **`gpu_stress`**: 8 processes, released together, JIT the same Triton kernels and the same
  FlashInfer op into one directory. Then a fresh process loads them, and the run counts failures,
  leftovers and rebuilds. Repeated 3 times.
- **`tilelang`**: 8 TileLang GEMM kernels (4 shapes × 2 tile configs), compiled cold in one process
  and warm in a new one.

## 4. Fetch and summarize

```bash
modal volume get cache-270-results / results/modal/
.venv/bin/python harness/summarize.py results/modal/ab/*.json > results/ab_summary.md
cat results/modal/audit/*/report.md results/modal/audit/*-gate/report.md
```

`results/ab_summary_MOCK_format_only.md` shows the table format. It comes from a mock run and is
not a measurement.

## 5. Limits to state with the results

- **No Lustre.** A Modal Volume stands in for capstor, so the seed and promote times are not
  capstor's. Lustre's flock and rename semantics can only be tested on Alps.
- **Not GH200.** H100 SXM is sm90 like GH200, but x86 instead of aarch64, and PCIe/NVLink topology
  differs. Cache keys that include the GPU name differ from GH200's.
- **Containers are not VMs.** The page cache can't be dropped, so weights are pre-read. The phases
  measure JIT and compile, not weight loading.
- **Not the platform's images.** `lmsysorg/sglang:v0.5.20` and `vllm/vllm-openai:v0.32.0` are the
  upstream images, not the private `sglang_base` / `vllm_base`, which are pip installs of the same
  versions:
  - the upstream vLLM image runs `flashinfer download-kernels`, and vllm_base most likely does not;
  - the SGLang image's FA3 on x86 comes from `sglang-kernel`, while on aarch64 FA4/CuTe paths JIT.
- **Small-model scale.** DeepSeek-V4.1-Flash (2 nodes) cannot run on Modal. Its saving is projected
  from the TileLang per-kernel numbers and labelled as a projection.
