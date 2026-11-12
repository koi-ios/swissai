# Modal runbook: 05 Python start-up

Everything runs on the exact upstream images: `lmsysorg/sglang:v0.5.20` and `vllm/vllm-openai:v0.33.0`. Nothing is installed into them; our code is mounted at `/opt/pystartup-src`. The no-pyc, timestamp and unchecked-hash arms are derived in the container with `PYTHONPYCACHEPREFIX`. That covers both the platform's pip-built images (pycs) and its uv-built images (none).

## Setup (once per account)

```bash
pip install modal && modal setup
```

```bash
cd opportunities/05-python-startup && python modal_app.py --dry-run
```

## Runs

| step | command | resources | time | approx USD |
|---|---|---|---|---|
| 1 | `modal run modal_app.py::smoke` | 2 × 2 vCPU | 3 min | 0.02 |
| 2 | `modal run modal_app.py::cpu_study --engine sglang` | 8 vCPU / 16 GiB | 30 min | 0.30 |
| 3 | `modal run modal_app.py::cpu_study --engine vllm` | 8 vCPU / 16 GiB | 25 min | 0.25 |
| 4 | `modal run modal_app.py::prewarm_study` (needs step 2's manifest) | 9 single-use containers | 20 min | 0.25 |
| 5 | `PYSTARTUP_GPU=L40S:2 modal run modal_app.py::gpu_e2e` | 2 GPUs, 16 vCPU | 35 min | 2.5 (H100:2: about 5) |

Total is about $3.3, well under the $15 cap, so one GPU retry fits. Split: account A runs steps 1–4 (about $1); account B runs step 5. Volumes are per account, so step 4 must use the account that ran step 2. Prices are approximate; check modal.com/pricing.

## What each step settles

- **cpu_study** covers leads 1–4 on the real image.
  - pyc audit of the image as shipped.
  - Per role (SGLang main / scheduler / detokenizer; vLLM main / engine_core / worker): import time with warm pycs, without pycs (read-only case) and with `PYTHONPYCACHEPREFIX` cold then warm. Each is split into compile / read / unmarshal / extension / exec, with modules and files compiled.
  - Fix (a): unchecked-hash compileall time and size, for the import set and for the whole image.
  - Fix (b): spawn vs zygote vs fork, using the scheduler's import set and 4 workers.
  - Zygote calibration: how much of the scheduler's import set preloads cleanly under emulated sm90 facts.
- **prewarm_study** covers fix (d).
  - Each arm runs in a fresh container: `cold`, `prewarm-then-import` and `prewarm-concurrent`, 32 threads.
  - Caveat: Modal's lazily-loaded image stands in for the `.sqsh` on Lustre. Modal may cache image files per host, and gVisor may make `mincore` / `fadvise` meaningless. `residency_at_start` and `second_import_wall_s` are recorded so this can be judged; report it honestly.
- **gpu_e2e**: TP2 with `Qwen/Qwen2.5-0.5B-Instruct`, CUDA graphs off to keep launches short.
  - **Proof of fork safety**: `zygote_smoke.txt` holds the zygote status with real NVML facts. Check that `torch_cuda_initialized` is false and `nvidia_fds` is empty. Its forked children then initialise CUDA themselves, which would fail if the zygote had.
  - Arms are interleaved, 3 repeats each:
    - `spawn` (stock);
    - `zygote` (default list);
    - `zygote-cal` (calibrated on the node);
    - `spawn-nopyc`;
    - `zygote-cal-nopyc`.
  - Metrics: Popen → `server_args=`, `server_args=` → last TP `Init torch distributed begin`, → ready, and greedy outputs, which must be identical.

## Fetch and summarise

```bash
modal volume get pystartup-results / results/modal/
```

- `cpu_study_<engine>.json`: see `imports.<role>.summary`, `compileall`, `spawn.summary` and `calibration`.
- `prewarm_*.json`.
- `e2e_*/e2e.json`: see `summary` per arm.

## Risks

- `lmsysorg/sglang:v0.5.20` is built for CUDA 13.0, so the GPU host needs driver ≥ 580. `gpu_e2e` records `nvidia-smi` first. If the driver is too old, set `PYSTARTUP_SGLANG_IMAGE` to a CUDA 12 build of the same version, if one exists.
- Containers have a small `/dev/shm`, so `e2e.py` sets `NCCL_SHM_DISABLE=1` when it is under 1 GiB. Custom all-reduce is disabled.
- There is no Lustre or squashfs on Modal, so the cold-image numbers are an analogue, not the Alps value. The Alps validation is the same `python -m pystartup.e2e` inside the `sglang_base` container under srun.
