# 05 · Python start-up before torch.distributed init: status

Updated 12 Nov 2026. The findings, the related work and the pitch are in WRITEUP.md; the run plan is in MODAL.md.

## Leads: verdicts

| lead | verdict | evidence |
|---|---|---|
| spawn makes each child re-import torch, transformers and SGLang | **Confirmed, with one correction.** The TokenizerManager runs in the main process; the spawned children are the schedulers (1 per TP rank) and 1 detokenizer. The main process already imports the scheduler tree. Workers take 11–16 s on every launch (logs). | `engine.py` `_set_envs_and_config`, `http_server.py:78`; `results/log_timeline.md` |
| the image is a squashfs on Lustre | **The path is confirmed** (`sglang_base-{arch}.sqsh` on capstor). The mount mechanism is not documented. | model-launch envs; CSCS docs (`writable` defaults to true, one container per node, local images not cached) |
| SyntaxWarning means modules are compiled at launch | **Confirmed, and refined.** 44 of 72 logs; once per process that compiles first, in the critical path. The image ships no valid pyc; the root is writable. Logs without it are a v0.5.10 image with no wrapper pre-import. | logs + Dockerfiles |
| uv doesn't compile | **Confirmed, and it hits the platform.** Every model-launch vLLM image is uv-built with no pycs (local check: uv left 1,003 of 1,291 modules without a pyc). Upstream `lmsysorg/sglang` deletes its pycs. | `images/*/Dockerfile`; `results/local_imports_proxy_macos.json` |
| (new) the 25–30 s is a lower bound | **Measured.** Youssef's `sitecustomize` pre-imported torch and SGLang before servekit's clock started. Without it, the main process alone takes 28–64 s. | `results/log_timeline.md` |
| vLLM has the same cost | **Partly.** It forks by default, so there is no full re-import per worker, but the uv images mean every process compiles. Import numbers on the real image are pending (`cpu_study --engine vllm`). | `vllm/utils/system_utils.py` |

## Done

- `analysis/log_timeline.py`: splits the start-up phase from the raw logs.
- `src/pystartup/`, standard library only:
  - **profiler** (`profile_startup`, `importprof`, `_importprobe`) and `pycheck`;
  - `spawnbench`;
  - fix (b), the **zygote** (`zygote`, `zygote_preload`, `cudafacts`), and the **SGLang wrapper** (`sglang_launch`);
  - fix (d), `prewarm`, plus residency and eviction helpers;
  - fix (c), `PYTHONPYCACHEPREFIX`, via `--pycache-prefix` and the profiler's prefix modes;
  - `tracer/`, the process-tree tracer;
  - `e2e` launcher.
- **Tests**: 41 pass locally (`.venv/bin/python -m pytest -q tests`).
- **Patches** (see `patches/README.md`):
  - SGLang worker zygote (applies to v0.5.20, tested in a stand-in package);
  - SGLang Dockerfile bytecode step;
  - model-launch bytecode steps plus `SML_SGLANG_ENTRYPOINT`.
- `modal_app.py` (`--dry-run` validated) and MODAL.md: about $3.3 in total against a $15 cap.
- WRITEUP.md, and `drafts/model-launch-bytecode-note.md`.
- **Local results** (macOS M-series, CPython 3.12, proxies; `results/local_*.json`):
  - workers ready, spawn vs zygote vs fork: 8.48 / 4.19 / 4.15 s;
  - bytecode: 0.97 s with pycs vs 2.13 s without;
  - compiling `sglang.srt` (31 MB): 1.96 s;
  - shipping pycs for a whole site-packages: 5.7 s, +1.2× the source size.

## Not done or blocked

- **No Modal run** (no account here). Not yet measured on the real images: module counts, compile time, spawn vs zygote, preloadable fraction, prewarm effect, end-to-end saving.
- **model-launch unit tests not run**: installing the package was not permitted in this session. Its only entrypoint assertion still holds by construction.
- **Dockerfile changes not built.**
- **Alps validation needs CSCS access.** Steps: pyc audit of the private `sglang_base` / `vllm_base`, a TP4 e2e, and the prewarm A/B on a cold node.

## Next 3 actions

1. **Account A (about $1):** `modal run modal_app.py::smoke`, then `cpu_study --engine sglang`, `cpu_study --engine vllm`, then `prewarm_study`. Fill in WRITEUP §4 "measured" from `cpu_study_*.json`.
2. **Account B (about $2.5):** `PYSTARTUP_GPU=L40S:2 modal run modal_app.py::gpu_e2e`.
   - Check `zygote_smoke.txt`: `torch_cuda_initialized` false, `nvidia_fds` empty.
   - Check that the arms' greedy outputs are identical, and the per-arm worker-phase numbers.
3. **On Clariden:** run `python -m pystartup.pycheck` inside `sglang_base` and `vllm_base`, then decide whether to send `drafts/model-launch-bytecode-note.md` and which patch to propose first. Both are the user's decision; nothing has been posted.
