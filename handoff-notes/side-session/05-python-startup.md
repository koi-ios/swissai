# 05-python-startup: hand-over (12 Nov 2026)

Folder: `opportunities/05-python-startup/`. Read STATUS.md first, then WRITEUP.md and MODAL.md.

## Headline findings (from the lab's own logs and Dockerfiles)

- **The 25–30 s "process start + TP worker spawn" in EVALUATION.md is a lower bound.**
  - Youssef's servekit-eval sbatch puts `patches/sitecustomize.py` on PYTHONPATH. It imports SGLang, and so torch, in every Python process, so servekit's wrappers warmed the page cache and wrote the pycs before SGLang's clock started.
  - Launches without that pre-import spend 28–64 s in the main process alone; Popen → dist init is 33–79 s.
  - Source: `results/log_timeline.md`, 59 launches.
- **Workers: 11–16 s on every launch.** SGLang 0.5.20 forces `spawn`; the schedulers (1 per TP rank) and the detokenizer re-import everything, and the main process imports the scheduler tree too.
- **Bytecode:**
  - upstream `lmsysorg/sglang:v0.5.20` deletes all `__pycache__` (and SGLang is an editable install);
  - **every model-launch vLLM image is uv-built with no pycs**;
  - the model-launch SGLang image is pip-built and has pycs;
  - upstream `vllm/vllm-openai` sets `UV_COMPILE_BYTECODE=1`.
  - The torchao SyntaxWarning in 44 of 72 logs is the first-process compile; the container root is writable.
- **vLLM forks its workers by default**, so it avoids the full re-import, but the uv images mean every process compiles.

## Built and tested (41 tests pass locally)

- Profiler for the real images: CPU containers, with GPU import paths emulated by served device facts.
- CUDA-safe worker zygote, plus a no-source-change SGLang wrapper (`python -m pystartup.sglang_launch -- …`).
  - Proof of fork safety: torch uninitialised, no `/dev/nvidia*` fds, one thread. Otherwise spawn is kept.
  - Locally: 4 workers ready in 4.19 s vs 8.48 s with spawn.
- Upstream SGLang patch (opt-in `SGLANG_WORKER_ZYGOTE=1`).
- Dockerfile bytecode patches for SGLang and model-launch, plus `SML_SGLANG_ENTRYPOINT`.
- Prewarm and `PYTHONPYCACHEPREFIX` tools.

## Pending

- All Modal runs, about $3.3: `smoke`, `cpu_study` for sglang and vllm, `prewarm_study`, `gpu_e2e` TP2.
- Alps validation.
- Predicted saving (not measured): 9–13 s per launch from the zygote, plus a few seconds from bytecode on uv/upstream images; the prewarm ceiling is the cold gap.
