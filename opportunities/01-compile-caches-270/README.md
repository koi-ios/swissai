# 01-compile-caches-270

Persistent JIT/compile caches for the platform's `vllm_base` and `sglang_base` images
(swiss-ai/model-launch#270). It provides an audit tool that keeps the cache list correct, a patch
for model-launch, and a Modal validation. Start with [WRITEUP.md](WRITEUP.md), then
[STATUS.md](STATUS.md).

| path | what |
|---|---|
| `cache_audit/` | the audit tool (stdlib only): `python -m cache_audit {run,static,kb-verify,match}` |
| `kb/caches.toml` | knowledge base: which variable relocates which cache, its key and write pattern, with evidence ids |
| `kb/engines.citations.json`, `research/*.citations.json` | 1,137 file:line citations into `vendor/` (the shipped versions), checked by `tests/test_kb.py` |
| `research/*.md` | source reviews per component (torch/Triton, FlashInfer/tvm-ffi/TileLang, DeepGEMM/CuTe DSL/misc, SGLang 0.5.20, vLLM 0.32.0) |
| `integration/sml_jit_cache.py` | key / arch / seed / promote / gc helper; embedded into master.sh by the patch |
| `patches/` | `git format-patch` against model-launch main @02b74ce (`after-272/`: env commit on PR #272) |
| `harness/` | Modal A/B driver (`ab.py`), one-shot warmup for audits, summarizer, local CPU A/B |
| `stress/` | write-safety stress (real libraries on CPU; Triton/FlashInfer on GPU), TileLang CUDA benchmark |
| `modal_app.py`, `MODAL.md` | Modal experiments and runbook (`python modal_app.py --dry-run`) |
| `drafts/` | the #270 comment, env TOML diffs, images-CI pre-warm design |
| `results/` | local measurements (audit reports, CPU A/B, stress); Modal results go to `results/modal/` |
| `tools/` | `fetch_sources.sh` (vendor the pinned sources), `make_citations.py` |

## Quick use

```bash
uv venv .venv --python 3.12 && uv pip install --python .venv/bin/python torch==2.13.0 apache-tvm-ffi==0.1.11 tilelang==0.1.12 ninja pytest modal
.venv/bin/python -m pytest -q                       # all tests (the real-library audits take ~3 min)
.venv/bin/python -m cache_audit run --relocate --readonly -- .venv/bin/python tests/workloads/tvm_ffi_jit.py
.venv/bin/python -m cache_audit run --env-toml harness/envs/vllm_base.toml -- <warmup command>   # CI gate
.venv/bin/python -m cache_audit static              # GPU-free drift check of installed packages
```
