# Patches (local only, never pushed)

| patch | against | what | regenerate |
|---|---|---|---|
| `sglang-v0.5.20-worker-zygote.patch` | sgl-project/sglang `v0.5.20` | Opt-in worker zygote (`SGLANG_WORKER_ZYGOTE=1`). Adds the new stdlib-only package `sglang/_worker_zygote`, a 3-line hook at the top of `sglang/__init__.py`, and the start-method switch in `engine.py`, a NUMA fallback and env registrations. 8 files, +885/−2. | `python patches/make_sglang_patch.py` (generated from `src/pystartup`) |
| `sglang-v0.5.20-dockerfile-bytecode.patch` | sgl-project/sglang `v0.5.20` | Runtime stage: compiles site-packages and the editable SGLang tree with unchecked-hash pycs. 1 file, +8. | `python patches/make_sglang_dockerfile_patch.py` |
| `model-launch-python-startup.patch` | swiss-ai/model-launch `02b74ce` | Gap-filling unchecked-hash `compileall` in 6 Dockerfiles (SGLang and vLLM templates plus generated images), and `SML_SGLANG_ENTRYPOINT` (default unchanged). 7 files, +53/−2. | `python patches/make_model_launch_patch.py` |

**Tested:**
- `tests/test_sglang_patch.py` checks that the zygote patch applies to a clean v0.5.20 worktree and that the patched files compile.
- The same test runs the patch's zygote end to end in a stand-in `sglang` package: workers are forked from the zygote, have torch preloaded and see env vars set after the zygote started.

**Not tested:**
- model-launch's own unit tests were not run here: installing the package was not permitted in this session. Its only entrypoint assertion (`"python3 -m sglang.launch_server" in head`) still holds, because the new default contains that string.
- The Dockerfile steps have not been built.

The wrapper alternative, `python -m pystartup.sglang_launch -- <sglang args>`, needs no SGLang change.
