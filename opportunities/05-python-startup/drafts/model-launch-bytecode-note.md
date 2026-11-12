# Draft: note for model-launch (bytecode in the engine images)

*For the user to adapt and send if they choose; nothing has been posted. It fits as a comment on #270 (startup caches) or as a short PR description for `patches/model-launch-python-startup.patch`.*

---

**Python bytecode is a start-up cache too, and the vLLM images ship none**

While looking at what a launch does before `torch.distributed` init, I found that the vLLM Dockerfiles in `images/` install with `uv pip install` without `UV_COMPILE_BYTECODE` or `--compile-bytecode`. uv does not compile by default.

- The affected Dockerfiles are `vllm_0.30.0`, `vllm_cuda13`, `vllm_cuda13_v2` and `vllm_apertus_1.5_release`. The last also imports vLLM from a git checkout through `PYTHONPATH`.
- So the images contain no `.pyc` files, and the first process of every launch compiles torch, transformers and vLLM from source. With `writable = false`, every process does.
- Upstream `vllm/vllm-openai` sets `UV_COMPILE_BYTECODE=1`. Upstream `lmsysorg/sglang` deletes `__pycache__` instead. The SGLang image here is built with pip, which does compile.

The lab's own logs show this happening. In Youssef's servekit runs on `lmsysorg/sglang:v0.5.16`, the first process in each container emits a compile-time `SyntaxWarning` for `torchao/quantization/quant_api.py`, and later processes in the same container don't.

**Suggested fix:** one gap-filling step at the end of each Dockerfile:

```dockerfile
RUN python -m compileall -q -j 0 --invalidation-mode unchecked-hash \
      "$(python -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])')" \
    || echo "compileall: some sources did not compile; continuing"
```

- Unchecked-hash pycs (PEP 552) stay valid however mtimes change on the way to the `.sqsh`.
- Locally, compiling a 166 MB site-packages takes about 6 s on 14 cores and adds about 1.2× the source size.

**To check the private images in a minute:**

```bash
python -m pystartup.pycheck $(python -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])')
```

Run it inside `vllm_base` and `sglang_base` on Clariden. It reports how many sources would be compiled at import.

A one-line, backward-compatible hook makes the SGLang entrypoint overridable at job runtime (`${SML_SGLANG_ENTRYPOINT:-python3 -m sglang.launch_server}`). With it, a deployment can try the worker zygote (`python3 -m pystartup.sglang_launch --`) through the environment alone.
