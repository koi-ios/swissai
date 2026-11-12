# Handoff notes

Partial research notes from seven workers that the vLLM-prelim session ("Claude.md and plans review") started on 11 Nov at 20:05. That was the same moment the opportunity sessions in [../opportunities/](../opportunities/) started on the same ideas. Both efforts drew on one usage limit and stopped at 20:27.

**Status.** The vLLM-prelim session did not restart its workers. The opportunity sessions are the canonical effort. Each note below was forwarded to the session that owns its idea.

**Caveats.** These are working notes, not deliverables. They contain file:line references at the versions named inside. Verify a fact before relying on it.

| Notes | Forwarded to | Facts that may be new there |
|---|---|---|
| [servekit-vllm-NOTES.md](servekit-vllm-NOTES.md) | 03-servekit-vllm | servekit's own fixture `tests/fixtures/llama70b-vllm-ngc.log` (Clariden, vLLM 0.24, Llama-3.1-70B TP4 from capstor). Lustre auto-prefetch was on, and weight loading still took 233 s of about 422 s to ready (single run). |
| [compile-cache-270-NOTES.md](compile-cache-270-NOTES.md) | 01-compile-caches-270, 02-servekit-cache-fix | The platform EDFs don't mount `/users`, so writes under `~` land in the container's writable layer and are lost at job end (inference from the CSCS docs). Youssef's EDFs set `HOME=/root` on purpose. Also Lustre flock modes. |
| [lazy-cuda-graphs/NOTES.md](lazy-cuda-graphs/NOTES.md) | 04-lazy-cuda-graphs | vLLM 0.33's default capture list is exactly 51 sizes. Upstream anticipates lazy capture, but 0.33 blocks it after startup. TileLang mHC kernels use a dynamic token dimension and are warmed before capture. |
| [python-startup/](python-startup/) | 05-python-startup | The only `SyntaxWarning` is torchao `quant_api.py:1731`. It is raised by each process that compiles the file before a `.pyc` exists. That is twice on the head node, where the wrapper and the bench race, and once on every other node (refined by the 05 session). The vLLM images install with `uv`, without `--compile-bytecode`, so they likely ship no `.pyc` files. |
| [opentela-73/NOTES.md](opentela-73/NOTES.md) | 06-opentela-73 | Leading hypothesis: priority (Lamport height) regression. The worker's store is wiped on shutdown, so its new rows lose to its own higher-priority `LEFT` row. |
| [servekit-correctness/NOTES.md](servekit-correctness/NOTES.md) | 08-preshard-roundtrip, 07-verify-tolerance | servekit's #19 backport no longer applies on SGLang 0.5.20 and 0.5.23. vLLM 0.33 MLA registers `W_UV`/`W_UK_T` as non-contiguous views, so its `_filter_subtensors` should raise on save (a prediction). SGLang 0.5.20 has a `PreshardedModelLoader`. |
