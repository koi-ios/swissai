# Cold-start ideas: where the work lives

The ideas are being researched and built in [opportunities/](opportunities/). There is one session and one folder per idea, and the rules and verified facts are in [opportunities/SHARED_CONTEXT.md](opportunities/SHARED_CONTEXT.md).

| Folder | Idea |
|---|---|
| 01-compile-caches-270 | model-launch #270: persistent compile caches for vLLM and SGLang, keyed per image and GPU, with an automated cache audit |
| 02-servekit-cache-fix | servekit: SGLang 0.5.20's moved cache directories, and an image-keyed cache |
| 03-servekit-vllm | servekit for vLLM: fast weight loading (milestone 1); uses [serve-kit-vllm-prelim/](serve-kit-vllm-prelim/) |
| 04-lazy-cuda-graphs | Traffic-aware, lazy CUDA-graph capture |
| 05-python-startup | Python start, TP worker spawn and distributed init |
| 06-opentela-73 | OpenTela #73: a restarted worker stays unroutable for minutes |
| 07-verify-tolerance | Calibrated correctness checks for low-precision models |
| 08-preshard-roundtrip | `sharded_state` round-trip for MLA and MXFP4 models (servekit #19, #14) |
| 09-barrier-vast | model-launch-images #22: TP32 post-load barrier; VAST readiness |

Supporting material:

- [serve-kit-vllm-prelim/HARNESS.md](serve-kit-vllm-prelim/HARNESS.md): one measured launch of vLLM or SGLang under `servekit profile`, with hooks for load tests, extra environment, wrappers and cache audits. Self-test: 55 checks against a mock engine.
- [handoff-notes/](handoff-notes/): partial notes from a parallel attempt on 11 Nov, each forwarded to the session that owns its idea.
- [sglang-cold-start/EVALUATION.md](sglang-cold-start/EVALUATION.md): the evaluation these ideas came from.
