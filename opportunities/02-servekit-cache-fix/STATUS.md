# Status: 02-servekit-cache-fix (12 Nov 2026)

## Done

- **Patches**: `patches/0001…0003` on eth-easl/servekit main `cd8f0b8`. They are also on branch `jit-cache-keyed` in `vendor/servekit`.
  - 0001 `fix: keep sglang 0.5.20's JIT caches`. This is the cache table: 11 variables, all set every time.
  - 0002 `feat: key JIT caches by environment, prune old ones, warm a new image's`.
  - 0003 `feat: report JIT output the engine leaves outside the kept caches` (`--jit-cache-strict` → exit 3).
- **Patches checked**:
  - They apply with `git am` to a fresh clone.
  - Each commit passes the suite on its own: 154, then 176, then 187 tests.
  - The final state passes 187 tests on Python 3.12 and 3.9. The only 2 errors are the macOS-only `O_DIRECT` stager tests, which error on main too.
  - They merge cleanly with PR #26 (196 tests): `results/patches-apply-and-merge-26.txt`.
- **The regression test reproduces the 0.5.20 miss.** It fails on cd8f0b8 (`results/regression-on-main-cd8f0b8.txt`) and passes with the patches.
- **Demo of the new log lines**: `results/demo-log.clean.txt`.
- **`drafts/PR.md`**: credits #18 and cites file:line at sglang v0.5.16 and v0.5.20.
- **`WRITEUP.md`**: about 2 pages.
- **`modal_app.py` + `MODAL.md` + `tools/` + `tests/test_tools.py`**: validated locally (dry run, import with modal 1.6.4, 8 tests).
  - The parser reproduces the lab A/B from its logs: dist init 19.54 / 2.74 s, first prefill step 28.49 / 5.69 s.
  - The runner runs end to end against both servekit trees with a fake engine.

## Facts confirmed (file:line)

- **servekit cd8f0b8**:
  - `jit_cache.py:16-22`: three variables.
  - `jit_cache.py:14`: `/tmp/servekit/caches`.
  - `prepare.py:87-88,138-141`: builds at `<root>/<artifact-name>` and copies into `<artifact>/servekit-caches`.
  - `launch.py:317-323`: copies back node-locally.
- **sglang v0.5.16**: `jit_kernel/utils/compile.py:244`, `TVM_FFI_CACHE_DIR`.
- **sglang v0.5.20**:
  - `kernels/jit/utils/compile/cache.py:302`: `SGLANG_JIT_CACHE_DIR or ~/.cache/sglang/jit`.
  - `srt/environ.py:1154`: deep_gemm. `:1164`: `SGLANG_CACHE_DIR`. `:1168`: cute_aot. `:1171`: jit. `:1181`: `SGLANG_CRASH_ON_JIT_COMPILE`. `:1923-1946`: third-party `setdefault`.
  - `kernels/ops/communication/all_reduce.py:47`: `load_jit("communicator")`.
- **The lab's A/B** (`servekit-jit-caching-eval`, 0.5.16): dist init 19.54 s without cache (81673) vs 2.74 s with it (81671).
- **The five slow launches** from `llama70b-tp4-sharded` (80990, 80991, 3184373, 3184455, 3184963): dist init 18.98–21.61 s. "Most likely another image" is still an inference: that prepare log is not in the repo.

## Not done or blocked

- **The GPU measurement has not run.** There is no Modal account here, so `modal run modal_app.py::experiment` (~$6) is for the user.
- **Not run on Linux locally**: the Docker daemon is off. `modal run modal_app.py::smoke` runs servekit's tests inside the sglang 0.5.20 image.
- **The e2e prepare-serve job is updated but not run.** It needs Clariden.
- **Untested `--only-prepare` cases**: the warm path with multi-node, which behaves like prepare (workers wait for the job to end), and pipeline parallelism.

## Next 3 actions

1. **Run Modal**: `download`, `smoke`, `gpu_check`, then `experiment`. Then `modal volume get` and `tools/summarize_modal.py`. Put the table into `drafts/PR.md` and `WRITEUP.md`. If `gpu_check` shows no NVLink or a driver older than CUDA 13, stop and change the image or GPU first.
2. **Send patch 0001 alone as the first PR.** It is small and obviously right. Offer 0002–0003 as the follow-up once the Modal table is in. Sending is the user's action.
3. **Ask Xiaozhe for one Clariden node-hour** to run the updated e2e job and a 3+3 Llama-70B TP4 A/B on the 0.5.20 image.
