# STATUS: 08 preshard round trip (12 Nov 2026)

## Done
- **Read, with citations** (WRITEUP.md): SGLang 0.5.20's `ShardedStateLoader` and `PreshardedModelLoader`, the MLA alias and MXFP4 triton paths, and servekit's prepare, patches and guard.
- **Upstream PR status:** #35715 open, #34558 closed unmerged, #34448 open, #35702 auto-closed. Main removed the `triton_kernel` runner (#49660).
- **New findings:**
  - servekit's #35715 backport stopped applying at v0.5.19 (2d8484740d);
  - the guard misses gpt-oss on the default backend;
  - `GemmaRMSNorm.gemma_weight` goes stale (predicted);
  - vLLM 0.33's `sharded_state` has the same design (read only, credited to a sibling session).
- **servekit branch `preshard-roundtrip`** (vendor/servekit, 2 local commits, not pushed), exported to `patches/servekit/`. It adds:
  - the raw preshard;
  - the deep-state checker and `servekit deepstate`;
  - `--preshard-format raw`;
  - the backport and guard fixes.
- **Tests:** 176 servekit tests pass (2 pre-existing Linux-only errors), and 9 reproductions in `repro/` against SGLang's own code pass.
- **Refreshed SGLang patch** for #35715 (`patches/sglang/`, v0.5.20 and main).
- **Modal:** `modal_app.py` (imports with modal 1.6.3), `modal_driver.py` (dry run verified), `summarize.py`, `MODAL.md`.
- **Results:** `results/cpu_repro.json`, `results/cpu_overhead.json`.
- **Drafts** in `drafts/`, none sent:
  - servekit PR 1, the backport fix: send first, it is small and urgent;
  - servekit PR 2, raw preshard, deep state and guard;
  - a comment on SGLang #35715, with the refreshed patch;
  - a status comment on #34448;
  - a new `gemma_weight` issue, held until a GPU confirms it.
- **Hand-over note** in the shared notes folder (`08-preshard-roundtrip.md`). Replied to the vLLM-prelim session.

## Not done
- **GPU validation:** the Modal run has not been executed (no account here); every GPU outcome in MODAL.md is a prediction.

## Next 3 actions
1. Run `modal run modal_app.py::smoke`, then both experiments (about $10–15), then `python summarize.py results/modal`, and update WRITEUP.md's results with the measured numbers.
2. Send servekit PR 1 (and the #35715 comment) now; send PR 2 and the #34448 comment with the Modal numbers attached.
3. Measure the raw preshard's load cost at scale (Kimi-K3 or GLM-5.x on Alps). Make unread replay entries `meta` tensors to cut per-tensor cost.
