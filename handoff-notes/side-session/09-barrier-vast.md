# 09-barrier-vast: hand-over note (12 Nov 2026)

Deliverables are in `opportunities/09-barrier-vast/`: `WRITEUP.md`, `STATUS.md`, `RUNBOOK.md`, `repro/`, `estimate/`, `patches/`, `drafts/`, `results/` and `tests/` (37 passing).

Facts other sessions may reuse (each checked in the source):
- **The barrier timeout is a constant, and nothing reaches it.** SGLang's post-load barrier is a hard-coded 480 s from 0.5.10.post1 to main as of 11 Nov 2026 (v0.5.20 `load_model_utils.py:51`).
  - `--dist-timeout` does not reach it.
  - `SGLANG_UNBALANCED_MODEL_LOADING_TIMEOUT_S` (set by model-launch-factory) is read by no release.
- **The old image is SGLang 0.5.10.post1.** `ci/sglang_cuda13.sqsh` is pinned to it since 17 Apr 2026, with torch 2.9.1.
- **gloo `monitored_barrier(wait_all_ranks=True)` waits a full timeout for each missing rank in turn,** dead or alive (CPU reproduction).
- **Measured node-to-node spread of mmap loads.** GLM-4.7 on 4 nodes, from Youssef's per-node logs: 83 s on Bristen and 132 s on Clariden (CV 4–6%).
- **Prediction for GLM-5.1 BF16 at TP32.** Each node faults 622 GB of checkpoint pages, 1.3× its 480 GB of DRAM. `--ep-size 32` brings this down to 194 GB.
- **CSCS storage.** Ritom is VAST over NFS: 10 PB, 900 GB/s read, on Slingshot (docs.cscs.ch).
- **Readahead.** Linux NFS readahead has defaulted to 128 KiB since 5.4. Lustre handles mmap faults with its own readahead engine (`ra_pages = 0`).
