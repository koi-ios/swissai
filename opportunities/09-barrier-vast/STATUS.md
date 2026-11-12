# STATUS: 09-barrier-vast (12 Nov 2026)

## Done

### A. Post-load barrier (model-launch-images#22)
- **Old image version found:** `ci/sglang_cuda13.sqsh` is SGLang 0.5.10.post1, per the Dockerfile pin since 17 Apr 2026. Rob should confirm it inside the `.sqsh`.
- **Barrier located in 0.5.10.post1 and 0.5.20, with history:**
  - Added at 300 s (#3988), raised to 480 s (#12759).
  - Still a constant on main.
  - `--dist-timeout` cannot reach it.
  - The factory's `SGLANG_UNBALANCED_MODEL_LOADING_TIMEOUT_S` is a no-op.
- **Loader and dependency changes between the two versions listed** with PR numbers.
- **`repro/barrier_gloo.py` plus 9 tests.** These measure the gloo `monitored_barrier(wait_all_ranks=True)` semantics. Rank 0 waits a full timeout for each missing rank in turn, dead or alive. So "all 31 ranks listed" means a wait of about 4 h. The reproduction also shows that a staggered spread larger than the timeout passes, `--dist-timeout` has no effect, and an env knob fixes it.
- **Measured from Youssef's per-node logs:** node-to-node spread of the default mmap load, 83 s and 132 s on GLM-4.7 at 4 nodes (CV 4.0% and 5.9%).
- **`estimate/tp_load_model.py` plus 4 tests** (prediction):
  - GLM-5.1 BF16 at TP32 reads 622 GB per node, 1.30× node DRAM, about 52 min cold.
  - With `--ep-size 32`, 194 GB and about 16 min.
  - GLM-5.1-FP8 at TP16 reads 374 GB, 0.78×.
  - Node spread alone fails the barrier 0–4% of the time at the measured CV.
- **Patches:**
  - `patches/sglang-barrier-timeout-env.diff` (applies to v0.5.20; the patched files compile);
  - `patches/sitecustomize.py` (a lazy hook, no rebuild needed) with 3 tests.
- **Drafts:** `drafts/questions_aryan_rob.md`, including a guide to reading the barrier log lines.

### B. VAST readiness
- **Research** done by a subagent; I checked the key facts myself:
  - CSCS storage table: Ritom is VAST NFS, 900 GB/s read;
  - the Linux 5.4 NFS readahead commit (128 KiB default);
  - Lustre `ra_pages = 0`.
- **`bench_storage.py`, `RUNBOOK.md` and 21 tests.** Local cold sweep on the Mac: 54 runs in 19 s, results in `results/storage-mac-local-cold.jsonl(.summary.md)`.
- **WRITEUP.md** covers sections A and B.

## Not done / open
- **Root cause of #22 not confirmed.** It needs the failing runs' logs (rank-0 `Rank k failed` lines with timestamps, any OOM kills). Why the old image passes is still unexplained.
- **Upstream patch** checked against v0.5.20 only, not current main (same code block at `load_model_utils.py:473-487`). Not sent anywhere.
- **Patch semantics:** the worst-case wait under `wait_all_ranks=True` is (N−1) × timeout. A bounded total wait would need a different barrier design; that is not explored.
- **No runs on Alps, and no Modal app.** Nothing here needs a GPU, and Modal has neither Lustre nor VAST.
- **Unknown for Ritom:** whether it is mounted with vastnfs or upstream NFS, over TCP or RDMA.

## Next 3 actions
1. Send the drafted questions to Aryan and Rob (user's call). With the logs, decide between memory pressure and a 0.5.20 regression in one read.
2. Ask for one 8-node GLM-5.1 BF16 launch with `--ep-size 32`, and one with the `sitecustomize` knob at 1800 s, recording `/proc/pressure/memory` on every node.
3. Get `bench_storage.py` run on capstor, iopsstor and Ritom (RUNBOOK §1–3, about 1 node-hour) to answer whether servekit's staging still pays on VAST.
