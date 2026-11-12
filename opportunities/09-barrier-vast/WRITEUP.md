# 09: why 8-node TP32 launches fail SGLang's post-load barrier, and what VAST changes

12 Nov 2026. Everything below comes from source code, the lab's own logs, public pages and CPU-only experiments; nothing ran on a GPU or on Alps. Labels: **MEASURED** (here, or in the lab's logs) · **PREDICTION** (a model) · **VENDOR** (an unverified claim).

## A. model-launch-images#22: the post-load barrier

**Problem.** Per model-launch PR #271, the barrier fails for GLM-5 ×2, GLM-5.1 ×2 and Kimi-K3 (all 8 nodes, TP32). In the PR's words, rank 0 "finishes loading and the other 31 ranks never do".

| model | checkpoint size | config | result |
|---|---|---|---|
| GLM-5, GLM-5.1 BF16 | 1,508 GB | 8 nodes, TP32 | fails |
| Kimi-K3 (MXFP4) | 1,561 GB | 8 nodes, TP32 | fails |
| GLM-5(.1)-FP8 | 756 GB | 4 nodes, TP16 | passes |
| GLM-4.6, DeepSeek-V3.1, Qwen3.5-397B | — | 4 nodes | pass |

- **The old image is SGLang 0.5.10.post1.** `sglang.toml` → `ci/sglang_cuda13.sqsh`. Its Dockerfile has pinned 0.5.10.post1 since 17 Apr 2026, with torch 2.9.1, sgl-kernel 0.3.21 and a hand-built FA3. PR #235 (on hold) would move that image to 0.5.20.
- **Kimi-K3 cannot run on the old image at all:** it first appears in SGLang #32541, Aug 2026.

**The code.**
- **The barrier.** `dist_barrier_after_load` calls `monitored_barrier(tp_cpu_group, timeout=480 s, wait_all_ranks=True)` (v0.5.20 `model_executor/model_runner_components/load_model_utils.py:51,367-387`; called from `model_runner.py:1291,1314`). 0.5.10.post1 has the same code inline at `model_runner.py:253,1276-1294`.
- **History** (found by bisection): added at 300 s in #3988 (Mar 2025); raised to 480 s in #12759 (Nov 2025, Ascend Kimi-K2-Thinking). It is still a constant on main at 95999790a4 (11 Nov 2026).
- **No switch reaches it.**
  - `--dist-timeout` goes to `init_process_group` and, from 0.5.20, to the NCCL subgroups (`parallel_state.py:2401-2425,369-381`).
  - The gloo group keeps a fixed 120 min (`:293,401-404`).
  - The barrier passes its own 480 s.
- **The factory's variable is a no-op.** model-launch-factory's `SGLANG_UNBALANCED_MODEL_LOADING_TIMEOUT_S=3600` (Kimi-K3.1 TOML) is read by no SGLang release.
- **Between 0.5.10.post1 and 0.5.20** (7,001 commits; 100 touch `model_loader/`), the default mmap + thread-pool path is unchanged. What is new:
  - opt-in prefetch (#20843);
  - drop-cache after load (#29156);
  - overlap mode (#32017);
  - sorted file order (#26937);
  - a pre-barrier kernel-precompile hook, used only by Kimi-K3 (#32541);
  - torch 2.9.1 → 2.13.0 and transformers 5.3 → 5.12.

**What I built.**

1. **`repro/barrier_gloo.py`** runs SGLang's barrier code on N CPU processes over gloo. MEASURED with torch 2.15; the scenario times are scaled, with T = 1 s standing for 480 s.

   | ranks 1..N−1 | rank 0 | waits | lists |
   |---|---|---|---|
   | on time, or 0.6 T late | passes | — | — |
   | one rank 1.5 T late | fails | 1.0 T | that rank |
   | all 1.5 T late | fails | 1.5 T | **only rank 1** |
   | staggered 0.6 T apart (total 4.2 T, N = 8) | **passes** | 4.2 T | — |
   | dead, hung, or more than (N−1) T late | fails | **(N−1) T** (7.0 T at N = 8) | all |
   | all 1.5 T late, 10× `--dist-timeout` | fails | 1.5 T | rank 1 |
   | all 1.5 T late, env knob = 3 T | passes | — | — |

   Rank 0 waits a full timeout for each missing rank in turn, dead or alive. So **an error listing all 31 ranks means rank 0 waited about 31 × 480 s ≈ 4 h** and no other rank arrived. That is dead, hung or hours-late ranks, not a spread of a few minutes. Because model-launch kills the job when any node's `srun` step exits (`launchers/framework.py:590-610`, `wait -n`), the follower processes must still have been alive.

2. **`estimate/tp_load_model.py`** combines the configs, SGLang's TP slicing and 4 KiB-page faults with Youssef's per-node mmap rate. That rate is MEASURED in his logs: GLM-4.7 TP4·PP4 on 4 nodes, 177 GB per node in 820–983 s (0.18–0.22 GB/s per node), node-to-node CV 4.0% (Bristen) and 5.9% (Clariden). The table below is PREDICTION. Rank 0's wait follows the barrier rules measured above.

   | case | GB per node | vs 480 GB DRAM | cold mmap load per node | barrier fails from node spread alone (CV 4 / 6 / 10%) |
   |---|---|---|---|---|
   | GLM-5.1 BF16, TP32, 8 nodes | 622 | **1.30×** | 52 min (48–58) | 0 / 4 / 24% |
   | same, with `--ep-size 32` | 194 | 0.40× | 16 min | 0 / 0 / 0% |
   | GLM-5.1-FP8, TP16, 4 nodes | 374 | 0.78× | 31 min | 0 / 0 / 4% |

   At TP32 each node still faults every page of the row-split expert `down_proj` (4 KiB rows, 128 B per rank). That is why a node reads 0.42 of the checkpoint, not 1/8. The model reproduces GLM-4.7's checkpoint size to within 2%.

**Hypotheses, ranked.**
1. **Memory pressure: the node's checkpoint footprint exceeds its DRAM.**
   - For: only the failing configuration overflows node DRAM (1.30× vs 0.78×). Youssef's barrier patch notes within-node spreads "well past" 480 s on a cold 1.5 TB read. Reclaim stalls or an OOM-killed rank on a still-running node fit "all 31 listed".
   - Against: the old image has the same footprint and passes.
   - Test: `sacct`/`dmesg` for OOM kills, and `/proc/pressure/memory` during a load.
2. **Something in 0.5.20 makes loading slower or hungrier.** Candidates: torch 2.13 host copies, more mmap views held alive, or Kimi-K3's pre-barrier JIT. Test: the same nodes on both images, with per-rank `Load weight end` times and RSS.
3. **Plain node spread: least likely on its own.** It gives 0–4% failures per launch at the measured CV, which cannot produce 5 failures out of 5.

The discriminating evidence already exists in the failing logs: the number of `[Rank 0]: Rank k failed…` lines and their timestamps. Questions for Aryan and Rob are drafted in `drafts/questions_aryan_rob.md`.

**Fixes.**
1. **Now, without a rebuild:** `patches/sitecustomize.py`, a lazy post-import hook that honours `SGLANG_UNBALANCED_MODEL_LOADING_TIMEOUT_S`.
2. **Upstream:** `patches/sglang-barrier-timeout-env.diff`. It applies to v0.5.20 and names the variable in the error message.
   - Caveat: the worst-case wait is (N−1) × timeout, so set the variable to the expected spread (about 30 min), not hours.
3. **Shrink the footprint.**
   - `--ep-size 32` (PREDICTION: 3.2× fewer bytes per node). The factory already uses it for Kimi-K3.1 on Beverin. It needs a throughput check.
   - Or servekit's presharded `/dev/shm` staging: 188 GB per node. GLM-5.x needs servekit's MLA patch; Kimi-K3 is blocked on sglang#34448.
   - Avoid `--weight-loader-prefetch-checkpoints` at TP32: it adds page-cache pressure.

## B. VAST readiness

**Facts.** I verified each one at its source, except those marked VENDOR.
- **Ritom is VAST over NFS:** 10 PB, 900 GB/s read, 317 GB/s write, on Slingshot (docs.cscs.ch/alps/storage). The docs give no protocol, transport or client.
- **Linux NFS faults are read-around of `read_ahead_kb`.**
  - The default has been **128 KiB since Linux 5.4** (commit c128e575).
  - It can only be changed through sysfs or udev, not a mount option.
  - `pread`/O_DIRECT issue concurrent rsize RPCs (≤ 1 MiB).
  - O_DIRECT on NFS has no alignment requirement.
  - `FADV_DONTNEED` drops clean pages that are not mapped.
- **Lustre handles mmap faults with its own readahead engine:** `llite_lib.c` sets the bdi's `ra_pages = 0`.
- **VENDOR, vastnfs client:**
  - multipath (`remoteports`), `spread_reads`, NFS/RDMA and GDS;
  - 8–8.5 GB/s per RDMA mount, about 11 GB/s with TCP + nconnect on 100 GbE;
  - "expanding readahead" for mmap, with a claimed 5–10× faster SGLang loads and no setup given.

**Predictions: which servekit techniques still matter on VAST.**
- **Presharding: yes, on any filesystem.** It removes TP-slice amplification (622 → 188 GB per node) and caps the footprint below DRAM.
- **Parallel large-block staging: probably yes on stock NFS.** With faults capped at 128 KiB per round trip (about 0.1–0.3 ms), each faulting thread gets 0.4–1.3 GB/s. The gap closes if Ritom uses vastnfs with expanding readahead or a raised `read_ahead_kb`. `bench_storage.py` measures exactly that.
- **Overlap and JIT caches: unchanged.** Neither depends on the filesystem.

**Built.** `bench_storage.py` (stdlib only), `RUNBOOK.md` and 21 tests.
- It compares `mmap`, `mmap-stride` (a TP slice: 128 B of every 4 KiB), `pread` and `odirect`, with 1–64 reader processes and a range of block sizes.
- Each run is client-cold (`posix_fadvise` DONTNEED, checked with `mincore`) and reads fresh regions.
- It records mount options and readahead with every result.
- The local test is MEASURED on this Mac's APFS NVMe: 384 MiB per run, median of 3 (min–max), at most 6% resident before each run. The whole sweep took 19 s.

  | method | GB/s |
  |---|---|
  | mmap, 1 reader | 0.91 (0.85–1.02) |
  | mmap, 16 readers | 2.59 (2.31–2.73) |
  | mmap-stride, 1 reader | 0.73 (0.72–0.75) |
  | pread 16M, 1 reader | 6.34 (4.59–6.51) |
  | F_NOCACHE 16M, 16 readers | 4.66 (4.08–5.98) |

  The runs are short and macOS has no O_DIRECT, so this validates the tool, not the disk.

## Run it
```bash
uv venv .venv && uv pip install --python .venv/bin/python torch pytest numpy
.venv/bin/python -m pytest tests -q                       # 37 tests, ~30 s
.venv/bin/python repro/barrier_gloo.py --scenario all --world 8 --timeout 1
.venv/bin/python estimate/tp_load_model.py
python3 bench_storage.py --quick <dir>                    # see RUNBOOK.md for Alps
```
There is no Modal app: nothing here needs a GPU, and Modal has neither Lustre nor VAST.

## What it would take on Alps, and the risks
- **(A)** Logs from the failing runs, then one 8-node launch each with the knob, with EP32, and with servekit staging. About 3 jobs × 8 nodes × 1 h.
- **(B)** About 15 min on one node each for capstor, iopsstor and Ritom, plus the 8-node step in RUNBOOK §3.
- **Risks:**
  - The memory-pressure hypothesis does not yet explain why the old image passes.
  - EP32 changes MoE communication.
  - vastnfs may already close the mmap gap; the benchmark will show it.

## Pitch
- Your 8-node launches die at a 480 s barrier that no flag reaches; the factory's environment variable is read by nothing.
- A CPU reproduction shows the "all 31 ranks" error means about 4 h of rank 0 waiting on dead or hung ranks. My byte model puts the failing case at 1.3× node RAM, and `--ep-size 32` cuts that 3.2×.
- I've drafted a fix and a benchmark that tells you, in 15 minutes per filesystem, whether servekit still pays off on VAST.

**Credit:**
- Youssef Boughizane's write-up, logs and barrier patch;
- servekit (eth-easl);
- model-launch PRs #235, #245 and #271 (Aryan Ahadinia, Rob);
- SGLang (sgl-project);
- CSCS docs, Linux and Lustre sources;
- VAST docs (vendor claims, marked as such).
