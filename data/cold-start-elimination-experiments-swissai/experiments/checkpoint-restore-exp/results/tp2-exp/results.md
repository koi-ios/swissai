# TP=2 findings — checkpoint/restore of a 2-GPU tensor-parallel SGLang server

Date: 2026-08-21. Same box as the rest of this experiment (2x RTX 3060, non-root user,
`criu` carrying `cap_checkpoint_restore` as a file cap, `--unprivileged`). Model:
Qwen2.5-3B-Instruct, `--tensor-parallel-size 2` via `GPU_IDS=0,1` (see `config.sh`).
Scripts: `scripts/tp2-exp/{baseline,checkpoint,restore}.sh`, thin wrappers around the
shared `scripts/*.sh` (see their headers) that pin both GPUs and redirect output under
`results/tp2-exp/`. `config.sh`/NOTES.md flagged TP>1 as explicitly out of scope for the
TP=1 pass — this is the first attempt.

## Status: baseline PASSES, checkpoint dump PASSES, restore FAILS with `--tcp-close` (sections 1-3 below) — but PASSES with `--tcp-established` + a CAP_NET_ADMIN criu binary, naive + thin, and N-times restorable per snapshot (see "Update 2026-08-27" at the bottom).

## 1. Baseline (TP=2 cold launch) — PASS

`scripts/tp2-exp/baseline.sh --cold`: 91.21s cold start (vs 76.73s at TP=1 on this same
box — two ranks' init/NCCL bootstrap costs more than the smaller per-GPU weight shard
saves), 272.0 tok/s post-launch throughput (vs 253.7 tok/s at TP=1 — the expected TP=2
speedup), correct greedy-decode output. `results/tp2-exp/baseline/`.

Re-run 2026-08-27 with a genuinely cold compile cache (`.cold-compile-cache` wiped and
verified repopulated): **92.87s** to ready — essentially unchanged from the 91.21s above.
Phase split (`baseline-20260827T100458Z-profile.json`): `process_startup` 11.7s,
`tp_worker_spawn` 16.5s, `torch_distributed_init` 1.9s, `weight_loading` 5.7s,
`cuda_graph_capture` 28.1s, `piecewise_cuda_graph_capture` 24.2s, `http_bind`/`warmup`
4.5s. Unlike TP=1 (fully-cold 109.8s vs warm 76.7s — a ~33s JIT-compile gap), a cold
cache barely moves TP=2: flashinfer's attention kernels are AOT-shipped, so "cold" only
recompiles a few triton/inductor kernels (~13s, hidden inside the graph-capture phases).
What dominates TP=2 startup is rank spawn + NCCL init + CUDA-graph *capture* (~46s), none
of which a warm cache helps. Post-restore throughput measured 402-408 tok/s across this
and every keep-tcp restore below (higher than the 272 tok/s recorded 08-21 — a real
decode speedup on this box between those dates, unrelated to cold-start).

## 2. Naive checkpoint dump — PASS, but needed one new CRIU flag

First attempt failed immediately:

```
Error (criu/files-reg.c:1031): Can't dump ghost file /dev/shm/nccl-N8nvmY of 33034240 size, increase limit
```

NCCL allocates a `/dev/shm` ring-buffer file for inter-GPU coordination that's tens of MB
under TP=2 — over CRIU's default 1MB `--ghost-limit`. Not an issue at TP=1 (no NCCL
process group needed for a single rank). Fixed by making `--ghost-limit` overridable via
a `GHOST_LIMIT` env var in `REUSABLE_DUMP_ARGS` (`scripts/utils/common.sh`), set to `64M`
by `scripts/tp2-exp/checkpoint.sh`; default TP=1 behavior unchanged since the var is
unset there. With the fix: 66.0s total checkpoint (17.7s GPU evict + 48.3s criu dump),
**29G snapshot** (vs 15G at TP=1 — two ranks' worth of CPU-side process state + weights).

## 3. Restore — intermittent CRIU bind race, then a real post-restore crash

### 3a. First restore attempt after every fresh checkpoint fails on a bind conflict

```
102865: Error (criu/sk-inet.c:1059): inet: Can't bind inet socket (id 613): Address already in use
102864: Error (criu/sk-inet.c:1059): inet: Can't bind inet socket (id 481): Address already in use
Error (criu/cr-restore.c:1269): 93369 killed by signal 9: Killed
```

Reproduced on both checkpoint attempts (2 for 2). No live process was ever holding the
port (`pgrep`/`ss` clean each time), and `ss -tn state time-wait` showed nothing either,
so this isn't an ordinary TIME_WAIT wait-out — more likely a CRIU restore-ordering race
specific to a socket shared between the two rank processes (fork-inherited fd or
SO_REUSEPORT bind, restored as two independent `bind()` calls instead of one restore +
fd-table dup). **Retrying the restore immediately succeeds** (2 for 2 so far) — treat as
a known flake for now, not yet root-caused or fixed.

### 3b. Once CRIU restore + GPU reattach succeed, the server crashes trying to serve

criu restore (~12-18s) and `cuda-checkpoint` GPU reattach (~9s) both succeed cleanly. The
server comes up far enough to answer one lightweight request (`GET /v1/models` — no
cross-rank GPU work needed), then:

```
[rank0]: TCPStore.cpp:106] [c10d] sendBytes failed on SocketImpl(...): Broken pipe
[rank0]: ProcessGroupNCCL.cpp:1771] Failed to check the "should dump" flag on TCPStore,
         (maybe TCPStore server has shut down too early), with error: Broken pipe
Subprocess scheduler_0 (pid=...) crashed with exit code 1. Triggering SIGQUIT for cleanup...
```

**Root cause**: `checkpoint.sh`'s `REUSABLE_DUMP_ARGS` includes `--tcp-close` (needed
because `TCP_REPAIR` requires `CAP_NET_ADMIN`, which we don't have — see NOTES.md). This
force-closes every established TCP connection at dump time, including
`ProcessGroupNCCL`'s TCPStore connection between the two ranks. At TP=1 this is harmless
(no cross-rank connection exists). At TP=2, `ProcessGroupNCCL`'s background
`HeartbeatMonitor` thread keeps polling that same TCPStore connection for the lifetime of
the process (not just at init) — after restore it's dead, so the poll throws, and the
uncaught exception kills the rank.

Web research (see below) confirms this is not a workaround-able CRIU flag issue: **nothing
automatically reconnects a torch-distributed TCPStore/NCCL connection after its underlying
socket is closed.** vLLM's own CUDA-checkpoint RFC
([vllm-project/vllm#34303](https://github.com/vllm-project/vllm/issues/34303)) treats
NCCL state as fully destroyed by any checkpoint and explicitly **destroys the process
group before suspend** (`cleanup_dist_env_and_memory()`) and **rebuilds it after resume**
(`init_worker_distributed_environment()`: fresh TCP rendezvous + `ncclCommInitRank` per
group) — an application-level (SGLang scheduler) hook, not a CRIU-level fix. They also
flag that CUDA graphs capturing NCCL collectives embed the old `ncclComm_t` handle and
likely need re-capture after a rebuild — a second risk on top of the first, relevant here
since our launches use CUDA graphs by default.

### First fix attempt: `TORCH_NCCL_ENABLE_MONITORING=0` — env var never reached the subprocess

`TORCH_NCCL_ENABLE_MONITORING=0` is PyTorch's own documented way to disable exactly that
heartbeat thread (tradeoff: loses NCCL-deadlock auto-detection — an acceptable trade for a
short-lived checkpoint experiment). Baked into the launch env
(`TORCH_NCCL_ENABLE_MONITORING=0 ./scripts/tp2-exp/checkpoint.sh`) and re-tested: **identical
crash, byte-for-byte.**

Root-caused by inspecting `/proc/<pid>/environ` live: SGLang's scheduler subprocess
(`mp.set_start_method("spawn", ...)` in `engine.py`) does **not** end up with our exported
var — present in the multiprocessing bootstrap process, absent by the time the process
renames itself to `sglang::scheduler_TP1`. This is the *exact same* env-scrubbing NOTES.md
already documents for `USE_LIBUV` (`ensure_shim()`'s whole reason for existing is routing
an env var past this scrubbing via a `.pth` file loaded fresh in every spawned Python
process, sys-path-based rather than envp-based). Not yet fixed: needs
`TORCH_NCCL_ENABLE_MONITORING=0` added the same way (`.pth`-based `os.environ.setdefault`
in `ensure_shim()`), not as a plain exported env var.

**Even if that lands, per the vLLM RFC above it's very unlikely to be a full fix** — it
only silences the watchdog's crash-on-broken-pipe; the underlying NCCL communicator is
still built on a dead connection, so an actual generation request (needing real cross-rank
all-reduce, unlike the `/v1/models` probe that got through) would likely hang rather than
serve correctly. Untested — the `.pth` fix hasn't been applied yet.

## Next steps (not yet done)

1. Route `TORCH_NCCL_ENABLE_MONITORING=0` through `ensure_shim()`'s `.pth` file so it
   actually reaches the scheduler subprocess, and confirm whether that alone lets a real
   generation request (not just `/v1/models`) succeed post-restore.
2. If (likely) it doesn't fully work, this needs an SGLang-side hook: destroy the TP
   process group before dump, rebuild it (fresh TCPStore + `ncclCommInitRank`, possibly
   re-capturing CUDA graphs) after restore — real engineering scope, not a wrapper-script
   fix.
3. Root-cause or work around the CRIU bind-race in 3a (currently just "retry once").

---

## Update 2026-08-27 — keep the TCPStore connection alive instead of closing it: TP=2 restore WORKS

Rather than the app-level rebuild in step 2, tried the thing NOTES.md ruled out on the
"we don't have CAP_NET_ADMIN" grounds — which no longer holds on this full-root VM. Built
a second criu binary `/usr/local/sbin/criu-netadmin` (a copy of the 4.2.1 build with
`setcap cap_net_admin,cap_sys_ptrace,cap_checkpoint_restore=ep`) so criu can snapshot an
established TCP connection via `TCP_REPAIR` instead of force-closing it. Both ends of the
inter-rank TCPStore connection (launcher + both scheduler ranks) are inside the dump, so
the connection can be restored intact.

New wrappers: `scripts/tp2-exp/checkpoint-keeptcp.sh` / `restore-keeptcp.sh`, which set
`CRIU=/usr/local/sbin/criu-netadmin` and swap `--tcp-close` for `--tcp-established`
(`CRIU_TCP_ARGS` / `RESTORE_TCP_ARGS` overrides added to `utils/common.sh` and
`restore.sh`; TP=1 defaults unchanged). `--network-lock skip` kept — criu's default
iptables lock fails under `--unprivileged` (`netfilter.c:105: Iptables configuration
failed`), and skip is safe here since every connection peer is inside the dump.

**Result — first restore from a fresh checkpoint, both naive and thin: `GATE: PASS`.**

| phase | naive | thin (`--enable-memory-saver`, release weights+kv before dump) |
|---|---:|---:|
| launch + warm | 81.9 s | 68.8 s |
| weight+kv release before dump | — | 0.4 s (GPU −9944 MiB/rank, VmRSS Δ 0) |
| cuda-checkpoint GPU evict (both ranks) | 21.7 s | 4.3 s |
| criu dump (`--tcp-established`, no `--tcp-close`) | 40.2 s → **29 G** | 13.8 s → **9.5 G** |
| **checkpoint total** | **61.9 s** | **18.1 s** |
| criu restore | 17.5 s | 7.5 s |
| cuda-checkpoint GPU reattach | 11.2 s | 5.8 s |
| resume_memory_occupation | — | 0.2 s |
| update_weights_from_disk | — | 5.9 s |
| **restore → serving** | **29.2 s → 3.12× vs 91.2 s cold** | **19.8 s → 4.62× vs cold, 1.18× vs naive** |
| bench | **407.7 tok/s, 64/64, 0 errors** | **409.3 tok/s, 64/64, 0 errors** |

Both produced byte-identical greedy output (`'Paris. The capital of Germany is Berlin...'`).

- **3b is gone**: the TCPStore connection survives, `ProcessGroupNCCL`'s HeartbeatMonitor
  doesn't hit a broken pipe, and — the real test — the bench issues 64 actual generation
  requests (cross-rank all-reduce) that all succeed. No `TORCH_NCCL_ENABLE_MONITORING=0`,
  no process-group rebuild. Holds for the thin path too — `resume_memory_occupation` +
  `update_weights_from_disk` run against the restored (still-connected) process group
  without incident.
- **3a is gone too**: criu restore succeeded on the *first* try in **both** runs (was
  0-for-2 before). Supports the guess in 3a that the bind race came from `--tcp-close`
  restoring a rank-shared socket as two independent `bind()`s — with `--tcp-established`
  the socket is restored once and inherited through the fd table.

### N-times restorable at TP=2 — fixed

First pass: restore **#2** from the same snapshot failed:

```
21213: Error (criu/files-reg.c:2353): Can't open file dev/shm/psm_8b0b10ec on restore: No such file or directory
```

`/dev/shm/psm_*` (a libfabric/PSM shared-memory segment, only present at TP>1) still had
`st_nlink >= 1` at dump time, so criu recorded it as an open-by-path, not a ghost. The
first restore's process unlinks it on exit, so restore #2 had nothing to open — the exact
same bug class as the semaphore-reuse issue, which `drop_tree_sems()` already fixed for
`/dev/shm/sem.*` only. (Reproduced identically on the thin snapshot, `psm_e5e5bd4c` —
orthogonal to naive vs. thin.)

**Fix:** `drop_tree_sems()` → `drop_tree_shm()` in `utils/common.sh`, widened to also
match `/dev/shm/psm_*` and to gather the tree's referenced inodes from `/proc/*/fd` as
well as `/proc/*/maps` (the psm segment is held as a plain fd, not always mmap'd). Same
mechanism as before: unlink by inode pre-dump so criu ghosts the content.

Verified on a fresh naive TP=2 keep-tcp snapshot (`run-20260827T102951Z`): `drop_tree_shm`
reported `unlinked: 3` (was 2 = semaphores only), image carries 23 ghost files (was 22),
and **3/3 restores from the one snapshot PASS** — criu restore first-try each time (no
bind race), byte-identical greedy output, 406.3 / 407.7 / 407.2 tok/s, 0 errors. The
"one checkpoint per restore measurement" constraint is now gone at TP=2 as well.
