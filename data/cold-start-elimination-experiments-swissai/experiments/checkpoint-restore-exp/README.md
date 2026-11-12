# CRIU checkpoint/restore of a warm SGLang server (local machine)

This is the experiment behind section V (CRIU) of the [main README](../../README.md): can we
checkpoint a warm, already-serving SGLang server to disk (NVIDIA `cuda-checkpoint` for the
CUDA contexts + CRIU for the CPU process tree) and restore it faster than a cold launch?

This cannot run on CSCS (CRIU needs `CAP_CHECKPOINT_RESTORE` + `CAP_SYS_PTRACE`, and enroot
containers add a seccomp filter CRIU refuses to run under — see the main README), so it runs
on a local machine: **2x RTX 3060 (12 GB each), Ubuntu 22.04, SGLang v0.5.10**,
`Qwen2.5-3B-Instruct` (the box's predecessor ran Apertus-8B on a 24 GB RTX 3090; these 12 GB
cards don't fit it, so the model is swapped — the numbers measure the *mechanism* and speedup
ratios, not absolute parity).

## Results

**TP=1** (GPU 0 only), full writeup in [`results/results.md`](results/results.md):

| path | time to serving | speedup vs cold start | checkpoint time |
|---|---|---|---|
| cold launch | 109.8s | 1x | — |
| naive restore | 23.3s | 4.72x | 34.7s (15GB snapshot) |
| thin restore | 17.4s | 6.31x | 11.6s (4.6GB snapshot) |

*Naive* dumps everything including the GPU weights; *thin* frees the weights right before the
dump via SGLang's `/release_memory_occupation` + `/resume_memory_occupation` endpoints and
reloads them from disk on restore. All runs produce byte-identical greedy output and
consistent post-restore throughput (252–254 tok/s). Snapshots are reusable: one naive image
was restored 3/3 times, one thin image 2/2 times.

**TP=2** (`--tensor-parallel-size 2`), full writeup in
[`results/tp2-exp/results.md`](results/tp2-exp/results.md):

| path | time to serving | speedup vs cold start | checkpoint time |
|---|---|---|---|
| cold launch | 91.2s | 1x | — |
| naive restore | 29.2s | 3.12x | 61.9s (29GB snapshot) |
| thin restore | 19.8s | 4.62x | 18.1s (9.5GB snapshot) |

TP>1 additionally needs `CAP_NET_ADMIN` to preserve the TCPStore/NCCL connection across the
dump (`--tcp-established`); with plain `--tcp-close` the restored rank crashes when the
heartbeat thread polls the dead connection, and working around it at the application level
(rebuild the process group) would invalidate the captured CUDA graphs — defeating the point.

**Minimal capability set** for the unprivileged round trip, in
[`results/caps-exp/results.md`](results/caps-exp/results.md): `cap_checkpoint_restore` +
`cap_sys_ptrace` as a file capability on the `criu` binary, executed by a non-root user —
nothing else is required.

## What it takes to make SGLang checkpointable (and reusable)

Each discovered the hard way; full rationale in [`NOTES.md`](NOTES.md):

1. `cuda-checkpoint` **every** process holding `/dev/nvidia` VMAs (launcher + scheduler +
   detokenizer), not just the one `nvidia-smi` lists as a compute app.
2. No `io_uring` anywhere (CRIU can't dump it): a uvloop→epoll shim ([`shim/`](shim)) loaded
   via `.pth`, plus `USE_LIBUV=0` for torch TCPStore's bundled libuv.
3. CRIU flags: `--tcp-close`/`--tcp-established`, `--network-lock skip`, `--shell-job
   --file-locks --ext-unix-sk` (see `CRIU_DUMP_ARGS`/`REUSABLE_DUMP_ARGS` in
   `scripts/utils/common.sh`).
4. **Reusable snapshots** require unlinking the tree's held-open `/dev/shm` files (POSIX
   semaphores; `psm_*` libfabric segments at TP>1) before the dump, so CRIU embeds them as
   ghost files instead of hard-linking a path the first restore consumes; and resetting the
   server log to its dump-time size before every restore (CRIU records it by path *and*
   size). At TP=2, `--ghost-limit` must be raised (NCCL's `/dev/shm` ring buffer is tens of
   MB, over the 1 MB default).

## Repo layout

```
config.sh                     # per-machine config (model, port, GPU pinning, ...)
scripts/                      # what you run: baseline / checkpoint [--thin] / restore [--cold]
scripts/utils/                # env probe + feasibility gates + shared helpers
scripts/tp2-exp/              # TP=2 wrappers around the shared scripts
scripts/caps-exp/             # minimal-criu-file-caps experiment
shim/uvloop/                  # uvloop -> epoll shim (io_uring is not CRIU-dumpable)
src/                          # CUDA counter used by the feasibility gates
results/                      # per-run meta.txt + bench JSON, phase profiles, writeups
PLAN.md / NOTES.md            # working docs (PLAN references a predecessor experiment,
                              # checkpoint-restore-local-exp, which lives in the internal
                              # serving-stack repo, not here)
```

## Reproducing

On a bare-metal machine with NVIDIA GPUs, no container, and `criu` ≥ 4.0 available (Ubuntu
22.04's apt `criu` 3.16.1 segfaults on restore against a 6.8 kernel — build 4.2.1 from
source if you hit that):

```
bash scripts/setup_env.sh          # venv, sglang, servekit, cuda-checkpoint binary
bash scripts/utils/probe_env.sh    # capabilities / seccomp / ptrace_scope sanity check
bash scripts/utils/gate_criu_cpu.sh && bash scripts/utils/gate_criu_gpu.sh
bash scripts/baseline.sh --cold    # cold-start reference
bash scripts/checkpoint.sh [--thin]
bash scripts/restore.sh --cold
```
