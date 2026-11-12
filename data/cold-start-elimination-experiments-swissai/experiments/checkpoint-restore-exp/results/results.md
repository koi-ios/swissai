# Results — 2x RTX 3060 VM (Phase 0–3)

Date: 2026-08-20. Machine: full-root KVM VM (`hostname=ubuntu`), 2x RTX 3060 (12 GB
each), driver 580.95.05, Ubuntu 22.04.5 LTS, kernel 6.8.0-59-generic. Not a container
(no `/.dockerenv`, `/proc/1/cgroup` shows `0::/init.scope`, `CapBnd` is the full set,
`Seccomp:0`) — this is the capable machine the plan's Phase 0 blocker report said was
needed after the first vast.ai Docker-container instance turned out structurally unable
to run CRIU.

## Model deviation from the local-3090 baseline

`experiments/checkpoint-restore-local-exp/` measured Apertus-8B (16.1 GB bf16 weights)
on a 24 GB RTX 3090. This box's GPUs are RTX 3060, **12 GB VRAM each** — Apertus-8B's
weights alone don't fit on one, regardless of `mem-fraction-static`. Per direction, this
run stays single-GPU (no TP=2, no fp8 quantization — both were considered and rejected:
TP=2 was out of scope for this pass, fp8 would break the byte-identical-output check)
and instead swaps to **Qwen2.5-3B-Instruct** (3.09B params, ~6.2 GB bf16 — verified via
the HF API before downloading), which fits comfortably in 12 GB with room for KV cache.

**Consequence: the numbers below are not comparable in absolute terms to the
local-3090's Apertus-8B numbers.** Different model, different GPU generation/VRAM.
(CUDA graphs run on both sides of this comparison — see the environment-fixes section
below; an earlier version of this doc had them off, which was a workaround for a bug
that turned out to be unrelated to graph capture.) What *is* comparable, and is the
actual point of this pass, is whether the checkpoint/restore **mechanism** still works
unprivileged on a different box, and what **speedup ratios** it produces there.

## Phase 0 — feasibility gate, re-run fresh on this box

| gate | proves | result |
|---|---|---|
| `probe_env` | `CapBnd` is the full set, `Seccomp:0` — not container-restricted | recorded |
| `criu-cpu` | CRIU can dump→disk→restore a plain process | ✅ **PASS** |
| `criu-gpu` | `cuda-checkpoint` + CRIU full round trip through a **kill** | ✅ **PASS** |
| `gate_pytorch_cuda` (3a) | Python+torch.cuda round trip (de-risk) | ✅ **PASS** |

Two environment problems surfaced and were fixed before any gate could pass, both
worth keeping in mind if this is repeated on a fourth machine:

1. **Ubuntu 22.04's apt `criu` (3.16.1) segfaults on `restore`** against this box's 6.8
   kernel (`killed by signal 11: Segmentation fault`). Built criu **4.2.1 from source**
   (same version the local-3090 box used) — passes cleanly. Unlike the vast.ai
   container, `setcap` on the built binary actually takes effect here (real ext4 root,
   not an overlay mounted `nouserxattr`).
2. Job-control gotcha in gate scripts: backgrounding the test counter process without
   `disown` occasionally corrupted `criu dump`'s view of the process (restore segfault
   traced to this before the criu-version issue was found) — added `disown` to
   `gate_criu_cpu.sh` and `gate_criu_gpu.sh`.

## Phase 1–2 — new experiment directory

Built per `PLAN.md`'s layout: `config.sh`, `scripts/utils/{common.sh,probe_env.sh,
gate_criu_cpu.sh,gate_criu_gpu.sh,gate_pytorch_cuda.sh}`, `scripts/{setup_env.sh,
baseline.sh,checkpoint.sh,restore.sh,cache_tools.py}`, `shim/uvloop/`, `src/`.
`checkpoint.sh [--thin]` / `restore.sh [--cold]` merge the old 4-script naive/thin pair
as planned, `meta.txt`'s `released_tags` auto-detects which restore path to take.

Deviations from the plan's Phase 1–2 text, both load-bearing:

- **servekit**: the plan assumed the servekit submodule would just be there; it wasn't
  initialized on this box. Rather than depend on `git submodule update` (SSH remote),
  `setup_env.sh` now does `pip install "servekit @ git+https://github.com/eth-easl/
  servekit@<pinned-commit>"` — portable to a machine with no SSH key configured for the
  private-submodule remote.
- **CRIU_ARGS**: the old local-3090 scripts passed `--unprivileged` to every `criu`
  invocation (needed there because criu ran via a file-capability grant to a non-root
  user). This box runs as full root with the complete capability set — `criu`'s
  `--unprivileged` flag doesn't even exist in criu 4.2.1's CLI when invoked that way, so
  it's dropped from `CRIU_DUMP_ARGS`/gate scripts entirely here.

## Environment fixes needed before Phase 3 would run at all

None of these are CRIU/checkpoint-restore problems — they're generic "new box, no CUDA
toolkit yet" problems, but each one blocked `baseline.sh` until fixed:

1. **No `nvcc`.** `apt install nvidia-cuda-toolkit` initially pulled CUDA **11.5**
   (Ubuntu 22.04's stock package) — too old to build SGLang/flashinfer's JIT kernels,
   which need `cuda_fp8.h` (CUDA 11.8+). Installed `cuda-toolkit-12-8` from NVIDIA's own
   apt repo (already configured on this box) to match torch's `cu128` wheels.
2. **nvcc-as-symlink breaks its own path heuristics.** A plain `symlink /usr/local/bin/
   nvcc -> .../cuda-12.8/bin/nvcc` made nvcc unable to find its own `include/` (nvcc
   derives its install root from argv0's directory, and a symlink one level removed from
   the toolkit's own `bin/` breaks that). Fixed by using a tiny wrapper **script**
   (`exec /usr/local/cuda-12.8/bin/nvcc "$@"`) instead of a symlink.
3. **Same heuristic breaks `torch.utils.cpp_extension.CUDA_HOME`.** Torch derives
   `CUDA_HOME` the same way (`dirname(dirname(which nvcc))`), which — because of the
   wrapper script living at `/usr/local/bin/nvcc` rather than inside the toolkit's own
   `bin/` — resolved to `/usr/local` instead of the toolkit root, feeding a bogus
   `-L/usr/local/lib64` into every JIT kernel's link step (`cannot find -lcudart`).
   Fixed by exporting `CUDA_HOME=/usr/local/cuda` explicitly in `utils/common.sh`.
4. **Stale CUDA 11 runtime shadowed the CUDA 12 one at link time.** The now-removed
   `nvidia-cuda-toolkit` (11.5) apt package had left `/usr/lib/x86_64-linux-gnu/
   libcudart.so -> libcudart.so.11.0` (via the `libcudart11.0` dependency package) in
   the default linker search path, ahead of the CUDA 12.8 toolkit's own lib dir. The
   JIT-compiled rope kernel silently linked against the CUDA **11** runtime while torch
   itself was on CUDA **12** — two separate CUDA runtime instances in one process, so a
   stream handle minted by one was meaningless to the other. Symptom: `RuntimeError:
   ... CUDA error: invalid resource handle` from `sglang/jit_kernel/csrc/elementwise/
   rope.cuh`, on the very first forward pass (not graph-capture-specific, despite
   looking like it at first). Fixed by `apt purge nvidia-cuda-toolkit libcudart11.0` and
   forcing `LD_LIBRARY_PATH` to the venv's own bundled `nvidia/cuda_runtime/lib` ahead
   of any system path, in `utils/common.sh` (this second half stays as a durable
   safeguard even after the purge, in case a future package reintroduces the shadow).
5. **CUDA graph capture failure was the SAME bug as #4, not a separate one.** The very
   first symptom actually seen (before any of fixes 1–4) was `invalid resource handle`
   during `Capture cuda graph`, which looked graph-capture-specific and was initially
   worked around with `--disable-cuda-graph`. That workaround was a red herring: once
   #4's stray CUDA 11 `libcudart` was purged and `LD_LIBRARY_PATH`/`CUDA_HOME` were
   fixed, re-testing **with CUDA graphs re-enabled** (`DISABLE_CUDA_GRAPH=0`, now the
   default in `config.sh`) launched and served cleanly — `cuda graph: True` in the
   engine logs, no error. The graph-capture crash and the plain-eager-mode crash were
   both just "a JIT kernel linked against the wrong CUDA runtime," surfacing at whatever
   point the process happened to hit a JIT-compiled kernel first. No workaround needed;
   `DISABLE_CUDA_GRAPH` stays in `config.sh` as a documented off-switch, defaulted off.

## Phase 3 — measured results (this box, Qwen2.5-3B-Instruct, GPU 0 only)

Measured with CUDA graphs **enabled** (`cuda graph: True` in engine logs on all three
runs — see environment-fixes item 5 above for why an earlier pass had them off).

| path | time to serving | vs local-3090 (Apertus-8B, for shape only — not comparable) |
|---|---:|---|
| cold launch (weights + JIT cold) | **109.8 s** | 59.3 s |
| naive restore (15 GB snapshot, cache cold) | **23.3 s (4.72x)** | 18.2 s (3.3x) |
| thin restore (4.6 GB snapshot + weight reload, cache cold) | **17.4 s (6.31x cold, 1.34x vs naive)** | 10.5 s (5.6x) |
| checkpoint (one-time, naive) | 34.7 s → 15 GB snapshot | 21.4 s → 26 GB |
| checkpoint (one-time, thin) | 11.6 s → 4.6 GB snapshot | 4.0 s → 6.0 GB |

All three server runs (baseline, naive restore, thin restore) produced **byte-identical
greedy output** for the same probe prompts, e.g.:

```
'The capital of France is' -> 'Paris. The capital of Germany is Berlin. The capital of '
'Explain in one sentence why the sk' -> "The sky appears blue because the Earth's atmosphere scat"
```

Post-restore throughput was consistent across baseline (252.4 tok/s), naive restore
(253.9 tok/s), and thin restore (253.9 tok/s) — as expected once CUDA graphs run on all
three; the throughput mismatch in an earlier draft of this doc (with graphs off) was an
artifact of that workaround, not a property of the restore path. The thin checkpoint's
weight release was verified genuine, not just relocated: `S1_RSS` and `S2_RSS` (host
process-tree VmRSS, straddling `/release_memory_occupation`) differed by 0 MiB while GPU
memory dropped by 10,038 MiB.

## Phase 4 — reusable snapshots (ported from local-3090 commit `50db0ba`, 2026-08-20)

The local-3090 experiment later found (`checkpoint-restore-local-exp` commit `50db0ba`,
merged into `main` after this exp's Phase 0–3 pass was first written) that a snapshot
taken with `--link-remap` is **single-use**: that flag hard-links SGLang's `/dev/shm`
POSIX semaphore into the live filesystem, the first restore consumes the link, and the
restored process deletes the inode on exit — so restore #2 from the same image fails.
The fix is two-part: `drop_tree_sems()` unlinks the semaphore *by inode* (its path in
`/proc/*/maps` is stale — Python renames it after `mmap`) before the dump, so criu embeds
its content in the image (a "ghost" file) instead of needing `--link-remap`; separately,
criu also records the checkpointed server's fd-1 log by path *and size* and refuses to
restore onto a file whose size has changed (including changes the restored server itself
appended), so the log must be reset to its dump-time size before every restore.

This was ported into `checkpoint.sh`/`restore.sh`/`utils/common.sh` here (`drop_tree_sems`,
`REUSABLE_DUMP_ARGS` in place of `CRIU_DUMP_ARGS`, `log`/`log_size` in `meta.txt`, log
truncate in `restore.sh` before every restore) and verified on this box:

- **Naive**: 1 snapshot (`naive_checkpoint/run-20260820T143316Z`, 15 GB), **3/3 restores**
  from the same image — all `GATE: PASS`, byte-identical output each time
  (`'Paris. The capital of Germany is Berlin...'`), throughput 254.1 / 254.0 / 253.4 tok/s.
- **Thin**: 1 snapshot (`thin_checkpoint/run-20260820T144112Z`, 4.6 GB), **2/2 restores**
  from the same image — both `GATE: PASS`, same byte-identical output, 253.9 / 253.6 tok/s.
  The local-3090 write-up never verified reusability on the thin path (only naive); it
  holds here too — reusability comes from the semaphore/log fix, which is orthogonal to
  whether weights were released before the dump.

**A third bug surfaced and was fixed while testing this, not present in the local-3090
write-up:** `criu restore` failed with `Can't fork for <pid>: File exists` on restore #2,
because criu restores onto the *exact original pids*, and the prior restore's process —
just killed — was still a zombie holding that pid slot (a fixed `sleep 1` after
`kill_servers` wasn't reliably long enough; the pid frees only once something reaps the
zombie). `restore.sh` now actively waits (up to 10 s, polling `/proc/<pid>`) for every pid
in `meta.txt`'s `gpu_pids` to actually disappear before calling `criu restore`, instead of
a fixed sleep — this is what makes repeated restores from one snapshot reliable rather
than merely possible.

**Consequence for future runs on this box:** the "one checkpoint per restore
measurement" constraint is gone here too, same as the local-3090 finding — A/B'ing
restore latency across N runs now needs one `checkpoint.sh` call, not N.

## Bottom line

The checkpoint/restore **mechanism** — `cuda-checkpoint` + CRIU dump/restore of a
warmed, unprivileged SGLang server, including the thin variant that releases and
reloads weights, with CUDA graphs enabled — reproduces cleanly on a second, materially
different machine (2x RTX 3060 VM vs. the original bare-metal RTX 3090): same
qualitative shape (naive restore faster than cold, thin restore faster still,
byte-identical output, consistent post-restore throughput, thin checkpoint one order of
magnitude smaller on disk), similar speedup ratios (naive 4.72x here vs 3.3x there; thin
6.31x here vs 5.6x there). The absolute numbers differ because the model and GPU
generation differ — see the deviations section above — so this is validation of the
*technique*, not a same-hardware regression check against the local-3090 numbers.
