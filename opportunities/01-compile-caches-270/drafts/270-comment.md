<!-- Draft comment for swiss-ai/model-launch#270. Not posted. Review numbers marked "predicted"
     before sending; replace them with the Modal results once harness/summarize.py has run. -->

I looked at what both base images write at startup, for vLLM and for SGL as asked above. I built a
patch for both and a small tool that keeps the variable list correct as the images change. Summary
below; patches and tool are linked at the end.

**Why nothing persists today.** No env TOML sets a cache variable. The EDFs don't mount `/users`, so
`$HOME` inside the container lives in the writable layer and disappears with the job. On top of that,
vLLM files its Triton and Inductor caches under a per-compile-hash directory, and that hash includes
the checkpoint path.

**What each image writes, checked against source** (sglang_base = SGLang 0.5.20; vllm_base: vLLM
0.32.0 as far as I can tell; both on torch 2.13 / Triton 3.7.1):

| cache | variable | relocatable? | read-only copy works? | note |
|---|---|---|---|---|
| vLLM torch.compile/AOT | `VLLM_CACHE_ROOT` | yes | no | keyed by **checkpoint path** (a new Apertus checkpoint recompiles its graphs) |
| Triton | `TRITON_CACHE_DIR` | **no** (group files hold absolute paths) | hits only | vLLM keeps it per compile hash unless set, so set it to share kernels across checkpoints |
| TileLang (DeepSeek-V4.x mHC) | `TILELANG_CACHE_DIR` | yes | hits only | **nothing sets it today** |
| DeepGEMM (DeepJIT) | `DG_JIT_CACHE_DIR` | yes | yes (`rw:ro` list) | vLLM puts it in `VLLM_CACHE_ROOT/deep_gemm`; paths that skip lazy init fall back to `~/.dj` |
| FlashInfer JIT | `FLASHINFER_WORKSPACE_BASE` | no (ninja) | no | versioned path `<ver>/<arch>/cached_ops` |
| SGLang JIT (custom all-reduce v2, KV store) | `SGLANG_JIT_CACHE_DIR` | yes | hits only | does **not** follow `SGLANG_CACHE_DIR`; ~17 s of the measured 54 s saving |
| SGLang root | `SGLANG_CACHE_DIR` | — | — | `import sglang` puts Triton/Inductor/ComputeCache/FlashInfer/DeepGEMM/cute_aot under it |
| tvm-ffi | `TVM_FFI_CACHE_DIR` | no | **hangs** on its lock | key has no GPU arch |
| CuTe DSL, Inductor | `CUTE_DSL_CACHE_DIR`, (vLLM sets Inductor) | partly | Inductor: no (lock files) | |

All 1,137 file:line citations behind this table are checked mechanically against the vendored
sources. The relocation and read-only columns for Inductor, tvm-ffi and TileLang were also measured
by running the real libraries on CPU.

**Design (patch below).**
- **Path.** `<shared-caches>/<image key>/<gpu arch>/<component>`.
- **Image key.** A `<image>.sqsh.digest` sidecar if the images CI writes one; otherwise a hash of the
  file size and the 96-byte squashfs superblock. That is one small read, and it changes on every
  in-place rebuild.
- **GPU arch.** From `nvidia-smi` compute_cap.
- **Overlay.** Each node seeds a node-local overlay at a fixed path, `/tmp/sml-jit-cache`. A fixed
  path is required because Triton and ninja store absolute paths. `{jit_cache}` in the env TOML
  resolves there, the same way `{arch}` is resolved.
- **Promote.** After `/health` answers, and every 30 min after that, each node publishes new entries:
  - copy to `.staging`, then `rename` a directory or `link` a file into place;
  - the first writer wins and nothing is overwritten;
  - readers never see partial entries, and no cross-node `flock` is needed (I don't know capstor's
    flock mode);
  - lock files, temp files and the per-node P2P JSONs are never shared;
  - files are made group-readable (some writers create 0600 files).
- **Modes.** `--jit-cache overlay|shared|off` for the mode and `--jit-cache-root` for the location.
  Users without write access still get warm starts.
- **Housekeeping.** `gc` evicts least-recently-used digests by size and age.
- **Tests.** 482 unit tests pass, including a run of the generated bash checking that its key is
  byte-identical to the helper's. 16 concurrent promoters each land an entry once; 8 jobs that built
  different kernels all land theirs.

**Why the shared cache isn't a plain read-write directory on capstor.**
- With umask 022, the first user's directories lock out other infra01 members.
- Inductor and FlashInfer need writable lock files even on a hit; tvm-ffi spins on its lock forever.
- Inductor's temp names (`.<pid>.<tid>.tmp`) can collide across nodes.
- Small-file lookups are metadata-heavy on HDD Lustre.

`--jit-cache shared` remains available as an option.

**Expected gain (predicted, not measured yet).**
- SGL: about the 54 s per 70B TP4 launch the lab measured.
  - servekit's three variables miss `SGLANG_JIT_CACHE_DIR` on 0.5.20, so about 17 s of that is
    currently lost even under servekit.
- vLLM: the TileLang and DeepGEMM warmups (40 and 8–10 min) should drop to load time after the first
  launch of an image digest. I have not verified how much of the 40 min is TileLang compile; vLLM's
  TileLang kernels are not compiled per captured batch size.
- I have a Modal run ready to measure cold vs warm vs servekit's three variables on both images
  (Apertus-8B TP1/TP2 and a small FP8 MoE). I'll post the numbers here.

**Pre-warming in the images CI** would make even the first launch of a new image warm:
1. after promotion, run one `sml` launch per flagship model with this protocol;
2. write the sidecar digest.

Happy to draft that for model-launch-images if useful.

**Keeping it correct.** `cache-audit` runs a launch cold in a fresh HOME, traces every write
(strace / inotify / snapshots), maps each root to its variable, and verifies the mapping by re-running.
It exits non-zero when a new image writes a cache the env TOML doesn't redirect. For example, the
0.5.16 → 0.5.20 move from `TVM_FFI_CACHE_DIR` to `SGLANG_JIT_CACHE_DIR` is exit 12. A GPU-free static
mode checks the installed packages in seconds.

Patches (4 commits on `feat/shared-jit-caches`, plus a variant of the env commit for #272's renamed
files), the tool and the evidence: <link>.

Questions:
1. Is `/capstor/.../sml/shared-caches` group-writable by infra01, and is capstor mounted with `flock`?
2. Can model-launch-images write `<image>.sqsh.digest` next to each released image?
3. Should the default be `overlay`, or `off` until one Clariden A/B confirms the gain?
