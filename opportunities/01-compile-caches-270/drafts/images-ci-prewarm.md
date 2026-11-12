# Pre-warming JIT caches in the images CI: evaluation and design (draft)

For swiss-ai/model-launch-images, which is private: this is a design note only, with no patch. It builds on the `sml_jit_cache.py` protocol in
`patches/0001` (`<root>/<image key>/<gpu arch>/`, seed and promote).

## The idea

The CI already builds every image on the cluster through FirecREST (model-launch history: `build_image.py`,
removed in #259). After it publishes an image, it would also do two things.

1. **Write the key.** It writes `ci/<image>-<arch>.sqsh.digest` next to the image. The file holds the OCI digest the CI
   pushed to GHCR, in the form `sha256:<hex>`.
   - sml prefers this sidecar over the superblock hash. It is cheap, exact and auditable.
2. **Warm the caches.** It runs `sml` once per warm-set model with `--jit-cache overlay`, in a short job:
   - the job runs until `/health` answers and then stops;
   - the normal promote step publishes the compiled kernels into `<root>/<digest>/<arch>/`.
   Every later launch of that image, including its very first user launch, then seeds a warm cache.

## Why it is the strongest version of #270

- **Images change in place.** Each rebuild changes the digest and starts a cold cache directory. Without pre-warming, the
  first user launch after every rebuild pays the full compile:
  - DeepSeek-V4.1-Flash: about 40 min (#270);
  - GLM-5.3: 8–10 min of DeepGEMM warmup;
  - SGLang 70B TP4: about 54 s.
  That user is often a maintainer validating the new image under a deadline.
- **Cost moves off the critical path.** CI already holds the GPU nodes for build validation. Warming then costs
  GPU-minutes in CI instead of minutes of downtime during a `--consecutive` handover.
- **Caches are shared across checkpoints.** TileLang, DeepGEMM, FlashInfer and Triton entries are keyed by kernel, not by
  checkpoint path. Warming one representative model per architecture family therefore warms all its checkpoints. The
  Apertus family alone had 43 short-lived checkpoints in the OpenTela trace.
  - The exception is vLLM's torch.compile and Inductor artifacts, which are keyed by checkpoint path.

## What it cannot do

- **Model-specific shapes.** Kernels are compiled per shape: hidden size, heads, expert count, TP degree and quantization.
  - A tiny model warms the shape-independent kernels: the custom all-reduce JIT, samplers and some FlashInfer ops.
  - Warming the 40 minutes of DeepSeek-V4.1 requires launching DeepSeek-V4.1 with its production TP layout.
  - So the warm set is a curated list: the flagships plus one model per architecture family.
- **vLLM torch.compile artifacts.** These are keyed by checkpoint path. A warmed Apertus checkpoint does not warm a new
  one, although that new one still hits Triton and the other kernel caches.
  - A small vLLM change would extend reuse across checkpoints: key the compile cache on the HF config instead of the
    model path.
- **GPU cost per rebuild.** Images are rebuilt "from main" frequently. Sensible policies:
  - warm only on promotion to the release path, not for every PR build;
  - warm the flagships only when their engine version changed. The audit's static mode tells you which libraries changed.

## Cost and benefit (estimate, not measured)

- **Cost.** Warming DeepSeek-V4.1-Flash costs one 2-node job of about 70 min per image release.
- **Benefit.** That is the compile time each launch otherwise pays:
  - every `--consecutive` handover;
  - every restart;
  - every user launch of that image.
- **Break-even.** It is reached as soon as the model is launched twice per image release. On the platform's long-running
  services that happens within a day.

## CI check (no GPU)

`python -m cache_audit static` in the freshly built image (seconds, CPU only) fails the build in two cases:
- a library renamed its cache variable (exit 12);
- a library grew a new cache variable the env TOML does not cover (exit 11).

The GPU audit (`cache-audit run --env-toml envs/<image>.toml -- warmup`) can run inside the warm job itself, since the
GPUs are already allocated. It fails when a launch writes any cache outside `{jit_cache}`.

## Steps

1. After `release`: write `${IMAGE}.sqsh.digest` from the GHCR push digest. This needs no hashing of the image.
2. Warm job:
   ```bash
   sml advanced --slurm-environment envs/<image>.toml --jit-cache overlay ...
   ```
   Wait for `/health`, wait for the promote log line, cancel.
3. Gate: run `cache-audit static` (CPU) in the image.
4. Optionally run `cache-audit run --env-toml` in the warm job.
