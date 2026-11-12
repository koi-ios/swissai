# servekit for vLLM: fast weight loading, ported and checked against vLLM's own loaders

*12 Nov 2026. Target: vLLM 0.33.0 (4ba787ec) and servekit main (cd8f0b8). Built and tested without a GPU; the GPU runs are prepared, not yet run.*

## The problem, with numbers

- Every model added to the Swiss AI platform since September runs on vLLM. servekit, whose fast loading the lab measured, wraps only SGLang; its roadmap item "Support vllm fast weight loading" is open.
- On SGLang, Youssef Boughizane's measurements show the stakes for Llama-3.1-70B at TP 4 on capstor, an HDD-backed Lustre:
  - default mmap loading takes 453.7 s on Bristen and 794.2 s on Clariden;
  - servekit's presharding, `/dev/shm` staging and `--overlap` bring that to 9.7 s and 6.0 s.
- The one vLLM run the lab has on capstor is servekit's own fixture `tests/fixtures/llama70b-vllm-ngc.log`:
  - setup: Clariden job 2918061, vLLM 0.24, same model, TP 4, a single run;
  - vLLM detected Lustre and auto-prefetched into the page cache on all four ranks;
  - yet `Loading weights took 231.17 seconds`, out of about 422 s to ready.
- So vLLM's built-in answer to network file systems does not close the gap on capstor. The likely reasons, read from the code:
  - each file gets one sequential reader, while a Lustre file needs dozens: one `dd` reader on a capstor file gives 0.36 GB/s and 64 give 6.8 GB/s (median of 3);
  - the prefetch starts only after engine startup;
  - the mmap reads apparently did not hit the prefetched pages. That last one is a hypothesis to check on Alps.

## Related work, and what is new here

Notes with citations are in `research/related_work.md` and `research/vllm-loading-paths.md`.

- **ServerlessLLM (OSDI '24).** The closest prior art: per-GPU partitioned checkpoints, multi-threaded `O_DIRECT` reads into pinned RAM, and loading overlapped with engine init. It uses its own format and a patched vLLM, on local NVMe.
- **fastsafetensors (arXiv 2505.23072) and InstantTensor.** They read unmodified checkpoints with deep direct I/O, then exchange tensors over the world group (NCCL), so every GPU receives the whole model.
- **Run:ai Model Streamer.** Concurrent range reads per file. Its `runai_streamer_sharded` variant reads exactly servekit's presharded files, but only inside the load step.
- **λScale and BlitzScale (OSDI '25), ModelExpress / NIXL, and vLLM's `vllm preload` + `ipc_cache`.** They load from GPUs that already hold the weights, so they speed up the second replica or a restart, not the first cold load.
- **None of these works reports Lustre numbers.**
- **What is new here:**
  - servekit's measured Lustre recipe on vLLM, with no fork of vLLM;
  - a code-level audit of when vLLM's `sharded_state` is safe. vLLM's `sharded_state` stores weights *after* `process_weights_after_loading` and runs that hook *again* on reload (base_loader.py:88, 109). It also only warns while narrowing or casting tensors (sharded_state_loader.py:224-244). So a checkpoint presharded for TP 4 and served at TP 2 answers fluently from wrong weights;
  - the fix that vLLM's own `ipc_cache` loader already contains, and `sharded_state` lacks.

## What was built

The servekit change is `patches/`: 3 commits on cd8f0b8; the main commit adds 1,838 lines, 770 of them tests. It applies cleanly to a fresh clone. The command is `servekit launch --servekit-artifact-path DIR [--overlap] -- vllm serve MODEL -tp N`.

1. **Prepare.** vLLM's own sharded-state saver runs from the serve command's flags, with eager mode and one replica forced. The manifest also pins the vLLM minor version, because the shards are post-processed weights. On a multi-node layout, a follower node serves headless until the head writes the manifest.
2. **Rewrite and stage.**
   - The positional model path, `-tp/-pp` and `--nnodes/--node-rank/--master-addr` are all read.
   - Each node stages only its own ranks' files with servekit's measured sliced stager. vLLM's pipeline files are named `model-pp-*-tp-*`, not SGLang's `-rank-`.
   - `--served-model-name` is pinned to the original path; otherwise vLLM would rename the model to `/dev/shm/...` for every client and router.
3. **A vLLM plugin instead of a fork.** It is a `vllm.general_plugins` entry point, which vLLM loads in every worker. It keeps the name `sharded_state`, which vLLM special-cases, and puts a subclass behind it that:
   - waits on servekit's stage barrier just before the read;
   - turns the narrowing and casting warnings into errors;
   - records every rank's load time. vLLM logs `Model loading took` on local rank 0 only, so without these records the slowest rank is invisible, and a headless node cannot tell when to free its copy.

   Under `--overlap`, shards are renamed into place only when complete, so an engine running without the plugin fails loudly instead of reading half-written files.
4. **Guard.** vLLM's own loader serves, untouched, what `sharded_state` cannot round-trip:
   - quantized checkpoints;
   - MLA models: `W_UV` and `W_UK_T` are created after loading (mla_attention.py:1347-1350);
   - expert parallelism, LoRA and speculative decoding;
   - Ray-placed ranks.

   Each of these prints a log line saying why.
5. **Compile caches.**
   - `VLLM_CACHE_ROOT` and TileLang join servekit's JIT caches.
   - vLLM's torch.compile cache key includes the model path (config/model.py:457-525). The staged path is the same on every launch, so caches are written back beside the artifact after the first launch, never overwriting.
   - servekit never sets `VLLM_PLUGINS`, which vLLM hashes into that key.
6. **Measurement.**
   - The prelim harness gains `servekit_serial` and `servekit_overlap` arms beside vLLM's own options.
   - `modal_app.py` runs everything in `vllm/vllm-openai:v0.33.0`; `MODAL.md` is the runbook.

## Results so far

| | what | status |
|---|---|---|
| servekit tests | 195 pass, 3 skip (Linux-only stager); 59 new vLLM tests, incl. an end-to-end `--overlap` run through a mock `vllm serve` subprocess that checksums every shard it reads | **measured** (macOS, no GPU) |
| patches on a fresh clone of cd8f0b8 | `git am` clean, same 195 pass | **measured** |
| prelim harness selftest | 57/57 (the 55 existing checks still pass; the real-stager servekit block runs on Linux only) | **measured** |
| `modal_app.py --dry-run` | OK, app builds with 5 functions; estimated **$13.7** on A10G | **measured** (cost is an estimate) |
| vLLM auto-prefetch on capstor, Llama-70B TP 4 | 231 s weight loading despite prefetch | **measured by the lab** (1 run, vLLM 0.24) |
| servekit-vLLM on capstor | ~1-10 s for Apertus-8B…Llama-70B, as on SGLang (same stager, same file layout, same overlap) | **prediction**, not measured |
| Modal Volume arms (14 arms × 3 repeats × TP 1/2) | — | **not run** (no Modal account here) |

**Decision: which vLLM-native option servekit should use, and when.** It rests on the capstor fixture, Youssef's SGLang measurements and the vLLM code. Rows marked "to measure" are covered by the Modal arms.

| case | servekit should use | why |
|---|---|---|
| Model servekit can preshard (bf16, no MLA), multiprocessing backend, single node or multi-node | its own path: `sharded_state` + per-node sliced staging + `--overlap` | No native option matches staging on Lustre. Auto-prefetch took 231 s on capstor. fastsafetensors (48.6 s on Clariden, SGLang) and InstantTensor read the whole model per node over NCCL. |
| Restart within a job on the same GPUs (OpenTela #73) | `vllm preload` + `--load-format ipc_cache` | Weights stay mapped in GPU memory, so no disk is read at all; staging cannot beat it. servekit could stage the daemon's first load. |
| Ray-placed multi-node (model-launch today) | vLLM's own loader. Measure `runai_streamer_sharded` with `concurrency=64` on a servekit artifact (to measure) | servekit cannot tell which node runs which rank. Run:ai's sharded variant reads only each rank's files, with in-file concurrency, like the stager. Better: move model-launch's vLLM multi-node to `--nnodes/--node-rank`, as it already does for SGLang. |
| Quantized or MLA models (the platform's vLLM flagships) | vLLM's own loader. Never force `--safetensors-load-strategy lazy`. | vLLM's `sharded_state` cannot round-trip them (for MLA the save itself should fail). Raw-byte presharding replayed through the default loader can (next step). |

## How to run it

- **Locally:**
  ```
  cd vendor/servekit && PYTHONPATH=src python -m pytest tests -q
  ./serve-kit-vllm-prelim/tests/selftest.sh
  python modal_app.py --dry-run
  ```
- **On Modal:** see `MODAL.md`. Run `download`, `smoke`, then `experiment_tp1` and `experiment_tp2`; that is about $14 across two accounts.

## What it would take on Alps, and the risks

- **One job pair per model.** For example `prepare` then `launch --overlap`, inside the `vllm_base` image, plus `servekit verify` against the default loader. servekit also needs installing in that image (`pip install`), so the plugin's entry point is visible.
- **Risk: platform fit.** model-launch launches vLLM multi-node through Ray. Per-node staging needs vLLM's multiprocessing backend.
- **Risk: coverage.** Most new platform models are quantized or MLA, so today's guard sends them to vLLM's own loader. For MLA, vLLM's `sharded_state` save should already fail: `_filter_subtensors` calls `view(-1)` on the transposed `W_UV`/`W_UK_T` views before checking contiguity (sharded_state_loader.py:152-158). Two routes would cover these models, in order of preference:
  - **Raw-byte presharding.** Store the checkpoint bytes each rank reads, then replay them through vLLM's ordinary default loader, so `process_weights_after_loading` runs exactly as in a normal load. The sibling preshard-roundtrip session (08) built an engine-agnostic core for this on servekit branch `preshard-roundtrip` (`rawshard/`), plus a tolerance-free per-tensor check (`servekit deepstate`). A vLLM port needs only a hook that wraps vLLM's weights iterator.
  - **A presharded loader on vLLM's `ipc_cache` contract.** Build the model on the meta device, replace parameters instead of copying into them, re-run post-processing under `weights_already_processed()`, and refuse quant methods without `supports_pre_processed_weights`. It belongs upstream in vLLM, but covers only the quant methods that declare support.

- **Risk: correctness.** `servekit verify` stays mandatory per model. The plugin's strict mode catches only shape and dtype mismatches.

## Pitch (for Xiaozhe or Ana)

servekit's Lustre loading now runs on vLLM, with no fork. It ships as a plugin, a guard, per-rank load records and cache write-back, with 59 tests and a $14 Modal run ready.
Your own capstor log shows vLLM's auto-prefetch still spends 231 s loading Llama-70B, so the gap servekit closed on SGLang is open on vLLM.
Next: cover quantized and MLA models (DeepSeek-V4, GLM-5) by replaying each rank's raw checkpoint bytes through vLLM's own loader, building on the preshard-roundtrip work.
