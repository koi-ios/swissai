# Status: servekit for vLLM (12 Nov 2026)

## Done
- **Research**, with file:line in vLLM v0.33.0 (4ba787ec):
  - `research/vllm-loading-paths.md`: every loading path, the plugin system, and `sharded_state` under TP × PP, quantization and MLA.
  - `research/related_work.md`: ServerlessLLM, fastsafetensors, Run:ai, ModelExpress/NIXL, InstantTensor, λScale, BlitzScale, Tensorizer, ipc_cache.
- **servekit change**: `patches/` (3 commits on cd8f0b8; applies cleanly to a fresh clone). It covers:
  - `launch`/`prepare` for `vllm serve`, with model-path rewrite and pinned served name;
  - manifest and version checks;
  - per-node sliced staging;
  - `--overlap` through a vLLM loader plugin (no fork);
  - a guard;
  - per-rank load records;
  - `VLLM_CACHE_ROOT` and friends, with write-back;
  - multi-node through the multiprocessing backend.

  Tests: 195 pass, 3 skip (Linux-only stager), including 59 new vLLM tests.
- **Prelim harness**: adds the `servekit_serial` and `servekit_overlap` arms, `run_servekit_prepare.sh` and `run_once.py --servekit-launch`. The mock now matches vLLM 0.33's sharded_state behaviour and logs on rank 0 only. Selftest passes 57/57.
- **Modal**: `modal_app.py` passes `--dry-run` (estimate $13.7 on A10G); `MODAL.md` is the runbook.
- **Write-ups**: `WRITEUP.md` (including the decision table on vLLM-native options) and `drafts/PR.md`.

## Not done, or blocked
- **No GPU run of anything.** The Modal runs need an account, so the Modal-Volume numbers in `WRITEUP.md` are still empty.
- **The real stager and `servekit launch` arms ran only through mocks on macOS.** On Linux they run in the Modal `selftest`, `smoke` and `experiment` functions.
- **The plugin was tested against a stand-in `vllm` package, not real vLLM.** The Modal smoke checks that vLLM logs "Registered model loader ... sharded_state".
- **Multi-node prepare and launch are designed and unit-tested only.** That covers the head running the sharder, followers running headless, and per-node frees. model-launch uses Ray for vLLM multi-node, which servekit refuses.
- **Quantized and MLA models are not presharded at all.** The preferred route is raw-byte presharding replayed through vLLM's default loader, reusing session 08's engine-agnostic `rawshard` core (servekit branch `preshard-roundtrip`); the alternative is a loader on vLLM's `ipc_cache` contract. See `WRITEUP.md`.

## Next 3 actions
1. On Modal account A, run `download`, `selftest`, `smoke` (about $0.75). Go on to `experiment_tp1` only if the smoke shows both servekit arms passing verify.
2. On account B, run `experiment_tp2`. Then summarize and fill `WRITEUP.md`'s results with measured medians and min–max.
3. With Xiaozhe: agree on Ray versus `--nnodes` in model-launch for vLLM. Add a vLLM hook to 08's `rawshard` core and try it on a small FP8 and a small MLA model, checked with `servekit deepstate`. Then send the servekit PR from `drafts/PR.md`.
