# Notes

Working notes for the preliminary result: what shapes the numbers, what surprised, what failed, and questions for Xiaozhe Yao. Fill the empty sections on the VM.

## Facts that shape the numbers

Fill from ENV.md and the logs.

- Network volume file system (`findmnt -T /net/models`): …
- `python evict.py --verify /net/models/apertus-8b`: cold … GB/s, cached … GB/s, second cold read … GB/s. Is the storage server caching?
- vLLM 0.33 reads safetensors with mmap by default, but **auto-prefetches into the page cache when it detects NFS or Lustre and the checkpoint fits in 90% of free RAM**. The log line "Filesystem type for checkpoints" says which, and `runs.jsonl` records it as `checkpoint_fs` / `prefetch_started`. So `default_net` may already prefetch, and `lazy_net` is plain mmap, the setting the servekit article's 453.7 s default compares to.

## Found while building the harness (11 Nov)

- servekit `main` already profiles vLLM live: `profile.py` has a vLLM spec, and its README shows `servekit profile -- vllm serve`. Open PR #26 adds `servekit profile --log` for saved vLLM logs, with patterns per vLLM version (0.14.1, 0.18.0, 0.24.0, 0.28.1rc1).
- `servekit verify` speaks the OpenAI API (`/v1/completions` with `echo` and `logprobs`), so it works against vLLM unchanged. Its default tolerance is 1e-6 per token, effectively bitwise. That is why `run_arms.sh` checks `default_net` against itself first. If that self-check fails, rerun with `VERIFY_TOKEN_TOL=1e-4 VERIFY_NLL_TOL=1e-5`, and write down why.
- vLLM 0.33 always points `TORCHINDUCTOR_CACHE_DIR` at `<VLLM_CACHE_ROOT>/…/inductor_cache`. It uses the caller's `TRITON_CACHE_DIR` only when that is set, and its own `triton_cache` otherwise. This matters for #270: for torch.compile artifacts, `VLLM_CACHE_ROOT` is the variable that counts.
- servekit's `jit_cache.py` also persists `TVM_FFI_CACHE_DIR`, which #270 does not list.
- `vllm bench startup` times the offline `LLM()` constructor. Its "cold" only swaps `VLLM_CACHE_ROOT` and never evicts the page cache, so it cross-checks `warm_weights`, not `cold`.
- vLLM logs CUDA-graph capture as `%.0f secs`: whole seconds only.
- servekit's `profile` keeps reading its pipe until every process holding it exits, so one straggler keeps it alive. `run_once.py` kills the process group 10 s after the server's port closes, and detects a crashed engine within about a second instead of waiting out the timeout.

## Found while porting servekit to vLLM (12 Nov)

- vLLM 0.33 logs "Model loading took", "Loading weights took" and "Graph capturing finished" with `info_once` (scope "local"): local rank 0 only (vllm/logger.py:141-151). So `weight_loading` at TP 2 is rank 0's, not the max; the mock now does the same. The servekit arms add every rank's own time (`vllm:sk_load_max_rank`).
- servekit's own fixture `tests/fixtures/llama70b-vllm-ngc.log` is the lab's one vLLM run on capstor (Clariden job 2918061, vLLM 0.24, Llama-3.1-70B TP 4, single run): "Filesystem type for checkpoints: LUSTRE ... Available RAM: 686.79 GiB", auto-prefetch started on all four ranks and "finished" in 4.5-11.4 s, yet "Loading weights took 231.17 seconds" (model loading 233.24 s, max over ranks). The prefetched pages apparently did not serve the mmap reads; check the Lustre client's `llite.*.max_cached_mb` on Alps. For comparison, Youssef's servekit overlap on SGLang loads the same model on Clariden in 6.0 s.
- `servekit_*` arms need servekit's own artifact (`run_servekit_prepare.sh`), whose manifest names `$MODEL_NET` as its source: the command keeps naming the source and servekit swaps in the artifact.

## Surprises

- …

## Failures

- …

## Three questions for Xiaozhe

Drafts; keep the three that the results make sharpest.

1. Where should vLLM support live: in `servekit launch` (rewrite `--model`, add `--load-format sharded_state`, stage per node), or as servekit choosing and configuring vLLM's own loaders per model where they already match RAM staging?
2. vLLM ≥ 0.33 auto-prefetches on Lustre when a checkpoint fits in RAM. Has the platform measured that on capstor, and which vLLM version does `vllm_base` pin?
3. For #270, should persistent compile caches be a servekit option (it already keeps Triton, TVM-FFI and FlashInfer caches for SGLang) or mounts and environment variables in model-launch's `vllm_base` and `sglang_base`?
