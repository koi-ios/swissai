# feat: vllm fast weight loading (`servekit launch -- vllm serve ...`)

Closes the roadmap item "Support vllm fast weight loading". This is a draft. Nothing here has run on a GPU yet; the Modal runs are prepared (see below).

## What it does
- **The same command shape as SGLang:**

  ```
  servekit launch --servekit-artifact-path DIR [--overlap] -- vllm serve MODEL -tp N
  ```

  It accepts the positional model or `--model`, `-tp/-pp`, and `--nnodes/--node-rank/--master-addr`.
- **Presharding** uses vLLM's own `sharded_state`. A vLLM sharder (adapted from vLLM's example) forces eager mode and one replica, and skips stale `*.index.json`. Its manifest pins the vLLM minor version, because the shards are post-processed weights.
- **The served name is kept.** The command pins `--served-model-name` to the original path. Otherwise vLLM would rename the model to `/dev/shm/servekit/...` for every client and router.
- **`--overlap` works through a vLLM plugin, with no fork.** A `vllm.general_plugins` entry point does nothing unless servekit launched the engine. It replaces the class behind `sharded_state` with a subclass that:
  - waits on the existing stage barrier (`_shim.wait_for_stage`) right before the read;
  - turns ShardedStateLoader's narrowing and dtype-cast warnings into errors. A TP-mismatched checkpoint otherwise serves wrong weights.
  - records each rank's load time. vLLM 0.33 logs `Model loading took` on local rank 0 only.

  Shards are renamed into place only when complete. An engine running without the plugin therefore fails loudly instead of reading half-written files.
- **Multi-node** uses the multiprocessing backend: each node stages its ranks' `model-rank-*` or `model-pp-*-tp-*` files. A headless node frees its copy once all its local ranks have recorded their load. Ray-placed ranks fall back to vLLM's own loader.
- **`vllm_guard`** keeps on vLLM's own loader everything `sharded_state` cannot round-trip: quantized checkpoints, MLA, expert parallelism, LoRA and speculative decoding.
  - vLLM's `sharded_state` re-runs `process_weights_after_loading` on post-processed weights (base_loader.py:88, 109).
  - MLA creates `W_UV`/`W_UK_T` after loading (mla_attention.py:1347-1350).
- **Compile caches**: `VLLM_CACHE_ROOT` and TileLang are added in a separate module (`jit_cache.py` is untouched). They are written back after the first launch, because vLLM's cache key includes the served model path.

## Not changed
- The SGLang path behaves the same: its tests are untouched apart from the two vLLM placeholders.
- The stager is still byte-identical to the experiment's.
- `_shim` and `jit_cache` are unchanged.

## Tests
`tests/test_vllm.py` (59 tests) covers:
- a fake sharder (head and follower);
- a fake engine (rewrite, guard, Ray, overlap visibility, worker-node free);
- the plugin against a stand-in `vllm` package;
- cache write-back;
- one end-to-end `--overlap` run through a mock `vllm serve` subprocess behind the real `run_profile`, which checksums every shard it reads.

The suite reports 195 passed and 3 skipped. A separate commit makes the O_DIRECT fixture skip on platforms without O_DIRECT (macOS), instead of erroring.

## Overlap with PR #26
This touches the VLLM `FrameworkSpec`, but only by adding fields. #26 (`profile --log`) changes its patterns, so a rebase should be mechanical.

## Before merge
- Run the Modal smoke (`opportunities/03-servekit-vllm/MODAL.md`), and one Apertus-8B run on Clariden with `servekit verify` against the default loader.

🤖 Generated with [Claude Code](https://claude.com/claude-code)
