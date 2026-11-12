# vLLM 0.33.0: weight-loading paths, plugins, and `sharded_state`

Target: **vLLM v0.33.0**, tag commit `4ba787ec` (31 Oct 2026), the newest stable
release on 11 Nov 2026 (`v0.34.0rc0` exists but is not stable). Every reference
below is `path:line` in that tree (`vendor/vllm`, shallow clone). Everything here
comes from reading the code: nothing ran on a GPU.

## 1. Every loading path

`--load-format` picks a loader class from a dict (`vllm/model_executor/model_loader/__init__.py:50-66`);
`get_model_loader` looks it up (`:123-127`) when the GPU worker loads the model
(`vllm/v1/worker/gpu_model_runner.py:4992-4997`). `BaseModelLoader.load_model`
always does, in order, `create_model` (on the GPU), `load_weights`, then
`process_weights_after_loading` (`base_loader.py:62-115`).

| `--load-format` | loader | what it reads, per rank | how |
|---|---|---|---|
| `auto`/`safetensors`/`hf` (default) | `DefaultModelLoader` | **the whole checkpoint** (each rank narrows to its shard) | `safetensors_weights_iterator`: `safe_open` = mmap per file (`weight_utils.py:1243-1249`) |
| … `--safetensors-load-strategy eager` | same | whole checkpoint | `open(f).read()` per file, one file at a time (`weight_utils.py:1207-1212`) |
| … `--safetensors-load-strategy prefetch` (or **auto**, see §2) | same | whole checkpoint via mmap, after a background page-cache prefetch | §2 |
| … `--model-loader-extra-config '{"enable_multithread_load":true,"num_threads":N}'` | same | whole checkpoint | N threads, `safetensors.load_file` per file (`weight_utils.py:1252-1278`, `default_loader.py:328-339`) |
| `fastsafetensors` | `DefaultModelLoader` | files split over the **world** group, then NCCL broadcast | `ParallelLoader(pg=WORLD)` (`weight_utils.py:1319-1386`); GDS off when world > 1 (`:1343`) |
| `instanttensor` | `DefaultModelLoader` | collective over the **world** device group, direct I/O + pipelined prefetch | `instanttensor.safe_open(..., process_group=world)` (`weight_utils.py:1394-1485`, group `:1418`; docstring `config/load.py:38-40`) |
| `runai_streamer` | `RunaiModelStreamerLoader` | whole checkpoint, or files split over ranks with `{"distributed": true}` | Run:ai streamer, concurrent range reads, `concurrency`/`memory_limit` extra config (`runai_streamer_loader.py:131-168`, `:225-263`) |
| `sharded_state` | `ShardedStateLoader` | **only this rank's presharded files** | glob of `model-rank-{rank}-part-*.safetensors` or `model-pp-{pp_rank}-tp-{tp_rank}-part-*` (`sharded_state_loader.py:38-39, 196-209`), `safe_open` = mmap (`:262-266`) |
| `runai_streamer_sharded` | `ShardedStateLoader` | only this rank's presharded files | same glob, Run:ai iterator (`:257-258`) |
| `ipc_cache` | `IpcModelLoader` | nothing from disk: maps the GPU weights a `vllm preload` daemon holds, via CUDA IPC | `weight_cache/ipc_loader.py:70-97, 295-313`; `entrypoints/cli/preload.py:1-18` |
| `modelexpress` | `ModelExpressModelLoader` | from a ModelExpress server | `modelexpress_loader.py` |
| `tensorizer` | `TensorizerLoader` | a tensorizer-serialized file | `tensorizer_loader.py` |

Not a load format, but adjacent: `vllm snapshot create` (`entrypoints/cli/snapshot.py:19-40`)
requires TP = PP = DP = 1 and a remote model pinned to a 40-character revision, so it
does not apply to TP > 1 serving on Alps.

## 2. Auto-prefetch, the native option closest to RAM staging

`safetensors_weights_iterator` (`weight_utils.py:1115-1249`):

- File system of the **first** file, from `/proc/mounts` (`:990-1019`). Network = `nfs`, `nfs4`, `lustre` (`:1137`).
- Fits = checkpoint size ≤ 90% of available RAM, cgroup-aware (`:1138-1141`, `:969-987`).
- With no `--safetensors-load-strategy`: network FS + fits → prefetch (`:1150-1152`); network FS + does not fit → warning, plain mmap (`:1153-1162`).
- Prefetch (`_prefetch_all_checkpoints`, `:1041-1111`): **each torch.distributed rank takes `sorted_files[rank::world_size]`** (`:1051-1057`), reads each of its files sequentially in 16 MiB `read()`s (`:1023-1038`) on `--safetensors-prefetch-num-threads` threads (default 8, `config/load.py:12`), in a background thread that starts inside the weight iterator, i.e. after process start, distributed init and model construction.

Consequences:

1. Concurrency is at most one sequential reader per file. There is no within-file parallelism, which is what Lustre needs: one reader on a single-striped capstor file gives 0.36 GB/s, and 64 give 6.8 GB/s (`data/cold-start-elimination-experiments-swissai/experiments/parallel-reads-lustre/results/parallel-read-sweep-81985.out`, medians of 3).
2. It cannot overlap with engine startup; servekit's `--overlap` does.
3. **Multi-node:** files are split over global ranks, so each node's page cache gets only 1/nnodes of the files. Every rank of the default loader still reads every file, so the rest is mmapped cold. Not measured; this is read from the code.
4. A checkpoint larger than 90% of a node's RAM (GLM-4.7, 717 GB) gets no prefetch at all, so loading is plain mmap: the 800+ s regime of Youssef's tables.

## 3. Plugins: overlap without a vLLM fork

- `vllm.general_plugins` entry points are loaded in every process (`plugins/__init__.py:16-18, 77-90`): the CLI parser (`engine/arg_utils.py:3256`), `EngineArgs` (`:929-931`), the engine core (`v1/engine/core.py:132-135`) and **every worker, before its model loads** (`v1/worker/worker_base.py:287-289`). Workers are spawned fresh, so a plugin is the only way code reaches them without a fork, the same reason servekit's SGLang shim is a plugin.
- `register_model_loader(name)` (`model_executor/model_loader/__init__.py:69-120`) replaces or adds a loader class. It warns when a name is overwritten (`:97-103`). `load_format` is a free string, and "other custom values can be supported via plugins" (`config/load.py:30, 60`).
- **Keep the name `sharded_state`** rather than inventing one. `VllmConfig` unties word embeddings for every format except `dummy`, `sharded_state` and `runai_streamer_sharded` (`config/vllm.py:1612-1620`). A new name would untie a tied-embedding model whose sharded dump stores the tie under `lm_head` only, so the load would fail on a missing key.
- `VLLM_PLUGINS` is part of the torch.compile cache key: `compile_factors` hashes every known vLLM variable except an ignore list (`envs.py:2353-2448`), and `VLLM_PLUGINS` is not on it. servekit must therefore never set it. A `SERVEKIT_*` variable is not a vLLM variable, so it is never hashed.
- `--worker-extension-cls` with `init_worker_extension` runs "before model loading" (`config/parallel.py:309-316`). It is an alternative hook, but it blocks before model construction, not just before the read, and RL frameworks already use that flag.

## 4. `sharded_state` under TP × PP, quantization and MLA

- **Save:** `save_sharded_state` is a `collective_rpc` that every worker answers before it returns, remote nodes included (`v1/executor/abstract.py:328-338`, `v1/worker/gpu_worker.py:1490-1503`). So, unlike SGLang 0.5.x (servekit's `wait_for_writes`), no done-marker barrier is needed. It dumps `_filter_subtensors(model.state_dict())` **after** `process_weights_after_loading` (`sharded_state_loader.py:268-303`).
- **TP × PP:** the file name carries both ranks: `model-pp-{pp_rank}-tp-{tp_rank}-part-{part}` when PP > 1 (`:38-39, 60-67`). Patterns are validated so ranks cannot collide (`:103-135`). Global rank = `pp_rank * TP + tp_rank` (with DP and PCP outermost/innermost, `distributed/parallel_state.py:2158-2164`), so a node holds a contiguous run of stages: servekit's SGLang slicing carries over with `-tp-` in place of SGLang's `-rank-`.
- **DP / EP:** the names carry no DP or EP rank. Dense DP replicas read the same files, which is fine. With `--enable-expert-parallel`, each DP rank holds different experts under the same name, so they collide. servekit refuses EP.
- **Silent narrowing:** a dtype mismatch only logs a warning, then `copy_` casts (`:224-231`). A tensor smaller than its parameter is narrowed with a warning (`:232-244`). A checkpoint presharded for TP=4 and loaded at TP=2 globs rank 0's and rank 1's files and narrows them into the bigger TP=2 parameters: **a fluent server on wrong weights**. Only missing keys raise (`:251-252`). This is why servekit's manifest check matters even more for vLLM, and why servekit's loader turns those two warnings into errors.
- **Quantized models:** the dump is post-processing state, but on reload `BaseModelLoader.load_model` runs `process_weights_after_loading` **again** (`base_loader.py:88, 109`), and loads into fresh parameters in checkpoint layout. vLLM's own weight cache (`ipc_cache`) handles exactly this case correctly:
  - it builds the model on the meta device;
  - it refuses any quant method that does not declare `supports_pre_processed_weights` (`weight_cache/protocol.py:97-117`, `ipc_loader.py:300`);
  - it replaces parameters instead of copying into them;
  - it re-runs post-processing under `weights_already_processed()` (`ipc_loader.py:305-313`; contract in `model_executor/layers/quantization/base_config.py:36-41`, gate in `model_loader/utils.py:139-166`).

  `ShardedStateLoader` does none of this. servekit therefore refuses quantized checkpoints for vLLM presharding until a loader follows the IPC contract (§6).
- **MLA** (DeepSeek-V2/V3, Kimi, GLM-5 MLA): `process_weights_after_loading` registers **new** parameters `W_UV` and `W_UK_T` as transposed/permuted views (`model_executor/layers/attention/mla_attention.py:1347-1350`; `replace_parameter` registers one when none exists, `model_executor/utils.py:71-90`). The **save** should already fail: `_filter_subtensors` calls `get_end_ptr`, i.e. `tensor.view(-1)`, on every tensor before any contiguity check (`sharded_state_loader.py:152-158`), and `view(-1)` raises on a non-contiguous view. Were it saved, the reload would still fail at `state_dict[key]` (`:223`), since a freshly built model has no such keys. That is the vLLM twin of SGLang #35702. Read from the code by this session and independently by the preshard-roundtrip session (08); not run. servekit refuses configs with `kv_lora_rank`.

## 5. What the log does and does not say

- `"Model loading took %s GiB memory and %.6f seconds"` (`gpu_model_runner.py:5088-5092`), `"Loading weights took"` (`sharded_state_loader.py:247-250`) and `"Graph capturing finished in %.0f secs"` (`gpu_model_runner.py:6516-6520`) are all `info_once` with the default scope `"local"`. That scope prints on **local rank 0 only** (`logger.py:141-151`, `distributed/parallel_state.py:2721-2739`).
  - So `servekit profile`'s vLLM `weight_loading` is local rank 0's time, not the slowest rank's.
  - A headless worker node prints exactly one such line.
  - servekit's loader plugin writes every rank's own load time into a marker, so launch can report the max over ranks.
- A headless worker node (`vllm serve ... --headless --node-rank k>0`, `entrypoints/cli/serve.py:218-233`) prints no ready line.

## 6. Compile caches and the rewritten model path

- The cache dir is `VLLM_CACHE_ROOT/torch_compile_cache/<sha256(env, config, code, compiler)[:10]>/rank_{rank}_{dp}/<prefix>` (`compilation/backends.py:1086-1104`).
  - Inductor's cache is forced under it (`compilation/compiler_interface.py:29-34`).
  - Triton's cache follows a caller-set `TRITON_CACHE_DIR`, and otherwise also goes under it (`:35-44`).
- **The config hash includes the model path**: `ModelConfig.compute_hash` does not ignore `model` (`config/model.py:457-525`, `config/utils.py:376-399`).
  - servekit serves from `/dev/shm/servekit/<artifact-name>`, the same string on every launch, so its cache key is stable across launches.
  - But it differs from a plain run's key on the source path, and from a sharding run's key: `prepare` cannot warm the serve-time compile cache unless it loads from the staged path.
- `VLLM_CACHE_ROOT` itself is excluded from the key (`envs.py:2366`), so relocating it is safe. `LoadConfig.compute_hash` hashes nothing (`config/load.py:119-134`), so `--load-format` does not change the key.

## 7. Multi-node in vLLM 0.33 (MP backend)

`--nnodes/-n`, `--node-rank/-r`, `--master-addr`, `--master-port` (`engine/arg_utils.py:1175-1178`, `config/parallel.py:317-329`); worker nodes add `--headless`. `world_size % nnodes == 0` is checked (`engine/arg_utils.py:2399-2418`). Data-parallel multi-node (`--data-parallel-size-local`, `--data-parallel-address`) runs whole replicas per node instead, so every node needs every shard.
