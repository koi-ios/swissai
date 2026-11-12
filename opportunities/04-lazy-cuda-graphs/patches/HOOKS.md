# Hook map: SGLang v0.5.20 (94602c9) + lazy CUDA-graph capture patch

Line numbers are in the **unpatched** tag (`git show v0.5.20:<file>`). Paths are relative to `python/sglang/srt/`.

| file:line (unpatched) | existing code | patch change |
|---|---|---|
| `model_executor/cuda_graph_config.py:76` | `ALLOWED_KEYS_PER_PHASE` | accept `initial_bs`, `deferred_bs`, `deferred_priority` for both phases |
| `model_executor/cuda_graph_config.py:109` | `PhaseConfig.full_prefill_prefix_chunk_tokens` | add the three optional fields (None = capture all at startup, today's behaviour) |
| `environ.py:1391` | `SGLANG_EAGER_INPUT_NO_COPY` (section "CUDA graphs and execution buffers") | add `SGLANG_LAZY_CUDA_GRAPH_{MIN_IDLE_S, IDLE_COST_RATIO, PROMOTE_AFTER, MIN_FREE_MB}` |
| `model_executor/runner/decode_cuda_graph_runner.py:213` | `class DecodeCudaGraphRunner` | class attribute `lazy_schedule = None` (EAGLE/MTP draft runners override `__init__` but reuse `capture()`) |
| `…/decode_cuda_graph_runner.py:309` | `self.capture_bs, self.compile_bs = get_batch_sizes_to_capture(...)` | build `lazy_schedule`; buffers stay sized for the planned `capture_bs` |
| `…/decode_cuda_graph_runner.py:680` | `is_bs_supported = ... cuda_graph_bs <= self.max_bs` | additionally require the planned bucket to be captured; else count an eager hit (promotion) |
| `…/decode_cuda_graph_runner.py:993-1057` | `capture()` | index reset moved into `_reset_buffers_for_capture()`; capture only `initial_bs` when lazy; mark them captured |
| `…/decode_cuda_graph_runner.py:1060` | `_capture_one_stream(stream_idx)` | takes `sizes` and `show_progress`; tqdm guard uses `isinstance` |
| new: `DecodeCudaGraphRunner.capture_deferred(bs)` | | reset index buffers → `graph_capture` → `capture_session` → one bucket → device sync → mark captured |
| `model_executor/runner/prefill_cuda_graph_runner.py:277` | `class PrefillCudaGraphRunner` | class attribute `lazy_schedule = None` |
| `…/prefill_cuda_graph_runner.py:569` | `# --- capture ---` in `__init__` | build `lazy_schedule`; `backend.keep_output_buffer(rows=max_num_tokens)` |
| `…/prefill_cuda_graph_runner.py:1209` | `if not self.can_replay_locally(...)` in `can_run_graph` | then require the planned bucket to be captured; else count an eager hit |
| `…/prefill_cuda_graph_runner.py:1405-1451` | `capture()` / `_capture_one_stream()` | split into `_capture_sizes(sizes)`; startup captures `initial_bs` |
| new: `PrefillCudaGraphRunner.capture_deferred(n)` | | **`buffers.reset_index_buffers()` first** (stale `out_cache_loc` points at KV slots the radix cache may own), then as decode |
| `model_executor/runner_backend/breakable_cuda_graph_backend.py:95-100` | `capture_session` resets `_shared_output_buffer` per session | keep it when `keep_output_buffer()` was called |
| `…/breakable_cuda_graph_backend.py:133` | `_alloc_full_buffer(warmup_out, size)` | allocate `max(size, rows)` rows once |
| `managers/scheduler.py:726` | `self.init_rank_consensus_checker()` in `__init__` | then `self.maybe_init_lazy_graph_capture()` |
| `managers/scheduler.py:2066-2079` | `ingest_requests`: leader polls `_poll_timeout_aborts()` | `begin_iteration()`; leader appends `propose(fully_idle=is_fully_idle())` to `local_reqs`; after `recv_requests`, `take_ticks()` |
| `managers/scheduler.py:4721-4799` | `on_idle`, fully idle path, before `maybe_sleep_on_idle()` | `run_pending(fully_idle=True)` |
| unchanged: `model_executor/model_runner.py:1794`, `:1840` | `_forward_raw` asks `can_run_graph` | (dispatch needs no change; frozen file untouched) |
| new: `model_executor/runner/lazy_capture.py` | | `LazyCaptureSchedule` (planned / captured / pending, bucket-exact `can_replay`, promotion), `pick_next_phase`; stdlib only |
| new: `model_executor/runner/lazy_capture_setup.py` | | eligibility checks, schedule from config, `has_capture_headroom()` |
| new: `managers/scheduler_components/lazy_graph_capture.py` | | `LazyGraphCaptureTick`, `LazyGraphCaptureController` (propose / accept / run_pending, MIN all-reduce agreement, disable on divergence), factory |

Facts the design relies on (unpatched code):
- **Request broadcast.** Each iteration, TP rank 0 broadcasts the polled requests plus its `local_reqs` to all TP ranks (`managers/scheduler_components/request_receiver.py:170-217`, `broadcast_pyobj` over `tp_cpu_group`). Iterations are therefore in lockstep, and the batch decision (and with it `on_idle`) is identical on every rank.
- **Custom all-reduce during warmups.**
  - Inside a capture session, the warmup forwards' custom all-reduce returns `torch.zeros_like(input)` (`distributed/device_communicators/custom_all_reduce.py:314-327`).
  - A real forward must therefore never run inside a capture session, which rules out capture on a background thread.
  - Buffers are registered at session end, collectively (`:182-194`).
- **Reserved KV slot.** KV slot 0 is reserved for dummy writes of padded tokens (`mem_cache/allocator/token.py:42-46`, `paged.py:337-341`).
- **Memory reserve.** The mem-fraction heuristic reserves memory for the full planned bucket lists (`arg_groups/memory_hook.py:350-380`, `reserve_for_graph_mb`), unless `SGLANG_ENABLE_POST_CAPTURE_KV_SIZING`, which the patch rejects for lazy mode.
