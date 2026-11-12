# Working notes, idea 1 (servekit for vLLM). Scratch; the deliverables are RELATED_WORK.md and DESIGN.md.

S = /private/tmp/claude-501/-Users-koiios--claude-spares-spare-qHWvhZ/85cbb828-b415-43aa-adc6-b088d62505ae/scratchpad
SK = $S/shared/repos/servekit (@cd8f0b8); V = $S/src/vllm (v0.33.0, 4ba787e)

## servekit facts (verified, file:line)
- launch.py: overlap = one barrier per node: SGLang plugin blocks ModelRunner.load_model until marker says ok
  (_shim/__init__.py:175 `_around_run_load` -> wait_for_stage; marker .<name>.stage published by launch.py:274,302).
  Not per-file pipelining. Metadata copied up front (launch.py:241-249,296).
- stage: vendored bash stager, dd O_DIRECT 16M blocks, 64 slices per file, ALL files at once (xargs -P NTASK),
  truncate first, numactl --interleave over CPU nodes, size-parity gate only (stage_to_shm_sliced.sh).
  Measured 11.9 GB/s Bristen, 17.0 GB/s Clariden (stage.py:3-6). BASH_ENV in NGC images -> 0.41 GB/s (stage.py:41-45).
- topology.py: shard_glob per node (model-rank-[lo-hi]-part-*.safetensors; pp variant), <10 index limit (_index_glob).
- engine_args/manifest: manifest fields format, engine, engine_version, tp, pp, dp, dtype, quantization, source.
  check_manifest: source path, engine, tp/pp/dp, dtype, quantization -> problems -> fallback to plain command.
- VLLM FrameworkSpec in profile.py:108-147 has NO model_flags, parallel_flags, dist_flags -> launch refuses vLLM
  (tests/test_launch.py:72 "Phase 4's job").
- jit_cache: TRITON_CACHE_DIR, TVM_FFI_CACHE_DIR, FLASHINFER_WORKSPACE_BASE under /tmp/servekit/caches/<name>.
- quant_guard: SGLang rules (mxfp4 triton, NVFP4 DeepSeek, GGUF, megamoe, act-order GPTQ, flashinfer_mxfp4).
- prepare: runs _prepare/save_sharded_state.py (SGLang Engine + collective_rpc save_sharded_model) + manifest + jit caches.
- entry point: [project.entry-points."sglang.srt.plugins"] servekit_pp = servekit._shim:install

## Lab's own vLLM data point (!!)
servekit tests/fixtures/llama70b-vllm-ngc.log: job 2918061, Clariden nid006183, cpus=288, 28 Jul 2026,
vLLM 0.24.0+092c4842.dev (NGC), Llama-3.1-70B-Instruct TP4 from /capstor/store/.../hf_models, load_format=auto.
- "Filesystem type for checkpoints: LUSTRE. Checkpoint size: 131.42 GiB. Available RAM: 686.79 GiB." -> auto-prefetch ON
- prefetch per rank 8 threads, 16 MiB; ranks got 8/8/7/7 of 30 files (= global rank::world split);
  "finished in 4.50s / 4.64 / 4.71 / 11.43 s"  (?? too fast; maybe partially cached or Lustre readahead; check)
- tqdm 30 files over 3:51 at ~7.7 s/it; "Loading weights took 231.17 seconds"; "Model loading took 32.89 GiB and 233.24 s" (max)
- torch.compile 35.70 s; graph capture 11 s; init engine 56.09 s (compilation 37.24 s); startup 16:58:06 -> ready ~17:05:08 (~422 s)
- => single run: weights = 233 s of ~422 s (55%) WITH auto-prefetch on. SGLang same model/cluster: default mmap 794.2 s,
  nommap 27.8 s (4 thr), fastsafetensors 48.6 s, servekit overlap 6.0 s (Youssef README tables, Clariden).

## Youssef's numbers (coldstart-exp README)
- Lustre: 64 O_DIRECT dd readers on one single-striped 4.6 GB file: 6.7 GB/s, 18x one reader, not flat yet.
- Bristen Llama-70B SGLang 0.5.10: default 453.7 s, nommap 45.7, fst 59.1, shm+mmap 20.1+7.7, shm+preshard 8.8+7.4, overlap 9.7.
- 0.5.16 sweep weight_loading (s): Bristen default/nommap/fst/sk-serial/sk-overlap:
  Apertus-8B 78.0/9.5/17.9/3.3+2.0/2.1; Llama-70B 737.0/46.0/61.3/9.3+14.2/14.3; GLM-4.7 827.9/294.9(thr4)/143.8/10.6+16.4/16.2
  Clariden: Apertus 92.8/4.5/11.9/1.1+0.9/0.9; Llama 794.2/27.8(thr4)/48.6/5.1+6.0/6.0; GLM 861.6/263.7(thr2)/113.8/40.8+6.5/6.7
- CPU sweep (Bristen Llama70B): nommap 192.2 s @16 CPUs -> 48.9 @128; overlap stage 29.2 @16 -> 15.2 @128.
- fastsafetensors: no multi-node; GLM result patched; NCCL over Slingshot scales badly.

## vLLM 0.33 code facts (V = shared clone; V2 = my clone at scratch/serve-kit-vllm-prelim/src/vllm, same tag)
- prefetch: weight_utils.py:1040-1112 `_prefetch_all_checkpoints`: files split by GLOBAL torch.distributed rank:
  `sorted_files[rank::world_size]` (1051-1057); 8 threads (DEFAULT_SAFETENSORS_PREFETCH_NUM_THREADS, config/load.py:12),
  buffered f.read(16 MiB) whole files (1023-1037); background daemon thread; loader does NOT wait for it.
- auto rule: weight_utils.py:1136-1198: fs in (nfs,nfs4,lustre) and total ckpt bytes <= 0.9*available RAM (psutil+cgroup).
  Whole checkpoint size, not per-node need. Else warning "Skipping auto-prefetch" -> plain mmap.
- iterator: every rank iterates ALL files (sorted), safe_open (mmap) + get_tensor (lazy view) (1200-1249). eager: f.read() whole file per rank (1207-1212).
- multithread (extra_config enable_multithread_load): load_file per file, 8 threads, rejects non-lazy strategies (default_loader.py:135-145,329-336).
- fastsafetensors: ParallelLoader over torch.distributed.group.WORLD, nogds when world>1 (weight_utils.py:1319-1385).
- instanttensor: world device_group collective loading unless EP filter/cpu engram (1394-1476).
- runai_streamer: per-rank full list unless extra_config distributed=true (runai_streamer_loader.py:131-146,225-263); device cpu unless distributed.
- runai_streamer_sharded = ShardedStateLoader + runai iterator with is_distributed=False -> CPU buffer + tensor.clone() + copy_ (sharded_state_loader.py:257-258; weight_utils.py:1282-1316). NOT straight to GPU.
- ShardedStateLoader: patterns model-rank-{rank}-part-{part} / model-pp-{pp_rank}-tp-{tp_rank}-part-{part} (38-39) (servekit SGLang PP uses -rank-, differs);
  glob order (unsorted) (204); safe_open mmap + copy_ (262-266, 244); dtype mismatch only warns (224-231); missing keys -> ValueError.
- BaseModelLoader.load_model ALWAYS runs process_weights_after_loading (base_loader.py:109) -> for sharded_state the saved
  (already post-processed, gpu_worker.py:1490-1502 saves model_runner.model) weights get processed AGAIN. No weights_already_processed guard.
  ipc_cache path does it right: meta-device model + _apply_entries + process_weights_after_loading under weights_already_processed()
  (ipc_loader.py ~280-320; model_executor/utils.py:13-32; model_loader/utils.py:139-170 supports_pre_processed_weights).
- register_model_loader(load_format) exists (model_loader/__init__.py:69-120); LoadConfig doc: "Other custom values can be supported via plugins".
- vllm.general_plugins entry points loaded in ALL processes incl. workers (plugins/__init__.py:17-19, 77-90; worker_base.py:287-289 inside init_worker before worker class constructed).
  VLLM_PLUGINS allowlist if set.
- preload/ipc_cache: daemon loads with configured loader (daemon.py:357-382) -> could load sharded_state from /dev/shm; PP rejected; fingerprint hashes safetensors metadata (docs/features/preload.md).
- VLLM_ENABLE_STARTUP_PLAN=1 persists KV memory profiling under VLLM_CACHE_ROOT/startup_plan (v1/worker/startup_plan.py) -> idea 2.
- multi-node: MP backend with --nnodes/--node-rank/--master-addr, followers --headless (config/parallel.py:317-326,1064,1109; docs/serving/parallelism_scaling.md:139-159).
  MP global rank = node_rank*local_world + local_rank (multiproc_executor.py:192-208). Ray: driver-first bundle sort (ray_utils.py:141-; ray_executor_v2.py:309-319).

## model-launch facts (@02b74ce)
- vLLM multi-node = Ray: head `ray start --head` + poll loop `python3 -c 'import ray; ray.init(...)'` every 5 s until all GPUs, then
  `vllm serve --distributed-executor-backend ray`; followers `ray start --address=head:6379 --block` (launchers/framework.py:240-320).
  SGLang multi-node = same command per node with --dist-init-addr/--nnodes/--node-rank (framework.py:180-237).
  => servekit cannot wrap a vLLM follower's engine command (there is none); rank->node set by vLLM/Ray.
- vLLM flagships: DeepSeek-V4.1-Flash TP8, 2 nodes (vllm_base); GLM-5.3 TP16, 4 nodes; GLM-5.3-Flash TP8, 2 nodes;
  MiMo-V2.6-Pro-RL TP4 PP4, 4 nodes; DSv4.1-Flash-1017 TP8 2 nodes on vllm_0.30.0.toml (not in envs dir).
- vllm_base built in private model-launch-images; public images/: vllm_cuda13 (nightly `-U vllm`), vllm_0.30.0, vllm_cuda13_v2 (0.20.2), nightly-7f1a539.
  => vllm_base's vLLM version NOT verifiable from public repo. EDF sets no VLLM_PLUGINS, no cache vars.
