# Related work: fast LLM weight loading

Compiled 11 Nov 2026 for the servekit → vLLM v0.33.0 port. Sources are paper PDFs, official READMEs and docs, and the vLLM v0.33.0 tree in `vendor/vllm`.

- **[inferred]** means my reading of code or docs, not a claim the source states.
- **[unverified]** means I could not confirm it from a primary source.

**servekit, for reference.** It works in four steps:
1. Preshard once per TP layout with vLLM's `sharded_state` saver.
2. Stage each node's ranks' files from Lustre into `/dev/shm`, using 64 parallel `dd O_DIRECT` slices per file.
3. Start the engine during staging; each worker waits at a barrier just before its weight read.
4. Check logprobs against the default loader.

---

## 1. ServerlessLLM (OSDI '24): loading-optimized checkpoints and multi-tier loading

- **Problem.** Cold-start latency in serverless LLM inference, dominated by checkpoint download and load.
- **Mechanism.**
  - **Format.** A one-time conversion produces a "loading-optimized" checkpoint:
    - per-GPU partition files that hold only raw tensor bytes;
    - a tensor index mapping each name to (GPU id, offset, size), with tensors word-aligned.
  - **Loading.** A separate model-manager process reads the partitions:
    - with `O_DIRECT` and several I/O threads per tier;
    - into a fixed-size pinned-memory chunk pool;
    - pipelining remote store → local SSD → DRAM → GPU.
  - **Overlap with engine init.** Meanwhile the inference process builds the model. It then points each tensor at base + offset in GPU memory, using CUDA IPC handles from the manager.
  - **Cluster level.** Locality-aware scheduling and live migration.
- **Headline numbers.**
  - **Paper.** LLaMA-2-70B FP16 loads in **10.3 s**, vs 48.0 s with Safetensors and 84.0 s with PyTorch (4.7× / 8.2×).
    - Setup: one server with 8× A5000 and 1 TB DDR4; RAID0 of two PCIe 4.0 NVMe SSDs (12 GB/s); page cache dropped.
    - The seconds are read off Fig. 6a; the ratios are stated in the text.
  - **Paper, end to end.** 10–200× lower latency than serverless baselines (abstract).
  - **Current README.** Qwen3-32B on H100 + NVMe: 20.6 s → 3.2 s ("random" scenario).
- **Conversion.** Yes, its own format:
  - `sllm-store save` for Transformers;
  - `examples/sllm_store/save_vllm_model.py --tensor-parallel-size N` for vLLM.
- **Reads per rank.** Its own partition only.
- **Multi-node / TP.**
  - Multi-GPU within a server, through the partition plan.
  - A cluster of servers for scheduling and migration.
  - Cross-node TP for one instance is not described **[unverified]**.
- **vLLM today.** Not upstream. `sllm-store` ships a vLLM patch (`sllm_load.patch`, tested with vLLM 0.9.0.1) that adds `load_format="serverless_llm"`. Its docs describe it as compatible with `sharded_state`.
- **Citation.** Yao Fu, Leyang Xue, Yeqi Huang, Andrei-Octavian Brabete, Dmitrii Ustiugov, Yuvraj Patel, Luo Mai. "ServerlessLLM: Low-Latency Serverless Inference for Large Language Models." OSDI '24, pp. 135–153, 2024.
  - Paper: https://www.usenix.org/conference/osdi24/presentation/fu
  - Code: https://github.com/ServerlessLLM/ServerlessLLM
- **vs servekit.** The closest prior art, sharing per-GPU partitioned files, parallel `O_DIRECT` reads into host RAM, and loading overlapped with engine init; servekit instead reuses vLLM's own `sharded_state` files (no custom format or patched vLLM), stages into node-local `/dev/shm` for multi-node TP jobs, and reads from shared HDD-backed Lustre rather than local NVMe.

## 2. fastsafetensors (IBM)

- **Problem.** Deserializing each tensor into host memory leaves storage bandwidth unused for large safetensors models.
- **Mechanism.**
  - **Bulk copy.** Whole files (groups of tensors) are copied into GPU memory, in one of two ways:
    - with GPUDirect Storage (cuFile) when available;
    - otherwise with `pread` through double-buffered host bounce buffers.
  - **On the GPU.** Tensors are built in place with DLPack, and sharding and dtype conversion run on the GPU.
  - **Multi-GPU.** Files are assigned round-robin to ranks (`add_filenames`). Tensors are then broadcast or scattered to the ranks that need them over torch.distributed (NCCL).
- **GPUDirect Storage.** Optional.
  - In vLLM v0.33.0, GDS is tried only when world size is 1.
  - With TP > 1, vLLM forces `nogds=True` (`weight_utils.py`) to avoid creating CUDA contexts on every GPU.
- **Headline numbers.**
  - **Standalone.** 4.8–7.5× faster than safetensors 0.4.3 for Llama-7B/13B/70B, Falcon-40B and Bloom-176B.
    - Setup: one server with 2 NUMA nodes, each with 4× A100-80GB and 4× Samsung 3.2 TB NVMe; the four SSDs top out at 28 GB/s.
  - **Throughput.** Llama-70B on 4 GPUs reached **26.4 GB/s** with GDS (25.8 GB/s without), vs a ~5 GB/s baseline.
  - **In vLLM.** Llama-2-13B on 4× L40S: startup 12.39 s → 4.74 s.
- **Conversion.** None; it reads standard safetensors.
- **Reads per rank.** About 1/N of the files from storage. In vLLM's `ParallelLoader` path every tensor is then broadcast, so each GPU receives the full checkpoint.
- **Multi-node / TP.** TP is supported. vLLM passes its WORLD group, so loading can span nodes, but the paper evaluates only a single node.
- **vLLM today.** `--load-format fastsafetensors`.
- **Citation.** Takeshi Yoshimura, Tatsuhiro Chiba, Manish Sethi, Daniel Waddington, Swaminathan Sundararaman. "Speeding up Model Loading with fastsafetensors." arXiv:2505.23072, 2025.
  - Venue: the arXiv comments field says IEEE CLOUD 2025; I found no IEEE DOI **[unverified]**.
  - Paper: https://arxiv.org/abs/2505.23072
  - Code: https://github.com/foundation-model-stack/fastsafetensors
- **vs servekit.** Both split the read work across ranks to approach device bandwidth, but fastsafetensors keeps unmodified HF files and pays with an all-ranks NCCL broadcast of the whole model, whereas servekit's per-rank files make the reads disjoint with no exchange (and fastsafetensors was evaluated only on local NVMe and tmpfs).

## 3. Run:ai Model Streamer

- **Problem.** Slow cold starts, especially from network and object storage.
- **Mechanism.**
  - **Reading.** A C++ engine runs a pool of OS threads (`RUNAI_STREAMER_CONCURRENCY`) that read tensor bytes concurrently into a bounded, reusable CPU buffer (`RUNAI_STREAMER_MEMORY_LIMIT`).
  - **Streaming.** Python yields each tensor as it arrives, so the host-to-GPU copy overlaps later reads.
  - **Storage.** Local and network file systems, S3 (and S3-compatible), GCS, and Azure Blob.
- **Distributed streaming.** Each process reads a disjoint subset of the files and broadcasts it over torch.distributed.
  - Scope: per node by default, or cluster-wide with `RUNAI_STREAMER_DIST_GLOBAL=1`.
  - In the library it is on by default for object storage to CUDA/XPU.
  - In vLLM it is opt-in: `--model-loader-extra-config '{"distributed": true}'`.
- **Sharded variant (`runai_streamer_sharded`).** This is vLLM's `ShardedStateLoader` with the streamer as the file reader.
  - It expects `model-rank-{rank}-part-{part}.safetensors`, the `sharded_state` layout written by `examples/features/sharded_state/save_sharded_state_offline.py`.
  - Each worker reads only its own files.
  - The `concurrency` and `memory_limit` options apply.
- **Headline numbers.** From the NVIDIA blog and the repo's `benchmarks.md`.
  - Setup: Llama-3-8B (15 GB, one safetensors file); AWS g5.12xlarge using 1× A10G; cold start; streamer 0.6.0.
  - **S3:** **4.88 s** at concurrency 32 (Tensorizer's best: 37.36 s).
  - **io2 SSD:** 7.53 s at concurrency 8, vs 47 s for safetensors.
  - **gp3 SSD:** 14.34 s, vs 47.99 s for safetensors.
  - **vLLM 0.5.5 time to ready:** 23.18 s from S3, vs 66.13 s with safetensors on gp3.
- **Conversion.** None for `runai_streamer`. `runai_streamer_sharded` needs a `sharded_state` pre-shard, the same one servekit makes.
- **Reads per rank.**
  - Default: the whole checkpoint.
  - `distributed`: 1/N of the files, then a broadcast.
  - `_sharded`: its own shard only.
- **Multi-node / TP.** Yes.
- **vLLM today.** `--load-format runai_streamer` or `runai_streamer_sharded`, plus `--model-loader-extra-config '{"concurrency":..,"memory_limit":..,"distributed":..}'`.
- **Citation.** Software, no paper or white paper found.
  - Repo: https://github.com/run-ai/runai-model-streamer. Per its README, it was renamed "Model Streamer" and moved to https://github.com/dsx-ai-factory/model-streamer on 13 Sep 2026. The PyPI package `runai-model-streamer` keeps its name; 0.17.3 shipped on 11 Nov 2026.
  - Numbers: O. Dayan, N. Neria, E. Karabulut. "Reducing Cold Start Latency for LLM Inference with NVIDIA Run:ai Model Streamer." NVIDIA Technical Blog, 16 Sep 2025. https://developer.nvidia.com/blog/reducing-cold-start-latency-for-llm-inference-with-nvidia-runai-model-streamer
- **vs servekit.** Its concurrency knob is the same idea as servekit's 64 slices per file, and `runai_streamer_sharded` reads exactly servekit's pre-sharded files (making it the natural in-tree baseline), but it reads inside each worker's load step through a CPU buffer, with nothing staged before engine start and no overlap with engine init.

## 4. NVIDIA ModelExpress + NIXL

- **Problem.** Replica scale-out and restart time: avoid re-reading storage when a compatible replica is already serving.
- **Mechanism.**
  - **Metadata.** A Rust ModelExpress server brokers metadata only, backed by Redis, Kubernetes CRDs, or a k8s-service backend.
  - **Peer path.** A vLLM worker loading through `MxModelLoader` looks for a READY source with the same identity (revision, dtype, quantization, …) and the same global `worker_rank`. It pulls that source's already post-processed tensors with NIXL RDMA reads (UCX backend; NVLink when the two are co-located).
  - **Fallback chain.** Server cache → InstantTensor → ModelStreamer (`MX_MODEL_URI`: S3/GCS/Azure/local) → GDS → vLLM default loader.
  - **After loading.** The worker publishes itself as a source. It can optionally transfer JIT and kernel caches too.
- **NIXL.** NVIDIA's point-to-point transfer library. It offers one API over GPU/CPU memory and file, block and object storage, through backend plugins (UCX, LIBFABRIC, GDS, POSIX, OBJ, Mooncake, …).
- **Headline numbers.** From `docs/BENCHMARKS.md`.
  - Setup: DeepSeek-V4-Pro; vLLM 0.23.0; TP=8; 8× B200 with ConnectX-7.
  - **Weight load:**

    | Path | Time |
    |---|---|
    | P2P over NIXL/RDMA | **11 s** |
    | Local storage, cold page cache | 1 m 10 s |
    | ModelStreamer from S3 | 3 m 16 s |
    | Pull from Hugging Face | 8 m 53 s |

  - **API-ready:** 1 m 44 s with P2P weights and kernel artifacts, vs 8 m 1 s for a cold start from VAST.
- **Conversion.** None.
- **Reads per rank.**
  - P2P transfer is rank-matched, so each target pulls its own post-processed tensors **[inferred from ARCHITECTURE.md]**.
  - The GDS fallback reads full tensors on every TP rank (per the README).
- **Multi-node / TP.** TP, DP and PP are validated, with a two-node TP+PP example. EP is not validated end to end.
- **vLLM today.**
  - `--load-format modelexpress` is native in v0.33.0, as a 70-line wrapper that imports `modelexpress.engines.vllm.loader.MxModelLoader` (needs `pip install modelexpress`).
  - The ModelExpress README says native support starts at vLLM 0.23.0.
  - The ModelExpress plugin also registers `mx`; older images used `mx-source` / `mx-target`.
  - Separately, `--weight-transfer-config '{"backend":"modelexpress"}'` is an RL weight-refit backend.
- **Citation.** Software, no paper.
  - ModelExpress: https://github.com/ai-dynamo/modelexpress (PyPI `modelexpress` 0.7.0, 3 Oct 2026).
  - NIXL: https://github.com/ai-dynamo/nixl (PyPI `nixl` 1.5.2, 29 Oct 2026).
- **vs servekit.** Complementary: ModelExpress takes storage out of the path only when a warm replica exists, while the first replica (servekit's case) still falls back to a storage loader, a cold-start slot servekit could fill on Alps.

## 5. InstantTensor (`--load-format instanttensor`)

- **Who.** ScitiX AI (GitHub org `scitix`), with the Wenfei Wu Lab at Peking University.
  - PyPI `instanttensor`: 0.1.0 on 2 Mar 2026, 0.2.3 on 8 Nov 2026.
  - No paper.
- **Problem.** Loading safetensors into GPU memory underuses I/O bandwidth.
- **Mechanism.**
  - **Read planning.** One `safe_open` call over all files lets it plan reads across files.
  - **I/O backends.** Direct I/O by default (io_uring, else AIO), mmap on tmpfs, and optional GDS (cuFile / FGDS).
  - **Pipelining.** It overlaps I/O, host-to-device copies and prefetch through reusable staging buffers.
  - **Multi-GPU.** With an NCCL process group, ranks share the read work and every rank yields the full selected tensor set. The split and the exchange happen in native code **[inferred from the Python wrapper and README]**.
- **Headline numbers.** From `docs/benchmark.md`, instanttensor 0.1.4, on "50 GB/s storage". The storage type, file system and page-cache state are not stated.

  | Model / GPUs | InstantTensor | safetensors | fastsafetensors | Run:ai |
  |---|---|---|---|---|
  | Qwen3-30B-A3B, 1× H200 | **1.77 s** (34.5 GB/s) | 57.4 s | 22.2 s | 9.29 s |
  | DeepSeek-R1, 8× H200, TP8/EP8 | **15.3 s** (45 GB/s) | 160 s | 75.8 s | 101 s; 192 s distributed |

  On tmpfs, DeepSeek-R1 loads in 12.1 s vs 22.6 s.
- **Conversion.** None.
- **Reads per rank.**
  - Storage reads are split across ranks, but every GPU receives the whole checkpoint, and vLLM slices out each rank's TP shard.
  - When filtering experts (EP), vLLM drops the collective mode, and each rank then reads only its own experts.
- **Multi-node / TP.** vLLM passes its world group, so loading can span nodes. All benchmarks are single-node.
- **vLLM today.** `--load-format instanttensor` (`pip install vllm[instanttensor]`). It is also the first disk strategy in ModelExpress's chain.
- **Citation.** https://github.com/scitix/InstantTensor
- **vs servekit.** Both rely on direct I/O with many outstanding requests, but InstantTensor reads unmodified HF files and delivers every tensor to every rank, whereas servekit's pre-sharding removes that traffic and its `/dev/shm` staging runs the slow Lustre reads before and alongside engine start; InstantTensor is the strongest unmodified-checkpoint baseline to run on capstor.

## 6. λScale (arXiv 2502.09922)

- **Problem.** Scale out serverless LLM instances quickly under bursty load, without keeping many warm replicas.
- **Mechanism.**
  - **Multicast.** The model is split into blocks and multicast over RDMA with GPUDirect RDMA, on a binomial-pipeline schedule built on Derecho's RDMC. It can optionally send k-way from k sources.
  - **Execute-while-load.** λPipe chains receiving nodes that hold different blocks into an inference pipeline, so serving starts before any node has the full model.
  - **Mode switch.** Once a node has every block, it switches to local execution, recomputing KV caches from the tokens generated so far.
  - **Where models live.** GPU memory, host memory, or remote nodes; SSD appears only as a baseline.
- **Headline numbers.**
  - Setup: up to 12 nodes × 1 H800, or 6 nodes × 4 H800, each node with one 400 Gb/s InfiniBand NIC; Llama-2 7B/13B/70B.
  - Llama-2-13B scaled to 8 nodes in **under 1 s**, up to 1.5× faster than NCCL.
  - Multicast is up to 1.82× faster than FaaSNet and 1.53× faster than NCCL.
  - On traces: 2.4–5× better P90 latency and 17.8–31.3% lower GPU cost than ServerlessLLM, FaaSNet and NCCL.
- **Conversion.** None described; tensors are packed into contiguous blocks at runtime.
- **Reads per rank.** Not applicable: blocks arrive by network multicast, and a node may run with only some of them.
- **Multi-node / TP.** Yes; the 70B model uses 4 GPUs per node.
- **vLLM today.** None.
  - It is its own system: 10K lines of Python plus 4K of C++, extending Meta's Llama reference code.
  - The authors say code will be "released upon acceptance".
- **Citation.** Minchen Yu, Rui Yang, Chaobo Jia, Zhaoyuan Su, Sheng Yao, Tingfeng Lan, Yuchen Yang, Zirui Wang, Yue Cheng, Wei Wang, Ao Wang, Ruichuan Chen. "λScale: Enabling Fast Scaling for Serverless Large Language Model Inference." arXiv:2502.09922 (v1 Feb 2025; v3 5 Mar 2026). https://arxiv.org/abs/2502.09922
  - Venue: none listed, and none found.
- **vs servekit.** λScale starts from a warm copy in another node's GPU or host memory rather than cold storage, and its execute-while-load goes further than servekit's overlap, which hides I/O behind engine init but serves only after all weights are in.

## 7. BlitzScale (OSDI '25)

- **Problem.** Autoscaling for model-as-a-service is stuck trading scaling speed against host-memory caching of parameters.
- **Mechanism.**
  - **Network loading.** New instances pull parameters from GPUs already serving the model, over the compute network (RDMA plus NVLink).
  - **Multicast chains.** The chains are planned on a leaf/spine network model, so only O(1) host-cached copies are needed cluster-wide.
  - **Within an NVLink group.** Each target GPU receives 1/g of the parameters, and the group then AllGathers over NVLink.
  - **Live scaling.** Scaling works at layer granularity (ZigZag scheduling), so overloaded instances can offload layers to new instances before those finish loading.
- **Headline numbers.**
  - Up to 94% lower tail latency than ServerlessLLM.
  - 49% less GPU time than DistServe and vLLM at the same SLA.
  - Setup: Llama3-8B, Mistral-24B and Qwen2.5-72B on BurstGPT and Azure traces.
    - Cluster A: 4×8 A800-80GB, NVLink, 100 Gbps RDMA.
    - Cluster B: 2×8 A100-80GB PCIe, 100 Gbps RDMA.
  - Example: six 24B prefill instances scaled in 1,200 ms, vs about 2,000 ms for an all-host-cache ServerlessLLM variant.
- **Conversion.** None described.
- **Reads per rank.** Not applicable: parameters come from peer GPU memory, with no storage on the hot path.
- **Multi-node / TP.** Yes; the 72B model uses TP on cluster A.
- **vLLM today.** None. It is its own 24K-line Rust/C++ system on FlashInfer kernels, and I found no public code.
- **Citation.** Dingyan Zhang, Haotian Wang, Yang Liu, Xingda Wei, Yizhou Shan, Rong Chen, Haibo Chen. "BlitzScale: Fast and Live Large Model Autoscaling with O(1) Host Caching." OSDI '25, pp. 275–293, 2025. arXiv:2412.17246.
  - Paper: https://www.usenix.org/conference/osdi25/presentation/zhang-dingyan
- **vs servekit.** BlitzScale makes the same "network beats storage" bet as ModelExpress and λScale, so it needs a live replica and is orthogonal to servekit's cold path; its layer-level live execution is a finer-grained version of servekit's overlap.

## 8. Tensorizer (CoreWeave)

- **Mechanism.**
  - **Format.** A model's tensors are serialized into a single file in Tensorizer's own format.
  - **Loading.** The deserializer streams from local disk, HTTP(S), Redis or S3 into (GPU) memory, using concurrent range readers (`num_readers`) and reusing buffers.
  - **Encryption.** Optional, via libsodium.
- **Headline numbers.**
  - README: GPT-J (20 GB) at **~5 GB/s** ("wire speed") on 40 GbE.
  - Run:ai's benchmark, Llama-3-8B on A10G, best reader count: 16.11 s on gp3, 10.36 s on io2, 37.36 s on S3.
- **Conversion.** Yes. Serialize with `examples/features/tensorize_vllm_model.py`. For TP > 1 there is one file per rank (`model-rank-%03d.tensors`), and vLLM fills in the TP rank.
- **Reads per rank.** Its own shard.
- **Multi-node / TP.** Per-rank files work with any TP size. Multi-node use is not specifically documented.
- **vLLM today.** `--load-format tensorizer` (`pip install vllm[tensorizer]`).
- **Citation.** CoreWeave, tensorizer, https://github.com/coreweave/tensorizer (PyPI 2.12.1, 21 Apr 2026). No paper.
- **vs servekit.** Closest on per-TP-rank files read by many concurrent readers, but Tensorizer needs its own `.tensors` format and targets HTTP/S3 streaming, whereas servekit keeps vLLM's `sharded_state` safetensors.

## 9. vLLM `ipc_cache` / `vllm preload`

- **What it is.**
  - **Daemon.** One weight-cache daemon per GPU:
    - loads its rank's shard with any disk `--load-format`;
    - runs quantization post-processing;
    - exports every parameter as a CUDA IPC handle over a Unix socket.
  - **Engine.** An engine started with `--load-format ipc_cache` builds the model on the meta device and maps the daemon's tensors. Mapping is `zero_copy` by default; `copy` clones the tensors and then the daemon frees its copy.
  - **Safety check.** Before mapping, the engine compares a fingerprint:
    - checkpoint-metadata hash and architecture;
    - TP/DP rank, dtype, and quantization;
    - vLLM version and GPU UUID.

    On any mismatch it falls back to disk.
  - **Extras.** `vllm preload` also pre-runs FlashInfer autotune, and can seed autotune caches from another replica.
- **When it landed.**
  - Daemon and `ipc_cache`: vLLM PR #54921 "Fast Start", merged 4 Sep 2026, released in **v0.30.0** (22 Sep 2026). Multi-node TP came in the same release (#55465 / #55468).
  - `vllm preload` launcher CLI: PR #56680, merged 25 Sep 2026, released in **v0.31.0** (5 Oct 2026), together with DP support (#57386).
- **Headline numbers.**
  - None measured in the PRs; the docs only say restarts go from "minutes" to "seconds".
  - For comparison, SGLang's analogous daemon (LMSYS blog, 21 Aug 2026) reports weight loading falling from ~495 s to ~0.63 s for Ling-2.6-1T FP8 on 8× H20-3e. https://www.lmsys.org/blog/2026-08-21-sglang-fast-recovery
- **Conversion.** None.
- **Reads per rank.** Its own post-processed shard, node-local, with no I/O.
- **Multi-node / TP.** Yes, with one launcher per node, since CUDA IPC is node-local.
  - Covers TP, EP and DP, and PP in the v0.33.0 code.
  - One note in the docs still says PP is rejected.
- **Citation.**
  - Docs: https://docs.vllm.ai/en/latest/features/preload/
  - PRs: https://github.com/vllm-project/vllm/pull/54921 and https://github.com/vllm-project/vllm/pull/56680
- **vs servekit.** Complementary and node-local: it removes reloads across restarts on one node, but the daemon's first load (through any disk loader) is still the cold load servekit speeds up, and ServerlessLLM's model manager already used the same CUDA-IPC hand-off in 2024.

---

## In-tree baseline to report: vLLM's safetensors prefetch (v0.33.0 code)

- **Auto-prefetch.** vLLM's default loader detects NFS/Lustre from `/proc/mounts`. If the checkpoint fits in 90% of available RAM, it prefetches the files into the page cache in the background:
  - 8 threads per rank;
  - buffered reads in 16 MiB blocks;
  - each global rank takes `files[rank::world_size]`.
- **Overrides.** `--safetensors-load-strategy {lazy,eager,prefetch}` changes this; the docstring recommends `eager` for Lustre/NFS.
- **Multi-node effect.** With multi-node TP, each node warms only the files assigned to its local ranks **[inferred from code]**.
- **Action for us.** servekit's "default loader" baseline should state which strategy was active.

## Gaps for the write-up

- **No Lustre results anywhere.** None of the sources above reports results on HDD-backed Lustre. Their storage is:
  - local NVMe or tmpfs (ServerlessLLM, fastsafetensors, InstantTensor);
  - EBS or S3 (Run:ai);
  - HTTP/S3 (Tensorizer);
  - VAST, S3 or Hugging Face (ModelExpress);
  - peer GPU and host memory (λScale, BlitzScale).
- **Per-rank pre-sharded files are not new.** They already exist in ServerlessLLM (own format), Tensorizer (own format), and vLLM's `sharded_state` / `runai_streamer_sharded`, which is the format servekit reuses.
- **What remains specific to servekit:**
  - node-local RAM staging from a shared parallel file system, run before and alongside engine start and gated by a per-worker barrier, on multi-node TP;
  - logprob verification.

## Summary table

| work | mechanism | reads per rank | needs conversion | multi-node | vLLM flag | best headline number + setup |
|---|---|---|---|---|---|---|
| ServerlessLLM | Per-GPU raw-tensor partitions; multi-threaded `O_DIRECT` reads; pinned chunk pool; SSD→DRAM→GPU pipeline; load overlaps engine init | shard | yes (own format) | server cluster; cross-node TP not described | none upstream (`serverless_llm` via sllm-store patch, vLLM 0.9.0.1) | LLaMA-2-70B in 10.3 s vs 48.0 s Safetensors (4.7×); 8× A5000, RAID0 2× NVMe (12 GB/s) |
| fastsafetensors | Bulk file→GPU copy (GDS or bounce buffers), DLPack; files round-robin to ranks, then NCCL broadcast/scatter | 1/N of files from storage; full model per GPU via NCCL | no | group can span nodes; evaluated on 1 node | `fastsafetensors` (GDS only at world size 1) | 4.8–7.5× vs safetensors; Llama-70B at 26.4 GB/s on 4× A100 + 4 NVMe (28 GB/s max) |
| Run:ai Model Streamer | C++ thread pool; concurrent reads into bounded CPU buffer; overlapped H2D; S3/GCS/Azure; optional distributed broadcast | whole (default); 1/N + broadcast (`distributed`); shard (`_sharded`) | no; `_sharded` needs `sharded_state` | yes | `runai_streamer`, `runai_streamer_sharded` | Llama-3-8B from S3 in 4.88 s (concurrency 32); io2 7.53 s vs 47 s safetensors; 1× A10G |
| ModelExpress + NIXL | RDMA read (NIXL) of post-processed weights from a serving replica; fallback InstantTensor → ModelStreamer → GDS → default | own rank's tensors from peer [inferred] | no | yes (TP/DP/PP validated) | `modelexpress` (`pip install modelexpress`) | DeepSeek-V4-Pro load 11 s P2P vs 1 m 10 s local cold; TP8, 8× B200, CX-7 |
| InstantTensor | Direct I/O (io_uring/AIO), cross-file read planning, pipelined prefetch, NCCL-coordinated reads | storage reads split; full model per GPU | no | world group can span nodes; single-node results only | `instanttensor` | DeepSeek-R1 in 15.3 s (45 GB/s) vs 160 s; 8× H200, unspecified "50 GB/s storage" |
| λScale | Block-wise binomial-pipeline RDMA/GDR multicast plus execute-while-load pipelines | n/a (blocks from peers) | none described | yes | none | Llama-2-13B scaled to 8 nodes in <1 s; H800, 400 Gb/s IB |
| BlitzScale | Multicast of parameters from serving GPUs over RDMA/NVLink, O(1) host cache, layer-level live scaling | n/a (peer GPU memory) | none described | yes | none | Up to 94% lower tail latency vs ServerlessLLM; 4×8 A800, 100 Gbps RDMA |
| Tensorizer | Own single-file format; concurrent range readers streaming HTTP/S3/local → GPU | shard (per-rank file) | yes (`.tensors`) | per-rank files; not specifically documented | `tensorizer` | GPT-J 20 GB at ~5 GB/s on 40 GbE |
| vLLM `ipc_cache` / `preload` | Per-GPU daemon holds post-processed shard; engines map it via CUDA IPC | own shard, no I/O | no | yes (one launcher per node) | `vllm preload` + `--load-format ipc_cache` (v0.30 / v0.31) | No vLLM number; SGLang analogue 495 s → 0.63 s (Ling-2.6-1T, 8× H20-3e) |
| *servekit (for contrast)* | Preshard plus `/dev/shm` staging (64 `O_DIRECT` slices per file) plus overlap barrier | shard | yes (`sharded_state`, once per TP layout) | yes | n/a (this port) | Llama-3.1-70B 453.7 s → 9.7 s on capstor Lustre |
