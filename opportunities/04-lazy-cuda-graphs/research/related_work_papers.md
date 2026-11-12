# Related work: lazy, traffic-driven CUDA-graph capture in LLM serving engines

Compiled 2026-11-11/12 for the write-up on lazy, traffic-driven CUDA-graph capture (SGLang 0.5.20 first, then vLLM).

**Our idea (for reference).** The server starts serving after capturing only a small, traffic-chosen set of CUDA graphs. It captures the remaining sizes one at a time while the scheduler is idle, in lockstep across tensor-parallel (TP) ranks, because the graphs contain NCCL collectives. The order is expected value = frequency × graph speedup ÷ capture cost. Optionally it promotes hot sizes that are running eagerly. It needs no privileges, no driver features and no offline step. CRIU and GPU snapshotting are not allowed on CSCS Alps. On the lab's GH200 runs, once weight loading is fixed, graph capture is 35–47% of a 95–117 s cold start.

**Legend.**
- **Read:** I read the primary source's full text in this session. The method is noted for each source.
- **UNVERIFIED:** I did not confirm it from the primary source in this session.
- **Derived:** my own arithmetic on numbers the source reports.

Each source gets at most one short quote. Everything else is paraphrased.

---

## 0. Bottom line

| Work | Attacks graph capture? | How | Offline step | Privileges / driver features | Serves before all graphs are ready? |
|---|---|---|---|---|---|
| Medusa (ASPLOS'25) | Yes | Offline materialization of graph topology; pointer patching by allocation index; kernel lookup by name | Yes, per (GPU type, model) | None stated by the mechanism; artifact checklist lists root access (reason not given) | No |
| Foundry (arXiv 2604.06664) | Yes | Offline SAVE of topology plus execution context; deterministic VMM layout; kernel-binary reload; topology templates plus `cuGraphExecUpdate` | Yes, per config, on 1 GPU | LD_PRELOAD driver hook and CUDA VMM APIs; no root, CRIU or cuda-checkpoint | Not stated; it specializes non-template sizes on first replay (~1–3 ms each) |
| vLLM RFC #20098 / PR #23184 (2025) | Yes | Lazy or delayed capture at runtime; idle-time capture suggested but never built | No | None | Yes (PR); closed, not merged |
| LMDeploy (PyTorch engine) | Implicitly | Graph runner captures on the first forward with a new size key | No | None | Yes in v0.7.0 (no pre-capture found); a startup `warmup()` exists by v0.10.0 |
| cuda-checkpoint family (Modal, Dynamo Snapshot, Parasail, vLLM RFC #34303, SGLang RFC #43965) | Yes, as part of a process snapshot | Process-level C/R | Yes (snapshot creation) | Driver C/R API (≥550; NCCL and IPC caveats); CRIU and privileged agents in most setups | No |
| ServerlessLLM, λScale, BlitzScale, DeepServe, HydraServe | No (weights, process and pod) | Faster or overlapped weight loading, pre-warmed pools, serving during load | Varies | RDMA, NPU links, pools | λScale and BlitzScale serve during weight load |

The idea that already exists closest to ours is in the vLLM community. RFC #20098 and PR #23184 proposed lazy and delayed capture, and a maintainer explicitly suggested capturing while the server is idle. That suggestion was deferred and never implemented. The PR was put on hold and then closed. The closest paper is Foundry: it is unprivileged but needs an offline step. Medusa is also close, and it explicitly argues that deferring capture is ineffective.

---

## 1. Assigned works

### 1.1 Medusa (ASPLOS 2025)

1. **Citation.** Shaoxun Zeng, Minhui Xie, Shiwei Gao, Youmin Chen, Youyou Lu (Tsinghua University). *Medusa: Accelerating Serverless LLM Inference with Materialization.* ASPLOS '25, Volume 1, Rotterdam, Mar 30–Apr 3 2025, pp. 653–668. DOI [10.1145/3669940.3707285](https://doi.org/10.1145/3669940.3707285). Open access (CC BY 4.0). I found no arXiv version. Artifact: https://github.com/thustorage/Medusa (Apache-2.0).
   - **Read:** full 16-page text through the ACM DL ePDF reader. WebFetch got HTTP 403 from ACM.
2. **What it does about cold start and CUDA graphs.**
   - It targets the "loading phase" of serverless vLLM cold starts.
   - It identifies two LLM-specific stages: the KV-cache-initialization profiling forward, and capture of CUDA graphs for 35 batch sizes (vLLM's default at the time).
   - It materializes both offline and restores them online. It does not address torch.compile or JIT.
   - §2.4 explicitly considers deferring capture until requests of each batch size arrive, and rejects it. The paper's argument is that deferral "merely delays and disperses it across different requests". It also rejects naive asynchronous overlap of loading stages, because capture depends on KV-cache init.
3. **Mechanism.**
   - **Offline phase,** run once per <GPU type, model type>:
     - Runs a normal cold start while intercepting buffer allocations and frees and `cudaLaunchKernel`.
     - Captures all 35 graphs, saving nodes and dependencies, and records the free GPU memory for the KV cache.
     - An analysis stage maps every 8-byte pointer-like kernel parameter to an *indirect index pointer*: its index in the allocation sequence. It matches backwards from each launch to avoid false positives from reused addresses.
     - It validates the result by comparing outputs of the original and restored graphs.
   - **Online:**
     - Replays the allocation and free sequence and patches the new addresses into node parameters. Constants are restored by value.
     - Rebuilds the graphs with the explicit graph-construction APIs.
     - Restores contents only for "permanent" buffers. Weights are allocated before capture; temporaries are freed after capture. Only 9.0% of kernels need permanent buffers, each two 4-byte magic numbers.
   - **Kernel addresses:**
     - Exported kernels are resolved with `dlopen`/`dlsym` plus `cudaGetFuncBySymbol` (69.2% of kernels for Llama2-13B at bs=1).
     - Hidden kernels such as cuBLAS are found by forcing module load with "triggering kernels", then `cuModuleEnumerateFunctions` plus `cuFuncGetName`.
     - The trigger is running and capturing the model's **first layer** online for each batch size. This assumes structurally identical layers.
   - **KV cache:** the offline free-memory value replaces the profiling forward. KV init is reordered before weight loading, so capture warmup overlaps with weight loading.
   - **Memory pools and relocation:** pointers are relocated by allocation-sequence replay. Device-side allocations and indirect pointers are not handled; the paper found none in 139,364 nodes.
   - **Does not serve before capture finishes.** All 35 graphs are restored in the loading phase.
   - The paper also notes that capturing several graphs at once in a process causes CUDA errors, so graphs are captured one by one.
4. **Key numbers.**
   - **Setup:**
     - Hardware: 4× A100-40GB SXM4 with NVLink, 2× EPYC 7642. Most experiments used one GPU.
     - Software: PyTorch 2.2.0, CUDA 12.4.
     - Baseline: vLLM, with 35 capture sizes.
     - Models: 10 models (Falcon-7B, Llama2-7B/13B, Qwen1.5-0.5B/1.8B/4B/7B/14B, Yi-6B/9B) with 139,364 graph nodes in total.
   - **Breakdown, Qwen1.5-4B under vLLM:**
     - Cold start: runtime init 22%, loading phase 76%, first token 2%.
     - Loading phase 2.85 s: model-structure init 0.85 s, weights 0.39 s, tokenizer 0.21 s, KV init 0.50 s, capture 0.90 s.
   - **Average over the 10 models:** capture is 32% of the loading phase and KV init 18%. The two stages together are described as 47% (§2.1) and as 50% (intro), so the paper is internally inconsistent here.
   - **CUDA graphs** speed up end-to-end request latency by up to 2.4× (prompt 161 / output 338 tokens).
   - **Loading phase:**
     - −42.5% on average vs vLLM, and −34.4% vs vLLM + async loading.
     - Cold start −34.9% on average.
     - Best case Llama2-13B (−42.9%); worst case Qwen1.5-0.5B (−21.1%).
   - **Qwen1.5-4B after Medusa:**
     - Loading phase 2.85 s → 1.67 s.
     - Capture stage 0.90 s → 0.57 s, because it still warms up and captures the first layer.
     - KV init 0.50 s → 0.02 s.
   - **Offline phase:** 39.2 s on average, of which the capture stage averages 9.7 s.
   - **P99 TTFT** (ShareGPT, Poisson arrivals):
     - −50.5% for Llama2-7B and −53.0% for Qwen1.5-4B at RPS 2.
     - At about 4.5 QPS (Llama2-7B), 43.0%, 29.9% and 27.0% lower than vLLM, vLLM + async, and no CUDA graphs respectively.
5. **Requirements.**
   - An offline capture and analysis run per (GPU type, model).
   - Interception of allocation and launch calls, inside the Medusa build; the exact hook mechanism is UNVERIFIED.
   - Explicit graph APIs and module-enumeration APIs.
   - Single GPU only; multi-GPU is future work.
   - Not CRIU and not cuda-checkpoint.
   - The artifact checklist lists root access without a reason.
   - Foundry reports that Medusa's rules target driver 550.54.14 and specific kernels. That is Foundry's claim, not checked in Medusa.
6. **How it differs from our idea.**
   - Medusa removes capture from the critical path by moving it **offline**, with kernel-specific patching and a single-GPU restriction. We need no offline artifact, serve immediately, and pay capture cost only in idle gaps, across TP ranks.
   - Medusa's §2.4 argument against deferral is the main objection we must answer. Idle-time capture puts no capture on the request path, except for the bounded head-of-line stall of one in-flight capture. EV ordering also front-loads the valuable graphs.
   - The two are complementary in principle: a materialized set could seed the initial set. In practice, Medusa's brittleness and its offline step rule it out on Alps.

### 1.2 Foundry (arXiv 2604.06664)

1. **Citation.** Xueshen Liu (U. Michigan), Yongji Wu (UC Berkeley; corresponding), Yuncheng Yao (Duke), Danyang Zhuo (Duke), Ion Stoica (UC Berkeley), Z. Morley Mao (U. Michigan). *Foundry: Template-Based CUDA Graph Context Materialization for Fast LLM Serving Cold Start.* arXiv:[2604.06664](https://arxiv.org/abs/2604.06664) v1, 8 Apr 2026, cs.DC. No venue listed. Code: https://github.com/foundry-org/foundry.
   - **Read:** arXiv HTML plus PDF text, all sections and appendices. The id resolves correctly.
2. **What it does about cold start and CUDA graphs.**
   - It argues that once weights load over RDMA in 1–2 s, CUDA-graph capture is the dominant cold-start cost.
   - It persists both graph topology and **execution context** (memory layout, kernel binaries) in an offline SAVE. It rebuilds executable graphs at LOAD, with no warmup or capture.
   - Loading kernel binaries from the archive also avoids torch.compile's bytecode transformation and autotuning at start.
   - Integrated with vLLM v0.11.2.
3. **Mechanism.**
   - **Driver hook.** A C/C++ library injected via `LD_PRELOAD` redirects every CUDA allocation to the CUDA **VMM** APIs, in a reserved virtual-address region at a fixed base.
     - Allocation is monotonic, so the same allocation sequence yields an identical layout. No kernel argument is rewritten, which also covers cuBLAS's opaque argument structs.
     - Transient allocations made inside the capture window are recorded at SAVE and replayed at LOAD. This is why disabling ASLR would not be enough.
     - LOAD maps the whole recorded range in one call, after which allocations are pointer bumps.
   - **Kernel binaries.**
     - Intercepts the `cuModuleLoad`/`cuLibraryLoad` families and extracts in-memory module payloads (e.g., from `libcublas.so`).
     - Records the load API and options plus a catalog keyed by (content hash, mangled name).
     - Pre-links multi-segment device code into a cubin, and flags modules that need NVSHMEM's `nvshmemx_cumodule_init` (DeepEP).
   - **Topology templates.**
     - Graphs for batch sizes 1–512 collapse to 12–25 unique topologies (22 for Qwen3-14B).
     - LOAD builds one template per topology with the explicit graph APIs.
     - Every other size is specialized **on demand at first replay** with `cuGraphExecUpdate`, which swaps kernel function, argument buffer and launch dimensions. In the paper's words, Foundry "specializes the template on demand at serving time".
     - Cluster-dimension attributes set via `cuGraphKernelNodeSetAttribute` are treated as topology.
     - Parallel graph construction contends inside the driver (Appendix A). Templates are therefore built on one thread while worker threads prepare parameters.
   - **Multi-GPU.** SAVE runs on one GPU with a stub over NCCL and NVSHMEM that does dummy communication. LOAD swaps in real kernel handles and rank-specific arguments.
     - Claimed for DP, TP and EP, but evaluated only on DP and EP. Pipeline parallelism is not supported.
     - Archive storage drops proportionally to rank count (e.g., 64× for 64 GPUs).
   - **Constraints.** The KV-cache size must be fixed before SAVE and LOAD so the layout matches.
   - **Timing.** Reconstruction runs asynchronously with weight and KV initialization. The paper does not say requests are admitted before LOAD finishes.
4. **Key numbers.**
   - **Setup:**
     - Hardware: 8×H200 DGX (Xeon 8480C) and 8×B200 DGX.
     - Software: driver 590.48, CUDA 13.1, vLLM 0.11.2, PyTorch 2.9.
     - 512 graphs for batch sizes 1–512.
     - Pre-warmed environments, weight loading excluded, warm torch.compile cache.
   - **Fig. 1** (vLLM worker init, Qwen3-14B, 2×H200): total 49.9 s, of which graph capture is 29.4 s, about 59% (derived).
     - The other printed values, by legend order, are NCCL init 2.7, model loading 6.7, torch.compile 7.6, memory profiling 3.4, and sampler warmup 0.05 s. I inferred the mapping from the extracted figure text; the values sum to 49.85 s.
     - The text says vLLM still takes 8 s with warm torch.compile caches.
   - **Cold start, vLLM with graphs → Foundry:**
     - Qwen3-14B DP1–8: 36–48 s → 1.7–1.8 s.
     - Llama3-8B: 28 s → 1.3 s.
     - Gemma3-12B: 45 s → 2.0 s.
     - Qwen3-30B-A3B EP2–8 BF16: 112–154 s → 2.7–2.8 s.
     - Qwen3-235B-A22B EP8 BF16: 650 s → 3.9 s.
     - FP8 Qwen3-30B-A3B: 72–83 s → 2.8–2.9 s.
     - FP8 Qwen3-235B EP4–8: 136–146 s → 3.5 s.
     - For reference, vLLM in eager mode takes 1.2–62 s.
   - **Per-graph cost** (Qwen3-14B DP4 and Qwen3-30B-A3B EP4, H200):
     - Stream capture: **59.7–198.6 ms**.
     - Template build: 31.1–69.5 ms.
     - `cuGraphExecUpdate` specialization: 0.98–2.89 ms.
     - Building 512 graphs sequentially would take about 35.6 s for Qwen3-30B-A3B.
   - **vs cuda-checkpoint + CRIU** (single GPU, 4 configs):
     - C/R restore takes 5.7–6.1 s; Foundry takes 1.3–2.3 s.
     - The archive is 4–5× smaller than the checkpoint image (Qwen3-14B: 1.1 GB vs 3.7 GB). Qwen3-235B EP8 needs a 2.2 GB archive, of which 1.4 GB is binaries.
     - cuda-checkpoint lacks the IPC memory that DeepEP needs.
   - TPOT is unchanged, and generated tokens are identical.
5. **Requirements.**
   - An offline SAVE per (model, parallelism config, software stack), run on a single GPU.
   - The `LD_PRELOAD` interposition, a free fixed-base VA range, CUDA VMM APIs and `cuGraphExecUpdate` (standard driver API), and a fixed KV size.
   - It needs **no root, CRIU or cuda-checkpoint**, and no modified driver.
   - Tested only on x86 DGX with driver 590.48 and CUDA 13.1. Portability across GPU types, drivers or aarch64 (GH200) is not discussed (UNVERIFIED).
6. **How it differs from our idea.**
   - Foundry eliminates capture cost with an offline artifact and a determinism contract on the allocator. We avoid any artifact or interposition and instead hide capture in idle time.
   - Foundry's readiness is 1.3–3.9 s; ours starts serving earlier but with partial graph coverage.
   - It is **complementary**: if an archive exists, LOAD is the fast path, and EV ordering could also choose which templates to build or specialize first.
   - Foundry needs no privileges, so it is a plausible competitor on Alps if an offline SAVE step and LD_PRELOAD are acceptable there. The write-up must argue the *no offline step, no version-locked artifact* case explicitly.
   - Its on-demand `cuGraphExecUpdate` is itself a small lazy step on the request path, which supports *lazy is fine when cheap*.

### 1.3 λScale (arXiv 2502.09922)

1. **Citation.** Minchen Yu (CUHK-Shenzhen), Rui Yang (UVA), Chaobo Jia (CUHK-SZ), Zhaoyuan Su (UVA), Sheng Yao (HKUST), Tingfeng Lan (UVA), Yuchen Yang (HKUST), Zirui Wang (UVA), Yue Cheng (UVA), Wei Wang (HKUST), Ao Wang (Alibaba), Ruichuan Chen (Nokia Bell Labs). *λScale: Enabling Fast Scaling for Serverless Large Language Model Inference.* arXiv:[2502.09922](https://arxiv.org/abs/2502.09922). v1 14 Feb 2025; v3 5 Mar 2026. No venue on the arXiv page.
   - **Read:** v3 PDF text. The id resolves correctly.
2. **What it does about cold start and CUDA graphs.**
   - It addresses scale-out of model weights: RDMA multicast plus **"execute-while-load"** distributed inference.
   - It does **not** mention CUDA graphs, warmup or compilation; grep of the full text confirms this.
3. **Mechanism.**
   - λPipe splits the model into blocks and multicasts them with a binomial-pipeline algorithm over GPUDirect RDMA. It is built on Derecho RDMC with one-sided RDMA and GDR extensions.
   - It forms execution pipelines across receiving nodes, so they serve before any node holds the full model. When loading completes, it switches to local execution.
   - Locality-driven startup: GPU (hot), host memory (warm), or remote (cold).
   - It packs tensors per block and pre-allocates GPU memory for blocks and activations.
   - The inference engine is built on Meta's Llama inference code; the KV cache is managed by the engine.
4. **Key numbers.**
   - **Setup:**
     - Shared HPC cluster with H800 GPUs.
     - Testbed1: 12 nodes × 1 H800. Testbed2: 6 nodes × 4 H800.
     - Each node has 400 Gb/s InfiniBand, 1 TB RAM and 4 TB NVMe.
     - Models: Llama-2 7B, 13B and 70B.
   - Scales Llama-13B to 8 nodes in under 1 s, up to 1.5× faster than NCCL.
   - On real traces vs ServerlessLLM, FaaSNet and NCCL: 2.4–5× better P90 tail latency and 17.8–31.3% lower cost.
   - Only models that fit in one multi-GPU node are supported.
5. **Requirements.** RDMA/InfiniBand with GPUDirect RDMA, and host-memory caching. No offline step or privileges are stated.
6. **How it differs from our idea.**
   - λScale is orthogonal: it covers weights, not graphs.
   - It is a conceptual precedent for **serving in a degraded mode, then upgrading in place** (pipelined → local), analogous to our eager → graph promotion. It is complementary.

### 1.4 ServerlessLLM (OSDI 2024)

1. **Citation.** Yao Fu, Leyang Xue, Yeqi Huang, Andrei-Octavian Brabete (U. Edinburgh), Dmitrii Ustiugov (NTU Singapore), Yuvraj Patel, Luo Mai (U. Edinburgh). *ServerlessLLM: Low-Latency Serverless Inference for Large Language Models.* OSDI '24, Santa Clara, July 10–12 2024. Pages 135–153 per Foundry's bibliography.
   - URL: https://www.usenix.org/conference/osdi24/presentation/fu. PDF: https://www.usenix.org/system/files/osdi24-fu.pdf.
   - arXiv:2401.14351: the id appeared in a search-result listing under this title; I did not open the abs page.
   - **Read:** USENIX PDF text.
2. **What it does about cold start and CUDA graphs.**
   - Checkpoint (weight) loading and locality: a loading-optimized checkpoint format, multi-tier loading, live migration of inference, and startup-time-aware scheduling.
   - **No mention of CUDA graphs, warmup or compilation;** grep of the full text confirms this.
3. **Mechanism.**
   - Chunked, partitioned per-GPU checkpoint files plus a tensor index, so tensors are set by base address plus offset.
   - Direct I/O (`O_DIRECT`), chunk-based parallel reads and pinned memory.
   - A model-manager process loads the data via CUDA IPC handles, while the inference process builds the model object.
   - Live migration is token-based: the KV cache is recomputed at the destination.
4. **Key numbers.**
   - **Setup:** a server with 8× A5000 and RAID0 NVMe (12 GB/s), plus a cluster of 4 nodes × 4 A40 on 10 GbE.
   - Loading is 3.6–8.2× faster than PyTorch and Safetensors:
     - OPT-2.7B: 6× and 3.6×.
     - LLaMA-2-70B: 8.2× and 4.7×.
   - A LoRA adapter loads in 83.5 ms vs 370 ms.
   - End-to-end latency is 10–200× lower than the serverless baselines.
   - Motivation numbers: OPT-30B onto 4 GPUs takes 34 s with PyTorch; LLaMA-2-70B onto 8 GPUs takes 84 s.
5. **Requirements.** Local SSD and DRAM tiers, and conversion to its checkpoint format. No privileges are stated.
6. **How it differs from our idea.** It is orthogonal and complementary. Once weights load fast, graph capture becomes the dominant remaining cost, as Foundry's Fig. 1 shows. That is our premise.

### 1.5 BlitzScale (OSDI 2025)

1. **Citation.** Dingyan Zhang, Haotian Wang, Yang Liu, Xingda Wei (SJTU), Yizhou Shan (Huawei Cloud), Rong Chen, Haibo Chen (SJTU). *BlitzScale: Fast and Live Large Model Autoscaling with O(1) Host Caching.* OSDI '25, Boston, July 7–9 2025, pp. 275–293.
   - URL: https://www.usenix.org/conference/osdi25/presentation/zhang-dingyan. arXiv:[2412.17246](https://arxiv.org/abs/2412.17246) (v2, 15 Jun 2025). Code: github.com/blitz-serving/blitz-scale.
   - **Read:** USENIX PDF text and the arXiv abs page. The id resolves correctly.
2. **What it does about cold start and CUDA graphs.**
   - It optimizes the **data plane**: parameters are loaded over the compute network with multicast, using O(1) host caching.
   - Scaling is **live**: layer-level cooperative execution lets new instances take over layers before loading finishes.
   - It explicitly leaves the control plane (CUDA context and kernel loading) to C/R work, citing PhOS/ParallelGPUOS and Medusa, and to its native engine.
   - **No CUDA-graph discussion.**
3. **Mechanism.**
   - Network-based parameter loading with multicast chains; ZigZag scheduling between deployed and scaling instances.
   - A native C++/Rust serving engine that avoids PyTorch init.
   - A **CUDA context pool with pre-loaded kernels (cuModule)**, because context creation takes about 500 ms.
   - A pre-created full-mesh connection pool instead of NCCL group setup (about 100 ms).
4. **Key numbers.**
   - Up to 94% lower tail latency vs ServerlessLLM.
   - 49% less GPU time vs non-autoscaling DistServe and vLLM at the same SLA.
   - Llama3-8B inference takes 80–900 ms on an A800; loading it from a 10 Gbps SSD takes 12.8 s.
   - Testbeds include A800 SXM and A100 PCIe with 200 Gbps RDMA.
5. **Requirements.** An RDMA or NVLink compute network and its native engine. No privileges or offline step are stated.
6. **How it differs from our idea.** It is orthogonal (weights and context). Its "live" scaling is the same *serve-before-ready* principle we apply to graphs. It is complementary.

### 1.6 DeepServe, formerly DeepFlow (USENIX ATC 2025; arXiv 2501.14417, Huawei Cloud)

1. **Citation.**
   - Authors: Junhao Hu (Peking University), Jiang Xu, Zhixia Liu, Yulong He, Yuetao Chen, Hao Xu, Jiang Liu, Jie Meng, Baoquan Zhang, Shining Wan, Gengyuan Dan, Zhiyu Dong, Zhihao Ren, Changhong Liu, Tao Xie (PKU), Dayun Lin, Qin Zhang, Yue Yu, Hao Feng, Xusheng Chen, Yizhou Shan (Huawei Cloud).
   - *DeepServe: Serverless Large Language Model Serving at Scale.* USENIX ATC '25, Boston, pp. 57–72. URL: https://www.usenix.org/conference/atc25/presentation/hu-junhao.
   - arXiv:[2501.14417](https://arxiv.org/abs/2501.14417): v1 (24 Jan 2025) was titled *"DeepFlow: Serverless Large Language Model Serving at Scale"*; v3 (9 Jun 2025) is titled DeepServe. The v1 author list also includes Chao He.
   - **"DeepFlow" in the candidate list is this same paper, not a separate system.** HydraServe cites it as DeepFlow.
   - **Read:** arXiv v3 PDF text plus the USENIX page.
2. **What it does about cold start and CUDA graphs.**
   - An Ascend NPU platform, not CUDA. Fast scaling comes from:
     - pre-warmed pods;
     - pre-warmed, model-agnostic and parallelism-agnostic engine processes (TEs), which take Python import, NPU init and HCCL setup off the critical path;
     - DRAM pre-loading of predicted models;
     - NPU-fork (weights copied from a running TE over HCCS or RoCE).
   - For the TE-Post-Load step: warm-up profiling of KV HBM is replaced by offline profiling and asynchronous block allocation. A **dummy request after startup** absorbs first-request slowness.
   - Graph backends are FlowTurbo (torch-based dynamic graph) and ATB (C++ static graph). Graph compilation and capture time is not analysed.
3. **Mechanism.**
   - Scaling steps: Scaler-Pre, TE-Pre-Load, TE-Load, TE-Post-Load, Scaler-Post.
   - TE-Pre-Load is cut by about 35% through late imports and parallel init, but it still dominates scaling time.
   - It scales to 64 instances in seconds.
4. **Key numbers.**
   - Per-step seconds appear only in Figs. 9 and 10. I could not map them reliably from the extracted text (UNVERIFIED).
   - Text states that PyTorch tensor init takes about 0.3 s on the DRAM-hit path, and that 1.5 TB DRAM per machine is enough to pre-load 10 × 70B or 100 × 7B models.
5. **Requirements.** Ascend 910B/910C clusters with HCCS or RoCE, standing pools of pre-warmed resources (idle capacity), and offline profiling.
6. **How it differs from our idea.**
   - It is complementary and off-platform. Its pre-warming keeps resources idle; we use otherwise-idle scheduler time.
   - The post-startup dummy request is a primitive *warm in the background* step.

---

## 2. Other work found (2023–2026)

### 2.1 Direct precedents for lazy, deferred or idle-time graph capture (most important)

**vLLM RFC #20098, "[RFC]: Lazy CUDA Graph capture"** ([link](https://github.com/vllm-project/vllm/issues/20098))
- **Status:** opened 25 Jun 2025 by lionelvillard; labels RFC, startup-ux, stale. Closed as *not planned* by the stale bot on 26 Dec 2025.
- **Motivation:** vLLM captures 67 graphs by default, which can take more than 10 s.
  - charlesfrye (Modal) said boots take about 45 s without graphs, so about 10 s more is about 20%.
  - A LoRA user reported startup of up to 7 minutes for an 8B model with `max_loras` 36.
  - BoyuanFeng estimated about 5,000 piecewise graphs for a 70B model.
  - By Jul 2025, lionelvillard reported about 50 ms per graph for an 8B model.
- **Ideas raised:**
  - Capture lazily in the piecewise backend.
  - Keep separate eager and lazy size lists; charlesfrye wanted 1–2 eager graphs.
  - ProExpertProg suggested a scheduler feature that "lazily captures cudagraphs if there are no requests to be scheduled". That is **idle-time capture, proposed in June 2025.**
- **Objections:**
  - zou3519: less predictable serving.
  - BoyuanFeng: abrupt performance drops.
  - yeqcharlotte and LucasWilkinson: OOM, because graph memory is not counted in memory profiling.
  - youkaichao (member, 22 Aug 2025): strongly opposed, citing maintenance burden.
  - ProExpertProg (Aug 2025): capture is now fast enough not to pursue further.
- **Read:** the GitHub issue web page; individual comments verified with targeted re-reads.

**vLLM PR #23184, "ON HOLD - [Core] Lazy/Delayed CUDA graph"** ([link](https://github.com/vllm-project/vllm/pull/23184))
- **Status:** by diegocastanibm (IBM), opened 19 Aug 2025, on hold 26 Aug 2025. Closed 2 Mar 2026 by ProExpertProg after going stale. Not merged.
- **Design:**
  - Flag `use_delay_cudagraph_capture`.
  - At each execution step it captures one graph: the current runtime shape if not yet captured, otherwise the largest uncaptured shape.
  - After review, the largest size is captured at init so later captures reuse its memory.
  - gc is frozen during capture.
- **Measurement (Llama-3.1-8B, ShareGPT, 1,000 prompts):**
  - Init capture of 4 s (0.54 GiB) removed.
  - Mean TTFT 20,493.77 → 20,431.48 ms.
  - TPOT 79.60 → 79.64 ms.
  - Throughput 13.73 → 13.50 req/s.
- **Review:**
  - WoosukKwon requested changes, saying capturing "at runtime affects reliability a lot", and asked to restrict capture to init time.
  - ProExpertProg asked for a mechanism to capture while the server is idle; the author deferred this to a separate PR that never appeared.
  - A user reported **garbled output on the first request under multi-TP**; the author could not reproduce it.
- **Read:** the PR web page; key sentences verified with targeted re-reads.

**LMDeploy PyTorch engine: lazy, on-request capture**
- **Source read:** `lmdeploy/pytorch/backends/cuda/graph_runner.py` at tags v0.7.0 and v0.10.0, via raw.githubusercontent.com, not the API.
- `CUDAGraphRunner.__call__` captures a new graph when `graph_key not in self._runner_map` on a live forward. It runs one eager warmup inside `capture()` first.
  - The key is (next power of two of `num_tokens`, `is_decoding`) in v0.7.0. v0.10.0 uses `get_capture_batch_sizes()`: powers of 2 up to `min(256, max_batches)`, then steps of 256.
- v0.7.0's `model_agent.py` has **no** startup pre-capture.
- v0.10.0's `model_agent.py` has a `warmup()` that runs one dummy decode per capture size, largest first. I did not verify its call site.
- So **lazy on-request capture shipped as LMDeploy's behaviour at least in v0.7.0**. It is SPMD-implicit: every TP rank sees the same miss in the same step.

**PyTorch CUDA Graph Trees** (`torch.compile(mode="reduce-overhead")`)
- **Read:** PyTorch 2.5 docs page `torch.compiler_cudagraph_trees.html`.
- The first run warms up and the second records and replays. They "re-record CUDAGraph for every unique shape of an input tensor".
- A single memory pool is shared across recordings.
- A flag `cudagraph_skip_dynamic_graphs` exists to avoid this.
- This is lazy per-shape capture on the critical path.

**llama.cpp CUDA graphs** (NVIDIA blog, Alan Gray, 7 Aug 2024, [link](https://developer.nvidia.com/blog/optimizing-llama-cpp-ai-inference-with-cuda-graphs/))
- **Read:** the blog page.
- Graphs are reused across tokens and "only recaptured when required".
- `cudaGraphExecUpdate` updates the executable instead of re-instantiating it, and kernel parameters are patched per token.
- Batch size 1 only. Up to 1.2× speedup (Llama 7B, H100).
- Precedent for cheap runtime update over capture.

**TensorRT-LLM**
- `ExtendedRuntimePerfKnobConfig` has `cuda_graph_mode` (default False) and `cuda_graph_cache_size` (default 0), described as the number of graphs "cached in the runtime" ([1.4.0rc1 API reference](https://nvidia.github.io/TensorRT-LLM/1.4.0rc1/llm-api/reference/ExtendedRuntimePerfKnobConfig.html), **read**).
  - A bigger cache is faster but uses more memory.
  - The page does not say when the graphs are captured, or for which runtime, so whether this cache is filled lazily at runtime is **UNVERIFIED**.
- **TRT-LLM tech blog "Tuning CUDA Graph Batch Sizes for Higher Output Throughput"** ([link](https://nvidia.github.io/TensorRT-LLM/blogs/tech_blog/blog20_Tuning_CUDA_Graph_Batch_Sizes_for_Higher_Output_Throughput.html); no date in the header, footer © 2025, TRT-LLM v1.3.0rc8). **Read.**
  - Graphs are captured at startup, twice: a dry run for memory, then live.
  - DeepSeek-R1 TP=4 on GB200: first capture 10.77 s (23 graphs, default ×2 set) vs 25.90 s (49 graphs, +64 set). That is about 0.47–0.53 s per graph (derived).
  - Total init 183.62 → 215.13 s.
  - The +64 set gives up to 1.3× TPS aggregated, and up to 1.5× for DeepSeek-R1 disaggregated; it became the default.
  - Future work: "parse serving logs to compute the best set of CUDA graph batch sizes", and a dry run that captures only 2–4 graphs.
  - Traffic-driven size selection is therefore listed as **future work** by NVIDIA, not done.

**SGLang upstream**
- **PR #32017** (merged 13 Aug 2026; [link](https://github.com/sgl-project/sglang/pull/32017)) overlaps checkpoint page-cache staging with CUDA-graph capture. Capture runs on capture-safe placeholder weights before the real-weight commit.
  - Qwen3-32B BF16 on H100, decode graphs up to bs 256: TP1 84.83 → 35.60 s and TP2 78.60 → 45.16 s, mostly from prefetching.
  - The reordering alone gives 8.6–11.7%. Capture time is not reported separately.
- **RFC #43965** (opened 13 Oct 2026, open; [link](https://github.com/sgl-project/sglang/issues/43965)) proposes initialization snapshots via CRIU plus cuda-checkpoint, TP=1 only.
  - Qwen3-8B: cold start 91.78 s (cold JIT) or 58.90 s (warm JIT) → restore 18.74 s.
- **RFC #32432** (opened 26 Jul 2026; closed by the inactivity bot 10 Oct 2026, tracking moved to #34297; [link](https://github.com/sgl-project/sglang/issues/32432)) defines a capture-start contract.
  - Lazy initialization, module loading and first-use JIT must finish *before* the capture region.
  - Ranks that capture together must rendezvous at the same boundary. The related fixes add a device synchronize plus a TP barrier before graph construction.
  - It concerns startup capture, not idle-time capture, but it is direct support for our lockstep design.
- I found no lazy, deferred or idle-time capture in upstream SGLang. The v0.5.20 tag's `model_executor/runner/` listing on GitHub shows 9 files and no `lazy_capture.py`; the listing was marked truncated.
- **Read:** the PR and RFC web pages.

**Other recent vLLM items** (**read**; startup-time work, not lazy capture)
- **PR #46523, "Prewarm full cudagraph capture forward pass"** (opened 23 Jun 2026, still open).
  - Runs the capture-compatible forward eagerly once before entering `torch.cuda.graph`, so Inductor and Triton first-call autotuning stays out of the capture region.
  - This is startup capture with no measurements.
- **RFC #55259, startup self-benchmarking** (opened 3 Sep 2026, open, no maintainer reply).
  - An optional sweep at startup that times forward passes at and just above each configured capture size, plus an eager tail.
  - It does not choose sizes, use observed traffic, or defer work.
  - It could supply the "graph speedup" term of our EV model.
- **Issue #62556, "When do larger CUDA graph capture sizes help?"** (opened 19 Oct 2026, open).
  - Raising the capture ceiling to 4K tokens gives large gains for MoE and hybrid models with MTP:
    - Qwen3.8-Flash-Next-FP8 MTP3 on 4×B300: +96.8% throughput at concurrency 128.
    - GLM 5.3 Flash TP4: +54.8% at c64, with graph memory going from 6.23 to 12.88 GiB per GPU.
  - It gives no gain or small losses for dense models: Qwen3-32B on H100 TP1 lost 4.0% throughput while capture time rose from 9 s to 15–16 s.
  - Evidence that graph *coverage* value is strongly model-dependent, which supports a speedup term in the EV model rather than a fixed list.

### 2.2 Snapshot / restore (needs driver C/R and usually CRIU or a privileged agent; not allowed on Alps)

- **NVIDIA cuda-checkpoint** ([README](https://github.com/NVIDIA/cuda-checkpoint), **read**).
  - Lock, checkpoint, restore and unlock a process's CUDA state, with CRIU handling the CPU side.
  - Driver ≥550; 570 adds NVML and CRIU 4.0+ integration; 580 adds GPU migration; 595 adds ARM (aarch64); 610 adds `cuIpcGetMemHandle` IPC.
  - No UVM, and no IPC memory from `cuMemExportToShareableHandle`.
  - The README states no privilege requirements, and says nothing on NCCL or CUDA graphs.
- **CRIUgpu.** Stoyanov, Spišaková, Ramos, Gurfinkel, Vagin, Reber, Armour, Bruno. arXiv:[2502.16631](https://arxiv.org/abs/2502.16631) (23 Feb 2025). Transparent CUDA and ROCm checkpointing via the new driver features, with no steady-state overhead. **Abstract only;** graph handling is UNVERIFIED.
- **Modal, "GPU Memory Snapshots: Supercharging sub-second startup"** (Luis Capelo, Colin Weld, 30 Jul 2025; [link](https://modal.com/blog/gpu-mem-snapshots); **read**).
  - Uses the `cuCheckpointProcess*` APIs plus gVisor C/R. Restores weights, kernels, torch.compile output and captured graphs.
  - Drivers 570/575; alpha; restore needs compatible GPU hardware.
  - vLLM Qwen2.5-0.5B: 45 s → 5 s. The GPU is not stated.
- **NVIDIA Dynamo Snapshot** (blog, 27 May 2026; Saereesitthipitak, Feigin, Mailthody, Khadkevich; [link](https://developer.nvidia.com/blog/nvidia-dynamo-snapshot-fast-startup-for-inference-workloads-on-kubernetes/); **read**).
  - CRIU plus cuda-checkpoint, run by a "privileged DaemonSet". The checkpoint is taken after init, including graph capture.
  - Single-GPU vLLM and SGLang only; multi-GPU is future work.
  - CRIU restore: Qwen3-8B (26 GiB) 24 s → 4.7 s with AIO plus parallel memfd; gpt-oss-120b 119 s → 15 s.
- **Parasail** (Meghana Madhyastha, 29 Jun 2026; [link](https://www.parasail.io/blogs/cutting-cold-start-latency-snapshotting); **read**).
  - cuda-checkpoint plus a patched CRIU; NCCL is torn down and rebuilt around suspend.
  - H200: Qwen2.5-7B TP1 63.56 → 15.22 s; Llama-3.3-70B TP1 138.19 → 119.78 s; Qwen3-30B TP2 132.16 → 76.69 s.
- **vLLM RFC #34303, CUDA Checkpoint/Restore** (opened 11 Feb 2026, open; [link](https://github.com/vllm-project/vllm/issues/34303); **read**).
  - Tiers: in-process suspend, CRIU to disk, future auto-suspend.
  - Driver 570+ and CUDA 12.8+. **NCCL communicators must be destroyed and rebuilt.** The RFC flags that graphs containing NCCL collectives may need re-capture.
  - `cuCheckpointProcessRestore` returns error 801 on H100 NVLink while NCCL is active.
  - Community data: SGLang on H200, TP1, Qwen3-0.6B cold 33 s → restore 17 s.
- **GCR.** Shaoxun Zeng, Tingxu Ren, Jiwu Shu, Youyou Lu. *GPU Checkpoint/Restore Made Fast and Lightweight.* FAST '26, pp. 239–254 ([link](https://www.usenix.org/conference/fast26/presentation/zeng)).
  - A hybrid: driver-integrated C/R for control state plus interception for data buffers.
  - Checkpoint latency −72.1% vs cuda-ckpt and −63.6% vs PhOS; restore −54.2% and −87.1%; overhead under 1%.
  - Evaluated with vLLM.
  - **Read:** USENIX page plus a grep of the PDF.
  - Foundry says interception-based C/R does not restore captured graphs; I did not verify that for GCR itself.
- **PhoenixOS (PhOS).** Xingda Wei, Zhuobin Huang, Tianle Sun, Yingyi Hao, Rong Chen, Mingcong Han, Jinyu Gu, Haibo Chen. *PhoenixOS: Concurrent OS-level GPU Checkpoint and Restore with Validated Speculation.* SOSP '25, DOI 10.1145/3731569.3764813. arXiv:[2405.12079](https://arxiv.org/abs/2405.12079) (v1 May 2024, v4 Sep 2025).
  - BlitzScale cites the arXiv version under the name "ParallelGPUOS"; the arXiv page lists no earlier titles.
  - It predicts the GPU memory each kernel touches from its launch arguments and validates the prediction by binary instrumentation. This enables concurrent C/R, copy-on-write and on-demand restore, plus a pool of execution contexts.
  - It reports being orders of magnitude faster than cuda-checkpoint.
  - **Abstract only.** It does not mention CUDA graphs.
- **Modal + Mistral 3** (2 Dec 2025, [link](https://modal.com/blog/mistral-3), **read**).
  - vLLM with Ministral 3 3B: about 118 s → about 12 s (medians). Requires vLLM Sleep Mode and `enable_gpu_snapshot`.
  - It does not mention graphs. The GPU is not stated.

### 2.3 Measurements of capture cost (2025–2026)

- **"Breaking the Ice: Analyzing Cold Start Latency in vLLM."** Huzaifa Shaaban Kabakibo, Animesh Trivedi, Lin Wang (Paderborn U., IBM Research Europe). MLSys 2026. arXiv:[2606.07362](https://arxiv.org/abs/2606.07362) (v3 29 Jun 2026). **Read** (PDF).
  - **Setup:** vLLM v0.10.1.1, H100 and L40S, 22 models.
  - Startup is mostly CPU-bound. Capture is one of only two GPU-bound steps, and the only one 1.2× faster on H100 than L40S.
  - Capture takes **0.91 s (Qwen-0.5B) to 1.51 s (Qwen-MoE-14.3B)**. For Llama2-7B it grows from 0.33 s (3 sizes) to **1.8 s (60 sizes)**, about 30 ms per graph (derived). In the paper's words, capture "time scales linearly".
  - Llama3.2-3B total startup is 20.32 s.
  - vLLM startup varied more than 4× across 9 releases (OPT-6.7B, H100).
  - The authors suggest overlapping torch.compile, KV profiling and capture as future work.
- **vLLM issue #19824, "Improve startup time UX"** (Jun 2025; [link](https://github.com/vllm-project/vllm/issues/19824); **read**).
  - Llama-70B-FP8 at TP=8: P2P check ~60 s, weights ~10 s (hot page cache), Dynamo ~15 s, Inductor ~70 s, **CUDA-graph capture ~60 s**.
  - The first request then took 30–60 s, possibly due to PTX JIT.
  - Graph serialization was listed as a possible research project.
- **InstantInfer.** Yitao Yuan, Yongchao He, Shaoke Fang, Wenfei Wu. arXiv:[2607.18957](https://arxiv.org/abs/2607.18957) (21 Jul 2026). **Read** (HTML).
  - Qwen3-30B-A3B stages Process Init / Model Init / Optimize: vLLM 26/58/23 s, SGLang 27/58/56 s.
  - "Optimize" is KV init plus graph capture. The paper lists baselines vLLM 0.13.0 and SGLang 0.5.9 on H20 at TP=1, but does not explicitly tie the breakdown to that setup.
  - The SGLang Optimize stage is about 40% of the total (derived), consistent with our premise.
  - It does not optimize this stage; it calls Medusa-style techniques complementary.
- **Tensorfuse, "Reduce vLLM GPU cold start time on Kubernetes"** (1 Sep 2025; [link](https://tensorfuse.io/docs/blogs/reducing_gpu_cold_start); **read**).
  - Llama-3.1-8B on L40S: capture 54 s → 7 s by restricting capture sizes to 1, 2, 4, 8, 16, 24, 32, 64.
  - torch.compile cache: graph compilation 42 → 13 s; Dynamo stays at 10 s.
  - Total 294 → 82 s.
- **vLLM PR #23184:** 4 s of capture for Llama-3.1-8B (see 2.1).
- **TRT-LLM blog:** 10.77–25.90 s for DeepSeek-R1 TP4 on GB200 (see 2.1).
- **Foundry and Medusa:** see §1.

**Per-graph capture costs usable in the EV model**, all from the sources above:

| Source | Setting | Cost per graph |
|---|---|---|
| Medusa | Qwen1.5-4B, A100, vLLM, includes warmup | ~26 ms (derived: 0.90 s / 35) |
| Breaking the Ice | Llama2-7B, H100, vLLM 0.10.1.1 | ~30 ms (derived: 1.8 s / 60) |
| vLLM RFC #20098 | 8B model | ~50 ms (comment) |
| Foundry | Qwen3-14B DP4 / Qwen3-30B-A3B EP4, H200, vLLM 0.11.2 | 59.7–198.6 ms (measured) |
| TRT-LLM blog | DeepSeek-R1 TP4, GB200 | ~470–530 ms (derived) |

Costs vary by about 20× across settings, so the EV model needs a per-deployment cost estimate. Measuring the first capture is enough.

### 2.4 Cold-start systems that do not touch graphs (context)

- **HydraServe** (formerly *"Towards Swift Serverless LLM Cold Starts with ParaServe"*, arXiv v1).
  - Chiheng Lou, Sheng Qi, Chao Jin, Dapeng Nie, Haoran Yang, Yu Ding, Xuanzhe Liu, Xin Jin. NSDI '26 (accepted per arXiv comments). The pages 415–430 come from a search snippet (UNVERIFIED).
  - arXiv:[2502.15524](https://arxiv.org/abs/2502.15524). **Read** (NSDI prepub PDF).
  - Pipeline-parallel model fetching across servers, overlapped worker startup, and pipeline consolidation.
  - Cold start 1.7–4.7× lower; SLO attainment 1.43–1.74× higher. No offline profiling.
  - Setup: vLLM, Llama2-7B on A10.
  - It states that "prior work has largely eliminated CUDA graph and KV cache initialization overhead", meaning Medusa.
- **WarmServe.** Lou et al. ICML 2026 per arXiv. arXiv:[2512.09472](https://arxiv.org/abs/2512.09472).
  - One-for-many GPU prewarming of parameters from workload forecasts. Tail TTFT up to 50.8× lower.
  - **Abstract only;** whether it prewarms graphs is UNVERIFIED.
- **Tangram.** Wenbin Zhu, Zhaoyan Shen, Hongjun Dai, Zili Shao, Feng Chen. arXiv:[2512.01357](https://arxiv.org/abs/2512.01357) (1 Dec 2025). **Read** (HTML).
  - Reuses tensors left in GPU memory and allocates the KV cache on demand. 1.8–6.2× faster loading.
  - Its strongest baseline is SLLM + CRIU + Medusa. No mention of CUDA graphs.
- **TIDAL.** Weihao Cui, Ziyi Xu, Han Zhao, Quan Chen, Zijun Li, Bingsheng He, Minyi Guo. arXiv:[2503.06421](https://arxiv.org/abs/2503.06421) (9 Mar 2025). **Read** (HTML).
  - Traces execution to build function templates. Proactive kernel-module loading counters the CUDA runtime's lazy module loading: a cold kernel call costs about 180 ms. Adaptive state forking is the second technique.
  - Cold start 1.79–2.11× lower. PyTorch 2.4. No CUDA graphs.
- **Aegaeon.** Yuxing Xiang, Xue Li, Kun Qian, Yufan Yang, Diwen Zhu, Wenyuan Yu, Ennan Zhai, Xuanzhe Liu, Xin Jin, Jingren Zhou. SOSP '25, DOI 10.1145/3731569.3764815. **Read** (PDF, relevant sections).
  - Token-level auto-scaling. Reuses initialized engine components and caches profiling: unoptimized init is up to 26.9 s for a 13B model, and component reuse removes over 80% of scale-up latency.
  - 97% lower overhead overall. No CUDA-graph mention found.
- **SpotServe.** Xupeng Miao et al. arXiv:[2311.15566](https://arxiv.org/abs/2311.15566). The venue is ASPLOS '24 (UNVERIFIED, not on the arXiv PDF header I read).
  - Re-initialization cost of preempted instances (e.g., about 2 minutes to launch and initialize an engine) is handled by context migration. No graph discussion found by grep.
- **C2CServe.** Shutian Luo, Ali Zafar Sadiq, Rui Yang, Mingye Zhang, Haiying Shen, Wei Wang, Yue Cheng. arXiv:[2605.19481](https://arxiv.org/abs/2605.19481) (19 May 2026).
  - GH200 NVLink-C2C plus MIG: weights stay in host memory and are streamed. Cold start up to 7.1× lower (dense) and 4.6× (MoE).
  - **Abstract only;** graph handling is UNVERIFIED. Relevant because it also targets GH200.
- **InstaInfer.** Y. Sui, H. Yu, Y. Hu, J. Li, H. Wang, *Pre-warming is not enough: Accelerating serverless inference with opportunistic pre-loading*, ACM SoCC 2024, per HydraServe's bibliography. **Not read: UNVERIFIED.**

### 2.5 Compilation caching as a cold-start tool

- **PyTorch Mega-Cache.** `torch.compiler.save_cache_artifacts()` and `load_cache_artifacts()` ([tutorial](https://docs.pytorch.org/tutorials/recipes/torch_compile_caching_tutorial.html), **read**).
  - Portable bundle of the PGO, AOTAutograd, Inductor, Triton and autotuning caches.
  - Requires the same PyTorch and Triton versions and the same GPU.
  - **Does not mention CUDA graphs.**
  - A **Beta** feature in PyTorch 2.7 ([release blog](https://pytorch.org/blog/pytorch-2-7/), **read**).
- **AOTInductor** ([PyTorch 2.5 docs](https://docs.pytorch.org/docs/2.5/torch.compiler_aot_inductor.html), **read**).
  - Compiles an exported model into a shared library run from C++. Dynamic dimensions such as batch are supported without recompiling.
  - Prototype status. The page does not mention CUDA graphs.
- **Unsourced claim to discount.** A Spheron blog ([link](https://www.spheron.network/blog/gpu-cold-start-llm-inference-2026/), read) claims vLLM reuses CUDA graphs across restarts via a `CUDA_GRAPHS_CACHE_DIR` environment variable.
  - It cites no source.
  - It conflicts with vLLM #19824 and with Foundry (Apr 2026), which treat graph serialization as an open research problem.
  - Treat it as unreliable.
- **Effect in vLLM:**
  - With warm torch.compile caches, vLLM still spends about 8 s (7.6 s in Fig. 1) on Dynamo bytecode transformation plus cache loading (Foundry).
  - The cache cut compile from 42 to 13 s while Dynamo stayed at 10 s (Tensorfuse).
  - Uncached Inductor took about 70 s for a 70B model at TP8 (vLLM #19824).
- **Implication:** caches remove compilation but **not** capture. No public compile cache stores CUDA graphs.

---

## 3. Conceptual precedent: tiered and background compilation

The pattern is to run cold code in a slow tier, compile hot code in the background, then install it. That is exactly our eager → captured-graph promotion. These citations are the canonical anchors.

1. **HotSpot (interpret first, compile hot methods, then install).**
   - Michael Paleczny, Christopher Vick, Cliff Click (Sun Microsystems). *The Java HotSpot™ Server Compiler.* Java Virtual Machine Research and Technology Symposium (JVM '01), USENIX, Monterey, Apr 23–24 2001. PDF: https://usenix.org/event/jvm01/full_papers/paleczny/paleczny.pdf.
     - **Read** (PDF text).
     - Methods start in the interpreter, which keeps method-entry and backward-branch counters.
     - Past a `CompileThreshold`, the runtime compiles the method and registers the code with the method, so later invocations jump to it. On-stack replacement handles hot loops. Deoptimization falls back to the interpreter.
   - The tiered form is documented by Oracle in *Java HotSpot™ Virtual Machine Performance Enhancements*, section "Tiered Compilation": https://docs.oracle.com/javase/8/docs/technotes/guides/vm/performance-enhancements-7.html.
     - **Read.** Introduced in Java SE 7 and the default for the server VM.
     - Tiers: interpreter → client compiler (profiled code) → server compiler. The goal is client-like startup together with server-level peak performance.
   - HotSpot's `-XX:+BackgroundCompilation` flag semantics are **UNVERIFIED**: I found no primary documentation, and the OpenJDK bug tracker returned 403.
2. **Cost-benefit ordering of recompilation in a background thread (the closest analogue to our EV ordering).**
   - Matthew Arnold, Stephen Fink, David Grove, Michael Hind, Peter F. Sweeney. *Adaptive Optimization in the Jalapeño JVM.* OOPSLA 2000, published as ACM SIGPLAN Notices 35(10), pp. 47–65, Oct 2000. DOI [10.1145/354222.353175](https://doi.org/10.1145/354222.353175).
     - Bibliographic data and abstract **read** on the ACM page. The abstract describes a federation of threads with asynchronous communication, and multi-level optimization driven by sampling.
     - The PDF body text is not machine-readable. Only figure labels such as "Compilation Thread" and "Install New Code" extracted, so the paper's exact formulas are **UNVERIFIED** from the paper itself.
   - The **Jikes RVM User Guide, "Adaptive Optimization System"** (https://www.jikesrvm.org/UserGuide/AdaptiveOptimizationSystem/index.html, **read**) cites that OOPSLA 2000 paper as the original design. It documents the model:
     - Recompile when (compile cost + expected future time at the new level) < expected future time at the current level.
     - Future time is estimated as equal to the time spent so far, from samples. Compile cost is linear in method size, and per-level speedups are offline constants.
     - The **recompilation queue is ordered by expected improvement.**
     - By default a dedicated thread recompiles **asynchronously while the method keeps running at its current tier**.
   - Mapping to our idea: frequency ≈ samples, speedup ≈ per-level speedup, capture cost ≈ compile cost, and the queue is ordered by expected benefit. Our version adds the constraint that "compilation" (capture) must run while the device is idle and in lockstep across ranks.
3. **V8 background compilation.**
   - Ross McIlroy. *Background compilation.* V8 blog, 26 Mar 2018: https://v8.dev/blog/background-compilation. **Read.**
   - From Chrome 66, most bytecode compilation of top-level scripts runs on a background thread. Only a short finalization step stays on the main thread.
   - Most pages saw a 5–20% cut in main-thread compilation time.
   - Inner functions are still compiled lazily on the main thread at first call. That is the same split as ours: a lazy, first-use path versus background work.

---

## 4. Synthesis for the write-up

### 4.1 Closest works, and what is genuinely new in ours

1. **vLLM RFC #20098 and PR #23184.** They are the same idea family: lazy or delayed capture, partial initial sets, and idle-time capture *suggested*.
   - The PR's policy is *current shape first, else the largest uncaptured*. It captures inside the forward step, which is implicitly lockstep across ranks.
   - It was never merged. Idle-time capture was never implemented.
2. **Foundry.** It removes capture with an offline artifact, an unprivileged LD_PRELOAD hook, and lazy `cuGraphExecUpdate` specialization.
3. **Medusa.** Offline materialization; it rejects deferral.
4. **LMDeploy and PyTorch CUDA Graph Trees.** Lazy capture on first use, on the request path.
5. **TRT-LLM batch-size tuning blog.** Workload-driven size selection is only future work there.

**New in ours** relative to all of the above:
- (a) Capture happens **only in scheduler-idle gaps**, one graph at a time, with a bounded worst-case stall. Nobody has implemented or evaluated this publicly.
- (b) An **explicit TP-lockstep protocol** for runtime capture of graphs with collectives, and a correctness argument for mixing eager and graph paths across ranks. PR #23184 drew an unreproduced multi-TP garbled-output report, and RFC #34303 flags NCCL-in-graph problems.
- (c) **EV ordering** by frequency × speedup ÷ cost, instead of first-seen or largest-first order or static lists. This is new for CUDA graphs, but **not conceptually new**: Jalapeño and Jikes RVM order background recompilation by expected improvement net of compile cost (§3). Cite it as the precedent.
- (d) A **traffic-chosen initial set**. NVIDIA lists log-driven size selection as future work.
- (e) **Frequency-based promotion** of hot eager sizes. This is only partly new: PR #23184 already prioritizes the current shape on first sight.
- (f) **Zero offline step, privilege or driver feature.** This fits Alps, unlike the C/R family and unlike Medusa and Foundry's offline steps.
- (g) **SGLang.** All prior engine work is vLLM, LMDeploy or TRT-LLM.

Lazy capture *per se* is **not** new.

### 4.2 Sources that contradict or weaken our premises

- **Medusa §2.4:** deferral only postpones capture and spreads it across requests. We must show that idle-only capture keeps capture off requests, and quantify the residual stall when a request arrives mid-capture.
- **vLLM maintainers on RFC #20098 and PR #23184:**
  - reliability (Kwon);
  - TTFT unpredictability (zou3519, ProExpertProg);
  - OOM risk, because graph memory is outside profiling (yeqcharlotte, LucasWilkinson);
  - maintenance burden (youkaichao, strongly opposed);
  - capture is now fast enough (ProExpertProg; about 50 ms per graph for 8B).
- **Breaking the Ice:** in vLLM 0.10.1.1 on a single H100, capture is only about 1–2 s of a roughly 20 s start. The 35–47% share holds only for larger, TP or MoE, many-size or SGLang settings.
  - Evidence that it does hold there: Foundry (59% of worker init), vLLM #19824 (~60 s at TP8 70B), InstantInfer (SGLang Optimize 56 s of 141 s), TRT-LLM (10–26 s). The write-up should state the regime.
- **Foundry needs no privileges.** If an offline SAVE plus LD_PRELOAD is acceptable on Alps, it would beat us on time-to-full-performance. Our case rests on *no offline step or version-locked artifact* and on composability.
- **HydraServe** claims graph and KV init overhead is largely eliminated by prior work. That holds only with Medusa's offline step.
- **LMDeploy** already captures lazily on the request path in v0.7.0, so *engines capture everything at startup* is not universal.

### 4.3 Risks others raised that the design must answer

- Graph-pool memory must be accounted for before KV sizing. Medusa, PR #23184 (capture the largest graph first) and the RFC #20098 OOM concern all point at this.
- Capture must be strictly serialized: Medusa notes CUDA errors with concurrent capture, and Foundry notes driver contention.
- Replay and eager execution may take different all-reduce paths, so they must not mix across TP ranks within one forward.
  - This rationale comes from our local prototype's docstring, not a published source.
  - The closest published evidence is the unreproduced multi-TP first-request garbage report on vLLM PR #23184, and RFC #34303's note that graphs containing NCCL collectives may need re-capture.
- First-use JIT and module loading inside the capture region:
  - SGLang RFC #32432 says to finish lazy init, module loading and JIT before capture, and to rendezvous ranks at the capture boundary.
  - vLLM PR #46523 does an eager pre-run before capture.
  - TIDAL measures the cost of lazy module loading; Foundry notes that kernels are loaded lazily.
- Graph coverage value is model-dependent: large for MoE, hybrid and MTP models, near zero or negative for dense models (vLLM #62556). Use a measured or estimated speedup per size, not a constant.

---

## 5. Verification log and unread leads

- **Method.**
  - ACM: the ACM DL ePDF reader in the built-in browser, with text extracted through the page's pdf.js. One attempt to open the raw PDF URL triggered a download dialog, which I did not complete.
  - USENIX and arXiv PDFs: `pdftotext` on files that WebFetch cached in the session's tool-results directory.
  - arXiv HTML and GitHub issue and PR pages: WebFetch, with targeted re-reads for every quoted sentence.
  - LMDeploy source: raw.githubusercontent.com. The GitHub API was not used.
- **Local note.** The vendored tree `vendor/sglang-v0.5.20/.../model_executor/runner/lazy_capture.py` has a newer mtime (Nov 11 20:23) than the rest of the vendor (20:07). Upstream v0.5.20's directory listing does not show this file, so it is local work, not prior art.
- **Unread leads (UNVERIFIED; titles from search results only):**
  - vLLM #16716 "V1 engine CUDA Graph Capturing time is too long"
  - vLLM PR #21146 "Freeze gc during cuda graph capture to speed up init". Possibly the optimization behind the ~50 ms per graph cited in RFC #20098; that link is my inference.
  - vLLM PR #51360 (same-host TP=1 snapshot)
  - SGLang #34297 (follow-up to RFC #32432)
  - SGLang Fast Engine Recovery roadmap #33522
  - Cerebrium snapshot blog
  - `vllm bench startup` docs
- **Could not open:**
  - The AOTInductor page for the current docs version: the redirect target rendered empty, so I used the 2.5 page.
  - The OpenJDK bug JDK-6217807 (HTTP 403).
  - The Jalapeño PDF body text: the text layer is not machine-readable.
