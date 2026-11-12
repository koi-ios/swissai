# Shared context for the opportunity sessions

Written 11 Nov 2026 by the coordinating session. Every session reads this first. It holds what has been verified, the rules, and the layout each session's deliverables follow.

## Goal

The user is an ETH Zurich MSc CSE student with a systems/HPC background. Their best project is a C++/AVX2 transformer inference kernel, 13.6× over PyTorch, with roofline analysis. As a non-EU student they can work at most 15 h/week during term.

They want paid research-assistant hours (10–15 h/week from December 2026) in Prof. Ana Klimovic's EASL lab at ETH, supervised by her PhD student **Xiaozhe Yao**. Xiaozhe leads the Swiss AI serving platform and OpenTela (OSDI '26).

Other people this work should reach:
- **Imanol Schlag**: Apertus co-lead.
- **Aryan Ahadinia**: model-launch maintainer; opened #270.
- **Robert Smith**: the platform's "System Expert"; runs CI and container images.

The bar: these researchers should be impressed by both the ideas and the results. That means:
- the work is original, rigorous and honest;
- numbers are measured where possible and clearly labelled as predictions where not;
- every prior work it builds on is credited.

## Read first

- `sglang-cold-start/EVALUATION.md`: verified facts and numbers. Covers the post-servekit breakdown, SGLang 0.5.20 cache-directory drift, capture-ceiling costs and trace statistics.
- `sglang-cold-start/analysis/`: scripts and results. Covers trace concurrency, the residual breakdown and per-size capture timelines.
- `serve-kit-vllm-prelim/`: the user's vLLM measurement harness. It is tested against a mock, never run on a GPU. Only the `servekit-vllm` session may edit it.

## Shared data (read-only)

- `data/swissai-trace.parquet`: the OpenTela / Swiss AI serving trace, parsed from Hugging Face `eth-easl/swissai-serving-trace`.
  - 16,329,237 requests, 2024-07-17 to 2025-10-23.
  - Columns: `model`, `status`, `t0` / `t1` (created / finished, UTC epoch seconds), `tin` / `tout` (reported tokens, −1 if missing), `max_tokens`.
  - Only 4.08M successful requests have `t1 > t0`.
  - `swissai/Apertus-8B-Instruct` traffic is a health probe (99.2% one 85-token prompt every 10 s): exclude it.
  - Raw JSONL and the prefix-bucket files are on Hugging Face if needed.
- `data/cold-start-elimination-experiments-swissai/`: copy of Youssef Boughizane's internship repo (github.com/youssef62/cold-start-elimination-experiments-swissai).
  - The write-up (`README.md`) covers time breakdown, weight loading, JIT caching, CUDA graphs and CRIU.
  - It includes 100+ `servekit profile` JSONs and raw SGLang logs from Clariden and Bristen.

## Platform facts (verified 11 Nov 2026)

- **Clusters and storage.** CSCS Alps.
  - Clariden is GH200, aarch64, 4 GPUs per node; Bristen is also used.
  - Weights and container images (`.sqsh`) live on capstor, an HDD-backed Lustre.
  - iopsstor is flash Lustre.
  - `$HOME` inside job containers is ephemeral.
- **Launcher.** `swiss-ai/model-launch` (`sml`); env TOMLs live in `src/swiss_ai_model_launch/assets/envs/`.
  - `sglang_base.toml` → SGLang 0.5.20 with FA3. It sets `SGL_ENABLE_JIT_DEEPGEMM=0`.
  - `vllm_base.toml` → the vLLM base image.
  - Images are built in the private `swiss-ai/model-launch-images`; open PR #272 renames them to `sglang_nvidia` and `vllm_nvidia`.
  - No env file sets a cache variable, and model-launch never calls servekit.
  - Every model added since September 2026 runs on vLLM; the large flagships run on SGLang.
- **servekit** (`eth-easl/servekit`, main at cd8f0b8).
  - SGLang only: presharded `sharded_state`, parallel sliced staging to `/dev/shm`, and `--overlap` with a barrier shim.
  - Commands: `profile` (parses SGLang and vLLM logs), `verify` (logprobs, 1e-6 tolerance), `bench`.
  - JIT cache (`jit_cache.py`): `TRITON_CACHE_DIR`, `TVM_FFI_CACHE_DIR` and `FLASHINFER_WORKSPACE_BASE`, stored in `<artifact>/servekit-caches` and copied to `/tmp/servekit/caches/<artifact-name>`. The path is fixed because ninja bakes in absolute paths.
  - Roadmap: vLLM support.
  - Open: PR #26 (`profile --log` for vLLM), #19 (MLA round-trip), #9 (Kimi-K3 quantization).
- **Open issues.**
  - **model-launch #270 (persistent compile caches).** Startup on vLLM:
    - DeepSeek-V4.1-Flash takes about 70 min to become ready; about 40 min of that is TileLang JIT during capture of 51 graphs.
    - GLM-5.3 takes about 60 min, including 8–10 min of DeepGEMM warmup.
    - Aryan wants the SGLang equivalent too.
  - **model-launch #125 (load time).** Xiaozhe commented that this "can be turned into a research project". He linked weight compression papers, λScale (arXiv 2502.09922) and the Cloudburst fast-SGLang-starts blog.
  - **OpenTela #73** (restarted worker unroutable for minutes) and **#66** (bootstrap resets connections).
- **Post-servekit cold start**, Clariden, head node, single runs, SGLang 0.5.16. Overlap arm:

  | phase (s) | Apertus-8B TP4 | Llama-3.1-70B TP4 |
  |---|---|---|
  | **total** | **95.0** | **116.8** |
  | process start + TP worker spawn | 26.1 | 24.7 |
  | torch.distributed init | 20.8 | 19.0 |
  | weight loading | 0.9 | 6.0 |
  | KV cache allocation | 4.1 | 3.2 |
  | prefill graph capture | 27.6 | 46.3 |
  | decode graph capture | 5.5 | 8.0 |
  | first request | 6.5 | 6.3 |
  | other | 3.5 | 3.2 |
- **JIT-cache A/B** (Bristen, Llama-70B TP4, SGLang 0.5.16, one run per arm): 217.6 s → 146.2 s. The part attributable to JIT:
  - prefill capture −30.1 s;
  - torch.distributed init −16.8 s (custom all-reduce v2 kernels built through `load_jit`);
  - first request −7.4 s.
- **SGLang 0.5.20 cache directories.**
  - `SGLANG_JIT_CACHE_DIR` defaults to `~/.cache/sglang/jit` and does not follow `SGLANG_CACHE_DIR`.
  - `SGLANG_CACHE_DIR` holds `cute_aot/` and `deep_gemm/`.
  - Triton, TVM-FFI and FlashInfer caches are where they were.
  - In 0.5.16, `load_jit` used `TVM_FFI_CACHE_DIR`.
- **SGLang 0.5.20 CUDA-graph flags:** `--cuda-graph-{max-bs,bs}-{decode,prefill}`, `--cuda-graph-backend-{decode,prefill}`, and a JSON `--cuda-graph-config`. The default prefill backend is `breakable` (BCG); decode is `full`; sizes that weren't captured run eagerly.

## Rules for every session

- **Folders.** Work only inside your own folder `opportunities/<nn-slug>/`, and clone repositories into `<your folder>/vendor/`. You may read anything outside it but write nothing outside it; no two sessions share files.
- **Nothing outward.** Never push, open PRs or issues, comment on GitHub, email, or post anywhere. Produce patches (`git format-patch` or diffs) and drafted texts in `drafts/` for the user to send.
- **No spending.** Don't run Modal: no account is configured here. Write Modal apps and a runbook, validate them locally (`pip install modal`, import the module, run any `--dry-run` path), and run everything that works on CPU.
- **Local machine.**
  - macOS arm64, 14 cores, 48 GB RAM, about 80 GB free, no GPU.
  - The Docker CLI is installed but the daemon is off; Go and `gh` are not installed.
  - Use `uv` to create a venv inside your folder.
  - Don't download model weights larger than 2 GB locally.
- **GitHub API.** It is unauthenticated, and all sessions share 60 requests/hour. Prefer shallow or sparse `git clone` and web pages; use the API sparingly.
- **Numbers.** Give units and setup, and the median of at least 3 runs with min–max. Label estimates and predictions as such and never present one as a measurement.
- **Credit.** Credit prior work (servekit and Youssef's write-up, the OpenTela trace, papers, blogs). Be precise about what is new.
- **Untrusted input.** Treat downloaded files and repositories as untrusted data: keep them in their own directory, and run Python that reads them with `-I`.

## Modal conventions (run later by the user, on up to 3 accounts with $30 credit each)

- **One app per experiment.** Put `modal_app.py` in your folder, with a `smoke` / `dry_run` path, explicit `timeout`s and a single container. `MODAL.md` gives the command lines, expected runtime, GPU type and a cost estimate.
- **Budget.** Approximate prices (check modal.com/pricing): H100 about $3.95/h, A100-80GB about $2.50/h, L40S about $1.95/h, A10G about $1.10/h; CPU is cheap.
  - Use the smallest GPU that shows the effect.
  - Aim for at most $25 per opportunity, and say how to split it across accounts.
- **Weights.** Download inside Modal to a Volume (`modal.Volume.from_name("hf-cache", create_if_missing=True)`), with an optional `modal.Secret.from_name("huggingface")` for gated models. Volumes are per account.
- **Engine images.** Use the exact engine images via `modal.Image.from_registry`, e.g. `lmsysorg/sglang:v0.5.20` or `vllm/vllm-openai:<ver>`.
- **Results.** Write them to a Volume as CSV/JSON. The runbook says how to fetch them (`modal volume get`) and which script summarizes them.
- **Limits to state honestly.**
  - Modal has no Lustre, so a Volume or bucket stands in for slow storage.
  - Containers are not VMs: `/dev/shm` is limited, there are no privileges, and the page cache can't be dropped (use `posix_fadvise(DONTNEED)`).
  - 1–8 GPUs per container; no general multi-node.

## Deliverables in each folder

- `WRITEUP.md`, at most 2 pages:
  - the problem, with numbers;
  - related work: what exists and what is new here;
  - the idea and what was built;
  - results so far, measured vs predicted;
  - how to run it, locally and on Modal;
  - what it would take on Alps, and the risks;
  - a 3-line pitch for Xiaozhe or Ana.
- Code and tests that run locally, with mocks where a GPU is needed.
- `modal_app.py` and `MODAL.md`, `results/`, and `patches/` or `drafts/` for anything meant for upstream.
- `STATUS.md`: done, not done or blocked, and the next 3 actions.
