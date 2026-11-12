# Modal runbook: does presharding round-trip the model?

One experiment per model, SGLang **v0.5.20** (`lmsysorg/sglang:v0.5.20`, the platform's version), **TP=2** on two GPUs in one container. The image carries servekit from `vendor/servekit` on branch `preshard-roundtrip`. That branch holds the raw preshard, the deep-state check, the fixed MLA backport and the guard fix, and installs servekit's SGLang plugin.

| model | why | GPUs (default) | checkpoint |
|---|---|---|---|
| `deepseek-ai/DeepSeek-V2-Lite` | MLA: servekit #19, sglang #35702 | 2x L40S | 31.4 GB BF16 |
| `openai/gpt-oss-20b` | MXFP4 on `triton_kernel`: servekit #9/#14, sglang #34448 | 2x H100 (SM90, like GH200) | 13.8 GB |

TP=2 rather than 1 on purpose: at TP=1 every rank reads every byte, so raw presharding's per-rank slicing would not be exercised.

## What runs (modal_driver.py, in this order)

| step | what | prediction (not yet measured) |
|---|---|---|
| 1 `default` | default loader; `servekit verify --record gold.json`; deep-state reference | - |
| 2 `default_again` | default loader again; verify against gold; second deep-state reference | pass, exact: the noise floor for both checks |
| 3 `sharded_state_sglang` | SGLang's own save (`modal_scripts/sglang_save_sharded.py`, CUDA graphs **on**, so a forward runs before the save) and reload with `--load-format sharded_state`; no servekit | **DSV2-Lite:** load fails, `KeyError: ...attn_mha.kv_b_proj.weight`. **gpt-oss:** loads, verify **fails**, artifact ≈ 31% of the checkpoint (servekit #14 measured 4.30 of 13.76 GB at TP=4), deep-state flags `...quant_method.w13_weight_triton_tensor.storage.data`, and the state_dict-only check passes |
| 4 `raw` | `servekit prepare --preshard-format raw` (records the default load) on **unpatched** SGLang; serve with the ordinary loader, which servekit's plugin replays | pass, exact; the plugin's own read verification passes |
| 5 `sharded_state_servekit` | DSV2-Lite only: `apply_all()` (the fixed #35715 backport), then servekit's sharded_state prepare and reload | pass, exact |
| 6 `timing` | `--runs` starts per correct arm, page cache warmed first; weight-loading seconds (SGLang's `Load weight end. elapsed=`) and process start to ready | default > raw ≥ sharded_state; raw with verification off ≈ sharded_state + post-processing |
| 7 `deepstate` | `servekit deepstate default <arm> --again default_again`, all tensors and registered-only | as above |

Serve runs disable CUDA graphs (`--disable-prefill-cuda-graph --disable-decode-cuda-graph`; 0.5.20 has no `--disable-cuda-graph`) to save time: every arm uses the same flags. Only the first run of each arm dumps the deep state, with `torch.empty` made deterministic, so the timing runs are not slowed by it.

## Commands

```bash
pip install "modal>=1.0"      # validated with modal 1.6.3 (import, function registration)
```

```bash
modal setup
```

Free, local: print every command the experiment would run.

```bash
python modal_driver.py --model dsv2-lite --dry-run
```

Pre-flight on one L4 (about 5–10 min including the first image build). It checks that the CUDA 13 image runs on Modal's driver, that SGLang loads servekit's plugin and its four hooks, that the hook targets exist, that the MLA backport's anchors match, and the gpt-oss backend rule.

```bash
modal run modal_app.py::smoke
```

The two experiments: each downloads into the `hf-cache` Volume (CPU only) and then runs on two GPUs.

```bash
modal run modal_app.py --model dsv2-lite
```

```bash
modal run modal_app.py --model gpt-oss-20b
```

Options: `--runs 3` (timing repeats), `--gpu l40s|h100|a100`, `--steps default,default_again,raw` (a subset), `--skip-download`. If the smoke test shows that the CUDA 13 image does not start on Modal's driver, retry with a CUDA 12 build of the same release, for example `SGLANG_IMAGE=lmsysorg/sglang:v0.5.20-cu129 modal run ...` (check that the tag exists first).

Fetch and summarize (writes `summary.md` next to each `results.json`):

```bash
modal volume get preshard-results dsv2-lite results/modal/
```

```bash
modal volume get preshard-results gpt-oss-20b results/modal/
```

```bash
python summarize.py results/modal
```

Each run directory holds `results.json`, `gold.json`, `verify-*.json`, `deepstate/<arm>/deepstate-target-rank-*.json`, `deepstate-<arm>-{full,registered_only}.json` and every server, prepare and verify log.

## Cost (estimate; check modal.com/pricing)

GPU prices from SHARED_CONTEXT: H100 $3.95/h, L40S $1.95/h, L4 about $0.80/h. Each experiment container also requests 8 CPUs and 96 GiB of memory, about $1.2/h extra at Modal's CPU and memory rates (estimate).

| run | time (estimate) | cost (estimate) |
|---|---|---|
| smoke, 1x L4 | 10 min | $0.15 |
| downloads (CPU) | 10–15 min each | < $0.10 each |
| DSV2-Lite, 2x L40S: 8 prepare/serve starts + 12 timing starts + 5 verify | 60–90 min | $5–8 |
| gpt-oss-20b, 2x H100: 6 + 9 starts + 4 verify | 30–45 min | $4.5–7 |
| **total, one pass** | | **$10–15**, under the $25 budget with room for one retry |

Split across accounts: **account A** runs `smoke` and `dsv2-lite` (about $8), **account B** runs `gpt-oss-20b` (about $7), **account C** is the reserve for a retry or `--runs 5`. Volumes are per account, so each account downloads its own model. Cheaper variant: `--gpu l40s` for gpt-oss (about $4). But L40S is SM89, while the platform's GH200 is SM90, and SGLang picks kernels per architecture.

## Limits, stated honestly

- **No Lustre.** Weights are copied from the Volume to local NVMe and pre-read into the page cache before each timing run. That stands in for servekit's `/dev/shm` staging, so timings compare the *loaders*, not Lustre I/O. The default arm reads the full checkpoint on every rank (mmap), as on the platform.
- The page cache cannot be dropped in a container, so "cold" starts are cold for the engine, not for storage.
- TP=2 on one node only: no PP, no multi-node, no EP. The CPU tests cover EP slicing; the hooks key shards on the world rank and check the recorded parallel layout.
- `servekit verify` uses its default tolerance of 1e-6. If `default_again` itself fails it, the platform's nondeterminism is above the tolerance (the sibling verify-tolerance session's subject). The deep-state comparison is then the arbiter, because it compares bytes, not logprobs.
- The raw arm's first run fingerprints with `fill_uninitialized_memory`, which touches every replay buffer page. Only the timing runs are representative of load time.
