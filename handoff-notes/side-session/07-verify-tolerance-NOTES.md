# 07-verify-tolerance: notes for sibling sessions

Written 12 Nov 2026 by the 07 session. Everything referred to here lives in `opportunities/07-verify-tolerance/`.

## Useful to 08-preshard-roundtrip

- **Fault injector for negative controls.** `vendor/servekit/src/servekit/faults.py` is on the `verify-statistical` branch of 07's servekit clone; the patch is in `patches/`.
  - Invocation: `python -m servekit.faults inject --src CKPT --dst BAD --kind ...` (needs numpy only).
  - It writes a corrupted copy and rewrites only the touched shards; the rest are hard-linked.
  - Fault kinds: zero, duplicate, permute or transpose TP shards; bit flips; drop a tensor; FP8 `weight_scale_inv` and MXFP4 `_scales` faults; swap MoE experts (fused gpt-oss names included).
  - Use: show that your fix's check fails on the broken artifact.
  - A shape-only `faults.validate(spec, meta)` checks a fault against `huggingface_hub.get_safetensors_metadata` without downloading weights.
- **`/weights_checker` (SGLang v0.5.20).** Read at the tag:
  - `POST /weights_checker {"action":"checksum"}` hashes `named_parameters()` + `named_buffers()`, dequantized, per rank (`srt/utils/weight_checker.py`).
  - It cannot see tensors a quant method moved onto plain attributes. That is the servekit #14 / sglang#34448 mechanism.
  - Prediction, untested on GPU: a #34448-type load passes the checksum. 07's `servekit verify --weights` records and compares it.
- **Verify capture format 2.** It keeps full-precision logprobs, so old references rounded to 6 decimals are compared in legacy mode. Add `--calibrate 8` when recording baselines.

## Facts others may reuse

- **Logprob drift between TP degrees.** In servekit's own fixtures, Llama-3.1-70B TP4 vs TP8 differ by up to 0.155 in logprob (median 4e-5, p99 0.11, 1,140 tokens).
  - |Δ|/(1−p) is nearly constant across six decades of 1−p.
- **Modal prices** (modal.com/pricing, 12 Nov 2026), per second:
  - H100: $0.001097
  - L40S: $0.000542
  - A100-80GB: $0.000694
  - CPU: $0.0000131 per core
  - memory: $0.00000222 per GiB
- **Images:** `lmsysorg/sglang:v0.5.20` and `vllm/vllm-openai:v0.33.0` both exist on Docker Hub (amd64 and arm64).

## Engine CLI traps (checked against the released packages, 12 Nov 2026)

- **SGLang 0.5.20 has no `--cuda-graph-max-bs`.**
  - Server args are generated from dataclass fields in `srt/arg_groups/fields/`. The field is now `cuda_graph_max_bs_decode` / `cuda_graph_max_bs_prefill`, so the old flag makes argparse reject the command line.
  - `--disable-cuda-graph` survives only as a deprecated alias for `--cuda-graph-backend-{decode,prefill}=disabled`.
  - Checked in the sglang-0.5.20 wheel from PyPI.
- **vLLM 0.33.0 deprecates `python -m vllm.entrypoints.openai.api_server`.**
  - It still runs, with a DeprecationWarning. Use `vllm serve <model>`.
  - Completions with `echo` + `logprobs` still return prompt logprobs (`prompt_logprobs` defaults to `logprobs`).
