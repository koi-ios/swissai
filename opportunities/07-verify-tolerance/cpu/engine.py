"""A CPU stand-in for a BF16 serving engine, built on a transformers model.

`servekit verify`'s statistics must be validated against real floating-point
nondeterminism, not synthetic noise, and there is no GPU here. This engine runs
a Hugging Face decoder (Qwen3, or Llama-style without q/k norms) in fp32 with
BF16 rounding at every kernel boundary -- which is what a BF16 GPU engine does
with fp32 accumulation -- and exposes the knobs that make a real engine's
numerics depend on its load:

  split_k      each GEMM's K dim summed as s fp32 partials (cuBLAS/CUTLASS
               choose split-K by problem size, i.e. by batch size)
  attn_splits  attention keys split into s partitions merged by log-sum-exp
               (split-KV / flash-decoding, also chosen by batch size)
  tp           row-parallel layers (o_proj, down_proj) as tp per-rank GEMMs
               whose BF16 outputs are all-reduced in BF16
  chunk        chunked prefill: the prompt computed `chunk` tokens at a time
               against the KV cache
  prefix       a prefix-cache hit: the first `prefix` tokens come from an
               earlier pass

Weights are read from safetensors files by `load_checkpoint`, which
dequantizes FP8 block-scaled (`weight_scale_inv`, 128x128 blocks, activations
quantized per 1x128 group at run time) and MXFP4 (`*_blocks` + E8M0 `*_scales`,
32-element blocks) tensors the way a loader does, so a corrupted file reaches
the arithmetic exactly as it would on a GPU. Missing tensors load as zeros, as
in a loader that does not check (sglang#34448 was such a case).

The model config, tokenizer and rotary embedding come from transformers, and
with rounding off the forward pass matches transformers' own to fp32 precision
(`python -m cpu.engine --check`).
"""
from __future__ import annotations

import json
import sys
import math
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import torch
from safetensors.torch import load_file

FP8_MAX = 448.0
FP4_VALUES = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0])
ROW_PARALLEL = ("o_proj", "down_proj")


def bf16(x: torch.Tensor) -> torch.Tensor:
    return x.to(torch.bfloat16).to(torch.float32)


def _ident(x: torch.Tensor) -> torch.Tensor:
    return x


@dataclass(frozen=True)
class Numerics:
    split_k: int = 1
    attn_splits: int = 1
    tp: int = 1
    chunk: Optional[int] = None
    prefix: Optional[int] = None
    round_bf16: bool = True

    def to_dict(self) -> dict:
        return asdict(self)


def _bounds(n: int, parts: int, align: int = 1) -> List[Tuple[int, int]]:
    """Split [0, n) into `parts` contiguous pieces, cut at multiples of `align`."""
    if parts <= 1:
        return [(0, n)]
    cuts = [0]
    for i in range(1, parts):
        c = int(round(n * i / parts / align)) * align
        cuts.append(min(max(c, cuts[-1]), n))
    cuts.append(n)
    return [(a, b) for a, b in zip(cuts, cuts[1:]) if b > a]


# --------------------------------------------------------------------------
# loading


def _dequant_fp8(w: torch.Tensor, scale_inv: torch.Tensor, block: Sequence[int]) -> torch.Tensor:
    bo, bi = block
    s = scale_inv.float().repeat_interleave(bo, 0).repeat_interleave(bi, 1)
    return w.float() * s[: w.shape[0], : w.shape[1]]


def _dequant_mxfp4(blocks: torch.Tensor, scales: torch.Tensor) -> torch.Tensor:
    """[out, n_blocks, 16] uint8 (two E2M1 per byte, low nibble first) + [out, n_blocks] E8M0 -> [out, n_blocks*32]."""
    lo = (blocks & 0x0F).long()
    hi = (blocks >> 4).long()
    vals = torch.stack([FP4_VALUES[lo], FP4_VALUES[hi]], dim=-1).reshape(*blocks.shape[:-1], 32)
    exp = scales.to(torch.int32) - 127
    w = vals * torch.pow(2.0, exp.float()).unsqueeze(-1)
    return w.reshape(blocks.shape[0], -1)


@dataclass
class Checkpoint:
    config: object
    weights: Dict[str, torch.Tensor]  # fp32, dequantized
    fp8: set = field(default_factory=set)  # linear names whose activations are FP8-quantized
    missing: List[str] = field(default_factory=list)
    source: str = ""


def expected_names(config) -> List[str]:
    names = ["model.embed_tokens.weight", "model.norm.weight"]
    if not getattr(config, "tie_word_embeddings", False):
        names.append("lm_head.weight")
    for i in range(config.num_hidden_layers):
        p = f"model.layers.{i}."
        names += [p + "input_layernorm.weight", p + "post_attention_layernorm.weight"]
        names += [p + f"self_attn.{n}.weight" for n in ("q_proj", "k_proj", "v_proj", "o_proj")]
        names += [p + f"mlp.{n}.weight" for n in ("gate_proj", "up_proj", "down_proj")]
        if config.model_type == "qwen3":
            names += [p + "self_attn.q_norm.weight", p + "self_attn.k_norm.weight"]
    return names


def from_tensors(config, raw: Dict[str, torch.Tensor], source: str = "") -> Checkpoint:
    """Dequantize a raw state dict (as read from safetensors) into fp32 weights; absent ones are zeros."""
    weights, fp8 = dequantize(config, raw)
    if getattr(config, "tie_word_embeddings", False) and "model.embed_tokens.weight" in weights:
        weights["lm_head.weight"] = weights["model.embed_tokens.weight"]
    missing = [name for name in expected_names(config) if name not in weights]
    for name in missing:
        weights[name] = torch.zeros(_shape_of(config, name))
        if name == "model.embed_tokens.weight" and getattr(config, "tie_word_embeddings", False):
            weights["lm_head.weight"] = weights[name]
    return Checkpoint(config=config, weights=weights, fp8=fp8, missing=missing, source=source)


def dequantize(config, raw: Dict[str, torch.Tensor]):
    """(fp32 weights, names of FP8 linears) for the tensors in `raw`; a missing scale reads as zeros."""
    qcfg = getattr(config, "quantization_config", None) or {}
    if not isinstance(qcfg, dict):
        qcfg = qcfg.to_dict()
    block = qcfg.get("weight_block_size") or [128, 128]
    weights, fp8 = {}, set()
    for name, t in raw.items():
        if name.endswith("_scale_inv") or name.endswith("_scales"):
            continue
        if name.endswith(".weight_blocks"):
            base = name[: -len("_blocks")]
            scales = raw.get(base + "_scales")
            if scales is None:  # a missing scale tensor reads back as zeros
                scales = torch.zeros(t.shape[:-1], dtype=torch.uint8)
            weights[base] = _dequant_mxfp4(t, scales)
            continue
        if t.dtype == torch.float8_e4m3fn:
            scale = raw.get(name + "_scale_inv")
            if scale is None:
                scale = torch.zeros(math.ceil(t.shape[0] / block[0]), math.ceil(t.shape[1] / block[1]))
            weights[name] = _dequant_fp8(t, scale, block)
            fp8.add(name)
            continue
        weights[name] = t.float()
    return weights, fp8


def _shape_of(config, name: str) -> Tuple[int, ...]:
    h, inter = config.hidden_size, config.intermediate_size
    hd = getattr(config, "head_dim", None) or h // config.num_attention_heads
    nq, nkv = config.num_attention_heads * hd, config.num_key_value_heads * hd
    table = {
        "embed_tokens.weight": (config.vocab_size, h),
        "lm_head.weight": (config.vocab_size, h),
        "norm.weight": (h,),
        "layernorm.weight": (h,),
        "q_norm.weight": (hd,),
        "k_norm.weight": (hd,),
        "q_proj.weight": (nq, h),
        "k_proj.weight": (nkv, h),
        "v_proj.weight": (nkv, h),
        "o_proj.weight": (h, nq),
        "gate_proj.weight": (inter, h),
        "up_proj.weight": (inter, h),
        "down_proj.weight": (h, inter),
    }
    for suffix, shape in table.items():
        if name.endswith(suffix):
            return shape
    raise KeyError(name)


def read_raw(model_dir: Path) -> Dict[str, torch.Tensor]:
    raw = {}
    for f in sorted(Path(model_dir).glob("*.safetensors")):
        raw.update(load_file(str(f)))
    return raw


def load_checkpoint(model_dir: Path) -> Checkpoint:
    from transformers import AutoConfig

    config = AutoConfig.from_pretrained(str(model_dir))
    return from_tensors(config, read_raw(model_dir), source=str(model_dir))


# --------------------------------------------------------------------------
# the forward pass


class Engine:
    """Teacher-forced scoring and greedy decoding under a given `Numerics`."""

    def __init__(self, ckpt: Checkpoint, threads: Optional[int] = None):
        from transformers.models.qwen3.modeling_qwen3 import Qwen3RotaryEmbedding

        if threads:
            torch.set_num_threads(threads)
        self.ckpt = ckpt
        c = ckpt.config
        self.config = c
        self.w = ckpt.weights
        self.n_layers = c.num_hidden_layers
        self.n_heads = c.num_attention_heads
        self.n_kv = c.num_key_value_heads
        self.head_dim = getattr(c, "head_dim", None) or c.hidden_size // c.num_attention_heads
        self.eps = c.rms_norm_eps
        self.qk_norm = "model.layers.0.self_attn.q_norm.weight" in self.w
        self.rotary = Qwen3RotaryEmbedding(config=c)

    # ---- kernels

    def _r(self, num: Numerics):
        return bf16 if num.round_bf16 else _ident

    def _rmsnorm(self, x, weight, num):
        r = self._r(num)
        var = x.pow(2).mean(-1, keepdim=True)
        return r(weight * r(x * torch.rsqrt(var + self.eps)))

    def _gemm(self, x, w, split_k):
        if split_k <= 1:
            return x @ w.T
        out = None
        for a, b in _bounds(w.shape[1], split_k, align=16):
            p = x[:, a:b] @ w[:, a:b].T
            out = p if out is None else out + p
        return out

    def _quant_act_fp8(self, x):
        t, k = x.shape
        pad = (-k) % 128
        xp = torch.nn.functional.pad(x, (0, pad)) if pad else x
        g = xp.reshape(t, -1, 128)
        s = (g.abs().amax(-1, keepdim=True) / FP8_MAX).clamp_min(1e-12)
        q = (g / s).clamp(-FP8_MAX, FP8_MAX).to(torch.float8_e4m3fn).float() * s
        return q.reshape(t, -1)[:, :k]

    def _linear(self, x, name, num):
        r = self._r(num)
        w = self.w[name]
        if name in self.ckpt.fp8:
            x = self._quant_act_fp8(x)
        if num.tp > 1 and any(p in name for p in ROW_PARALLEL):
            out = None
            for a, b in _bounds(w.shape[1], num.tp, align=16):
                part = r(self._gemm(x[:, a:b], w[:, a:b], num.split_k))
                out = part if out is None else r(out + part)
            return out
        return r(self._gemm(x, w, num.split_k))

    def _rope(self, x, cos, sin, num):
        r = self._r(num)
        half = x.shape[-1] // 2
        rot = torch.cat((-x[..., half:], x[..., :half]), dim=-1)
        return r(x * r(cos) + rot * r(sin))

    def _attention(self, q, k, v, q_start, num):
        """q [H, Tq, D] at positions q_start..; k, v [Hkv, Tk, D] at positions 0..Tk-1.

        Keys are cut into segments, each attended separately into a BF16 partial
        output plus its log-sum-exp, then merged -- what split-KV decoding does
        (`attn_splits`), and what SGLang's FlashInfer extend path does at the
        boundary between cached prefix and new tokens (ragged + paged, then
        merge_state), which is how chunked prefill and prefix hits move numbers.
        """
        r = self._r(num)
        rep = self.n_heads // self.n_kv
        k = k.repeat_interleave(rep, 0)
        v = v.repeat_interleave(rep, 0)
        tq, tk = q.shape[1], k.shape[1]
        scale = self.head_dim ** -0.5
        s = (q @ k.transpose(1, 2)) * scale
        qpos = torch.arange(q_start, q_start + tq).unsqueeze(1)
        kpos = torch.arange(tk).unsqueeze(0)
        s = s.masked_fill(kpos > qpos, float("-inf"))
        cuts = {0, tk}
        if num.attn_splits > 1:
            cuts.update(b for _, b in _bounds(tk, num.attn_splits, align=16))
        if q_start > 0 and tq > 1:  # extend over a cached prefix
            cuts.add(q_start)
        cuts = sorted(cuts)
        if len(cuts) == 2:
            p = torch.softmax(s, dim=-1)
            return r(r(p) @ v)
        lses, outs = [], []
        for a, b in zip(cuts, cuts[1:]):
            sp = s[..., a:b]
            m = sp.amax(-1, keepdim=True)
            m = torch.where(torch.isinf(m), torch.zeros_like(m), m)
            e = torch.exp(sp - m)
            l = e.sum(-1, keepdim=True)
            o = (r(e) @ v[:, a:b]) / l.clamp_min(1e-30)
            outs.append(r(o))
            lses.append(torch.where(l > 0, m + torch.log(l.clamp_min(1e-30)), torch.full_like(l, float("-inf"))))
        lse = torch.logsumexp(torch.stack(lses), dim=0)
        merged = sum(torch.exp(x - lse) * o for x, o in zip(lses, outs))
        return r(merged)

    def _layer(self, i, h, cos, sin, cache, q_start, num):
        r = self._r(num)
        p = f"model.layers.{i}."
        t = h.shape[0]
        x = self._rmsnorm(h, self.w[p + "input_layernorm.weight"], num)
        q = self._linear(x, p + "self_attn.q_proj.weight", num).view(t, self.n_heads, self.head_dim)
        k = self._linear(x, p + "self_attn.k_proj.weight", num).view(t, self.n_kv, self.head_dim)
        v = self._linear(x, p + "self_attn.v_proj.weight", num).view(t, self.n_kv, self.head_dim)
        if self.qk_norm:
            q = self._rmsnorm(q, self.w[p + "self_attn.q_norm.weight"], num)
            k = self._rmsnorm(k, self.w[p + "self_attn.k_norm.weight"], num)
        q = self._rope(q.transpose(0, 1), cos, sin, num)
        k = self._rope(k.transpose(0, 1), cos, sin, num)
        v = v.transpose(0, 1)
        if cache[i] is not None:
            k = torch.cat([cache[i][0], k], dim=1)
            v = torch.cat([cache[i][1], v], dim=1)
        cache[i] = (k, v)
        a = self._attention(q, k, v, q_start, num).transpose(0, 1).reshape(t, -1)
        h = r(h + self._linear(a, p + "self_attn.o_proj.weight", num))
        x = self._rmsnorm(h, self.w[p + "post_attention_layernorm.weight"], num)
        g = self._linear(x, p + "mlp.gate_proj.weight", num)
        u = self._linear(x, p + "mlp.up_proj.weight", num)
        m = r(r(torch.nn.functional.silu(g)) * u)
        return r(h + self._linear(m, p + "mlp.down_proj.weight", num))

    def _forward(self, ids: torch.Tensor, cache, start: int, num: Numerics) -> torch.Tensor:
        """Logits (fp32, from BF16 lm_head output) for `ids` at positions start..start+len-1."""
        h = self.w["model.embed_tokens.weight"][ids]
        pos = torch.arange(start, start + ids.shape[0]).unsqueeze(0)
        cos, sin = self.rotary(h.unsqueeze(0), pos)
        cos, sin = cos[0].unsqueeze(0), sin[0].unsqueeze(0)
        for i in range(self.n_layers):
            h = self._layer(i, h, cos, sin, cache, start, num)
        x = self._rmsnorm(h, self.w["model.norm.weight"], num)
        return self._linear(x, "lm_head.weight", num)

    def _pieces(self, n: int, num: Numerics) -> List[Tuple[int, int]]:
        cuts = {0, n}
        if num.prefix and 0 < num.prefix < n:
            cuts.add(num.prefix)
        if num.chunk:
            cuts.update(range(num.chunk, n, num.chunk))
        cuts = sorted(cuts)
        return list(zip(cuts, cuts[1:]))

    @torch.no_grad()
    def prefill(self, ids: Sequence[int], num: Numerics):
        ids = torch.tensor(list(ids), dtype=torch.long)
        cache = [None] * self.n_layers
        logits = []
        for a, b in self._pieces(len(ids), num):
            logits.append(self._forward(ids[a:b], cache, a, num))
        return torch.cat(logits, dim=0), cache

    @torch.no_grad()
    def score(self, ids: Sequence[int], num: Numerics, top_k: int = 2):
        """Logprob of each token given its prefix (positions 1..n-1), plus top-k (ids, logprobs) there."""
        logits, _ = self.prefill(ids, num)
        lp = torch.log_softmax(logits[:-1].float(), dim=-1)
        target = torch.tensor(list(ids[1:]), dtype=torch.long)
        tok_lp = lp.gather(1, target.unsqueeze(1)).squeeze(1)
        tv, ti = lp.topk(top_k, dim=-1)
        return tok_lp.tolist(), ti.tolist(), tv.tolist(), lp

    @torch.no_grad()
    def greedy(self, ids: Sequence[int], n_tokens: int, num: Numerics, top_k: int = 2):
        logits, cache = self.prefill(ids, num)
        pos = len(ids)
        last = logits[-1]
        out, tops = [], []
        for _ in range(n_tokens):
            lp = torch.log_softmax(last.float(), dim=-1)
            tv, ti = lp.topk(top_k)
            nxt = int(ti[0])
            out.append(nxt)
            tops.append((ti.tolist(), tv.tolist()))
            last = self._forward(torch.tensor([nxt]), cache, pos, num)[-1]
            pos += 1
        return out, tops


# --------------------------------------------------------------------------
# captures in servekit verify's format


def capture(engine: Engine, tokenizer, prompts: List[Tuple[str, str]], num_for, *, prompt_set="builtin-v1",
            greedy_prompts: int = 8, greedy_tokens: int = 32, top_k: int = 2, model: str = "cpu",
            full_lp: bool = False) -> dict:
    """A capture dict as `servekit verify` writes it (format 2).

    `num_for(i, key)` gives the Numerics for prompt i, so a run can vary them per request
    as a loaded server would. `full_lp` also keeps each prompt's full log-softmax
    (as `_lp`, not JSON) for exact KL measurements.
    """
    tok = lambda i: tokenizer.decode([i])  # noqa: E731
    out = []
    for i, (key, text) in enumerate(prompts):
        ids = tokenizer(text).input_ids
        num = num_for(i, key)
        lps, ti, tv, full = engine.score(ids, num, top_k)
        entry = {
            "key": key,
            "text": text,
            "n_tokens": len(ids) - 1,
            "tokens": [tok(t) for t in ids[1:]],
            "token_logprobs": lps,
            "mean_nll": -sum(lps) / len(lps),
            "top_logprobs": [[[tok(a), b] for a, b in zip(ia, va)] for ia, va in zip(ti, tv)],
            "numerics": num.to_dict(),
        }
        if full_lp:
            entry["_lp"] = full
        if i < greedy_prompts:
            g, tops = engine.greedy(ids, greedy_tokens, num, top_k)
            entry["greedy_tokens"] = [tok(t) for t in g]
            entry["greedy_top_logprobs"] = [[[tok(a), b] for a, b in zip(ia, va)] for ia, va in tops]
        out.append(entry)
    return {
        "format": 2,
        "model": model,
        "prompt_set": prompt_set,
        "greedy_prompts": greedy_prompts,
        "greedy_tokens": greedy_tokens,
        "top_logprobs": top_k,
        "prompts": out,
    }


def _check(model_dir: str) -> None:
    """With rounding off, this forward must match transformers' fp32 forward."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokzr = AutoTokenizer.from_pretrained(model_dir)
    hf = AutoModelForCausalLM.from_pretrained(model_dir, dtype=torch.float32, attn_implementation="eager").eval()
    eng = Engine(load_checkpoint(Path(model_dir)))
    ids = tokzr("The capital of France is Paris, and the capital of Germany is").input_ids
    with torch.no_grad():
        ref = torch.log_softmax(hf(torch.tensor([ids])).logits[0].float(), -1)
    for num in (Numerics(round_bf16=False), Numerics(round_bf16=False, chunk=4, attn_splits=3, split_k=2, tp=2)):
        logits, _ = eng.prefill(ids, num)
        got = torch.log_softmax(logits.float(), -1)
        print(f"{num}: max |dlogprob| vs transformers fp32 = {(got - ref).abs().max().item():.3e}")
    t0 = time.time()
    logits, _ = eng.prefill(ids, Numerics())
    got = torch.log_softmax(logits.float(), -1)
    print(f"bf16 emulation: max |dlogprob| vs fp32 = {(got - ref).abs().max().item():.3e} ({time.time() - t0:.2f}s)")


if __name__ == "__main__":
    import sys

    _check(sys.argv[1] if len(sys.argv) > 1 else str(Path(__file__).resolve().parent.parent / "models" / "Qwen3-0.6B"))
