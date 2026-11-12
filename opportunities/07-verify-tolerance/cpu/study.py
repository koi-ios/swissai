"""The CPU validation study: noise floor, false alarms and detection power, end to end.

For each checkpoint variant (BF16 Qwen3-0.6B, its official FP8 block-scaled
release, and an MXFP4-MLP variant made by cpu/mxfp4.py) this captures, in the
format `servekit verify` writes:

  reference   the reference schedule (fixed numerics for every request)
  repeat      the reference schedule again: does it reproduce bitwise?
  request     per-request perturbed numerics (split-K, split-KV, chunked prefill,
              prefix hits) -- seeds 1..16 calibrate, seeds >= 101 are held out
  tp2, tp4    the same, at another tensor-parallel degree than the reference
  fault       a corrupted checkpoint (servekit.faults, on the real safetensors
              tensors), captured under request-level perturbation

and, for every run, the true effect on the model: mean KL(P_ref || P_run) over
all prompt positions with full distributions, and for faults the same under
the reference schedule plus the change in NLL on a held-out passage.

Captures go to results/cpu/<variant>.jsonl.gz, one run per line; `report.py`
fits the noise models and computes every statistic from them, so the decision
rule can change without re-running anything.

    python -I cpu/study.py --variant bf16 --workers 4
"""
from __future__ import annotations

import argparse
import gzip
import json
import math
import os
import random
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "vendor" / "servekit" / "src"))
os.environ.setdefault("HF_HUB_OFFLINE", "1")

import numpy as np  # noqa: E402
import torch  # noqa: E402

from cpu.engine import Checkpoint, Engine, Numerics, capture, dequantize, load_checkpoint  # noqa: E402
from servekit import faults  # noqa: E402

VARIANTS = {
    "bf16": ROOT / "models" / "Qwen3-0.6B",
    "fp8": ROOT / "models" / "Qwen3-0.6B-FP8",
    "mxfp4": ROOT / "models" / "Qwen3-0.6B-MXFP4-mlp",
}
TOKENIZER = ROOT / "models" / "Qwen3-0.6B"
PROMPTS = ROOT / "vendor" / "servekit" / "src" / "servekit" / "_verify" / "prompts.json"
OUT = ROOT / "results" / "cpu"
GREEDY_PROMPTS, GREEDY_TOKENS = 8, 16

# Not in the verify prompt set: what the corrupted model does to text nobody checked.
HELD_OUT = (
    "The committee met on Tuesday to review the budget for the coming year. After a long discussion, "
    "the members agreed to increase funding for the public library and to delay the renovation of the "
    "town hall until the spring. Several residents spoke during the open session, most of them in favour "
    "of extending the bus service to the new housing estate north of the river.\n\n"
    "def merge_sorted(a, b):\n    out, i, j = [], 0, 0\n    while i < len(a) and j < len(b):\n"
    "        if a[i] <= b[j]:\n            out.append(a[i]); i += 1\n        else:\n"
    "            out.append(b[j]); j += 1\n    return out + a[i:] + b[j:]\n\n"
    "Photosynthesis converts light energy into chemical energy. In the light-dependent reactions, water "
    "is split and oxygen is released; the Calvin cycle then fixes carbon dioxide into sugars."
)

PROJ = r"layers\.\d+\.(self_attn\.(q|k|v|o)_proj|mlp\.(gate|up|down)_proj)\.weight$"
FP8_SCALES = r"layers\.\d+\.(self_attn\.(q|k|v|o)_proj|mlp\.(gate|up|down)_proj)\.weight_scale_inv$"
MX_SCALES = r"mlp\.(gate|up|down)_proj\.weight_scales$"
N_LAYERS = 28


def _layer_proj(layer: int, suffix: str = "weight") -> str:
    return rf"layers\.{layer}\.(self_attn\.(q|k|v|o)_proj|mlp\.(gate|up|down)_proj)\.{suffix}$"


def fault_grid(variant: str, seeds=(0, 1, 2)):
    """[(class, magnitude label, spec dict)]: every fault at every magnitude, for each seed."""
    out = []
    for s in seeds:
        layer = random.Random(1000 + s).randrange(N_LAYERS)
        if variant == "bf16":
            for tp in (8, 32, 128):
                out.append(("zero_slice", f"1/{tp} of a tensor", dict(kind="zero_slice", match=PROJ, tp=tp, shard=s % tp, seed=s)))
            out.append(("zero_vocab_shard", "1/8 of the vocab", dict(kind="zero_slice", match=r"embed_tokens\.weight$", tp=8, shard=(3 + s) % 8, seed=s)))
            for tp in (2, 4, 8):
                out.append(("dup_slice", f"tp={tp}", dict(kind="dup_slice", match=PROJ, tp=tp, src_shard=0, dst_shard=1, seed=s)))
            for tp in (2, 4):
                out.append(("permute_shards", f"tp={tp}", dict(kind="permute_shards", match=PROJ, tp=tp, seed=s)))
            out.append(("transpose", "square k/v_proj", dict(kind="transpose", match=r"self_attn\.(k|v)_proj\.weight$", seed=s)))
            for n in (1, 4):
                out.append(("bitflip_exp_msb", f"n={n}", dict(kind="bitflip", match=PROJ, bit="exp_msb", n=n, seed=s)))
            for n in (1, 16, 256, 4096):
                out.append(("bitflip_exp_lsb", f"n={n}", dict(kind="bitflip", match=PROJ, bit="exp_lsb", n=n, seed=s)))
            for n in (16, 256, 4096, 65536):
                out.append(("bitflip_mant_msb", f"n={n}", dict(kind="bitflip", match=PROJ, bit="mant_msb", n=n, seed=s)))
            out.append(("drop_tensor", "a norm weight", dict(kind="drop_tensor", match=r"(layernorm|q_norm|k_norm)\.weight$", seed=s)))
            out.append(("drop_tensor", "a projection", dict(kind="drop_tensor", match=PROJ, seed=s)))
            for sigma in (1e-3, 3e-3, 1e-2, 3e-2, 1e-1):
                out.append(("noise_one_layer", f"sigma={sigma:g}", dict(kind="noise", match=_layer_proj(layer), count=0, sigma=sigma, seed=s)))
            for factor in (1.01, 1.03, 1.1, 1.3):
                out.append(("scale_norm", f"x{factor:g}", dict(kind="scale", match=r"post_attention_layernorm\.weight$", factor=factor, seed=s)))
        elif variant == "fp8":
            out.append(("fp8_scale_roll", "one tensor", dict(kind="scale_blocks", mode="roll", match=FP8_SCALES, seed=s)))
            for sigma in (0.01, 0.03, 0.1, 0.3):
                out.append(("fp8_scale_jitter", f"sigma={sigma:g}", dict(kind="scale_blocks", mode="jitter", sigma=sigma, match=_layer_proj(layer, "weight_scale_inv"), count=0, seed=s)))
            out.append(("fp8_scale_factor", "5% of blocks x2", dict(kind="scale_blocks", mode="factor", factor=2.0, fraction=0.05, match=FP8_SCALES, seed=s)))
            out.append(("fp8_scale_ones", "one tensor", dict(kind="scale_blocks", mode="ones", match=FP8_SCALES, seed=s)))
            out.append(("fp8_scale_transpose", "square k/v grid", dict(kind="scale_blocks", mode="transpose", match=r"self_attn\.(k|v)_proj\.weight_scale_inv$", seed=s)))
            out.append(("fp8_scale_swap_tp", "tp=2", dict(kind="scale_blocks", mode="swap_tp", tp=2, match=FP8_SCALES, seed=s)))
            for n in (1, 16):
                out.append(("fp8_bitflip_exp_msb", f"n={n}", dict(kind="bitflip", bit="exp_msb", n=n, match=PROJ, seed=s)))
        elif variant == "mxfp4":
            for frac in (0.001, 0.01, 0.1):
                out.append(("mxfp4_scale_shift+1", f"{frac:g} of blocks", dict(kind="mxfp4_scales", mode="shift", k=1, fraction=frac, match=MX_SCALES, seed=s)))
            out.append(("mxfp4_scale_shift-1", "0.01 of blocks", dict(kind="mxfp4_scales", mode="shift", k=-1, fraction=0.01, match=MX_SCALES, seed=s)))
            out.append(("mxfp4_scale_roll", "one tensor", dict(kind="mxfp4_scales", mode="roll", match=MX_SCALES, seed=s)))
            out.append(("mxfp4_scale_zero", "0.01 of blocks", dict(kind="mxfp4_scales", mode="zero", fraction=0.01, match=MX_SCALES, seed=s)))
            out.append(("mxfp4_zero_blocks", "1/8 of a tensor", dict(kind="zero_slice", match=r"mlp\.(gate|up|down)_proj\.weight_blocks$", tp=8, shard=s % 8, seed=s)))
    return out


def run_specs(variant: str, n_calib: int = 16, n_heldout: int = 40, n_tp: int = 10, fault_seeds=(0, 1, 2)):
    specs = [{"family": "reference", "seed": 0}, {"family": "repeat", "seed": 0}]
    specs += [{"family": "request", "seed": k, "role": "calibration"} for k in range(1, n_calib + 1)]
    specs += [{"family": "request", "seed": 100 + k, "role": "heldout"} for k in range(1, n_heldout + 1)]
    for tp in (2, 4):
        specs += [{"family": f"tp{tp}", "seed": 100 * tp + k, "role": "heldout"} for k in range(1, n_tp + 1)]
    for i, (cls, mag, spec) in enumerate(fault_grid(variant, fault_seeds)):
        specs.append({"family": "request", "seed": 5000 + i, "role": "fault", "class": cls, "magnitude": mag,
                      "fault": spec})
    for s in specs:
        s["variant"] = variant
        s["id"] = f"{variant}:{s['family']}:{s['seed']}"
    return specs


# --------------------------------------------------------------------------
# worker process


_W = {}


def _to_torch(t: faults.Tensor) -> torch.Tensor:
    a = np.array(t.data, copy=True)
    if t.dtype == "BF16":
        return torch.from_numpy(a.view(np.int16)).view(torch.bfloat16)
    if t.dtype == "F8_E4M3":
        return torch.from_numpy(a).view(torch.float8_e4m3fn)
    return torch.from_numpy(a)


def _init(variant: str, threads: int):
    from transformers import AutoTokenizer

    torch.set_num_threads(threads)
    path = VARIANTS[variant]
    raw = {}
    for f in faults.checkpoint_files(path):
        raw.update(faults.read_file(f)[0])
    clean = load_checkpoint(path)
    tok = AutoTokenizer.from_pretrained(str(TOKENIZER))
    prompts = [(p["key"], p["text"]) for p in json.loads(PROMPTS.read_text())["prompts"]]
    _W.update(variant=variant, raw=raw, clean=clean, tok=tok, prompts=prompts, ref_lp=None, held=None)


def faulted_checkpoint(spec: dict):
    """The clean checkpoint with `spec` applied to its raw tensors, re-dequantized where it matters."""
    clean: Checkpoint = _W["clean"]
    raw = dict(_W["raw"])
    report = faults.apply(raw, faults.FaultSpec.from_dict(spec))
    touched = {c.name for c in report.changes} | set(report.dropped)
    bases = set()
    for name in touched:
        for suf in ("_scale_inv", "_blocks", "_scales"):
            if name.endswith(suf):
                name = name[: -len(suf)]
        bases.add(name)
    subset = {}
    for base in bases:
        for name in (base, base + "_scale_inv", base + "_blocks", base + "_scales"):
            if name in raw:
                subset[name] = _to_torch(raw[name])
    new_w, new_fp8 = dequantize(clean.config, subset)
    weights = dict(clean.weights)
    for base in bases:
        if base in new_w:
            weights[base] = new_w[base]
        elif base in weights:
            weights[base] = torch.zeros_like(clean.weights[base])  # dropped: an unchecking loader leaves zeros
    tie = getattr(clean.config, "tie_word_embeddings", False)
    if tie and "model.embed_tokens.weight" in bases:
        weights["lm_head.weight"] = weights["model.embed_tokens.weight"]
    fp8 = set(clean.fp8)
    ck = Checkpoint(config=clean.config, weights=weights, fp8=fp8, missing=sorted(report.dropped), source="fault")
    return ck, report


def _numerics_for(family: str, seed: int):
    if family in ("reference", "repeat"):
        return lambda i, key: Numerics()
    tp = {"request": 1, "tp2": 2, "tp4": 4}[family]
    rng = random.Random(seed)
    cache = {}

    def num(i, key):
        if i not in cache:
            n_tok = len(_W["tok"](_W["prompts"][i][1]).input_ids)
            prefix = rng.randrange(4, n_tok - 2) if n_tok > 8 and rng.random() < 0.3 else None
            cache[i] = Numerics(split_k=rng.choice([1, 1, 2, 4]), attn_splits=rng.choice([1, 1, 2, 4]), tp=tp,
                                chunk=rng.choice([None, None, 64, 128, 256]), prefix=prefix)
        return cache[i]

    return num


def _ref_lp():
    """Full log-softmax of the clean model under the reference schedule, per prompt (cached)."""
    if _W["ref_lp"] is None:
        eng = Engine(_W["clean"])
        _W["ref_lp"] = [eng.score(_W["tok"](t).input_ids, Numerics(), 2)[3] for _, t in _W["prompts"]]
        ids = _W["tok"](HELD_OUT).input_ids
        lps = eng.score(ids, Numerics(), 2)[0]
        _W["held"] = (ids, -sum(lps) / len(lps))
    return _W["ref_lp"]


def _kl(ref_lps, lps) -> float:
    tot, n = 0.0, 0
    for a, b in zip(ref_lps, lps):
        p = a.exp()
        kl = (p * (a - b)).sum(-1)
        kl = torch.where(torch.isfinite(kl), kl, torch.full_like(kl, float("inf")))
        tot += float(kl.double().sum())
        n += kl.numel()
    return tot / n


def run_one(spec: dict) -> dict:
    t0 = time.time()
    fault_report = None
    if spec.get("fault"):
        ck, rep = faulted_checkpoint(spec["fault"])
        fault_report = rep.to_dict()
    else:
        ck = _W["clean"]
    eng = Engine(ck)
    cap = capture(eng, _W["tok"], _W["prompts"], _numerics_for(spec["family"], spec["seed"]),
                  greedy_prompts=GREEDY_PROMPTS, greedy_tokens=GREEDY_TOKENS, top_k=2,
                  model=f"Qwen3-0.6B/{spec['variant']}", full_lp=True)
    cap["schedule"] = {"kind": spec["family"] if spec["family"] in ("reference", "repeat") else "perturbed",
                       "family": spec["family"], "seed": spec["seed"]}
    ref = _ref_lp()
    effect = {"kl_vs_ref": _kl(ref, [p.pop("_lp") for p in cap["prompts"]])}
    if spec.get("fault"):
        # The fault alone, with the reference's own numerics: what exact mode would see.
        lps_fault = []
        changed_tokens = 0
        for (key, text), r in zip(_W["prompts"], ref):
            ids = _W["tok"](text).input_ids
            tok_lp, _, _, full = eng.score(ids, Numerics(), 2)
            lps_fault.append(full)
            target = torch.tensor(ids[1:])
            ref_t = r.gather(1, target.unsqueeze(1)).squeeze(1)
            changed_tokens += int((torch.tensor(tok_lp) != ref_t).sum())
        effect["kl_fault_only"] = _kl(ref, lps_fault)
        effect["tokens_changed_ref_schedule"] = changed_tokens
        ids, nll0 = _W["held"]
        lps = eng.score(ids, Numerics(), 2)[0]
        nll = -sum(lps) / len(lps)
        effect["heldout_nll_delta"] = nll - nll0 if math.isfinite(nll) else float("inf")
    return {"spec": spec, "capture": cap, "effect": effect, "fault_report": fault_report,
            "seconds": time.time() - t0}


# --------------------------------------------------------------------------
# driver


def _done_ids(path: Path) -> set:
    if not path.exists():
        return set()
    ids = set()
    with gzip.open(path, "rt") as fh:
        for line in fh:
            try:
                ids.add(json.loads(line)["spec"]["id"])
            except (json.JSONDecodeError, KeyError):
                pass
    return ids


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", choices=sorted(VARIANTS), required=True)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--threads", type=int, default=3)
    ap.add_argument("--heldout", type=int, default=40)
    ap.add_argument("--tp-runs", type=int, default=10)
    ap.add_argument("--calib", type=int, default=16)
    ap.add_argument("--fault-seeds", type=int, default=3)
    ap.add_argument("--only", default=None, help="run only specs whose id contains this")
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args(argv)

    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / f"{args.variant}.jsonl.gz"
    specs = run_specs(args.variant, args.calib, args.heldout, args.tp_runs, tuple(range(args.fault_seeds)))
    done = _done_ids(path)
    todo = [s for s in specs if s["id"] not in done and (args.only is None or args.only in s["id"])]
    if args.limit:
        todo = todo[: args.limit]
    print(f"{args.variant}: {len(specs)} runs, {len(done)} done, {len(todo)} to go", flush=True)
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=args.workers, initializer=_init,
                             initargs=(args.variant, args.threads)) as ex:
        futures = [ex.submit(run_one, s) for s in todo]
        for k, fut in enumerate(as_completed(futures), 1):
            rec = fut.result()
            with gzip.open(path, "at") as fh:
                fh.write(json.dumps(rec) + "\n")
            sp = rec["spec"]
            print(f"[{k}/{len(todo)} {time.time() - t0:6.0f}s] {sp['id']} {sp.get('class', '')} "
                  f"{sp.get('magnitude', '')} kl={rec['effect']['kl_vs_ref']:.3g} ({rec['seconds']:.1f}s)", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
