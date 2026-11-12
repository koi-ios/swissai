"""Benign logprob drift in servekit's own e2e fixtures: Llama-3.1-70B-Instruct at TP4 vs TP8.

Both fixtures were recorded by `servekit verify --record` on the platform (SGLang,
BF16) with the built-in prompt set; only the TP degree (and so the reduction order
of every row-parallel layer) differs. Prints the drift, its scaling with the
observed token's probability, and the greedy agreement. Stdlib only.

    python -I analysis/fixtures_noise.py
"""
import json
import math
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FIX = ROOT / "vendor" / "servekit" / "tests" / "e2e" / "fixtures"


def q(xs, p):
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(p * (len(xs) - 1) + 0.5))]


def main():
    a = json.loads((FIX / "llama70b-tp4-bf16.json").read_text())
    b = json.loads((FIX / "llama70b-tp8-bf16.json").read_text())
    ref, delta, keys = [], [], []
    greedy = []
    for pa, pb in zip(a["prompts"], b["prompts"]):
        assert pa["key"] == pb["key"] and pa["n_tokens"] == pb["n_tokens"]
        ref += pa["token_logprobs"]
        delta += [y - x for x, y in zip(pa["token_logprobs"], pb["token_logprobs"])]
        keys += [pa["key"]] * pa["n_tokens"]
        if "greedy_tokens" in pa:
            ga, gb = pa["greedy_tokens"], pb["greedy_tokens"]
            first = next((i for i, (x, y) in enumerate(zip(ga, gb)) if x != y), None)
            greedy.append((pa["key"], first))
    ad = [abs(d) for d in delta]
    print(f"tokens: {len(ad)}; |delta logprob| median {statistics.median(ad):.2e}, p90 {q(ad, .9):.3g}, "
          f"p99 {q(ad, .99):.3g}, max {max(ad):.3g}; exactly equal: {sum(d == 0 for d in ad) / len(ad):.1%}")
    print(f"signed mean delta {statistics.fmean(delta):+.2e} (sd {statistics.pstdev(delta):.3g}): no NLL bias")
    print("\n|delta| / (1 - p) by the reference's -logprob (p = exp(logprob) of the observed token):")
    print(f"{'-logprob bin':>16} {'n':>5} {'median':>9} {'p90':>9}")
    bins = [(0, 1e-3), (1e-3, 1e-2), (1e-2, 5e-2), (5e-2, .2), (.2, .7), (.7, 1.5), (1.5, 3), (3, 6), (6, 30)]
    for lo, hi in bins:
        u = [x / max(-math.expm1(r), 1e-12) for x, r in zip(ad, ref) if lo < -r <= hi]
        if u:
            print(f"({lo:6.3f},{hi:5.2f}] {len(u):5d} {statistics.median(u):9.3f} {q(u, .9):9.3f}")
    print("\nfirst greedy divergence (TP4 vs TP8):", ", ".join(f"{k}@{f}" for k, f in greedy if f is not None) or "none")
    print("(the first bin is inflated by the fixtures' 6-decimal rounding: 5e-7 / 1e-5 = 0.05)")


if __name__ == "__main__":
    main()
