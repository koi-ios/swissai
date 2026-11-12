"""Where an SGLang cold start goes once servekit has fixed weight loading.

Reads the `servekit profile` JSON reports published in Youssef Boughizane's experiment
repository (https://github.com/youssef62/cold-start-elimination-experiments-swissai),
clone it first:

    git clone --depth 1 https://github.com/youssef62/cold-start-elimination-experiments-swissai.git

Usage:
    python residual_breakdown.py <repo> --out <markdown file>

Groups servekit's phases into buckets and reports the median [min-max] per bucket over
the head node of every servekit run, per cluster and model. All runs are the lab's own
(Clariden GH200 / Bristen, SGLang v0.5.16 for servekit-eval and the JIT-cache A/B).
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import statistics as st
from collections import defaultdict

BUCKETS = [
    ("python_start_and_worker_spawn", "process start + TP worker spawn (imports)"),
    ("dist_init", "torch.distributed init"),
    ("weights", "weight loading (+ stage when not overlapped)"),
    ("kv_alloc", "KV cache allocation"),
    ("prefill_graphs", "prefill CUDA graph capture (breakable)"),
    ("decode_graphs", "decode CUDA graph capture"),
    ("warmup_jit", "first warmup request (JIT)"),
    ("other", "other gaps (unknown, http bind)"),
]


def bucketize(phases):
    b = defaultdict(float)
    seen_dist = False
    for p in phases:
        n, d = p["name"], float(p["duration_s"])
        if n in ("process_startup", "tp_worker_spawn"):
            b["python_start_and_worker_spawn"] += d
        elif n == "unknown" and not seen_dist:
            # before torch.distributed init: worker spawn on multi-node runs, where
            # servekit cannot name it
            b["python_start_and_worker_spawn"] += d
        elif n == "torch_distributed_init":
            b["dist_init"] += d
            seen_dist = True
        elif n in ("weight_loading", "stage"):
            b["weights"] += d
        elif n == "kv_cache_alloc":
            b["kv_alloc"] += d
        elif n in ("prefill_cuda_graph_capture", "piecewise_cuda_graph_capture"):
            b["prefill_graphs"] += d
        elif n == "cuda_graph_capture":
            b["decode_graphs"] += d
        elif n.startswith("warmup_request"):
            b["warmup_jit"] += d
        else:
            b["other"] += d
    return b


def fmt(xs):
    if not xs:
        return "-"
    if len(xs) == 1:
        return f"{xs[0]:.1f}"
    return f"{st.median(xs):.1f} [{min(xs):.1f}-{max(xs):.1f}]"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("repo")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    groups = defaultdict(list)
    for p in sorted(glob.glob(os.path.join(a.repo, "experiments/servekit-eval/results/*/*servekit*/run*.json"))):
        if os.path.basename(p) not in ("run.json", "run.node0.json"):
            continue  # head node only: it is the one that serves
        d = json.load(open(p))
        if not d.get("success"):
            continue
        cluster_model = os.path.basename(os.path.dirname(os.path.dirname(p)))  # e.g. clariden-llama70b
        arm = "overlap" if "overlap" in p else "serial"
        groups[(cluster_model, f"servekit ({arm})")].append(d)
    for name in ("no-cache", "with-cache"):
        for p in glob.glob(os.path.join(a.repo, f"experiments/servekit-jit-caching-eval/results/*/{name}.json")):
            groups[("bristen-llama70b (JIT A/B)", f"servekit overlap, JIT {name}")].append(json.load(open(p)))

    lines = ["| cluster-model | arm | n | total s | " + " | ".join(label for _, label in BUCKETS) + " |",
             "|---|---|---|---|" + "---|" * len(BUCKETS)]
    shares = []
    for (cm, arm), runs in sorted(groups.items()):
        totals = [float(r["total_duration_s"]) for r in runs]
        cols = []
        for key, _ in BUCKETS:
            cols.append(fmt([bucketize(r["phases"]).get(key, 0.0) for r in runs]))
        lines.append(f"| {cm} | {arm} | {len(runs)} | {fmt(totals)} | " + " | ".join(cols) + " |")
        med_total = st.median(totals)
        sh = {key: st.median([bucketize(r["phases"]).get(key, 0.0) for r in runs]) / med_total for key, _ in BUCKETS}
        shares.append((cm, arm, med_total, sh))
    lines += ["", "Share of the median total (medians of each bucket / median total):", "",
              "| cluster-model | arm | median total s | " + " | ".join(k for k, _ in BUCKETS) + " |",
              "|---|---|---|" + "---|" * len(BUCKETS)]
    for cm, arm, tot, sh in shares:
        lines.append(f"| {cm} | {arm} | {tot:.1f} | " + " | ".join(f"{sh[k]:.0%}" for k, _ in BUCKETS) + " |")
    text = "\n".join(lines) + "\n"
    open(a.out, "w").write(text)
    print(text)


if __name__ == "__main__":
    main()
