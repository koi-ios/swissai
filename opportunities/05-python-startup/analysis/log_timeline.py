"""Split the lab's "process start + TP worker spawn" phase into its parts, from raw logs.

Reads every SGLang job log (`*.out`) in Youssef Boughizane's experiment repository
(github.com/youssef62/cold-start-elimination-experiments-swissai) and, for each launch,
lines up three clocks:

* `started_at` from servekit's profile report (epoch, sub-second): the moment servekit
  Popen()s `python -m sglang.launch_server`;
* SGLang's own logger timestamps (1 s resolution, CEST) for `server_args=` (the main
  process has finished its imports and argument post-processing and is about to spawn
  the schedulers) and `Init torch distributed begin` (every TP rank's scheduler is up);
* where the torchao compile-time SyntaxWarning appears (before or after servekit hands
  over to SGLang), which tells us which process compiled that module from source.

Usage:
    python log_timeline.py <experiments-repo> --csv out.csv --md out.md

Only the head node of single-node runs is split (multi-node logs interleave four nodes'
output). Nothing here is re-measured: these are the lab's runs, one job per row.
"""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import glob
import json
import os
import re
import statistics as st
from collections import defaultdict

CEST = dt.timezone(dt.timedelta(hours=2))  # all runs are Aug-Sep 2026, CEST
TS = re.compile(r"^\[(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})(?: (TP\d+|DP\d+|PP\d+)[^\]]*)?\] ")
SERVER_ARGS = re.compile(r"server_args=ServerArgs\(")
DIST_BEGIN = re.compile(r"Init torch distributed begin")
CHAT_TEMPLATE = re.compile(r"Using default HuggingFace chat template|chat template")
SYNTAX_WARNING = re.compile(r"SyntaxWarning: invalid escape sequence")
HANDOVER = re.compile(r"\[SERVEKIT\] (launching:|profiling cold start)")
STARTED_AT = re.compile(r'"started_at":\s*([\d.]+)')


def parse_ts(s: str) -> float:
    return dt.datetime.strptime(s, "%Y-%m-%d %H:%M:%S").replace(tzinfo=CEST).timestamp()


def classify(path: str) -> dict:
    p = path.replace(os.sep, "/")
    if "/servekit-eval/" in p:
        exp, image, wrapper = "servekit-eval", "v0.5.16", "sitecustomize imports torch+sglang in every python process"
    elif "/servekit-jit-caching-eval/" in p:
        exp, image, wrapper = "jit-cache A/B", "v0.5.16", "none"
    elif "/cpu-sweep/" in p:
        exp, image, wrapper = "cpu-sweep", "v0.5.10", "none"
    elif "/lustre-loading-exp/" in p:
        exp, image, wrapper = "lustre-loading", "v0.5.10", "none"
    else:
        exp, image, wrapper = "other", "?", "?"
    m = re.search(r"results/(bristen|clariden)-([a-z0-9.]+)/", p)
    if m:
        cluster, model = m.group(1), m.group(2)
    else:
        cluster = "bristen"  # every non-servekit-eval experiment ran on bristen
        model = "llama70b" if ("llama" in p or exp != "other") else "?"
        if "apertus" in p:
            model = "apertus8b"
    return {"experiment": exp, "image": image, "preimport_wrapper": wrapper, "cluster": cluster, "model": model}


def started_at_for(path: str, text: str):
    """servekit's started_at for this log: embedded report, sibling run dir, or results/<jobid>/."""
    m = STARTED_AT.search(text)
    if m:
        return float(m.group(1))
    base = os.path.splitext(os.path.basename(path))[0]
    job = base.rsplit("-", 1)[-1]
    d = os.path.dirname(path)
    cands = [os.path.join(d, base, "run.json"), os.path.join(d, base, "run.node0.json")]
    cands += glob.glob(os.path.join(d, job, "*.json"))
    cands += glob.glob(os.path.join(d, f"*{job}*profile.json"))
    for c in cands:
        if os.path.exists(c):
            try:
                return float(json.load(open(c))["started_at"]), c
            except Exception:
                continue
    return None


def report_phases(path: str):
    base = os.path.splitext(os.path.basename(path))[0]
    job = base.rsplit("-", 1)[-1]
    d = os.path.dirname(path)
    cands = [os.path.join(d, base, "run.json"), os.path.join(d, base, "run.node0.json")]
    cands += glob.glob(os.path.join(d, job, "*.json")) + glob.glob(os.path.join(d, f"*{job}*profile.json"))
    for c in cands:
        if os.path.exists(c):
            try:
                rep = json.load(open(c))
                ph = {p["name"]: p["duration_s"] for p in rep["phases"]}
                return rep, ph
            except Exception:
                continue
    return None, {}


def analyse(path: str) -> dict | None:
    text = open(path, errors="replace").read()
    lines = text.splitlines()
    n_args = sum(1 for l in lines if SERVER_ARGS.search(l))
    if n_args != 1:
        return None  # multi-node (interleaved) or no SGLang launch
    row = classify(path)
    row["log"] = os.path.relpath(path)
    rep, ph = report_phases(path)
    sa = rep["started_at"] if rep else None
    if sa is None:
        m = STARTED_AT.search(text)
        sa = float(m.group(1)) if m else None
    row["spawn_epoch"] = sa
    t_args = t_dist_first = t_dist_last = t_first_ts = t_tp_first = None
    handover_idx = None
    sw_before = sw_after = 0
    for i, l in enumerate(lines):
        if handover_idx is None and HANDOVER.search(l):
            handover_idx = i
        if SYNTAX_WARNING.search(l):
            if handover_idx is None:
                sw_before += 1
            else:
                sw_after += 1
        m = TS.match(l)
        if not m:
            continue
        t = parse_ts(m.group(1))
        if t_first_ts is None:
            t_first_ts = t
        if t_args is None and SERVER_ARGS.search(l):
            t_args = t
        if m.group(2) and t_tp_first is None:
            t_tp_first = t
        if DIST_BEGIN.search(l):
            t_dist_first = t if t_dist_first is None else t_dist_first
            t_dist_last = t
    row["syntaxwarn_before_handover"] = sw_before
    row["syntaxwarn_after_handover"] = sw_after
    row["servekit_process_startup_s"] = ph.get("process_startup")
    row["servekit_tp_worker_spawn_s"] = ph.get("tp_worker_spawn")
    if sa and t_args and t_dist_last:
        # log timestamps are truncated to the second: +0.5 s is the unbiased midpoint
        row["main_to_server_args_s"] = round(t_args + 0.5 - sa, 1)
        row["children_to_dist_begin_s"] = round(t_dist_last - t_args, 1)
        row["spawn_to_dist_begin_s"] = round(t_dist_last + 0.5 - sa, 1)
        row["first_output_s"] = ph.get("process_startup")
    return row


def fmt(xs):
    xs = [x for x in xs if x is not None]
    if not xs:
        return "-"
    if len(xs) == 1:
        return f"{xs[0]:.1f}"
    return f"{st.median(xs):.1f} [{min(xs):.1f}-{max(xs):.1f}]"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("repo")
    ap.add_argument("--csv", required=True)
    ap.add_argument("--md", required=True)
    a = ap.parse_args()
    rows = []
    for p in sorted(glob.glob(os.path.join(a.repo, "experiments", "**", "*.out"), recursive=True)):
        r = analyse(p)
        if r:
            rows.append(r)
    keys = ["log", "experiment", "cluster", "model", "image", "preimport_wrapper", "spawn_epoch",
            "servekit_process_startup_s", "servekit_tp_worker_spawn_s", "main_to_server_args_s",
            "children_to_dist_begin_s", "spawn_to_dist_begin_s", "syntaxwarn_before_handover",
            "syntaxwarn_after_handover"]
    with open(a.csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(r)
    groups = defaultdict(list)
    for r in rows:
        if r.get("main_to_server_args_s") is None:
            continue
        groups[(r["cluster"], r["experiment"], r["image"], "yes" if r["preimport_wrapper"].startswith("sitecustomize") else "no")].append(r)
    out = ["| cluster | experiment | image | torch pre-imported by a wrapper before SGLang starts | n | servekit `process_startup` (to first output line) | main process: spawn → `server_args=` | TP schedulers: `server_args=` → last `Init torch distributed begin` | total: spawn → dist init | SyntaxWarnings from wrapper / from SGLang (per log) |",
           "|---|---|---|---|---|---|---|---|---|---|"]
    for k in sorted(groups):
        rs = groups[k]
        sw = f"{fmt([r['syntaxwarn_before_handover'] for r in rs])} / {fmt([r['syntaxwarn_after_handover'] for r in rs])}"
        out.append(f"| {k[0]} | {k[1]} | {k[2]} | {k[3]} | {len(rs)} | {fmt([r['servekit_process_startup_s'] for r in rs])} | "
                   f"{fmt([r['main_to_server_args_s'] for r in rs])} | {fmt([r['children_to_dist_begin_s'] for r in rs])} | "
                   f"{fmt([r['spawn_to_dist_begin_s'] for r in rs])} | {sw} |")
    out += ["", "Median [min-max] over jobs, seconds. Log timestamps have 1 s resolution (taken at +0.5 s).",
            "`spawn` is servekit's `started_at` (Popen of `python -m sglang.launch_server`)."]
    open(a.md, "w").write("\n".join(out) + "\n")
    print("\n".join(out))
    print(f"\n{len(rows)} single-node launches parsed")


if __name__ == "__main__":
    main()
