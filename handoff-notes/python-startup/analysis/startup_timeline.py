"""Per-launch timeline of the first ~40 s of an SGLang cold start, from the lab's raw logs.

Reads every SGLang job log (`*.out`) in Youssef Boughizane's experiment repository
(github.com/youssef62/cold-start-elimination-experiments-swissai) and joins it with the
`servekit profile` JSON written by the same job. For each launch it lines up:

  t_job     `started=` echoed by the sbatch script on the host, before `srun`
  t_popen   servekit's `started_at`: Popen of `python -m sglang.launch_server` (epoch, sub-second)
  t_first   t_popen + servekit's `process_startup` (first line the engine prints)
  t_args    SGLang's `server_args=` line (main process done importing + parsing)
  t_child   first line tagged `TPn` (a scheduler child is alive and logging)
  t_dbeg    last `Init torch distributed begin` over the TP ranks of the head node
  t_nccl    `sglang is using nccl==`
  t_ar      `All Reduce config:` (custom all-reduce set up)
  t_dend    last `Init torch distributed ends. elapsed=`

SGLang's logger has 1 s resolution in local time (CEST, +02:00). Its stamps are taken
at +0.5 s, so differences against t_popen carry +/-0.5 s of error.

It also counts per log: pynvml FutureWarnings (one per python process that imports
torch.cuda), torchao SyntaxWarnings (one per process that COMPILES quant_api.py from
source), and whether those warnings come before servekit hands over to SGLang.

Nothing here is re-measured: one row per job of the lab's runs.

Usage:
    python startup_timeline.py <experiments-dir> --csv out.csv --md out.md
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

CEST = dt.timezone(dt.timedelta(hours=2))  # every run is Aug-Sep 2026
TS = re.compile(r"^\[(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})(?: ([A-Z]+\d+)[^\]]*)?\] ")
STARTED = re.compile(r"^started=(\S+)")
SERVER_ARGS = re.compile(r"server_args=ServerArgs\(")
DIST_BEGIN = re.compile(r"Init torch distributed begin")
DIST_END = re.compile(r"Init torch distributed ends\. elapsed=([\d.]+) s")
NCCL = re.compile(r"sglang is using nccl==")
ALLRED = re.compile(r"All Reduce config:")
FUTURE = re.compile(r"FutureWarning: The pynvml package is deprecated")
SYNTAX = re.compile(r"SyntaxWarning: invalid escape sequence")
HANDOVER = re.compile(r"\[SERVEKIT\] (launching:|profiling cold start)")
STAGED = re.compile(r"\[SERVEKIT\] staged ([\d.]+) GB in ([\d.]+) s")
OVERLAP = re.compile(r"\(overlapped|--overlap")


def ts(s: str) -> float:
    return dt.datetime.strptime(s, "%Y-%m-%d %H:%M:%S").replace(tzinfo=CEST).timestamp() + 0.5


def classify(p: str) -> dict:
    p = p.replace(os.sep, "/")
    d = {"experiment": "other", "image": "?", "prewarm": "?"}
    if "/servekit-eval/" in p:
        d.update(experiment="servekit-eval", image="v0.5.16", prewarm="yes (sitecustomize)")
    elif "/servekit-jit-caching-eval/" in p:
        d.update(experiment="jit-cache-ab", image="v0.5.16", prewarm="no")
    elif "/cpu-sweep/" in p:
        d.update(experiment="cpu-sweep", image="v0.5.10", prewarm="no")
    elif "/lustre-loading-exp/" in p:
        d.update(experiment="lustre-loading", image="v0.5.10", prewarm="no")
    m = re.search(r"results/(bristen|clariden)-([a-z0-9.]+)/", p)
    if m:
        d["cluster"], d["model"] = m.group(1), m.group(2)
    else:
        d["cluster"] = "bristen"  # cpu-sweep, lustre-loading and jit A/B ran on bristen only
        d["model"] = "llama70b"
    base = os.path.basename(p)[:-4]
    d["job"] = base
    m = re.search(r"-(servekit-overlap|servekit|mmap2|mmap|nommap|fst|preshard|shm_mmap|overlap|no-cache|with-cache|prepare-fa3)-", base + "-")
    d["arm"] = m.group(1) if m else base
    m = re.search(r"cpu(\d+)", p)
    d["cpus"] = int(m.group(1)) if m else 128
    return d


def find_profile(log: str) -> dict | None:
    stem = log[:-4]
    jobid = re.findall(r"(\d{5,8})", os.path.basename(stem))
    cands = []
    if os.path.isdir(stem):
        cands += sorted(glob.glob(os.path.join(stem, "run.json")))
        cands += sorted(glob.glob(os.path.join(stem, "run.node0.json")))
    d = os.path.dirname(log)
    if jobid:
        j = jobid[-1]
        cands += sorted(glob.glob(os.path.join(d, j, "*.json")))
        cands += sorted(glob.glob(os.path.join(d, "*", f"*-{j}-*profile.json")))
        cands += sorted(glob.glob(os.path.join(d, f"*-{j}-*profile.json")))
    for c in cands:
        try:
            with open(c) as f:
                js = json.load(f)
            if "started_at" in js:
                js["_path"] = c
                return js
        except Exception:
            continue
    return None


def parse(log: str) -> dict | None:
    with open(log, errors="replace") as f:
        lines = f.read().splitlines()
    r = classify(log)
    ev = defaultdict(list)
    handover_idx = None
    n_future = n_syntax = n_syntax_before = n_future_before = 0
    nnodes = 1
    for i, line in enumerate(lines):
        if handover_idx is None and HANDOVER.search(line):
            handover_idx = i
        m = STARTED.match(line)
        if m and "t_job" not in r:
            r["t_job"] = dt.datetime.fromisoformat(m.group(1)).timestamp()
        if line.startswith("tp=") and "nodes=" in line:
            nnodes = int(re.search(r"nodes=(\d+)", line).group(1))
        if FUTURE.search(line):
            n_future += 1
            if handover_idx is None:
                n_future_before += 1
        if SYNTAX.search(line):
            n_syntax += 1
            if handover_idx is None:
                n_syntax_before += 1
        m = STAGED.search(line)
        if m:
            r["stage_gb"], r["stage_s"] = float(m.group(1)), float(m.group(2))
        if OVERLAP.search(line):
            r["overlap"] = True
        m = TS.match(line)
        if not m:
            continue
        t, tag = ts(m.group(1)), m.group(2)
        if SERVER_ARGS.search(line):
            ev["args"].append(t)
        if tag and tag.startswith("TP"):
            ev["child"].append(t)
        if DIST_BEGIN.search(line):
            ev["dbeg"].append(t)
        if NCCL.search(line):
            ev["nccl"].append(t)
        if ALLRED.search(line):
            ev["ar"].append(t)
        md = DIST_END.search(line)
        if md:
            ev["dend"].append(t)
            ev["dend_elapsed"].append(float(md.group(1)))
    if not ev["args"]:
        return None
    r["nnodes"] = nnodes
    r.setdefault("overlap", False)
    r.update(n_future=n_future, n_future_before_handover=n_future_before,
             n_syntax=n_syntax, n_syntax_before_handover=n_syntax_before)
    prof = find_profile(log)
    if prof:
        r["t_popen"] = prof["started_at"]
        ph = {p["name"]: p["duration_s"] for p in prof.get("phases", [])}
        r["process_startup"] = ph.get("process_startup")
        r["tp_worker_spawn"] = ph.get("tp_worker_spawn")
        r["dist_init_reported"] = ph.get("torch_distributed_init")
        r["total"] = prof.get("total_duration_s")
        r["profile"] = os.path.relpath(prof["_path"], os.path.dirname(log))
    r["t_args"] = ev["args"][0]
    r["t_child"] = min(ev["child"]) if ev["child"] else None
    # head node only for multi-node logs: the first 4 TP ranks to report are the head's
    r["t_dbeg"] = max(ev["dbeg"][:4]) if ev["dbeg"] else None
    r["t_nccl"] = ev["nccl"][0] if ev["nccl"] else None
    r["t_ar"] = ev["ar"][0] if ev["ar"] else None
    r["t_dend"] = max(ev["dend"][:4]) if ev["dend"] else None
    r["dist_elapsed_max"] = max(ev["dend_elapsed"][:4]) if ev["dend_elapsed"] else None
    return r


def d(a, b):
    if a is None or b is None:
        return None
    return round(a - b, 1)


def derive(r: dict) -> dict:
    out = dict(r)
    out["job_to_popen"] = d(r.get("t_popen"), r.get("t_job"))
    out["popen_to_first"] = r.get("process_startup")
    out["popen_to_args"] = d(r.get("t_args"), r.get("t_popen"))
    out["args_to_dbeg"] = d(r.get("t_dbeg"), r.get("t_args"))
    out["popen_to_dbeg"] = d(r.get("t_dbeg"), r.get("t_popen"))
    out["dbeg_to_nccl"] = d(r.get("t_nccl"), r.get("t_dbeg"))
    out["nccl_to_ar"] = d(r.get("t_ar"), r.get("t_nccl"))
    out["ar_to_dend"] = d(r.get("t_dend"), r.get("t_ar"))
    st_s = r.get("stage_s") if not r.get("overlap") else 0.0
    if out["job_to_popen"] is not None:
        out["job_to_popen_minus_stage"] = round(out["job_to_popen"] - (st_s or 0.0), 1)
    return out


COLS = ["experiment", "cluster", "model", "arm", "job", "image", "prewarm", "cpus", "nnodes", "overlap",
        "stage_gb", "stage_s", "job_to_popen", "job_to_popen_minus_stage", "popen_to_first", "popen_to_args",
        "args_to_dbeg", "popen_to_dbeg", "process_startup", "tp_worker_spawn", "dist_init_reported",
        "dbeg_to_nccl", "nccl_to_ar", "ar_to_dend", "dist_elapsed_max", "total",
        "n_future", "n_future_before_handover", "n_syntax", "n_syntax_before_handover", "profile"]


def med(xs):
    xs = [x for x in xs if x is not None]
    if not xs:
        return "-"
    return f"{st.median(xs):.1f} [{min(xs):.1f}, {max(xs):.1f}] (n={len(xs)})"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("experiments")
    ap.add_argument("--csv", required=True)
    ap.add_argument("--md", required=True)
    a = ap.parse_args()
    rows = []
    for log in sorted(glob.glob(os.path.join(a.experiments, "**", "*.out"), recursive=True)):
        r = parse(log)
        if r:
            rows.append(derive(r))
    with open(a.csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=COLS, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(r)
    groups = defaultdict(list)
    for r in rows:
        if r["nnodes"] != 1:
            continue
        key = (r["experiment"], r["cluster"], r["image"], r["prewarm"])
        groups[key].append(r)
    hdr = ["experiment", "cluster", "image", "torch pre-imported before Popen", "n",
           "job start -> Popen, minus serial staging (s)", "Popen -> first line (s)", "Popen -> server_args (s)",
           "server_args -> last TP 'dist begin' (s)", "dist begin -> nccl line (s)", "nccl -> All Reduce config (s)",
           "All Reduce config -> dist end (s)", "pynvml FutureWarnings / log", "SyntaxWarnings / log"]
    md = ["| " + " | ".join(hdr) + " |", "|" + "---|" * len(hdr)]
    for key, rs in sorted(groups.items()):
        md.append("| " + " | ".join([*key, str(len(rs)),
                                     med([r.get("job_to_popen_minus_stage") for r in rs]),
                                     med([r.get("popen_to_first") for r in rs]),
                                     med([r.get("popen_to_args") for r in rs]),
                                     med([r.get("args_to_dbeg") for r in rs]),
                                     med([r.get("dbeg_to_nccl") for r in rs]),
                                     med([r.get("nccl_to_ar") for r in rs]),
                                     med([r.get("ar_to_dend") for r in rs]),
                                     med([r.get("n_future") for r in rs]),
                                     med([r.get("n_syntax") for r in rs])]) + " |")
    md.append("")
    md.append("Single-node launches only (head node). median [min, max] (n), seconds. SGLang log stamps have 1 s resolution.")
    md.append("`job start -> Popen` covers srun + container start + `pip install -e servekit` + (servekit-eval only) the")
    md.append("wrapper's own cold import of torch+sglang through sitecustomize; serial staging time is subtracted.")
    with open(a.md, "w") as f:
        f.write("\n".join(md) + "\n")
    print("\n".join(md))
    print(f"\n{len(rows)} launches parsed")


if __name__ == "__main__":
    main()
