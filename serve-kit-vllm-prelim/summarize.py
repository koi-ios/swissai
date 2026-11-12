#!/usr/bin/env python3
"""Turn results/*.csv into results/summary.md, and README's results block with --readme.

Every number is a median over repeats, with [min, max] and n; cells with fewer
than 3 runs are marked. Nothing is typed in by hand, so nothing in the README or
the email can claim more than the CSV files show.

    python summarize.py            # print and write results/summary.md
    python summarize.py --readme   # also replace the block between the results markers in README.md
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
import os
import re
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

RESULTS = Path(os.environ.get("RESULTS", "results"))
README = Path(__file__).resolve().parent / "README.md"
MARK_START, MARK_END = "<!-- results:start -->", "<!-- results:end -->"
RUN_ORDER = ["cold", "warm_weights", "warm"]
NATIVE_ARMS = ["lazy_net", "eager_net", "prefetch8_net", "prefetch32_net", "instanttensor_net",
               "fastsafetensors_net", "runai_net", "runai_sharded_net"]
ARM_NOTES = {
    "default_net": "vLLM default loader, network volume (reference)",
    "default_local": "vLLM default loader, local NVMe (control)",
    "lazy_net": "--safetensors-load-strategy lazy (plain mmap)",
    "eager_net": "--safetensors-load-strategy eager",
    "prefetch8_net": "prefetch, 8 threads",
    "prefetch32_net": "prefetch, 32 threads",
    "instanttensor_net": "--load-format instanttensor",
    "fastsafetensors_net": "--load-format fastsafetensors",
    "runai_net": "--load-format runai_streamer",
    "sharded_net": "presharded, --load-format sharded_state, network volume",
    "runai_sharded_net": "presharded, --load-format runai_streamer_sharded",
    "staged_shm": "presharded, staged to /dev/shm by stage.py (staging counted), sharded_state",
    "servekit_serial": "servekit launch: its artifact staged to /dev/shm by its sliced stager (staging counted)",
    "servekit_overlap": "servekit launch --overlap: staging hidden behind engine startup (barrier wait counted)",
}


def read_csv(path: Path) -> List[dict]:
    if not path.exists():
        return []
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def stats(vals: List[float]) -> Optional[Tuple[float, float, float, int]]:
    vals = [v for v in vals if v is not None]
    if not vals:
        return None
    return statistics.median(vals), min(vals), max(vals), len(vals)


def cell(vals: List[float], digits: int = 1) -> str:
    s = stats(vals)
    if s is None:
        return "–"
    med, lo, hi, n = s
    text = f"{med:.{digits}f} [{lo:.{digits}f}, {hi:.{digits}f}]"
    return text + (f" (n={n})" if n < 3 else "")


def med(vals: List[float]) -> Optional[float]:
    s = stats(vals)
    return s[0] if s else None


def by_repeat(rows: List[dict], key: str) -> Dict[tuple, Dict[str, Dict[str, float]]]:
    """{(tp, key value): {repeat: {phase: seconds}}}"""
    out: Dict[tuple, Dict[str, Dict[str, float]]] = defaultdict(lambda: defaultdict(dict))
    for r in rows:
        out[(int(r["tp"]), r[key])][r["repeat"]][r["phase"]] = float(r["seconds"])
    return out


def col(reps: Dict[str, Dict[str, float]], *names: str) -> List[float]:
    """One value per repeat: the first of `names` that the repeat has."""
    vals = []
    for phases in reps.values():
        v = next((phases[n] for n in names if n in phases), None)
        if v is not None:
            vals.append(v)
    return vals


def env_facts() -> dict:
    p = RESULTS / "env.json"
    if p.exists():
        try:
            return json.loads(p.read_text())
        except json.JSONDecodeError:
            pass
    return {}


def baseline_section(rows: List[dict], env: dict) -> List[str]:
    if not rows:
        return []
    groups = by_repeat(rows, "run")
    model = rows[0]["model"]
    out = [f"### Day 1: vLLM cold start, {model}",
           "",
           "Seconds, median [min, max] over repeats. *Cold*: weights evicted from the page cache and every "
           "compile cache empty. *Rest* is the total minus the three named phases: imports, process spawn, "
           "distributed init, KV-cache profiling, API server.",
           "",
           "| TP | run | total | weight loading | torch.compile | CUDA graphs | rest | n |",
           "|---|---|---|---|---|---|---|---|"]
    keys = sorted(groups, key=lambda k: (k[0], RUN_ORDER.index(k[1]) if k[1] in RUN_ORDER else 99, k[1]))
    for tp, run in keys:
        reps = groups[(tp, run)]
        rest = []
        for phases in reps.values():
            tot = phases.get("total", phases.get("total_http"))
            if tot is None:
                continue
            named = [phases.get(a, phases.get(b)) for a, b in (("weight_loading", "vllm:model_loading"),
                                                                 ("torch_compile", "vllm:torch_compile_total"),
                                                                 ("cuda_graph_capture", "vllm:cuda_graph_capture"))]
            rest.append(tot - sum(v for v in named if v is not None))
        out.append(f"| {tp} | {run} | {cell(col(reps, 'total', 'total_http'))} | "
                   f"{cell(col(reps, 'weight_loading', 'vllm:model_loading'))} | "
                   f"{cell(col(reps, 'torch_compile', 'vllm:torch_compile_total'))} | "
                   f"{cell(col(reps, 'cuda_graph_capture', 'vllm:cuda_graph_capture'))} | {cell(rest)} | {len(reps)} |")
    out.append("")
    where = ", ".join(x for x in (env.get("gpu_summary"), env.get("net_storage")) if x) or "[GPU, storage]"
    for tp, run in keys:
        if run != "cold":
            continue
        reps = groups[(tp, run)]
        t, w = med(col(reps, "total", "total_http")), med(col(reps, "weight_loading", "vllm:model_loading"))
        c, g = med(col(reps, "torch_compile", "vllm:torch_compile_total")), med(col(reps, "cuda_graph_capture",
                                                                                     "vllm:cuda_graph_capture"))
        if None in (t, w, c, g):
            continue
        out.append(f"> On {where}, {model}'s vLLM cold start at TP {tp} takes {t:.0f} s: {w:.0f} s weights, "
                   f"{c:.0f} s compilation, {g:.0f} s CUDA graphs (median of {len(reps)}).")
    out.append("")
    return out


def verify_by_arm(vrows: List[dict]) -> Dict[str, str]:
    out: Dict[str, str] = {}
    for r in vrows:  # later rows win
        arm = r.get("arm") or r.get("run")
        if r["mode"] == "record":
            if arm not in out:
                out[arm] = "reference" if r["exit_code"] == "0" else "record FAILED"
            continue
        passed = r["passed"] in ("True", "true", "1")
        delta = r.get("worst_token_delta") or ""
        detail = f" (max abs. logprob diff {float(delta):.1e})" if delta not in ("", "None") else ""
        verdict = "pass" if passed else "**FAIL**"
        out[arm] = (f"reference; self-check {verdict}" if arm == "default_net" else verdict + detail)
    return out


def arms_section(rows: List[dict], vrows: List[dict], env: dict) -> List[str]:
    if not rows:
        return []
    groups = by_repeat(rows, "arm")
    verdicts = verify_by_arm(vrows)
    table = []
    for (tp, arm), reps in groups.items():
        weights = []
        for phases in reps.values():
            w = phases.get("weight_loading", phases.get("vllm:model_loading"))
            if w is not None:
                # the harness's own staging, or servekit launch's serial "stage" phase
                weights.append(w + phases.get("staging", phases.get("stage", 0.0)))
        totals = col(reps, "total_with_staging", "total", "total_http")
        table.append((tp, arm, weights, totals, len(reps)))
    ref = {tp: (med(w), med(t)) for tp, arm, w, t, _ in table if arm == "default_net"}
    table.sort(key=lambda r: (r[0], r[1] != "default_net", med(r[3]) or 1e9))
    model = rows[0]["model"]
    out = [f"### Day 2: weight-loading arms, {model}",
           "",
           "Cold runs only. Seconds, median [min, max]. For `staged_shm` the staging copy into /dev/shm is "
           "counted in both weight loading and total. Speedups are against `default_net`'s medians.",
           "",
           "| TP | arm | what it is | weight loading s | total cold start s | speedup (weights) | speedup (total) | verify | n |",
           "|---|---|---|---|---|---|---|---|---|"]
    for tp, arm, weights, totals, n in table:
        rw, rt = ref.get(tp, (None, None))
        mw, mt = med(weights), med(totals)
        sw = f"{rw / mw:.1f}×" if rw and mw else "–"
        stt = f"{rt / mt:.2f}×" if rt and mt else "–"
        out.append(f"| {tp} | `{arm}` | {ARM_NOTES.get(arm, '')} | {cell(weights)} | {cell(totals)} | {sw} | {stt} | "
                   f"{verdicts.get(arm, 'not run')} | {n} |")
    out.append("")
    for tp in sorted(ref):
        a = ref[tp][0]
        staged = next((r for r in table if r[0] == tp and r[1] == "staged_shm"), None)
        natives = [(med(r[2]), r[1]) for r in table if r[0] == tp and r[1] in NATIVE_ARMS and med(r[2])]
        if not staged or not a or not med(staged[2]):
            continue
        b = med(staged[2])
        best = min(natives) if natives else None
        v = verdicts.get("staged_shm", "not run")
        vtext = "servekit verify passes" if v.startswith("pass") else f"servekit verify: {v}"
        where = ", ".join(x for x in (env.get("gpu_summary"), env.get("net_storage")) if x) or "[setup]"
        line = (f"> On {where}, staging presharded weights in RAM cut vLLM's weight loading for {model} at TP {tp} "
                f"from {a:.1f} s to {b:.1f} s ({a / b:.1f}×)")
        line += f", versus {best[0]:.1f} s for vLLM's best built-in option (`{best[1]}`)" if best else ""
        out.append(line + f", and {vtext}.")
    out.append("")
    return out


def staging_section(rows: List[dict]) -> List[str]:
    if not rows:
        return []
    groups: Dict[tuple, List[float]] = defaultdict(list)
    sizes = set()
    for r in rows:
        groups[(int(r["readers"]), int(r["block_mib"]), r["read_mode"])].append(float(r["gbps"]))
        sizes.add(int(r["bytes"]))
    gb = ", ".join(f"{b / 1e9:.2f} GB" for b in sorted(sizes))
    out = ["### Staging bandwidth into /dev/shm (stage.py)",
           "",
           f"Decimal GB/s, median [min, max]; {gb} per staging; source evicted before each.",
           "",
           "| readers | block | read mode | GB/s | n |",
           "|---|---|---|---|---|"]
    for (n, b, mode), vals in sorted(groups.items(), key=lambda kv: (kv[0][2], kv[0][1], kv[0][0])):
        out.append(f"| {n} | {b} MiB | {mode} | {cell(vals, 2)} | {len(vals)} |")
    out.append("")
    return out


def ingest_bench_startup() -> List[dict]:
    """results/bench_startup_tp*.json -> rows, also written to results/bench_startup.csv."""
    rows = []
    for path in sorted(glob.glob(str(RESULTS / "bench_startup_tp*.json"))):
        tp = int(re.search(r"tp(\d+)", path).group(1))
        data = json.loads(Path(path).read_text())
        date = Path(path).stat().st_mtime
        for kind in ("cold", "warm"):
            for metric, phase in (("startup", "total_startup"), ("compilation", "compilation")):
                for i, v in enumerate(data.get(f"{kind}_{metric}_times", []), 1):
                    rows.append({"date": int(date), "tp": tp, "run": f"bench_{kind}", "repeat": i,
                                 "phase": phase, "seconds": float(v)})
    if rows:
        with open(RESULTS / "bench_startup.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["date", "tp", "run", "repeat", "phase", "seconds"])
            w.writeheader()
            w.writerows(rows)
    return rows


def bench_section(rows: List[dict]) -> List[str]:
    if not rows:
        return []
    groups: Dict[tuple, List[float]] = defaultdict(list)
    for r in rows:
        groups[(r["tp"], r["run"], r["phase"])].append(r["seconds"])
    out = ["### Cross-check: `vllm bench startup`",
           "",
           "Offline `LLM()` construction, no API server; weights already in the page cache; its *cold* empties "
           "only VLLM_CACHE_ROOT. Compare with the warm_weights and warm rows above.",
           "",
           "| TP | kind | startup s | compilation s | n |",
           "|---|---|---|---|---|"]
    for tp in sorted({k[0] for k in groups}):
        for kind in ("bench_cold", "bench_warm"):
            tot = groups.get((tp, kind, "total_startup"), [])
            if tot:
                out.append(f"| {tp} | {kind[6:]} | {cell(tot)} | {cell(groups.get((tp, kind, 'compilation'), []))} | "
                           f"{len(tot)} |")
    out.append("")
    return out


def runs_section() -> List[str]:
    p = RESULTS / "runs.jsonl"
    if not p.exists():
        return []
    failed, test_mode, warm_cold = [], 0, []
    for line in p.read_text().splitlines():
        try:
            m = json.loads(line)
        except json.JSONDecodeError:
            continue
        test_mode += bool(m.get("test_mode"))
        if m.get("status") != "ok":
            failed.append(f"- {m.get('date')} `{m.get('arm') or m.get('run')}` TP {m.get('tp')} repeat "
                          f"{m.get('repeat')}: {m.get('reason') or 'failed'}")
        if m.get("evict") and (m.get("resident_pct_after_evict") or 0) > 1.0:
            warm_cold.append(m.get("run_id"))
    out = []
    if test_mode:
        out += [f"**{test_mode} run(s) were made in SK_TEST_MODE (mock engine, no eviction): not measurements.**", ""]
    if warm_cold:
        out += [f"**Runs whose eviction left >1% resident: {', '.join(warm_cold)}.**", ""]
    if failed:
        out += ["### Failed runs (not in the tables)", ""] + failed + [""]
    return out


def setup_line(env: dict) -> List[str]:
    if not env:
        return []
    bits = [env.get("gpu_summary"), f"vLLM {env['vllm']}" if env.get("vllm") else None,
            f"torch {env['torch']}" if env.get("torch") else None, env.get("net_storage")]
    return ["Setup: " + "; ".join(b for b in bits if b) + ". Details in [ENV.md](ENV.md).", ""]


def build_headline() -> str:
    """What README.md opens with: the arms table and the two sentences; the rest stays in summary.md."""
    env = env_facts()
    arms = arms_section(read_csv(RESULTS / "arms.csv"), read_csv(RESULTS / "verify.csv"), env)
    day1 = [ln for ln in baseline_section(read_csv(RESULTS / "baseline.csv"), env) if ln.startswith("> ")]
    if not arms and not day1:
        return "_No results yet: run the steps below, then `python summarize.py --readme`._\n"
    parts = setup_line(env) + arms + day1 + ([""] if day1 else [])
    parts += ["All tables (baseline phases, staging bandwidth, `vllm bench startup`, failed runs): "
              "[results/summary.md](results/summary.md)."]
    return "\n".join(parts).rstrip() + "\n"


def build() -> str:
    env = env_facts()
    parts = ["## Results", ""] + setup_line(env)
    parts += baseline_section(read_csv(RESULTS / "baseline.csv"), env)
    parts += bench_section(ingest_bench_startup())
    parts += staging_section(read_csv(RESULTS / "staging.csv"))
    parts += arms_section(read_csv(RESULTS / "arms.csv"), read_csv(RESULTS / "verify.csv"), env)
    parts += runs_section()
    if len(parts) <= 4:
        parts.append("No results yet.")
    return "\n".join(parts).rstrip() + "\n"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--readme", action="store_true", help="also update README.md between the results markers")
    args = ap.parse_args(argv)
    text = build()
    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / "summary.md").write_text(text)
    print(text)
    if args.readme:
        readme = README.read_text()
        if MARK_START not in readme or MARK_END not in readme:
            raise SystemExit(f"README.md has no {MARK_START} ... {MARK_END} block")
        head, rest = readme.split(MARK_START, 1)
        _, tail = rest.split(MARK_END, 1)
        README.write_text(f"{head}{MARK_START}\n{build_headline()}{MARK_END}{tail}")
        print(f"updated {README}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
