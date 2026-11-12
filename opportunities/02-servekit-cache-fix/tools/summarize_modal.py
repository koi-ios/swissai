"""Per-phase table for the JIT-cache A/B, from the runs.jsonl the Modal app writes.

    python tools/summarize_modal.py results/modal/<stamp>/runs.jsonl [--out results/modal/<stamp>/summary.md]

Median and min-max over each arm's measured runs, in seconds. Runs that failed
are counted and left out of the medians.
"""
from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path
from typing import Dict, List, Optional

ARMS = ("none", "main", "patched")
ARM_LABEL = {
    "none": "no caches",
    "main": "servekit main (cd8f0b8) caches",
    "patched": "servekit + patches/ caches",
}
METRICS = (
    ("dist_init_s", "torch.distributed init (custom all-reduce JIT)"),
    ("first_prefill_step_s", "first prefill-capture step"),
    ("prefill_capture_s", "prefill graph capture, all sizes"),
    ("first_request_s", "first request (warmup)"),
    ("total_s", "total to ready (excl. weight stage)"),
)


def load(path: Path) -> List[dict]:
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def _value(run: dict, metric: str) -> Optional[float]:
    if metric == "prefill_capture_s":
        return run.get("phases", {}).get("prefill_cuda_graph_capture")
    if metric == "total_s":
        total = run.get("total_s")
        stage = run.get("phases", {}).get("stage", 0.0)
        return None if total is None else round(total - stage, 2)
    return run.get(metric)


def cell(values: List[float]) -> str:
    if not values:
        return "n/a"
    if len(values) == 1:
        return f"{values[0]:.1f} (1 run)"
    return f"{statistics.median(values):.1f} [{min(values):.1f}-{max(values):.1f}]"


def table(runs: List[dict]) -> str:
    measured = [r for r in runs if r.get("arm") in ARMS and not r.get("prime") and not r.get("prepare")]
    ok = [r for r in measured if r.get("rc") == 0]
    arms = [a for a in ARMS if any(r["arm"] == a for r in measured)]
    lines = ["| phase (s) | " + " | ".join(ARM_LABEL[a] for a in arms) + " |",
             "|---|" + "---|" * len(arms)]
    for metric, label in METRICS:
        row = []
        for arm in arms:
            vals = [v for v in (_value(r, metric) for r in ok if r["arm"] == arm) if v is not None]
            row.append(cell(vals))
        lines.append(f"| {label} | " + " | ".join(row) + " |")
    counts = {a: (sum(r["arm"] == a for r in ok), sum(r["arm"] == a for r in measured)) for a in arms}
    lines.append("")
    lines.append("Runs ok/attempted: " + ", ".join(f"{a} {k}/{n}" for a, (k, n) in counts.items()))
    ar = {r.get("custom_all_reduce") for r in ok}
    lines.append(f"Custom all-reduce v2 enabled in every run: {ar == {True}} (seen: {sorted(map(str, ar))})")
    return "\n".join(lines)


def left_behind(runs: List[dict]) -> str:
    """What each arm's jobs compiled into their own ephemeral $HOME / temp dir (median bytes)."""
    by_arm: Dict[str, Dict[str, List[int]]] = {}
    for r in runs:
        if r.get("rc") != 0 or r.get("prime"):
            continue
        label = r["arm"] + (" (prepare)" if r.get("prepare") else "")
        for path, v in (r.get("left_behind") or {}).items():
            by_arm.setdefault(label, {}).setdefault(path, []).append(v["bytes"])
    if not by_arm:
        return "Nothing left behind in any job's $HOME or temp dir."
    lines = ["| job | directory | median MB |", "|---|---|---|"]
    for label in sorted(by_arm):
        for path, sizes in sorted(by_arm[label].items(), key=lambda kv: -statistics.median(kv[1])):
            lines.append(f"| {label} | `{path}` | {statistics.median(sizes) / 1e6:.1f} |")
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("runs")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    runs = load(Path(args.runs))
    meta = next((r for r in runs if r.get("kind") == "meta"), {})
    runs = [r for r in runs if r.get("kind") != "meta"]
    text = "\n\n".join([
        f"# JIT-cache A/B: {meta.get('model', '?')}, TP{meta.get('tp', '?')}, {meta.get('image', '?')}",
        f"{meta.get('gpus', '')}".strip(),
        "Median [min-max] over each arm's runs, seconds. Each run is a fresh job: new $HOME, $TMPDIR and "
        "node-local cache dir; only the artifact carries over.",
        table(runs),
        "## Compiled output each job left in its own $HOME or temp dir (lost when the job ends)",
        left_behind(runs),
    ])
    print(text)
    if args.out:
        Path(args.out).write_text(text + "\n")


if __name__ == "__main__":
    main()
