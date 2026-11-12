"""Summarize Modal results: python summarize.py results/modal [more dirs or results.json files]

Prints, per model run: what each arm did (load, verify vs the default loader,
deep-state check), the artifact sizes, and weight-loading time as
median (min-max) over the timing runs. Writes the same as markdown next to
each results.json (summary.md).
"""
from __future__ import annotations

import json
import statistics
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional


def _fmt(values: List[Optional[float]]) -> str:
    vals = [v for v in values if v is not None]
    if not vals:
        return "-"
    med = statistics.median(vals)
    return f"{med:.2f} ({min(vals):.2f}-{max(vals):.2f}), n={len(vals)}"


def _deep(res: Dict[str, Any], arm: str, label: str) -> str:
    r = res.get("deepstate", {}).get(f"{arm}/{label}")
    if not r:
        return "-"
    if r.get("rc") == 0:
        return "exact"
    summary = r.get("summary", "")
    counts = [l.strip() for l in summary.splitlines() if l.strip().startswith(("missing:", "extra:", "differ:", "alias"))]
    return "differs: " + "; ".join(counts[:3]) if counts else f"rc={r.get('rc')}"


def summarize(res: Dict[str, Any]) -> str:
    lines = [f"## {res['model']} ({res['repo']}), TP={res['tp']}, run {res['run_id']}", ""]
    env = res.get("env", {})
    if env:
        lines += [f"- sglang/torch/cuda: `{env.get('sglang_torch_cuda')}`; GPUs: {', '.join(env.get('gpus', []))}; "
                  f"servekit {env.get('servekit_commit')}", ""]
    ck = res.get("checkpoint_bytes", 0)
    lines += ["| arm | load | verify vs default (1e-6) | deep-state (all tensors) | state_dict-only check | artifact GB | prepare s |",
              "|---|---|---|---|---|---|---|"]
    for arm, a in res.get("arms", {}).items():
        serve = a.get("serve", {})
        load = "ok" if serve.get("ready") else ("prepare failed" if a.get("status") == "prepare_failed" else "FAILED")
        err = (a.get("error") or serve.get("error") or [""])[-1][:90]
        if load != "ok" and err:
            load += f": `{err}`"
        v = a.get("verify", {})
        verify = "recorded" if arm == "default" else ("pass" if v.get("passed") else ("FAIL" if v else "-"))
        gb = f"{a.get('bytes', 0) / 1e9:.2f}" if a.get("bytes") else (f"{ck / 1e9:.2f} (source)" if arm.startswith("default") else "-")
        prep = a.get("prepare", {}).get("seconds", "-")
        lines.append(f"| {arm} | {load} | {verify} | {_deep(res, arm, 'full')} | {_deep(res, arm, 'registered_only')} | {gb} | {prep} |")
    if res.get("timing"):
        lines += ["", "| arm (timing) | weight loading s, median (min-max) | process start to ready s |", "|---|---|---|"]
        for arm, samples in res["timing"].items():
            lines.append(f"| {arm} | {_fmt([s.get('weight_load_s') for s in samples])} | {_fmt([s.get('ready_s') for s in samples])} |")
    raw = res.get("arms", {}).get("raw", {}).get("ranks")
    if raw:
        stored = sum(r["stored_bytes"] for r in raw.values())
        lines += ["", f"- raw preshard: {len(raw)} ranks store {stored / 1e9:.2f} GB in total for a {ck / 1e9:.2f} GB checkpoint; "
                  + ", ".join(f"{k}: {r['tensors_full']} full / {r['tensors_partial']} sliced / {r['tensors_unread']} unread"
                              for k, r in raw.items())]
    return "\n".join(lines) + "\n"


def main(argv: List[str]) -> int:
    paths: List[Path] = []
    for arg in argv or ["results/modal"]:
        p = Path(arg)
        paths += sorted(p.rglob("results.json")) if p.is_dir() else [p]
    if not paths:
        print("no results.json found", file=sys.stderr)
        return 1
    for path in paths:
        text = summarize(json.loads(path.read_text()))
        (path.parent / "summary.md").write_text(text)
        print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
