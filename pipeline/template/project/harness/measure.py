#!/usr/bin/env python3
"""measure.py: one run record per run, an interleaved schedule, and the small-n analysis.

Implements decision D6 of pipeline/DOCTRINE.md. Standard library only, so it runs inside engine
images and on a laptop. Copy it with the template; do not edit it inside a project (a frozen
harness includes this file).

  record    append one JSON line for a finished run (environment captured automatically)
  schedule  print an interleaved run order: random order inside each pair
  analyze   medians, ranges, ratio of medians, paired ratios, exact sign test, A/A noise floor
  selftest  check the statistics against known values

Examples
  python3 measure.py schedule --arms base,fix --pairs 6 --seed 7
  python3 measure.py record --out results/runs.jsonl --arm fix --pair 3 --label MEASURED \
      --metric total_s=95.0 --cache page=cold --cache compile=miss --engine-commit 94602c9
  python3 measure.py analyze results/runs.jsonl --metric total_s --base base --test fix
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import math
import os
import platform
import random
import shutil
import statistics
import subprocess
import sys

LABELS = ("MEASURED", "PREDICTED", "ESTIMATE")


def _run(cmd: list[str]) -> str:
    try:
        return subprocess.run(cmd, capture_output=True, text=True, timeout=20).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return ""


def environment() -> dict:
    """What a reader needs to compare this run with another one."""
    env = {
        "utc": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        "host": platform.node(),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "cpus": os.cpu_count(),
        "image_digest": os.environ.get("IMAGE_DIGEST", ""),
        "harness_commit": _run(["git", "rev-parse", "--short", "HEAD"]),
        "gpus": [],
    }
    if shutil.which("nvidia-smi"):
        out = _run(["nvidia-smi", "--query-gpu=name,uuid,driver_version,memory.total", "--format=csv,noheader"])
        for line in out.splitlines():
            name, uuid, driver, mem = (x.strip() for x in line.split(",")[:4])
            env["gpus"].append({"name": name, "uuid": uuid, "driver": driver, "memory": mem})
    return env


def _pairs(items: list[str], what: str, cast=str) -> dict:
    out = {}
    for item in items or []:
        if "=" not in item:
            sys.exit(f"--{what} wants KEY=VALUE, got {item!r}")
        key, value = item.split("=", 1)
        out[key] = cast(value)
    return out


def cmd_record(a: argparse.Namespace) -> int:
    if a.label not in LABELS:
        sys.exit(f"--label must be one of {', '.join(LABELS)}")
    rec = {
        "run_id": a.run_id or dt.datetime.now(dt.timezone.utc).strftime("run-%Y%m%dT%H%M%SZ"),
        "arm": a.arm,
        "pair": a.pair,
        "label": a.label,
        "status": a.status,
        "metrics": _pairs(a.metric, "metric", float),
        "cache": _pairs(a.cache, "cache"),
        "engine_commit": a.engine_commit,
        "model_revision": a.model_revision,
        "note": a.note,
        "env": environment(),
    }
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    line = json.dumps(rec, sort_keys=True) + "\n"
    fd = os.open(a.out, os.O_WRONLY | os.O_APPEND | os.O_CREAT, 0o644)
    try:
        os.write(fd, line.encode())  # one write call: a record is never half there
        os.fsync(fd)
    finally:
        os.close(fd)
    print(f"recorded {rec['run_id']} arm={a.arm} pair={a.pair} -> {a.out}")
    return 0


def schedule(arms: list[str], pairs: int, seed: int) -> list[tuple[int, str]]:
    rng = random.Random(seed)
    order = []
    for p in range(1, pairs + 1):
        block = arms[:]
        rng.shuffle(block)
        order += [(p, arm) for arm in block]
    return order


def cmd_schedule(a: argparse.Namespace) -> int:
    arms = [x for x in a.arms.split(",") if x]
    if len(arms) < 2:
        sys.exit("--arms needs at least two names; for an A/A pilot use e.g. --arms base,base2")
    for p, arm in schedule(arms, a.pairs, a.seed):
        print(f"pair {p}\t{arm}")
    print(f"# {a.pairs} pairs, seed {a.seed}. Every run is a fresh process; a pair shares one container or host.",
          file=sys.stderr)
    return 0


def sign_test_p(wins: int, losses: int) -> float:
    """Exact two-sided sign test; ties are dropped by the caller."""
    n = wins + losses
    if n == 0:
        return 1.0
    k = min(wins, losses)
    tail = sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n
    return min(1.0, 2 * tail)


def range_covers_median(n: int) -> float:
    """Chance that min..max of n independent runs contains the true median."""
    return 1 - 2 * 0.5 ** n


def load(path: str, metric: str) -> dict[str, dict[int, float]]:
    arms: dict[str, dict[int, float]] = {}
    with open(path, encoding="utf-8") as fh:
        for n, line in enumerate(fh, 1):
            if not line.strip():
                continue
            rec = json.loads(line)
            if rec.get("status", "ok") != "ok" or rec.get("label", "MEASURED") != "MEASURED":
                continue
            if metric not in rec.get("metrics", {}):
                continue
            pair = rec.get("pair")
            slot = arms.setdefault(rec["arm"], {})
            slot[pair if pair is not None else -n] = rec["metrics"][metric]
    return arms


def describe(name: str, values: list[float]) -> str:
    vals = ", ".join(f"{v:g}" for v in sorted(values))
    return (f"{name}: n={len(values)} median={statistics.median(values):g} "
            f"min={min(values):g} max={max(values):g} values=[{vals}]")


def cmd_analyze(a: argparse.Namespace) -> int:
    arms = load(a.path, a.metric)
    for need in (a.base, a.test):
        if need not in arms:
            sys.exit(f"no MEASURED ok runs for arm {need!r} and metric {a.metric!r}; arms seen: {sorted(arms)}")
    base, test = arms[a.base], arms[a.test]
    print(describe(a.base, list(base.values())))
    print(describe(a.test, list(test.values())))
    ratio = statistics.median(test.values()) / statistics.median(base.values())
    print(f"ratio of medians ({a.test}/{a.base}) = {ratio:.3f}")
    shared = sorted(p for p in base if p in test and p is not None and p > 0)
    if shared:
        ratios = [test[p] / base[p] for p in shared]
        wins = sum(r < 1 for r in ratios)
        losses = sum(r > 1 for r in ratios)
        print(f"paired ratios over {len(shared)} pairs: min={min(ratios):.3f} max={max(ratios):.3f}; "
              f"{a.test} lower in {wins}, higher in {losses}; sign test p={sign_test_p(wins, losses):.4f}")
    if a.aa and a.aa in arms:
        twin = arms[a.aa]
        both = sorted(p for p in base if p in twin and p is not None and p > 0)
        if both:
            aa = [twin[p] / base[p] for p in both]
            print(f"A/A noise floor ({a.aa}/{a.base}) over {len(both)} pairs: min={min(aa):.3f} max={max(aa):.3f}")
    n = min(len(base), len(test))
    if n < 6 or len(shared) < 6:
        print(f"verdict: DESCRIPTIVE ONLY. Fewer than 6 pairs; min..max of {n} runs contains the true median "
              f"{range_covers_median(n):.0%} of the time. Make no claim that one arm beats the other.")
    else:
        print("verdict: a comparative claim is allowed if the paired ratios lie outside the A/A range "
              "and the sign test p is below 0.05. State n, setup and date with it.")
    return 0


def cmd_selftest(_: argparse.Namespace) -> int:
    checks = [
        (abs(sign_test_p(6, 0) - 0.03125) < 1e-12, "6 of 6 pairs gives p = 0.03125"),
        (abs(sign_test_p(5, 0) - 0.0625) < 1e-12, "5 of 5 pairs gives p = 0.0625"),
        (abs(sign_test_p(3, 3) - 1.0) < 1e-12, "3 against 3 gives p = 1"),
        (abs(range_covers_median(3) - 0.75) < 1e-12, "min..max of 3 runs covers the median 75% of the time"),
        (abs(range_covers_median(6) - 0.96875) < 1e-12, "min..max of 6 runs covers it 96.9% of the time"),
        (len(schedule(["a", "b"], 6, 1)) == 12, "6 pairs of 2 arms are 12 runs"),
        (all({arm for p, arm in schedule(["a", "b"], 6, 1) if p == k} == {"a", "b"} for k in range(1, 7)),
         "each pair holds every arm once"),
        (schedule(["a", "b"], 6, 1) == schedule(["a", "b"], 6, 1), "a seed fixes the order"),
    ]
    for ok, text in checks:
        print(("ok   " if ok else "FAIL ") + text)
    return 0 if all(ok for ok, _ in checks) else 1


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    r = sub.add_parser("record")
    r.add_argument("--out", required=True)
    r.add_argument("--arm", required=True)
    r.add_argument("--pair", type=int)
    r.add_argument("--run-id")
    r.add_argument("--label", default="MEASURED")
    r.add_argument("--status", default="ok")
    r.add_argument("--metric", action="append")
    r.add_argument("--cache", action="append")
    r.add_argument("--engine-commit", default="")
    r.add_argument("--model-revision", default="")
    r.add_argument("--note", default="")
    r.set_defaults(fn=cmd_record)

    s = sub.add_parser("schedule")
    s.add_argument("--arms", required=True)
    s.add_argument("--pairs", type=int, default=6)
    s.add_argument("--seed", type=int, default=1)
    s.set_defaults(fn=cmd_schedule)

    an = sub.add_parser("analyze")
    an.add_argument("path")
    an.add_argument("--metric", required=True)
    an.add_argument("--base", required=True)
    an.add_argument("--test", required=True)
    an.add_argument("--aa", help="name of the second baseline arm of an A/A pilot")
    an.set_defaults(fn=cmd_analyze)

    sub.add_parser("selftest").set_defaults(fn=cmd_selftest)
    args = p.parse_args()
    return args.fn(args)


if __name__ == "__main__":
    sys.exit(main())
