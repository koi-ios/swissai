#!/usr/bin/env python3
"""measure.py: one run record per run, an interleaved schedule, and the small-n analysis.

Implements decision D6 of pipeline/DOCTRINE.md. Standard library only, so it runs inside engine
images and on a laptop. Copy it with the template; do not edit it inside a project (a frozen
harness includes this file).

  record    append one JSON line for a finished run (environment captured automatically)
  schedule  print an interleaved run order: random order inside each pair
  analyze   medians, ranges, ratio of medians, paired ratios, exact sign test, A/A noise floor
  selftest  check the statistics and the analysis against known values

A run belongs to a block (one pilot, one main run, one day's session). A pair number is only
meaningful inside its block, so the A/A pilot and the main run never mix.

Examples
  python3 measure.py schedule --arms base,base2 --pairs 6 --seed 7          # the A/A pilot
  python3 measure.py record --out results/runs.jsonl --block pilot --arm base --pair 1 \
      --metric total_s=95.0 --cache page=cold --engine-commit 94602c9 --exit-code 0
  python3 measure.py analyze results/runs.jsonl --metric total_s --block main \
      --base base --test fix --aa-block pilot --aa base2
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import math
import os
import platform
import random
import shutil
import statistics
import subprocess
import sys
import tempfile

LABELS = ("MEASURED", "PREDICTED", "ESTIMATE")


def _run(cmd: list[str]) -> str:
    try:
        return subprocess.run(cmd, capture_output=True, text=True, timeout=20).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return ""


def harness_sha(explicit: str = "") -> str:
    """Digest of HARNESS.sha256, which lists the frozen harness files; empty if not frozen yet."""
    if explicit:
        return explicit
    here = os.path.dirname(os.path.abspath(__file__))
    for path in (os.path.join(here, "..", "HARNESS.sha256"), "HARNESS.sha256"):
        if os.path.isfile(path):
            with open(path, "rb") as fh:
                return hashlib.sha256(fh.read()).hexdigest()[:16]
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
        "gpus": [],
    }
    if shutil.which("nvidia-smi"):
        out = _run(["nvidia-smi", "--query-gpu=name,uuid,driver_version,memory.total", "--format=csv,noheader"])
        for line in out.splitlines():
            name, uuid, driver, mem = (x.strip() for x in line.split(",")[:4])
            env["gpus"].append({"name": name, "uuid": uuid, "driver": driver, "memory": mem})
    return env


def _pairs(items: list[str] | None, what: str, cast=str) -> dict:
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
    status = a.status or ("ok" if a.exit_code == 0 else "failed")
    rec = {
        "run_id": a.run_id or dt.datetime.now(dt.timezone.utc).strftime("run-%Y%m%dT%H%M%S%fZ"),
        "block": a.block,
        "arm": a.arm,
        "pair": a.pair,
        "label": a.label,
        "status": status,
        "exit_code": a.exit_code,
        "metrics": _pairs(a.metric, "metric", float),
        "phases": _pairs(a.phase, "phase", float),
        "cache": _pairs(a.cache, "cache"),
        "engine_commit": a.engine_commit,
        "model_revision": a.model_revision,
        "harness_sha": harness_sha(a.harness_sha),
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
    print(f"recorded {rec['run_id']} block={a.block} arm={a.arm} pair={a.pair} status={status} -> {a.out}")
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
    if len(arms) < 2 or len(set(arms)) != len(arms):
        sys.exit("--arms needs at least two different names; for an A/A pilot use --arms base,base2")
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


def load(path: str, metric: str) -> tuple[dict, dict]:
    """Return {block: {arm: {pair: value}}} and counts of what was left out. Nothing is dropped silently."""
    blocks: dict[str, dict[str, dict]] = {}
    left_out = {"not ok": 0, "not MEASURED": 0, "no such metric": 0}
    dupes = []
    with open(path, encoding="utf-8") as fh:
        for n, line in enumerate(fh, 1):
            if not line.strip():
                continue
            rec = json.loads(line)
            if rec.get("status", "ok") != "ok":
                left_out["not ok"] += 1
                continue
            if rec.get("label", "MEASURED") != "MEASURED":
                left_out["not MEASURED"] += 1
                continue
            if metric not in rec.get("metrics", {}):
                left_out["no such metric"] += 1
                continue
            block, arm = rec.get("block", "main"), rec["arm"]
            pair = rec.get("pair")
            key = pair if pair is not None else f"unpaired-{n}"
            slot = blocks.setdefault(block, {}).setdefault(arm, {})
            if key in slot:
                dupes.append(f"line {n}: block={block} arm={arm} pair={pair} appears twice")
            slot[key] = rec["metrics"][metric]
    if dupes:
        sys.exit("duplicate records; give reruns a new --block or mark the bad run --status failed:\n  "
                 + "\n  ".join(dupes))
    return blocks, left_out


def describe(name: str, values: list[float]) -> str:
    vals = ", ".join(f"{v:g}" for v in sorted(values))
    return (f"{name}: n={len(values)} median={statistics.median(values):g} "
            f"min={min(values):g} max={max(values):g} values=[{vals}]")


def paired_ratios(num: dict, den: dict) -> list[float]:
    shared = sorted(p for p in den if p in num and isinstance(p, int))
    return [num[p] / den[p] for p in shared]


def analyze(blocks: dict, block: str, base: str, test: str, aa_block: str = "", aa: str = "") -> tuple[str, list[str]]:
    """Return (verdict word, report lines). Verdicts: DESCRIPTIVE, NO-PILOT, NOT-DISTINGUISHABLE, SUPPORTED."""
    arms = blocks.get(block, {})
    for need in (base, test):
        if need not in arms:
            sys.exit(f"block {block!r} has no usable runs for arm {need!r}; arms there: {sorted(arms)}")
    b, t = arms[base], arms[test]
    out = [describe(base, list(b.values())), describe(test, list(t.values()))]
    ratio = statistics.median(t.values()) / statistics.median(b.values())
    out.append(f"ratio of medians ({test}/{base}) = {ratio:.3f}")
    ratios = paired_ratios(t, b)
    wins, losses = sum(r < 1 for r in ratios), sum(r > 1 for r in ratios)
    p = sign_test_p(wins, losses)
    if ratios:
        out.append(f"paired ratios over {len(ratios)} pairs: min={min(ratios):.3f} max={max(ratios):.3f}; "
                   f"{test} lower in {wins}, higher in {losses}, tied in {len(ratios) - wins - losses}; "
                   f"sign test p={p:.4f}")
    floor = None
    if aa_block and aa:
        pilot = blocks.get(aa_block, {})
        if base in pilot and aa in pilot:
            noise = paired_ratios(pilot[aa], pilot[base])
            if noise:
                floor = (min(noise), max(noise))
                out.append(f"A/A noise floor ({aa}/{base} in block {aa_block}) over {len(noise)} pairs: "
                           f"min={floor[0]:.3f} max={floor[1]:.3f}")
        if floor is None:
            out.append(f"A/A pilot not found: block {aa_block!r} needs paired runs of {base!r} and {aa!r}")
    untied = wins + losses
    if untied < 6:
        n = min(len(b), len(t))
        return "DESCRIPTIVE", out + [
            f"verdict: DESCRIPTIVE ONLY. {untied} untied pairs, fewer than 6; min..max of {n} runs contains "
            f"the true median {range_covers_median(n):.0%} of the time. Make no claim that one arm beats the other."]
    if floor is None:
        return "NO-PILOT", out + [
            "verdict: NO COMPARATIVE CLAIM. There is no A/A pilot to compare with; run one and pass "
            "--aa-block and --aa."]
    inside = sum(floor[0] <= r <= floor[1] for r in ratios)
    if p >= 0.05 or floor[0] <= ratio <= floor[1]:
        why = f"sign test p={p:.4f}" if p >= 0.05 else "the ratio of medians lies inside the A/A range"
        return "NOT-DISTINGUISHABLE", out + [
            f"verdict: NOT DISTINGUISHABLE from noise ({why}). Report it as such, or go to 10 pairs once."]
    note = f" Caution: {inside} of {len(ratios)} paired ratios lie inside the A/A range." if inside else ""
    return "SUPPORTED", out + [
        f"verdict: COMPARATIVE CLAIM SUPPORTED. Sign test p={p:.4f}; the ratio of medians {ratio:.3f} is outside "
        f"the A/A range {floor[0]:.3f} to {floor[1]:.3f}.{note} State n, setup and date with it."]


def cmd_analyze(a: argparse.Namespace) -> int:
    blocks, left_out = load(a.path, a.metric)
    block = a.block
    if not block:
        candidates = [b for b in blocks if b != a.aa_block]
        if len(candidates) != 1:
            sys.exit(f"the file has blocks {sorted(blocks)}; name one with --block")
        block = candidates[0]
    _, lines = analyze(blocks, block, a.base, a.test, a.aa_block, a.aa)
    print(f"block {block}; left out: " + ", ".join(f"{v} {k}" for k, v in left_out.items()))
    print("\n".join(lines))
    return 0


def _selftest_analysis() -> list[tuple[bool, str]]:
    def rec(block, arm, pair, value, status="ok"):
        return json.dumps({"block": block, "arm": arm, "pair": pair, "status": status,
                           "label": "MEASURED", "metrics": {"t": value}})

    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "runs.jsonl")
        rows = []
        for p in range(1, 7):  # pilot: about 100 s; main run on another day: about 200 s
            rows += [rec("pilot", "base", p, 100 + p), rec("pilot", "base2", p, 101 + p)]
            rows += [rec("main", "base", p, 200 + p), rec("main", "fix", p, 160 + p)]
        rows.append(rec("main", "fix", 7, 999, status="failed"))
        with open(path, "w") as fh:
            fh.write("\n".join(rows) + "\n")
        blocks, left = load(path, "t")
        verdict, lines = analyze(blocks, "main", "base", "fix", "pilot", "base2")
        floor = next(ln for ln in lines if ln.startswith("A/A"))
        checks = [
            (verdict == "SUPPORTED", "six pairs all one way, outside the noise floor: claim supported"),
            ("min=1.009 max=1.010" in floor, "the noise floor uses the pilot's own baseline, not the main run's"),
            (left["not ok"] == 1, "a failed run is counted as left out, not ignored"),
            (analyze(blocks, "main", "base", "fix")[0] == "NO-PILOT", "no pilot given: no comparative claim"),
        ]
        blocks["main"]["fix"][6] = 300.0  # five of six agree
        checks.append((analyze(blocks, "main", "base", "fix", "pilot", "base2")[0] == "NOT-DISTINGUISHABLE",
                       "five of six pairs (p = 0.22): not distinguishable"))
        three = {"main": {arm: {p: v for p, v in vals.items() if p <= 3} for arm, vals in blocks["main"].items()}}
        checks.append((analyze(three, "main", "base", "fix")[0] == "DESCRIPTIVE", "three pairs: descriptive only"))
        with open(path, "a") as fh:
            fh.write(rec("main", "base", 1, 250) + "\n")
        try:
            load(path, "t")
            checks.append((False, "a duplicate (block, arm, pair) is refused"))
        except SystemExit:
            checks.append((True, "a duplicate (block, arm, pair) is refused"))
    return checks


def cmd_selftest(_: argparse.Namespace) -> int:
    checks = [
        (abs(sign_test_p(6, 0) - 0.03125) < 1e-12, "6 of 6 pairs gives p = 0.03125"),
        (abs(sign_test_p(5, 0) - 0.0625) < 1e-12, "5 of 5 pairs gives p = 0.0625"),
        (abs(sign_test_p(5, 1) - 0.21875) < 1e-12, "5 of 6 pairs gives p = 0.21875"),
        (abs(sign_test_p(3, 3) - 1.0) < 1e-12, "3 against 3 gives p = 1"),
        (abs(range_covers_median(3) - 0.75) < 1e-12, "min..max of 3 runs covers the median 75% of the time"),
        (abs(range_covers_median(6) - 0.96875) < 1e-12, "min..max of 6 runs covers it 96.9% of the time"),
        (len(schedule(["a", "b"], 6, 1)) == 12, "6 pairs of 2 arms are 12 runs"),
        (all({arm for p, arm in schedule(["a", "b"], 6, 1) if p == k} == {"a", "b"} for k in range(1, 7)),
         "each pair holds every arm once"),
        (schedule(["a", "b"], 6, 1) == schedule(["a", "b"], 6, 1), "a seed fixes the order"),
    ] + _selftest_analysis()
    for ok, text in checks:
        print(("ok   " if ok else "FAIL ") + text)
    return 0 if all(ok for ok, _ in checks) else 1


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    r = sub.add_parser("record")
    r.add_argument("--out", required=True)
    r.add_argument("--block", default="main", help="pilot, main, or any name; pairs are numbered inside a block")
    r.add_argument("--arm", required=True)
    r.add_argument("--pair", type=int)
    r.add_argument("--run-id")
    r.add_argument("--label", default="MEASURED")
    r.add_argument("--status", help="ok or failed; default follows --exit-code")
    r.add_argument("--exit-code", type=int, default=0)
    r.add_argument("--metric", action="append", help="NAME=number, repeatable")
    r.add_argument("--phase", action="append", help="NAME=seconds, repeatable")
    r.add_argument("--cache", action="append", help="NAME=state, e.g. page=cold, repeatable")
    r.add_argument("--engine-commit", default="")
    r.add_argument("--model-revision", default="")
    r.add_argument("--harness-sha", default="", help="default: digest of HARNESS.sha256 if the harness is frozen")
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
    an.add_argument("--block", default="", help="the block holding the comparison")
    an.add_argument("--base", required=True)
    an.add_argument("--test", required=True)
    an.add_argument("--aa-block", default="", help="the block holding the A/A pilot")
    an.add_argument("--aa", default="", help="the second baseline arm of the pilot")
    an.set_defaults(fn=cmd_analyze)

    sub.add_parser("selftest").set_defaults(fn=cmd_selftest)
    args = p.parse_args()
    return args.fn(args)


if __name__ == "__main__":
    sys.exit(main())
