#!/usr/bin/env python3
"""Summarize harness/ab.py results: per engine/model/TP and arm, the median [min-max] of
time-to-ready, first request and every servekit-profile phase, the delta against cold, and
the cache components' builds avoided (binaries written by populate vs by warm runs).

    python harness/summarize.py results/ab/*.json > results/ab_summary.md
"""

from __future__ import annotations

import json
import statistics
import sys
from collections import defaultdict


def mmm(xs: list[float]) -> str:
    xs = [x for x in xs if x is not None]
    if not xs:
        return "-"
    return f"{statistics.median(xs):.1f} [{min(xs):.1f}-{max(xs):.1f}]"


def med(xs: list[float]) -> float | None:
    xs = [x for x in xs if x is not None]
    return statistics.median(xs) if xs else None


def ready(r: dict) -> float | None:
    """servekit's total (spawn to the ready line) when present, else the harness wall clock."""
    return (r.get("profile") or {}).get("total_duration_s") or r.get("ready_s")


def phases(r: dict) -> dict[str, float]:
    out: dict[str, float] = {}
    for p in (r.get("profile") or {}).get("phases", []):
        out[p["name"]] = out.get(p["name"], 0.0) + p["duration_s"]
    return out


def main(paths: list[str]) -> int:
    rows = []
    for p in paths:
        rows += json.load(open(p))
    groups: dict[tuple, list[dict]] = defaultdict(list)
    for r in rows:
        groups[(r["engine"], r["model"].rstrip("/").split("/")[-1], r["tp"])].append(r)
    print("# Cold vs warm vs servekit's three variables\n")
    print("Medians over the measured runs with [min-max]; seconds. Warm arms exclude their populate run.")
    print("Persisted arms seed a node-local overlay from a Modal Volume and promote back, as sml would.\n")
    for (engine, model, tp), rs in sorted(groups.items()):
        arms: dict[str, list[dict]] = defaultdict(list)
        populate: dict[str, dict] = {}
        for r in rs:
            if not r.get("ok"):
                continue
            if r["run"].endswith("populate"):
                populate[r["arm"]] = r
            else:
                arms[r["arm"]].append(r)
        print(f"## {engine} · {model} · TP{tp}\n")
        print("| arm | runs | ready | first request | seed | delta ready vs cold |")
        print("|---|---:|---:|---:|---:|---:|")
        cold_ready = med([ready(r) for r in arms.get("cold", [])])
        for arm in ("cold", "servekit3", "full"):
            xs = arms.get(arm, [])
            if not xs:
                continue
            rdy = med([ready(r) for r in xs])
            delta = f"{rdy - cold_ready:+.1f}" if (rdy is not None and cold_ready is not None) else "-"
            seed = mmm([(r.get("seed") or {}).get("seconds") for r in xs])
            print(f"| {arm} | {len(xs)} | {mmm([ready(r) for r in xs])} | "
                  f"{mmm([r['first_request_s'] for r in xs])} | {seed} | {delta} |")
        # per-phase table
        names: list[str] = []
        for xs in arms.values():
            for r in xs:
                for n in phases(r):
                    if n not in names:
                        names.append(n)
        if names:
            print("\n| phase | " + " | ".join(a for a in ("cold", "servekit3", "full") if a in arms) + " | full - cold |")
            print("|---|" + "---:|" * (len([a for a in ("cold", "servekit3", "full") if a in arms]) + 1))
            for n in names:
                vals = {a: [phases(r).get(n) for r in arms[a]] for a in ("cold", "servekit3", "full") if a in arms}
                d = "-"
                if "cold" in vals and "full" in vals and med(vals["cold"]) is not None and med(vals["full"]) is not None:
                    d = f"{med(vals['full']) - med(vals['cold']):+.1f}"
                print(f"| {n} | " + " | ".join(mmm(v) for v in vals.values()) + f" | {d} |")
        # builds avoided
        print("\n| cache component | arm | binaries built by populate | binaries rebuilt per warm run (median) | builds avoided |")
        print("|---|---|---:|---:|---:|")
        for arm in ("servekit3", "full"):
            pop = populate.get(arm, {}).get("written", {})
            warm = arms.get(arm, [])
            comps = sorted(set(pop) | {c for r in warm for c in r.get("written", {})})
            for c in comps:
                b0 = pop.get(c, {}).get("binaries", 0)
                bw = med([r.get("written", {}).get(c, {}).get("binaries", 0) for r in warm]) or 0
                if b0 or bw:
                    print(f"| {c} | {arm} | {b0} | {bw:g} | {b0 - bw:g} |")
        print()
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
