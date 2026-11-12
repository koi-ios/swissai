#!/usr/bin/env python3
"""Summarize e2e_*.csv: median (min-max) seconds per binary x scenario x delay.

Usage: python3 -I summarize.py [results_dir]   (prints a Markdown table)
"""
import csv
import statistics
import sys
from collections import defaultdict
from pathlib import Path


def fmt(xs):
    if not xs:
        return "n/a"
    if len(xs) == 1:
        return f"{xs[0]:.1f}"
    return f"{statistics.median(xs):.1f} ({min(xs):.1f}-{max(xs):.1f})"


def main():
    root = Path(sys.argv[1] if len(sys.argv) > 1 else Path(__file__).parent)
    cells = defaultdict(lambda: {"ttr": [], "stuck": 0, "n": 0, "post": [], "guard": [], "self": set()})
    for path in sorted(root.glob("e2e_*.csv")):
        scenario = path.stem.split("_")[1]
        with open(path, newline="") as f:
            for row in csv.DictReader(f):
                key = (row["binary"], scenario, int(row["history"]), int(row["delay_s"]))
                c = cells[key]
                c["n"] += 1
                if row["outcome"] == "routable":
                    c["ttr"].append(float(row["time_to_routable_s"]))
                elif row["outcome"].startswith("not routable"):
                    c["stuck"] += 1
                if row.get("after_post_s"):
                    c["post"].append(float(row["after_post_s"]))
                if row.get("guard_s"):
                    c["guard"].append(float(row["guard_s"]))
                if row.get("worker_self_status"):
                    c["self"].add(f'{row["worker_self_status"]}/{row["worker_self_services"]} svc')

    print("| binary | scenario | history | restart delay (s) | runs | routable after restart, s: median (min-max) "
          "| not routable within timeout | worker's own view | POST workaround, s | guard fired, s |")
    print("|---|---|---|---|---|---|---|---|---|---|")
    order = {"base": 0, "fixed": 1, "rejoin": 2}
    for key in sorted(cells, key=lambda k: (k[1], k[2], k[3], order.get(k[0], 9))):
        b, scenario, history, delay = key
        c = cells[key]
        print(f"| {b} | {scenario} | {history} | {delay} | {c['n']} | {fmt(c['ttr'])} | {c['stuck']}/{c['n']} "
              f"| {', '.join(sorted(c['self'])) or 'n/a'} | {fmt(c['post'])} | {fmt(c['guard'])} |")


if __name__ == "__main__":
    main()
