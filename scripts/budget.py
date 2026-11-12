#!/usr/bin/env python3
"""Budget gate and spend ledger (decision D10). A person launches paid runs; this says whether one may.

  budget.py gate --project 02 --cost 6        exit 1 if the run would pass a cap   (make budget)
  budget.py spend --project 02 --step "gpu_check" --gpu "H100!" --seconds 900 \
                  --planned 6 --actual 5.40 --app-id ap-xyz [--note ...]            (make spend)
  budget.py summary

Caps come from pipeline/workspace.toml. The ledger is budget/ledger.csv, append-only.
The dashboard spend limit on the provider's side is the second stop; this is the first.
"""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import sys

from lib import ROOT, find_project, money, read_csv, workspace

LEDGER = ROOT / "budget/ledger.csv"
FIELDS = ["date", "project", "step", "gpu", "seconds", "planned_usd", "actual_usd", "app_id", "note"]


def state(project: str | None = None) -> dict:
    ledger = read_csv(LEDGER)
    b = workspace()["budget"]
    month = dt.date.today().strftime("%Y-%m")
    return {
        "project_spent": money(ledger, "actual_usd", project) if project else 0.0,
        "month_spent": money(ledger, "actual_usd", month=month),
        "total_spent": money(ledger, "actual_usd"),
        "project_cap": b["project_cap_usd"], "month_cap": b["month_usd"], "total_cap": b["total_usd"], "month": month,
    }


def cmd_gate(a: argparse.Namespace) -> int:
    p = find_project(a.project)
    cap = (p["card"].get("budget") or {}).get("cap_usd")
    s = state(p["slug"])
    if cap is not None:
        s["project_cap"] = cap
    checks = [
        ("project", s["project_spent"], s["project_cap"]),
        (f"month {s['month']}", s["month_spent"], s["month_cap"]),
        ("total", s["total_spent"], s["total_cap"]),
    ]
    ok = True
    for name, spent, limit in checks:
        after = spent + a.cost
        verdict = "ok" if after <= limit else "OVER"
        ok &= after <= limit
        print(f"budget: {name:<14} spent ${spent:.2f} + this run ${a.cost:.2f} = ${after:.2f} of ${limit:.2f}  {verdict}")
    if not ok:
        print(f"budget: REFUSED for {p['slug']}. Lower the cost, wait for next month's credit, or change the caps "
              "in pipeline/workspace.toml on purpose.")
        return 1
    print(f"budget: allowed for {p['slug']}. Launch it yourself, then record it with `make spend`.")
    return 0


def cmd_spend(a: argparse.Namespace) -> int:
    p = find_project(a.project)
    new = not LEDGER.exists()
    row = {"date": a.date or dt.date.today().isoformat(), "project": p["slug"], "step": a.step, "gpu": a.gpu,
           "seconds": a.seconds, "planned_usd": a.planned, "actual_usd": a.actual, "app_id": a.app_id, "note": a.note}
    with open(LEDGER, "a", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDS)
        if new:
            w.writeheader()
        w.writerow(row)
    print(f"budget: recorded ${a.actual:.2f} for {p['slug']} ({a.step}); run `make portfolio`")
    return 0


def cmd_summary(_: argparse.Namespace) -> int:
    s = state()
    plan = read_csv(ROOT / "budget/plan.csv")
    print(f"planned ${money(plan, 'est_usd'):.2f} in {len(plan)} steps; spent ${s['total_spent']:.2f} of "
          f"${s['total_cap']:.2f}; this month ({s['month']}) ${s['month_spent']:.2f} of ${s['month_cap']:.2f}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    g = sub.add_parser("gate")
    g.add_argument("--project", required=True)
    g.add_argument("--cost", type=float, required=True)
    g.set_defaults(fn=cmd_gate)
    s = sub.add_parser("spend")
    s.add_argument("--project", required=True)
    s.add_argument("--step", required=True)
    s.add_argument("--gpu", default="")
    s.add_argument("--seconds", type=int, default=0)
    s.add_argument("--planned", type=float, default=0.0)
    s.add_argument("--actual", type=float, required=True)
    s.add_argument("--app-id", default="")
    s.add_argument("--note", default="")
    s.add_argument("--date", default="")
    s.set_defaults(fn=cmd_spend)
    sub.add_parser("summary").set_defaults(fn=cmd_summary)
    a = ap.parse_args()
    return a.fn(a)


if __name__ == "__main__":
    sys.exit(main())
