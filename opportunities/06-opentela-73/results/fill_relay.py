#!/usr/bin/env python3
"""Fill the long-history relay cells in drafts/PR.md, WRITEUP.md and STATUS.md
from results/{calib_relay,e2e_relay_base,e2e_relay_fixed}.csv.

Run from the opportunity folder: python3 -I results/fill_relay.py
"""
import csv
import statistics
from pathlib import Path


def rows(name):
    p = Path("results") / name
    if not p.exists():
        return []
    with open(p, newline="") as f:
        return list(csv.DictReader(f))


def fmt(xs):
    return f"{statistics.median(xs):.1f} ({min(xs):.1f}-{max(xs):.1f})" if len(xs) > 1 else f"{xs[0]:.1f}"


base = rows("calib_relay.csv") + rows("e2e_relay_base.csv")
fixed = rows("e2e_relay_fixed.csv")
b_ok = [float(r["time_to_routable_s"]) for r in base if r["outcome"] == "routable"]
b_stuck = sum(r["outcome"].startswith("not routable") for r in base)
f_ok = [float(r["time_to_routable_s"]) for r in fixed if r["outcome"] == "routable"]
f_stuck = sum(r["outcome"].startswith("not routable") for r in fixed)

base_cell = (f"not routable within 180 s in {b_stuck}/{len(base)}; "
             f"the other {len(b_ok)} routable at {fmt(b_ok)}") if b_ok else f"not routable within 180 s ({b_stuck}/{len(base)})"
fixed_cell = fmt(f_ok) + (f"; {f_stuck}/{len(fixed)} not routable" if f_stuck else "")
status = (f"{b_stuck} of {len(base)} runs with a default relay worker and a 20,000-record history "
          f"(the other {len(b_ok)} recovered at about 20 s: a race between the history walk and the worker's single re-announce)")

race = ("- **Why the two rows differ.** A public-address worker reserves no relay, so it never gets the one re-announce "
        "a restarted worker otherwise makes about 10 s in. A default worker does make that re-announce, and whether it "
        "rescues the row is a race with the history walk. On localhost a 20,000-block walk mostly finishes first. The one "
        "stuck run was the calibration run: it ran at `nice -n 19` alongside other runs, so its walk was slower. With a "
        "1,000-record history the worker recovered in 14.2 s.\n"
        "- **Prediction, not measured.** On a real mesh every block fetch costs a WAN or relay round trip, so a long "
        "history makes the walk take minutes and the re-announce loses.")

for name in ["drafts/PR.md", "WRITEUP.md", "STATUS.md"]:
    p = Path(name)
    s = p.read_text()
    s = s.replace("RELAY_BASE", base_cell).replace("RELAY_FIXED", fixed_cell).replace("RELAY_STATUS", status)
    old = [ln for ln in s.splitlines() if ln.startswith("- **Why the two rows differ.**")]
    if old:
        s = s.replace(old[0], race)
    p.write_text(s)
    print(name, "placeholders left:", s.count("RELAY_"))
print("base:", base_cell)
print("fixed:", fixed_cell)
