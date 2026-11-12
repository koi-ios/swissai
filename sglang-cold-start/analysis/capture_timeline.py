"""Per-size CUDA-graph capture timeline from an SGLang startup log.

SGLang prints a tqdm bar while it captures graphs, largest size first:
    Capturing num tokens (num_tokens=8192 ...):   2%| | 1/58 [00:05<05:24,  5.69s/it]
    Capturing batches (bs=256 ...):               3%| | 1/36 [00:12<07:29, 12.84s/it]
The bar's elapsed time is in whole seconds, so per-size times here are +-1 s; the
cumulative curve is what matters.

Usage:
    python capture_timeline.py <sglang.log> [<sglang.log> ...] --out <csv>

Writes one row per (log, phase, size): elapsed seconds after that size finished, the
size's share of the phase, and the time spent on sizes above each threshold.
"""
from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path

ANSI = re.compile(r"\x1b\[[0-9;]*[a-zA-Z]")
BAR = re.compile(
    r"Capturing (?P<kind>num tokens|batches) \((?:num_tokens|bs)=(?P<size>\d+)[^)]*\):\s+\d+%[^|]*\|[^|]*\|\s*"
    r"(?P<done>\d+)/(?P<total>\d+) \[(?P<mm>\d+):(?P<ss>\d+)<"
)
PHASE_END = re.compile(r"Capture (?:target )?(?P<which>prefill|decode|piecewise)? ?CUDA graph end\. (?:elapsed=|Time elapsed: )(?P<s>[\d.]+) s", re.I)


def parse(path: Path):
    text = ANSI.sub("", path.read_text(errors="replace").replace("\r", "\n"))
    # The bar shows "size=N  k/T" both before and after N is captured; the size captured
    # in step k is the one shown when the counter moves from k-1 to k.
    phases = {}
    for line in text.split("\n"):
        m = BAR.search(line)
        if not m:
            continue
        kind = "prefill" if m["kind"] == "num tokens" else "decode"
        done, total = int(m["done"]), int(m["total"])
        elapsed = int(m["mm"]) * 60 + int(m["ss"])
        ph = phases.setdefault(kind, {"total": total, "shown": {}, "elapsed": {}})
        # last label shown at count k = the size captured in step k+1
        ph["shown"][done] = int(m["size"])
        ph["elapsed"].setdefault(done, elapsed)
    reported = {}
    for m in PHASE_END.finditer(text):
        which = (m["which"] or "decode").lower()
        which = "prefill" if which in ("prefill", "piecewise") else "decode"
        reported.setdefault(which, float(m["s"]))
    out = []
    for kind, ph in phases.items():
        prev = 0
        for k in range(1, ph["total"] + 1):
            if k not in ph["elapsed"] or (k - 1) not in ph["shown"]:
                continue
            size = ph["shown"][k - 1]
            el = ph["elapsed"][k]
            out.append({"log": path.name, "phase": kind, "k": k, "size": size, "elapsed_s": el,
                        "step_s": el - prev, "reported_phase_s": reported.get(kind)})
            prev = el
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("logs", nargs="+")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    rows = []
    for p in a.logs:
        rows += parse(Path(p))
    with open(a.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["log", "phase", "k", "size", "elapsed_s", "step_s", "reported_phase_s"])
        w.writeheader()
        w.writerows(rows)
    # Summary: time spent on sizes above a ceiling (captured first, so it is the elapsed
    # time at the step where the size first drops to <= ceiling).
    by = {}
    for r in rows:
        by.setdefault((r["log"], r["phase"]), []).append(r)
    # Sizes are captured largest first, so step 1 also carries one-off costs (JIT, first
    # forward, allocator warmup). Those move to the new first size under a lower ceiling,
    # so the saving of a ceiling counts only steps 2..n above it.
    for (log, phase), rs in by.items():
        rs.sort(key=lambda r: r["k"])
        total = rs[-1]["elapsed_s"]
        n = len(rs)
        first = rs[0]["step_s"]
        print(f"{log} [{phase}] sizes={n} bar_total={total}s reported={rs[0]['reported_phase_s']}s "
              f"step1(size {rs[0]['size']})={first}s")
        ceilings = [256, 512, 1024, 2048, 4096] if phase == "prefill" else [8, 16, 32, 64, 128]
        for c in ceilings:
            saved = sum(r["step_s"] for r in rs[1:] if r["size"] > c)
            kept = sum(1 for r in rs if r["size"] <= c)
            print(f"   ceiling {c:>5}: keep {kept:>2}/{n} sizes, saving (excl. step 1) ~ {saved:>3d}s of {total}s")


if __name__ == "__main__":
    main()
