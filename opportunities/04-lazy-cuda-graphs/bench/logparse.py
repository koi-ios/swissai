"""Parse SGLang 0.5.20 startup logs and the lazy-capture controller's lines.

Input: (t, line) pairs, t = seconds since the server process was started
(stamped by the reader thread in arm.py; SGLang's own timestamps have 1 s
resolution).

Lines matched (formats from the v0.5.20 source and the patch):
  model_runner_components/cuda_graph_setup.py:
    "Capture target prefill CUDA graph end. elapsed=27.60 s, mem usage=1.23 GB, ..."
    "Capture target decode CUDA graph end. elapsed=5.50 s, mem usage=0.40 GB, ..."
  scheduler_components/lazy_graph_capture.py (patch):
    "Lazy CUDA graph capture: decode bucket 8 captured in 0.213 s (step 0, 41 buckets pending)."
    "Lazy CUDA graph capture: all planned buckets captured."
  model_executor/runner/lazy_capture_setup.py (patch):
    "Lazy CUDA graph capture [decode]: 3 of 36 buckets at startup [1, 2, 4], 33 deferred."
  http_server: "The server is fired up and ready to roll!"
"""

from __future__ import annotations

import re
from typing import Dict, Iterable, List, Optional, Tuple

ANSI = re.compile(r"\x1b\[[0-9;]*[a-zA-Z]")
PHASE_END = re.compile(
    r"Capture (?:target|draft) (?P<phase>prefill|decode|verify) (?:CUDA )?graph end\. "
    r"elapsed=(?P<s>[\d.]+) s, mem usage=(?P<mem>-?[\d.]+) GB"
)
LAZY_STEP = re.compile(
    r"Lazy CUDA graph capture: (?P<phase>decode|prefill) bucket (?P<size>\d+) "
    r"captured in (?P<s>[\d.]+) s \(step (?P<step>\d+), (?P<pending>\d+) buckets pending\)"
)
LAZY_DONE = re.compile(r"Lazy CUDA graph capture: all planned buckets captured")
LAZY_SETUP = re.compile(
    r"Lazy CUDA graph capture \[(?P<phase>decode|prefill)\]: (?P<init>\d+) of "
    r"(?P<planned>\d+) buckets at startup"
)
LAZY_DISABLED = re.compile(r"Lazy CUDA graph capture disabled for (?P<phase>\w+) \((?P<why>[^)]*)\)")
READY = re.compile(r"The server is fired up and ready to roll")
TQDM_STEP = re.compile(
    r"Capturing (?P<kind>num tokens|batches) \((?:num_tokens|bs)=(?P<size>\d+)"
)


def parse(lines: Iterable[Tuple[float, str]]) -> Dict:
    out: Dict = {
        "startup_capture": {},
        "lazy_setup": {},
        "lazy_disabled": {},
        "lazy_steps": [],
        "lazy_all_captured_t": None,
        "ready_log_t": None,
        "errors": [],
    }
    for t, raw in lines:
        line = ANSI.sub("", raw)
        m = PHASE_END.search(line)
        if m:
            out["startup_capture"][m["phase"]] = {
                "t": t,
                "elapsed_s": float(m["s"]),
                "mem_gb": float(m["mem"]),
            }
            continue
        m = LAZY_STEP.search(line)
        if m:
            out["lazy_steps"].append(
                {
                    "t": t,
                    "phase": m["phase"],
                    "size": int(m["size"]),
                    "seconds": float(m["s"]),
                    "step": int(m["step"]),
                    "pending": int(m["pending"]),
                }
            )
            continue
        if LAZY_DONE.search(line):
            out["lazy_all_captured_t"] = t
            continue
        m = LAZY_SETUP.search(line)
        if m:
            out["lazy_setup"][m["phase"]] = {
                "initial": int(m["init"]),
                "planned": int(m["planned"]),
            }
            continue
        m = LAZY_DISABLED.search(line)
        if m:
            out["lazy_disabled"][m["phase"]] = m["why"]
            continue
        if out["ready_log_t"] is None and READY.search(line):
            out["ready_log_t"] = t
            continue
        if "Traceback" in line or "CUDA out of memory" in line or "Error" in line[:200]:
            if len(out["errors"]) < 20:
                out["errors"].append(line.strip()[:300])
    out["startup_capture_total_s"] = round(
        sum(p["elapsed_s"] for p in out["startup_capture"].values()), 3
    )
    out["lazy_capture_total_s"] = round(sum(s["seconds"] for s in out["lazy_steps"]), 3)
    return out


def captured_by(parsed: Dict, phase: str, t: float) -> List[int]:
    """Buckets of `phase` captured lazily before time t."""
    return [s["size"] for s in parsed["lazy_steps"] if s["phase"] == phase and s["t"] <= t]


def first_step_t(parsed: Dict) -> Optional[float]:
    steps = parsed["lazy_steps"]
    return steps[0]["t"] if steps else None
