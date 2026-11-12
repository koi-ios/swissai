"""Read one SGLang launch for the JIT-cache A/B: the phases it compares, and
which compiled output the job left behind in its $HOME and temp dir.

Used inside the Modal container (run_arm.py) and locally (summarize_modal.py,
tests). Standard library only.
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Dict, Optional, Tuple

ANSI = re.compile(r"\x1b\[[0-9;]*[a-zA-Z]")

# SGLang captures prefill graphs largest size first, under a tqdm bar. At 1/N the
# bar's rate is the first size's own time, to 0.01 s; its elapsed counter has
# whole seconds only. The first size carries the one-off costs: JIT, the first
# forward pass, allocator warmup (sglang-cold-start/analysis/capture_timeline.py).
FIRST_PREFILL = re.compile(
    r"Capturing num tokens \(num_tokens=(?P<size>\d+)[^)]*\):[^|]*\|[^|]*\|\s*1/\d+ "
    r"\[[^,\]]*,\s*(?P<rate>[\d.]+)(?P<unit>s/it|it/s)\]"
)
DIST_INIT = re.compile(r"Init torch distributed ends\. elapsed=([\d.]+) s")
# custom_all_reduce_v2.py:292 logs its config only once it is enabled.
AR_ENABLED = re.compile(r"All Reduce config:")
AR_DISABLED = re.compile(r"CustomAllReduce\w* is disabled[^\n]*")
SGLANG_VERSION = re.compile(r"sglang(?:==| version[: ]+)v?(\d+\.\d+\.\d+\S*)", re.I)


def clean(text: str) -> str:
    return ANSI.sub("", text.replace("\r", "\n"))


def first_prefill_step(text: str) -> Optional[Tuple[int, float]]:
    """(size, seconds) of the first prefill graph captured, or None if none was."""
    m = FIRST_PREFILL.search(clean(text))
    if m is None:
        return None
    rate = float(m["rate"])
    seconds = rate if m["unit"] == "s/it" else (1.0 / rate if rate else 0.0)
    return int(m["size"]), round(seconds, 2)


def dist_init(text: str) -> Optional[float]:
    """torch.distributed init, slowest rank (each TP rank logs its own)."""
    values = [float(v) for v in DIST_INIT.findall(text)]
    return max(values) if values else None


def custom_all_reduce(text: str) -> Optional[bool]:
    """Whether sglang's custom all-reduce v2 came up; None if the log says neither."""
    if AR_ENABLED.search(text):
        return True
    if AR_DISABLED.search(text):
        return False
    return None


def phases(report: dict) -> Dict[str, float]:
    """servekit's phase table, by name; a phase reported twice keeps the larger."""
    out: Dict[str, float] = {}
    for phase in report.get("phases", []):
        out[phase["name"]] = max(out.get(phase["name"], 0.0), float(phase["duration_s"]))
    return out


def _bytes(path: Path) -> Tuple[int, int]:
    n = size = 0
    for p in path.rglob("*") if path.is_dir() else [path]:
        if p.is_file() and not p.is_symlink():
            n += 1
            size += p.stat().st_size
    return n, size


def inventory(home: Path, tmp: Path) -> Dict[str, Dict[str, int]]:
    """Every compiled-output directory a job left in its own $HOME and temp dir.

    Both start empty for each job, so everything here is what the job wrote and
    the next job on another node will not have: ~/.cache two levels deep
    (~/.cache/sglang/jit apart from ~/.cache/sglang/triton), the other cache
    roots and the temp dir one level deep.
    """
    out: Dict[str, Dict[str, int]] = {}

    def add(label: str, path: Path) -> None:
        n, size = _bytes(path)
        if size:
            out[label] = {"files": n, "bytes": size}

    cache = home / ".cache"
    if cache.is_dir():
        for top in sorted(cache.iterdir()):
            subs = sorted(top.iterdir()) if top.is_dir() else []
            if subs and all(s.is_dir() for s in subs):
                for sub in subs:
                    add(f"~/.cache/{top.name}/{sub.name}", sub)
            else:
                add(f"~/.cache/{top.name}", top)
    for name in (".triton", ".nv", ".tilelang"):
        if (home / name).is_dir():
            for top in sorted((home / name).iterdir()):
                add(f"~/{name}/{top.name}", top)
    if tmp.is_dir():
        for top in sorted(tmp.iterdir()):
            add(f"$TMPDIR/{top.name}", top)
    return out


def read_run(run_dir: Path) -> dict:
    """Everything the summary needs from one run directory written by run_arm.py."""
    run_dir = Path(run_dir)
    meta = json.loads((run_dir / "result.json").read_text())
    log = (run_dir / "engine.log").read_text(errors="replace") if (run_dir / "engine.log").is_file() else ""
    report = json.loads((run_dir / "report.json").read_text()) if (run_dir / "report.json").is_file() else {}
    first = first_prefill_step(log)
    table = phases(report)
    return {
        **meta,
        "total_s": report.get("total_duration_s"),
        "phases": table,
        "dist_init_s": dist_init(log) or table.get("torch_distributed_init"),
        "first_prefill_size": first[0] if first else None,
        "first_prefill_step_s": first[1] if first else None,
        "first_request_s": table.get("warmup_request(JIT)"),
        "custom_all_reduce": custom_all_reduce(log),
        "jit_cache": report.get("jit_cache"),
    }
