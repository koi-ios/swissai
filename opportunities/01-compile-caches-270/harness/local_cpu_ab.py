#!/usr/bin/env python3
"""Cold vs warm on CPU with the real libraries the images pin (torch 2.13.0 Inductor,
apache-tvm-ffi 0.1.11, TileLang 0.1.12): 3 cold runs (fresh cache each), then one populate
run and 3 warm runs (same cache, fresh HOME/TMPDIR each, i.e. a new job). Median [min-max]
wall seconds of the whole process and binaries written per run.

    python harness/local_cpu_ab.py --out results/local_cpu_ab.json
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import statistics
import subprocess
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
WORKLOADS = {
    "torch_compile_cpu": "TORCHINDUCTOR_CACHE_DIR",
    "tvm_ffi_jit": "TVM_FFI_CACHE_DIR",
    "tilelang_cpu": "TILELANG_CACHE_DIR",
}
BIN = (".so", ".cubin", ".dylib")


def binaries(d: Path) -> set[str]:
    return {str(p) for p in d.rglob("*") if p.is_file() and (p.suffix in BIN or ".so" in p.name)} if d.exists() else set()


def run(workload: str, cache: Path, base: Path, tag: str) -> dict:
    home, tmp = base / f"home-{tag}", base / f"tmp-{tag}"
    home.mkdir(parents=True)
    tmp.mkdir(parents=True)
    env = dict(os.environ, HOME=str(home), TMPDIR=str(tmp), **{WORKLOADS[workload]: str(cache)})
    before = binaries(cache)
    t0 = time.perf_counter()
    p = subprocess.run([sys.executable, str(ROOT / "tests/workloads" / f"{workload}.py")], env=env,
                       capture_output=True, text=True, timeout=900)
    dt = time.perf_counter() - t0
    return {"tag": tag, "ok": p.returncode == 0, "wall_s": round(dt, 3),
            "binaries_written": len(binaries(cache) - before), "stdout": p.stdout.strip()[-200:]}


def mmm(xs: list[float]) -> str:
    return f"{statistics.median(xs):.2f} [{min(xs):.2f}-{max(xs):.2f}]"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="results/local_cpu_ab.json")
    ap.add_argument("--repeats", type=int, default=3)
    a = ap.parse_args()
    res = {"host": platform.platform(), "python": sys.version.split()[0], "workloads": {}}
    lines = ["| workload (real library) | cold wall s | warm wall s | binaries built cold | rebuilt warm |",
             "|---|---:|---:|---:|---:|"]
    for w in WORKLOADS:
        base = Path(tempfile.mkdtemp(prefix=f"cpu-ab-{w}-"))
        cold = [run(w, base / f"cache-cold{i}", base, f"cold{i}") for i in range(a.repeats)]
        warm_cache = base / "cache-warm"
        populate = run(w, warm_cache, base, "populate")
        warm = [run(w, warm_cache, base, f"warm{i}") for i in range(a.repeats)]
        res["workloads"][w] = {"cold": cold, "populate": populate, "warm": warm}
        assert all(r["ok"] for r in cold + warm + [populate]), (w, [r["stdout"] for r in cold + warm])
        lines.append(f"| {w} | {mmm([r['wall_s'] for r in cold])} | {mmm([r['wall_s'] for r in warm])} | "
                     f"{statistics.median([r['binaries_written'] for r in cold]):g} | "
                     f"{statistics.median([r['binaries_written'] for r in warm]):g} |")
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(res, indent=1))
    md = Path(a.out).with_suffix(".md")
    md.write_text(f"# Local CPU cold vs warm ({res['host']}, Python {res['python']})\n\n"
                  f"Median [min-max] over {a.repeats} runs; whole-process wall seconds incl. Python and "
                  "library import (seconds, not the GPU compile times; the point is the hit/miss behaviour).\n\n"
                  + "\n".join(lines) + "\n")
    print(md.read_text())
    return 0


if __name__ == "__main__":
    sys.exit(main())
