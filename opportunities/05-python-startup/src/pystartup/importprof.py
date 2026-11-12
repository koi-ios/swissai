"""Parent side: run `pystartup._importprobe` in fresh interpreters under different pyc regimes.

Modes (how the child finds bytecode):
    asis         whatever the image has; if a pyc is missing and the tree is writable the child
                 writes it (so a second `asis` run is warm)
    nopyc        PYTHONPYCACHEPREFIX=<empty dir> + PYTHONDONTWRITEBYTECODE=1: every module is
                 compiled from source and nothing is written (= read-only image without pycs)
    prefix-cold  PYTHONPYCACHEPREFIX=<fresh dir>: compile and write into the prefix (fix (c), first process)
    prefix-warm  same prefix again: load from the prefix (fix (c), every later process)
    prefix:DIR   use an existing prefix, e.g. one filled by `compileall --invalidation-mode
                 unchecked-hash` (fix (a) without touching the image)

Frozen stdlib modules (os, site, codecs, ...) never need a pyc, so they are the same in all
modes. Each run is a separate interpreter; wall time is measured around the whole process.
"""
from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
from typing import Dict, List, Optional

PKG_PARENT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
IMPORTTIME = re.compile(r"^import time:\s+(\d+)\s+\|\s+(\d+)\s+\|(\s*)(\S+)\s*$")


def parse_importtime(stderr: str) -> List[dict]:
    """Parse `-X importtime` lines: self/cumulative microseconds, nesting depth, module."""
    rows = []
    for line in stderr.splitlines():
        m = IMPORTTIME.match(line)
        if not m:
            continue
        rows.append({"self_us": int(m.group(1)), "cum_us": int(m.group(2)),
                     "depth": (len(m.group(3)) - 1) // 2, "module": m.group(4)})
    return rows


def summarize_importtime(rows: List[dict], top: int = 25) -> dict:
    """Totals per top-level package (self time, so nothing is double counted)."""
    by_pkg: Dict[str, int] = {}
    for r in rows:
        pkg = r["module"].split(".")[0]
        by_pkg[pkg] = by_pkg.get(pkg, 0) + r["self_us"]
    total = sum(r["self_us"] for r in rows)
    roots = sorted([r for r in rows if r["depth"] == 0], key=lambda r: -r["cum_us"])[:top]
    return {
        "modules": len(rows),
        "self_total_s": round(total / 1e6, 3),
        "by_top_package_s": {k: round(v / 1e6, 3) for k, v in sorted(by_pkg.items(), key=lambda kv: -kv[1])[:top]},
        "slowest_roots": [{"module": r["module"], "cum_s": round(r["cum_us"] / 1e6, 3)} for r in roots],
        "slowest_self": [{"module": r["module"], "self_s": round(r["self_us"] / 1e6, 3)}
                         for r in sorted(rows, key=lambda r: -r["self_us"])[:top]],
    }


def run_probe(targets: List[str], mode: str = "asis", *, python: Optional[str] = None,
              importtime: bool = True, fake_cuda: bool = False, audit_pyc: bool = False,
              dump_modules: bool = False, dump_files: bool = False, env_extra: Optional[Dict[str, str]] = None,
              prefix_dir: Optional[str] = None, timeout: float = 1800.0, workdir: Optional[str] = None) -> dict:
    """One fresh interpreter importing `targets`. Returns the probe JSON plus wall time."""
    workdir = workdir or tempfile.mkdtemp(prefix="pystartup-probe-")
    out = os.path.join(workdir, f"probe-{mode.replace(':', '_').replace('/', '_')}-{time.time_ns()}.json")
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join([PKG_PARENT] + [p for p in env.get("PYTHONPATH", "").split(os.pathsep) if p])
    env.pop("PYTHONPYCACHEPREFIX", None)
    env.pop("PYTHONDONTWRITEBYTECODE", None)
    cleanup = None
    if mode == "nopyc":
        cleanup = tempfile.mkdtemp(prefix="pystartup-emptyprefix-")
        env["PYTHONPYCACHEPREFIX"] = cleanup
        env["PYTHONDONTWRITEBYTECODE"] = "1"
    elif mode in ("prefix-cold", "prefix-warm"):
        if not prefix_dir:
            raise ValueError(f"{mode} needs prefix_dir")
        env["PYTHONPYCACHEPREFIX"] = prefix_dir
    elif mode.startswith("prefix:"):
        env["PYTHONPYCACHEPREFIX"] = mode.split(":", 1)[1]
    elif mode != "asis":
        raise ValueError(mode)
    env.update(env_extra or {})
    cmd = [python or sys.executable]
    if importtime:
        cmd += ["-X", "importtime"]
    cmd += ["-m", "pystartup._importprobe", "--out", out, "--targets", ",".join(targets)]
    if fake_cuda:
        cmd.append("--fake-cuda")
    if audit_pyc:
        cmd.append("--audit-pyc")
    if dump_modules:
        cmd.append("--dump-modules")
    if dump_files:
        cmd.append("--dump-files")
    t = time.perf_counter()
    r = subprocess.run(cmd, capture_output=True, text=True, env=env, timeout=timeout, cwd=workdir)
    wall = time.perf_counter() - t
    res: dict = {"mode": mode, "returncode": r.returncode, "wall_s": round(wall, 4)}
    try:
        with open(out) as f:
            res.update(json.load(f))
    except (OSError, ValueError):
        res["stderr_tail"] = r.stderr[-3000:]
    if importtime:
        res["importtime"] = summarize_importtime(parse_importtime(r.stderr))
    if cleanup:
        shutil.rmtree(cleanup, ignore_errors=True)
    return res


def compileall_prefix(roots: List[str], prefix: str, *, invalidation: str = "unchecked-hash",
                      workers: int = 0, python: Optional[str] = None, files: Optional[List[str]] = None) -> dict:
    """Byte-compile `roots` (or just `files`) into a PYTHONPYCACHEPREFIX tree; returns time and size.

    This is fix (a) done out of tree, so it can run in an unmodified image.
    """
    env = dict(os.environ)
    env["PYTHONPYCACHEPREFIX"] = prefix
    env.pop("PYTHONDONTWRITEBYTECODE", None)
    cmd = [python or sys.executable, "-m", "compileall", "-q", "-j", str(workers),
           "--invalidation-mode", invalidation]
    listfile = None
    if files:
        listfile = tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False)
        listfile.write("\n".join(files))
        listfile.close()
        cmd += ["-i", listfile.name]
    else:
        cmd += roots
    t = time.perf_counter()
    r = subprocess.run(cmd, capture_output=True, text=True, env=env)
    dt = time.perf_counter() - t
    size = n = 0
    for dirpath, _, fns in os.walk(prefix):
        for fn in fns:
            if fn.endswith(".pyc"):
                n += 1
                size += os.path.getsize(os.path.join(dirpath, fn))
    if listfile:
        os.unlink(listfile.name)
    # the prefix also receives pycs of the stdlib modules compileall itself imported
    from pystartup.pycheck import pyc_path_for

    req = [f for f in (files or []) if f.endswith(".py")]
    req_ok = sum(1 for f in req if os.path.exists(pyc_path_for(f, prefix)))
    return {"seconds": round(dt, 3), "pyc_files": n, "pyc_bytes": size, "requested": len(req),
            "requested_compiled": req_ok, "returncode": r.returncode,
            "invalidation": invalidation, "workers": workers, "stderr_tail": r.stderr[-1000:]}
