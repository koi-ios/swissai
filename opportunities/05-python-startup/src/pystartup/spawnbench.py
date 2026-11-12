"""Worker start-up under spawn, fork and an early-started zygote, modelled on an engine launch.

The engine's main process imports its own stack, then starts N workers that need the same
stack. Scenarios, all measured from t0 = parent start of the scenario:

    spawn   parent imports targets; then N spawn children each import targets again
            (SGLang 0.5.20: mp.set_start_method("spawn", force=True))
    fork    parent imports targets; then N fork children inherit them (vLLM's default when CUDA
            is not initialised; unsafe once the parent has initialised CUDA)
    zygote  t0: start the guarded forkserver preloading targets; the parent imports targets in
            parallel; then N children are forked from the zygote

Each scenario runs in its own fresh orchestrator interpreter so earlier imports don't leak.

    python -m pystartup.spawnbench --targets torch,transformers --workers 4 --repeats 3 --json out.json
"""
from __future__ import annotations

import argparse
import json
import os
import statistics as st
import subprocess
import sys
import time
from typing import List, Optional

PKG_PARENT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _import_targets(targets):
    """Import targets; under PYSTARTUP_FAKE_CUDA (CPU containers) behind the same fact guard
    the zygote uses, so spawn, fork and zygote import exactly the same code paths."""
    guard = None
    if os.environ.get("PYSTARTUP_FAKE_CUDA") and any(t not in sys.modules for t in targets):
        try:
            from pystartup.cudafacts import CudaGuard, probe_facts

            import torch  # noqa: F401

            guard = CudaGuard(probe_facts()).arm()
        except ImportError:
            guard = None
    try:
        for t in targets:
            __import__(t)
    finally:
        if guard is not None:
            guard.disarm()


def _child(q, targets):
    t_enter = time.time()
    pre = [t for t in targets if t in sys.modules]
    _import_targets(targets)
    q.put({"pid": os.getpid(), "t_enter": t_enter, "t_ready": time.time(), "preloaded": len(pre),
           "zygote_child": os.environ.get("PYSTARTUP_ZYGOTE_CHILD") is not None})


def _scenario(method: str, targets: List[str], workers: int, facts) -> dict:
    import multiprocessing as mp

    t0 = time.time()
    z = None
    if method == "zygote":
        from pystartup import zygote

        z = zygote.start(targets, facts=facts, strict=False)
        zygote.install_parent_hooks()
    tp = time.time()
    _import_targets(targets)  # the parent's own imports (the engine's main process does these)
    parent_import_s = time.time() - tp
    zstatus = None
    if z is not None:
        zstatus = z.wait(600)
        if not zstatus.get("ok"):
            return {"method": method, "error": zstatus.get("reasons")}
        ctx = mp.get_context("forkserver")
    else:
        ctx = mp.get_context(method)
    q = ctx.Queue()
    t_launch = time.time()
    ps = [ctx.Process(target=_child, args=(q, targets)) for _ in range(workers)]
    for p in ps:
        p.start()
    res = [q.get(timeout=900) for _ in ps]
    for p in ps:
        p.join(60)
    t_all = max(r["t_ready"] for r in res)
    out = {
        "method": method, "workers": workers, "parent_import_s": round(parent_import_s, 3),
        "launch_to_ready_s": [round(r["t_ready"] - t_launch, 3) for r in res],
        "launch_to_target_enter_s": [round(r["t_enter"] - t_launch, 3) for r in res],
        "t0_to_all_ready_s": round(t_all - t0, 3),
        "children_preloaded": [r["preloaded"] for r in res],
    }
    if zstatus:
        out["zygote_preload_s"] = zstatus.get("preload_s")
        out["zygote_waited_s"] = round(t_launch - t0, 3)
        out["zygote_invariants"] = zstatus.get("invariants")
    return out


def run_scenario_subprocess(method: str, targets: List[str], workers: int, facts=None,
                            python: Optional[str] = None, env_extra=None, timeout: float = 1800) -> dict:
    code = ("import json,sys; sys.path.insert(0, %r); from pystartup.spawnbench import _scenario; "
            "print('SPAWNBENCH_JSON ' + json.dumps(_scenario(%r, %r, %d, %r)))"
            % (PKG_PARENT, method, targets, workers, facts))
    env = dict(os.environ)
    env.update(env_extra or {})
    env["PYTHONPATH"] = os.pathsep.join([PKG_PARENT] + [p for p in env.get("PYTHONPATH", "").split(os.pathsep) if p])
    t = time.perf_counter()
    r = subprocess.run([python or sys.executable, "-c", code], capture_output=True, text=True,
                       timeout=timeout, env=env)
    wall = time.perf_counter() - t
    for line in r.stdout.splitlines():
        if line.startswith("SPAWNBENCH_JSON "):
            d = json.loads(line[len("SPAWNBENCH_JSON "):])
            d["process_wall_s"] = round(wall, 3)
            return d
    return {"method": method, "error": r.stderr[-3000:], "returncode": r.returncode}


def bench(targets: List[str], workers: int, repeats: int, methods: List[str], facts=None,
          python: Optional[str] = None) -> dict:
    runs = []
    for rep in range(repeats):
        for m in methods:  # interleaved so drift hits every method alike
            d = run_scenario_subprocess(m, targets, workers, facts=facts, python=python)
            d["repeat"] = rep
            runs.append(d)
            print(json.dumps({k: d.get(k) for k in ("method", "repeat", "parent_import_s",
                                                    "t0_to_all_ready_s", "error")}), flush=True)
    summary = {}
    for m in methods:
        ok = [r for r in runs if r["method"] == m and "error" not in r]
        if not ok:
            summary[m] = {"error": [r.get("error") for r in runs if r["method"] == m][:1]}
            continue
        tot = [r["t0_to_all_ready_s"] for r in ok]
        child = [max(r["launch_to_ready_s"]) for r in ok]
        summary[m] = {"n": len(ok), "t0_to_all_ready_s": _mmm(tot), "slowest_child_after_launch_s": _mmm(child),
                      "parent_import_s": _mmm([r["parent_import_s"] for r in ok])}
    return {"targets": targets, "workers": workers, "repeats": repeats, "python": sys.version.split()[0],
            "platform": sys.platform, "summary": summary, "runs": runs}


def _mmm(xs):
    return {"median": round(st.median(xs), 3), "min": round(min(xs), 3), "max": round(max(xs), 3)}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--targets", default="torch,transformers")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--methods", default="spawn,zygote,fork")
    ap.add_argument("--facts", default="none", help='"none", "probe" or JSON facts for the zygote')
    ap.add_argument("--json", default=None)
    a = ap.parse_args(argv)
    facts = None if a.facts == "none" else ("probe" if a.facts == "probe" else json.loads(a.facts))
    res = bench(a.targets.split(","), a.workers, a.repeats, a.methods.split(","), facts=facts)
    print(json.dumps(res["summary"], indent=1))
    if a.json:
        with open(a.json, "w") as f:
            json.dump(res, f, indent=1)
    return 0


if __name__ == "__main__":
    sys.exit(main())
