"""End-to-end SGLang launches: baseline spawn vs the zygote, with and without bytecode.

    python -m pystartup.e2e --model /hf/Qwen2.5-0.5B-Instruct --tp 2 --repeats 3 \
        --arms spawn,zygote,zygote-cal,spawn-nopyc,zygote-cal-nopyc --out /results/e2e

For every launch it records, from the launcher's own clock (sub-second, unlike SGLang's
1 s log stamps): Popen -> `server_args=` (main process imports + argument processing),
-> the last TP rank's `Init torch distributed begin` (worker start), -> ready; the greedy
output of a fixed prompt (must be identical across arms); the wrapper's zygote report; and
the process tree from tracer/sitecustomize.py.

Arms:
    spawn            stock `python -m sglang.launch_server`
    spawn-nopyc      same, every process compiles from source and writes nothing
                     (= read-only image without pycs, e.g. upstream lmsysorg/sglang)
    zygote           pystartup.sglang_launch, default CUDA-free third-party preload
    zygote-cal       pystartup.sglang_launch with a calibrated preload list (strict replay)
    zygote-cal-nopyc zygote-cal without pycs: the zygote compiles once for all workers

`--calibrate` (default on when a zygote-cal arm is requested and no list is given) runs one
traced spawn launch, takes the TP0 scheduler's module list and calibrates it on this node.
Runs are interleaved (arm order rotates per repeat); an untimed warm-up launch comes first
so every timed launch sees the same warm page cache.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import re
import shutil
import signal
import statistics as st
import subprocess
import sys
import tempfile
import threading
import time
import urllib.request
from typing import Dict, List, Optional

PKG_PARENT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TRACER_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "tracer")
READY = re.compile(r"The server is fired up and ready to roll")
SERVER_ARGS = re.compile(r"server_args=")
DIST_BEGIN = re.compile(r"\[[^\]]*TP(\d+)[^\]]*\] Init torch distributed begin")
DIST_END = re.compile(r"\[[^\]]*TP(\d+)[^\]]*\] Init torch distributed ends")
PROMPT = "The capital of France is"


def sglang_cmd(model: str, tp: int, port: int, extra: List[str]) -> List[str]:
    return ["--model-path", model, "--tp-size", str(tp), "--host", "127.0.0.1", "--port", str(port),
            "--random-seed", "42", "--mem-fraction-static", "0.5", "--disable-custom-all-reduce"] + extra


def arm_command(arm: str, sargs: List[str], cal: Optional[str], report: str) -> List[str]:
    py = sys.executable
    if arm.startswith("spawn"):
        return [py, "-m", "sglang.launch_server"] + sargs
    w = ["--report", report]
    if arm.startswith("zygote-cal"):
        if not cal:
            raise ValueError("zygote-cal needs a calibration file")
        w += ["--zygote-modules", cal]
    return [py, "-m", "pystartup.sglang_launch"] + w + ["--"] + sargs


def kill_tree(proc: subprocess.Popen, grace: float = 15.0):
    try:
        os.killpg(proc.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    t = time.time()
    while time.time() - t < grace and proc.poll() is None:
        time.sleep(0.2)
    try:
        os.killpg(proc.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    proc.wait()
    time.sleep(2)  # let the GPUs release memory before the next launch


def generate(port: int, timeout: float = 120) -> Optional[str]:
    body = json.dumps({"text": PROMPT, "sampling_params": {"temperature": 0, "max_new_tokens": 16}}).encode()
    req = urllib.request.Request(f"http://127.0.0.1:{port}/generate", data=body,
                                 headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return json.loads(r.read())["text"]
    except Exception as e:  # noqa: BLE001
        return f"ERROR {e!r}"


def launch(arm: str, sargs: List[str], cal: Optional[str], run_dir: str, tp: int, port: int,
           ready_timeout: float, env_extra: Dict[str, str]) -> dict:
    os.makedirs(run_dir, exist_ok=True)
    trace = os.path.join(run_dir, "trace")
    report = os.path.join(run_dir, "zygote_report.json")
    env = dict(os.environ)
    env.update(env_extra)
    env["PYTHONPATH"] = os.pathsep.join([TRACER_DIR, PKG_PARENT] +
                                        [p for p in env.get("PYTHONPATH", "").split(os.pathsep) if p])
    env["PYSTARTUP_TRACE_DIR"] = trace
    env["PYSTARTUP_TRACE_MODULES"] = "1"
    env["PYTHONUNBUFFERED"] = "1"
    nopyc_dir = None
    if arm.endswith("-nopyc"):
        nopyc_dir = tempfile.mkdtemp(prefix="pystartup-nopyc-")
        env["PYTHONPYCACHEPREFIX"] = nopyc_dir
        env["PYTHONDONTWRITEBYTECODE"] = "1"
    cmd = arm_command(arm, sargs, cal, report)
    lines: List[tuple] = []
    marks: Dict[str, float] = {}
    ranks_begin, ranks_end = {}, {}
    t_spawn = time.time()
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, env=env, text=True,
                            bufsize=1, start_new_session=True)
    ready = threading.Event()

    def reader():
        for line in proc.stdout:
            now = time.time()
            lines.append((now, line.rstrip("\n")))
            if "server_args" not in marks and SERVER_ARGS.search(line):
                marks["server_args"] = now
            m = DIST_BEGIN.search(line)
            if m:
                ranks_begin.setdefault(int(m.group(1)), now)
            m = DIST_END.search(line)
            if m:
                ranks_end.setdefault(int(m.group(1)), now)
            if READY.search(line):
                marks["ready"] = now
                ready.set()
        ready.set()

    th = threading.Thread(target=reader, daemon=True)
    th.start()
    ready.wait(ready_timeout)
    out = generate(port) if "ready" in marks else None
    kill_tree(proc)
    th.join(10)
    with open(os.path.join(run_dir, "engine.log"), "w") as f:
        for t, l in lines:
            f.write(f"{t - t_spawn:9.3f} {l}\n")
    rec = {"arm": arm, "cmd": cmd, "t_spawn": t_spawn, "returncode": proc.returncode, "output": out,
           "ok": "ready" in marks and out is not None and not str(out).startswith("ERROR")}
    if "server_args" in marks:
        rec["spawn_to_server_args_s"] = round(marks["server_args"] - t_spawn, 3)
    if len(ranks_begin) >= tp:
        last = max(ranks_begin.values())
        rec["spawn_to_dist_begin_s"] = round(last - t_spawn, 3)
        if "server_args" in marks:
            rec["server_args_to_dist_begin_s"] = round(last - marks["server_args"], 3)
    if len(ranks_end) >= tp:
        rec["dist_init_s"] = round(max(ranks_end.values()) - max(ranks_begin.values() or [0]), 3)
    if "ready" in marks:
        rec["spawn_to_ready_s"] = round(marks["ready"] - t_spawn, 3)
    if os.path.exists(report):
        with open(report) as f:
            rep = json.load(f)
        zs = rep.get("zygote_status") or {}
        rec["zygote"] = {"used": rep.get("use_forkserver"), "blocked_on_zygote_s": rep.get("blocked_on_zygote_s"),
                         "preload_s": zs.get("preload_s"), "ok": zs.get("ok"), "reasons": zs.get("reasons"),
                         "invariants": zs.get("invariants"), "loaded": len(zs.get("loaded", [])),
                         "facts": zs.get("facts"), "skipped": len(zs.get("skipped", [])),
                         "failed": zs.get("failed", [])[:5], "tripped": zs.get("tripped", [])[:5]}
    try:
        from pystartup.profile_startup import tree_summary

        rec["tree"] = tree_summary(trace)
    except Exception as e:  # noqa: BLE001
        rec["tree"] = {"error": repr(e)}
    if nopyc_dir:
        shutil.rmtree(nopyc_dir, ignore_errors=True)
    return rec


def scheduler_module_list(trace_dir: str) -> List[str]:
    """Ordered modules of the TP0 scheduler (the longest-lived worker record) from a traced launch."""
    best = None
    for p in glob.glob(os.path.join(trace_dir, "proc-*.json")):
        with open(p) as f:
            r = json.load(f)
        cmd = " ".join(r.get("cmdline", []))
        if "spawn_main" not in cmd and not r.get("forked_from"):
            continue
        if best is None or r.get("imports", 0) > best.get("imports", 0):
            best = r
    if not best:
        return []
    seen, out = set(), []
    for _, name in best.get("modules", []):
        if name not in seen:
            seen.add(name)
            out.append(name)
    return out


def summarize(runs: List[dict]) -> dict:
    keys = ["spawn_to_server_args_s", "server_args_to_dist_begin_s", "spawn_to_dist_begin_s", "dist_init_s",
            "spawn_to_ready_s"]
    out = {}
    for arm in sorted({r["arm"] for r in runs}):
        rs = [r for r in runs if r["arm"] == arm and r.get("ok")]
        d = {"n_ok": len(rs), "n": sum(1 for r in runs if r["arm"] == arm)}
        for k in keys:
            xs = [r[k] for r in rs if k in r]
            if xs:
                d[k] = {"median": round(st.median(xs), 2), "min": round(min(xs), 2), "max": round(max(xs), 2)}
        d["outputs"] = sorted({str(r.get("output")) for r in rs})
        z = [r["zygote"] for r in rs if r.get("zygote")]
        if z:
            d["zygote_used"] = sum(1 for x in z if x.get("used"))
            d["zygote_preload_s"] = [x.get("preload_s") for x in z]
            d["blocked_on_zygote_s"] = [x.get("blocked_on_zygote_s") for x in z]
        out[arm] = d
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--tp", type=int, default=2)
    ap.add_argument("--port", type=int, default=30000)
    ap.add_argument("--arms", default="spawn,zygote,zygote-cal,spawn-nopyc,zygote-cal-nopyc")
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--calibration", default=None, help="existing calibration JSON for zygote-cal arms")
    ap.add_argument("--keep-cuda-graphs", action="store_true")
    ap.add_argument("--ready-timeout", type=float, default=900)
    ap.add_argument("--out", required=True)
    ap.add_argument("--extra", default="", help="extra SGLang arguments, space separated")
    a = ap.parse_args(argv)
    os.makedirs(a.out, exist_ok=True)
    extra = a.extra.split() if a.extra else []
    if not a.keep_cuda_graphs:
        extra += ["--cuda-graph-backend-decode", "disabled", "--cuda-graph-backend-prefill", "disabled"]
    sargs = sglang_cmd(a.model, a.tp, a.port, extra)
    env_extra = {}
    try:
        if os.statvfs("/dev/shm").f_bavail * os.statvfs("/dev/shm").f_frsize < (1 << 30):
            env_extra["NCCL_SHM_DISABLE"] = "1"  # small /dev/shm in containers
    except OSError:
        pass
    arms = a.arms.split(",")
    meta = {"argv": sys.argv, "sglang_args": sargs, "env_extra": env_extra, "t": time.time()}
    print(f"[e2e] warm-up launch (untimed)", flush=True)
    warm = launch("spawn", sargs, None, os.path.join(a.out, "warmup"), a.tp, a.port, a.ready_timeout, env_extra)
    meta["warmup"] = {k: warm.get(k) for k in ("ok", "spawn_to_ready_s", "output", "returncode")}
    if not warm.get("ok"):
        meta["error"] = "warm-up launch failed; see warmup/engine.log"
        with open(os.path.join(a.out, "e2e.json"), "w") as f:
            json.dump(meta, f, indent=1)
        print(json.dumps(meta["warmup"]), flush=True)
        return 1
    cal = a.calibration
    if any(x.startswith("zygote-cal") for x in arms) and not cal:
        from pystartup import zygote

        cands = scheduler_module_list(os.path.join(a.out, "warmup", "trace"))
        meta["calibration_candidates"] = len(cands)
        res = zygote.calibrate(cands, facts="probe")
        cal = os.path.join(a.out, "calibration.json")
        with open(cal, "w") as f:
            json.dump(res, f, indent=1)
        meta["calibration"] = {"modules": len(res["modules"]), "trips": len(res["trips"]),
                               "seconds": res.get("calibration_s"), "facts": res.get("facts"),
                               "not_ok_targets": sum(1 for v in res["targets"].values() if v != "ok")}
        print(f"[e2e] calibration: {json.dumps(meta['calibration'])}", flush=True)
    runs = []
    for rep in range(a.repeats):
        order = arms[rep % len(arms):] + arms[:rep % len(arms)]
        for arm in order:
            r = launch(arm, sargs, cal, os.path.join(a.out, f"{arm}-r{rep}"), a.tp, a.port, a.ready_timeout, env_extra)
            r["repeat"] = rep
            runs.append(r)
            print(f"[e2e] {arm} r{rep}: " + json.dumps({k: r.get(k) for k in (
                "ok", "spawn_to_server_args_s", "server_args_to_dist_begin_s", "spawn_to_ready_s")}), flush=True)
            with open(os.path.join(a.out, "e2e.json"), "w") as f:
                json.dump({"meta": meta, "summary": summarize(runs), "runs": runs}, f, indent=1)
    print(json.dumps(summarize(runs), indent=1), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
