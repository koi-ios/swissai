#!/usr/bin/env python3
"""Cold vs warm vs servekit-3-variables A/B for one engine/model/TP, inside the engine image.

Every run is a fresh "job": new HOME and TMPDIR, caches in a node-local overlay at a fixed path
(/tmp/sml-jit-cache, as sml does), seeded from and promoted to a shared directory (a Modal Volume
standing in for capstor) with the same helper sml ships (integration/sml_jit_cache.py).

Arms
  cold       nothing persists (the platform today)
  servekit3  only TRITON_CACHE_DIR, TVM_FFI_CACHE_DIR, FLASHINFER_WORKSPACE_BASE persist (servekit's list)
  full       every variable from the platform env TOML (patches/0002) persists
Persisting arms start with one populate run (recorded, not counted as warm), then N warm runs.

Per run it records servekit-profile phases (servekit.profile.run_profile, the lab's own parser),
time to ready, first-request latency, and per-component cache files and binaries written during the
run (KB-matched): binaries written by a warm run are rebuilds, the rest are builds avoided.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import signal
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE.parent / "integration"))

from cache_audit import kb as kbmod  # noqa: E402
from cache_audit.classify import classify  # noqa: E402
import sml_jit_cache as sjc  # noqa: E402

OVERLAY = os.environ.get("AB_OVERLAY", "/tmp/sml-jit-cache")  # fixed path, as sml uses
PORT = int(os.environ.get("AB_PORT", "8080"))

SERVEKIT3 = ["TRITON_CACHE_DIR", "TVM_FFI_CACHE_DIR", "FLASHINFER_WORKSPACE_BASE"]
# The [env] cache lines of patches/0002 (vllm_base / sglang_base), {jit_cache} -> overlay.
FULL = {
    "vllm": {
        "VLLM_CACHE_ROOT": "vllm", "TRITON_CACHE_DIR": "triton", "DG_JIT_CACHE_DIR": "deep_gemm",
        "TILELANG_CACHE_DIR": "tilelang", "FLASHINFER_WORKSPACE_BASE": "flashinfer",
        "TVM_FFI_CACHE_DIR": "tvm-ffi", "CUTE_DSL_CACHE_DIR": "cute-dsl", "CUDA_CACHE_PATH": "nv",
    },
    "sglang": {
        "SGLANG_CACHE_DIR": "sglang", "SGLANG_JIT_CACHE_DIR": "sglang-jit", "TRITON_CACHE_DIR": "triton",
        "TILELANG_CACHE_DIR": "tilelang", "TVM_FFI_CACHE_DIR": "tvm-ffi", "CUTE_DSL_CACHE_DIR": "cute-dsl",
    },
}
# Platform parity (envs/*_base.toml), minus the Slingshot/libfabric NCCL settings.
PLATFORM_ENV = {
    "vllm": {"VLLM_ALLREDUCE_USE_SYMM_MEM": "0"},
    "sglang": {"SGL_ENABLE_JIT_DEEPGEMM": "0", "SGLANG_ENABLE_JIT_DEEPGEMM": "false"},
}


def engine_cmd(engine: str, model_dir: str, tp: int, extra: list[str]) -> list[str]:
    if engine == "sglang":
        return [sys.executable, "-m", "sglang.launch_server", "--model-path", model_dir, "--tp", str(tp),
                "--host", "0.0.0.0", "--port", str(PORT), "--context-length", "8192",
                "--mem-fraction-static", "0.85", "--served-model-name", "m", *extra]
    return ["vllm", "serve", model_dir, "--tensor-parallel-size", str(tp), "--host", "0.0.0.0",
            "--port", str(PORT), "--max-model-len", "8192", "--gpu-memory-utilization", "0.85",
            "--served-model-name", "m", *extra]


def snapshot(root: Path) -> dict[str, tuple[int, int]]:
    out = {}
    if root.is_dir():
        for d, _dirs, files in os.walk(root):
            for f in files:
                p = os.path.join(d, f)
                try:
                    st = os.lstat(p)
                except OSError:
                    continue
                out[p] = (st.st_size, st.st_mtime_ns)
    return out


def new_files_by_component(before: dict, after: dict, env: dict, kb) -> dict[str, dict[str, int]]:
    res: dict[str, dict[str, int]] = {}
    memo: dict = {}
    for p, st in after.items():
        if before.get(p) == st:
            continue
        m = kb.match(p, env, memo)
        label = m[0].id if m else "UNKNOWN:" + os.path.relpath(p, OVERLAY).split("/")[0]
        c = res.setdefault(label, {"files": 0, "binaries": 0, "bytes": 0})
        c["files"] += 1
        c["bytes"] += st[0]
        c["binaries"] += classify(p) == "binary"
    return res


def first_request(timeout: float = 600) -> float:
    body = json.dumps({"model": "m", "prompt": "The capital of Switzerland is", "max_tokens": 16,
                       "temperature": 0}).encode()
    req = urllib.request.Request(f"http://127.0.0.1:{PORT}/v1/completions", data=body,
                                 headers={"Content-Type": "application/json"})
    t0 = time.monotonic()
    with urllib.request.urlopen(req, timeout=timeout) as r:
        r.read()
    return time.monotonic() - t0


def one_run(engine: str, model_dir: str, tp: int, arm: str, shared: Path | None, persist: dict[str, str],
            workdir: Path, tag: str, extra: list[str], kb, timeout: float) -> dict:
    job = workdir / tag
    home, tmp = job / "home", job / "tmp"
    for d in (home, tmp):
        d.mkdir(parents=True)
    overlay = Path(OVERLAY)
    shutil.rmtree(overlay, ignore_errors=True)
    overlay.mkdir(parents=True)
    rec: dict = {"engine": engine, "model": model_dir, "tp": tp, "arm": arm, "run": tag}
    if shared is not None:
        rec["seed"] = sjc.seed(shared, overlay)
    env = dict(os.environ)
    for k in set(SERVEKIT3) | set(FULL["vllm"]) | set(FULL["sglang"]):
        env.pop(k, None)
    env.update(HOME=str(home), TMPDIR=str(tmp), **PLATFORM_ENV[engine])
    # persisted variables point into the overlay; everything else stays at defaults (fresh HOME/TMPDIR)
    env.update({k: f"{OVERLAY}/{v}" for k, v in persist.items()})
    rec["persisted_vars"] = sorted(persist)
    before = {**snapshot(overlay), **snapshot(home), **snapshot(tmp)}

    out = job / "profile.json"
    log = job / "server.log"
    t0 = time.monotonic()
    with open(log, "w") as lf:
        proc = subprocess.Popen([sys.executable, "-m", "servekit.cli", "profile", "--out", str(out),
                                 "--timeout", str(int(timeout)), "--",
                                 *engine_cmd(engine, model_dir, tp, extra)],
                                env=env, stdout=lf, stderr=subprocess.STDOUT, start_new_session=True)
        try:
            while not out.exists():
                if proc.poll() is not None:
                    raise RuntimeError(f"server exited {proc.returncode} before ready")
                if time.monotonic() - t0 > timeout:
                    raise TimeoutError("not ready")
                time.sleep(0.2)
            rec["ready_s"] = round(time.monotonic() - t0, 2)
            rec["first_request_s"] = round(first_request(), 2)
            rec["ok"] = True
        except Exception as e:  # recorded, the arm continues
            rec["ok"] = False
            rec["error"] = repr(e)
        finally:
            os.killpg(proc.pid, signal.SIGTERM)
            try:
                proc.wait(timeout=60)
            except subprocess.TimeoutExpired:
                os.killpg(proc.pid, signal.SIGKILL)
    if out.exists():
        rec["profile"] = json.loads(out.read_text())
    after = {**snapshot(overlay), **snapshot(home), **snapshot(tmp)}
    rec["written"] = new_files_by_component(before, after, env, kb)
    if shared is not None and rec.get("ok"):
        rec["promote"] = sjc.promote(shared, overlay, settle=0)
    rec["log_tail"] = log.read_text(errors="replace").splitlines()[-30:]
    shutil.rmtree(job, ignore_errors=True)
    return rec


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--engine", choices=["sglang", "vllm"], required=True)
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--tp", type=int, default=1)
    ap.add_argument("--arms", default="cold,servekit3,full")
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--shared-root", required=True, help="persistent dir (Modal Volume)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--timeout", type=float, default=2400)
    ap.add_argument("--extra", default="", help="extra engine args")
    a = ap.parse_args()
    kb = kbmod.load()
    workdir = Path(os.environ.get("AB_WORKDIR", "/tmp/ab-jobs"))
    results = []
    outp = Path(a.out)
    outp.parent.mkdir(parents=True, exist_ok=True)
    name = Path(a.model_dir).name
    for arm in a.arms.split(","):
        if arm == "cold":
            persist, shared = {}, None
            runs = [f"cold-{i}" for i in range(a.repeats)]
        else:
            persist = FULL[a.engine] if arm == "full" else {k: FULL[a.engine].get(k, k.lower()) for k in SERVEKIT3}
            shared = Path(a.shared_root) / a.engine / name / f"tp{a.tp}" / arm
            shutil.rmtree(shared, ignore_errors=True)
            shared.mkdir(parents=True)
            runs = ["populate"] + [f"warm-{i}" for i in range(a.repeats)]
        for tag in runs:
            r = one_run(a.engine, a.model_dir, a.tp, arm, shared, persist, workdir, f"{arm}-{tag}",
                        a.extra.split(), kb, a.timeout)
            results.append(r)
            outp.write_text(json.dumps(results, indent=1))
            print(json.dumps({k: r.get(k) for k in ("arm", "run", "ok", "ready_s", "first_request_s")}), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
