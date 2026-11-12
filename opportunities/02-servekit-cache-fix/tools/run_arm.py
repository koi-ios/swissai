"""One job of the JIT-cache A/B, inside the Modal container.

Runs `servekit launch` (or `--only-prepare`) from the servekit tree it is given,
as a new SLURM job on a new node would: a fresh $HOME and $TMPDIR, an empty
node-local /tmp/servekit/caches, nothing carried over but the artifact. Once the
server reports ready it is stopped, the way the lab's sbatch scripts did.

    python run_arm.py --servekit /opt/servekit-main --artifact /work/artifacts/main \
        --model /work/models/Qwen2.5-1.5B-Instruct --tp 2 --run-dir /work/runs/r1 [--prepare]

The weight stage is a plain file copy in every arm. What the A/B compares is the
JIT caches, and servekit's sliced O_DIRECT stager is built for Lustre and a large
/dev/shm, neither of which a Modal container has.

Writes into --run-dir report.json (servekit's report, launch only) and
result.json (exit code, wall time, and the inventory of what the job left in
its $HOME and temp dir). The caller captures stdout, the engine's included,
as engine.log.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import signal
import sys
import tempfile
import threading
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import runlog  # noqa: E402

# Every variable either servekit version may set, so none leaks in from the
# container's own environment.
CACHE_VARS = (
    "TRITON_CACHE_DIR", "TVM_FFI_CACHE_DIR", "FLASHINFER_WORKSPACE_BASE", "SGLANG_CACHE_DIR",
    "SGLANG_JIT_CACHE_DIR", "SGLANG_DG_CACHE_DIR", "TORCHINDUCTOR_CACHE_DIR", "CUDA_CACHE_PATH",
    "TORCH_EXTENSIONS_DIR", "TILELANG_CACHE_DIR", "CUTE_DSL_CACHE_DIR", "XDG_CACHE_HOME",
)
NODE_LOCAL = Path("/tmp/servekit/caches")


def plain_stage(src, dest, slices=64, file_pattern="*"):
    from servekit.stage import StageResult

    t0 = time.time()
    dest = Path(dest)
    dest.mkdir(parents=True, exist_ok=True)
    total = 0
    for f in sorted(Path(src).glob(file_pattern)):
        if f.is_file():
            shutil.copyfile(f, dest / f.name)
            total += f.stat().st_size
    wall = max(time.time() - t0, 1e-6)
    return StageResult(dest=dest, wall_s=wall, gbps=total / wall / 1e9, bytes=total)


def stop_when_ready(report: Path, deadline_s: float) -> None:
    """SIGTERM ourselves once servekit has written its report: servekit forwards it to the engine."""
    t0 = time.monotonic()
    while not report.exists() and time.monotonic() - t0 < deadline_s:
        time.sleep(0.5)
    time.sleep(2)  # let servekit finish its own lines (the audit, the free)
    os.kill(os.getpid(), signal.SIGTERM)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--servekit", required=True, help="a servekit checkout; its src/ is put first on sys.path")
    ap.add_argument("--artifact", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--tp", type=int, default=2)
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--port", type=int, default=30000)
    ap.add_argument("--prepare", action="store_true", help="servekit launch --only-prepare")
    ap.add_argument("--timeout", type=float, default=1200)
    ap.add_argument("--stage-root", default="/work/stage")
    ap.add_argument("--node-local", default=str(NODE_LOCAL), help="servekit's node-local cache root (tests only)")
    ap.add_argument("engine_args", nargs="*", help="extra sglang flags, after --")
    args = ap.parse_args()

    run = Path(args.run_dir)
    home, tmp = run / "home", run / "tmp"
    for d in (home, tmp):
        shutil.rmtree(d, ignore_errors=True)
        d.mkdir(parents=True)
    for var in CACHE_VARS:
        os.environ.pop(var, None)
    os.environ.update(HOME=str(home), TMPDIR=str(tmp), PYTHONUNBUFFERED="1")
    tempfile.tempdir = None  # re-read TMPDIR
    shutil.rmtree(args.node_local, ignore_errors=True)  # a new node has nothing node-local
    os.environ["PYTHONPATH"] = f"{args.servekit}/src" + os.pathsep + os.environ.get("PYTHONPATH", "")
    sys.path.insert(0, f"{args.servekit}/src")

    import servekit.launch as launch_mod
    from servekit.cli import main as servekit

    launch_mod.stage = plain_stage
    launch_mod.DEFAULT_ROOT = Path(args.stage_root)
    launch_mod.DEFAULT_CACHE_ROOT = Path(args.node_local)

    report = run / "report.json"
    report.unlink(missing_ok=True)
    command = [
        sys.executable, "-m", "sglang.launch_server", "--model-path", args.model,
        "--tp", str(args.tp), "--host", "127.0.0.1", "--port", str(args.port), "--random-seed", "42",
        *args.engine_args,
    ]
    argv = ["launch", "--servekit-artifact-path", args.artifact, "--out", str(report), "--timeout", str(args.timeout)]
    if args.prepare:
        argv.append("--only-prepare")
    else:
        threading.Thread(target=stop_when_ready, args=(report, args.timeout + 60), daemon=True).start()

    t0 = time.time()
    rc = servekit(argv + ["--"] + command)
    result = {
        "rc": rc,
        "wall_s": round(time.time() - t0, 2),
        "servekit": args.servekit,
        "artifact": args.artifact,
        "prepare": args.prepare,
        "left_behind": runlog.inventory(home, tmp),
    }
    (run / "result.json").write_text(json.dumps(result, indent=2))
    return 0 if rc == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
