"""harness/ab.py end to end against a faithful mock `vllm` (no GPU): every arm runs, servekit's
own profiler parses the phases, the overlay is seeded and promoted, and the builds-avoided
accounting shows what servekit's three variables miss."""

import json
import os
import socket
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
pytest.importorskip("servekit.profile")


def free_port() -> int:
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    p = s.getsockname()[1]
    s.close()
    return p


def test_ab_harness_with_mock_vllm(tmp_path):
    bindir = tmp_path / "bin"
    bindir.mkdir()
    vllm = bindir / "vllm"
    vllm.write_text(f"#!/bin/sh\nexec {sys.executable} {ROOT / 'tests/workloads/mock_vllm.py'} \"$@\"\n")
    vllm.chmod(0o755)
    model = tmp_path / "models" / "Apertus-8B"
    model.mkdir(parents=True)
    out = tmp_path / "ab.json"
    env = dict(os.environ, PATH=f"{bindir}:{os.environ['PATH']}", AB_PORT=str(free_port()),
               AB_OVERLAY=str(tmp_path / "overlay"), AB_WORKDIR=str(tmp_path / "jobs"))
    p = subprocess.run([sys.executable, str(ROOT / "harness/ab.py"), "--engine", "vllm", "--model-dir", str(model),
                        "--arms", "cold,servekit3,full", "--repeats", "2", "--shared-root", str(tmp_path / "shared"),
                        "--out", str(out), "--timeout", "120"], env=env, capture_output=True, text=True, timeout=600)
    assert p.returncode == 0, p.stderr[-3000:]
    runs = json.loads(out.read_text())
    assert len(runs) == 2 + 3 + 3 and all(r["ok"] for r in runs), [r.get("error") for r in runs]
    by = {(r["arm"], r["run"].split("-", 1)[1]): r for r in runs}
    phases = {ph["name"] for ph in by[("cold", "cold-0")]["profile"]["phases"]}
    assert {"torch_compile", "cuda_graph_capture"} <= phases            # servekit parsed the log

    def binaries(r, comp):
        return r["written"].get(comp, {}).get("binaries", 0)
    # cold: everything rebuilt every time
    assert binaries(by[("cold", "cold-1")], "tilelang") == 3
    # full: populate builds, warm rebuilds nothing
    assert binaries(by[("full", "populate")], "deep_gemm") == 2
    for comp in ("triton", "tilelang", "deep_gemm", "inductor.fxgraph"):
        assert binaries(by[("full", "warm-0")], comp) == 0, comp
    # servekit3: Triton is persisted, but TileLang and DeepGEMM are rebuilt on every launch
    assert binaries(by[("servekit3", "warm-0")], "triton") == 0
    assert binaries(by[("servekit3", "warm-0")], "tilelang") == 3
    assert binaries(by[("servekit3", "warm-0")], "deep_gemm") == 2
    assert by[("full", "warm-1")]["seed"]["files"] > 0 and by[("full", "populate")]["promote"]["promoted"] > 0
    # the summary renders
    s = subprocess.run([sys.executable, str(ROOT / "harness/summarize.py"), str(out)], capture_output=True, text=True)
    assert s.returncode == 0 and "builds avoided" in s.stdout and "| full |" in s.stdout
