import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
WORKLOADS = ROOT / "tests" / "workloads"


def run_audit(tmp_path: Path, workload: str, *extra: str, env: dict | None = None, timeout: int = 600):
    """Run `python -m cache_audit run` on a workload; return (exit code, report dict)."""
    out = tmp_path / "report"
    cmd = [sys.executable, "-m", "cache_audit", "run", "--workdir", str(tmp_path / "runs"),
           "--out", str(out), *extra, "--", sys.executable, str(WORKLOADS / workload)]
    e = dict(os.environ)
    e["PATH"] = str(Path(sys.executable).parent) + os.pathsep + e.get("PATH", "")
    if env:
        e.update(env)
    p = subprocess.run(cmd, cwd=ROOT, env=e, capture_output=True, text=True, timeout=timeout)
    rep = json.loads((out / "report.json").read_text()) if (out / "report.json").exists() else {}
    return p.returncode, rep, p.stdout + p.stderr


@pytest.fixture
def audit(tmp_path):
    return lambda workload, *extra, **kw: run_audit(tmp_path, workload, *extra, **kw)
