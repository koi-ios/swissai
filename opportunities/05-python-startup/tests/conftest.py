import json
import os
import subprocess
import sys
import textwrap

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(ROOT, "src")
if SRC not in sys.path:
    sys.path.insert(0, SRC)

NO_GPU_FACTS = {"count": 0, "major": 0, "minor": 0, "name": "none", "source": "test"}
H100_FACTS = {"count": 2, "major": 9, "minor": 0, "name": "NVIDIA H100 80GB HBM3", "source": "test"}


def run_py(code: str, env_extra=None, cwd=None, timeout=300):
    """Run `code` in a fresh interpreter with src/ (and `cwd`) importable; return CompletedProcess."""
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join([p for p in [cwd, SRC, env.get("PYTHONPATH", "")] if p])
    env.pop("PYSTARTUP_ZYGOTE_CONFIG", None)
    env.update(env_extra or {})
    return subprocess.run([sys.executable, "-c", textwrap.dedent(code)], capture_output=True, text=True,
                          env=env, cwd=cwd, timeout=timeout)


def run_json(code: str, marker: str = "RESULT ", **kw):
    r = run_py(code, **kw)
    for line in r.stdout.splitlines():
        if line.startswith(marker):
            return json.loads(line[len(marker):])
    raise AssertionError(f"no {marker!r} line; rc={r.returncode}\nstdout:\n{r.stdout[-3000:]}\nstderr:\n{r.stderr[-6000:]}")


@pytest.fixture
def modsdir(tmp_path):
    """A directory of tiny test modules for zygote scenarios."""
    d = tmp_path / "mods"
    d.mkdir()
    marker = tmp_path / "exec_count.txt"
    (d / "zt_counted.py").write_text(textwrap.dedent(f"""
        # appends one line each time this module body executes
        with open({str(marker)!r}, "a") as f:
            f.write("x\\n")
        VALUE = 42
    """))
    (d / "zt_trips.py").write_text(textwrap.dedent("""
        import torch
        free, total = torch.cuda.mem_get_info()   # needs a CUDA context: must trip in the zygote
    """))
    (d / "zt_swallows.py").write_text(textwrap.dedent("""
        import torch
        try:
            torch.cuda.mem_get_info()
        except BaseException:                      # a library that swallows everything
            pass
        OK = True
    """))
    (d / "zt_uses_facts.py").write_text(textwrap.dedent("""
        import torch
        CAP = torch.cuda.get_device_capability()   # served from facts
        AVAILABLE = torch.cuda.is_available()
        N = torch.cuda.device_count()
    """))
    (d / "zt_thread.py").write_text(textwrap.dedent("""
        import threading, time
        threading.Thread(target=time.sleep, args=(30,), daemon=True).start()
    """))
    (d / "zt_denied_dep.py").write_text("import zt_denied_target\n")
    (d / "zt_denied_target.py").write_text("X = 1\n")
    (d / "zt_partial.py").write_text(textwrap.dedent(f"""
        with open({str(marker)!r}, "a") as f:
            f.write("partial\\n")
        import torch
        torch.cuda.synchronize()                   # trips after a side effect
    """))
    return {"dir": str(d), "marker": str(marker)}
