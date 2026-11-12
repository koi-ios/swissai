"""Python audit-hook recorder (PEP 578): attributes each write to a source line.

A ``sitecustomize.py`` placed first on ``PYTHONPATH`` installs
``sys.addaudithook`` in every Python process of the run (including
multiprocessing ``spawn`` workers, which inherit the environment). It logs
write-mode ``open``, ``os.rename``/``os.replace``, ``os.mkdir``, ``os.remove``,
``shutil.*``, ``tempfile.mk*temp`` and ``subprocess.Popen`` with the first
non-stdlib stack frame, e.g. ``.../triton/runtime/cache.py:123``. That turns
"a file appeared under ~/.triton" into "triton/runtime/cache.py:123 wrote it
via os.replace", which is also an empirical check of the knowledge base's
source citations.

Cross-platform and unprivileged; blind to writes done in C/C++ (ninja, nvcc,
DeepGEMM's host JIT), which the snapshot/strace/inotify tracers cover.
Processes started with ``python -I`` or ``-E`` ignore ``PYTHONPATH`` and are
not recorded.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

from . import FileEvent

ENV_LOG = "CACHE_AUDIT_PYLOG"

SITECUSTOMIZE = r'''
# Installed by cache-audit. Records file writes made by Python code.
import os as _os
import sys as _sys


def _cache_audit_install():
    log_dir = _os.environ.get("CACHE_AUDIT_PYLOG")
    if not log_dir:
        return
    import json, sysconfig, threading, time
    try:
        _os.makedirs(log_dir, exist_ok=True)
        f = open(_os.path.join(log_dir, "%d.jsonl" % _os.getpid()), "a", buffering=1)
    except OSError:
        return
    here = _os.path.dirname(_os.path.abspath(__file__))
    std = []
    for k in ("stdlib", "platstdlib"):
        p = sysconfig.get_paths().get(k)
        if p:
            std.append(_os.path.realpath(p))
            std.append(p)
    skip = tuple(std) + (here, "<frozen")
    lock = threading.Lock()
    W = _os.O_WRONLY | _os.O_RDWR | _os.O_CREAT | _os.O_TRUNC | getattr(_os, "O_APPEND", 0)
    tl = threading.local()

    def frames():
        fr = _sys._getframe(2)
        out = []
        while fr is not None and len(out) < 3:
            fn = fr.f_code.co_filename
            if not fn.startswith(skip):
                out.append("%s:%d" % (fn, fr.f_lineno))
            fr = fr.f_back
        return out

    def p(x):
        if isinstance(x, int) or x is None:
            return None
        x = _os.fsdecode(x)
        return x if _os.path.isabs(x) else _os.path.abspath(x)

    def hook(event, args):
        if getattr(tl, "on", False):
            return
        try:
            if event == "open":
                path, mode, flags = args
                w = (mode is not None and any(c in mode for c in "wax+")) or (
                    mode is None and flags is not None and flags & W)
                if not w or p(path) is None:
                    return
                rec = {"ev": "open_w", "path": p(path)}
            elif event in ("os.rename", "os.link", "os.symlink", "shutil.copyfile",
                           "shutil.copytree", "shutil.move"):
                rec = {"ev": event, "src": p(args[0]), "path": p(args[1])}
            elif event in ("os.mkdir", "os.remove", "os.rmdir", "os.truncate"):
                if p(args[0]) is None:
                    return
                rec = {"ev": event, "path": p(args[0])}
            elif event in ("tempfile.mkstemp", "tempfile.mkdtemp"):
                rec = {"ev": event, "path": p(args[0])}
            elif event == "subprocess.Popen":
                exe, a, cwd, env = args
                a = list(a) if isinstance(a, (list, tuple)) else [a]
                rec = {"ev": event, "args": [_os.fsdecode(x) for x in a[:16]],
                       "cwd": _os.fsdecode(cwd) if cwd else _os.getcwd()}
            else:
                return
            tl.on = True
            try:
                rec["pid"] = _os.getpid()
                rec["t"] = time.time()
                rec["by"] = frames()
                line = json.dumps(rec)
                with lock:
                    f.write(line + "\n")
            finally:
                tl.on = False
        except Exception:
            pass

    _sys.addaudithook(hook)

    # Chain to any sitecustomize we shadow.
    try:
        import importlib.machinery, importlib.util
        rest = [d for d in _sys.path if _os.path.abspath(d or ".") != here]
        spec = importlib.machinery.PathFinder.find_spec("sitecustomize", rest)
        if spec and spec.origin and _os.path.abspath(spec.origin) != _os.path.abspath(__file__):
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
    except Exception:
        pass


_cache_audit_install()
'''


def install(hook_dir: str | os.PathLike[str]) -> str:
    """Write the sitecustomize into ``hook_dir``; returns the dir to prepend to PYTHONPATH."""
    d = Path(hook_dir)
    d.mkdir(parents=True, exist_ok=True)
    (d / "sitecustomize.py").write_text(SITECUSTOMIZE)
    return str(d)


def env_for(base_env: dict[str, str], hook_dir: str, log_dir: str) -> dict[str, str]:
    env = dict(base_env)
    env["PYTHONPATH"] = hook_dir + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    env[ENV_LOG] = log_dir
    return env


_OPS = {
    "open_w": "open_w", "os.rename": "rename_to", "os.link": "link_to", "os.symlink": "symlink",
    "os.mkdir": "mkdir", "os.remove": "delete", "os.rmdir": "delete", "os.truncate": "modify",
    "shutil.copyfile": "create", "shutil.copytree": "create", "shutil.move": "rename_to",
    "tempfile.mkstemp": "tmp_create", "tempfile.mkdtemp": "tmp_mkdir",
}


def collect(log_dir: str) -> tuple[list[FileEvent], list[dict]]:
    """Read every ``<pid>.jsonl`` written during the run."""
    events: list[FileEvent] = []
    procs: list[dict] = []
    d = Path(log_dir)
    if not d.is_dir():
        return events, procs
    for f in sorted(d.glob("*.jsonl")):
        for line in f.read_text(errors="replace").splitlines():
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            if rec.get("ev") == "subprocess.Popen":
                procs.append(rec)
                continue
            op = _OPS.get(rec.get("ev", ""))
            if not op or not rec.get("path"):
                continue
            by = rec.get("by") or []
            events.append(FileEvent(rec["path"], op, pid=rec.get("pid"), source="audithook",
                                    writer=by[0] if by else "", detail=rec.get("src") or ""))
    return events, procs
