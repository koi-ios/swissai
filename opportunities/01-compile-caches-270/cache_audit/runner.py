"""Run one command in a fresh HOME/TMPDIR sandbox under the best available tracer."""

from __future__ import annotations

import os
import secrets
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

from . import tracers
from .tracers import FileEvent, Recording, audithook, snapshot
from .tracers import inotify as inotify_tracer
from .tracers import strace as strace_tracer


@dataclass
class Sandbox:
    """Fresh, uniquely named HOME/TMPDIR (and optionally XDG) dirs for one run."""

    root: str
    home: str
    tmp: str
    record: str
    xdg: bool = False

    @classmethod
    def create(cls, base: str, tag: str, xdg: bool = False) -> "Sandbox":
        root = os.path.join(os.path.abspath(base), f"{tag}-{secrets.token_hex(3)}")
        home, tmp, rec = (os.path.join(root, x) for x in ("home", "tmp", ".record"))
        for d in (home, tmp, rec):
            os.makedirs(d, exist_ok=True)
        if xdg:
            for x in ("xdg-cache", "xdg-config", "xdg-data", "xdg-state"):
                os.makedirs(os.path.join(root, x), exist_ok=True)
        return cls(root, home, tmp, rec, xdg)

    def fresh_home(self, tag: str) -> "Sandbox":
        """New HOME/TMPDIR under the same root (an ephemeral job HOME, same cache)."""
        home = os.path.join(self.root, f"home-{tag}")
        tmp = os.path.join(self.root, f"tmp-{tag}")
        rec = os.path.join(self.root, f".record-{tag}")
        for d in (home, tmp, rec):
            os.makedirs(d, exist_ok=True)
        return Sandbox(self.root, home, tmp, rec, self.xdg)

    def env(self, base: Mapping[str, str]) -> dict[str, str]:
        env = dict(base)
        env.update(HOME=self.home, TMPDIR=self.tmp, TMP=self.tmp, TEMP=self.tmp)
        if self.xdg:
            env.update(XDG_CACHE_HOME=os.path.join(self.root, "xdg-cache"),
                       XDG_CONFIG_HOME=os.path.join(self.root, "xdg-config"),
                       XDG_DATA_HOME=os.path.join(self.root, "xdg-data"),
                       XDG_STATE_HOME=os.path.join(self.root, "xdg-state"))
        return env

    def needles(self) -> list[bytes]:
        """Byte strings whose presence in a cache file means 'absolute path baked in'."""
        out = {self.root, os.path.realpath(self.root)}
        return [os.fsencode(x) for x in sorted(out)]


def pick_tracer(requested: str) -> str:
    if requested != "auto":
        return requested
    if strace_tracer.available():
        return "strace"
    if inotify_tracer.available():
        return "inotify"
    return "snapshot"


def default_watch_roots(sb: Sandbox, cwd: str, extra: list[str]) -> list[str]:
    roots = [sb.root, cwd, "/dev/shm", "/var/tmp"]
    tmp = "/tmp"
    if not (os.path.realpath(sb.root) + "/").startswith(os.path.realpath(tmp) + "/"):
        roots.append(tmp)
    roots += extra
    seen, out = set(), []
    for r in roots:
        rp = os.path.realpath(r)
        if os.path.isdir(rp) and rp not in seen:
            seen.add(rp)
            out.append(r)
    return out


def record(argv: list[str], env: Mapping[str, str], sb: Sandbox, *, cwd: str | None = None,
           tracer: str = "auto", audit_hook: bool = True, watch: list[str] | None = None,
           timeout: float | None = None, log_name: str = "run") -> Recording:
    cwd = os.path.abspath(cwd or os.getcwd())
    tracer = pick_tracer(tracer)
    rec = Recording(tracers=[tracer] + (["audithook"] if audit_hook else []) +
                    (["snapshot"] if tracer != "snapshot" else []))
    roots = default_watch_roots(sb, cwd, watch or [])
    exclude = [sb.record, os.path.realpath(sb.record)]

    run_env = dict(env)
    pylog = os.path.join(sb.record, f"py-{log_name}")
    if audit_hook:
        hook_dir = audithook.install(os.path.join(sb.record, "pyhook"))
        run_env = audithook.env_for(run_env, hook_dir, pylog)

    before = snapshot.take(roots, exclude=exclude)
    watcher = None
    if tracer == "inotify":
        watcher = inotify_tracer.Watcher(roots, exclude=exclude).start()
    cmd = list(argv)
    slog = os.path.join(sb.record, f"strace-{log_name}.log")
    if tracer == "strace":
        cmd = strace_tracer.command(slog, cmd)

    out_path = os.path.join(sb.record, f"{log_name}.out")
    t0 = time.monotonic()
    with open(out_path, "wb") as out:
        try:
            p = subprocess.run(cmd, env=run_env, cwd=cwd, stdout=out, stderr=subprocess.STDOUT,
                               timeout=timeout)
            rec.returncode = p.returncode
        except subprocess.TimeoutExpired:
            rec.returncode = -9
            rec.notes.append(f"timeout after {timeout}s")
    rec.wall_s = time.monotonic() - t0

    if watcher is not None:
        rec.events += watcher.stop()
        if watcher.incomplete:
            rec.notes.append(watcher.incomplete)
    after = snapshot.take(roots, exclude=exclude)
    rec.events += snapshot.diff(before, after)
    if tracer == "strace" and os.path.exists(slog):
        ev, procs = strace_tracer.parse(Path(slog).read_text(errors="replace"), cwd)
        rec.events += ev
        rec.processes.update(procs)
    if audit_hook:
        ev, subs = audithook.collect(pylog)
        rec.events += ev
        rec.subprocesses += subs
    # drop events on our own bookkeeping
    rr = (sb.record.rstrip("/") + "/", os.path.realpath(sb.record).rstrip("/") + "/")
    rec.events = [e for e in rec.events if not e.path.startswith(rr)]
    return rec


def tail(sb: Sandbox, log_name: str = "run", n: int = 30) -> str:
    p = os.path.join(sb.record, f"{log_name}.out")
    try:
        lines = Path(p).read_text(errors="replace").splitlines()
    except OSError:
        return ""
    return "\n".join(lines[-n:])


def copy_tree(src: str, dst: str) -> None:
    shutil.copytree(src, dst, symlinks=True, dirs_exist_ok=True)


def set_tree_writable(path: str, writable: bool) -> None:
    for d, dirs, files in os.walk(path):
        for name in dirs + files:
            p = os.path.join(d, name)
            try:
                st = os.lstat(p)
                if os.path.islink(p):
                    continue
                mode = st.st_mode | 0o200 if writable else st.st_mode & ~0o222
                os.chmod(p, mode)
            except OSError:
                pass
    try:
        st = os.stat(path)
        os.chmod(path, st.st_mode | 0o200 if writable else st.st_mode & ~0o222)
    except OSError:
        pass


PY = sys.executable
