"""strace tracer: complete per-process write log, unprivileged (ptrace of own children).

Runs ``strace -f -qq -y -s 4096 -o LOG -e trace=<file-writing syscalls>`` and
parses the log. ``-y`` makes strace print the path behind every fd, so a
successful ``openat`` reports its absolute path in the return value and
``*at`` dirfds come with their directory. Relative paths with ``AT_FDCWD``
are resolved against a per-pid working directory tracked through
``chdir``/``fchdir`` and inherited across ``clone``/``fork``.

Works in containers where ptrace of descendants is allowed (Docker's default
seccomp profile on kernels >= 4.8, enroot, gVisor); ``available()`` checks.
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
from dataclasses import dataclass, field

from . import FileEvent

SYSCALLS = [
    "?open", "openat", "?openat2", "?creat", "?mkdir", "mkdirat", "?rename", "renameat",
    "?renameat2", "?link", "linkat", "?symlink", "symlinkat", "?unlink", "unlinkat",
    "?rmdir", "truncate", "chdir", "fchdir", "execve", "?execveat",
    "clone", "?clone3", "?fork", "?vfork",
]

_WRITE_FLAGS = ("O_WRONLY", "O_RDWR", "O_CREAT", "O_TRUNC", "O_APPEND")
_DENIED = {"EACCES", "EROFS", "EPERM", "EDQUOT", "ENOSPC", "EXDEV"}

_LINE = re.compile(r"^(\d+)\s+(.*)$")
_CALL = re.compile(r"^(\w+)\((.*)\)\s+=\s+(-?\d+|\?|0x[0-9a-fA-F]+)(.*)$", re.S)
_RESUMED = re.compile(r"^<\.\.\. (\w+) resumed>(.*)$", re.S)
_FD = re.compile(r"^(-?\d+|AT_FDCWD)(?:<(.*)>)?$", re.S)


def available() -> bool:
    exe = shutil.which("strace")
    if not exe:
        return False
    try:
        r = subprocess.run([exe, "-f", "-qq", "-o", os.devnull, "true"],
                           capture_output=True, timeout=20)
        return r.returncode == 0
    except Exception:
        return False


def command(log_path: str, argv: list[str]) -> list[str]:
    return ["strace", "-f", "-qq", "-y", "-s", "4096", "-o", log_path,
            "-e", "trace=" + ",".join(SYSCALLS), "--", *argv]


def _split_args(s: str) -> list[str]:
    """Split a strace argument list on top-level commas."""
    out, cur, depth, in_str, esc = [], [], 0, False, False
    for ch in s:
        if in_str:
            cur.append(ch)
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
        elif ch in "([{<":
            depth += 1
        elif ch in ")]}>":
            depth -= 1
        elif ch == "," and depth == 0:
            out.append("".join(cur).strip())
            cur = []
            continue
        cur.append(ch)
    if cur or out:
        out.append("".join(cur).strip())
    return out


def _cstring(tok: str) -> str | None:
    """Decode a strace C string literal (``"a\\nb"...``) to str."""
    tok = tok.strip()
    if tok.endswith("..."):
        tok = tok[:-3]
    if not (tok.startswith('"') and tok.endswith('"')):
        return None
    body = tok[1:-1]
    out = bytearray()
    i = 0
    simple = {"n": 10, "t": 9, "r": 13, "v": 11, "f": 12, "\\": 92, '"': 34, "'": 39, "a": 7, "b": 8}
    while i < len(body):
        c = body[i]
        if c != "\\":
            out += c.encode()
            i += 1
            continue
        nxt = body[i + 1] if i + 1 < len(body) else ""
        if nxt == "x":
            out.append(int(body[i + 2:i + 4], 16))
            i += 4
        elif nxt.isdigit():
            j = i + 1
            while j < len(body) and j < i + 4 and body[j] in "01234567":
                j += 1
            out.append(int(body[i + 1:j], 8))
            i = j
        else:
            out.append(simple.get(nxt, ord(nxt) if nxt else 92))
            i += 2
    return out.decode("utf-8", "surrogateescape")


@dataclass
class _State:
    cwd: dict[int, str] = field(default_factory=dict)
    pending: dict[int, str] = field(default_factory=dict)
    procs: dict[int, str] = field(default_factory=dict)
    parent_of: dict[int, int] = field(default_factory=dict)


def _dirfd(tok: str, pid: int, st: _State) -> str | None:
    m = _FD.match(tok.strip())
    if not m:
        return None
    if m.group(2):  # -y decorates fds (and, in newer strace, AT_FDCWD) with their path
        return m.group(2)
    if m.group(1) == "AT_FDCWD":
        return st.cwd.get(pid)
    return None


def _join(base: str | None, p: str | None) -> str | None:
    if p is None:
        return None
    if os.path.isabs(p):
        return os.path.normpath(p)
    if base is None:
        return None
    return os.path.normpath(os.path.join(base, p))


def parse(text: str, root_pid_cwd: str) -> tuple[list[FileEvent], dict[int, str]]:
    """Parse an ``strace -f -y -o`` log into FileEvents and pid->argv."""
    st = _State()
    events: list[FileEvent] = []
    first_pid: int | None = None

    def cwd_of(pid: int) -> str:
        if pid not in st.cwd:
            parent = st.parent_of.get(pid)
            st.cwd[pid] = st.cwd.get(parent, root_pid_cwd) if parent else root_pid_cwd
        return st.cwd[pid]

    for raw in text.splitlines():
        m = _LINE.match(raw)
        if not m:
            continue
        pid, rest = int(m.group(1)), m.group(2)
        if first_pid is None:
            first_pid = pid
            st.cwd[pid] = root_pid_cwd
        cwd_of(pid)
        if rest.startswith(("+++", "---")):
            continue
        r = _RESUMED.match(rest)
        if r:
            head = st.pending.pop(pid, None)
            if head is None:
                continue
            rest = head + r.group(2)
        if rest.endswith("<unfinished ...>"):
            st.pending[pid] = rest[: -len("<unfinished ...>")].rstrip()
            continue
        c = _CALL.match(rest)
        if not c:
            continue
        name, argstr, ret, tail = c.group(1), c.group(2), c.group(3), c.group(4)
        args = _split_args(argstr)
        ok = ret not in ("-1", "?")
        err = ""
        if not ok:
            em = re.search(r"\b(E[A-Z]+)\b", tail)
            err = em.group(1) if em else "ERR"
        _apply(name, args, ok, err, ret, tail, pid, st, events)

    return events, st.procs


def _apply(name, args, ok, err, ret, tail, pid, st: _State, events: list[FileEvent]) -> None:
    def ev(path, op, detail=""):
        if path:
            events.append(FileEvent(path, op, pid=pid, source="strace",
                                    writer=st.procs.get(pid, ""), detail=detail))

    if name in ("clone", "clone3", "fork", "vfork") and ok:
        try:
            child = int(ret)
        except ValueError:
            return
        st.parent_of[child] = pid
        st.cwd[child] = st.cwd.get(pid, "/")
        st.procs.setdefault(child, st.procs.get(pid, ""))
        return
    if name in ("execve", "execveat") and ok:
        exe = _cstring(args[0]) if args else None
        argv = args[1] if len(args) > 1 else ""
        st.procs[pid] = f"{exe} {argv}".strip()
        return
    if name == "chdir" and ok:
        p = _join(st.cwd.get(pid), _cstring(args[0]))
        if p:
            st.cwd[pid] = p
        return
    if name == "fchdir" and ok:
        p = _dirfd(args[0], pid, st)
        if p:
            st.cwd[pid] = p
        return

    if name in ("open", "creat", "openat", "openat2"):
        if name == "openat" or name == "openat2":
            base, path_tok, flags = _dirfd(args[0], pid, st), args[1], (args[2] if len(args) > 2 else "")
        else:
            base, path_tok = st.cwd.get(pid), args[0]
            flags = "O_CREAT|O_WRONLY|O_TRUNC" if name == "creat" else (args[1] if len(args) > 1 else "")
        if not any(f in flags for f in _WRITE_FLAGS):
            return
        path = _join(base, _cstring(path_tok))
        if ok:
            fm = re.match(r"\s*<(.*)>", tail)  # -y: "= 3</abs/path>"
            ev(fm.group(1) if fm else path, "open_w", detail=flags)
        elif err in _DENIED:
            ev(path, "write_failed", detail=err)
        return
    if name in ("mkdir", "mkdirat"):
        base = st.cwd.get(pid) if name == "mkdir" else _dirfd(args[0], pid, st)
        path = _join(base, _cstring(args[0 if name == "mkdir" else 1]))
        if ok:
            ev(path, "mkdir")
        elif err in _DENIED:
            ev(path, "write_failed", detail=err)
        return
    if name in ("rename", "renameat", "renameat2", "link", "linkat"):
        if name in ("rename", "link"):
            src = _join(st.cwd.get(pid), _cstring(args[0]))
            dst = _join(st.cwd.get(pid), _cstring(args[1]))
        else:
            src = _join(_dirfd(args[0], pid, st), _cstring(args[1]))
            dst = _join(_dirfd(args[2], pid, st), _cstring(args[3]))
        op = "rename_to" if name.startswith("rename") else "link_to"
        if ok:
            ev(dst, op, detail=src or "")
            if op == "rename_to":
                ev(src, "rename_from", detail=dst or "")
        elif err in _DENIED:
            ev(dst, "write_failed", detail=err)
        elif err == "EEXIST":
            ev(dst, "exists", detail=src or "")
        return
    if name in ("symlink", "symlinkat"):
        if name == "symlink":
            dst = _join(st.cwd.get(pid), _cstring(args[1]))
        else:
            dst = _join(_dirfd(args[1], pid, st), _cstring(args[2]))
        if ok:
            ev(dst, "symlink", detail=_cstring(args[0]) or "")
        return
    if name in ("unlink", "unlinkat", "rmdir"):
        if name == "unlinkat":
            path = _join(_dirfd(args[0], pid, st), _cstring(args[1]))
        else:
            path = _join(st.cwd.get(pid), _cstring(args[0]))
        if ok:
            ev(path, "delete")
        return
    if name == "truncate" and ok:
        ev(_join(st.cwd.get(pid), _cstring(args[0])), "modify")
