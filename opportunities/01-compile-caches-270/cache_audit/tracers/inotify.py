"""inotify tracer (Linux): recursive watches on the writable roots, no privilege.

New directories get a watch as soon as their IN_CREATE arrives and are then
scanned, so files created before the watch landed are still reported. If the
kernel queue overflows (IN_Q_OVERFLOW) or the watch limit is hit, the
recording is marked incomplete; the snapshot diff still covers persistence.
"""

from __future__ import annotations

import ctypes
import ctypes.util
import errno
import os
import select
import struct
import sys
import threading

from . import FileEvent

IN_MODIFY = 0x00000002
IN_ATTRIB = 0x00000004
IN_CLOSE_WRITE = 0x00000008
IN_MOVED_FROM = 0x00000040
IN_MOVED_TO = 0x00000080
IN_CREATE = 0x00000100
IN_DELETE = 0x00000200
IN_Q_OVERFLOW = 0x00004000
IN_IGNORED = 0x00008000
IN_ISDIR = 0x40000000
IN_ONLYDIR = 0x01000000
IN_NONBLOCK = 0o4000
IN_CLOEXEC = 0o2000000

MASK = IN_MODIFY | IN_CLOSE_WRITE | IN_MOVED_FROM | IN_MOVED_TO | IN_CREATE | IN_DELETE
_HDR = struct.Struct("iIII")


def available() -> bool:
    return sys.platform.startswith("linux") and _libc() is not None


def _libc():
    name = ctypes.util.find_library("c")
    if not name:
        return None
    try:
        lib = ctypes.CDLL(name, use_errno=True)
        lib.inotify_init1  # noqa: B018
        return lib
    except (OSError, AttributeError):
        return None


class Watcher:
    def __init__(self, roots: list[str], exclude: list[str] | None = None):
        self.libc = _libc()
        if self.libc is None:
            raise OSError("inotify unavailable")
        self.fd = self.libc.inotify_init1(IN_NONBLOCK | IN_CLOEXEC)
        if self.fd < 0:
            raise OSError(ctypes.get_errno(), "inotify_init1")
        self.wd_path: dict[int, str] = {}
        self.exclude = [os.path.normpath(e) for e in (exclude or [])]
        self.events: list[FileEvent] = []
        self.incomplete = ""
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        for r in roots:
            if os.path.isdir(r):
                self._add_tree(os.path.normpath(r), report=False)

    def _excluded(self, p: str) -> bool:
        return any(p == e or p.startswith(e + "/") for e in self.exclude)

    def _add(self, d: str) -> bool:
        wd = self.libc.inotify_add_watch(self.fd, os.fsencode(d), MASK | IN_ONLYDIR)
        if wd < 0:
            e = ctypes.get_errno()
            if e == errno.ENOSPC:
                self.incomplete = "inotify watch limit reached (fs.inotify.max_user_watches)"
            return False
        self.wd_path[wd] = d
        return True

    def _add_tree(self, root: str, report: bool) -> None:
        stack = [root]
        while stack:
            d = stack.pop()
            if self._excluded(d) or not self._add(d):
                continue
            try:
                with os.scandir(d) as it:
                    for ent in it:
                        if ent.is_dir(follow_symlinks=False):
                            stack.append(ent.path)
                            if report:
                                self.events.append(FileEvent(ent.path, "mkdir", source="inotify"))
                        elif report:  # created before our watch on its directory landed
                            self.events.append(FileEvent(ent.path, "create", source="inotify"))
            except OSError:
                pass

    def start(self) -> "Watcher":
        self._thread.start()
        return self

    def stop(self) -> list[FileEvent]:
        self._stop.set()
        self._thread.join(timeout=10)
        self._drain()
        os.close(self.fd)
        return self.events

    def _loop(self) -> None:
        while not self._stop.is_set():
            r, _, _ = select.select([self.fd], [], [], 0.2)
            if r:
                self._drain()

    def _drain(self) -> None:
        while True:
            try:
                buf = os.read(self.fd, 1 << 16)
            except BlockingIOError:
                return
            except OSError:
                return
            if not buf:
                return
            off = 0
            while off + _HDR.size <= len(buf):
                wd, mask, cookie, ln = _HDR.unpack_from(buf, off)
                name = buf[off + _HDR.size: off + _HDR.size + ln].rstrip(b"\0").decode("utf-8", "surrogateescape")
                off += _HDR.size + ln
                if mask & IN_Q_OVERFLOW:
                    self.incomplete = "inotify queue overflow"
                    continue
                if mask & IN_IGNORED:
                    continue
                base = self.wd_path.get(wd)
                if base is None:
                    continue
                path = os.path.join(base, name) if name else base
                if mask & IN_ISDIR:
                    if mask & (IN_CREATE | IN_MOVED_TO):
                        self.events.append(FileEvent(path, "mkdir", source="inotify"))
                        self._add_tree(path, report=True)
                    continue
                if mask & IN_CREATE:
                    op = "create"
                elif mask & IN_MOVED_TO:
                    op = "rename_to"
                elif mask & IN_MOVED_FROM:
                    op = "rename_from"
                elif mask & IN_DELETE:
                    op = "delete"
                elif mask & (IN_CLOSE_WRITE | IN_MODIFY):
                    op = "modify"
                else:
                    continue
                self.events.append(FileEvent(path, op, source="inotify", detail=str(cookie) if cookie else ""))
