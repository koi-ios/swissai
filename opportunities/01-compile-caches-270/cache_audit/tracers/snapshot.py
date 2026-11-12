"""Before/after snapshots of writable roots: always available, needs no privilege.

Records (size, mtime_ns, inode) per file. A file is *created* if it is new,
*modified* if any of the three changed, *deleted* if it vanished. Temporary
files that were created and removed during the run are invisible here, which
is what we want for "what persisted"; the live tracers see them.
"""

from __future__ import annotations

import os
import stat
from dataclasses import dataclass
from typing import Iterable

from . import FileEvent

# Pseudo or huge filesystems never worth walking.
_SKIP_PREFIXES = ("/proc", "/sys", "/dev/pts", "/run/udev")


@dataclass(frozen=True)
class FileState:
    size: int
    mtime_ns: int
    ino: int
    is_link: bool = False


def take(roots: Iterable[str], exclude: Iterable[str] = (), max_entries: int = 3_000_000,
         same_device: bool = True) -> dict[str, FileState]:
    """Walk ``roots`` (no symlink following) and record every non-directory entry."""
    exclude = tuple(os.path.normpath(e) for e in exclude)
    out: dict[str, FileState] = {}
    seen_roots: list[str] = []
    for root in roots:
        root = os.path.normpath(root)
        if not os.path.isdir(root) or any(_under(root, r) for r in seen_roots):
            continue
        seen_roots.append(root)
        try:
            root_dev = os.stat(root).st_dev
        except OSError:
            continue
        stack = [root]
        while stack:
            d = stack.pop()
            if d.startswith(_SKIP_PREFIXES) or any(_under(d, e) for e in exclude):
                continue
            try:
                it = os.scandir(d)
            except OSError:
                continue
            with it:
                for ent in it:
                    try:
                        st = ent.stat(follow_symlinks=False)
                    except OSError:
                        continue
                    if stat.S_ISDIR(st.st_mode):
                        if same_device and st.st_dev != root_dev:
                            continue  # do not wander into other mounts (e.g. Lustre under /)
                        stack.append(ent.path)
                        continue
                    out[ent.path] = FileState(st.st_size, st.st_mtime_ns, st.st_ino,
                                              stat.S_ISLNK(st.st_mode))
                    if len(out) >= max_entries:
                        raise RuntimeError(f"snapshot exceeded {max_entries} entries under {roots}")
    return out


def diff(before: dict[str, FileState], after: dict[str, FileState]) -> list[FileEvent]:
    ev: list[FileEvent] = []
    for p, st in after.items():
        old = before.get(p)
        if old is None:
            ev.append(FileEvent(p, "create", source="snapshot"))
        elif old != st:
            ev.append(FileEvent(p, "modify", source="snapshot"))
    for p in before.keys() - after.keys():
        ev.append(FileEvent(p, "delete", source="snapshot"))
    return ev


def _under(path: str, root: str) -> bool:
    return path == root or path.startswith(root.rstrip("/") + "/")
