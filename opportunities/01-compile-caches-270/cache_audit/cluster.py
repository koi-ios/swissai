"""Cluster persisted files into cache roots and attach evidence to each root.

Known roots come from the knowledge base (longest pattern match). Files no KB
entry claims are grouped by where they live: ``$HOME/.cache/<x>``,
``$HOME/.local/share/<x>``, ``$HOME/.<x>``, ``$TMPDIR/<x>``, ``/tmp/<x>``,
``/dev/shm/<x>`` or ``<cwd>/<x>``; each group is an UNKNOWN root to look at.

From the live tracers each root also gets *how* it was written: the share of
its files that arrived through ``rename``/``link`` (the temp-file + atomic
publish idiom) versus direct ``open(O_WRONLY|O_CREAT)``, and the writers
(Python ``file:line`` from the audit hook, or the process from strace).
"""

from __future__ import annotations

import os
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Iterable, Mapping

from .classify import Contents
from .kb import KB, CacheEntry
from .tracers import FileEvent

_TWO_LEVEL = {".cache", ".config", "Library", ".local"}


@dataclass
class Root:
    path: str
    entry: CacheEntry | None
    anchor: str                       # HOME, TMPDIR, /tmp, cwd, ...
    files: list[str] = field(default_factory=list)
    contents: Contents = field(default_factory=Contents)
    writers: Counter = field(default_factory=Counter)
    ops: Counter = field(default_factory=Counter)
    failed_writes: list[str] = field(default_factory=list)

    @property
    def label(self) -> str:
        return self.entry.id if self.entry else "UNKNOWN"

    def write_pattern(self) -> str:
        """Empirical write idiom, from live-tracer events on this root's files."""
        atomic = self.ops.get("rename_to", 0) + self.ops.get("link_to", 0)
        direct = self.ops.get("open_w", 0)
        if atomic == 0 and direct == 0:
            return "no live trace"
        if atomic and atomic >= 0.5 * len(self.files):
            return f"atomic publish ({atomic} renames/links, {direct} direct opens)"
        if atomic:
            return f"mixed ({atomic} renames/links, {direct} direct opens)"
        return f"direct writes ({direct} opens, no rename)"


def anchors_for(env: Mapping[str, str], cwd: str, extra: Iterable[str] = ()) -> list[tuple[str, str]]:
    """(name, path) pairs, most specific first."""
    out = []
    for name in ("XDG_CACHE_HOME", "TMPDIR", "HOME"):
        if env.get(name):
            out.append((name, os.path.normpath(env[name])))
    out += [("/tmp", "/tmp"), ("/var/tmp", "/var/tmp"), ("/dev/shm", "/dev/shm"), ("cwd", os.path.normpath(cwd))]
    out += [(p, os.path.normpath(p)) for p in extra]
    # most specific (longest) first so $HOME/.cache wins over $HOME
    return sorted(out, key=lambda kv: -len(kv[1]))


def _unknown_root(path: str, anchors: list[tuple[str, str]]) -> tuple[str, str]:
    for name, a in anchors:
        if path == a or path.startswith(a.rstrip("/") + "/"):
            rel = os.path.relpath(path, a).split(os.sep)
            if len(rel) == 1:  # a file directly in the anchor
                return name, path
            depth = 1
            if name == "HOME" and rel[0] in _TWO_LEVEL:
                depth = 3 if rel[:2] == [".local", "share"] else 2
            depth = min(depth, len(rel) - 1)
            return name, os.path.join(a, *rel[:depth])
    return "other", os.path.dirname(path)


def cluster(paths: Iterable[str], kb: KB, env: Mapping[str, str], cwd: str,
            events: Iterable[FileEvent] = (), needles: Iterable[bytes] = (),
            extra_anchors: Iterable[str] = (), ignore: Iterable[str] = ()) -> list[Root]:
    anchors = anchors_for(env, cwd, extra_anchors)
    ignore = tuple(os.path.normpath(i) for i in ignore)
    roots: dict[str, Root] = {}
    memo: dict = {}
    needles = list(needles)
    for p in sorted(set(paths)):
        if any(p == i or p.startswith(i + "/") for i in ignore):
            continue
        m = kb.match(p, env, memo)
        if m:
            entry, rpath = m
            anchor = next((n for n, a in anchors if rpath.startswith(a.rstrip("/") + "/") or rpath == a), "other")
        else:
            entry = None
            anchor, rpath = _unknown_root(p, anchors)
        r = roots.get(rpath)
        if r is None:
            r = roots[rpath] = Root(rpath, entry, anchor)
        r.files.append(p)
        r.contents.add(p, needles)

    # live-tracer evidence: ops and writers per root
    by_file: dict[str, Root] = {f: r for r in roots.values() for f in r.files}
    dir_roots = sorted(roots.values(), key=lambda r: -len(r.path))
    for e in events:
        if e.source == "snapshot":
            continue
        r = by_file.get(e.path)
        if r is None:
            r = next((x for x in dir_roots if e.path.startswith(x.path.rstrip("/") + "/")), None)
        if r is None:
            continue
        r.ops[e.op] += 1
        if e.op == "write_failed":
            r.failed_writes.append(f"{e.path} ({e.detail})")
        if e.writer:
            r.writers[_short_writer(e.writer)] += 1
    return sorted(roots.values(), key=lambda r: (r.entry is None, -r.contents.bytes))


def _short_writer(w: str) -> str:
    """Trim a site-packages path to ``pkg/module.py:line``; keep process names short."""
    for marker in ("site-packages/", "dist-packages/"):
        if marker in w:
            return w.split(marker, 1)[1]
    if " " in w:  # strace argv
        exe = w.split(" ", 1)[0]
        return os.path.basename(exe.strip('"'))
    return w


def failed_outside(events: Iterable[FileEvent], roots: list[Root]) -> list[FileEvent]:
    """Failed writes not under any root (e.g. EROFS on a read-only cache dir)."""
    rp = [r.path.rstrip("/") + "/" for r in roots]
    return [e for e in events if e.op == "write_failed" and not any(e.path.startswith(x) for x in rp)]


def group_events_by_dir(events: Iterable[FileEvent]) -> dict[str, Counter]:
    out: dict[str, Counter] = defaultdict(Counter)
    for e in events:
        out[os.path.dirname(e.path)][e.op] += 1
    return out
