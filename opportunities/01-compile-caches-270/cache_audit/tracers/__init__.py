"""Unprivileged ways to record which files a command writes.

All tracers produce the same thing: a set of ``FileEvent``s (absolute path,
operation, pid if known, writer attribution if known). ``record()`` runs the
command under the best available tracer and always adds a before/after
snapshot diff, which is the ground truth for *what persisted*; the live
tracers (strace, inotify, Python audit hooks) add *who wrote it* and catch
transient files and failed writes (e.g. EROFS on a read-only cache).
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class FileEvent:
    path: str
    op: str                      # create | modify | delete | rename_to | mkdir | write_failed
    pid: int | None = None
    source: str = ""             # snapshot | strace | inotify | audithook
    writer: str = ""             # e.g. "triton/runtime/cache.py:123" or "ninja"
    detail: str = ""             # errno for failures, rename source, etc.


@dataclass
class Recording:
    events: list[FileEvent] = field(default_factory=list)
    processes: dict[int, str] = field(default_factory=dict)   # pid -> argv (strace execve)
    subprocesses: list[dict] = field(default_factory=list)    # audithook subprocess.Popen records
    tracers: list[str] = field(default_factory=list)
    returncode: int | None = None
    wall_s: float = 0.0
    notes: list[str] = field(default_factory=list)

    def persisted(self) -> set[str]:
        """Paths that exist after the run and were created or modified by it."""
        snap = {e.path for e in self.events if e.source == "snapshot" and e.op in ("create", "modify")}
        return snap
