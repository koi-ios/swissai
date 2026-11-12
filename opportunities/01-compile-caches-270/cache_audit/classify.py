"""Classify cache contents and detect absolute paths baked into them.

Classes: binary (ELF/Mach-O .so, .cubin, fatbin), ir (.ptx, .ttir, .ttgir, .llir,
.mlir), source (generated .cu/.cpp/.py), build (ninja files, objects), autotune
(tuning results), serialized (pickled graphs, .bin/.pt artifacts), metadata
(.json/.txt), lock, bytecode (.pyc), temp, other.

``embeds_path`` reports whether a file contains one of the sandbox's absolute
roots: such a cache is not relocatable as-is (ninja build trees, JSON metadata
with absolute source paths), which is what forces servekit to copy its caches
to a fixed ``/tmp`` path, and what decides between a fixed-path per-job overlay
and a shared directory.
"""

from __future__ import annotations

import os
import re
from collections import Counter
from dataclasses import dataclass, field
from typing import Iterable

_ELF = b"\x7fELF"
_MACHO = (b"\xcf\xfa\xed\xfe", b"\xce\xfa\xed\xfe", b"\xca\xfe\xba\xbe")
_FATBIN = b"\x50\xed\x55\xba"  # 0xBA55ED50 little endian

_EXT = {
    ".so": "binary", ".cubin": "binary", ".fatbin": "binary", ".hsaco": "binary", ".dylib": "binary",
    ".ptx": "ir", ".ttir": "ir", ".ttgir": "ir", ".llir": "ir", ".mlir": "ir", ".ll": "ir",
    ".amdgcn": "ir", ".sass": "ir", ".bc": "ir",
    ".cu": "source", ".cuh": "source", ".cpp": "source", ".cc": "source", ".c": "source",
    ".h": "source", ".hpp": "source", ".py": "source", ".cl": "source",
    ".o": "build", ".obj": "build", ".d": "build", ".a": "build",
    ".pkl": "serialized", ".pickle": "serialized", ".pt": "serialized", ".bin": "serialized",
    ".pt2": "serialized", ".bundle": "serialized",
    ".json": "metadata", ".txt": "metadata", ".yaml": "metadata", ".toml": "metadata",
    ".lock": "lock", ".lck": "lock",
    ".pyc": "bytecode",
}
_BUILD_NAMES = {"build.ninja", ".ninja_log", ".ninja_deps", "CMakeCache.txt", "Makefile"}
_AUTOTUNE = re.compile(r"(autotun|best_config|tuning|tuned|\.autotune|configs?_\w+\.json)", re.I)
# Python's tempfile names are "tmp" + 8 chars of [a-z0-9_] (mkstemp, NamedTemporaryFile, mkdtemp).
_TEMP = re.compile(r"(\.tmp|^tmp[a-z0-9_]{8}(\.|$)|tmp[_.-]?[0-9a-z]{6,}|\.partial|\.swp|~$|\.\d{3,}\.\w{6,}$)", re.I)


def classify(path: str, head: bytes | None = None) -> str:
    name = os.path.basename(path)
    low = name.lower()
    if name in _BUILD_NAMES:
        return "build"
    if low.endswith(".lock") or name in ("lock", "LOCK", ".lock") or low.endswith(".lck"):
        return "lock"
    if _TEMP.search(name):
        return "temp"
    if _AUTOTUNE.search(path) and low.endswith((".json", ".best_config", ".txt", ".pkl")):
        return "autotune"
    if low.endswith(".best_config"):
        return "autotune"
    ext = os.path.splitext(low)[1]
    if ext in _EXT:
        return _EXT[ext]
    if head is None:
        head = _head(path)
    if head.startswith(_ELF) or head.startswith(_MACHO) or head.startswith(_FATBIN):
        return "binary"
    return "other"


def _head(path: str, n: int = 8) -> bytes:
    try:
        with open(path, "rb") as f:
            return f.read(n)
    except OSError:
        return b""


def embeds_path(path: str, needles: Iterable[bytes], max_bytes: int = 8 << 20) -> bool:
    """True if the file contains any of ``needles`` (absolute sandbox roots)."""
    needles = [n for n in needles if n]
    if not needles:
        return False
    try:
        size = os.path.getsize(path)
        if size > max_bytes:
            return False
        with open(path, "rb") as f:
            data = f.read()
    except OSError:
        return False
    return any(n in data for n in needles)


@dataclass
class Contents:
    files: int = 0
    bytes: int = 0
    by_class: Counter = field(default_factory=Counter)
    bytes_by_class: Counter = field(default_factory=Counter)
    path_baked: list[str] = field(default_factory=list)

    def add(self, path: str, needles: Iterable[bytes] = ()) -> None:
        try:
            size = os.lstat(path).st_size
        except OSError:
            return
        cls = classify(path)
        self.files += 1
        self.bytes += size
        self.by_class[cls] += 1
        self.bytes_by_class[cls] += size
        if cls in ("build", "metadata", "source", "binary", "other", "serialized") and embeds_path(path, needles):
            self.path_baked.append(path)

    def summary(self) -> str:
        parts = [f"{k} {v}" for k, v in self.by_class.most_common()]
        return ", ".join(parts)


def human(n: int) -> str:
    x = float(n)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if x < 1024 or unit == "TB":
            return f"{x:.0f} {unit}" if unit == "B" else f"{x:.1f} {unit}"
        x /= 1024
    return f"{n} B"
