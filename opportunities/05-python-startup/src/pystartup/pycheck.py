"""Audit Python bytecode caches: which sources have a .pyc the interpreter would actually use.

CPython (PEP 3147 / PEP 552) loads `__pycache__/<name>.<tag>.pyc` instead of compiling the
source only when the pyc header matches:

* magic number == this interpreter's `importlib.util.MAGIC_NUMBER`;
* timestamp pyc (flags == 0): recorded source mtime and size == the source's current
  `int(st_mtime) & 0xFFFFFFFF` and `st_size & 0xFFFFFFFF`;
* checked-hash pyc (flags == 0b11): SipHash of the current source == recorded hash;
* unchecked-hash pyc (flags == 0b01): always used (unless --check-hash-based-pycs always).

A missing or stale pyc means the module is compiled from source on every import, and the
result is written back only if the directory is writable and PYTHONDONTWRITEBYTECODE is unset.
On a read-only image every process pays the compile; on a writable per-job overlay the first
process of every job pays it.

Usage:
    python -m pystartup.pycheck <dir> [<dir> ...] [--json out.json] [--prefix <pycache prefix>]
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import struct
import sys
from dataclasses import asdict, dataclass, field
from typing import Dict, Iterable, List, Optional

MAGIC = importlib.util.MAGIC_NUMBER
CACHE_TAG = sys.implementation.cache_tag  # e.g. cpython-312

# outcome labels
MISSING = "missing"
STALE = "stale"           # header does not match the source (mtime/size/hash) -> recompiled
BAD_MAGIC = "bad_magic"   # compiled by another interpreter version -> recompiled
VALID_TIMESTAMP = "valid_timestamp"
VALID_CHECKED_HASH = "valid_checked_hash"
VALID_UNCHECKED_HASH = "valid_unchecked_hash"
VALID = {VALID_TIMESTAMP, VALID_CHECKED_HASH, VALID_UNCHECKED_HASH}


def pyc_path_for(source: str, prefix: Optional[str] = None) -> str:
    """Where this interpreter looks for the pyc of `source`.

    With `prefix` (PYTHONPYCACHEPREFIX semantics) the pyc lives at
    <prefix>/<absolute source dir>/<name>.<tag>.pyc, otherwise in __pycache__.
    """
    if prefix is None:
        head, tail = os.path.split(source)
        base = tail.rsplit(".", 1)[0]
        return os.path.join(head, "__pycache__", f"{base}.{CACHE_TAG}.pyc")
    head, tail = os.path.split(os.path.abspath(source))
    base = tail.rsplit(".", 1)[0]
    return os.path.join(prefix, head.lstrip(os.sep), f"{base}.{CACHE_TAG}.pyc")


def classify(source: str, pyc: Optional[str] = None, *, read_source_for_hash: bool = True) -> str:
    """Classify one source file's pyc the way SourceLoader.get_code would."""
    pyc = pyc or pyc_path_for(source)
    try:
        with open(pyc, "rb") as f:
            header = f.read(16)
    except OSError:
        return MISSING
    if len(header) < 16:
        return STALE
    if header[:4] != MAGIC:
        return BAD_MAGIC
    flags = struct.unpack("<I", header[4:8])[0]
    if flags & ~0b11:
        return STALE  # invalid flags: the loader rejects the pyc
    hash_based = bool(flags & 0b1)
    check_source = bool(flags & 0b10)
    if hash_based:
        if not check_source:
            return VALID_UNCHECKED_HASH
        if not read_source_for_hash:
            return VALID_CHECKED_HASH  # optimistic: caller chose not to hash sources
        try:
            with open(source, "rb") as f:
                data = f.read()
        except OSError:
            return STALE
        if importlib.util.source_hash(data) == header[8:16]:
            return VALID_CHECKED_HASH
        return STALE
    try:
        st = os.stat(source)
    except OSError:
        return STALE
    mtime, size = struct.unpack("<II", header[8:16])
    if mtime == (int(st.st_mtime) & 0xFFFFFFFF) and size == (st.st_size & 0xFFFFFFFF):
        return VALID_TIMESTAMP
    return STALE


@dataclass
class AuditResult:
    roots: List[str]
    prefix: Optional[str]
    interpreter: str
    cache_tag: str
    counts: Dict[str, int] = field(default_factory=dict)
    source_bytes: Dict[str, int] = field(default_factory=dict)
    pyc_bytes_total: int = 0
    py_files: int = 0
    py_bytes: int = 0
    examples: Dict[str, List[str]] = field(default_factory=dict)

    @property
    def needs_compile(self) -> int:
        return sum(v for k, v in self.counts.items() if k not in VALID)

    @property
    def needs_compile_bytes(self) -> int:
        return sum(v for k, v in self.source_bytes.items() if k not in VALID)

    def to_dict(self) -> dict:
        d = asdict(self)
        d["needs_compile"] = self.needs_compile
        d["needs_compile_bytes"] = self.needs_compile_bytes
        return d


def iter_sources(roots: Iterable[str]) -> Iterable[str]:
    for root in roots:
        if os.path.isfile(root):
            if root.endswith(".py"):
                yield root
            continue
        for dirpath, dirnames, filenames in os.walk(root):
            dirnames[:] = [d for d in dirnames if d != "__pycache__"]
            for fn in filenames:
                if fn.endswith(".py"):
                    yield os.path.join(dirpath, fn)


def audit(paths: Iterable[str], prefix: Optional[str] = None, *, roots: Optional[List[str]] = None,
          max_examples: int = 5, hash_sources: bool = True) -> AuditResult:
    """Classify every source in `paths` (files, or directories walked recursively)."""
    paths = list(paths)
    res = AuditResult(roots=roots or paths, prefix=prefix, interpreter=sys.version.split()[0],
                      cache_tag=CACHE_TAG)
    for src in iter_sources(paths):
        try:
            size = os.path.getsize(src)
        except OSError:
            continue
        pyc = pyc_path_for(src, prefix)
        outcome = classify(src, pyc, read_source_for_hash=hash_sources)
        res.py_files += 1
        res.py_bytes += size
        res.counts[outcome] = res.counts.get(outcome, 0) + 1
        res.source_bytes[outcome] = res.source_bytes.get(outcome, 0) + size
        if outcome in VALID:
            try:
                res.pyc_bytes_total += os.path.getsize(pyc)
            except OSError:
                pass
        ex = res.examples.setdefault(outcome, [])
        if len(ex) < max_examples:
            ex.append(src)
    return res


def audit_modules(module_files: Iterable[str], prefix: Optional[str] = None) -> AuditResult:
    """Audit only the sources an import set actually loaded (e.g. sys.modules __file__s)."""
    files = sorted({f for f in module_files if f and f.endswith(".py") and os.path.exists(f)})
    return audit(files, prefix, roots=["<import set>"])


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("roots", nargs="+")
    ap.add_argument("--prefix", default=None, help="audit a PYTHONPYCACHEPREFIX tree instead of __pycache__")
    ap.add_argument("--json", default=None)
    ap.add_argument("--no-hash", action="store_true", help="do not hash sources for checked-hash pycs")
    a = ap.parse_args(argv)
    res = audit(a.roots, a.prefix, hash_sources=not a.no_hash)
    d = res.to_dict()
    if a.json:
        with open(a.json, "w") as f:
            json.dump(d, f, indent=1)
    print(f"{res.py_files} .py files, {res.py_bytes / 1e6:.1f} MB of source under {', '.join(a.roots)}")
    for k in sorted(res.counts):
        print(f"  {k:22s} {res.counts[k]:7d} files  {res.source_bytes[k] / 1e6:8.1f} MB source")
    print(f"  would be compiled at import: {res.needs_compile} files, {res.needs_compile_bytes / 1e6:.1f} MB")
    return 0


if __name__ == "__main__":
    sys.exit(main())
