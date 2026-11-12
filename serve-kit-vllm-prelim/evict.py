#!/usr/bin/env python3
"""Drop a model's files from the page cache, so the next read really is cold.

    python evict.py /net/models/apertus-8b            # evict every file under the directory
    python evict.py --verify /net/models/apertus-8b   # prove it: cached read vs read after eviction
    python evict.py --warm /net/models/apertus-8b     # the opposite: read every file into the page cache
    python evict.py --residency /net/models/apertus-8b

Eviction is posix_fadvise(POSIX_FADV_DONTNEED) over each file, after fdatasync,
because dirty pages are never dropped. It clears THIS machine's page cache only:
on a network volume the storage server may still answer from its own RAM, and no
client can clear that. --verify measures what a read after eviction costs, and
compares two cold reads, so a server-side cache shows up instead of hiding.

Residency is counted with mincore(2) on a private read-only mapping, the same
way vmtouch and fincore do. Linux only for eviction; residency also runs on macOS.
"""
from __future__ import annotations

import argparse
import ctypes
import ctypes.util
import json
import mmap
import os
import sys
import time
from pathlib import Path
from typing import Iterable, Iterator, List, Optional, Tuple

PAGE = mmap.PAGESIZE
READ_BLOCK = 16 * 1024 * 1024

_libc = ctypes.CDLL(ctypes.util.find_library("c"), use_errno=True)
_libc.mincore.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.POINTER(ctypes.c_ubyte)]
_libc.mincore.restype = ctypes.c_int


def iter_files(paths: Iterable[Path]) -> Iterator[Path]:
    """Every regular file under `paths` (files are yielded as given), sorted, no directory symlinks."""
    for root in paths:
        root = Path(root)
        if root.is_file():
            yield root
            continue
        if not root.is_dir():
            raise FileNotFoundError(f"{root}: no such file or directory")
        for dirpath, dirnames, filenames in os.walk(root):
            dirnames.sort()
            for name in sorted(filenames):
                p = Path(dirpath) / name
                if p.is_file():  # follows a file symlink (HF cache layout), skips sockets etc.
                    yield p


def evict_file(path: Path) -> int:
    """Drop one file's pages from the page cache; return its size in bytes."""
    if not hasattr(os, "posix_fadvise"):
        raise RuntimeError("posix_fadvise is not available on this platform; eviction needs Linux")
    fd = os.open(path, os.O_RDONLY)
    try:
        try:
            os.fdatasync(fd)  # DONTNEED skips dirty pages; a fresh download or copy has some
        except OSError:
            pass
        os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
        return os.fstat(fd).st_size
    finally:
        os.close(fd)


def evict(paths: Iterable[Path]) -> Tuple[int, int]:
    """Evict every file under `paths`; return (files, bytes)."""
    n = total = 0
    for p in iter_files(paths):
        total += evict_file(p)
        n += 1
    return n, total


def drop_all_caches() -> None:
    """The global hammer (root only): sync, then drop page cache, dentries and inodes."""
    if os.geteuid() != 0:
        raise PermissionError("--drop-caches needs root (it writes /proc/sys/vm/drop_caches)")
    os.sync()
    with open("/proc/sys/vm/drop_caches", "w") as f:
        f.write("3\n")


def file_residency(path: Path) -> Tuple[int, int]:
    """(resident bytes, size) of one file, from mincore on a fresh private mapping."""
    size = path.stat().st_size
    if size == 0:
        return 0, 0
    with open(path, "rb") as f:
        # ACCESS_COPY = MAP_PRIVATE, which ctypes can take an address of; nothing is ever written.
        mm = mmap.mmap(f.fileno(), size, access=mmap.ACCESS_COPY)
        try:
            anchor = ctypes.c_char.from_buffer(mm)
            try:
                pages = (size + PAGE - 1) // PAGE
                vec = (ctypes.c_ubyte * pages)()
                if _libc.mincore(ctypes.addressof(anchor), size, vec) != 0:
                    err = ctypes.get_errno()
                    raise OSError(err, f"mincore failed on {path}: {os.strerror(err)}")
                # bit 0 is "resident"; macOS also sets referenced/modified bits, so mask first
                resident_pages = bytes(vec).translate(_BIT0).count(1)
            finally:
                del anchor
        finally:
            mm.close()
    return min(resident_pages * PAGE, size), size


_BIT0 = bytes(b & 1 for b in range(256))


def residency(paths: Iterable[Path]) -> Tuple[int, int]:
    """(resident bytes, total bytes) over every file under `paths`."""
    res = tot = 0
    for p in iter_files(paths):
        r, s = file_residency(p)
        res += r
        tot += s
    return res, tot


def read_files(files: List[Path], limit: Optional[int] = None, block: int = READ_BLOCK) -> Tuple[int, float]:
    """Read `files` sequentially (at most `limit` bytes in all); return (bytes, seconds)."""
    buf = bytearray(block)
    view = memoryview(buf)
    done = 0
    t0 = time.perf_counter()
    for p in files:
        with open(p, "rb", buffering=0) as f:
            while limit is None or done < limit:
                want = block if limit is None else min(block, limit - done)
                n = f.readinto(view[:want])
                if not n:
                    break
                done += n
        if limit is not None and done >= limit:
            break
    return done, time.perf_counter() - t0


def warm(paths: Iterable[Path]) -> Tuple[int, float]:
    """Read every file under `paths` into the page cache; return (bytes, seconds)."""
    return read_files(list(iter_files(paths)))


def _gbps(nbytes: int, seconds: float) -> float:
    return nbytes / 1e9 / seconds if seconds > 0 else float("inf")


def verify(paths: List[Path], limit: int, min_ratio: float) -> dict:
    """Read, read again, evict, read again: is a read after eviction much slower than a cached one?"""
    files = list(iter_files(paths))
    if not files:
        raise ValueError("nothing to read")
    evict(files)
    b1, cold1 = read_files(files, limit)
    b2, cached = read_files(files, limit)
    res_cached, _ = residency(files)
    evict(files)
    res_after, total = residency(files)
    b3, cold2 = read_files(files, limit)
    out = {
        "files": len(files),
        "bytes_read": b1,
        "total_bytes": total,
        "cold_read_1_gbps": round(_gbps(b1, cold1), 3),
        "cached_read_gbps": round(_gbps(b2, cached), 3),
        "cold_read_2_gbps": round(_gbps(b3, cold2), 3),
        "resident_after_read_pct_of_read": round(min(100.0, 100.0 * res_cached / b1), 2) if b1 else 0.0,
        "resident_after_evict_pct": round(100.0 * res_after / total, 2) if total else 0.0,
    }
    ratio = cold2 / cached if cached > 0 else float("inf")
    out["cached_speedup_x"] = round(ratio, 1)
    problems = []
    if out["resident_after_evict_pct"] > 1.0:
        problems.append(f"{out['resident_after_evict_pct']}% still resident after eviction")
    if ratio < min_ratio:
        problems.append(
            f"a read after eviction is only {ratio:.1f}x slower than a cached read (want >= {min_ratio}x): "
            "the storage is very fast, or something below this machine is caching"
        )
    if cold1 > 0 and cold2 < 0.5 * cold1:
        problems.append(
            f"the second cold read ran {cold1 / cold2:.1f}x faster than the first: "
            "the storage server is probably caching these bytes, so 'cold' runs are partly warm"
        )
    out["problems"] = problems
    out["passed"] = not any("resident" in p or "slower" in p for p in problems)
    return out


def _parse_size(text: str) -> int:
    units = {"K": 1024, "M": 1024**2, "G": 1024**3, "T": 1024**4}
    text = text.strip().upper().rstrip("B").rstrip("I")
    if text and text[-1] in units:
        return int(float(text[:-1]) * units[text[-1]])
    return int(text)


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("paths", nargs="+", type=Path)
    mode = ap.add_mutually_exclusive_group()
    mode.add_argument("--verify", action="store_true", help="check that eviction makes the next read slow")
    mode.add_argument("--warm", action="store_true", help="read every file into the page cache instead")
    mode.add_argument("--residency", action="store_true", help="only report how much is in the page cache")
    ap.add_argument("--drop-caches", action="store_true", help="also drop the whole page cache (root)")
    ap.add_argument("--verify-bytes", default="4G", help="bytes to read per pass in --verify (default 4G)")
    ap.add_argument("--min-ratio", type=float, default=3.0, help="--verify fails below this cold/cached slowdown")
    ap.add_argument("--json", action="store_true", help="print one JSON object instead of text")
    args = ap.parse_args(argv)

    t0 = time.perf_counter()
    if args.verify:
        out = verify(args.paths, _parse_size(args.verify_bytes), args.min_ratio)
        if args.json:
            print(json.dumps(out))
        else:
            for k, v in out.items():
                if k != "problems":
                    print(f"{k:28s} {v}")
            for p in out["problems"]:
                print(f"! {p}")
            print("PASS" if out["passed"] else "FAIL")
        return 0 if out["passed"] else 1

    if args.warm:
        nbytes, secs = warm(args.paths)
        res, tot = residency(args.paths)
        out = {"action": "warm", "bytes": nbytes, "seconds": round(secs, 2), "gbps": round(_gbps(nbytes, secs), 3),
               "resident_pct": round(100.0 * res / tot, 2) if tot else 0.0}
    elif args.residency:
        res, tot = residency(args.paths)
        out = {"action": "residency", "bytes": tot, "resident_bytes": res,
               "resident_pct": round(100.0 * res / tot, 2) if tot else 0.0}
    else:
        n, nbytes = evict(args.paths)
        if args.drop_caches:
            drop_all_caches()
        res, tot = residency(args.paths)
        out = {"action": "evict", "files": n, "bytes": nbytes, "seconds": round(time.perf_counter() - t0, 2),
               "dropped_all_caches": bool(args.drop_caches),
               "resident_pct": round(100.0 * res / tot, 2) if tot else 0.0}
    print(json.dumps(out) if args.json else "  ".join(f"{k}={v}" for k, v in out.items()))
    if out["action"] == "evict" and out["resident_pct"] > 1.0:
        print(f"! {out['resident_pct']}% still resident after eviction (is a process holding the files mapped?)",
              file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, ValueError, RuntimeError) as e:
        print(f"error: {e}", file=sys.stderr)
        raise SystemExit(2)
