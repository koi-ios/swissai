"""Fix (d): warm the page cache for an engine's import working set, in parallel, early.

A cold `import torch, sglang...` reads thousands of files one after another from a single
thread; on a SquashFS image stored on Lustre every miss is a remote, latency-bound read. The
same bytes read by 32-64 threads at job start (while srun/container setup, the parent's
argument parsing or the other engine processes run) arrive far faster: capstor gave 362 MB/s
with 1 reader and 6.7 GB/s with 64 readers in the lab's own sweep
(experiments/parallel-reads-lustre, job 81985).

    python -m pystartup.prewarm record --probe-json probe.json --out manifest.txt
    python -m pystartup.prewarm run --manifest manifest.txt --threads 32 [--mode read|willneed]
    python -m pystartup.prewarm residency --manifest manifest.txt      # Linux: % in page cache
    python -m pystartup.prewarm evict --manifest manifest.txt          # Linux: POSIX_FADV_DONTNEED

The manifest is the ordered list of files a profiled launch touched (sources, pycs,
extension modules and every mapped shared library), so the prewarm reads exactly the working
set, first-needed first (the idea eStargz/Nydus prefetch lists use for lazily pulled images).
"""
from __future__ import annotations

import argparse
import ctypes
import ctypes.util
import json
import os
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, Iterable, List, Optional

CHUNK = 4 << 20


def load_manifest(path: str) -> List[str]:
    with open(path) as f:
        return [l.rstrip("\n") for l in f if l.strip() and not l.startswith("#")]


def write_manifest(paths: Iterable[str], out: str) -> int:
    seen, order = set(), []
    for p in paths:
        if p and p not in seen and os.path.isfile(p):
            seen.add(p)
            order.append(p)
    with open(out, "w") as f:
        f.write("# pystartup prewarm manifest: files in first-access order\n")
        f.write("\n".join(order) + "\n")
    return len(order)


def _read_file(path: str, mode: str) -> int:
    try:
        fd = os.open(path, os.O_RDONLY)
    except OSError:
        return 0
    try:
        if mode == "willneed" and hasattr(os, "posix_fadvise"):
            size = os.fstat(fd).st_size
            os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_WILLNEED)
            return size
        n = 0
        while True:
            b = os.read(fd, CHUNK)
            if not b:
                return n
            n += len(b)
    finally:
        os.close(fd)


def prewarm(paths: List[str], threads: int = 32, mode: str = "read") -> dict:
    """Read every file in `paths` with a pool of threads (I/O releases the GIL)."""
    t = time.perf_counter()
    total = 0
    lock = threading.Lock()

    def work(p):
        nonlocal total
        n = _read_file(p, mode)
        with lock:
            total += n

    with ThreadPoolExecutor(max_workers=max(1, threads)) as ex:
        list(ex.map(work, paths))
    dt = time.perf_counter() - t
    return {"files": len(paths), "bytes": total, "seconds": round(dt, 3), "threads": threads, "mode": mode,
            "GBps": round(total / dt / 1e9, 3) if dt > 0 else None}


def start_background(paths: List[str], threads: int = 32, mode: str = "read") -> threading.Thread:
    """Prewarm from a daemon thread of the current process (e.g. the engine's launcher)."""
    th = threading.Thread(target=prewarm, args=(paths, threads, mode), daemon=True, name="pystartup-prewarm")
    th.start()
    return th


# ---------------------------------------------------------------------------- Linux helpers
_libc = None


def _libc_handle():
    global _libc
    if _libc is None:
        _libc = ctypes.CDLL(ctypes.util.find_library("c") or "libc.so.6", use_errno=True)
        _libc.mmap.restype = ctypes.c_void_p
        _libc.mmap.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int, ctypes.c_int, ctypes.c_int,
                               ctypes.c_long]
        _libc.munmap.argtypes = [ctypes.c_void_p, ctypes.c_size_t]
        _libc.mincore.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_void_p]
    return _libc


def residency(paths: List[str]) -> dict:
    """Fraction of each file's pages in the page cache (mincore). Linux only."""
    if not sys.platform.startswith("linux"):
        return {"error": "mincore residency is Linux only"}
    libc = _libc_handle()
    page = os.sysconf("SC_PAGE_SIZE")
    tot_pages = res_pages = 0
    files = 0
    for p in paths:
        try:
            fd = os.open(p, os.O_RDONLY)
        except OSError:
            continue
        try:
            size = os.fstat(fd).st_size
            if size == 0:
                continue
            addr = libc.mmap(None, size, 1, 1, fd, 0)  # PROT_READ, MAP_SHARED
            if addr in (None, ctypes.c_void_p(-1).value):
                continue
            n = (size + page - 1) // page
            vec = (ctypes.c_ubyte * n)()
            if libc.mincore(addr, size, vec) == 0:
                tot_pages += n
                res_pages += sum(v & 1 for v in vec)
                files += 1
            libc.munmap(addr, size)
        finally:
            os.close(fd)
    return {"files": files, "pages": tot_pages, "resident_pages": res_pages,
            "resident_fraction": round(res_pages / tot_pages, 4) if tot_pages else None}


def evict(paths: List[str]) -> dict:
    """Ask the kernel to drop clean cached pages of these files (best effort, unprivileged)."""
    if not hasattr(os, "posix_fadvise"):
        return {"error": "posix_fadvise unavailable"}
    n = 0
    for p in paths:
        try:
            fd = os.open(p, os.O_RDONLY)
        except OSError:
            continue
        try:
            os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
            n += 1
        finally:
            os.close(fd)
    return {"files": n}


def manifest_from_probe(probe: dict) -> List[str]:
    """Ordered file list from an importprof result made with --dump-files."""
    return list(probe.get("files_in_order", []))


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("record")
    r.add_argument("--probe-json", required=True)
    r.add_argument("--out", required=True)
    w = sub.add_parser("run")
    w.add_argument("--manifest", required=True)
    w.add_argument("--threads", type=int, default=32)
    w.add_argument("--mode", default="read", choices=["read", "willneed"])
    for name in ("residency", "evict"):
        s = sub.add_parser(name)
        s.add_argument("--manifest", required=True)
    a = ap.parse_args(argv)
    if a.cmd == "record":
        with open(a.probe_json) as f:
            n = write_manifest(manifest_from_probe(json.load(f)), a.out)
        print(f"{n} files -> {a.out}")
    elif a.cmd == "run":
        print(json.dumps(prewarm(load_manifest(a.manifest), a.threads, a.mode)))
    elif a.cmd == "residency":
        print(json.dumps(residency(load_manifest(a.manifest))))
    else:
        print(json.dumps(evict(load_manifest(a.manifest))))
    return 0


if __name__ == "__main__":
    sys.exit(main())
