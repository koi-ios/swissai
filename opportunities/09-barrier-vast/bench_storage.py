#!/usr/bin/env python3
"""bench_storage.py: how fast can one node read model weights from a filesystem?

Compares the read paths LLM loaders use, on the files you point it at:
  mmap         page-fault reads through a shared file mapping (SGLang/vLLM default
               safetensors path): memcpy from the mapping into a buffer
  mmap-stride  like mmap, but copies only `--chunk` bytes of every `--stride` bytes,
               i.e. one tensor-parallel slice of a row-split weight (needs numpy)
  pread        buffered pread() into a reused buffer (--weight-loader-disable-mmap,
               servekit's stager, fastsafetensors without GDS)
  odirect      O_DIRECT pread() into an aligned buffer (bypasses the page cache;
               on macOS F_NOCACHE stands in)
with 1..64 parallel reader processes and several block sizes. Before every run it
drops the files' client page cache with posix_fadvise(DONTNEED) (Linux) and
measures how much is still resident with mincore(), so "cold" is checked, not
assumed. By default each run reads a fresh region of the file set (--rotate), so
server-side caches (Lustre OSS, VAST) are not re-hit by repeats.

It only ever reads the paths you give it. `--create DIR` writes random test files
into a new directory you name (random data: storage that compresses or dedupes
would otherwise look faster than it is).

Examples
  python bench_storage.py --dry-run /capstor/.../zai-org/GLM-4.7
  python bench_storage.py --quick /capstor/.../zai-org/GLM-4.7 --glob '*.safetensors'
  python bench_storage.py --create /vast/scratch/$USER/bench --create-size 64G
  python bench_storage.py /vast/scratch/$USER/bench --readers 1,4,16,64 --block-sizes 1M,16M,64M

Output: one JSON line per run (first line: environment) in --out, plus a summary
table (median, min-max over --repeats) on stdout and in <out>.summary.md.
Requires Python >= 3.8 and nothing else (numpy only for mmap-stride).
"""
from __future__ import annotations

import argparse
import ctypes
import ctypes.util
import fnmatch
import json
import mmap
import multiprocessing as mp
import os
import platform
import socket
import statistics
import subprocess
import sys
import time

PAGE = mmap.PAGESIZE
IS_LINUX = sys.platform.startswith("linux")
IS_DARWIN = sys.platform == "darwin"
F_NOCACHE = 48  # macOS fcntl
METHODS = ("mmap", "mmap-stride", "pread", "odirect")


# ----------------------------------------------------------------------------- utils
def parse_size(s: str) -> int:
    s = s.strip().upper().rstrip("B").rstrip("I")
    mult = {"K": 1 << 10, "M": 1 << 20, "G": 1 << 30, "T": 1 << 40}
    if s and s[-1] in mult:
        return int(float(s[:-1]) * mult[s[-1]])
    return int(s)


def fmt_size(n: int) -> str:
    for unit, m in (("T", 1 << 40), ("G", 1 << 30), ("M", 1 << 20), ("K", 1 << 10)):
        if n >= m and n % m == 0:
            return f"{n // m}{unit}"
    return str(n)


def list_files(paths, pattern):
    out = []
    for p in paths:
        if os.path.isdir(p):
            for root, _, names in os.walk(p):
                for n in sorted(names):
                    f = os.path.join(root, n)
                    if os.path.isfile(f) and not os.path.islink(f) and fnmatch.fnmatch(n, pattern):
                        out.append(f)
        elif os.path.isfile(p):
            out.append(p)
        else:
            raise SystemExit(f"not a file or directory: {p}")
    out = sorted(set(out))
    files = [(f, os.path.getsize(f)) for f in out]
    return [(f, s) for f, s in files if s > 0]


# ------------------------------------------------------------------ page cache control
def drop_cache(path: str) -> bool:
    """Drop clean cached pages of `path` on this client. True if supported."""
    fadvise = getattr(os, "posix_fadvise", None)
    if fadvise is None:
        return False
    fd = os.open(path, os.O_RDONLY)
    try:
        fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
    finally:
        os.close(fd)
    return True


_libc = None


def _get_libc():
    global _libc
    if _libc is None:
        _libc = ctypes.CDLL(ctypes.util.find_library("c"), use_errno=True)
        _libc.mincore.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_void_p]
        _libc.mincore.restype = ctypes.c_int
    return _libc


def resident_fraction(segments, max_probe=64, probe_bytes=4 << 20):
    """Fraction of pages resident in the page cache over up to `max_probe`
    windows of the (path, offset, length) segments that a run is about to read."""
    if not segments:
        return None
    step = max(1, len(segments) // max_probe)
    probes = segments[::step][:max_probe]
    resident = total = 0
    try:
        libc = _get_libc()
        for path, off, length in probes:
            start = off - off % mmap.ALLOCATIONGRANULARITY
            n = min(probe_bytes, length + (off - start))
            fsize = os.path.getsize(path)
            n = min(n, fsize - start)
            if n <= 0:
                continue
            fd = os.open(path, os.O_RDONLY)
            try:
                # a private writable mapping lets ctypes take its address; nothing is written
                mm = mmap.mmap(fd, n, flags=mmap.MAP_PRIVATE, prot=mmap.PROT_READ | mmap.PROT_WRITE, offset=start)
            finally:
                os.close(fd)
            try:
                buf = ctypes.c_char.from_buffer(mm)
                npages = (n + PAGE - 1) // PAGE
                vec = (ctypes.c_ubyte * npages)()
                rc = libc.mincore(ctypes.c_void_p(ctypes.addressof(buf)), ctypes.c_size_t(n), vec)
                del buf
                if rc != 0:
                    return None
                resident += sum(1 for v in vec if v & 1)
                total += npages
            finally:
                mm.close()
    except (OSError, ValueError, AttributeError, TypeError):
        return None
    return resident / total if total else None


# --------------------------------------------------------------------------- planning
def plan_run(files, cursor: int, nbytes: int, block: int, readers: int):
    """Pick `nbytes` of the file set starting at byte `cursor` of the concatenated
    files (wrapping), cut it into `block`-aligned (path, offset, length) segments
    and split those into `readers` contiguous groups of ~equal bytes."""
    total = sum(s for _, s in files)
    nbytes = min(nbytes, total)
    starts, acc = [], 0
    for _, s in files:
        starts.append(acc)
        acc += s
    cursor %= total
    # first file containing the cursor
    i = max(k for k, st in enumerate(starts) if st <= cursor)
    off = cursor - starts[i]
    off -= off % block
    segs, got = [], 0
    while got < nbytes:
        path, size = files[i]
        while off < size and got < nbytes:
            length = min(block, size - off, nbytes - got)
            segs.append((path, off, length))
            got += length
            off += block
        i = (i + 1) % len(files)
        off = 0
    # split into contiguous groups with ~equal bytes
    groups = [[] for _ in range(readers)]
    target = got / readers
    acc = 0
    for seg in segs:
        g = min(readers - 1, int(acc // target)) if target > 0 else 0
        groups[g].append(seg)
        acc += seg[2]
    return [g for g in groups if g], got


# ---------------------------------------------------------------------------- readers
def _read_mmap(segs, block, madvise, stride=None, chunk=None):
    dst = bytearray(block)
    dmv = memoryview(dst)
    maps = {}
    done = 0
    np = None
    if stride:
        import numpy as np  # noqa: F811
    try:
        for path, off, length in segs:
            mm = maps.get(path)
            if mm is None:
                fd = os.open(path, os.O_RDONLY)
                try:
                    mm = mmap.mmap(fd, 0, prot=mmap.PROT_READ)
                finally:
                    os.close(fd)
                if madvise and hasattr(mm, "madvise"):
                    mm.madvise({"sequential": mmap.MADV_SEQUENTIAL, "random": mmap.MADV_RANDOM,
                                "willneed": mmap.MADV_WILLNEED, "normal": mmap.MADV_NORMAL}[madvise])
                maps[path] = mm
            if stride:
                rows = length // stride
                if rows:
                    a = np.frombuffer(mm, dtype=np.uint8, count=rows * stride, offset=off)
                    part = a.reshape(rows, stride)[:, :chunk].copy()  # touches every row
                    done += part.nbytes
                    del a, part
            else:
                src = memoryview(mm)[off:off + length]
                dmv[:length] = src
                src.release()
                done += length
    finally:
        for mm in maps.values():
            try:
                mm.close()
            except BufferError:
                pass
    return done


def _read_pread(segs, block, direct):
    flags = os.O_RDONLY
    if direct and IS_LINUX:
        flags |= os.O_DIRECT
    buf = mmap.mmap(-1, block)  # page-aligned, as O_DIRECT requires
    fds = {}
    done = 0
    try:
        for path, off, length in segs:
            fd = fds.get(path)
            if fd is None:
                fd = os.open(path, flags)
                if direct and IS_DARWIN:
                    import fcntl
                    fcntl.fcntl(fd, F_NOCACHE, 1)
                fds[path] = fd
            want = length if not direct else ((length + PAGE - 1) // PAGE) * PAGE
            if hasattr(os, "preadv"):
                n = os.preadv(fd, [memoryview(buf)[:want]], off)
            else:
                n = len(os.pread(fd, want, off))
            done += min(n, length)
    finally:
        for fd in fds.values():
            os.close(fd)
        buf.close()
    return done


def _reader_proc(idx, method, segs, block, opts, barrier, q):
    try:
        barrier.wait()
        t0 = time.monotonic()
        if method == "mmap":
            n = _read_mmap(segs, block, opts.get("madvise"))
        elif method == "mmap-stride":
            n = _read_mmap(segs, block, opts.get("madvise"), opts["stride"], opts["chunk"])
        elif method == "pread":
            n = _read_pread(segs, block, direct=False)
        elif method == "odirect":
            n = _read_pread(segs, block, direct=True)
        else:
            raise ValueError(method)
        t1 = time.monotonic()
        q.put((idx, t0, t1, n, None))
    except Exception as e:  # report, don't hang the parent
        q.put((idx, None, None, 0, f"{type(e).__name__}: {e}"))


def run_once(method, groups, block, opts):
    ctx = mp.get_context("spawn")
    barrier = ctx.Barrier(len(groups))
    q = ctx.Queue()
    procs = [ctx.Process(target=_reader_proc, args=(i, method, g, block, opts, barrier, q))
             for i, g in enumerate(groups)]
    for p in procs:
        p.start()
    res = [q.get() for _ in procs]
    for p in procs:
        p.join()
    errs = [r[4] for r in res if r[4]]
    if errs:
        raise RuntimeError(errs[0])
    t0 = min(r[1] for r in res)
    t1 = max(r[2] for r in res)
    durs = [r[2] - r[1] for r in res]
    return {"seconds": t1 - t0, "bytes_read": sum(r[3] for r in res),
            "reader_min_s": min(durs), "reader_max_s": max(durs)}


# ------------------------------------------------------------------------ environment
def describe_env(path):
    env = {"host": socket.gethostname(), "platform": platform.platform(), "python": platform.python_version(),
           "cpus": os.cpu_count(), "page_size": PAGE, "path": os.path.abspath(path)}
    try:
        if IS_LINUX:
            st = os.stat(path)
            best = None
            with open("/proc/self/mountinfo") as f:
                for line in f:
                    parts = line.split()
                    mnt = parts[4]
                    sep = parts.index("-")
                    fstype, src, sopts = parts[sep + 1], parts[sep + 2], parts[sep + 3]
                    if os.path.abspath(path).startswith(mnt) and (best is None or len(mnt) > len(best[0])):
                        best = (mnt, fstype, src, parts[5] + "," + sopts, parts[2])
            if best:
                env.update(mount=best[0], fstype=best[1], source=best[2], mount_opts=best[3])
                majmin = best[4]
                for cand in (f"/sys/class/bdi/{majmin}/read_ahead_kb",):
                    if os.path.exists(cand):
                        env["bdi_read_ahead_kb"] = int(open(cand).read())
            with open("/proc/meminfo") as f:
                env["mem_total_kb"] = int(f.readline().split()[1])
            env["kernel"] = platform.release()
        elif IS_DARWIN:
            out = subprocess.run(["df", "-P", path], capture_output=True, text=True).stdout.splitlines()
            if len(out) > 1:
                dev = out[1].split()[0]
                mounts = subprocess.run(["mount"], capture_output=True, text=True).stdout
                for line in mounts.splitlines():
                    if line.startswith(dev + " "):
                        env["mount"] = line
            env["mem_total_kb"] = int(subprocess.run(["sysctl", "-n", "hw.memsize"], capture_output=True,
                                                     text=True).stdout) // 1024
    except Exception as e:  # environment capture must never stop a benchmark
        env["env_error"] = str(e)
    return env


# --------------------------------------------------------------------------- creation
def create_files(directory, total, nfiles):
    if os.path.exists(directory) and os.listdir(directory):
        raise SystemExit(f"--create: {directory} exists and is not empty; refusing to write there")
    os.makedirs(directory, exist_ok=True)
    per = total // nfiles
    chunk = 16 << 20
    for i in range(nfiles):
        p = os.path.join(directory, f"bench-{i:03d}.bin")
        fd = os.open(p, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
        try:
            if IS_DARWIN:
                import fcntl
                fcntl.fcntl(fd, F_NOCACHE, 1)  # keep the new data out of the cache
            left = per
            while left > 0:
                n = min(chunk, left)
                os.write(fd, os.urandom(n))
                left -= n
            os.fsync(fd)
        finally:
            os.close(fd)
        drop_cache(p)
    print(f"created {nfiles} files x {fmt_size(per)} in {directory}")


# ------------------------------------------------------------------------------ main
def summarize(rows):
    keyf = lambda r: (r["method"], r["readers"], r["block"], r.get("madvise") or "")  # noqa: E731
    groups = {}
    for r in rows:
        groups.setdefault(keyf(r), []).append(r)
    lines = ["| method | readers | block | median GB/s | min-max GB/s | reader spread (median) | resident before (max) | n |",
             "|---|---|---|---|---|---|---|---|"]
    out = []
    for k in sorted(groups, key=lambda k: (METHODS.index(k[0]), k[2], k[1])):
        g = groups[k]
        bw = [r["gbps"] for r in g]
        spread = statistics.median(r["reader_max_s"] - r["reader_min_s"] for r in g)
        res = [r["resident_before"] for r in g if r["resident_before"] is not None]
        name = k[0] + (f" ({k[3]})" if k[3] else "")
        lines.append(f"| {name} | {k[1]} | {fmt_size(k[2])} | {statistics.median(bw):.2f} | "
                     f"{min(bw):.2f}-{max(bw):.2f} | {spread:.2f} s | "
                     f"{(max(res) * 100 if res else float('nan')):.0f}% | {len(g)} |")
        out.append({"method": k[0], "readers": k[1], "block": k[2], "median_gbps": statistics.median(bw),
                    "min_gbps": min(bw), "max_gbps": max(bw), "n": len(g)})
    return "\n".join(lines), out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("paths", nargs="*", help="files or directories to read (never written)")
    ap.add_argument("--glob", default="*", help="file name filter inside directories, e.g. '*.safetensors'")
    ap.add_argument("--methods", default="mmap,pread,odirect", help=f"comma list of {','.join(METHODS)}")
    ap.add_argument("--readers", default="1,4,16,64", help="parallel reader processes per point")
    ap.add_argument("--block-sizes", default="1M,16M", help="read/copy size per call")
    ap.add_argument("--bytes-per-run", default="4G", help="bytes read per run (all readers together)")
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--warm", action="store_true", help="do not drop the page cache before runs")
    ap.add_argument("--no-rotate", action="store_true", help="re-read the same region every run")
    ap.add_argument("--madvise", choices=["normal", "sequential", "random", "willneed"], default=None)
    ap.add_argument("--stride", default="4K", help="mmap-stride: row length in bytes")
    ap.add_argument("--chunk", default="128", help="mmap-stride: bytes copied per row")
    ap.add_argument("--quick", action="store_true", help="readers 1,8; blocks 1M,16M; 1G per run")
    ap.add_argument("--create", metavar="DIR", help="write random test files into a new DIR, then exit")
    ap.add_argument("--create-size", default="16G")
    ap.add_argument("--create-files", type=int, default=8)
    ap.add_argument("--out", default=None, help="JSONL output (default results/storage-<host>-<time>.jsonl)")
    ap.add_argument("--dry-run", action="store_true", help="print the plan and exit without reading")
    a = ap.parse_args(argv)

    if a.create:
        create_files(a.create, parse_size(a.create_size), a.create_files)
        return 0
    if not a.paths:
        ap.error("give at least one file or directory")
    if a.quick:
        a.readers, a.block_sizes, a.bytes_per_run = "1,8", "1M,16M", "1G"
    methods = [m for m in a.methods.split(",") if m]
    for m in methods:
        if m not in METHODS:
            ap.error(f"unknown method {m}")
    readers = [int(x) for x in a.readers.split(",")]
    blocks = [parse_size(x) for x in a.block_sizes.split(",")]
    if any(b % PAGE for b in blocks):
        ap.error(f"block sizes must be multiples of the page size ({PAGE})")
    nbytes = parse_size(a.bytes_per_run)
    files = list_files(a.paths, a.glob)
    if not files:
        raise SystemExit("no non-empty files found")
    total = sum(s for _, s in files)

    points = []
    for m in methods:
        for b in (blocks if m in ("pread", "odirect") else [max(blocks)] if m == "mmap-stride" else [blocks[0]]):
            for r in readers:
                points.append((m, r, b))
    runs = len(points) * a.repeats
    read_total = runs * min(nbytes, total)
    print(f"{len(files)} files, {total / 1e9:.1f} GB; {len(points)} points x {a.repeats} repeats = {runs} runs, "
          f"{read_total / 1e9:.1f} GB read in total"
          + ("" if a.no_rotate or read_total <= total else " (more than the file set: some regions are re-read)"))
    print(f"runtime upper bound at 0.2 GB/s: {read_total / 0.2e9 / 60:.0f} min; at 2 GB/s: {read_total / 2e9 / 60:.1f} min")
    if a.dry_run:
        for p in points:
            print("  ", p[0], "readers=%d" % p[1], "block=%s" % fmt_size(p[2]))
        return 0
    if "mmap-stride" in methods:
        try:
            import numpy  # noqa: F401
        except ImportError:
            raise SystemExit("mmap-stride needs numpy")

    env = describe_env(files[0][0])
    env.update({"files": len(files), "file_bytes": total, "cold": not a.warm, "rotate": not a.no_rotate,
                "drop_cache_supported": hasattr(os, "posix_fadvise")})
    out = a.out or os.path.join("results", f"storage-{env['host']}-{time.strftime('%Y%m%d-%H%M%S')}.jsonl")
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    rows = []
    cursor = 0
    stride, chunk = parse_size(a.stride), parse_size(a.chunk)
    with open(out, "w") as fo:
        fo.write(json.dumps({"env": env}) + "\n")
        for rep in range(a.repeats):
            for (m, r, b) in points:  # repeats outermost: drift hits every point alike
                groups, got = plan_run(files, cursor, nbytes, b, r)
                if not a.no_rotate:
                    cursor += got
                segs = [s for g in groups for s in g]
                if not a.warm:
                    for path in sorted({s[0] for s in segs}):
                        drop_cache(path)
                res_before = resident_fraction(segs)
                opts = {"madvise": a.madvise, "stride": stride, "chunk": chunk}
                t = run_once(m, groups, b, opts)
                row = {"method": m, "readers": len(groups), "block": b, "madvise": a.madvise, "repeat": rep,
                       "bytes": got, "resident_before": res_before, **t,
                       # gbps: bytes of file swept per second (what the page cache/storage served);
                       # useful_gbps: bytes the loader kept (differs only for mmap-stride)
                       "gbps": got / t["seconds"] / 1e9,
                       "useful_gbps": t["bytes_read"] / t["seconds"] / 1e9,
                       "time": time.time()}
                rows.append(row)
                fo.write(json.dumps(row) + "\n")
                fo.flush()
                rb = "n/a" if res_before is None else f"{res_before * 100:.0f}%"
                print(f"rep {rep} {m:11s} readers={len(groups):3d} block={fmt_size(b):>4s} "
                      f"{row['gbps']:7.2f} GB/s  ({got / 1e9:.2f} GB in {t['seconds']:.2f} s, "
                      f"reader spread {t['reader_max_s'] - t['reader_min_s']:.2f} s, resident before {rb})")
    table, summary = summarize(rows)
    with open(out + ".summary.md", "w") as f:
        f.write(f"# {env['host']} {env.get('fstype', '')} {env['path']}\n\n{table}\n\n```\n{json.dumps(env, indent=1)}\n```\n")
    print("\n" + table)
    print(f"\nwrote {out} and {out}.summary.md")
    return 0


if __name__ == "__main__":
    sys.exit(main())
