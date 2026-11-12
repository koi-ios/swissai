#!/usr/bin/env python3
"""Stage a (presharded) model directory into RAM with N parallel reader processes.

    python stage.py /net/models/apertus-8b-tp2 /dev/shm/apertus-8b-tp2 --readers 16
    python stage.py SRC DEST --rank 0                       # one rank's shards + the config files
    python stage.py SRC DEST --sweep 1,2,4,8,16,32 --repeats 3 --csv results/staging.csv

A port of servekit's sliced stager (src/servekit/_stage/stage_to_shm_sliced.sh,
which runs `dd` over 64 slices per file) to Python, so it can rank-filter, count
and sweep:

- every file is cut into BLOCK-sized pieces, and the list of pieces, in file
  order, is split into N contiguous runs, one per reader process: each reader
  streams one disjoint byte range, and the readers finish together;
- a reader moves each block with os.pread (os.preadv into a page-aligned buffer
  under --direct) and os.pwrite at the same offset;
- each file is created at full size under a temporary name first and renamed to
  its real name only after every reader has finished and the sizes match, so a
  loader never sees a half-written file.

Buffered reads drop each block from the page cache after copying it
(POSIX_FADV_DONTNEED), so the model is not held twice in RAM and a repeat starts
cold. --direct uses O_DIRECT instead, which not every file system supports.
"""
from __future__ import annotations

import argparse
import csv
import mmap
import multiprocessing as mp
import os
import re
import shutil
import statistics
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

TMP_PREFIX = ".staging-"
DIRECT_ALIGN = 4096
# vLLM's ShardedStateLoader names (TP_ONLY_PATTERN / PP_AND_TP_PATTERN in vllm 0.33)
_TP_ONLY = re.compile(r"^model-rank-(\d+)-part-\d+\.safetensors$")
_PP_TP = re.compile(r"^model-pp-\d+-tp-(\d+)-part-\d+\.safetensors$")


@dataclass
class StageResult:
    src: str
    dest: str
    files: int
    bytes: int
    readers: int
    block_mib: int
    read_mode: str
    seconds: float
    gbps: float  # decimal GB/s, as servekit's stage_GBps


def _rank_of(name: str) -> Optional[int]:
    m = _TP_ONLY.match(name) or _PP_TP.match(name)
    return int(m.group(1)) if m else None


def plan_files(src: Path, ranks: Optional[Sequence[int]] = None) -> List[Tuple[Path, int]]:
    """[(path relative to src, size)] in a stable order; other ranks' shard files dropped if `ranks`."""
    out: List[Tuple[Path, int]] = []
    shard_files = kept_shards = 0
    for dirpath, dirnames, filenames in os.walk(src):
        dirnames.sort()
        for name in sorted(filenames):
            if name.startswith(TMP_PREFIX):
                continue
            p = Path(dirpath) / name
            if not p.is_file():
                continue
            rank = _rank_of(name)
            if rank is not None:
                shard_files += 1
                if ranks is not None and rank not in ranks:
                    continue
                kept_shards += 1
            out.append((p.relative_to(src), p.stat().st_size))
    if ranks is not None:
        if shard_files == 0:
            raise ValueError(f"--rank given, but {src} holds no sharded-state files (model-rank-*-part-*.safetensors)")
        if kept_shards == 0:
            raise ValueError(f"no shard files for rank(s) {list(ranks)} in {src}")
    if not out:
        raise ValueError(f"no files to stage in {src}")
    return out


def make_pieces(sizes: Sequence[int], block: int) -> List[Tuple[int, int, int]]:
    """(file index, offset, length) for every block of every file, in file order."""
    pieces = []
    for i, size in enumerate(sizes):
        for off in range(0, size, block):
            pieces.append((i, off, min(block, size - off)))
    return pieces


def split_runs(n_pieces: int, n_readers: int) -> List[Tuple[int, int]]:
    """Cut range(n_pieces) into n_readers contiguous, near-equal runs (empty runs dropped)."""
    runs = [(n_pieces * r // n_readers, n_pieces * (r + 1) // n_readers) for r in range(n_readers)]
    return [(a, b) for a, b in runs if b > a]


def _reader(job) -> int:
    srcs, dsts, pieces, block, direct, drop_src_cache = job
    sfds, dfds = {}, {}
    buf = mmap.mmap(-1, block) if direct else None  # anonymous mmap is page-aligned, as O_DIRECT needs
    can_drop = drop_src_cache and not direct and hasattr(os, "posix_fadvise")
    moved = 0
    try:
        for fi, off, length in pieces:
            if fi not in sfds:
                flags = os.O_RDONLY | (getattr(os, "O_DIRECT", 0) if direct else 0)
                try:
                    sfds[fi] = os.open(srcs[fi], flags)
                except OSError as e:
                    if direct:
                        raise OSError(e.errno, f"O_DIRECT open failed on {srcs[fi]} ({e.strerror}); "
                                               "this file system may not support it") from e
                    raise
                dfds[fi] = os.open(dsts[fi], os.O_WRONLY)
            sfd, dfd = sfds[fi], dfds[fi]
            if direct:
                want = (length + DIRECT_ALIGN - 1) // DIRECT_ALIGN * DIRECT_ALIGN
                n = os.preadv(sfd, [memoryview(buf)[:want]], off)
                if n < length:
                    raise IOError(f"short O_DIRECT read: {n} of {length} bytes at {off} in {srcs[fi]}")
                data = memoryview(buf)[:length]
            else:
                chunks, got = [], 0
                while got < length:
                    part = os.pread(sfd, length - got, off + got)
                    if not part:
                        raise IOError(f"unexpected end of file at {off + got} in {srcs[fi]}")
                    chunks.append(part)
                    got += len(part)
                data = memoryview(chunks[0] if len(chunks) == 1 else b"".join(chunks))
            written = 0
            while written < length:
                written += os.pwrite(dfd, data[written:length], off + written)
            if can_drop:
                os.posix_fadvise(sfd, off, length, os.POSIX_FADV_DONTNEED)
            moved += length
            del data
    finally:
        for fd in list(sfds.values()) + list(dfds.values()):
            os.close(fd)
        if buf is not None:
            try:
                buf.close()
            except BufferError:  # a view is still alive after an error; let the real error through
                pass
    return moved


def _tmp_name(final: Path) -> Path:
    return final.with_name(TMP_PREFIX + final.name)


def check_dest(dest: Path) -> None:
    """Refuse a destination whose replacement would delete more than one model copy."""
    p = Path(dest).resolve()
    if p in (Path("/"), Path("/dev/shm").resolve(), Path.home().resolve()) or len(p.parts) < 3 or os.path.ismount(p):
        raise ValueError(f"refusing to stage into {p}: use a subdirectory, e.g. /dev/shm/<model>")


def stage(
    src: Path,
    dest: Path,
    readers: int = 16,
    block_mib: int = 16,
    direct: bool = False,
    ranks: Optional[Sequence[int]] = None,
    force: bool = False,
    drop_src_cache: bool = True,
    headroom_bytes: int = 2 * 1024**3,
) -> StageResult:
    """Copy `src` into `dest`; the timing covers everything from pre-flight to the last rename."""
    t0 = time.perf_counter()
    src, dest = Path(src), Path(dest)
    if readers < 1 or block_mib < 1:
        raise ValueError("readers and block_mib must be >= 1")
    if not src.is_dir():
        raise FileNotFoundError(f"src is not a directory: {src}")
    check_dest(dest)
    files = plan_files(src, ranks)
    total = sum(size for _, size in files)

    if dest.exists():
        if any(dest.iterdir()):
            if not force:
                raise FileExistsError(f"{dest} exists and is not empty (use --force to replace it)")
            shutil.rmtree(dest)
    dest.mkdir(parents=True, exist_ok=True)
    free = shutil.disk_usage(dest).free
    if free < total + headroom_bytes:
        raise OSError(f"{dest} has {free / 1e9:.1f} GB free, need {total / 1e9:.1f} GB + "
                      f"{headroom_bytes / 1e9:.1f} GB headroom (a container's /dev/shm is often only 64 MB)")

    srcs, finals, tmps = [], [], []
    for rel, size in files:
        final = dest / rel
        final.parent.mkdir(parents=True, exist_ok=True)
        tmp = _tmp_name(final)
        fd = os.open(tmp, os.O_CREAT | os.O_WRONLY | os.O_TRUNC, 0o644)
        try:
            os.ftruncate(fd, size)  # full size before any writer runs (no racing creates)
        finally:
            os.close(fd)
        srcs.append(str(src / rel))
        finals.append(final)
        tmps.append(str(tmp))

    block = block_mib * 1024 * 1024
    pieces = make_pieces([size for _, size in files], block)
    runs = split_runs(len(pieces), readers)
    jobs = [(srcs, tmps, pieces[a:b], block, direct, drop_src_cache) for a, b in runs]
    if jobs:
        ctx = mp.get_context("fork") if "fork" in mp.get_all_start_methods() else mp.get_context()
        with ctx.Pool(processes=len(jobs)) as pool:
            moved = sum(pool.map(_reader, jobs, chunksize=1))
    else:
        moved = 0  # only empty files
    if moved != total:
        raise IOError(f"moved {moved} bytes, expected {total}")
    for (rel, size), tmp in zip(files, tmps):
        got = os.stat(tmp).st_size
        if got != size:
            raise IOError(f"size mismatch for {rel}: {got} != {size}")
    for tmp, final in zip(tmps, finals):
        os.replace(tmp, final)
    secs = time.perf_counter() - t0
    return StageResult(
        src=str(src), dest=str(dest), files=len(files), bytes=total, readers=readers, block_mib=block_mib,
        read_mode="direct" if direct else "buffered", seconds=round(secs, 3),
        gbps=round(total / 1e9 / secs, 3) if secs > 0 else 0.0,
    )


CSV_FIELDS = ["date", "src", "files", "bytes", "readers", "block_mib", "read_mode", "repeat", "seconds", "gbps"]


def append_csv(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    new = not path.exists() or path.stat().st_size == 0
    with open(path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=CSV_FIELDS, extrasaction="ignore")
        if new:
            w.writeheader()
        w.writerow(row)


def _evict_src(src: Path) -> None:
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import evict  # noqa: E402  (same directory)

    evict.evict([src])


def sweep(args) -> int:
    check_dest(args.dest)  # the sweep deletes dest between runs
    readers_list = [int(x) for x in args.sweep.split(",") if x]
    blocks = [int(x) for x in str(args.block_mib).split(",") if x]
    modes = [m for m in args.modes.split(",") if m]
    results = {}
    for rep in range(1, args.repeats + 1):
        # rotate the order each repeat so slow drift on shared storage does not favour one setting
        order = readers_list[(rep - 1) % len(readers_list):] + readers_list[:(rep - 1) % len(readers_list)]
        for n in order:
            for b in blocks:
                for mode in modes:
                    if not args.no_evict and os.environ.get("SK_TEST_MODE") != "1":
                        _evict_src(args.src)
                    shutil.rmtree(args.dest, ignore_errors=True)
                    try:
                        res = stage(args.src, args.dest, n, b, direct=(mode == "direct"), ranks=args.rank,
                                    force=True, drop_src_cache=not args.keep_src_cache)
                    except OSError as e:
                        print(f"! readers={n} block={b}MiB mode={mode}: {e}", file=sys.stderr)
                        continue
                    finally:
                        if not args.keep:
                            shutil.rmtree(args.dest, ignore_errors=True)
                    row = {**asdict(res), "date": _now(), "repeat": rep}
                    if args.csv:
                        append_csv(args.csv, row)
                    results.setdefault((n, b, mode), []).append(res.gbps)
                    print(f"repeat={rep} readers={n:>3} block={b}MiB mode={mode:8s} "
                          f"{res.bytes / 1e9:.2f} GB in {res.seconds:.2f} s = {res.gbps:.2f} GB/s", flush=True)
    print("\nreaders  block  mode      GB/s median [min, max]  n")
    for (n, b, mode), vals in sorted(results.items()):
        print(f"{n:>7}  {b:>4}M  {mode:8s}  {statistics.median(vals):6.2f} [{min(vals):.2f}, {max(vals):.2f}]  {len(vals)}")
    return 0 if results else 1


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("src", type=Path)
    ap.add_argument("dest", type=Path)
    ap.add_argument("--readers", type=int, default=16, help="parallel reader processes (default 16)")
    ap.add_argument("--block-mib", default="16", help="block size in MiB, 16-64 (a list in --sweep, e.g. 16,64)")
    ap.add_argument("--rank", type=int, action="append", help="stage only this TP rank's shard files (repeatable)")
    ap.add_argument("--direct", action="store_true", help="O_DIRECT reads (not on every file system)")
    ap.add_argument("--force", action="store_true", help="replace a non-empty dest")
    ap.add_argument("--keep-src-cache", action="store_true", help="do not drop copied blocks from the page cache")
    ap.add_argument("--csv", type=Path, help="append one row per staging to this CSV")
    ap.add_argument("--repeat", type=int, default=1, help="repeat number to record (single run)")
    ap.add_argument("--sweep", help="comma-separated reader counts; evicts src and clears dest before each")
    ap.add_argument("--repeats", type=int, default=3, help="repeats per setting in --sweep")
    ap.add_argument("--modes", default="buffered", help="read modes in --sweep: buffered,direct")
    ap.add_argument("--no-evict", action="store_true", help="--sweep: do not evict src first (testing only)")
    ap.add_argument("--keep", action="store_true", help="--sweep: keep the last dest instead of deleting it")
    args = ap.parse_args(argv)

    if args.sweep:
        return sweep(args)
    res = stage(args.src, args.dest, args.readers, int(args.block_mib), direct=args.direct, ranks=args.rank,
                force=args.force, drop_src_cache=not args.keep_src_cache)
    if args.csv:
        append_csv(args.csv, {**asdict(res), "date": _now(), "repeat": args.repeat})
    print(f"staged_files={res.files} staged_bytes={res.bytes} readers={res.readers} block_mib={res.block_mib} "
          f"read_mode={res.read_mode} stage_wall_s={res.seconds:.2f} stage_GBps={res.gbps:.2f}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, ValueError) as e:
        print(f"error: {e}", file=sys.stderr)
        raise SystemExit(2)
