#!/usr/bin/env python3
"""Per-image, per-GPU-arch shared JIT/compile cache for sml (swiss-ai/model-launch#270).

Layout on shared storage (capstor):

    <root>/<image-key>/<gpu-arch>/<component>/...      component = triton, inductor, vllm, tilelang, ...
    <root>/<image-key>/<gpu-arch>/.meta/{image,last_used,promotions.log}
    <root>/<image-key>/<gpu-arch>/.staging/<host>-<pid>-<rand>/   (same filesystem: atomic rename)

Lifecycle of one job (all unprivileged, standard library only, runs inside the engine image):

    key      image key: "<image>.digest" sidecar written by the images CI if present, else a hash of
             the file size and the 96-byte squashfs superblock (mkfs time, table offsets, bytes used):
             one small read, changes on every rebuild, even though images are updated in place.
    arch     "sm90" from nvidia-smi compute_cap (GH200/H100), "sm80" (A100), ...
    seed     copy the shared tree into a node-local overlay at a FIXED path (Triton group files and
             ninja trees store absolute paths, so entries must be read where they were written),
             in parallel; never copies lock files, temp files or node-specific runtime state.
    promote  after the server is ready (and periodically), publish entries the job created: each
             new subtree is copied into .staging, made group-readable (some writers create 0600
             files), then renamed into place. A directory rename onto an existing non-empty
             directory fails and a file is published with link(), which fails if it exists, so the
             first writer wins, nothing is overwritten and readers never see a partial entry. No
             cross-node locks are needed (Lustre may be mounted localflock or noflock).
    gc       LRU by .meta/last_used (touched by every seed) under a byte budget and a max age.
"""

from __future__ import annotations

import argparse
import concurrent.futures as cf
import contextlib
import fnmatch
import hashlib
import json
import os
import shutil
import socket
import stat
import subprocess
import sys
import time
import urllib.request
import uuid
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

DEFAULT_ROOT = "/capstor/store/cscs/swissai/infra01/sml/shared-caches"
# Fixed, node-local path on purpose: entries must be read where they were written.
DEFAULT_OVERLAY = "/tmp/sml-jit-cache"  # noqa: S108
MANIFEST = ".sml-seed-manifest.json"

# Never seed or promote: locks (a stale cpp_extension FileBaton blocks forever), temp and
# staging files of the libraries' own atomic writes, node-topology state, our own bookkeeping.
EXCLUDE = [
    "*.lock", "lock", "*.lck", "LOCK",
    "tmp.pid_*", ".staging*", "*.tmp", "*.tmp.*", ".*.tmp", "*.partial", "tmp",
    "gpu_p2p_access_cache_for_*.json",
    ".meta", MANIFEST, "__pycache__",
]


# ----------------------------------------------------------------------------------------- key
def squashfs_fingerprint(path: str) -> str | None:
    """sha256 over (file size, first 96 bytes) if the file is a squashfs image."""
    try:
        st = os.stat(path)
        with open(path, "rb") as f:
            sb = f.read(96)
    except OSError:
        return None
    if len(sb) < 96 or sb[:4] not in (b"hsqs", b"sqsh"):
        return None
    h = hashlib.sha256(str(st.st_size).encode() + b"\0" + sb).hexdigest()
    return "sqsh-" + h[:16]


def image_key(image: str) -> str:
    sidecar = image + ".digest"
    try:
        with open(sidecar) as f:
            d = f.readline().strip()
        if d:
            return d.replace(":", "-")[:23] if d.startswith("sha256:") else _clean(d)
    except OSError:
        pass
    if "@sha256:" in image:
        return "sha256-" + image.split("@sha256:", 1)[1][:16]
    fp = squashfs_fingerprint(image)
    if fp:
        return fp
    if os.path.isfile(image):  # not squashfs: size + first MiB
        with open(image, "rb") as f:
            head = f.read(1 << 20)
        return "file-" + hashlib.sha256(str(os.path.getsize(image)).encode() + head).hexdigest()[:16]
    # a registry tag: tags move, so this key is only as good as the tag
    return "ref-" + hashlib.sha256(image.encode()).hexdigest()[:16]


def _clean(s: str) -> str:
    return "".join(c if c.isalnum() or c in "-_." else "-" for c in s)[:64]


def gpu_arch() -> str:
    if os.environ.get("SML_GPU_ARCH"):
        return os.environ["SML_GPU_ARCH"]
    try:
        out = subprocess.run(  # noqa: S603
            ["nvidia-smi", "--query-gpu=compute_cap", "--format=csv,noheader"],  # noqa: S607
            capture_output=True,
            text=True,
            timeout=30,
        ).stdout
        caps = sorted({c.strip().replace(".", "") for c in out.splitlines() if c.strip()})
        if caps:
            return "+".join("sm" + c for c in caps)
    except (OSError, subprocess.SubprocessError):
        pass
    return "unknown-gpu"


# ------------------------------------------------------------------------------------- helpers
def _excluded(name: str) -> bool:
    return any(fnmatch.fnmatchcase(name, p) for p in EXCLUDE)


def _walk(root: Path) -> Iterator[tuple[Path, os.stat_result]]:
    """Yield (relative path, lstat) for files and symlinks, skipping excluded names."""
    stack = [root]
    while stack:
        d = stack.pop()
        try:
            it = os.scandir(d)
        except OSError:
            continue
        with it:
            for e in it:
                if _excluded(e.name):
                    continue
                try:
                    st = e.stat(follow_symlinks=False)
                except OSError:
                    continue
                if stat.S_ISDIR(st.st_mode):
                    stack.append(Path(e.path))
                else:
                    yield Path(e.path).relative_to(root), st


def _copy_file(src: Path, dst: Path) -> int:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if src.is_symlink():
        os.symlink(os.readlink(src), dst)
        return 0
    shutil.copy2(src, dst)
    return dst.stat().st_size


def _share_modes(path: Path) -> None:
    """Group/other readable, group writable for dirs (setgid keeps the group)."""
    for d, _dirs, files in os.walk(path):
        try:
            os.chmod(d, 0o2775)  # noqa: S103  # group-shared cache: setgid + group write
        except OSError:
            pass
        for f in files:
            p = os.path.join(d, f)
            try:
                st = os.lstat(p)
                if stat.S_ISREG(st.st_mode):
                    os.chmod(p, 0o775 if st.st_mode & 0o111 else 0o664)
            except OSError:
                pass


# ---------------------------------------------------------------------------------------- seed
def seed(shared: Path, local: Path, jobs: int = 16) -> dict[str, Any]:
    t0 = time.monotonic()
    local.mkdir(parents=True, exist_ok=True)
    files = list(_walk(shared)) if shared.is_dir() else []
    nbytes = 0
    with cf.ThreadPoolExecutor(max_workers=jobs) as ex:
        futs = {}
        for rel, _st in files:
            dst = local / rel
            if dst.exists() or dst.is_symlink():
                continue
            futs[ex.submit(_copy_file, shared / rel, dst)] = rel
        errors = 0
        for f in cf.as_completed(futs):
            try:
                nbytes += f.result()
            except OSError:
                errors += 1
    manifest = {str(rel): [st.st_size, int(st.st_mtime)] for rel, st in _walk(local)}
    (local / MANIFEST).write_text(json.dumps(manifest))
    meta = shared / ".meta"
    try:
        meta.mkdir(parents=True, exist_ok=True)
        (meta / "last_used").touch()
    except OSError:
        pass  # read-only users still seed
    return {"files": len(futs), "bytes": nbytes, "errors": errors, "seconds": round(time.monotonic() - t0, 2)}


# ------------------------------------------------------------------------------------- promote
def _new_subtrees(shared: Path, local: Path, settle: float = 0) -> tuple[list[Path], int, int]:
    """Minimal set of relative paths (dirs or files) that exist locally but not in shared,
    skipping files the seed brought in and units with a file modified in the last ``settle``
    seconds (a build may still be writing in place). Returns (paths, n_modified, n_unsettled)."""
    try:
        seeded = json.loads((local / MANIFEST).read_text())
    except (OSError, ValueError):
        seeded = {}
    out: list[Path] = []
    taken: set[Path] = set()
    modified = 0
    now = time.time()
    files = sorted(_walk(local), key=lambda x: str(x[0]))
    fresh = [rel for rel, st in files if settle and now - st.st_mtime < settle]
    for rel, st in files:
        if str(rel) in seeded:
            if seeded[str(rel)] != [st.st_size, int(st.st_mtime)]:
                modified += 1  # existing entries are immutable in shared: never overwrite
            continue
        if any(p == rel or p in rel.parents for p in taken):
            continue
        # highest ancestor that is missing in shared
        unit = rel
        for parent in reversed(rel.parents):
            if str(parent) == ".":
                continue
            if not (shared / parent).exists():
                unit = parent
                break
        if (shared / unit).exists():
            continue
        taken.add(unit)
        out.append(unit)
    unsettled = [u for u in out if any(f == u or u in f.parents for f in fresh)]
    out = [u for u in out if u not in unsettled]
    return out, modified, len(unsettled)


def promote(shared: Path, local: Path, settle: float = 0, max_passes: int = 5) -> dict[str, Any]:
    """Publish new local entries. A directory unit that loses the race to another job is
    re-examined in the next pass one level deeper, so entries only this job built still land."""
    t0 = time.monotonic()
    res: dict[str, Any] = {"promoted": 0, "files": 0, "bytes": 0, "lost_race": 0,
                           "skipped_modified": 0, "unsettled": 0, "errors": 0}
    stage_root = shared / ".staging" / f"{socket.gethostname()}-{os.getpid()}-{uuid.uuid4().hex[:8]}"
    try:
        for _ in range(max_passes):
            units, modified, unsettled = _new_subtrees(shared, local, settle)
            res["skipped_modified"], res["unsettled"] = modified, unsettled
            if not units:
                break
            try:
                stage_root.mkdir(parents=True, exist_ok=True)
            except OSError as e:
                res["errors"] += 1
                res["error"] = f"cannot write {shared}: {e.strerror}"
                break
            if not _promote_units(units, shared, local, stage_root, res):
                break  # no directory lost a race: nothing deeper to retry
    finally:
        shutil.rmtree(stage_root, ignore_errors=True)
    res["seconds"] = round(time.monotonic() - t0, 2)
    if res["promoted"] or res["errors"]:
        _log(shared, res)
    return res


def _log(shared: Path, res: dict[str, Any]) -> None:
    try:
        meta = shared / ".meta"
        meta.mkdir(parents=True, exist_ok=True)
        with open(meta / "promotions.log", "a") as f:
            f.write(json.dumps({"t": time.time(), "host": socket.gethostname(),
                                "job": os.environ.get("SLURM_JOB_ID"), **res}) + "\n")
    except OSError:
        pass


def _promote_units(units: list[Path], shared: Path, local: Path, stage_root: Path, res: dict[str, Any]) -> bool:
    """One pass. Returns True if a directory unit lost a race (worth another pass)."""
    retry = False
    for unit in units:
        src, dst, tmp = local / unit, shared / unit, stage_root / unit
        try:
            if src.is_dir() and not src.is_symlink():
                n = 0
                for rel, _ in _walk(src):
                    res["bytes"] += _copy_file(src / rel, tmp / rel)
                    n += 1
                if n == 0:
                    continue
                _share_modes(tmp)
                dst.parent.mkdir(parents=True, exist_ok=True)
                try:
                    os.rename(tmp, dst)          # fails if dst appeared meanwhile (non-empty)
                except OSError:
                    res["lost_race"] += 1
                    shutil.rmtree(tmp, ignore_errors=True)
                    retry = True
                    continue
                res["files"] += n
            else:
                res["bytes"] += _copy_file(src, tmp)
                _share_modes(tmp.parent)
                dst.parent.mkdir(parents=True, exist_ok=True)
                try:
                    os.link(tmp, dst)            # atomic create-if-absent
                except FileExistsError:
                    res["lost_race"] += 1
                    continue
                except OSError:                   # no hard links: rename if still absent
                    if dst.exists():
                        res["lost_race"] += 1
                        continue
                    os.rename(tmp, dst)
                res["files"] += 1
            res["promoted"] += 1
        except OSError:
            res["errors"] += 1
    return retry


def wait_ready(url: str, timeout: float) -> bool:
    if not url.startswith(("http://", "https://")):
        raise ValueError(f"not an http(s) URL: {url}")
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        with contextlib.suppress(OSError, ValueError):
            with urllib.request.urlopen(url, timeout=5) as r:  # noqa: S310  # scheme checked above
                if r.status == 200:
                    return True
        time.sleep(10)
    return False


# ------------------------------------------------------------------------------------------ gc
def _du(p: Path) -> int:
    return sum(st.st_size for _, st in _walk(p))


@dataclass
class _CacheDir:
    path: Path
    last_used: float
    bytes: int


def gc(root: Path, max_bytes: int, max_age_days: float, dry_run: bool = False) -> list[dict[str, Any]]:
    entries: list[_CacheDir] = []
    for key in sorted(root.iterdir()) if root.is_dir() else []:
        if key.name.startswith("."):
            continue
        for arch in sorted(key.iterdir()):
            if not arch.is_dir():
                continue
            lu = arch / ".meta" / "last_used"
            last = lu.stat().st_mtime if lu.exists() else arch.stat().st_mtime
            entries.append(_CacheDir(arch, last, _du(arch)))
    entries.sort(key=lambda e: e.last_used)  # oldest first
    total = sum(e.bytes for e in entries)
    now = time.time()
    evicted: list[dict[str, Any]] = []
    for e in entries:
        too_old = now - e.last_used > max_age_days * 86400
        if not too_old and total <= max_bytes:
            continue
        evicted.append({"path": str(e.path), "bytes": e.bytes, "idle_days": round((now - e.last_used) / 86400, 1)})
        total -= e.bytes
        if not dry_run:
            trash = root / ".trash" / f"{e.path.parent.name}-{e.path.name}-{int(now)}"
            trash.parent.mkdir(parents=True, exist_ok=True)
            os.rename(e.path, trash)  # atomic: a job mid-seed falls back to a cold start
            shutil.rmtree(trash, ignore_errors=True)
    return evicted


# ----------------------------------------------------------------------------------------- cli
def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    k = sub.add_parser("key")
    k.add_argument("image")
    sub.add_parser("arch")
    p = sub.add_parser("path", help="print <root>/<key>/<arch>")
    p.add_argument("image")
    p.add_argument("--root", default=os.environ.get("SML_JIT_CACHE_ROOT", DEFAULT_ROOT))
    s = sub.add_parser("seed")
    s.add_argument("--shared", required=True)
    s.add_argument("--local", default=DEFAULT_OVERLAY)
    s.add_argument("--jobs", type=int, default=16)
    pr = sub.add_parser("promote")
    pr.add_argument("--shared", required=True)
    pr.add_argument("--local", default=DEFAULT_OVERLAY)
    pr.add_argument("--when-ready", default=None, help="health URL to wait for before the first promote")
    pr.add_argument("--ready-timeout", type=float, default=6 * 3600)
    pr.add_argument("--interval", type=float, default=0, help="then promote again every N s (0: once)")
    pr.add_argument("--settle", type=float, default=60,
                    help="skip units with a file modified in the last N s (a build may be in progress)")
    g = sub.add_parser("gc")
    g.add_argument("--root", default=os.environ.get("SML_JIT_CACHE_ROOT", DEFAULT_ROOT))
    g.add_argument("--max-bytes", type=float, default=500e9)
    g.add_argument("--max-age-days", type=float, default=30)
    g.add_argument("--dry-run", action="store_true")
    d = sub.add_parser("du")
    d.add_argument("--root", default=os.environ.get("SML_JIT_CACHE_ROOT", DEFAULT_ROOT))
    a = ap.parse_args(argv)

    if a.cmd == "key":
        print(image_key(a.image))
    elif a.cmd == "arch":
        print(gpu_arch())
    elif a.cmd == "path":
        print(os.path.join(a.root, image_key(a.image), gpu_arch()))
    elif a.cmd == "seed":
        print(json.dumps({"seed": seed(Path(a.shared), Path(a.local), a.jobs)}))
    elif a.cmd == "promote":
        if a.when_ready and not wait_ready(a.when_ready, a.ready_timeout):
            print(json.dumps({"promote": "server never became ready"}))
            return 1
        while True:
            print(json.dumps({"promote": promote(Path(a.shared), Path(a.local), a.settle)}), flush=True)
            if a.interval <= 0:
                break
            time.sleep(a.interval)
    elif a.cmd == "gc":
        print(json.dumps({"evicted": gc(Path(a.root), int(a.max_bytes), a.max_age_days, a.dry_run)}, indent=1))
    elif a.cmd == "du":
        root = Path(a.root)
        for key in sorted(root.iterdir()) if root.is_dir() else []:
            for arch in sorted(key.iterdir()) if key.is_dir() and not key.name.startswith(".") else []:
                comps = {c.name: _du(c) for c in sorted(arch.iterdir()) if c.is_dir() and not c.name.startswith(".")}
                print(f"{key.name}/{arch.name}: " + ", ".join(f"{c} {b / 1e9:.2f} GB" for c, b in comps.items()))
    return 0


if __name__ == "__main__":
    sys.exit(main())
