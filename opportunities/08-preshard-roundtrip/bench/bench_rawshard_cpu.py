"""CPU micro-benchmark of raw presharding's own overheads (not a GPU load-time measurement).

A synthetic checkpoint of bf16 matrices is "loaded" the way a TP=4 rank loads
it: column-parallel tensors narrow dim 0, row-parallel ones narrow dim 1, and
the slice is copied into a preallocated parameter. Four ways, each timed as
the median (min-max) of 5 runs with the files in the page cache:

  plain       the ordinary iterator, no servekit
  record      the same load through RecordSession (what `servekit prepare --preshard-format raw` adds)
  replay      the load fed from the raw preshard, verification on (the default)
  replay_nv   the same with SERVEKIT_RAW_VERIFY=0

Two shapes: many small tensors (per-tensor Python overhead dominates) and fewer
large ones (byte copies dominate).

    ../.venv/bin/python bench/bench_rawshard_cpu.py   # writes results/cpu_overhead.json
"""
import json
import os
import platform
import statistics
import sys
import tempfile
import time
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "vendor" / "servekit" / "src"))
from servekit.rawshard.store import RawIndex, RecordSession  # noqa: E402

TP, RANK, REPEATS = 4, 1, 5


def make(directory: Path, n: int, rows: int, cols: int) -> list:
    g = torch.Generator().manual_seed(0)
    names, per_file, paths = [f"layer.{i}.weight" for i in range(n)], max(1, n // 4), []
    for f in range(0, n, per_file):
        part = {k: torch.randn(rows, cols, generator=g).bfloat16() for k in names[f : f + per_file]}
        path = directory / f"model-{f // per_file:05d}.safetensors"
        save_file(part, str(path))
        paths.append(path)
    return paths


def iterate(paths):
    for p in paths:
        with safe_open(str(p), framework="pt") as f:
            for k in f.keys():  # noqa: SIM118
                yield k, f.get_tensor(k)


def load(weights, rows, cols):
    params = {}
    for name, w in weights:
        i = int(name.split(".")[1])
        if i % 2 == 0:  # column parallel
            shard = rows // TP
            p = params.setdefault(name, torch.empty(shard, cols, dtype=torch.bfloat16))
            p.copy_(w.narrow(0, RANK * shard, shard))
        else:  # row parallel: a strided read
            shard = cols // TP
            p = params.setdefault(name, torch.empty(rows, shard, dtype=torch.bfloat16))
            p.copy_(w.narrow(1, RANK * shard, shard))
    return params


def timed(fn):
    samples = []
    for _ in range(REPEATS):
        t0 = time.perf_counter()
        fn()
        samples.append(time.perf_counter() - t0)
    return {"median_s": statistics.median(samples), "min_s": min(samples), "max_s": max(samples)}


def case(n: int, rows: int, cols: int) -> dict:
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        paths = make(tmp / "ck" if (tmp / "ck").mkdir() is None else tmp, n, rows, cols)
        ck_bytes = sum(p.stat().st_size for p in paths)
        load(iterate(paths), rows, cols)  # warm the page cache
        out = {"tensors": n, "shape": [rows, cols], "checkpoint_mb": round(ck_bytes / 1e6, 1)}
        out["plain"] = timed(lambda: load(iterate(paths), rows, cols))

        def record():
            d = tmp / "raw"
            session = RecordSession(str(d), RANK)
            load(session.stream("target/0", iterate(paths)), rows, cols)
            return session.finish()

        out["record"] = timed(record)
        stats = record()
        out["stored_mb"] = round(stats["stored_bytes"] / 1e6, 1)
        index = RawIndex(str(tmp / "raw"), RANK)
        out["replay"] = timed(lambda: (load(index.replay(0, verify=True), rows, cols), index.verdict()))
        out["replay_nv"] = timed(lambda: load(index.replay(0, verify=False), rows, cols))
        assert index.verdict() == [], index.verdict()
        reference = load(iterate(paths), rows, cols)
        replayed = load(index.replay(0, verify=False), rows, cols)
        assert all(torch.equal(reference[k], replayed[k]) for k in reference)
        for k in ("record", "replay", "replay_nv"):
            out[f"{k}_overhead_us_per_tensor"] = round((out[k]["median_s"] - out["plain"]["median_s"]) / n * 1e6, 1)
        return out


def main() -> int:
    torch.set_num_threads(min(8, os.cpu_count() or 1))
    results = {
        "machine": f"{platform.machine()} {platform.system()} {platform.release()}, torch {torch.__version__}, "
                   f"{torch.get_num_threads()} threads",
        "note": "CPU only; loads into CPU tensors. Measures servekit's own overhead, not GPU load time.",
        "cases": [case(4000, 64, 64), case(64, 4096, 2048)],
    }
    out = ROOT / "results" / "cpu_overhead.json"
    out.write_text(json.dumps(results, indent=2) + "\n")
    for c in results["cases"]:
        print(f"{c['tensors']} x {c['shape']} ({c['checkpoint_mb']} MB, stores {c['stored_mb']} MB for rank {RANK}/{TP}):")
        for k in ("plain", "record", "replay", "replay_nv"):
            r = c[k]
            extra = f"  (+{c[k + '_overhead_us_per_tensor']} us/tensor)" if k != "plain" else ""
            print(f"  {k:10s} {r['median_s'] * 1e3:8.1f} ms ({r['min_s'] * 1e3:.1f}-{r['max_s'] * 1e3:.1f}){extra}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
