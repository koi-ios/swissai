"""Convert the OpenTela / Swiss AI serving trace (trace.jsonl) to a compact Parquet file.

Source: https://huggingface.co/datasets/eth-easl/swissai-serving-trace (file trace.jsonl,
16,329,237 rows, 6.98 GB). Download it first, for example:

    curl -L -o trace.jsonl \
      https://huggingface.co/datasets/eth-easl/swissai-serving-trace/resolve/main/trace.jsonl

Usage:
    python trace_to_parquet.py <trace.jsonl> <out.parquet> [--workers N]

Keeps one row per request: model, status, created/finished time (UTC epoch seconds),
reported input/output tokens (-1 when missing) and the client's max_tokens (NaN when
missing or not a number). Splits the file into byte ranges and parses them in parallel.
"""
from __future__ import annotations

import argparse
import os
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime

import numpy as np
import orjson
import pyarrow as pa
import pyarrow.parquet as pq


def _epoch(s) -> float:
    if not s:
        return np.nan
    try:
        return datetime.fromisoformat(s.replace("Z", "+00:00")).timestamp()
    except ValueError:
        return np.nan


def _num(x) -> float:
    if isinstance(x, bool):
        return np.nan
    if isinstance(x, (int, float)):
        return float(x)
    if isinstance(x, str):
        try:
            return float(x)
        except ValueError:
            return np.nan
    return np.nan


def _ranges(path: str, n: int):
    size = os.path.getsize(path)
    step = size // n
    cuts = [0]
    with open(path, "rb") as f:
        for i in range(1, n):
            f.seek(i * step)
            f.readline()  # move to the start of the next full line
            cuts.append(f.tell())
    cuts.append(size)
    return [(cuts[i], cuts[i + 1]) for i in range(n) if cuts[i + 1] > cuts[i]]


def _parse(args):
    path, start, end = args
    model, status, t0, t1, tin, tout, maxtok = [], [], [], [], [], [], []
    bad = 0
    with open(path, "rb") as f:
        f.seek(start)
        while f.tell() < end:
            line = f.readline()
            if not line:
                break
            line = line.strip()
            if not line:
                continue
            try:
                r = orjson.loads(line)
            except orjson.JSONDecodeError:
                bad += 1
                continue
            mp = r.get("model_parameters") or {}
            model.append(r.get("model") or "")
            status.append(r.get("status") or "")
            t0.append(_epoch(r.get("created_at")))
            t1.append(_epoch(r.get("finished_at")))
            tin.append(int(r.get("reported_token_input", -1) if r.get("reported_token_input") is not None else -1))
            tout.append(int(r.get("reported_token_output", -1) if r.get("reported_token_output") is not None else -1))
            maxtok.append(_num(mp.get("max_tokens")) if isinstance(mp, dict) else np.nan)
    table = pa.table(
        {
            "model": pa.array(model, pa.string()).dictionary_encode(),
            "status": pa.array(status, pa.string()).dictionary_encode(),
            "t0": pa.array(t0, pa.float64()),
            "t1": pa.array(t1, pa.float64()),
            "tin": pa.array(tin, pa.int64()),
            "tout": pa.array(tout, pa.int64()),
            "max_tokens": pa.array(maxtok, pa.float64()),
        }
    )
    return table, bad


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("trace")
    ap.add_argument("out")
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 2))
    a = ap.parse_args()
    jobs = [(a.trace, s, e) for s, e in _ranges(a.trace, a.workers * 4)]
    tables, bad = [], 0
    with ProcessPoolExecutor(a.workers) as ex:
        for t, b in ex.map(_parse, jobs):
            tables.append(t.cast(pa.schema([
                ("model", pa.string()), ("status", pa.string()), ("t0", pa.float64()), ("t1", pa.float64()),
                ("tin", pa.int64()), ("tout", pa.int64()), ("max_tokens", pa.float64()),
            ])))
            bad += b
    table = pa.concat_tables(tables)
    pq.write_table(table, a.out, compression="zstd")
    print(f"rows={table.num_rows} unparseable_lines={bad} -> {a.out}")


if __name__ == "__main__":
    main()
