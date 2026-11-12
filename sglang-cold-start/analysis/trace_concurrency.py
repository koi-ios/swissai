"""Per-model load, concurrency and idle time from the Swiss AI serving trace.

Input: the Parquet file written by trace_to_parquet.py.
Usage:
    python trace_concurrency.py <trace.parquet> <out_dir> [--top 25]

Every number here is an ESTIMATE derived from a request log:
- "in-flight" counts requests between created_at and finished_at for one model name,
  summed over all of that model's replicas (the trace does not say how many there were)
  and including time spent queued or in prefill. It is therefore an upper bound on the
  decode batch size any one replica saw.
- "busy time" is time with at least one request in flight. Shares of busy time stand in
  for shares of decode steps (each step takes roughly the same time at small batch sizes).
- Only requests with status DEFAULT and finished_at > created_at are used for concurrency.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

# SGLang's default decode capture list when max_running_requests is 256
# (seen in the platform's own logs: 36 sizes).
DEFAULT_DECODE_BS = [1, 2, 4, 8, 12, 16] + list(range(24, 257, 8))
THRESHOLDS = [1, 2, 4, 8, 16, 32, 64, 128, 256]


def concurrency_profile(t0: np.ndarray, t1: np.ndarray):
    """Time-weighted histogram of in-flight requests. Returns (levels, seconds)."""
    times = np.concatenate([t0, t1])
    delta = np.concatenate([np.ones_like(t0, dtype=np.int64), -np.ones_like(t1, dtype=np.int64)])
    # at equal timestamps apply departures first
    order = np.lexsort((delta, times))
    times, delta = times[order], delta[order]
    level = np.cumsum(delta)
    dur = np.diff(times)
    lv = level[:-1]
    maxlv = int(lv.max()) if lv.size else 0
    secs = np.bincount(lv, weights=dur, minlength=maxlv + 1)
    return secs, times[0], times[-1]


def busy_quantile(secs: np.ndarray, q: float) -> int:
    busy = secs[1:]
    tot = busy.sum()
    if tot <= 0:
        return 0
    c = np.cumsum(busy) / tot
    return int(np.searchsorted(c, q) + 1)


def idle_gaps(t0: np.ndarray, t1: np.ndarray) -> np.ndarray:
    """Lengths (s) of the periods with no request in flight, between busy periods."""
    order = np.argsort(t0)
    s, e = t0[order], t1[order]
    run_end = np.maximum.accumulate(e)
    gaps = s[1:] - run_end[:-1]
    return gaps[gaps > 0]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("parquet")
    ap.add_argument("out_dir")
    ap.add_argument("--top", type=int, default=25)
    a = ap.parse_args()
    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    df = pd.read_parquet(a.parquet)
    df["model"] = df["model"].astype(str).str.strip()
    n_all = len(df)
    t_first = pd.to_datetime(df["t0"].min(), unit="s", utc=True)
    t_last = pd.to_datetime(df["t0"].max(), unit="s", utc=True)
    status_counts = df["status"].value_counts().to_dict()

    ok = df[(df["status"] == "DEFAULT") & np.isfinite(df["t0"]) & np.isfinite(df["t1"]) & (df["t1"] > df["t0"])].copy()
    ok["dur"] = ok["t1"] - ok["t0"]
    # drop absurd durations (clock problems, stuck streams) from concurrency: > 2 h
    long_mask = ok["dur"] > 7200
    n_long = int(long_mask.sum())
    ok = ok[~long_mask]

    counts = df.groupby("model").size().sort_values(ascending=False)
    top = [m for m in counts.index if m in set(ok["model"])][: a.top]

    rows = []
    cdf = {}
    for m in top:
        g = ok[ok["model"] == m]
        t0 = g["t0"].to_numpy()
        t1 = g["t1"].to_numpy()
        secs, start, end = concurrency_profile(t0, t1)
        window = end - start
        busy = secs[1:].sum()
        gaps = idle_gaps(t0, t1)
        days = pd.to_datetime(g["t0"], unit="s", utc=True).dt.floor("D")
        n_days = days.nunique()
        span_days = max(window / 86400.0, 1e-9)
        tin = g.loc[g["tin"] > 0, "tin"]
        tout = g.loc[g["tout"] > 0, "tout"]
        busy_share = {f"busy_le_{k}": float(secs[1 : k + 1].sum() / busy) if busy > 0 else np.nan for k in THRESHOLDS}
        row = {
            "model": m,
            "requests_all": int(counts[m]),
            "requests_ok": int(len(g)),
            "first": str(pd.to_datetime(start, unit="s", utc=True).date()),
            "last": str(pd.to_datetime(end, unit="s", utc=True).date()),
            "days_with_traffic": int(n_days),
            "span_days": round(span_days, 1),
            "req_per_active_day": round(len(g) / max(n_days, 1), 1),
            "dur_p50_s": round(float(g["dur"].quantile(0.5)), 2),
            "dur_p99_s": round(float(g["dur"].quantile(0.99)), 1),
            "in_p50": int(tin.quantile(0.5)) if len(tin) else -1,
            "in_p99": int(tin.quantile(0.99)) if len(tin) else -1,
            "out_p50": int(tout.quantile(0.5)) if len(tout) else -1,
            "out_p99": int(tout.quantile(0.99)) if len(tout) else -1,
            "idle_share_of_span": round(float(secs[0] / window), 4) if window > 0 else np.nan,
            "busy_hours": round(busy / 3600.0, 1),
            "inflight_p50": busy_quantile(secs, 0.50),
            "inflight_p90": busy_quantile(secs, 0.90),
            "inflight_p99": busy_quantile(secs, 0.99),
            "inflight_p999": busy_quantile(secs, 0.999),
            "inflight_max": int(len(secs) - 1),
            "idle_gaps_ge_5min_per_day": round(float((gaps >= 300).sum() / span_days), 2),
            "idle_gaps_ge_15min_per_day": round(float((gaps >= 900).sum() / span_days), 2),
            "idle_gaps_ge_60min_per_day": round(float((gaps >= 3600).sum() / span_days), 2),
            **{k: round(v, 5) for k, v in busy_share.items()},
        }
        rows.append(row)
        cdf[m] = (np.cumsum(secs[1:]) / busy).tolist() if busy > 0 else []

    res = pd.DataFrame(rows)
    res.to_csv(out / "per_model.csv", index=False)
    (out / "inflight_cdf.json").write_text(json.dumps(cdf))
    meta = {
        "rows": n_all,
        "first_request_utc": str(t_first),
        "last_request_utc": str(t_last),
        "status_counts": {str(k): int(v) for k, v in status_counts.items()},
        "ok_requests_used": int(len(ok)),
        "dropped_longer_than_2h": n_long,
        "distinct_models": int(df["model"].nunique()),
        "default_decode_capture_list": DEFAULT_DECODE_BS,
    }
    (out / "meta.json").write_text(json.dumps(meta, indent=1))
    print(json.dumps(meta, indent=1))
    with pd.option_context("display.max_columns", 50, "display.width", 250):
        print(res[["model", "requests_all", "first", "last", "days_with_traffic", "req_per_active_day", "dur_p50_s",
                   "in_p50", "out_p50", "idle_share_of_span", "inflight_p50", "inflight_p90", "inflight_p99",
                   "inflight_p999", "inflight_max", "busy_le_8", "busy_le_32", "busy_le_64",
                   "idle_gaps_ge_15min_per_day"]].to_string(index=False))


if __name__ == "__main__":
    main()
