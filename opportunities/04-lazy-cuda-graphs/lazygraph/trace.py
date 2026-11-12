"""Per-model traffic from the Swiss AI serving trace (OpenTela, OSDI '26).

Input: the Parquet file written by sglang-cold-start/analysis/trace_to_parquet.py
(columns model, status, t0, t1, tin, tout, max_tokens; t0/t1 in UTC seconds).

Every quantity here is an ESTIMATE derived from a request log:
- A request is "in flight" from t0 (created) to t1 (finished), summed over all
  replicas of the model and including queueing; so in-flight counts are an
  upper bound on any one replica's decode batch.
- Only status DEFAULT with t1 > t0 and duration <= 2 h is used (as in
  sglang-cold-start/analysis/trace_concurrency.py).
- One request = one prefill forward at bucket(tin); chunking above the
  prefill maximum and co-batched prefills are ignored.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

MAX_DURATION_S = 7200.0


@lru_cache(maxsize=2)
def load_trace(path: str) -> pd.DataFrame:
    df = pd.read_parquet(path, columns=["model", "status", "t0", "t1", "tin", "tout"])
    df["model"] = df["model"].astype(str).str.strip()
    ok = df[
        (df["status"] == "DEFAULT")
        & np.isfinite(df["t0"])
        & np.isfinite(df["t1"])
        & (df["t1"] > df["t0"])
        & ((df["t1"] - df["t0"]) <= MAX_DURATION_S)
    ]
    return ok.sort_values("t0").reset_index(drop=True)


@dataclass
class ModelTraffic:
    model: str
    t0: np.ndarray
    t1: np.ndarray
    tin: np.ndarray  # -1 where unknown
    tout: np.ndarray

    @property
    def span_s(self) -> float:
        return float(self.t1.max() - self.t0.min()) if len(self.t0) else 0.0

    def busy_intervals(self) -> np.ndarray:
        """Merged [start, end) intervals with >= 1 request in flight."""
        return merge_intervals(self.t0, self.t1)

    def inflight_seconds(self) -> np.ndarray:
        """secs[k] = seconds with exactly k requests in flight (k >= 0)."""
        return inflight_histogram(self.t0, self.t1)

    def prompt_tokens(self) -> np.ndarray:
        return self.tin[self.tin > 0]


def model_traffic(trace: pd.DataFrame, model: str) -> ModelTraffic:
    g = trace[trace["model"] == model]
    if g.empty:
        raise KeyError(f"no successful requests with durations for {model!r}")
    return ModelTraffic(
        model=model,
        t0=g["t0"].to_numpy(dtype=np.float64),
        t1=g["t1"].to_numpy(dtype=np.float64),
        tin=g["tin"].to_numpy(dtype=np.int64),
        tout=g["tout"].to_numpy(dtype=np.int64),
    )


def merge_intervals(t0: np.ndarray, t1: np.ndarray) -> np.ndarray:
    if len(t0) == 0:
        return np.empty((0, 2))
    order = np.argsort(t0, kind="stable")
    s, e = t0[order], t1[order]
    run_end = np.maximum.accumulate(e)
    # A new busy period starts where a request begins after all earlier ones ended.
    new = np.empty(len(s), dtype=bool)
    new[0] = True
    new[1:] = s[1:] > run_end[:-1]
    starts = s[new]
    idx = np.flatnonzero(new)
    ends = np.append(run_end[idx[1:] - 1], run_end[-1])
    return np.stack([starts, ends], axis=1)


def inflight_histogram(t0: np.ndarray, t1: np.ndarray) -> np.ndarray:
    times = np.concatenate([t0, t1])
    delta = np.concatenate(
        [np.ones(len(t0), dtype=np.int64), -np.ones(len(t1), dtype=np.int64)]
    )
    order = np.lexsort((delta, times))  # departures first at equal timestamps
    times, delta = times[order], delta[order]
    level = np.cumsum(delta)[:-1]
    dur = np.diff(times)
    return np.bincount(level, weights=dur, minlength=int(level.max()) + 1)


def sessions(busy: np.ndarray, min_gap_s: float) -> List[Tuple[float, float]]:
    """(launch_time, prev_gap_s) for busy periods preceded by an idle gap >= min_gap_s.

    Models scale-to-zero after `min_gap_s` of idleness: the first request after
    such a gap triggers a cold start.
    """
    out = []
    if len(busy) == 0:
        return out
    out.append((float(busy[0, 0]), float("inf")))
    gaps = busy[1:, 0] - busy[:-1, 1]
    for i in np.flatnonzero(gaps >= min_gap_s):
        out.append((float(busy[i + 1, 0]), float(gaps[i])))
    return out


def summarize(traffic: ModelTraffic) -> Dict[str, float]:
    busy = traffic.busy_intervals()
    gaps = busy[1:, 0] - busy[:-1, 1] if len(busy) > 1 else np.array([])
    secs = traffic.inflight_seconds()
    busy_s = float(secs[1:].sum())
    tin = traffic.prompt_tokens()
    span_days = max(traffic.span_s / 86400.0, 1e-9)

    def busy_quantile(q: float) -> int:
        c = np.cumsum(secs[1:]) / max(busy_s, 1e-9)
        return int(np.searchsorted(c, q) + 1)

    return {
        "requests": int(len(traffic.t0)),
        "span_days": round(traffic.span_s / 86400.0, 2),
        "idle_share": round(1.0 - busy_s / max(traffic.span_s, 1e-9), 4),
        "inflight_p50": busy_quantile(0.5),
        "inflight_p99": busy_quantile(0.99),
        "prompt_p50": int(np.quantile(tin, 0.5)) if len(tin) else -1,
        "prompt_p99": int(np.quantile(tin, 0.99)) if len(tin) else -1,
        "idle_gaps_ge_15min_per_day": round(float((gaps >= 900).sum()) / span_days, 2),
        "idle_gap_p50_s": round(float(np.quantile(gaps, 0.5)), 2) if len(gaps) else -1,
        "idle_gap_p90_s": round(float(np.quantile(gaps, 0.9)), 2) if len(gaps) else -1,
    }
