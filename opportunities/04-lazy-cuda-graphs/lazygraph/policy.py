"""Expected-value ranking of capture buckets and the startup set.

For every planned bucket j (phase decode or prefill):
  f_j  request-forwards per second of deployment time that would use bucket j
       (trace estimate; see trace.py). A decode step at batch size k counts k
       times, since an eager step delays all k requests; a prefill counts once,
  d_j  extra seconds per forward when j runs eagerly (costs.EagerPenalty),
  c_j  seconds to capture j (costs.CaptureCost),
  w_j = f_j * d_j   latency added per second while j is not captured,
  EV_j = w_j / c_j  value density.

Deferred order: decreasing EV_j. Capturing one bucket at a time on one
"machine" (the idle GPU) and accumulating penalty w_j until bucket j is done is
the single-machine weighted-completion-time problem 1||sum w_j C_j, for which
this order (Smith's rule, WSPT) is optimal when steps are not interrupted.

Startup set: capturing j before serving delays the lam requests that wait for
the server by c_j each; deferring it costs about w_j * H, where H is how long a
deferred bucket typically waits. Capture j at startup iff
  w_j * H > lam * c_j  <=>  EV_j > lam / H,
so one ranking and one threshold give both decisions. Bulk endpoints have
large f_j, so every bucket clears the threshold and the policy degenerates to
the default (capture everything at startup).
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from typing import Dict, List, Optional, Sequence

import numpy as np

from lazygraph.buckets import bucket_of, decode_buckets, prefill_buckets
from lazygraph.costs import EngineProfile
from lazygraph.trace import ModelTraffic


@dataclass(frozen=True)
class BucketValue:
    phase: str
    size: int
    rate_per_s: float
    penalty_s: float
    cost_s: float

    @property
    def weight(self) -> float:
        return self.rate_per_s * self.penalty_s

    @property
    def ev(self) -> float:
        return self.weight / self.cost_s if self.cost_s > 0 else float("inf")


def bucket_rates(
    traffic: ModelTraffic,
    *,
    prefill: Sequence[int],
    decode: Sequence[int],
    decode_steps_per_busy_s: float,
) -> Dict[str, Dict[int, float]]:
    span = max(traffic.span_s, 1e-9)
    pre: Dict[int, float] = {b: 0.0 for b in prefill}
    for tokens in traffic.prompt_tokens():
        b = bucket_of(int(tokens), prefill)
        if b is not None:  # above the maximum: always eager, not a bucket
            pre[b] += 1.0
    pre = {b: n / span for b, n in pre.items()}

    dec: Dict[int, float] = {b: 0.0 for b in decode}
    secs = traffic.inflight_seconds()
    for level in range(1, len(secs)):
        if secs[level] <= 0:
            continue
        b = bucket_of(level, decode)
        if b is not None:
            dec[b] += secs[level] * decode_steps_per_busy_s * level
    dec = {b: n / span for b, n in dec.items()}
    return {"prefill": pre, "decode": dec}


def rank_buckets(
    traffic: ModelTraffic,
    profile: EngineProfile,
    *,
    prefill: Optional[Sequence[int]] = None,
    decode: Optional[Sequence[int]] = None,
) -> List[BucketValue]:
    prefill = list(prefill or prefill_buckets())
    decode = list(decode or decode_buckets())
    rates = bucket_rates(
        traffic,
        prefill=prefill,
        decode=decode,
        decode_steps_per_busy_s=profile.decode_steps_per_busy_s,
    )
    values = []
    for phase, sizes in (("decode", decode), ("prefill", prefill)):
        for size in sizes:
            values.append(
                BucketValue(
                    phase=phase,
                    size=size,
                    rate_per_s=rates[phase][size],
                    penalty_s=profile.eager_penalty(phase, size),
                    cost_s=profile.capture_cost(phase, size),
                )
            )
    # Ties (e.g. zero traffic) broken by cheaper capture first, then decode.
    values.sort(key=lambda v: (-v.ev, v.cost_s, v.phase != "decode", v.size))
    return values


@dataclass(frozen=True)
class Plan:
    model: str
    profile: str
    threshold: float
    lam: float
    horizon_s: float
    initial: Dict[str, List[int]]
    deferred: Dict[str, List[int]]
    priority: Dict[str, List[float]]
    planned: Dict[str, List[int]]
    startup_capture_s: float
    deferred_capture_s: float
    default_capture_s: float
    coverage: Dict[str, float]

    @property
    def lazy(self) -> bool:
        return any(self.deferred[p] for p in self.deferred)

    def cuda_graph_config(self) -> dict:
        """JSON for SGLang 0.5.20 + this patch: --cuda-graph-config '<json>'."""
        cfg = {}
        for phase in ("decode", "prefill"):
            entry = {"bs": self.planned[phase]}
            if self.lazy:
                entry["initial_bs"] = self.initial[phase]
                entry["deferred_bs"] = self.deferred[phase]
                entry["deferred_priority"] = [
                    float(f"{p:.6g}") for p in self.priority[phase]
                ]
            cfg[phase] = entry
        return cfg

    def to_json(self) -> dict:
        d = asdict(self)
        d["cuda_graph_config"] = self.cuda_graph_config()
        d["lazy"] = self.lazy
        return d


def make_plan(
    traffic: ModelTraffic,
    profile: EngineProfile,
    *,
    lam: float = 1.0,
    horizon_s: float = 60.0,
    prefill: Optional[Sequence[int]] = None,
    decode: Optional[Sequence[int]] = None,
) -> Plan:
    prefill = list(prefill or prefill_buckets())
    decode = list(decode or decode_buckets())
    values = rank_buckets(traffic, profile, prefill=prefill, decode=decode)
    threshold = lam / horizon_s
    initial = {"decode": [], "prefill": []}
    deferred = {"decode": [], "prefill": []}
    priority = {"decode": [], "prefill": []}
    for v in values:  # EV order
        if v.ev > threshold:
            initial[v.phase].append(v.size)
        else:
            deferred[v.phase].append(v.size)
            priority[v.phase].append(v.ev)
    for phase in initial:
        initial[phase].sort()
    startup = sum(v.cost_s for v in values if v.size in initial[v.phase])
    later = sum(v.cost_s for v in values if v.size in deferred[v.phase])
    return Plan(
        model=traffic.model,
        profile=profile.name,
        threshold=threshold,
        lam=lam,
        horizon_s=horizon_s,
        initial=initial,
        deferred=deferred,
        priority=priority,
        planned={"decode": decode, "prefill": prefill},
        startup_capture_s=startup,
        deferred_capture_s=later,
        default_capture_s=startup + later,
        coverage=_coverage(values, initial),
    )


def _coverage(values: List[BucketValue], initial: Dict[str, List[int]]) -> Dict[str, float]:
    out = {}
    for phase in ("decode", "prefill"):
        total = sum(v.rate_per_s for v in values if v.phase == phase)
        hit = sum(
            v.rate_per_s for v in values if v.phase == phase and v.size in initial[phase]
        )
        out[f"{phase}_forwards_on_startup_graphs"] = (
            round(hit / total, 4) if total > 0 else 1.0
        )
    return out


def dump_plan(plan: Plan, path: str) -> None:
    with open(path, "w") as f:
        json.dump(plan.to_json(), f, indent=1)


def values_table(values: List[BucketValue], top: int = 20) -> str:
    rows = ["phase    size   rate/s      penalty_ms  cost_s  EV"]
    for v in values[:top]:
        rows.append(
            f"{v.phase:8s} {v.size:5d}  {v.rate_per_s:10.4g}  {v.penalty_s * 1e3:9.2f}  "
            f"{v.cost_s:6.3f}  {v.ev:.4g}"
        )
    return "\n".join(rows)


def quantiles(x: np.ndarray, qs=(0.5, 0.9, 0.99)) -> Dict[str, float]:
    if len(x) == 0:
        return {f"p{int(q * 100)}": float("nan") for q in qs}
    return {f"p{int(q * 100)}": float(np.quantile(x, q)) for q in qs}
