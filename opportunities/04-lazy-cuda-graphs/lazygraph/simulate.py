"""Trace-driven simulation of deferred CUDA-graph capture after a cold start.

Launches. A model scales to zero after `session_gap_s` without traffic; the
first request after such a gap triggers a cold start (the trace's first
request of a model counts as one too). Each launch is simulated for
`horizon_s` seconds of trace time.

Arms (only graph capture differs; process start, NCCL init, weights and KV
allocation are identical and left out):
  default  capture every planned bucket before serving (SGLang default)
  fixed    capture the startup set only; other buckets stay eager forever
  lazy     capture the startup set, then the deferred buckets one per step
           during idle time (the patch's controller rule)

Timeline model (APPROXIMATION, stated in the write-up):
  - Busy/idle periods are the trace's own (requests in flight from t0 to t1),
    so the server's speed in each arm does not feed back into the timeline.
  - Requests that arrive before the server is ready wait for it; that wait is
    the startup cost of the arm.
  - A deferred step may start in an idle period once the leader has seen
    min_idle_s + idle_cost_ratio * cost(step) of idleness. Steps are not
    preemptible: requests arriving during a step wait for its end.
  - Eager penalties: one prefill per request at bucket(tin); decode steps at
    the in-flight level k, steps_per_busy_s per second, each delaying k
    requests by penalty(bucket(k)).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from lazygraph.buckets import bucket_of
from lazygraph.costs import EngineProfile
from lazygraph.policy import Plan
from lazygraph.trace import ModelTraffic, merge_intervals, sessions


@dataclass(frozen=True)
class SimConfig:
    min_idle_s: float = 1.0
    idle_cost_ratio: float = 0.0
    horizon_s: float = 1800.0
    session_gap_s: float = 900.0
    max_launches: int = 2000


@dataclass
class ArmResult:
    ready_s: float  # capture time before serving (startup share of the arm)
    startup_wait_s: float = 0.0  # summed over requests that arrived before ready
    eager_prefill_requests: int = 0
    eager_prefill_penalty_s: float = 0.0
    eager_decode_penalty_s: float = 0.0
    interference_s: float = 0.0
    delayed_requests: int = 0
    max_delay_s: float = 0.0
    all_captured_after_s: Optional[float] = None  # since launch; lazy only
    steps_done: int = 0

    @property
    def total_extra_s(self) -> float:
        return (
            self.startup_wait_s
            + self.eager_prefill_penalty_s
            + self.eager_decode_penalty_s
            + self.interference_s
        )


@dataclass
class LaunchResult:
    launch_t: float
    prev_gap_s: float
    requests: int
    arms: Dict[str, ArmResult] = field(default_factory=dict)


def deferred_steps(plan: Plan, profile: EngineProfile) -> List[Tuple[str, int, float]]:
    """Deferred buckets merged across phases by priority, as the controller does."""
    heads = {p: 0 for p in ("decode", "prefill")}
    out = []
    while True:
        best = None
        for phase in ("decode", "prefill"):
            i = heads[phase]
            if i < len(plan.deferred[phase]):
                pr = plan.priority[phase][i]
                if best is None or pr > best[2]:
                    best = (phase, plan.deferred[phase][i], pr)
        if best is None:
            return out
        phase, size, _ = best
        heads[phase] += 1
        out.append((phase, size, profile.capture_cost(phase, size)))


def _place_steps(
    idle: np.ndarray, steps: List[Tuple[str, int, float]], cfg: SimConfig
) -> List[Tuple[str, int, float, float, Optional[float]]]:
    """Greedy placement: returns (phase, size, start, end, interrupted_at)."""
    placed = []
    k = 0
    for u, v in idle:
        t = u
        while k < len(steps):
            phase, size, cost = steps[k]
            start = max(t, u + cfg.min_idle_s + cfg.idle_cost_ratio * cost)
            if start >= v:
                break
            end = start + cost
            placed.append((phase, size, start, end, v if end > v else None))
            k += 1
            t = end
            if end > v:
                break
    return placed


def _inflight_segments(t0: np.ndarray, t1: np.ndarray) -> np.ndarray:
    """Rows (start, end, level) of the in-flight step function, level >= 1."""
    times = np.concatenate([t0, t1])
    delta = np.concatenate([np.ones(len(t0)), -np.ones(len(t1))])
    order = np.lexsort((delta, times))
    times, delta = times[order], delta[order]
    level = np.cumsum(delta)[:-1]
    seg = np.stack([times[:-1], times[1:], level], axis=1)
    return seg[(seg[:, 2] >= 1) & (seg[:, 1] > seg[:, 0])]


def simulate_launch(
    traffic: ModelTraffic,
    launch_t: float,
    prev_gap: float,
    plan: Plan,
    profile: EngineProfile,
    cfg: SimConfig,
) -> LaunchResult:
    sel = (traffic.t0 >= launch_t) & (traffic.t0 < launch_t + cfg.horizon_s)
    a = traffic.t0[sel] - launch_t
    e = np.minimum(traffic.t1[sel], launch_t + cfg.horizon_s) - launch_t
    tin = traffic.tin[sel]
    res = LaunchResult(launch_t=launch_t, prev_gap_s=prev_gap, requests=int(sel.sum()))
    prefill_planned = plan.planned["prefill"]
    decode_planned = plan.planned["decode"]
    segments = _inflight_segments(a, e)
    busy = merge_intervals(a, e)

    startup = {
        "default": plan.default_capture_s,
        "fixed": plan.startup_capture_s,
        "lazy": plan.startup_capture_s,
    }
    for arm, ready in startup.items():
        r = ArmResult(ready_s=ready)
        r.startup_wait_s = float(np.clip(ready - a, 0, None).sum())
        captured_at: Dict[Tuple[str, int], float] = {}
        if arm == "default" or not plan.lazy:
            for phase in ("decode", "prefill"):
                for s in plan.planned[phase]:
                    captured_at[(phase, s)] = 0.0
        else:
            for phase in ("decode", "prefill"):
                for s in plan.initial[phase]:
                    captured_at[(phase, s)] = 0.0
        if arm == "lazy" and plan.lazy:
            idle = _idle_after(busy, ready, cfg.horizon_s)
            placed = _place_steps(idle, deferred_steps(plan, profile), cfg)
            for phase, size, start, end, cut in placed:
                captured_at[(phase, size)] = end
                r.steps_done += 1
                if cut is not None:
                    waiting = (a >= cut) & (a < end)
                    r.delayed_requests += int(waiting.sum())
                    if waiting.any():
                        delays = end - a[waiting]
                        r.interference_s += float(delays.sum())
                        r.max_delay_s = max(r.max_delay_s, float(delays.max()))
            total = sum(len(plan.deferred[p]) for p in plan.deferred)
            if r.steps_done == total:
                r.all_captured_after_s = max(
                    (end for _, _, _, end, _ in placed), default=ready
                )
        _eager_penalties(
            r,
            a=a,
            tin=tin,
            segments=segments,
            ready=ready,
            captured_at=captured_at,
            prefill_planned=prefill_planned,
            decode_planned=decode_planned,
            profile=profile,
        )
        res.arms[arm] = r
    return res


def _idle_after(busy: np.ndarray, ready: float, horizon: float) -> np.ndarray:
    """Idle periods in [ready, horizon) given busy intervals (requests arriving
    before `ready` are served from `ready`, so the first busy period may start
    then)."""
    out = []
    t = ready
    for s, e in busy:
        if e <= t:
            continue
        if s > t:
            out.append((t, min(s, horizon)))
        t = max(t, e)
        if t >= horizon:
            break
    if t < horizon:
        out.append((t, horizon))
    return np.array([iv for iv in out if iv[1] > iv[0]]).reshape(-1, 2)


def _eager_penalties(
    r: ArmResult,
    *,
    a: np.ndarray,
    tin: np.ndarray,
    segments: np.ndarray,
    ready: float,
    captured_at: Dict[Tuple[str, int], float],
    prefill_planned: Sequence[int],
    decode_planned: Sequence[int],
    profile: EngineProfile,
) -> None:
    for t_arr, tokens in zip(a, tin):
        if tokens <= 0:
            continue
        b = bucket_of(int(tokens), prefill_planned)
        if b is None:
            continue  # above the maximum: eager in every arm
        t_run = max(t_arr, ready)
        if captured_at.get(("prefill", b), np.inf) > t_run:
            r.eager_prefill_requests += 1
            r.eager_prefill_penalty_s += profile.eager_penalty("prefill", b)
    steps_per_s = profile.decode_steps_per_busy_s
    for start, end, level in segments:
        b = bucket_of(int(level), decode_planned)
        if b is None:
            continue
        cap = captured_at.get(("decode", b), np.inf)
        lo, hi = max(start, ready), min(end, cap)
        if hi > lo:
            steps = (hi - lo) * steps_per_s
            r.eager_decode_penalty_s += steps * level * profile.eager_penalty(
                "decode", b
            )


def simulate_model(
    traffic: ModelTraffic, plan: Plan, profile: EngineProfile, cfg: SimConfig
) -> List[LaunchResult]:
    launches = sessions(traffic.busy_intervals(), cfg.session_gap_s)[: cfg.max_launches]
    return [
        simulate_launch(traffic, t, gap, plan, profile, cfg) for t, gap in launches
    ]


def aggregate(results: List[LaunchResult]) -> Dict[str, dict]:
    out: Dict[str, dict] = {"launches": len(results)}
    if not results:
        return out
    n_req = np.array([r.requests for r in results])
    out["requests_per_launch_p50"] = float(np.median(n_req))
    for arm in ("default", "fixed", "lazy"):
        rows = [r.arms[arm] for r in results]
        total = np.array([x.total_extra_s for x in rows])
        d = {
            "ready_s": rows[0].ready_s,
            "startup_wait_s_mean": float(np.mean([x.startup_wait_s for x in rows])),
            "eager_prefill_penalty_s_mean": float(
                np.mean([x.eager_prefill_penalty_s for x in rows])
            ),
            "eager_decode_penalty_s_mean": float(
                np.mean([x.eager_decode_penalty_s for x in rows])
            ),
            "eager_prefill_share": float(
                np.sum([x.eager_prefill_requests for x in rows])
                / max(1, int(n_req.sum()))
            ),
            "interference_s_mean": float(np.mean([x.interference_s for x in rows])),
            "delayed_request_share": float(
                np.sum([x.delayed_requests for x in rows]) / max(1, int(n_req.sum()))
            ),
            "max_delay_s": float(np.max([x.max_delay_s for x in rows])),
            "total_extra_s_mean": float(total.mean()),
            "total_extra_s_p50": float(np.median(total)),
        }
        if arm == "lazy":
            done = [x.all_captured_after_s for x in rows if x.all_captured_after_s is not None]
            d["launches_fully_captured_share"] = len(done) / len(rows)
            d["all_captured_after_s_p50"] = float(np.median(done)) if done else None
            d["all_captured_after_s_p90"] = (
                float(np.quantile(done, 0.9)) if done else None
            )
        out[arm] = d
    return out


def adaptive_plan(
    traffic: ModelTraffic,
    profile: EngineProfile,
    cfg: SimConfig,
    *,
    make_plan_fn,
    first_horizon_s: float = 60.0,
):
    """Plan whose threshold uses this model's own deferral horizon and waiters.

    Pass 1 plans with a nominal horizon and simulates it. H is then the
    request-weighted mean time until all deferred buckets were captured (the
    simulation horizon for launches that never finished): deferral cost is
    rate x time, and busy launches both carry more traffic and wait longer, so
    an unweighted median would understate it. lam is the mean number of
    requests that arrived while the default arm was still capturing. Pass 2
    re-plans with (lam, H).
    """
    plan0 = make_plan_fn(traffic, profile, lam=1.0, horizon_s=first_horizon_s)
    results = simulate_model(traffic, plan0, profile, cfg)
    if not results:
        return plan0, {"lam": 1.0, "horizon_s": first_horizon_s, "launches": 0}
    done = [
        r.arms["lazy"].all_captured_after_s
        if r.arms["lazy"].all_captured_after_s is not None
        else cfg.horizon_s
        for r in results
    ]
    weights = np.array([max(r.requests, 1) for r in results], dtype=float)
    horizon = (
        float(np.average(done, weights=weights)) if plan0.lazy else first_horizon_s
    )
    waiters = []
    for r in results:
        sel = (traffic.t0 >= r.launch_t) & (
            traffic.t0 < r.launch_t + plan0.default_capture_s
        )
        waiters.append(int(sel.sum()))
    lam = max(1.0, float(np.mean(waiters)))
    plan1 = make_plan_fn(traffic, profile, lam=lam, horizon_s=max(horizon, 1.0))
    return plan1, {"lam": lam, "horizon_s": horizon, "launches": len(results)}
