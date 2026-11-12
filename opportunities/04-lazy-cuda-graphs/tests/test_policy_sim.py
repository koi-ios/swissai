"""Tests for the trace-driven policy tool and simulator (lazygraph/)."""

from __future__ import annotations

import itertools
import json
import random
import sys
import types

import numpy as np
import pytest

from conftest import load_patch_module
from lazygraph.buckets import bucket_of, decode_buckets, prefill_buckets
from lazygraph.costs import CaptureCost, EagerPenalty, EngineProfile, load_profile
from lazygraph.policy import BucketValue, make_plan, rank_buckets
from lazygraph.simulate import (
    SimConfig,
    _idle_after,
    _place_steps,
    aggregate,
    deferred_steps,
    simulate_launch,
    simulate_model,
)
from lazygraph.trace import ModelTraffic, inflight_histogram, merge_intervals, sessions


def test_bucket_lists_match_sglang_defaults():
    p = prefill_buckets(8192)
    assert len(p) == 58 and p[0] == 4 and p[-1] == 8192  # 58 sizes in the lab logs
    d = decode_buckets(256)
    assert len(d) == 36 and d[:6] == [1, 2, 4, 8, 12, 16]  # 36 sizes in the lab logs
    assert decode_buckets(512)[-1] == 512
    assert bucket_of(5, d) == 8 and bucket_of(257, d) is None


def test_merge_intervals_and_inflight():
    t0 = np.array([0.0, 1.0, 5.0, 5.5])
    t1 = np.array([2.0, 3.0, 6.0, 7.0])
    assert merge_intervals(t0, t1).tolist() == [[0.0, 3.0], [5.0, 7.0]]
    secs = inflight_histogram(t0, t1)
    # level 1: [0,1),[2,3),[5,5.5),[6,7) = 3.5 s; level 2: [1,2),[5.5,6) = 1.5 s
    assert secs[1] == pytest.approx(3.5) and secs[2] == pytest.approx(1.5)
    busy = merge_intervals(t0, t1)
    assert sessions(busy, min_gap_s=1.5) == [(0.0, float("inf")), (5.0, 2.0)]


def toy_profile(**kw) -> EngineProfile:
    base = dict(
        name="toy",
        description="toy",
        one_off_s=0.0,
        capture={"prefill": CaptureCost(0.5, 0.0), "decode": CaptureCost(0.2, 0.0)},
        penalty={
            "prefill": EagerPenalty(0.01, 0.0, 0.0, 0.0),
            "decode": EagerPenalty(0.005, 0.0, 0.0, 0.0),
        },
        decode_steps_per_busy_s=100.0,
        capture_source="toy",
    )
    base.update(kw)
    return EngineProfile(**base)


def toy_traffic(n=200, gap=30.0, dur=2.0, tin=300, seed=0) -> ModelTraffic:
    rng = np.random.default_rng(seed)
    t0 = np.cumsum(rng.exponential(gap, n))
    t1 = t0 + dur
    return ModelTraffic(
        model="toy",
        t0=t0,
        t1=t1,
        tin=np.full(n, tin),
        tout=np.full(n, 100),
    )


def test_rank_buckets_puts_used_buckets_first():
    values = rank_buckets(toy_traffic(), toy_profile())
    top = values[0]
    assert top.phase == "decode" and top.size == 1  # one request in flight at a time
    used_prefill = [v for v in values if v.phase == "prefill" and v.rate_per_s > 0]
    assert [v.size for v in used_prefill] == [bucket_of(300, prefill_buckets())]
    evs = [v.ev for v in values]
    assert evs == sorted(evs, reverse=True)


def test_threshold_extremes():
    t = toy_traffic()
    lazy_everything = make_plan(t, toy_profile(), lam=1e9, horizon_s=1.0)
    assert lazy_everything.initial == {"decode": [], "prefill": []}
    eager_nothing = make_plan(t, toy_profile(), lam=1e-12, horizon_s=1e12)
    used = {v.size for v in rank_buckets(t, toy_profile()) if v.rate_per_s > 0}
    assert set(eager_nothing.initial["decode"]) | set(eager_nothing.initial["prefill"]) >= used


def test_ev_order_is_optimal_for_weighted_completion_time():
    """Smith's rule: decreasing w/c minimises sum w_j C_j on one machine."""
    rng = random.Random(3)
    for _ in range(50):
        jobs = [
            BucketValue("decode", i, rng.uniform(0, 2), rng.uniform(0.001, 0.01),
                        rng.uniform(0.1, 2.0))
            for i in range(6)
        ]

        def cost(order):
            t, total = 0.0, 0.0
            for j in order:
                t += j.cost_s
                total += j.weight * t
            return total

        best = min(cost(p) for p in itertools.permutations(jobs))
        smith = cost(sorted(jobs, key=lambda j: -j.ev))
        assert smith == pytest.approx(best, rel=1e-9)


def test_cuda_graph_config_is_accepted_by_patched_parser():
    """The tool's JSON parses with the patched SGLang 0.5.20 config parser."""
    stub = types.ModuleType("sglang.srt.runtime_context")
    stub.get_exec = lambda: (_ for _ in ()).throw(ValueError("not published"))
    sys.modules.setdefault("sglang.srt.runtime_context", stub)
    # default_prefill_backend() lazily imports sglang.srt.utils.is_cuda.
    utils_stub = types.ModuleType("sglang.srt.utils")
    utils_stub.is_cuda = lambda: True
    sys.modules.setdefault("sglang.srt.utils", utils_stub)
    cgc = load_patch_module("sglang.srt.model_executor.cuda_graph_config")
    plan = make_plan(toy_traffic(), toy_profile(), lam=1.0, horizon_s=60.0)
    assert plan.lazy
    raw = json.dumps(plan.cuda_graph_config())
    parsed = cgc.parse_cuda_graph_config_arg(raw)
    cfg = cgc.CudaGraphConfig.from_dict(parsed)
    assert cfg.decode.initial_bs == plan.initial["decode"]
    assert cfg.prefill.deferred_bs == plan.deferred["prefill"]
    assert len(cfg.prefill.deferred_priority) == len(cfg.prefill.deferred_bs)
    # Every bucket is either at startup or deferred, never both.
    for phase in ("decode", "prefill"):
        a, b = set(plan.initial[phase]), set(plan.deferred[phase])
        assert not a & b and a | b == set(plan.planned[phase])


def test_unknown_key_still_rejected_by_patched_parser():
    cgc = load_patch_module("sglang.srt.model_executor.cuda_graph_config")
    import argparse

    with pytest.raises(argparse.ArgumentTypeError):
        cgc.parse_cuda_graph_config_arg('{"decode": {"lazy": true}}')


def test_place_steps_respects_hysteresis_and_interruptions():
    idle = np.array([[0.0, 0.5], [2.0, 4.0], [10.0, 20.0]])
    steps = [("decode", 8, 1.0), ("decode", 16, 1.0), ("prefill", 512, 3.0)]
    cfg = SimConfig(min_idle_s=1.0, idle_cost_ratio=0.0)
    placed = _place_steps(idle, steps, cfg)
    # [0,0.5) too short for the 1 s hysteresis; [2,4): step at 3.0 ends 4.0 (no cut);
    # next would start 4.0 >= 4.0 -> next period; [10,20): starts 11.0 then 12.0.
    assert [(p[1], p[2], p[3], p[4]) for p in placed] == [
        (8, 3.0, 4.0, None),
        (16, 11.0, 12.0, None),
        (512, 12.0, 15.0, None),
    ]
    cut = _place_steps(np.array([[0.0, 2.5]]), [("prefill", 512, 3.0)], cfg)
    assert cut[0][4] == 2.5  # interrupted by the request at 2.5


def test_idle_after_ready():
    busy = np.array([[0.0, 1.0], [3.0, 5.0]])
    assert _idle_after(busy, ready=2.0, horizon=10.0).tolist() == [[2.0, 3.0], [5.0, 10.0]]
    assert _idle_after(busy, ready=4.0, horizon=10.0).tolist() == [[5.0, 10.0]]


def test_simulation_arms_on_synthetic_traffic():
    t = toy_traffic(n=300, gap=20.0)
    prof = toy_profile()
    plan = make_plan(t, prof, lam=1.0, horizon_s=60.0)
    cfg = SimConfig(min_idle_s=1.0, horizon_s=600.0, session_gap_s=60.0)
    results = simulate_model(t, plan, prof, cfg)
    agg = aggregate(results)
    assert agg["launches"] >= 1
    assert agg["default"]["eager_prefill_penalty_s_mean"] == 0.0
    assert agg["default"]["interference_s_mean"] == 0.0
    assert agg["lazy"]["ready_s"] < agg["default"]["ready_s"]
    # Deferral can only reduce eager exposure relative to the fixed arm.
    assert agg["lazy"]["eager_prefill_share"] <= agg["fixed"]["eager_prefill_share"]
    # Max delay is bounded by the largest single capture step.
    max_step = max(c for _, _, c in deferred_steps(plan, prof))
    assert agg["lazy"]["max_delay_s"] <= max_step + 1e-9


def test_request_arriving_during_step_waits_for_its_end():
    t = ModelTraffic(
        model="toy",
        t0=np.array([0.0, 3.5]),
        t1=np.array([0.5, 4.0]),
        tin=np.array([300, 300]),
        tout=np.array([10, 10]),
    )
    prof = toy_profile(
        capture={"prefill": CaptureCost(2.0, 0.0), "decode": CaptureCost(0.2, 0.0)}
    )
    plan = make_plan(t, prof, lam=1e9, horizon_s=1.0)  # defer everything
    res = simulate_launch(t, 0.0, float("inf"), plan, prof,
                          SimConfig(min_idle_s=1.0, horizon_s=60.0))
    lazy = res.arms["lazy"]
    # Idle from 0.5; steps start at 1.5. The request at 3.5 lands inside a step.
    assert lazy.delayed_requests == 1
    assert 0.0 < lazy.max_delay_s <= 2.0


def test_measured_penalty_overrides_prediction(tmp_path):
    f = tmp_path / "pen.json"
    f.write_text(json.dumps({"decode": {"1": 0.004, "64": 0.002}, "prefill": {}}))
    prof = load_profile("llama8b_tp1_h100", str(f))
    assert prof.eager_penalty("decode", 1) == pytest.approx(0.004)
    assert prof.eager_penalty("decode", 32) == pytest.approx(0.004 - 0.002 * 31 / 63)
    assert prof.eager_penalty("prefill", 512) == pytest.approx(
        prof.penalty["prefill"](512)
    )
