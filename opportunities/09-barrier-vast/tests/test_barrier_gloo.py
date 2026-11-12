"""Tests for repro/barrier_gloo.py: pin down gloo monitored_barrier semantics that
explain how model-launch-images#22 can be read from its logs.

Run: .venv/bin/python -m pytest tests/test_barrier_gloo.py -q   (~40 s, CPU only)
"""
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "repro"))
import barrier_gloo as bg  # noqa: E402

W, T = 4, 0.5  # 4 ranks; 0.5 s stands for SGLang's 480 s


def run(name):
    return bg.run_scenario(bg.build_scenarios(W, T)[name])


def test_parse_failed_ranks():
    assert bg._parse_failed("[Rank 0]: Ranks 1, 2, 31 failed to pass monitoredBarrier in 480000 ms") == [1, 2, 31]
    assert bg._parse_failed("[Rank 0]: Ranks 3 failed to pass monitoredBarrier in 1000 ms") == [3]
    assert bg._parse_failed("something else") == []


def test_env_knob(monkeypatch):
    monkeypatch.delenv(bg.ENV_KNOB, raising=False)
    assert bg.barrier_timeout_s(480) == 480
    monkeypatch.setenv(bg.ENV_KNOB, "3600")
    assert bg.barrier_timeout_s(480) == 3600


def test_all_on_time_passes():
    assert run("ok").rank0_outcome == "passed"


def test_one_rank_late_by_more_than_timeout_fails_after_one_timeout():
    r = run("one_slow")
    assert r.rank0_outcome == "raised"
    assert r.listed_failed_ranks == [W - 1]
    assert 0.8 * T <= r.rank0_wait_s <= 1.6 * T


def test_all_moderately_late_lists_only_first_rank():
    # every other rank is 1.5x the timeout late: only rank 1 misses its window,
    # the rest arrive during rank 2's window. So "all 31 ranks listed" cannot
    # come from a moderate spread.
    r = run("all_slow")
    assert r.rank0_outcome == "raised"
    assert r.listed_failed_ranks == [1]


def test_total_spread_above_timeout_can_still_pass():
    # windows are per rank and sequential, so what must stay under the timeout
    # is each successive gap, not the total spread.
    r = run("staggered")
    assert r.rank0_outcome == "passed"
    assert r.rank0_wait_s > 1.2 * T


def test_dead_ranks_cost_a_full_timeout_each():
    r = run("dead")
    assert r.rank0_outcome == "raised"
    assert r.listed_failed_ranks == list(range(1, W))
    assert r.rank0_wait_s >= 0.9 * (W - 1) * T


def test_dist_timeout_does_not_change_the_barrier():
    r = run("dist_timeout")
    assert r.rank0_outcome == "raised"
    assert r.listed_failed_ranks == [1]


def test_env_knob_makes_the_slow_load_pass():
    assert run("fixed").rank0_outcome == "passed"
