"""Unit tests for LazyCaptureSchedule (patch module, GPU-free)."""

from __future__ import annotations

import pytest

from conftest import load_patch_module

lc = load_patch_module("sglang.srt.model_executor.runner.lazy_capture")
LazyCaptureSchedule = lc.LazyCaptureSchedule
pick_next_phase = lc.pick_next_phase

# SGLang 0.5.20 default decode list for max_bs 64 (cuda_graph_hook.py:577-609).
DECODE = [1, 2, 4, 8, 12, 16, 24, 32, 40, 48, 56, 64]


def make(**kw):
    args = dict(phase="decode", planned=DECODE, initial=[1, 2, 4, 8])
    args.update(kw)
    return LazyCaptureSchedule(**args)


def capture_initial(s):
    for size in s.startup_sizes():
        s.mark_captured(size)
    return s


def test_default_order_is_ascending_and_startup_largest_first():
    s = make()
    assert s.pending == (12, 16, 24, 32, 40, 48, 56, 64)
    assert s.startup_sizes() == [8, 4, 2, 1]
    assert s.captured == ()


def test_initial_must_be_planned():
    with pytest.raises(ValueError):
        make(initial=[1, 3])


def test_deferred_order_validation():
    with pytest.raises(ValueError):
        make(deferred_order=[2])  # already initial
    with pytest.raises(ValueError):
        make(deferred_order=[12, 12])  # duplicate
    with pytest.raises(ValueError):
        make(deferred_order=[13])  # not planned


def test_policy_order_kept_and_missing_buckets_appended():
    s = make(deferred_order=[64, 16])
    assert s.pending[:2] == (64, 16)
    assert s.pending[2:] == (12, 24, 32, 40, 48, 56)


def test_priorities_default_and_tail():
    s = make(deferred_order=[24, 12], deferred_priority=[5.0, 3.0])
    assert s.next_pending() == (24, 5.0)
    s.mark_captured(24)
    assert s.next_pending() == (12, 3.0)
    s.mark_captured(12)
    head, prio = s.next_pending()
    assert head == 16 and prio < 3.0  # appended tail ranks below policy entries


def test_planned_bucket_and_bucket_exact_dispatch():
    s = capture_initial(make())
    assert s.planned_bucket(1) == 1
    assert s.planned_bucket(5) == 8
    assert s.planned_bucket(9) == 12
    assert s.planned_bucket(65) is None
    assert s.can_replay(5)  # bucket 8 captured
    # bucket 12 is planned but not captured: no padding to a larger captured
    # bucket (there is none) and never to 16 even if 16 were captured.
    assert not s.can_replay(9)
    s.mark_captured(16)
    assert not s.can_replay(9)  # still 12's batches, 12 not captured
    assert s.can_replay(13)  # bucket 16
    assert not s.can_replay(100)


def test_record_eager_counts_only_pending_buckets():
    s = capture_initial(make(promote_after=0))
    assert s.record_eager(3) is None  # bucket 4 captured -> not counted
    assert s.eager_hits(4) == 0
    s.record_eager(9)
    s.record_eager(11)
    assert s.eager_hits(12) == 2
    assert s.record_eager(1000) is None  # above max: not a planned bucket


def test_promotion_moves_hot_bucket_to_head_fifo():
    s = capture_initial(make(promote_after=3))
    assert s.next_pending()[0] == 12
    for _ in range(2):
        assert s.record_eager(50) is None  # bucket 56
    assert s.record_eager(50) == 56
    assert s.next_pending() == (56, float("inf"))
    for _ in range(3):
        s.record_eager(30)  # bucket 32
    assert s.pending[:2] == (56, 32)  # first promoted stays first
    s.mark_captured(56)
    assert s.next_pending()[0] == 32
    s.mark_captured(32)
    assert s.next_pending()[0] == 12  # back to policy order


def test_mark_failed_drops_bucket_and_stops_counting():
    s = capture_initial(make(promote_after=2))
    s.mark_failed(12)
    assert 12 not in s.pending and s.failed == (12,)
    assert s.record_eager(10) is None
    assert not s.can_replay(10)


def test_mark_captured_rejects_unplanned():
    s = make()
    with pytest.raises(ValueError):
        s.mark_captured(13)


def test_pick_next_phase_merges_by_priority():
    dec = capture_initial(make(deferred_order=[12, 16], deferred_priority=[2.0, 0.5]))
    pre = LazyCaptureSchedule(
        phase="prefill",
        planned=[64, 128, 256, 512],
        initial=[64],
        deferred_order=[256, 128],
        deferred_priority=[1.0, 0.1],
    )
    order = []
    while True:
        sched = pick_next_phase([dec, pre])
        if sched is None:
            break
        size = sched.next_pending()[0]
        order.append((sched.phase, size))
        sched.mark_captured(size)
    assert order[:4] == [
        ("decode", 12),
        ("prefill", 256),
        ("decode", 16),
        ("prefill", 128),
    ]
    assert pick_next_phase([None, None]) is None


def test_identical_inputs_give_identical_state_on_every_rank():
    """Rank consistency: same call sequence -> same queue on every 'rank'."""
    import random

    rng = random.Random(7)
    events = [rng.choice(["eager", "eager", "capture"]) for _ in range(200)]
    sizes = [rng.randint(1, 64) for _ in range(200)]
    snapshots = []
    for _rank in range(4):
        s = capture_initial(make(promote_after=3))
        for ev, raw in zip(events, sizes):
            if ev == "eager":
                if not s.can_replay(raw):
                    s.record_eager(raw)
            elif s.next_pending() is not None:
                s.mark_captured(s.next_pending()[0])
        snapshots.append((s.captured, s.pending, s.failed))
    assert all(snap == snapshots[0] for snap in snapshots)
