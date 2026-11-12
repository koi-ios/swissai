"""Multi-rank tests for LazyGraphCaptureController (patch module, GPU-free).

Each fake TP rank runs in its own thread for the step that contains the
collective (run_pending), and the MIN all-reduce is a two-phase barrier, so a
rank that skipped or added a collective would deadlock and the barrier timeout
fails the test. The scheduler loop around it mirrors the patched
Scheduler.ingest_requests / on_idle: begin_iteration -> leader propose ->
broadcast -> take_ticks -> (no batch) -> run_pending.
"""

from __future__ import annotations

import threading
from typing import List, Optional

import pytest

from conftest import load_patch_module

lc = load_patch_module("sglang.srt.model_executor.runner.lazy_capture")
lgc = load_patch_module("sglang.srt.managers.scheduler_components.lazy_graph_capture")

DECODE = [1, 2, 4, 8, 12, 16, 24, 32]
PREFILL = [64, 128, 256, 512, 1024]


class FakeClock:
    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now


class CollectiveGroup:
    """Element-wise MIN all-reduce across N threads."""

    def __init__(self, n: int, timeout: float = 5.0) -> None:
        self._contrib: List[Optional[List[int]]] = [None] * n
        self._result: List[int] = []
        self._timeout = timeout
        self._b1 = threading.Barrier(n, action=self._reduce)
        self._b2 = threading.Barrier(n)
        self.calls = 0

    def _reduce(self) -> None:
        self.calls += 1
        self._result = [min(col) for col in zip(*self._contrib)]

    def agree_min_for(self, rank: int):
        def agree_min(flags: List[int]) -> List[int]:
            self._contrib[rank] = list(flags)
            self._b1.wait(timeout=self._timeout)
            out = list(self._result)
            self._b2.wait(timeout=self._timeout)
            return out

        return agree_min


class FakeRunner:
    def __init__(self, schedule, *, headroom: bool = True) -> None:
        self.lazy_schedule = schedule
        self.headroom = headroom
        self.captured_log: List[int] = []

    def capture_deferred(self, size: int) -> None:
        self.captured_log.append(size)
        self.lazy_schedule.mark_captured(size)

    def lazy_has_headroom(self, size: int) -> bool:
        return self.headroom


def make_schedules(*, decode_order=None, promote_after=0):
    dec = lc.LazyCaptureSchedule(
        phase="decode",
        planned=DECODE,
        initial=[1, 2, 4],
        deferred_order=decode_order,
        promote_after=promote_after,
    )
    pre = lc.LazyCaptureSchedule(
        phase="prefill",
        planned=PREFILL,
        initial=[256],
        deferred_order=[512, 128, 64, 1024],
        deferred_priority=[0.5, 0.4, 0.3, 0.2],
    )
    for s in (dec, pre):
        for size in s.startup_sizes():
            s.mark_captured(size)
    return dec, pre


class Rank:
    def __init__(self, rank, group, clock, *, min_idle_s=1.0, idle_cost_ratio=0.0,
                 decode_order=None, headroom=True, promote_after=0):
        dec, pre = make_schedules(decode_order=decode_order, promote_after=promote_after)
        self.decode = FakeRunner(dec, headroom=headroom)
        self.prefill = FakeRunner(pre, headroom=headroom)
        self.ctrl = lgc.LazyGraphCaptureController(
            is_leader=rank == 0,
            targets={"decode": self.decode, "prefill": self.prefill},
            agree_min=group.agree_min_for(rank),
            min_idle_s=min_idle_s,
            idle_cost_ratio=idle_cost_ratio,
            clock=clock,
        )

    def state(self):
        return (
            self.decode.lazy_schedule.captured,
            self.decode.lazy_schedule.pending,
            self.prefill.lazy_schedule.captured,
            self.prefill.lazy_schedule.pending,
            self.ctrl.seq,
            self.ctrl.disabled,
        )


def run_on_all(ranks, fn):
    """Run fn(rank) concurrently on every rank; re-raise the first error."""
    errors = []

    def wrap(r):
        try:
            fn(r)
        except BaseException as e:  # noqa: BLE001 - surfaced below
            errors.append(e)

    threads = [threading.Thread(target=wrap, args=(r,)) for r in ranks]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=10)
    assert not any(t.is_alive() for t in threads), "deadlock: a rank never returned"
    if errors:
        raise errors[0]


def iteration(ranks, clock, *, new_requests=(), fully_idle=True):
    """One scheduler iteration on every rank; returns the tick that ran, if any."""
    leader = ranks[0]
    for r in ranks:
        r.ctrl.begin_iteration()
    tick = leader.ctrl.propose(fully_idle=fully_idle)
    broadcast = list(new_requests) + ([tick] if tick is not None else [])
    remaining = [r.ctrl.take_ticks(list(broadcast)) for r in ranks]
    assert all(rem == list(new_requests) for rem in remaining)
    if new_requests or not fully_idle:
        return None  # a batch is scheduled; on_idle is not reached on any rank
    ran = []
    run_on_all(ranks, lambda r: ran.append(r.ctrl.run_pending(fully_idle=True)))
    assert len({repr(t) for t in ran}) == 1, "ranks ran different steps"
    return ran[0]


def make_ranks(n=4, **kw):
    group = CollectiveGroup(n)
    clock = FakeClock()
    return [Rank(i, group, clock, **kw) for i in range(n)], clock, group


def test_hysteresis_then_lockstep_capture_of_every_bucket():
    ranks, clock, group = make_ranks(min_idle_s=1.0)
    steps = []
    for _ in range(200):
        tick = iteration(ranks, clock)
        if tick is not None:
            steps.append((round(clock.now, 2), tick.phase, tick.size))
        clock.now += 0.1
        if not ranks[0].ctrl.has_pending_work():
            break
    assert steps[0][0] >= 1.0  # nothing before the idle hysteresis
    states = [r.state() for r in ranks]
    assert all(s == states[0] for s in states)
    dec_cap, dec_pend, pre_cap, pre_pend, seq, disabled = states[0]
    assert dec_cap == tuple(DECODE) and pre_cap == tuple(PREFILL)
    assert dec_pend == () and pre_pend == () and not disabled
    assert seq == len(steps) == (len(DECODE) - 3) + (len(PREFILL) - 1)
    # Decode has default (count-based) priorities above the prefill EV scores.
    assert [p for _, p, _ in steps[:5]] == ["decode"] * 5
    assert [s for _, p, s in steps if p == "prefill"] == [512, 128, 64, 1024]


def test_busy_iterations_reset_the_idle_timer():
    ranks, clock, _ = make_ranks(min_idle_s=1.0)
    for _ in range(9):
        assert iteration(ranks, clock) is None
        clock.now += 0.1
    # A request arrives just before the threshold: no capture, timer resets.
    assert iteration(ranks, clock, new_requests=["req"]) is None
    clock.now += 0.1
    assert iteration(ranks, clock, fully_idle=False) is None
    clock.now += 0.1
    t_idle = clock.now
    tick = None
    while tick is None:
        tick = iteration(ranks, clock)
        clock.now += 0.1
    assert clock.now - 0.1 - t_idle >= 1.0 - 1e-9


def test_tick_in_same_iteration_as_request_is_dropped_everywhere():
    ranks, clock, group = make_ranks(min_idle_s=0.0)
    calls_before = group.calls
    # Leader proposes (idle at poll time) but a request is in the same broadcast.
    assert iteration(ranks, clock, new_requests=["req"]) is None
    assert group.calls == calls_before  # no collective was entered
    assert all(r.ctrl.seq == 0 for r in ranks)
    # Next iteration: the stale tick is gone; a fresh one is proposed and run.
    tick = iteration(ranks, clock)
    assert tick is not None and tick.seq == 0
    assert all(r.ctrl.seq == 1 for r in ranks)


def test_idle_cost_ratio_defers_expensive_buckets():
    ranks, clock, _ = make_ranks(min_idle_s=0.0, idle_cost_ratio=10.0)
    ctrl = ranks[0].ctrl
    for r in ranks:  # same observations everywhere is not required (leader-only)
        r.ctrl.observe_step(phase="decode", size=8, seconds=0.2)
    need = 10.0 * ctrl.estimate_cost(phase="decode", size=8)
    first = None
    while first is None:
        first = iteration(ranks, clock)
        clock.now += 0.05
    assert clock.now - 0.05 >= need - 1e-9


def test_request_waits_at_most_one_step():
    """Arrivals during a step are polled in the next iteration."""
    ranks, clock, _ = make_ranks(min_idle_s=0.5)
    step_cost = {"decode": 0.3, "prefill": 1.2}
    arrivals = [0.2, 0.9, 1.35, 2.0, 2.05, 4.4, 7.77]
    served = {}
    pending = sorted(arrivals)
    busy_until = 0.0
    dt = 0.01
    while clock.now < 12.0:
        arrived = [a for a in pending if a <= clock.now]
        pending = [a for a in pending if a > clock.now]
        for a in arrived:
            served[a] = clock.now
        busy = clock.now < busy_until
        if arrived:
            busy_until = clock.now + 0.25
        tick = iteration(
            ranks, clock, new_requests=["r"] * len(arrived), fully_idle=not busy
        )
        clock.now += step_cost[tick.phase] if tick is not None else dt
    max_step = max(step_cost.values())
    delays = [served[a] - a for a in arrivals]
    assert max(delays) <= max_step + dt + 1e-9
    assert any(d > dt for d in delays)  # some request did wait for a step


def test_memory_headroom_failure_skips_bucket_on_every_rank():
    group = CollectiveGroup(3)
    clock = FakeClock()
    ranks = [Rank(0, group, clock, min_idle_s=0.0),
             Rank(1, group, clock, min_idle_s=0.0, headroom=False),
             Rank(2, group, clock, min_idle_s=0.0)]
    assert iteration(ranks, clock) is None
    for r in ranks:
        assert r.decode.lazy_schedule.failed == (8,)
        assert r.ctrl.seq == 1 and r.decode.captured_log == []
    # Queue keeps moving to the next bucket, which also fails on rank 1 ...
    assert iteration(ranks, clock) is None
    assert all(r.decode.lazy_schedule.failed == (8, 12) for r in ranks)
    assert len({r.state() for r in ranks}) == 1


def test_diverged_rank_disables_lazy_capture_without_deadlock():
    group = CollectiveGroup(3)
    clock = FakeClock()
    ranks = [Rank(0, group, clock, min_idle_s=0.0),
             Rank(1, group, clock, min_idle_s=0.0),
             Rank(2, group, clock, min_idle_s=0.0, decode_order=[32, 8])]
    assert iteration(ranks, clock) is None
    assert all(r.ctrl.disabled for r in ranks)
    assert all(r.decode.captured_log == [] for r in ranks)
    assert ranks[0].ctrl.propose(fully_idle=True) is None


def test_promotion_is_rank_consistent_and_captured_first():
    ranks, clock, _ = make_ranks(min_idle_s=0.0, promote_after=3)
    # The same eager decode batches run on every rank (replicated scheduler).
    for r in ranks:
        sched = r.decode.lazy_schedule
        for raw in (20, 21, 22):  # bucket 24
            assert not sched.can_replay(raw)
            sched.record_eager(raw)
    tick = iteration(ranks, clock)
    assert (tick.phase, tick.size) == ("decode", 24)
    assert all(r.decode.captured_log == [24] for r in ranks)


def test_cost_model_fit():
    fit = lgc._fit_cost_model
    obs = [(s, 0.2 + 1e-4 * s) for s in (64, 512, 2048, 8192)]
    a, b = fit(obs, (9.0, 9.0))
    assert a == pytest.approx(0.2, abs=1e-9) and b == pytest.approx(1e-4, rel=1e-6)
    assert fit([], (0.3, 0.0)) == (0.3, 0.0)
    a1, b1 = fit([(1024, 0.5)], (0.4, 1.4e-4))
    assert b1 == 1.4e-4 and a1 == pytest.approx(0.5 - 1.4e-4 * 1024)


def test_tick_survives_pickle_broadcast():
    import pickle

    tick = lgc.LazyGraphCaptureTick(seq=3, phase="prefill", size=512)
    assert pickle.loads(pickle.dumps(tick)) == tick
