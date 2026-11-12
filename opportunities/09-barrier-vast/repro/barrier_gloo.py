"""CPU-only reproduction of SGLang's post-load barrier (model-launch-images#22).

SGLang (0.5.10.post1 through main, Nov 2026) ends model loading with

    dist.monitored_barrier(group=get_tp_group().cpu_group,
                           timeout=timedelta(seconds=480), wait_all_ranks=True)

(sglang/srt/model_executor/model_runner_components/load_model_utils.py:51,367-387
in v0.5.20). This script spawns N CPU processes on gloo, builds the same kind of
group SGLang builds (a gloo subgroup with a 120 min default timeout), lets each
rank "load" for a configurable time (or die, or hang), and runs a faithful copy
of that barrier with a scaled-down timeout (default 2 s stands for 480 s).

It answers three questions the issue cannot answer from the error text alone:
  1. How long does rank 0 wait, and which ranks does it list, when the other
     ranks are slow vs. dead vs. hung?
  2. Does --dist-timeout (the process-group timeout) change the barrier? (No.)
  3. Does an environment knob for the barrier timeout make a slow load pass?

Usage:
  python repro/barrier_gloo.py --scenario all --world 8 --timeout 2
  python repro/barrier_gloo.py --scenario all_slow --world 4 --timeout 1 --json out.json
"""

from __future__ import annotations

import argparse
import datetime
import json
import multiprocessing as mp
import os
import re
import socket
import sys
import time
from dataclasses import asdict, dataclass, field
from typing import Dict, List, Optional

# The knob proposed upstream (see patches/). The model-launch-factory already
# sets this name, but no SGLang release reads it today.
ENV_KNOB = "SGLANG_UNBALANCED_MODEL_LOADING_TIMEOUT_S"


def barrier_timeout_s(default_s: float) -> float:
    """Proposed behaviour: the env var overrides the hard-coded constant."""
    raw = os.environ.get(ENV_KNOB)
    return float(raw) if raw else default_s


def dist_barrier_after_load(*, group, tp_rank: int, timeout_s: float) -> None:
    """Copy of SGLang v0.5.20 dist_barrier_after_load (non-mooncake branch).

    The only difference: SGLang raises ``from None``; we chain the gloo error so
    the repro can record which ranks gloo reported.
    """
    import torch.distributed as dist

    try:
        dist.monitored_barrier(
            group=group,
            timeout=datetime.timedelta(seconds=timeout_s),
            wait_all_ranks=True,
        )
    except RuntimeError as e:
        raise ValueError(
            f"TP rank {tp_rank} could finish the model loading, but there are other "
            "ranks that didn't finish loading. It is likely due to unexpected "
            "failures (e.g., OOM) or a slow node."
        ) from e


@dataclass
class Behaviour:
    kind: str  # "load" | "die" | "hang"
    seconds: float = 0.0  # load time, or time before dying


@dataclass
class Scenario:
    name: str
    world: int
    timeout_s: float  # the barrier timeout (stands for 480 s)
    behaviours: List[Behaviour]
    pg_timeout_s: float = 3600.0  # --dist-timeout analogue (init_process_group)
    group_timeout_s: float = 7200.0  # SGLang's gloo cpu_group: 120 min
    env_timeout_s: Optional[float] = None  # value for the proposed env knob
    description: str = ""


def build_scenarios(world: int, t: float) -> Dict[str, Scenario]:
    base = 0.2  # every rank "loads" at least this long
    fast = Behaviour("load", base)

    def others(b: Behaviour) -> List[Behaviour]:
        return [fast] + [b] * (world - 1)

    return {
        "ok": Scenario("ok", world, t, [fast] * world,
                       description="all ranks finish together"),
        "under": Scenario("under", world, t, [fast] * (world - 1) + [Behaviour("load", base + 0.6 * t)],
                          description="last rank 0.6x timeout behind: passes"),
        "one_slow": Scenario("one_slow", world, t, [fast] * (world - 1) + [Behaviour("load", base + 1.5 * t)],
                             description="last rank 1.5x timeout behind"),
        "all_slow": Scenario("all_slow", world, t, others(Behaviour("load", base + 1.5 * t)),
                             description="ranks 1..N-1 all 1.5x timeout behind rank 0"),
        "staggered": Scenario("staggered", world, t,
                              [Behaviour("load", base + 0.6 * t * k) for k in range(world)],
                              description="rank k finishes 0.6k x timeout late: total spread > timeout"),
        "all_slow_long": Scenario("all_slow_long", world, t, others(Behaviour("load", base + (world + 1) * t)),
                                  description="ranks 1..N-1 behind by more than (N-1) x timeout"),
        "dead": Scenario("dead", world, t, others(Behaviour("die", base)),
                         description="ranks 1..N-1 exit (e.g. OOM-killed) before the barrier"),
        "hang": Scenario("hang", world, t, others(Behaviour("hang")),
                         description="ranks 1..N-1 alive but never reach the barrier"),
        "dist_timeout": Scenario("dist_timeout", world, t, others(Behaviour("load", base + 1.5 * t)),
                                 pg_timeout_s=10 * t, group_timeout_s=10 * t,
                                 description="all_slow, with a 10x larger process-group timeout (--dist-timeout)"),
        "fixed": Scenario("fixed", world, t, others(Behaviour("load", base + 1.5 * t)),
                          env_timeout_s=3 * t,
                          description="all_slow, with the proposed env knob set to 3x timeout"),
    }


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _worker(rank: int, sc: Scenario, port: int, epoch: float, q) -> None:
    os.environ.setdefault("GLOO_SOCKET_IFNAME", "lo0" if sys.platform == "darwin" else "lo")
    if sc.env_timeout_s is not None:
        os.environ[ENV_KNOB] = str(sc.env_timeout_s)
    import torch.distributed as dist

    dist.init_process_group(
        "gloo",
        init_method=f"tcp://127.0.0.1:{port}",
        rank=rank,
        world_size=sc.world,
        timeout=datetime.timedelta(seconds=sc.pg_timeout_s),
    )
    # SGLang's GroupCoordinator: cpu_group = new_group(ranks, backend="gloo", timeout=120 min)
    group = dist.new_group(
        list(range(sc.world)), backend="gloo",
        timeout=datetime.timedelta(seconds=sc.group_timeout_s),
    )
    b = sc.behaviours[rank]
    if b.kind == "die":
        time.sleep(b.seconds)
        q.put({"rank": rank, "outcome": "died", "t_enter": None, "t_exit": time.time() - epoch})
        q.close(); q.join_thread()  # flush before dying
        os._exit(1)
    if b.kind == "hang":
        q.put({"rank": rank, "outcome": "hanging", "t_enter": None, "t_exit": None})
        q.close(); q.join_thread()
        while True:
            time.sleep(3600)
    time.sleep(b.seconds)  # "weight loading"

    timeout_s = barrier_timeout_s(sc.timeout_s)
    t_enter = time.time() - epoch
    rec = {"rank": rank, "t_enter": t_enter, "timeout_s": timeout_s}
    try:
        dist_barrier_after_load(group=group, tp_rank=rank, timeout_s=timeout_s)
        rec["outcome"] = "passed"
    except ValueError as e:
        rec["outcome"] = "raised"
        rec["message"] = str(e)
        rec["gloo_error"] = str(e.__cause__).splitlines()[0][:300] if e.__cause__ else ""
    except Exception as e:  # non-zero ranks see gloo errors when rank 0 dies
        rec["outcome"] = "error"
        rec["gloo_error"] = f"{type(e).__name__}: {str(e).splitlines()[0][:300]}"
    rec["t_exit"] = time.time() - epoch
    q.put(rec)
    q.close(); q.join_thread()  # flush before a possible os._exit
    if rec["outcome"] != "passed":
        os._exit(1)  # SGLang's scheduler process dies on this exception
    dist.destroy_process_group()


@dataclass
class Result:
    scenario: str
    description: str
    world: int
    timeout_s: float
    rank0_outcome: str
    rank0_wait_s: Optional[float]
    listed_failed_ranks: List[int]
    gloo_error: str
    ranks: List[dict] = field(default_factory=list)


def _parse_failed(msg: str) -> List[int]:
    m = re.search(r"Ranks? ([\d, ]+) failed to pass monitoredBarrier", msg or "")
    return [int(x) for x in m.group(1).replace(" ", "").split(",") if x] if m else []


def run_scenario(sc: Scenario, deadline_s: Optional[float] = None) -> Result:
    ctx = mp.get_context("spawn")
    q = ctx.Queue()
    port = _free_port()
    epoch = time.time()
    procs = [ctx.Process(target=_worker, args=(r, sc, port, epoch, q), daemon=True)
             for r in range(sc.world)]
    for p in procs:
        p.start()
    if deadline_s is None:
        longest = max(b.seconds for b in sc.behaviours)
        deadline_s = 20 + 2 * longest + sc.world * barrier_bound(sc)
    recs: Dict[int, dict] = {}
    end = time.time() + deadline_s
    while time.time() < end and len(recs) < sc.world:
        try:
            r = q.get(timeout=0.2)
            recs[r["rank"]] = r
        except Exception:
            pass
        if 0 in recs and recs[0].get("outcome") in ("passed", "raised", "error"):
            # once rank 0 is done, give the others a moment to report, then stop
            if all(not p.is_alive() for p in procs[1:]) or time.time() > recs[0]["t_exit"] + epoch + 2:
                break
    while True:  # drain
        try:
            r = q.get_nowait()
            recs[r["rank"]] = r
        except Exception:
            break
    for p in procs:
        if p.is_alive():
            p.kill()
        p.join(timeout=5)
    r0 = recs.get(0, {"outcome": "no-report"})
    wait = (r0["t_exit"] - r0["t_enter"]) if r0.get("t_enter") is not None and r0.get("t_exit") else None
    return Result(
        scenario=sc.name, description=sc.description, world=sc.world,
        timeout_s=sc.env_timeout_s or sc.timeout_s,
        rank0_outcome=r0["outcome"], rank0_wait_s=None if wait is None else round(wait, 2),
        listed_failed_ranks=_parse_failed(r0.get("gloo_error", "")),
        gloo_error=r0.get("gloo_error", ""),
        ranks=[recs[k] for k in sorted(recs)],
    )


def barrier_bound(sc: Scenario) -> float:
    return sc.env_timeout_s or sc.timeout_s


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scenario", default="all",
                    help="one of: all, " + ", ".join(build_scenarios(2, 1).keys()))
    ap.add_argument("--world", type=int, default=8, help="ranks in the TP group (32 on the platform)")
    ap.add_argument("--timeout", type=float, default=2.0,
                    help="barrier timeout in seconds; stands for SGLang's 480 s")
    ap.add_argument("--json", help="write results here")
    args = ap.parse_args(argv)

    import torch
    scs = build_scenarios(args.world, args.timeout)
    names = list(scs) if args.scenario == "all" else [args.scenario]
    results = []
    print(f"torch {torch.__version__}, world={args.world}, barrier timeout={args.timeout}s "
          f"(1 s here = {480 / args.timeout:.0f} s in production)")
    for n in names:
        res = run_scenario(scs[n])
        results.append(res)
        listed = res.listed_failed_ranks
        listed_s = (f"{len(listed)} ranks [{listed[0]}..{listed[-1]}]" if len(listed) > 3 else str(listed))
        print(f"{n:14s} rank0={res.rank0_outcome:7s} wait={res.rank0_wait_s!s:>6}s listed={listed_s:22s} | {res.description}")
    if args.json:
        with open(args.json, "w") as f:
            json.dump({"torch": torch.__version__, "world": args.world, "timeout_s": args.timeout,
                       "results": [asdict(r) for r in results]}, f, indent=1)
    return 0


if __name__ == "__main__":
    sys.exit(main())
