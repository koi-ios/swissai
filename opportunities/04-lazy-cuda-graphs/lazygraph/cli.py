"""Command line for the lazy-capture policy and simulator.

  python -m lazygraph.cli profile  --trace T --model M [--model M2 ...]
  python -m lazygraph.cli policy   --trace T --model M --engine E [--out plan.json]
  python -m lazygraph.cli simulate --trace T --model M --engine E [--out sim.json]
  python -m lazygraph.cli replay   --trace T --model M --out arrivals.jsonl

Engines: see lazygraph.costs.PROFILES. --penalty takes the eager-penalty JSON
measured on Modal (modal_app.py summarize) instead of the predicted model.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np

from lazygraph.buckets import decode_buckets, prefill_buckets
from lazygraph.costs import PROFILES, load_profile
from lazygraph.policy import make_plan, rank_buckets, values_table
from lazygraph.simulate import SimConfig, adaptive_plan, aggregate, simulate_model
from lazygraph.trace import load_trace, model_traffic, sessions, summarize

DEFAULT_TRACE = str(
    Path(__file__).resolve().parents[3] / "data" / "swissai-trace.parquet"
)


def _common(p: argparse.ArgumentParser) -> None:
    p.add_argument("--trace", default=DEFAULT_TRACE)
    p.add_argument("--model", action="append", required=True)


def _engine(p: argparse.ArgumentParser) -> None:
    p.add_argument("--engine", choices=sorted(PROFILES), default="llama70b_tp4_gh200")
    p.add_argument("--penalty", default=None, help="measured eager penalties JSON")
    p.add_argument("--penalty-scale", type=float, default=1.0,
                   help="multiply eager penalties (sensitivity analysis)")
    p.add_argument("--prefill-penalty-scale", type=float, default=1.0,
                   help="multiply prefill eager penalties only")
    p.add_argument("--lam", type=float, default=1.0,
                   help="requests delayed by each startup second (>= 1)")
    p.add_argument("--horizon", type=float, default=60.0,
                   help="seconds a deferred bucket is expected to wait")
    p.add_argument("--fixed-threshold", dest="adaptive", action="store_false",
                   help="use --lam/--horizon as given instead of estimating them "
                        "from the trace (default: estimate)")
    p.add_argument("--max-decode-bs", type=int, default=256)
    p.add_argument("--max-prefill-tokens", type=int, default=8192)


def cmd_profile(args) -> int:
    trace = load_trace(args.trace)
    rows = {m: summarize(model_traffic(trace, m)) for m in args.model}
    print(json.dumps(rows, indent=1))
    return 0


def _plan(args, model):
    trace = load_trace(args.trace)
    traffic = model_traffic(trace, model)
    profile = replace(load_profile(args.engine, args.penalty),
                      penalty_scale=args.penalty_scale,
                      prefill_penalty_scale=getattr(args, "prefill_penalty_scale", 1.0))
    prefill = prefill_buckets(args.max_prefill_tokens)
    decode = decode_buckets(args.max_decode_bs)

    def plan_fn(t, p, *, lam, horizon_s):
        return make_plan(t, p, lam=lam, horizon_s=horizon_s,
                         prefill=prefill, decode=decode)

    if not args.adaptive:
        return traffic, profile, plan_fn(traffic, profile, lam=args.lam,
                                         horizon_s=args.horizon)
    cfg = SimConfig(
        min_idle_s=getattr(args, "min_idle", 1.0),
        idle_cost_ratio=getattr(args, "idle_cost_ratio", 0.0),
        horizon_s=getattr(args, "sim_horizon", 1800.0),
        session_gap_s=getattr(args, "session_gap", 900.0),
    )
    plan, info = adaptive_plan(traffic, profile, cfg, make_plan_fn=plan_fn,
                               first_horizon_s=args.horizon)
    args._adaptive_info = info
    return traffic, profile, plan


def cmd_policy(args) -> int:
    out = {}
    for model in args.model:
        traffic, profile, plan = _plan(args, model)
        values = rank_buckets(
            traffic,
            profile,
            prefill=plan.planned["prefill"],
            decode=plan.planned["decode"],
        )
        info = getattr(args, "_adaptive_info", None)
        print(f"\n== {model} ({profile.name}) threshold EV > {plan.threshold:.4g}/s "
              f"(lam={plan.lam:.2f}, H={plan.horizon_s:.0f} s{', estimated' if info else ''})")
        print(values_table(values, top=args.top))
        print(
            f"startup capture {plan.startup_capture_s:.1f} s "
            f"(default {plan.default_capture_s:.1f} s, deferred "
            f"{plan.deferred_capture_s:.1f} s), initial decode {plan.initial['decode']}, "
            f"initial prefill {plan.initial['prefill']}, coverage {plan.coverage}"
        )
        print("--cuda-graph-config '" + json.dumps(plan.cuda_graph_config()) + "'")
        out[model] = plan.to_json()
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(out, indent=1))
    return 0


def cmd_simulate(args) -> int:
    cfg = SimConfig(
        min_idle_s=args.min_idle,
        idle_cost_ratio=args.idle_cost_ratio,
        horizon_s=args.sim_horizon,
        session_gap_s=args.session_gap,
    )
    out = {"config": asdict(cfg), "models": {}}
    for model in args.model:
        traffic, profile, plan = _plan(args, model)
        results = simulate_model(traffic, plan, profile, cfg)
        agg = aggregate(results)
        agg["plan"] = {
            "startup_capture_s": plan.startup_capture_s,
            "default_capture_s": plan.default_capture_s,
            "initial": plan.initial,
            "n_deferred": {p: len(v) for p, v in plan.deferred.items()},
            "lazy": plan.lazy,
        }
        agg["engine"] = profile.name
        agg["penalty_source"] = profile.penalty_source
        out["models"][model] = agg
        print(f"\n== {model}: {agg['launches']} launches")
        print(json.dumps({k: agg[k] for k in ("default", "fixed", "lazy") if k in agg},
                         indent=1))
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(out, indent=1))
    return 0


def cmd_sweep(args) -> int:
    """Sensitivity of the lazy arm to eager penalties and idle hysteresis."""
    rows = []
    for model in args.model:
        for scale in args.scales:
            for min_idle in args.min_idles:
                for ratio in args.ratios:
                    ns = argparse.Namespace(**{**vars(args), "penalty_scale": scale})
                    traffic, profile, plan = _plan(ns, model)
                    cfg = SimConfig(min_idle_s=min_idle, idle_cost_ratio=ratio,
                                    horizon_s=args.sim_horizon,
                                    session_gap_s=args.session_gap)
                    agg = aggregate(simulate_model(traffic, plan, profile, cfg))
                    if agg["launches"] == 0:
                        continue
                    rows.append({
                        "model": model, "penalty_scale": scale, "min_idle_s": min_idle,
                        "idle_cost_ratio": ratio, "launches": agg["launches"],
                        "startup_capture_s": round(plan.startup_capture_s, 2),
                        "initial_decode": plan.initial["decode"],
                        "initial_prefill": plan.initial["prefill"],
                        **{f"{arm}_total_mean_s": round(agg[arm]["total_extra_s_mean"], 2)
                           for arm in ("default", "fixed", "lazy")},
                        "lazy_interference_mean_s": round(agg["lazy"]["interference_s_mean"], 3),
                        "lazy_delayed_share": round(agg["lazy"]["delayed_request_share"], 4),
                        "lazy_max_delay_s": round(agg["lazy"]["max_delay_s"], 2),
                        "lazy_full_capture_p50_s": agg["lazy"].get("all_captured_after_s_p50"),
                        "lazy_full_capture_share": agg["lazy"].get("launches_fully_captured_share"),
                    })
                    print(json.dumps(rows[-1]))
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(rows, indent=1))
    return 0


def cmd_replay(args) -> int:
    """Arrival windows after cold starts, for replay on Modal."""
    trace = load_trace(args.trace)
    rows = []
    for model in args.model:
        traffic = model_traffic(trace, model)
        launches = sessions(traffic.busy_intervals(), args.session_gap)
        for i, (t, gap) in enumerate(launches):
            sel = (traffic.t0 >= t) & (traffic.t0 < t + args.window)
            if sel.sum() < args.min_requests:
                continue
            rows.append(
                {
                    "model": model,
                    "launch": i,
                    "prev_gap_s": gap if np.isfinite(gap) else None,
                    "arrivals_s": np.round(traffic.t0[sel] - t, 3).tolist(),
                    "prompt_tokens": traffic.tin[sel].tolist(),
                    "output_tokens": traffic.tout[sel].tolist(),
                }
            )
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    print(f"wrote {len(rows)} launch windows to {args.out}")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="lazygraph")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("profile")
    _common(p)
    p.set_defaults(fn=cmd_profile)

    p = sub.add_parser("policy")
    _common(p)
    _engine(p)
    p.add_argument("--top", type=int, default=15)
    p.add_argument("--out")
    p.set_defaults(fn=cmd_policy)

    p = sub.add_parser("simulate")
    _common(p)
    _engine(p)
    p.add_argument("--min-idle", type=float, default=1.0)
    p.add_argument("--idle-cost-ratio", type=float, default=0.0)
    p.add_argument("--sim-horizon", type=float, default=1800.0)
    p.add_argument("--session-gap", type=float, default=900.0)
    p.add_argument("--out")
    p.set_defaults(fn=cmd_simulate)

    p = sub.add_parser("sweep")
    _common(p)
    _engine(p)
    p.add_argument("--scales", type=float, nargs="+", default=[0.5, 1.0, 2.0, 5.0])
    p.add_argument("--min-idles", type=float, nargs="+", default=[0.0, 1.0, 5.0])
    p.add_argument("--ratios", type=float, nargs="+", default=[0.0, 5.0])
    p.add_argument("--sim-horizon", type=float, default=1800.0)
    p.add_argument("--session-gap", type=float, default=900.0)
    p.add_argument("--out")
    p.set_defaults(fn=cmd_sweep)

    p = sub.add_parser("replay")
    _common(p)
    p.add_argument("--session-gap", type=float, default=900.0)
    p.add_argument("--window", type=float, default=600.0)
    p.add_argument("--min-requests", type=int, default=5)
    p.add_argument("--out", required=True)
    p.set_defaults(fn=cmd_replay)

    args = ap.parse_args(argv)
    return args.fn(args)


if __name__ == "__main__":
    sys.exit(main())
