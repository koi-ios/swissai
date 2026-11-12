"""Modal experiment: lazy CUDA-graph capture in SGLang 0.5.20 (see MODAL.md).

Arms (one fresh container = one cold start each):
  default  SGLang 0.5.20 + patch, no lazy keys (captures every bucket at startup)
  fixed    small fixed capture set: decode bs <= 32, prefill tokens <= 2048
  lazy     policy-chosen startup set, the rest captured while idle (the patch)
Extra sweeps:
  eager        eager-fallback latency per size: --disable-cuda-graph vs default
  correctness  greedy outputs + prefix-cache reuse after deferred captures

Usage:
  python modal_app.py --dry-run                      # local, CPU, mock server
  modal run modal_app.py --mode smoke                # 1 short lazy run, ~$0.5
  modal run modal_app.py --mode arms --repeats 3     # 9 cold starts, TP1
  modal run modal_app.py --mode eager
  modal run modal_app.py --mode correctness
  modal run modal_app.py --mode arms --tp 2 --repeats 2
  modal volume get lazy-cuda-graphs-results / ./results/modal
  python modal_app.py --summarize results/modal
"""

from __future__ import annotations

import argparse
import json
import os
import socket
import statistics
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

HERE = Path(__file__).resolve().parent
PATCH = HERE / "patches" / "0001-sglang-v0.5.20-lazy-cuda-graph-capture.patch"
POLICY = HERE / "results" / "policy" / "modal_apertus70b_traffic_8b_tp1.json"
REPLAY = HERE / "results" / "replay" / "apertus70b_windows.jsonl"
TRACE_MODEL = "swissai/Apertus-70B-Instruct"

SGLANG_IMAGE = os.environ.get("LAZY_SGLANG_IMAGE", "lmsysorg/sglang:v0.5.20")
MODEL = os.environ.get("LAZY_MODEL", "swiss-ai/Apertus-8B-Instruct-2509")
HF_SECRET = os.environ.get("LAZY_HF_SECRET")  # e.g. "huggingface" for gated models
APP_NAME = "lazy-cuda-graphs"
RESULTS_VOLUME = "lazy-cuda-graphs-results"
LAUNCHER = ["python3", "-m", "sglang.launch_server"]
GPU_PRICE_PER_H = {"H100": 3.95}  # approximate, check modal.com/pricing

# ---------------------------------------------------------------------------
# Arm configuration (pure Python, shared by Modal and the dry run)
# ---------------------------------------------------------------------------

PREFILL_DEFAULT = (
    list(range(4, 33, 4)) + list(range(48, 257, 16)) + list(range(288, 513, 32))
    + list(range(576, 1025, 64)) + list(range(1280, 4097, 256)) + list(range(4608, 8193, 512))
)
DECODE_DEFAULT = [1, 2, 4, 8, 12] + list(range(16, 257, 8))


def fixed_config() -> dict:
    """A small fixed set (prefill ceiling 2048, decode cap 32), no lazy capture."""
    return {
        "decode": {"bs": [b for b in DECODE_DEFAULT if b <= 32]},
        "prefill": {"bs": [n for n in PREFILL_DEFAULT if n <= 2048]},
    }


def lazy_config(policy_path: Path = POLICY) -> dict:
    plans = json.loads(policy_path.read_text())
    plan = plans[TRACE_MODEL] if TRACE_MODEL in plans else next(iter(plans.values()))
    cfg = plan["cuda_graph_config"]
    for phase in ("decode", "prefill"):
        p = cfg[phase]
        assert set(p["initial_bs"]) <= set(p["bs"]), phase
        assert not set(p["initial_bs"]) & set(p["deferred_bs"]), phase
        assert len(p["deferred_priority"]) == len(p["deferred_bs"]), phase
    return cfg


def replay_windows(path: Path = REPLAY, *, k: int = 8, time_scale: float = 0.5,
                   max_prompt: int = 8192, max_output: int = 256) -> dict:
    """Superpose the k launch windows nearest the median size (independent
    cold starts of the Apertus-70B arrival process => about k x its rate) and
    compress time by `time_scale`. Deterministic."""
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    rows.sort(key=lambda r: (len(r["arrivals_s"]), r["launch"]))
    mid = len(rows) // 2
    chosen = rows[max(0, mid - k // 2): max(0, mid - k // 2) + k]
    events = []
    for r in chosen:
        for a, p, o in zip(r["arrivals_s"], r["prompt_tokens"], r["output_tokens"]):
            events.append((a, p if p > 0 else 356, o if o > 0 else 124))
    events.sort()
    return {
        "windows": [r["launch"] for r in chosen],
        "arrivals_s": [e[0] for e in events],
        "prompt_tokens": [e[1] for e in events],
        "output_tokens": [e[2] for e in events],
        "time_scale": time_scale,
        "max_prompt": max_prompt,
        "max_output": max_output,
    }


def build_specs(mode: str, *, model: str, tp: int, repeats: int) -> List[dict]:
    lazy_env = {
        "SGLANG_LAZY_CUDA_GRAPH_MIN_IDLE_S": "1.0",
        "SGLANG_LAZY_CUDA_GRAPH_IDLE_COST_RATIO": "0",
        "SGLANG_LAZY_CUDA_GRAPH_PROMOTE_AFTER": "4",
    }
    base = {
        "model": model,
        "tp": tp,
        "server_args": ["--mem-fraction-static", "0.85"],
        "launch_request": {"prompt_tokens": 400, "output_tokens": 64},
        "timeout_s": 1800,
    }
    arms = {
        "default": {"cuda_graph_config": None, "env": {}},
        "fixed": {"cuda_graph_config": fixed_config(), "env": {}},
        "lazy": {"cuda_graph_config": lazy_config(), "env": lazy_env},
    }
    specs: List[dict] = []
    if mode == "smoke":
        return [{**base, **arms["lazy"], "arm": "lazy", "repeat": 0,
                 "replay": {**replay_windows(k=2), "time_scale": 0.25},
                 "wait_full_capture_s": 300}]
    if mode == "arms":
        replay = replay_windows()
        # Interleave arms so drift (node, time of day) does not bias one arm.
        for r in range(repeats):
            for arm in ("default", "fixed", "lazy"):
                specs.append({
                    **base, **arms[arm], "arm": arm, "repeat": r, "replay": replay,
                    "wait_full_capture_s": 600 if arm == "lazy" else 0,
                    "throughput": {"n": 512, "prompt_len": 512, "output_len": 128,
                                   "concurrency": 64},
                })
        return specs
    if mode == "eager":
        sweep = {"prefill_tokens": [512, 1536, 4096, 8192], "decode_bs": [1, 8, 32, 64],
                 "reps": 5, "decode_tokens": 128}
        for arm, extra in (("graphs", []), ("eager", ["--disable-cuda-graph"])):
            specs.append({**base, "arm": arm, "repeat": 0, "cuda_graph_config": None,
                          "env": {}, "server_args": base["server_args"] + extra,
                          "launch_request": None, "eager_sweep": sweep})
        return specs
    if mode == "correctness":
        corr = {"prompt_tokens": [96, 300, 1000, 3000, 6000], "prefix_tokens": 3000,
                "suffix_tokens": 200, "output_tokens": 32, "wait_s": 600}
        # Defer every prefill bucket so captures run while the prefix is cached.
        lazy = json.loads(json.dumps(arms["lazy"]["cuda_graph_config"]))
        lazy["prefill"]["deferred_bs"] = lazy["prefill"]["initial_bs"] + lazy["prefill"]["deferred_bs"]
        lazy["prefill"]["deferred_priority"] = (
            [1.0] * len(lazy["prefill"]["initial_bs"]) + lazy["prefill"]["deferred_priority"]
        )
        lazy["prefill"]["initial_bs"] = []
        for arm, cfg, env in (("default", None, {}), ("lazy", lazy, lazy_env)):
            specs.append({**base, "arm": arm, "repeat": 0, "cuda_graph_config": cfg,
                          "env": env, "launch_request": None, "correctness": corr})
        return specs
    raise ValueError(f"unknown mode {mode!r}")


def estimate_cost(mode: str, *, tp: int, repeats: int) -> float:
    minutes = {"smoke": 10, "arms": 14 * 3 * repeats, "eager": 2 * 12, "correctness": 2 * 12}[mode]
    return minutes / 60 * GPU_PRICE_PER_H["H100"] * tp


# ---------------------------------------------------------------------------
# Summaries (local)
# ---------------------------------------------------------------------------


def _median_range(xs: List[float]) -> str:
    xs = [x for x in xs if x is not None]
    if not xs:
        return "n/a"
    return f"{statistics.median(xs):.3g} [{min(xs):.3g}-{max(xs):.3g}] (n={len(xs)})"


def summarize(results_dir: Path) -> dict:
    recs = [json.loads(p.read_text()) for p in sorted(results_dir.rglob("*.json"))
            if p.name not in ("summary.json", "eager_penalty.json")]
    recs = [r for r in recs if "spec" in r]
    out: Dict[str, dict] = {}
    by_arm: Dict[str, List[dict]] = {}
    for r in recs:
        mode = str(r["spec"].get("run_id", "run")).split("-tp")[0]
        key = f"{mode}/{r['spec']['arm']}-tp{r['spec'].get('tp', 1)}"
        by_arm.setdefault(key, []).append(r)
    print(f"{len(recs)} records in {results_dir}")
    for key, rs in sorted(by_arm.items()):
        row = {
            "runs": len(rs),
            "time_to_ready_s": _median_range([r.get("time_to_ready_s") for r in rs]),
            "launch_ttft_s": _median_range([(r.get("launch_request") or {}).get("ttft_s") for r in rs]),
            "startup_capture_s": _median_range([r["log"].get("startup_capture_total_s") for r in rs]),
            "replay_ttft_p50_s": _median_range([((r.get("replay") or {}).get("all") or {}).get("ttft_p50_s") for r in rs]),
            "replay_ttft_p99_s": _median_range([((r.get("replay") or {}).get("all") or {}).get("ttft_p99_s") for r in rs]),
            "replay_tpot_p50_s": _median_range([((r.get("replay") or {}).get("all") or {}).get("tpot_p50_s") for r in rs]),
            "replay_tpot_p99_s": _median_range([((r.get("replay") or {}).get("all") or {}).get("tpot_p99_s") for r in rs]),
            "throughput_tok_s": _median_range([(r.get("throughput") or {}).get("output_tok_per_s") for r in rs]),
            "full_capture_t_s": _median_range([r.get("full_capture_t") for r in rs]),
            "lazy_steps": _median_range([len(r["log"].get("lazy_steps", [])) for r in rs]),
            "max_lazy_step_s": _median_range([max([s["seconds"] for s in r["log"].get("lazy_steps", [])], default=None) for r in rs]),
            "gpu_mem_after_full_mib": _median_range([(r.get("gpu_mem_after_full_capture_mib") or r.get("gpu_mem_after_replay_mib") or [None])[0] for r in rs]),
            "errors": sum(1 for r in rs if r.get("error") or r["log"].get("errors")),
        }
        out[key] = row
        print(f"\n[{key}]")
        for k, v in row.items():
            print(f"  {k:24s} {v}")
    penalty = _eager_penalty(by_arm)
    if penalty:
        (results_dir / "eager_penalty.json").write_text(json.dumps(penalty, indent=1))
        print("\neager penalty (s per forward):", json.dumps(penalty))
    corr = _correctness(by_arm)
    if corr:
        out["correctness"] = corr
        print("\ncorrectness:", json.dumps(corr))
    (results_dir / "summary.json").write_text(json.dumps(out, indent=1))
    return out


def _eager_penalty(by_arm: Dict[str, List[dict]]) -> Optional[dict]:
    g = [r for k, rs in by_arm.items() if k.split("/")[-1].startswith("graphs") for r in rs if r.get("eager_sweep")]
    e = [r for k, rs in by_arm.items() if k.split("/")[-1].startswith("eager") for r in rs if r.get("eager_sweep")]
    if not g or not e:
        return None
    gs, es = g[0]["eager_sweep"], e[0]["eager_sweep"]
    out = {"prefill": {}, "decode": {}}
    for n, v in gs["prefill_ttft_s"].items():
        ev = es["prefill_ttft_s"].get(n, {})
        if v.get("median") is not None and ev.get("median") is not None:
            out["prefill"][n] = max(0.0, ev["median"] - v["median"])
    for bs, v in gs["decode_tpot_s"].items():
        ev = es["decode_tpot_s"].get(bs, {})
        if v.get("median") is not None and ev.get("median") is not None:
            out["decode"][bs] = max(0.0, ev["median"] - v["median"])
    return out


def _correctness(by_arm: Dict[str, List[dict]]) -> Optional[dict]:
    ref = next((r for k, rs in by_arm.items() if k.split("/")[-1].startswith("default") for r in rs if r.get("correctness")), None)
    lazy = next((r for k, rs in by_arm.items() if k.split("/")[-1].startswith("lazy") for r in rs if r.get("correctness")), None)
    if not ref or not lazy:
        return None
    out = {}
    for phase in ("phase1", "phase2"):
        for name, ids in ref["correctness"][phase].items():
            other = lazy["correctness"][phase].get(name)
            same = other == ids
            first_diff = None
            if not same and other is not None:
                first_diff = next((i for i, (a, b) in enumerate(zip(ids, other)) if a != b), min(len(ids), len(other)))
            out[f"{phase}/{name}"] = {"identical": same, "first_diff": first_diff}
    return out


# ---------------------------------------------------------------------------
# Modal definitions
# ---------------------------------------------------------------------------

try:
    import modal
except ImportError:  # dry run without modal installed
    modal = None

if modal is not None:
    image = (
        modal.Image.from_registry(SGLANG_IMAGE)
        .add_local_file(str(PATCH), "/opt/lazy/0001.patch", copy=True)
        .add_local_file(str(HERE / "bench" / "apply_patch.py"), "/opt/lazy/apply_patch.py", copy=True)
        .run_commands("python3 /opt/lazy/apply_patch.py /opt/lazy/0001.patch --expect-version 0.5.20")
        .env({"HF_HOME": "/hf-cache", "PYTHONUNBUFFERED": "1"})
        .add_local_python_source("bench")
    )
    app = modal.App(APP_NAME)
    hf_cache = modal.Volume.from_name("hf-cache", create_if_missing=True)
    results_vol = modal.Volume.from_name(RESULTS_VOLUME, create_if_missing=True)
    secrets = [modal.Secret.from_name(HF_SECRET)] if HF_SECRET else []
    volumes = {"/hf-cache": hf_cache, "/results": results_vol}

    @app.function(image=image, volumes={"/hf-cache": hf_cache}, secrets=secrets,
                  timeout=3600, cpu=4)
    def download_model(model: str) -> str:
        from huggingface_hub import snapshot_download

        path = snapshot_download(model, allow_patterns=["*.json", "*.safetensors", "*.model",
                                                        "*.txt", "*.jinja", "tokenizer*"])
        hf_cache.commit()
        return path

    def _run(spec: dict) -> dict:
        from bench.arm import run_arm

        out = Path("/results") / spec.get("run_id", "run")
        rec = run_arm(spec, launcher=LAUNCHER, out_dir=out)
        results_vol.commit()
        return {k: rec.get(k) for k in ("tag", "time_to_ready_s", "full_capture_t", "error")}

    @app.function(image=image, gpu="H100", volumes=volumes, secrets=secrets, timeout=3600)
    def run_arm_tp1(spec: dict) -> dict:
        return _run(spec)

    @app.function(image=image, gpu="H100:2", volumes=volumes, secrets=secrets, timeout=3600)
    def run_arm_tp2(spec: dict) -> dict:
        return _run(spec)

    @app.local_entrypoint()
    def main(mode: str = "smoke", tp: int = 1, repeats: int = 3, model: str = MODEL):
        specs = build_specs(mode, model=model, tp=tp, repeats=repeats)
        run_id = f"{mode}-tp{tp}-{time.strftime('%Y%m%d-%H%M%S')}"
        print(f"{len(specs)} runs, estimated cost ${estimate_cost(mode, tp=tp, repeats=repeats):.1f}")
        print("downloading weights (CPU):", download_model.remote(model))
        fn = run_arm_tp1 if tp == 1 else run_arm_tp2
        # One container per cold start, sequential: arms must not share a node cache.
        for spec in specs:
            spec["run_id"] = run_id
            print(fn.remote(spec))
        print(f"results: modal volume get {RESULTS_VOLUME} /{run_id} ./results/modal/")


# ---------------------------------------------------------------------------
# Local dry run: every mode against bench/mock_server.py on CPU
# ---------------------------------------------------------------------------


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def dry_run(out_dir: Optional[Path] = None) -> int:
    from bench.arm import run_arm

    assert PATCH.exists(), PATCH
    assert POLICY.exists() and REPLAY.exists(), "run the policy/replay commands first"
    out_dir = out_dir or HERE / "results" / "dryrun"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "README.md").write_text(
        "MOCK DATA. Produced by `python modal_app.py --dry-run` against\n"
        "bench/mock_server.py on a CPU. These numbers measure nothing; they only\n"
        "show that the harness, parsers and summary run end to end.\n"
    )
    launcher = [sys.executable, "-m", "bench.mock_server", "--mock-time-scale", "0.02"]
    if modal is not None:
        print(f"modal {modal.__version__}: app '{APP_NAME}', image {SGLANG_IMAGE}, "
              "functions download_model, run_arm_tp1, run_arm_tp2")
    total = 0.0
    for mode in ("smoke", "arms", "eager", "correctness"):
        specs = build_specs(mode, model="mock/model", tp=1, repeats=1)
        total += estimate_cost(mode, tp=1, repeats=3 if mode == "arms" else 1)
        for spec in specs:
            # Shrink the workload so the dry run takes seconds, not minutes.
            if spec.get("replay"):
                spec["replay"] = {**spec["replay"], "time_scale": 0.004, "max_output": 8}
            if spec.get("throughput"):
                spec["throughput"] = {**spec["throughput"], "n": 16, "output_len": 8, "concurrency": 4}
            if spec.get("eager_sweep"):
                spec["eager_sweep"] = {**spec["eager_sweep"], "reps": 1, "decode_tokens": 8,
                                       "decode_bs": [1, 8]}
            if spec.get("correctness"):
                spec["correctness"] = {**spec["correctness"], "output_tokens": 8, "wait_s": 60}
            spec["timeout_s"] = 120
            spec["run_id"] = f"dry-{mode}"
            t = time.monotonic()
            rec = run_arm(spec, launcher=launcher, out_dir=out_dir / spec["run_id"], port=_free_port())
            status = rec.get("error") or "ok"
            print(f"  {mode:12s} {rec['tag']:20s} ready={rec.get('time_to_ready_s', float('nan')):.2f}s "
                  f"lazy_steps={len(rec['log']['lazy_steps'])} ({time.monotonic() - t:.1f}s) {status}")
            if rec.get("error"):
                return 1
    summary = summarize(out_dir)
    corr = summary.get("correctness", {})
    if corr and not all(v["identical"] for v in corr.values()):
        print("dry run: mock outputs differ across arms (harness bug)")
        return 1
    print(f"\ndry run OK; results in {out_dir}")
    print(f"estimated Modal cost (H100 ~${GPU_PRICE_PER_H['H100']}/h): smoke+arms(3x)+eager+"
          f"correctness = ${total:.1f} at TP1; TP2 arms x2 adds "
          f"${estimate_cost('arms', tp=2, repeats=2):.1f}")
    return 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--summarize", type=Path)
    a = ap.parse_args()
    if a.summarize:
        summarize(a.summarize)
        sys.exit(0)
    sys.exit(dry_run() if a.dry_run else 0)
