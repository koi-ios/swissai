"""Per-bucket capture cost and eager-fallback penalty models.

Capture cost, seconds per bucket: c(size) = a + b * size.
  MEASURED (fitted) for the two Clariden profiles from the lab's SGLang 0.5.16
  startup logs (sglang-cold-start/analysis/results/capture_timeline.csv, servekit
  runs). Fit: least squares of cumulative elapsed time against cumulative size
  over capture steps 2..n (step 1 carries the one-off JIT/allocator cost).
  tqdm reports whole seconds, so per-size values are +-1 s; the cumulative fit
  has max residual 2.6-3.1 s (prefill) and 0.6 s (decode).
    Llama-3.1-70B TP4 (3 runs): prefill a = 284 / 373 / 428 ms,
      b = 141 / 121 / 105 us per token; decode a = 216 / 288 / 285 ms.
    Apertus-8B TP4 (2 runs): prefill a = 170 / 195 ms, b = 48 / 46 us per token;
      decode a = 126 / 138 ms.
  Medians are used below. The one-off cost is step 1 minus the fitted cost of
  its own size (8192 tokens).

Eager penalty, extra seconds per forward when a bucket runs eagerly:
  PREDICTED, not measured. Model: eager forward = max(host time, GPU time),
  graph replay ~= GPU time, so the penalty is the host time the GPU cannot hide:
  penalty(size) = max(floor, host_s - (gpu_fixed_s + gpu_per_unit_s * size)).
  host_s ~ launches per forward x ~10 us + Python; gpu_fixed_s ~ weight bytes /
  HBM bandwidth (decode); gpu_per_unit_s ~ 2 * params / (TP * dense FLOP/s)
  per token (prefill) or KV read per sequence (decode). The Modal experiment
  (modal_app.py, sweep "eager") measures these; pass its JSON via --penalty.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field, replace
from typing import Dict, Optional


@dataclass(frozen=True)
class CaptureCost:
    a_s: float
    b_s: float

    def __call__(self, size: int) -> float:
        return self.a_s + self.b_s * size


@dataclass(frozen=True)
class EagerPenalty:
    host_s: float
    gpu_fixed_s: float
    gpu_per_unit_s: float
    floor_s: float

    def __call__(self, size: int) -> float:
        gpu = self.gpu_fixed_s + self.gpu_per_unit_s * size
        return max(self.floor_s, self.host_s - gpu)


@dataclass(frozen=True)
class EngineProfile:
    name: str
    description: str
    one_off_s: float
    capture: Dict[str, CaptureCost]
    penalty: Dict[str, EagerPenalty]
    # Decode steps per second of busy time (1 / TPOT); converts decode busy
    # time into forwards for the expected-value ranking.
    decode_steps_per_busy_s: float
    capture_source: str
    penalty_source: str = "PREDICTED (costs.py docstring); replace with Modal data"
    measured_penalty: Optional[Dict[str, Dict[int, float]]] = field(default=None)
    # Sensitivity knobs: multiply every eager penalty / prefill penalties only.
    penalty_scale: float = 1.0
    prefill_penalty_scale: float = 1.0

    def capture_cost(self, phase: str, size: int) -> float:
        return self.capture[phase](size)

    def eager_penalty(self, phase: str, size: int) -> float:
        if self.measured_penalty and phase in self.measured_penalty:
            value = _interpolate(self.measured_penalty[phase], size)
        else:
            value = self.penalty[phase](size)
        scale = self.penalty_scale * (self.prefill_penalty_scale if phase == "prefill" else 1.0)
        return value * scale

    def to_json(self) -> dict:
        d = asdict(self)
        return d


def _interpolate(points: Dict[int, float], size: int) -> float:
    xs = sorted(points)
    if size <= xs[0]:
        return max(0.0, points[xs[0]])
    if size >= xs[-1]:
        return max(0.0, points[xs[-1]])
    for lo, hi in zip(xs, xs[1:]):
        if lo <= size <= hi:
            w = (size - lo) / (hi - lo)
            return max(0.0, points[lo] * (1 - w) + points[hi] * w)
    raise AssertionError("unreachable")


PROFILES: Dict[str, EngineProfile] = {
    "llama70b_tp4_gh200": EngineProfile(
        name="llama70b_tp4_gh200",
        description="Llama-3.1-70B / Apertus-70B class, TP4, Clariden GH200, SGLang 0.5.16",
        one_off_s=16.0,
        capture={
            "prefill": CaptureCost(a_s=0.373, b_s=1.21e-4),
            "decode": CaptureCost(a_s=0.285, b_s=0.0),
        },
        penalty={
            # 80 layers; weights 141 GB / 4 GPUs at ~4 TB/s ~= 8.8 ms per step.
            "decode": EagerPenalty(
                host_s=17e-3, gpu_fixed_s=8.8e-3, gpu_per_unit_s=5e-5, floor_s=5e-4
            ),
            # 2 * 70e9 / 4 FLOP per token at ~7e14 FLOP/s ~= 5e-5 s per token.
            "prefill": EagerPenalty(
                host_s=17e-3, gpu_fixed_s=2e-3, gpu_per_unit_s=5e-5, floor_s=5e-4
            ),
        },
        decode_steps_per_busy_s=50.0,
        capture_source="MEASURED: fit to 3 Clariden servekit logs (capture_timeline.csv)",
    ),
    "apertus8b_tp4_gh200": EngineProfile(
        name="apertus8b_tp4_gh200",
        description="Apertus-8B, TP4, Clariden GH200, SGLang 0.5.16",
        one_off_s=11.4,
        capture={
            "prefill": CaptureCost(a_s=0.183, b_s=4.7e-5),
            "decode": CaptureCost(a_s=0.132, b_s=0.0),
        },
        penalty={
            "decode": EagerPenalty(
                host_s=12e-3, gpu_fixed_s=1.0e-3, gpu_per_unit_s=2e-5, floor_s=5e-4
            ),
            "prefill": EagerPenalty(
                host_s=12e-3, gpu_fixed_s=1e-3, gpu_per_unit_s=7e-6, floor_s=5e-4
            ),
        },
        decode_steps_per_busy_s=100.0,
        capture_source="MEASURED: fit to 2 Clariden servekit logs (capture_timeline.csv)",
    ),
    "llama8b_tp1_h100": EngineProfile(
        name="llama8b_tp1_h100",
        description="Llama-3.1-8B, TP1, H100 (Modal target), SGLang 0.5.20",
        one_off_s=8.0,
        capture={
            # Host part as on GH200 8B; GPU part x4 for TP1 (3 forwards/bucket).
            "prefill": CaptureCost(a_s=0.18, b_s=8e-5),
            "decode": CaptureCost(a_s=0.13, b_s=0.0),
        },
        penalty={
            # Weights 16 GB at ~3 TB/s ~= 5.3 ms per decode step.
            "decode": EagerPenalty(
                host_s=12e-3, gpu_fixed_s=5.3e-3, gpu_per_unit_s=3e-5, floor_s=5e-4
            ),
            "prefill": EagerPenalty(
                host_s=12e-3, gpu_fixed_s=1e-3, gpu_per_unit_s=2.7e-5, floor_s=5e-4
            ),
        },
        decode_steps_per_busy_s=120.0,
        capture_source="PREDICTED from the GH200 fits; Modal run measures it",
    ),
}


def load_profile(name: str, penalty_json: Optional[str] = None) -> EngineProfile:
    """A built-in profile, optionally with eager penalties measured on Modal.

    penalty_json: {"decode": {"1": 0.0061, "8": ...}, "prefill": {"512": ...}}
    in seconds per forward (output of `modal_app.py summarize`).
    """
    profile = PROFILES[name]
    if penalty_json is None:
        return profile
    with open(penalty_json) as f:
        raw = json.load(f)
    measured = {
        phase: {int(k): float(v) for k, v in points.items()}
        for phase, points in raw.items()
        if phase in ("decode", "prefill") and points
    }
    return replace(
        profile,
        measured_penalty=measured,
        penalty_source=f"MEASURED: {penalty_json}",
    )
