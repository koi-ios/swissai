"""The verify-tolerance study on a real engine: noise floor, false alarms and detection power.

One call studies one model on one engine, in one container, start to finish:

  1. reference   a clean server: the reference capture, the weight checksums,
                 `calibrate()` (a repeat + K perturbed captures) and H held-out
                 perturbed captures -- the request-level noise of one process
  2. restarts    the clean model on fresh processes: are restarts bitwise
                 reproducible, and is their drift inside the calibrated noise?
  3. configs     the clean model under other engine settings (eager instead of
                 CUDA graphs, small chunked-prefill size, another attention
                 backend): drift a fixed tolerance cannot tell from a fault
  4. determ.     the engine's batch-invariant mode (SGLang
                 --enable-deterministic-inference, vLLM VLLM_BATCH_INVARIANT=1):
                 its own reference and calibration, to test exact mode under load
  5. faults      each corruption from `servekit.faults`, injected into a copy of
                 the real checkpoint files: a fresh server on it, captured on the
                 reference schedule (the candidate as verify sees it) and once
                 under load; plus the weight checksums and held-out NLL

Every capture is written as it is taken (JSONL.gz, the record format of
cpu/study.py), so `python -I cpu/report.py <dir>` analyses real-engine results
and CPU results alike, and an interrupted run keeps what it has.

The same driver runs locally against the CPU engine's HTTP server
(`--engine cpu`), which is how it was tested before any GPU time is spent:

    python -I study_driver.py --engine cpu --model qwen3-0.6b --quick --out results/e2e_cpu
"""
from __future__ import annotations

import argparse
import gzip
import json
import os
import random
import shutil
import signal
import socket
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional

HERE = Path(__file__).resolve().parent
for p in (HERE / "vendor" / "servekit" / "src", Path("/opt/servekit/src")):
    if p.is_dir():
        sys.path.insert(0, str(p))

from servekit import faults  # noqa: E402
from servekit.bench import wait_for_ready  # noqa: E402
from servekit.verify import calibrate, capture, capture_perturbed, load_prompt_set, score, weights_checksum  # noqa: E402

HELD_OUT = (
    "The committee met on Tuesday to review the budget for the coming year. After a long discussion, "
    "the members agreed to increase funding for the public library and to delay the renovation of the "
    "town hall until the spring. Several residents spoke during the open session, most of them in favour "
    "of extending the bus service to the new housing estate north of the river.\n\n"
    "def merge_sorted(a, b):\n    out, i, j = [], 0, 0\n    while i < len(a) and j < len(b):\n"
    "        if a[i] <= b[j]:\n            out.append(a[i]); i += 1\n        else:\n"
    "            out.append(b[j]); j += 1\n    return out + a[i:] + b[j:]\n\n"
    "Photosynthesis converts light energy into chemical energy. In the light-dependent reactions, water "
    "is split and oxygen is released; the Calvin cycle then fixes carbon dioxide into sugars."
)


# --------------------------------------------------------------------------
# models and their fault grids


def _proj(layer="\\d+", suffix="weight"):
    return rf"layers\.{layer}\.(self_attn\.(q|k|v|o)_proj|mlp\.(gate|up|down)_proj)\.{suffix}$"


def grid_dense_bf16(n_layers: int, square: str, seed: int = 0) -> List[tuple]:
    layer = random.Random(seed).randrange(n_layers)
    return [
        ("zero_slice", "1/8 of a tensor", dict(kind="zero_slice", match=_proj(), tp=8, shard=1, seed=seed)),
        ("zero_slice", "1/128 of a tensor", dict(kind="zero_slice", match=_proj(), tp=128, shard=5, seed=seed)),
        ("zero_vocab_shard", "1/8 of the vocab", dict(kind="zero_slice", match=r"embed_tokens\.weight$", tp=8, shard=3, seed=seed)),
        ("dup_slice", "tp=2", dict(kind="dup_slice", match=_proj(), tp=2, src_shard=0, dst_shard=1, seed=seed)),
        ("permute_shards", "tp=2", dict(kind="permute_shards", match=_proj(), tp=2, seed=seed)),
        ("transpose", "square projection", dict(kind="transpose", match=square, seed=seed)),
        ("bitflip_exp_msb", "n=1", dict(kind="bitflip", match=_proj(), bit="exp_msb", n=1, seed=seed)),
        ("bitflip_exp_lsb", "n=16", dict(kind="bitflip", match=_proj(), bit="exp_lsb", n=16, seed=seed)),
        ("bitflip_exp_lsb", "n=4096", dict(kind="bitflip", match=_proj(), bit="exp_lsb", n=4096, seed=seed)),
        ("bitflip_mant_msb", "n=4096", dict(kind="bitflip", match=_proj(), bit="mant_msb", n=4096, seed=seed)),
        ("bitflip_mant_msb", "n=65536", dict(kind="bitflip", match=_proj(), bit="mant_msb", n=65536, seed=seed)),
        ("drop_tensor", "a norm weight", dict(kind="drop_tensor", match=r"(layernorm|q_norm|k_norm)\.weight$", seed=seed)),
        ("noise_one_layer", "sigma=0.001", dict(kind="noise", match=_proj(layer), count=0, sigma=1e-3, seed=seed)),
        ("noise_one_layer", "sigma=0.01", dict(kind="noise", match=_proj(layer), count=0, sigma=1e-2, seed=seed)),
        ("noise_one_layer", "sigma=0.03", dict(kind="noise", match=_proj(layer), count=0, sigma=3e-2, seed=seed)),
        ("scale_norm", "x1.03", dict(kind="scale", match=r"post_attention_layernorm\.weight$", factor=1.03, seed=seed)),
        ("scale_norm", "x1.3", dict(kind="scale", match=r"post_attention_layernorm\.weight$", factor=1.3, seed=seed)),
    ]


def grid_fp8(n_layers: int, seed: int = 0) -> List[tuple]:
    layer = random.Random(seed).randrange(n_layers)
    s = _proj(suffix="weight_scale_inv")
    return [
        ("fp8_scale_roll", "one tensor", dict(kind="scale_blocks", mode="roll", match=s, seed=seed)),
        ("fp8_scale_jitter", "sigma=0.01", dict(kind="scale_blocks", mode="jitter", sigma=0.01, match=_proj(layer, "weight_scale_inv"), count=0, seed=seed)),
        ("fp8_scale_jitter", "sigma=0.1", dict(kind="scale_blocks", mode="jitter", sigma=0.1, match=_proj(layer, "weight_scale_inv"), count=0, seed=seed)),
        ("fp8_scale_factor", "5% of blocks x2", dict(kind="scale_blocks", mode="factor", factor=2.0, fraction=0.05, match=s, seed=seed)),
        ("fp8_scale_ones", "one tensor", dict(kind="scale_blocks", mode="ones", match=s, seed=seed)),
        ("fp8_scale_swap_tp", "tp=2", dict(kind="scale_blocks", mode="swap_tp", tp=2, match=s, seed=seed)),
        ("fp8_bitflip_exp_msb", "n=1", dict(kind="bitflip", bit="exp_msb", n=1, match=_proj(), seed=seed)),
        ("zero_slice", "1/8 of an FP8 tensor", dict(kind="zero_slice", match=_proj(), tp=8, shard=2, seed=seed)),
    ]


def grid_gptoss(n_layers: int, seed: int = 0) -> List[tuple]:
    layer = random.Random(seed).randrange(n_layers)
    ex = r"mlp\.experts\.(gate_up|down)_proj"
    attn = rf"layers\.{layer}\.self_attn\.(q|k|v|o)_proj\.weight$"
    return [
        ("swap_experts", "one layer, all tensors", dict(kind="swap_experts", match=rf"layers\.{layer}\.mlp\.experts\.", seed=seed)),
        ("swap_experts", "one layer, down_proj only", dict(kind="swap_experts", match=rf"layers\.{layer}\.mlp\.experts\.down_proj", seed=seed)),
        ("mxfp4_scale_shift+1", "1% of one tensor's blocks", dict(kind="mxfp4_scales", mode="shift", k=1, fraction=0.01, match=ex + "_scales$", seed=seed)),
        ("mxfp4_scale_shift+1", "10% of one tensor's blocks", dict(kind="mxfp4_scales", mode="shift", k=1, fraction=0.1, match=ex + "_scales$", seed=seed)),
        ("mxfp4_scale_roll", "one tensor", dict(kind="mxfp4_scales", mode="roll", match=ex + "_scales$", seed=seed)),
        ("zero_expert_blocks", "all experts, one layer", dict(kind="zero_slice", tp=1, shard=0, count=0, match=rf"layers\.{layer}\.{ex}_blocks$", seed=seed)),
        ("zero_expert_blocks", "all experts, all layers (sglang#34448)", dict(kind="zero_slice", tp=1, shard=0, count=0, match=ex + "_blocks$", seed=seed)),
        ("bitflip_exp_msb", "n=1, attention", dict(kind="bitflip", bit="exp_msb", n=1, match=r"self_attn\.(q|k|v|o)_proj\.weight$", seed=seed)),
        ("noise_one_layer", "sigma=0.01, attention", dict(kind="noise", sigma=1e-2, count=0, match=attn, seed=seed)),
    ]


def _distinct_seeds(grid: List[tuple], base: int = 0) -> List[tuple]:
    """One seed per fault, so single-tensor faults land on different tensors."""
    for i, (_, _, spec) in enumerate(grid):
        spec["seed"] = base * 1000 + i
    return grid


@dataclass
class ModelSpec:
    key: str
    hf_id: str
    faults: List[tuple]
    alt_attention: str  # a second attention backend to compare against the default
    start_s: float  # expected server start time on an H100, for the plan (an estimate)
    capture_s: float  # expected seconds per capture of the built-in prompt set (an estimate)
    sglang_args: List[str] = field(default_factory=list)
    vllm_args: List[str] = field(default_factory=list)
    tokenizer: Optional[str] = None


MODELS: Dict[str, ModelSpec] = {
    "qwen3-1.7b": ModelSpec("qwen3-1.7b", "Qwen/Qwen3-1.7B",
                            _distinct_seeds(grid_dense_bf16(28, r"layers\.\d+\.self_attn\.(q|o)_proj\.weight$")), "triton", 60, 6),
    "qwen3-1.7b-fp8": ModelSpec("qwen3-1.7b-fp8", "Qwen/Qwen3-1.7B-FP8", _distinct_seeds(grid_fp8(28)), "triton", 70, 6),
    "gpt-oss-20b": ModelSpec("gpt-oss-20b", "openai/gpt-oss-20b", _distinct_seeds(grid_gptoss(24)), "triton", 150, 12),
    # Local only: the CPU engine behind cpu/server.py, for testing this driver.
    "qwen3-0.6b": ModelSpec("qwen3-0.6b", "Qwen/Qwen3-0.6B", _distinct_seeds(grid_dense_bf16(28, r"self_attn\.(k|v)_proj\.weight$")[:6]),
                            "", 8, 40),
}


# --------------------------------------------------------------------------
# servers


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _wait_gpu_free(limit_mib: int = 2048, timeout: float = 90.0) -> None:
    """Block until the GPU holds less than `limit_mib`, so the next server's static allocation fits."""
    smi = shutil.which("nvidia-smi")
    if smi is None:
        time.sleep(1)
        return
    t0 = time.time()
    while time.time() - t0 < timeout:
        try:
            out = subprocess.run([smi, "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                                 capture_output=True, text=True, timeout=10).stdout
            if max(int(x) for x in out.split()) < limit_mib:
                return
        except Exception:  # noqa: BLE001
            return
        time.sleep(2)
    print(f"[study] warning: GPU memory still in use after {timeout:g}s", flush=True)


class Server:
    """One engine process; stop() kills its whole process group and waits for the port to close."""

    def __init__(self, engine: str, model_dir: Path, extra: List[str], env: Dict[str, str], log: Path,
                 spec: ModelSpec):
        self.engine, self.model_dir, self.extra, self.env, self.log, self.spec = engine, model_dir, extra, env, log, spec
        self.port = _free_port()
        self.url = f"http://127.0.0.1:{self.port}"
        self.proc: Optional[subprocess.Popen] = None

    def command(self) -> List[str]:
        if self.engine == "sglang":
            return [sys.executable, "-m", "sglang.launch_server", "--model-path", str(self.model_dir),
                    "--host", "127.0.0.1", "--port", str(self.port), "--served-model-name", "m",
                    "--random-seed", "12345", "--mem-fraction-static", "0.8", "--cuda-graph-max-bs-decode", "32",
                    "--context-length", "8192", *self.spec.sglang_args, *self.extra]
        if self.engine == "vllm":
            # `python -m vllm.entrypoints.openai.api_server` is deprecated in 0.33; `vllm serve` is the documented CLI
            vllm = shutil.which("vllm") or str(Path(sys.executable).parent / "vllm")
            return [vllm, "serve", str(self.model_dir),
                    "--host", "127.0.0.1", "--port", str(self.port), "--served-model-name", "m", "--seed", "0",
                    "--gpu-memory-utilization", "0.8", "--max-model-len", "8192", *self.spec.vllm_args, *self.extra]
        if self.engine == "cpu":
            return [sys.executable, "-I", str(HERE / "cpu" / "server.py"), "--model-path", str(self.model_dir),
                    "--port", str(self.port), "--served-model-name", "m", *self.extra]
        raise ValueError(self.engine)

    def start(self, timeout: float) -> float:
        self.log.parent.mkdir(parents=True, exist_ok=True)
        t0 = time.time()
        fh = open(self.log, "w")
        self.proc = subprocess.Popen(self.command(), stdout=fh, stderr=subprocess.STDOUT, start_new_session=True,
                                     env={**os.environ, **self.env})
        deadline = t0 + timeout
        while True:
            if self.proc.poll() is not None:
                raise RuntimeError(f"server exited with {self.proc.returncode}; see {self.log}")
            try:
                wait_for_ready(self.url, min(30.0, max(1.0, deadline - time.time())), interval_s=2.0)
                return time.time() - t0
            except TimeoutError:
                if time.time() > deadline:
                    self.stop()
                    raise RuntimeError(f"server not ready after {timeout:g}s; see {self.log}")

    def stop(self) -> None:
        if self.proc is None:
            return
        try:
            os.killpg(self.proc.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
        try:
            self.proc.wait(timeout=60)
        except subprocess.TimeoutExpired:
            os.killpg(self.proc.pid, signal.SIGKILL)
            self.proc.wait(timeout=30)
        self.proc = None
        _wait_gpu_free()


# --------------------------------------------------------------------------
# the study


@dataclass
class Plan:
    calib: int = 32  # the CPU study: 16 runs missed a rare setting on MXFP4 (3/60 false alarms), 32 did not
    heldout: int = 24
    restarts: int = 2
    configs: bool = True
    deterministic: bool = True
    faults: bool = True
    load_captures: int = 1  # perturbed captures per fault / restart / config arm
    start_timeout: float = 900.0


QUICK = Plan(calib=4, heldout=3, restarts=1, configs=False, deterministic=False, faults=True, load_captures=1,
             start_timeout=300.0)


def config_arms(engine: str, spec: ModelSpec) -> List[tuple]:
    if engine == "sglang":
        # SGLang 0.5.20 spellings (checked in the release's arg_groups): --disable-cuda-graph is a deprecated alias
        arms = [("eager", ["--cuda-graph-backend-decode", "disabled", "--cuda-graph-backend-prefill", "disabled"], {}),
                ("chunk512", ["--chunked-prefill-size", "512"], {})]
        if spec.alt_attention:
            arms.append((f"attn-{spec.alt_attention}", ["--attention-backend", spec.alt_attention], {}))
        return arms
    if engine == "vllm":
        return [("eager", ["--enforce-eager"], {}), ("chunk512", ["--max-num-batched-tokens", "512"], {})]
    return [("autotune", ["--autotune-seed", "7"], {})]


def deterministic_arm(engine: str):
    if engine == "sglang":
        return ["--enable-deterministic-inference"], {}
    if engine == "vllm":
        return [], {"VLLM_BATCH_INVARIANT": "1"}
    return None


def estimate(spec: ModelSpec, engine: str, plan: Plan) -> dict:
    """Server starts, captures and minutes the plan needs (an estimate, from spec.start_s and spec.capture_s)."""
    starts = 1 + plan.restarts
    caps = 1 + 1 + plan.calib + plan.heldout + 1 + plan.restarts * (1 + plan.load_captures)
    if plan.configs:
        n = len(config_arms(engine, spec))
        starts += n
        caps += n * (1 + plan.load_captures)
    if plan.deterministic and deterministic_arm(engine):
        starts += 1
        caps += 1 + 1 + 8 + 4
    if plan.faults:
        starts += len(spec.faults)
        caps += len(spec.faults) * (1 + plan.load_captures + 1)
    minutes = (starts * (spec.start_s + 15) + caps * spec.capture_s) / 60 + 3
    return {"server_starts": starts, "captures": caps, "minutes": minutes}


class Study:
    def __init__(self, spec: ModelSpec, engine: str, clean: Path, work: Path, out: Path, plan: Plan,
                 budget_min: float, on_record: Callable[[], None] = lambda: None):
        self.spec, self.engine, self.clean, self.work, self.out, self.plan = spec, engine, clean, work, out, plan
        self.deadline = time.time() + budget_min * 60
        self.on_record = on_record
        self.prompt_set, self.prompts = load_prompt_set(None)
        self.done = self._done()
        self.variant = out.name[: -len(".jsonl.gz")] if out.name.endswith(".jsonl.gz") else out.stem
        self.log_dir = out.parent / "logs" / self.variant

    def _done(self) -> set:
        if not self.out.exists():
            return set()
        ids = set()
        with gzip.open(self.out, "rt") as fh:
            for line in fh:
                try:
                    ids.add(json.loads(line)["spec"]["id"])
                except Exception:  # noqa: BLE001 -- a truncated last line from an interrupted run
                    pass
        return ids

    def _write(self, spec: dict, cap: Optional[dict], effect: dict, extra: Optional[dict] = None) -> None:
        spec = dict(spec, variant=self.variant)
        spec["id"] = f"{spec['variant']}:{spec['family']}:{spec['seed']}"
        rec = {"spec": spec, "capture": cap, "effect": effect, **(extra or {})}
        self.out.parent.mkdir(parents=True, exist_ok=True)
        with gzip.open(self.out, "at") as fh:
            fh.write(json.dumps(rec) + "\n")
        self.done.add(spec["id"])
        self.on_record()
        print(f"[study] {spec['id']} {spec.get('class', '')} {spec.get('magnitude', '')} "
              f"{'(server failed)' if cap is None else ''}", flush=True)

    def _ids(self, family, seed):
        return f"{self.variant}:{family}:{seed}"

    def _out_of_time(self) -> bool:
        if time.time() > self.deadline:
            print("[study] budget reached; stopping before the next server start", flush=True)
            return True
        return False

    def _server(self, model_dir, extra, env, name) -> Server:
        return Server(self.engine, model_dir, extra, env, self.log_dir / f"{name}.log", self.spec)

    def _heldout_nll(self, url) -> float:
        lps, _, _ = score(url, "m", HELD_OUT, 600.0)
        return -sum(lps) / len(lps)

    def _cap(self, url, **kw):
        return capture(url, "m", self.prompts, self.prompt_set, **kw)

    def _pert(self, url, seed, conc=None, fill=None):
        rng = random.Random(seed)
        return capture_perturbed(url, "m", self.prompts, self.prompt_set, seed=seed,
                                 concurrency=conc or rng.choice([2, 4, 8]), fillers=fill if fill is not None else rng.choice([0, 2, 6]))

    def run(self) -> None:
        p = self.plan
        ref_cap = None
        # 1. the reference server
        if self._ids("reference", 0) not in self.done or any(
                self._ids("request", k) not in self.done for k in range(1, p.calib + 1)):
            # One process, written all at once: a preempted container must not leave a reference
            # whose calibration came from a different process.
            if self.out.exists():
                self._drop_records()
            srv = self._server(self.clean, [], {}, "reference")
            t = srv.start(p.start_timeout)
            batch = []
            try:
                ref_cap = self._cap(srv.url)
                ref_cap["weights"] = weights_checksum(srv.url) if self.engine != "vllm" else None
                nll0 = self._heldout_nll(srv.url)
                batch.append(({"family": "reference", "seed": 0, "role": "reference"}, ref_cap,
                              {"kl_vs_ref": None, "heldout_nll": nll0, "start_s": t}))
                cal = calibrate(srv.url, "m", self.prompts, self.prompt_set, p.calib)
                batch.append(({"family": "repeat", "seed": 0, "role": "repeat"}, cal[0], {"kl_vs_ref": None}))
                for k, c in enumerate(cal[1:], 1):
                    batch.append(({"family": "request", "seed": k, "role": "calibration"}, c, {"kl_vs_ref": None}))
                for k in range(1, p.heldout + 1):
                    batch.append(({"family": "request", "seed": 100 + k, "role": "heldout"},
                                  self._pert(srv.url, 100 + k), {"kl_vs_ref": None}))
            finally:
                srv.stop()
            for rec in batch:
                self._write(*rec)
        nll0 = self._reference_nll()

        # 2. restarts of the clean model
        for r in range(1, p.restarts + 1):
            if self._ids("restart", r) in self.done or self._out_of_time():
                continue
            self._arm(self.clean, [], {}, f"restart-{r}", "restart", r, nll0)

        # 3. other engine settings
        if p.configs:
            for i, (name, extra, env) in enumerate(config_arms(self.engine, self.spec), 1):
                if self._ids(f"config:{name}", i) in self.done or self._out_of_time():
                    continue
                self._arm(self.clean, extra, env, name, f"config:{name}", i, nll0)

        # 5. faults (before the deterministic arm: they matter more if time runs short)
        if p.faults:
            for i, (cls, mag, fspec) in enumerate(self.spec.faults):
                if self._ids("fault", 5000 + i) in self.done or self._out_of_time():
                    continue
                self._fault(i, cls, mag, fspec, nll0)

        # 4. the batch-invariant mode, as its own reference
        det = deterministic_arm(self.engine)
        if p.deterministic and det and not self._out_of_time():
            Study(self.spec, self.engine, self.clean, self.work,
                  self.out.with_name(self.out.name.replace(".jsonl.gz", "-det.jsonl.gz")),
                  Plan(calib=8, heldout=4, restarts=0, configs=False, deterministic=False, faults=False,
                       start_timeout=p.start_timeout),
                  (self.deadline - time.time()) / 60, self.on_record)._run_det(det)

    def _run_det(self, det) -> None:
        extra, env = det
        if self._ids("reference", 0) in self.done:
            return
        srv = self._server(self.clean, extra, env, "deterministic")
        try:
            srv.start(self.plan.start_timeout)
        except RuntimeError as e:
            print(f"[study] deterministic arm skipped: {e}", flush=True)
            return
        try:
            ref = self._cap(srv.url)
            self._write({"family": "reference", "seed": 0, "role": "reference"}, ref,
                        {"kl_vs_ref": None, "heldout_nll": self._heldout_nll(srv.url)})
            cal = calibrate(srv.url, "m", self.prompts, self.prompt_set, self.plan.calib)
            self._write({"family": "repeat", "seed": 0, "role": "repeat"}, cal[0], {"kl_vs_ref": None})
            for k, c in enumerate(cal[1:], 1):
                self._write({"family": "request", "seed": k, "role": "calibration"}, c, {"kl_vs_ref": None})
            for k in range(1, self.plan.heldout + 1):
                self._write({"family": "request", "seed": 100 + k, "role": "heldout"},
                            self._pert(srv.url, 100 + k, conc=8, fill=6), {"kl_vs_ref": None})
        finally:
            srv.stop()

    def _drop_records(self) -> None:
        """Start over: a half-finished reference step is worthless, and so is everything measured against it."""
        self.out.rename(self.out.with_name(self.out.name + f".stale-{int(time.time())}"))
        self.done = set()

    def _reference_nll(self) -> float:
        with gzip.open(self.out, "rt") as fh:
            for line in fh:
                rec = json.loads(line)
                if rec["spec"]["family"] == "reference":
                    return rec["effect"]["heldout_nll"]
        raise RuntimeError("no reference record")

    def _reference_capture(self) -> dict:
        with gzip.open(self.out, "rt") as fh:
            for line in fh:
                rec = json.loads(line)
                if rec["spec"]["family"] == "reference":
                    return rec["capture"]
        raise RuntimeError("no reference record")

    def _arm(self, model_dir, extra, env, name, family, seed, nll0, fault=None) -> None:
        srv = self._server(model_dir, extra, env, name)
        base = {"family": family, "seed": seed, "role": "heldout" if fault is None else "fault"}
        if fault is not None:
            base.update(fault)
        try:
            t = srv.start(self.plan.start_timeout)
        except RuntimeError as e:
            # A server that will not come up on a corrupted checkpoint is a loud failure: detected.
            self._write(base, None, {"kl_vs_ref": None, "server_failed": True, "error": str(e)[:300]})
            return
        try:
            seq = self._cap(srv.url)
            if self.engine != "vllm":
                seq["weights"] = weights_checksum(srv.url)
            ref = self._reference_capture()
            changed = sum(a != b for pr, pc in zip(ref["prompts"], seq["prompts"])
                          for a, b in zip(pr["token_logprobs"], pc["token_logprobs"]))
            effect = {"kl_vs_ref": None, "heldout_nll_delta": self._heldout_nll(srv.url) - nll0,
                      "tokens_changed_ref_schedule": changed, "start_s": t}
            self._write(base, seq, effect)
            for j in range(self.plan.load_captures):
                self._write(dict(base, family=family + "-load", seed=seed * 100 + j,
                                 role="heldout" if fault is None else "fault-load"),
                            self._pert(srv.url, 7000 + seed * 100 + j), {"kl_vs_ref": None})
        finally:
            srv.stop()

    def _fault(self, i, cls, mag, fspec, nll0) -> None:
        dst = self.work / f"fault-{i}"
        if dst.exists():
            shutil.rmtree(dst)
        try:
            report = faults.inject(self.clean, dst, faults.FaultSpec.from_dict(fspec))
        except ValueError as e:
            print(f"[study] fault {i} {cls} {mag} not applicable: {e}", flush=True)
            return
        try:
            self._arm(dst, [], {}, f"fault-{i}", "fault", 5000 + i, nll0,
                      fault={"class": cls, "magnitude": mag, "fault": fspec, "fault_report": report.to_dict()})
        finally:
            shutil.rmtree(dst, ignore_errors=True)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--engine", choices=("sglang", "vllm", "cpu"), required=True)
    ap.add_argument("--model", choices=sorted(MODELS), required=True)
    ap.add_argument("--model-dir", type=Path, default=None, help="local checkpoint (default: models/<name> for cpu)")
    ap.add_argument("--out", type=Path, required=True, help="results directory")
    ap.add_argument("--work", type=Path, default=None, help="scratch for corrupted copies (same file system as the model)")
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--budget-min", type=float, default=150.0)
    ap.add_argument("--plan-only", action="store_true")
    ap.add_argument("--max-faults", type=int, default=None, help="run only the first N faults of the grid")
    args = ap.parse_args(argv)
    spec = MODELS[args.model]
    if args.max_faults is not None:
        spec.faults = spec.faults[: args.max_faults]
    plan = QUICK if args.quick else Plan()
    est = estimate(spec, args.engine, plan)
    print(f"[study] {args.model} on {args.engine}: {est['server_starts']} server starts, {est['captures']} captures, "
          f"~{est['minutes']:.0f} min (estimate)", flush=True)
    if args.plan_only:
        return 0
    model_dir = args.model_dir or (HERE / "models" / spec.hf_id.split("/")[1])
    work = args.work or (args.out / "work")
    work.mkdir(parents=True, exist_ok=True)
    out = args.out / f"{args.model}-{args.engine}.jsonl.gz"
    Study(spec, args.engine, model_dir, work, out, plan, args.budget_min).run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
