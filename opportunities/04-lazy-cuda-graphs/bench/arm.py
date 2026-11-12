"""Run one experiment arm end to end and return a JSON-serialisable record.

Timeline (t = seconds since the server process was started):
  t=0     start `launch_server`; a client sends one request immediately
          (time-to-first-token for a request arriving at launch) and, if
          configured, starts replaying trace arrivals at their offsets
  ready   GET /health returns 200 (time-to-ready)
  ...     replay finishes; lazy arms then wait (idle) until the controller
          logs that every planned bucket is captured, or a timeout
  then    optional closed-loop throughput, eager-penalty sweep, correctness
"""

from __future__ import annotations

import json
import os
import statistics
import subprocess
import threading
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence

from bench import client, logparse


def pct(values: Sequence[float], q: float) -> Optional[float]:
    xs = sorted(v for v in values if v is not None)
    if not xs:
        return None
    k = max(0, min(len(xs) - 1, int(round(q * (len(xs) - 1)))))
    return xs[k]


def gpu_memory_used_mib() -> Optional[List[int]]:
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=20, check=True,
        ).stdout
        return [int(x) for x in out.split()]
    except (OSError, subprocess.SubprocessError, ValueError):
        return None


class ServerProcess:
    """launch_server subprocess whose output is timestamped line by line."""

    def __init__(self, cmd: List[str], env: Dict[str, str], log_path: Path) -> None:
        self.t0 = time.monotonic()
        self.lines: List[tuple] = []
        self._log = open(log_path, "w")
        self.proc = subprocess.Popen(
            cmd, env={**os.environ, **env}, stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT, text=True, bufsize=1,
        )
        self._reader = threading.Thread(target=self._read, daemon=True)
        self._reader.start()

    def _read(self) -> None:
        for line in self.proc.stdout:
            t = time.monotonic() - self.t0
            self.lines.append((t, line))
            self._log.write(f"{t:10.3f} {line}")
            self._log.flush()

    def wait_for(self, needle: str, timeout_s: float) -> Optional[float]:
        deadline = time.monotonic() + timeout_s
        seen = 0
        while time.monotonic() < deadline:
            while seen < len(self.lines):
                t, line = self.lines[seen]
                seen += 1
                if needle in line:
                    return t
            if self.proc.poll() is not None:
                return None
            time.sleep(0.2)
        return None

    def stop(self) -> None:
        if self.proc.poll() is None:
            self.proc.terminate()
            try:
                self.proc.wait(timeout=60)
            except subprocess.TimeoutExpired:
                self.proc.kill()
        self._reader.join(timeout=10)
        self._log.close()


def server_command(spec: dict, *, launcher: List[str], port: int) -> List[str]:
    cmd = list(launcher) + [
        "--model-path", spec["model"],
        "--tp", str(spec.get("tp", 1)),
        "--host", "127.0.0.1",
        "--port", str(port),
    ]
    if spec.get("cuda_graph_config") is not None:
        cmd += ["--cuda-graph-config", json.dumps(spec["cuda_graph_config"])]
    cmd += list(spec.get("server_args", []))
    return cmd


def _replay_summary(results: List[client.RequestResult], t_ready: float) -> dict:
    ok = [r for r in results if r.error is None]
    before = [r for r in ok if r.t_intended < t_ready]
    after = [r for r in ok if r.t_intended >= t_ready]

    def stats(rs):
        return {
            "n": len(rs),
            "ttft_p50_s": pct([r.ttft_s for r in rs], 0.5),
            "ttft_p99_s": pct([r.ttft_s for r in rs], 0.99),
            "tpot_p50_s": pct([r.tpot_s for r in rs], 0.5),
            "tpot_p99_s": pct([r.tpot_s for r in rs], 0.99),
        }

    return {
        "requests": len(results),
        "errors": len(results) - len(ok),
        "all": stats(ok),
        "arrived_before_ready": stats(before),
        "arrived_after_ready": stats(after),
    }


def eager_sweep(base_url: str, cfg: dict) -> dict:
    """Prefill TTFT per prompt size and decode TPOT per batch size (medians)."""
    out = {"prefill_ttft_s": {}, "decode_tpot_s": {}}
    reps = int(cfg.get("reps", 5))
    for n in cfg.get("prefill_tokens", []):
        samples = []
        for r in range(reps + 1):  # first run discarded (warm-up, JIT)
            t0 = time.monotonic()
            res = client.stream_generate(
                base_url, input_ids=client.make_prompt(n, 31 * n + r),
                max_new_tokens=2, t0=t0, t_intended=0.0, rid=r,
            )
            if r > 0 and res.ttft_s is not None:
                samples.append(res.ttft_s)
        out["prefill_ttft_s"][str(n)] = {
            "median": statistics.median(samples) if samples else None,
            "min": min(samples) if samples else None,
            "max": max(samples) if samples else None,
            "n": len(samples),
        }
    for bs in cfg.get("decode_bs", []):
        medians = []
        for r in range(reps):
            t0 = time.monotonic()
            results: List[client.RequestResult] = []
            lock = threading.Lock()

            def one(i, r=r, t0=t0, bs=bs):
                res = client.stream_generate(
                    base_url, input_ids=client.make_prompt(128, 7 * bs + 1000 * r + i),
                    max_new_tokens=int(cfg.get("decode_tokens", 128)), t0=t0,
                    t_intended=0.0, rid=i,
                )
                with lock:
                    results.append(res)

            ths = [threading.Thread(target=one, args=(i,)) for i in range(bs)]
            for t in ths:
                t.start()
            for t in ths:
                t.join()
            tpots = [x.tpot_s for x in results if x.tpot_s is not None]
            if tpots:
                medians.append(statistics.median(tpots))
        out["decode_tpot_s"][str(bs)] = {
            "median": statistics.median(medians) if medians else None,
            "min": min(medians) if medians else None,
            "max": max(medians) if medians else None,
            "n": len(medians),
        }
    return out


def correctness(base_url: str, cfg: dict, server: ServerProcess, lazy: bool) -> dict:
    """Greedy outputs for fixed prompts, plus a prefix-cache reuse probe.

    Phase 1 sends the long prompt P (its KV enters the radix cache). Lazy arms
    then idle until every bucket is captured, so deferred prefill captures run
    while P's KV is cached. Phase 2 sends P + suffix (radix hit) and the other
    prompts. Compare across arms: any difference in phase 2 but not phase 1
    points at state the deferred captures overwrote.
    """
    out_tokens = int(cfg.get("output_tokens", 32))
    prompts = {str(n): client.make_prompt(n, 4242 + n) for n in cfg.get("prompt_tokens", [])}
    long_p = client.make_prompt(int(cfg.get("prefix_tokens", 3000)), 777)
    res = {"phase1": {}, "phase2": {}}
    res["phase1"]["prefix"] = client.generate_ids(base_url, input_ids=long_p, max_new_tokens=out_tokens)
    if lazy:
        res["waited_for_full_capture_s"] = server.wait_for(
            "all planned buckets captured", float(cfg.get("wait_s", 600))
        )
    else:
        time.sleep(float(cfg.get("idle_s", 5)))
    suffix = client.make_prompt(int(cfg.get("suffix_tokens", 200)), 999)
    res["phase2"]["prefix+suffix"] = client.generate_ids(
        base_url, input_ids=long_p + suffix, max_new_tokens=out_tokens
    )
    for name, p in prompts.items():
        res["phase2"][name] = client.generate_ids(base_url, input_ids=p, max_new_tokens=out_tokens)
    return res


def run_arm(spec: dict, *, launcher: List[str], out_dir: Path, port: int = 30000) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    tag = f"{spec['arm']}-tp{spec.get('tp', 1)}-r{spec.get('repeat', 0)}"
    base_url = f"http://127.0.0.1:{port}"
    record = {"spec": {k: v for k, v in spec.items() if k != "replay"}, "tag": tag}
    if spec.get("replay"):
        record["spec"]["replay_requests"] = len(spec["replay"]["arrivals_s"])
    server = ServerProcess(
        server_command(spec, launcher=launcher, port=port),
        {k: str(v) for k, v in spec.get("env", {}).items()},
        out_dir / f"{tag}.log",
    )
    t0 = server.t0
    launch: Dict[str, client.RequestResult] = {}
    replayed: List[client.RequestResult] = []
    threads = []
    lr = spec.get("launch_request")
    if lr:
        def send_launch():
            launch["r"] = client.stream_generate(
                base_url, input_ids=client.make_prompt(lr["prompt_tokens"], 1),
                max_new_tokens=lr["output_tokens"], t0=t0, t_intended=0.0, rid=-1,
                connect_timeout_s=spec.get("timeout_s", 2400),
            )
        threads.append(threading.Thread(target=send_launch))
    rp = spec.get("replay")
    if rp:
        scale = float(rp.get("time_scale", 1.0))

        def send_replay():
            replayed.extend(client.replay(
                base_url, arrivals_s=[a * scale for a in rp["arrivals_s"]],
                prompt_tokens=rp["prompt_tokens"], output_tokens=rp["output_tokens"],
                t0=t0, max_prompt=rp.get("max_prompt", 8192),
                max_output=rp.get("max_output", 256), seed=spec.get("repeat", 0),
            ))
        threads.append(threading.Thread(target=send_replay))
    for t in threads:
        t.start()
    try:
        t_ready = client.wait_ready(base_url, timeout_s=spec.get("timeout_s", 2400)) - t0
        record["time_to_ready_s"] = t_ready
        for t in threads:
            t.join()
        if "r" in launch:
            r = launch["r"]
            record["launch_request"] = {"ttft_s": r.ttft_s, "tpot_s": r.tpot_s, "error": r.error}
        if rp:
            record["replay"] = _replay_summary(replayed, t_ready)
            record["replay_requests_raw"] = client.result_dicts(replayed)
        record["gpu_mem_after_replay_mib"] = gpu_memory_used_mib()
        if spec.get("wait_full_capture_s"):
            record["full_capture_t"] = server.wait_for(
                "all planned buckets captured", float(spec["wait_full_capture_s"])
            )
            record["gpu_mem_after_full_capture_mib"] = gpu_memory_used_mib()
        if spec.get("throughput"):
            record["throughput"] = client.throughput(base_url, **spec["throughput"])
        if spec.get("eager_sweep"):
            record["eager_sweep"] = eager_sweep(base_url, spec["eager_sweep"])
        if spec.get("correctness"):
            record["correctness"] = correctness(
                base_url, spec["correctness"], server, lazy=spec["arm"] == "lazy"
            )
    except Exception as e:  # noqa: BLE001 - keep the partial record
        record["error"] = repr(e)
    finally:
        server.stop()
        record["log"] = logparse.parse(server.lines)
    (out_dir / f"{tag}.json").write_text(json.dumps(record, indent=1, default=str))
    return record
