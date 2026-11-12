#!/usr/bin/env python3
"""One measured cold start of vLLM or SGLang: prepare weights and caches, launch,
wait for ready, optionally verify, shut down cleanly, and append the phases to a CSV.

    # Day 1, a cold run: weights evicted, every compile cache empty
    python run_once.py --csv results/baseline.csv --run cold --repeat 1 --tp 2 \
        --model /net/models/apertus-8b --evict /net/models/apertus-8b --caches fresh

    # Day 2, the staged arm: staging time is measured and counted
    python run_once.py --csv results/arms.csv --arm staged_shm --repeat 1 --tp 2 \
        --evict /net/models/apertus-8b-tp2 --stage-src /net/models/apertus-8b-tp2 \
        --stage-dest /dev/shm/apertus-8b-tp2 --caches fresh -- --load-format sharded_state

    # SGLang, caches kept between runs, a load test against the live server
    python run_once.py --engine sglang --csv results/sglang.csv --run warm --repeat 1 --tp 1 \
        --model /net/models/apertus-8b --caches persist:$HOME/sk-warm/sglang \
        --hold 'python replay.py --url "$BASE_URL" --model "$SERVED_NAME"' -- --cuda-graph-max-bs 32

The launch runs under `servekit profile`, so the phase table is servekit's own,
for either engine (vLLM: weight_loading = "Model loading took", max over ranks;
torch_compile; cuda_graph_capture. SGLang: torch_distributed_init; weight_loading;
the CUDA-graph captures; the warmup request. Plus the gaps between). Lines from the
engine's log add what that table does not break out, as `vllm:*` or `sglang:*`
phases. Every run, failed or not, also appends a JSON line with its settings,
status and log paths to results/runs.jsonl.

Everything after `--` is passed to the engine (`vllm serve` or SGLang's
`launch_server`) unchanged.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import re
import shlex
import shutil
import signal
import socket
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import evict as evictlib  # noqa: E402
import stage as stagelib  # noqa: E402

# Every compile/JIT cache a cold run must start without. #270 names the first
# four; the rest are defaults that would otherwise survive in $HOME.
CACHE_VARS = {
    "VLLM_CACHE_ROOT": "vllm",                 # torch.compile artifacts, model info
    "TRITON_CACHE_DIR": "triton",              # honoured by vLLM >= 0.33 instead of its own triton_cache
    "TORCHINDUCTOR_CACHE_DIR": "inductor",     # vLLM 0.33 re-points this under VLLM_CACHE_ROOT anyway
    "FLASHINFER_WORKSPACE_BASE": "flashinfer", # FlashInfer appends .cache/flashinfer itself
    "TILELANG_CACHE_DIR": "tilelang",
    "TVM_FFI_CACHE_DIR": "tvm-ffi",            # servekit's jit_cache.py persists this one too
    "DG_JIT_CACHE_DIR": "deepgemm",
    "XDG_CACHE_HOME": "xdg",                   # anything else that defaults to ~/.cache
    "CUDA_CACHE_PATH": "cuda",                 # the driver's PTX JIT cache, ~/.nv/ComputeCache
    # SGLang >= 0.5.20 (harmless for vLLM): CuTe AOT and DeepGEMM caches live under
    # SGLANG_CACHE_DIR, but its load_jit kernels (custom all-reduce included) do not.
    "SGLANG_CACHE_DIR": "sglang",
    "SGLANG_JIT_CACHE_DIR": "sglang-jit",
}
ENGINES = ("vllm", "sglang")
# The checkpoint is local: no Hub calls or usage pings inside the measured window.
QUIET_ENV = {
    "HF_HUB_OFFLINE": "1",
    "TRANSFORMERS_OFFLINE": "1",
    "VLLM_NO_USAGE_STATS": "1",
    "DO_NOT_TRACK": "1",
    "PYTHONUNBUFFERED": "1",
}
TEST_MODE = os.environ.get("SK_TEST_MODE") == "1"  # mock engine on a laptop: no eviction, no GPUs

# (phase, regex with the seconds as group 1, how to combine lines: per rank "max", or "sum" per rank then max)
LOG_PHASES = [
    ("loading_weights", re.compile(r"Loading weights took ([\d.]+) s"), "max"),
    ("model_loading", re.compile(r"Model loading took .*? and ([\d.]+) s"), "max"),
    ("prefetch", re.compile(r"Prefetching checkpoint files into page cache finished in ([\d.]+)\s*s"), "max"),
    ("dynamo_transform", re.compile(r"Dynamo bytecode transform time: ([\d.]+) s"), "max"),
    ("graph_compile", re.compile(r"Compiling a graph for .*? takes ([\d.]+) s"), "sum"),
    ("compiled_graph_cache_load", re.compile(r"Directly load the compiled graph\(s\) for .*? took ([\d.]+) s"), "sum"),
    ("torch_compile_total", re.compile(r"torch\.compile (?:took|takes) ([\d.]+) s in total"), "max"),
    ("initial_profile_run", re.compile(r"Initial profiling/warmup run took ([\d.]+) s"), "max"),
    ("cuda_graph_capture", re.compile(r"[Gg]raph capturing finished in ([\d.]+) secs?"), "max"),
    ("init_engine", re.compile(r"init engine \(profile, create kv cache, warmup model\) took ([\d.]+) s"), "max"),
    # `servekit launch` (vLLM): every rank's own load time from its plugin, where vLLM logs
    # local rank 0 only; and the stage, which an --overlap report leaves out of its phases
    ("sk_load_max_rank", re.compile(r"\[SERVEKIT\] weight load per rank: max ([\d.]+) s"), "max"),
    ("sk_stage", re.compile(r"\[SERVEKIT\] staged [\d.]+ GB in ([\d.]+) s"), "max"),
]
# SGLang reports these itself; servekit's table already has most of them, the
# split into decode, prefill and piecewise captures is what this adds.
SGLANG_LOG_PHASES = [
    ("dist_init", re.compile(r"Init torch distributed ends\. elapsed=([\d.]+) s"), "max"),
    ("load_weight", re.compile(r"Load weight end\. elapsed=([\d.]+) s"), "max"),
    ("decode_graphs", re.compile(
        r"Capture (?:cuda graph|target decode CUDA graph) end\. (?:Time elapsed: |elapsed=)([\d.]+) s"), "max"),
    ("prefill_graphs", re.compile(r"Capture target prefill CUDA graph end\. elapsed=([\d.]+) s"), "max"),
    ("piecewise_graphs", re.compile(r"Capture piecewise CUDA graph end\. Time elapsed: ([\d.]+) s"), "max"),
]
ENGINE_LOG_PHASES = {"vllm": LOG_PHASES, "sglang": SGLANG_LOG_PHASES}
_SOURCE = re.compile(r"\((?P<src>[A-Za-z][\w]*) pid=(?P<pid>\d+)\)")
# SGLang prefixes rank lines with "[2026-11-11 10:00:00 TP0]" (or "DP0 TP1", "TP0 EP0", ...)
_SGL_SOURCE = re.compile(r"\[\d{4}-\d\d-\d\d \d\d:\d\d:\d\d(?:\.\d+)?(?P<src>(?: [A-Z]{2}\d+)+)\]")
_SGL_KV = re.compile(r"KV Cache is allocated\. #tokens: (\d+)")
_VERSION = re.compile(r"(?:\bversion |Initializing a V1 LLM engine \(v)(\d+\.\d+\.\d+[\w.+-]*)")
_FS = re.compile(r"Filesystem type for checkpoints: (\S+?)\. Checkpoint size: ([\d.]+) GiB\. Available RAM: ([\d.]+) GiB")
_PREFETCH_ON = re.compile(r"Prefetching checkpoint files into page cache started")
_ERROR = re.compile(r"Traceback \(most recent call last\)|\bERROR\b|CUDA out of memory|Error:")


def now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def log(msg: str) -> None:
    print(f"[run_once {time.strftime('%H:%M:%S')}] {msg}", flush=True)


# --------------------------------------------------------------------------- caches

def cache_env(spec: str, fresh_base: Path, fresh_home: bool = False) -> Tuple[Dict[str, str], Optional[Path]]:
    """Env vars for `--caches`; returns (env, dir to delete afterwards or None).

    `fresh_home` also points HOME into the cache root, for caches that have no
    variable of their own and default to somewhere under ~ (an audit's safety net)."""
    if spec == "inherit":
        if fresh_home:
            raise ValueError("--fresh-home needs --caches fresh or persist:DIR")
        return {}, None
    if spec == "fresh":
        fresh_base.mkdir(parents=True, exist_ok=True)
        root = Path(tempfile.mkdtemp(prefix="cold-", dir=fresh_base))
        cleanup: Optional[Path] = root
    elif spec.startswith("persist:"):
        root = Path(spec.split(":", 1)[1]).expanduser()
        cleanup = None
    else:
        raise ValueError(f"--caches must be fresh, persist:DIR or inherit, not {spec!r}")
    env = {}
    for var, sub in CACHE_VARS.items():
        d = root / sub
        d.mkdir(parents=True, exist_ok=True)
        env[var] = str(d)
    if fresh_home:
        (root / "home").mkdir(parents=True, exist_ok=True)
        env["HOME"] = str(root / "home")
    return env, cleanup


def engine_command(args, extra: List[str]) -> List[str]:
    """The engine's own launch command, before servekit wraps it."""
    if args.engine == "vllm":
        cmd = [args.vllm_bin, "serve", str(args.model), "--tensor-parallel-size", str(args.tp)]
    else:
        cmd = [*shlex.split(args.sglang_cmd), "--model-path", str(args.model), "--tp-size", str(args.tp)]
    cmd += ["--host", args.host, "--port", str(args.port), "--served-model-name", args.served_model_name, *extra]
    return [*shlex.split(args.engine_prefix), *cmd] if args.engine_prefix else cmd


def parse_env_pairs(pairs: List[str]) -> Dict[str, str]:
    out = {}
    for p in pairs:
        if "=" not in p or p.startswith("="):
            raise ValueError(f"--env takes KEY=VALUE, not {p!r}")
        k, v = p.split("=", 1)
        out[k] = v
    return out


def run_hold(cmd: str, env: Dict[str, str], log_path: Path, timeout: float) -> Tuple[int, float]:
    """Run a command against the live server (a load test, a correctness check),
    with BASE_URL, SERVED_NAME, RUN_ID and RUN_LOG_DIR set. Returns (exit code, seconds)."""
    t0 = time.time()
    log(f"hold: {cmd}")
    try:
        with open(log_path, "w") as f:
            p = subprocess.run(["bash", "-c", cmd], stdout=f, stderr=subprocess.STDOUT, env=env, timeout=timeout)
        rc = p.returncode
    except subprocess.TimeoutExpired:
        rc = 124
    secs = time.time() - t0
    log(f"hold finished with exit code {rc} after {secs:.1f} s (output: {log_path})")
    return rc, secs


# --------------------------------------------------------------------------- machine state

def gpu_compute_pids() -> Optional[List[str]]:
    exe = shutil.which("nvidia-smi")
    if not exe:
        return None
    try:
        out = subprocess.run([exe, "--query-compute-apps=pid", "--format=csv,noheader"],
                             capture_output=True, text=True, timeout=30)
    except (OSError, subprocess.TimeoutExpired):
        return None
    if out.returncode != 0:
        return None
    return [ln.strip() for ln in out.stdout.splitlines() if ln.strip()]


def wait_gpus_idle(timeout: float) -> bool:
    deadline = time.time() + timeout
    while True:
        pids = gpu_compute_pids()
        if pids is None or not pids:
            return True
        if time.time() > deadline:
            log(f"GPUs still busy (pids {', '.join(pids)})")
            return False
        time.sleep(2)


def port_free(host: str, port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            s.bind((host, port))
            return True
        except OSError:
            return False


def wait_port_free(host: str, port: int, timeout: float) -> bool:
    deadline = time.time() + timeout
    while not port_free(host, port):
        if time.time() > deadline:
            return False
        time.sleep(1)
    return True


# --------------------------------------------------------------------------- HTTP

def http_ok(url: str, timeout: float = 2.0) -> bool:
    try:
        with urllib.request.urlopen(url, timeout=timeout) as r:
            return r.status == 200
    except (urllib.error.URLError, OSError, ValueError):
        return False


def completion(base: str, model: str, prompt: str, max_tokens: int, timeout: float = 120.0) -> dict:
    body = json.dumps({"model": model, "prompt": prompt, "max_tokens": max_tokens, "temperature": 0.0}).encode()
    req = urllib.request.Request(f"{base}/v1/completions", data=body, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


# --------------------------------------------------------------------------- process control

def group_alive(pgid: int) -> bool:
    try:
        os.killpg(pgid, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True


def child_pids(pid: int) -> Optional[Tuple[List[int], List[int]]]:
    """(live, zombie) direct children of `pid`, or None if that cannot be told.

    A crashed engine stays a zombie child of servekit, which never reaps it while
    it is blocked reading the pipe that the engine's orphans still hold."""
    exe = shutil.which("ps")
    if not exe:
        return None
    try:
        out = subprocess.run([exe, "-A", "-o", "pid=,ppid=,stat="], capture_output=True, text=True, timeout=10)
    except (OSError, subprocess.TimeoutExpired):
        return None
    if out.returncode != 0:
        return None
    live, zombies = [], []
    for line in out.stdout.splitlines():
        parts = line.split()
        if len(parts) >= 3 and parts[1] == str(pid):
            (zombies if parts[2].startswith("Z") else live).append(int(parts[0]))
    return live, zombies


def stop(proc: subprocess.Popen, host: str, port: int, grace: float = 60.0) -> float:
    """SIGTERM the wrapper (servekit forwards it to vLLM, which stops its own workers),
    then clear anything left in the process group. Returns seconds taken.

    servekit only exits once every process holding its output pipe is gone, so a
    straggler (a compile worker, an orphaned rank) would keep it up for the whole
    grace period: once vLLM's port has closed, give it 10 s, then kill the group.
    """
    t0 = time.time()
    if proc.poll() is None:
        proc.send_signal(signal.SIGTERM)
        closed_at: Optional[float] = None
        while time.time() - t0 < grace and proc.poll() is None:
            if closed_at is None and port_free(host, port):
                closed_at = time.time()
            if closed_at is not None and time.time() - closed_at > 10:
                log("server port closed but its process group lingers; killing the group")
                break
            time.sleep(0.5)
    pgid = proc.pid  # start_new_session=True made it the group leader
    for sig, wait in ((signal.SIGTERM, 15.0), (signal.SIGKILL, 10.0)):
        if not group_alive(pgid):
            break
        try:
            os.killpg(pgid, sig)
        except ProcessLookupError:
            break
        except PermissionError:
            pass  # macOS refuses when a member is a zombie, yet signals the rest
        deadline = time.time() + wait
        while time.time() < deadline:
            proc.poll()  # reap the leader once it exits, or its zombie keeps the group "alive"
            if not group_alive(pgid):
                break
            time.sleep(0.5)
    if proc.poll() is None:
        proc.kill()
        proc.wait()
    return time.time() - t0


def wait_ready(proc: subprocess.Popen, t0: float, health_url: str, report: Optional[Path],
               timeout: float) -> Tuple[Optional[float], str]:
    """Seconds from launch to the first HTTP 200 on /health, and why the wait ended.

    With servekit in front, a crashed engine can leave children holding servekit's
    pipe, and servekit then waits out its whole --timeout: so once servekit has had
    a child, losing it means the engine died, and the wait ends there.
    """
    ready_at: Optional[float] = None
    had_child, last_check = False, 0.0
    while True:
        rc = proc.poll()
        if rc is not None:
            return None, f"process exited with code {rc} before it was ready"
        now = time.time()
        if report is not None and ready_at is None and now - last_check > 1:
            last_check = now
            kids = child_pids(proc.pid)
            if kids is not None:
                live, zombies = kids
                had_child = had_child or bool(live or zombies)
                if had_child and not live:
                    return None, "the engine process exited before it was ready (see the log)"
        if ready_at is None and http_ok(health_url):
            ready_at = now
        if ready_at is not None:
            if report is None or report.exists():
                return ready_at - t0, "ready"
            if now - ready_at > 30:
                return ready_at - t0, "ready, but servekit wrote no report within 30 s"
        if now - t0 > timeout:
            return None, f"not ready after {timeout:.0f} s"
        time.sleep(0.25)


# --------------------------------------------------------------------------- log parsing

def parse_vllm_log(path: Path) -> Tuple[Dict[str, float], dict]:
    return parse_engine_log(path, "vllm")


def parse_engine_log(path: Path, engine: str = "vllm") -> Tuple[Dict[str, float], dict]:
    """Engine-reported phases (combined over ranks) and run facts from a vLLM or SGLang log."""
    table = ENGINE_LOG_PHASES[engine]
    per_src: Dict[str, Dict[str, List[float]]] = defaultdict(lambda: defaultdict(list))
    facts: dict = {"errors": []}
    text = path.read_text(errors="replace") if path.exists() else ""
    for line in text.splitlines():
        if engine == "sglang":
            m = _SGL_SOURCE.search(line)
            src = m.group("src").strip() if m else "main"
        else:
            m = _SOURCE.search(line)
            src = f"{m.group('src')}:{m.group('pid')}" if m else "main"
        for name, rx, _ in table:
            hit = rx.search(line)
            if hit:
                per_src[name][src].append(float(hit.group(1)))
                break
        if engine == "vllm":
            if "vllm_version" not in facts:
                v = _VERSION.search(line)
                if v:
                    facts["vllm_version"] = v.group(1)
            fs = _FS.search(line)
            if fs:
                facts["checkpoint_fs"] = fs.group(1)
                facts["checkpoint_gib"] = float(fs.group(2))
                facts["available_ram_gib"] = float(fs.group(3))
            if _PREFETCH_ON.search(line):
                facts["prefetch_started"] = True
        else:
            kv = _SGL_KV.search(line)
            if kv:
                facts["kv_cache_tokens"] = int(kv.group(1))
        if _ERROR.search(line) and len(facts["errors"]) < 8:
            facts["errors"].append(line.strip()[:300])
    phases: Dict[str, float] = {}
    for name, _, how in table:
        by_src = per_src.get(name)
        if not by_src:
            continue
        combine = sum if how == "sum" else max
        # the slowest rank decides when a TP-parallel phase is over (servekit takes the max too)
        phases[name] = round(max(combine(vals) for vals in by_src.values()), 3)
        if len(by_src) > 1:
            facts.setdefault("ranks_reporting", {})[name] = len(by_src)
    return phases, facts


# --------------------------------------------------------------------------- results

def append_csv(path: Path, header: List[str], rows: List[list]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and path.stat().st_size > 0:
        with open(path, newline="") as f:
            existing = next(csv.reader(f), [])
        if existing != header:
            raise ValueError(f"{path} has columns {existing}, expected {header}; use another --csv")
        new = False
    else:
        new = True
    with open(path, "a", newline="") as f:
        w = csv.writer(f)
        if new:
            w.writerow(header)
        w.writerows(rows)


def run_verify(servekit: str, base: str, mode: str, ref: Path, out_json: Path, log_path: Path) -> Tuple[int, dict]:
    cmd = [servekit, "verify", "--url", base, f"--{mode}", str(ref), "--wait-ready", "120", "--timeout", "300"]
    if mode == "reference":
        cmd += ["--out", str(out_json)]
        # servekit's defaults are 1e-6 (effectively bitwise); loosen only if default_net fails against itself
        for var, flag in (("VERIFY_TOKEN_TOL", "--token-tol"), ("VERIFY_NLL_TOL", "--nll-tol")):
            if os.environ.get(var):
                cmd += [flag, os.environ[var]]
    log(" ".join(cmd))
    try:
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=1800)
        rc, output = p.returncode, p.stdout + p.stderr
    except subprocess.TimeoutExpired as e:
        rc, output = 124, f"verify timed out: {e}"
    log_path.write_text(output)
    parsed = {}
    if mode == "reference" and out_json.exists():
        try:
            parsed = json.loads(out_json.read_text())
        except json.JSONDecodeError:
            pass
    tail = [ln for ln in output.splitlines() if ln.strip()][-3:]
    for ln in tail:
        log(f"verify: {ln}")
    return rc, parsed


# --------------------------------------------------------------------------- main

def parse_args(argv: List[str]):
    if "--" in argv:
        i = argv.index("--")
        argv, extra = argv[:i], argv[i + 1:]
    else:
        extra = []
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    who = ap.add_mutually_exclusive_group(required=True)
    who.add_argument("--run", help="run type for results/baseline.csv: cold, warm_weights, warm, ...")
    who.add_argument("--arm", help="arm name for results/arms.csv")
    ap.add_argument("--repeat", type=int, required=True)
    ap.add_argument("--tp", type=int, required=True)
    ap.add_argument("--engine", choices=ENGINES, default=os.environ.get("ENGINE", "vllm"))
    ap.add_argument("--model", help="path given to the engine (defaults to --stage-dest)")
    ap.add_argument("--model-name", default=os.environ.get("MODEL_NAME"), help="the `model` CSV column")
    ap.add_argument("--served-model-name", default=os.environ.get("SERVED_NAME", "model"))
    ap.add_argument("--csv", type=Path, help="append phase rows here")
    ap.add_argument("--no-record", action="store_true", help="measure but write no CSV rows (priming runs)")
    ap.add_argument("--evict", type=Path, action="append", default=[], help="evict these paths first (repeatable)")
    ap.add_argument("--drop-caches", action="store_true", help="also drop the whole page cache (root)")
    ap.add_argument("--prewarm", type=Path, action="append", default=[], help="read these into the page cache first")
    ap.add_argument("--caches", default="fresh", help="fresh | persist:DIR | inherit (default fresh)")
    ap.add_argument("--fresh-cache-base", type=Path, default=Path(os.environ.get("FRESH_CACHE_BASE", "/tmp/sk-caches")))
    ap.add_argument("--keep-caches", action="store_true", help="keep fresh cache dirs (to inspect what was written)")
    ap.add_argument("--fresh-home", action="store_true",
                    help="also point HOME into the cache root, so caches without a variable start empty too")
    ap.add_argument("--env", action="append", default=[], metavar="KEY=VALUE",
                    help="extra environment for the engine (repeatable; applied last, so it can override a cache dir)")
    ap.add_argument("--stage-src", type=Path)
    ap.add_argument("--stage-dest", type=Path)
    ap.add_argument("--stage-readers", type=int, default=int(os.environ.get("STAGE_READERS", 16)))
    ap.add_argument("--stage-block-mib", type=int, default=int(os.environ.get("STAGE_BLOCK_MIB", 16)))
    ap.add_argument("--keep-staged", action="store_true", help="leave the staged copy in place after the run")
    ap.add_argument("--verify-record", type=Path, help="record a servekit verify reference from this server")
    ap.add_argument("--verify-reference", type=Path, help="check this server against a servekit verify reference")
    ap.add_argument("--verify-csv", type=Path, default=None, help="append the verify outcome here")
    ap.add_argument("--no-smoke", action="store_true", help="skip the one-token request after ready")
    ap.add_argument("--hold", metavar="CMD",
                    help="after ready (and verify), run this shell command against the live server, then stop it; "
                         "it gets BASE_URL, SERVED_NAME, RUN_ID and RUN_LOG_DIR")
    ap.add_argument("--hold-timeout", type=float, default=3600.0)
    ap.add_argument("--host", default="127.0.0.1", help="bind address (keep it off the public interface)")
    ap.add_argument("--port", type=int, default=int(os.environ.get("PORT", 8000)))
    ap.add_argument("--timeout", type=float, default=float(os.environ.get("READY_TIMEOUT", 1200)))
    ap.add_argument("--no-servekit", action="store_true", help="launch the engine directly (phases from its log only)")
    ap.add_argument("--vllm-bin", default=os.environ.get("VLLM_BIN", "vllm"))
    ap.add_argument("--sglang-cmd", default=os.environ.get("SGLANG_CMD", "python3 -m sglang.launch_server"),
                    help="how to start SGLang's server (shell-split)")
    ap.add_argument("--engine-prefix", default=os.environ.get("ENGINE_PREFIX", ""),
                    help="a command put in front of the engine, inside servekit (e.g. 'strace -f -o x.txt')")
    ap.add_argument("--servekit-bin", default=os.environ.get("SERVEKIT_BIN", "servekit"))
    ap.add_argument("--servekit-launch", type=Path, metavar="ARTIFACT",
                    help="run under `servekit launch --servekit-artifact-path ARTIFACT` instead of `servekit "
                         "profile`: servekit stages the presharded artifact into /dev/shm itself (vLLM only)")
    ap.add_argument("--servekit-overlap", action="store_true", help="with --servekit-launch: add --overlap")
    ap.add_argument("--servekit-shm", type=Path, default=Path(os.environ.get("SERVEKIT_SHM", "/dev/shm/servekit")),
                    help="where servekit launch stages (its DEFAULT_ROOT), cleaned after each run")
    ap.add_argument("--servekit-cache-root", type=Path,
                    default=Path(os.environ.get("SERVEKIT_CACHE_ROOT", "/tmp/servekit/caches")),
                    help="servekit's node-local cache root (jit_cache.NODE_LOCAL_ROOT)")
    ap.add_argument("--results", type=Path, default=Path(os.environ.get("RESULTS", "results")))
    ap.add_argument("--dry-run", action="store_true", help="print what would run, run nothing")
    args = ap.parse_args(argv)
    if args.model is None:
        if args.stage_dest is None:
            ap.error("--model is required (or --stage-dest)")
        args.model = args.stage_dest
    if bool(args.stage_src) != bool(args.stage_dest):
        ap.error("--stage-src and --stage-dest go together")
    if args.verify_record and args.verify_reference:
        ap.error("--verify-record and --verify-reference are exclusive")
    if not args.no_record and args.csv is None:
        ap.error("--csv is required unless --no-record")
    if args.stage_dest:
        try:
            stagelib.check_dest(args.stage_dest)  # it is deleted after the run
        except ValueError as e:
            ap.error(str(e))
    if args.servekit_launch is not None:
        if args.engine != "vllm" or args.no_servekit or args.stage_src:
            ap.error("--servekit-launch is for vLLM, under servekit, without the harness's own staging")
    elif args.servekit_overlap:
        ap.error("--servekit-overlap needs --servekit-launch")
    if args.fresh_home and args.caches == "inherit":
        ap.error("--fresh-home needs --caches fresh or persist:DIR")
    try:
        args.env_extra = parse_env_pairs(args.env)
    except ValueError as e:
        ap.error(str(e))
    args.results = args.results.resolve()  # servekit writes its report from its own cwd
    return args, extra


def main(argv: Optional[List[str]] = None) -> int:
    args, extra = parse_args(sys.argv[1:] if argv is None else argv)
    label_col, label = ("arm", args.arm) if args.arm else ("run", args.run)
    model_name = args.model_name or Path(str(args.model)).name
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    eng = "" if args.engine == "vllm" else f"{args.engine}_"
    run_id = f"{stamp}_{eng}{label}_tp{args.tp}_r{args.repeat}"
    logs = args.results / "logs"
    log_path, report_path = logs / f"{run_id}.log", logs / f"{run_id}.profile.json"
    base = f"http://{args.host}:{args.port}"

    engine_cmd = engine_command(args, extra)
    if args.no_servekit:
        cmd = engine_cmd
    elif args.servekit_launch is not None:
        cmd = [args.servekit_bin, "launch", "--servekit-artifact-path", str(args.servekit_launch),
               "--out", str(report_path), "--timeout", str(int(args.timeout)),
               *(["--overlap"] if args.servekit_overlap else []), "--", *engine_cmd]
    else:
        cmd = [args.servekit_bin, "profile", "--out", str(report_path), "--timeout", str(int(args.timeout)),
               "--", *engine_cmd]

    meta: dict = {
        "run_id": run_id, "date": now_iso(), "engine": args.engine, label_col: label, "repeat": args.repeat,
        "tp": args.tp, "model": model_name, "model_path": str(args.model), "command": cmd, "caches": args.caches,
        "evict": [str(p) for p in args.evict], "prewarm": [str(p) for p in args.prewarm],
        "env_extra": args.env_extra, "test_mode": TEST_MODE, "recorded": not args.no_record, "log": str(log_path),
    }
    if args.dry_run:
        print(json.dumps(meta, indent=1))
        return 0
    logs.mkdir(parents=True, exist_ok=True)
    log(f"{label_col}={label} repeat={args.repeat} tp={args.tp}: {' '.join(cmd)}")

    if not wait_port_free(args.host, args.port, 30):
        log(f"port {args.port} is busy: is another server still running?")
        return 2
    if not TEST_MODE and not wait_gpus_idle(120):
        return 2

    # The server runs in its own session, so a dropped SSH connection (SIGHUP) or a kill
    # (SIGTERM) would orphan it on the GPUs; turn both into an exception so `finally` stops it.
    def _interrupt(signum, _frame):
        raise KeyboardInterrupt(f"signal {signum}")

    for sig in (signal.SIGTERM, signal.SIGHUP):
        signal.signal(sig, _interrupt)

    env_caches, fresh_dir = cache_env(args.caches, args.fresh_cache_base, args.fresh_home)
    env = {**os.environ, **QUIET_ENV, **env_caches, **args.env_extra}
    meta["cache_env"] = env_caches
    status, reason, rc = "failed", "", 1
    proc: Optional[subprocess.Popen] = None
    staging_s: Optional[float] = None
    try:
        # 1. weights: cold or warm, and check that it took
        if args.evict:
            if TEST_MODE:
                log("SK_TEST_MODE: eviction skipped, this run is NOT cold")
                meta["evict_skipped"] = True
            else:
                evictlib.evict(args.evict)
                if args.drop_caches:
                    evictlib.drop_all_caches()
                res, tot = evictlib.residency(args.evict)
                meta["resident_pct_after_evict"] = round(100.0 * res / tot, 2) if tot else 0.0
                log(f"evicted {', '.join(map(str, args.evict))}: {meta['resident_pct_after_evict']}% still resident")
                if meta["resident_pct_after_evict"] > 1.0:
                    raise RuntimeError("eviction left more than 1% resident; refusing to call this run cold")
        if args.prewarm:
            nbytes, secs = evictlib.warm(args.prewarm)
            res, tot = evictlib.residency(args.prewarm)
            meta["resident_pct_after_prewarm"] = round(100.0 * res / tot, 2) if tot else 0.0
            log(f"prewarmed {nbytes / 1e9:.1f} GB in {secs:.1f} s: {meta['resident_pct_after_prewarm']}% resident")

        # 2. staging, timed and counted
        if args.stage_src:
            st = stagelib.stage(args.stage_src, args.stage_dest, args.stage_readers, args.stage_block_mib, force=True)
            staging_s = st.seconds
            meta["staging"] = st.__dict__
            log(f"staged {st.bytes / 1e9:.2f} GB in {st.seconds:.2f} s ({st.gbps:.2f} GB/s, {st.readers} readers)")

        # servekit launch keeps compile caches beside the artifact and in a node-local copy,
        # and writes them back after a launch: a cold run must start without both.
        if args.servekit_launch is not None:
            kept = [args.servekit_launch / "servekit-caches", args.servekit_cache_root / args.servekit_launch.name]
            if args.caches == "fresh":
                for d in kept:
                    shutil.rmtree(d, ignore_errors=True)
                meta["servekit_caches"] = "cleared"
            else:
                meta["servekit_caches"] = "kept"
            shutil.rmtree(args.servekit_shm / args.servekit_launch.name, ignore_errors=True)

        # 3. launch and wait
        t0 = time.time()
        with open(log_path, "w") as logf:
            proc = subprocess.Popen(cmd, stdout=logf, stderr=subprocess.STDOUT, env=env, start_new_session=True)
        ready_s, reason = wait_ready(proc, t0, f"{base}/health", None if args.no_servekit else report_path,
                                     args.timeout + 30)
        meta["ready_http_s"] = round(ready_s, 3) if ready_s is not None else None
        log(f"{reason}" + (f" after {ready_s:.1f} s" if ready_s is not None else ""))

        if ready_s is not None:
            status, rc = "ok", 0
            if not args.no_smoke:
                t1 = time.time()
                try:
                    completion(base, args.served_model_name, "The capital of Switzerland is", 1)
                    meta["first_request_s"] = round(time.time() - t1, 3)
                except (urllib.error.URLError, OSError, ValueError) as e:
                    meta["first_request_error"] = str(e)[:300]
                    status, rc, reason = "failed", 1, f"server ready but a request failed: {e}"
            vmode = "record" if args.verify_record else "reference" if args.verify_reference else None
            if vmode and status == "ok":
                ref = args.verify_record or args.verify_reference
                vrc, vres = run_verify(args.servekit_bin, base, vmode, ref, logs / f"{run_id}.verify.json",
                                       logs / f"{run_id}.verify.txt")
                meta["verify"] = {"mode": vmode, "reference": str(ref), "exit_code": vrc,
                                  "passed": vres.get("passed") if vres else (vrc == 0),
                                  "worst_token_delta": vres.get("worst_token_delta"),
                                  "worst_nll_delta": vres.get("worst_nll_delta"),
                                  "failures": len(vres.get("failures", [])) if vres else None}
                if args.verify_csv:
                    append_csv(args.verify_csv,
                               ["date", "model", "tp", label_col, "repeat", "mode", "reference", "exit_code",
                                "passed", "worst_token_delta", "worst_nll_delta"],
                               [[meta["date"], model_name, args.tp, label, args.repeat, vmode, str(ref), vrc,
                                 meta["verify"]["passed"], meta["verify"]["worst_token_delta"],
                                 meta["verify"]["worst_nll_delta"]]])
                if vrc != 0:
                    rc = 3
            if args.hold and status == "ok":
                hold_out = logs / f"{run_id}.hold.txt"
                hold_env = {**os.environ, "BASE_URL": base, "SERVED_NAME": args.served_model_name,
                            "RUN_ID": run_id, "RUN_LOG_DIR": str(logs)}
                hrc, hsecs = run_hold(args.hold, hold_env, hold_out, args.hold_timeout)
                meta["hold"] = {"command": args.hold, "exit_code": hrc, "seconds": round(hsecs, 1),
                                "output": str(hold_out)}
                if hrc != 0 and rc == 0:
                    rc = 4
    except (OSError, ValueError, RuntimeError) as e:
        reason = f"{type(e).__name__}: {e}"
        log(reason)
    finally:
        for sig in (signal.SIGTERM, signal.SIGHUP):  # let the cleanup itself finish
            signal.signal(sig, signal.SIG_IGN)
        if proc is not None:
            served = meta.get("ready_http_s") is not None
            meta["shutdown_s"] = round(stop(proc, args.host, args.port, grace=60.0 if served else 5.0), 1)
            if not TEST_MODE:
                wait_gpus_idle(120)
            wait_port_free(args.host, args.port, 60)
        if fresh_dir is not None and not args.keep_caches:
            shutil.rmtree(fresh_dir, ignore_errors=True)
        if args.stage_dest and not args.keep_staged:
            shutil.rmtree(args.stage_dest, ignore_errors=True)
        if args.servekit_launch is not None and not args.keep_staged:
            # servekit frees it at ready; after a failed run it is left in place
            shutil.rmtree(args.servekit_shm / args.servekit_launch.name, ignore_errors=True)

    # 4. phases: servekit's table, vLLM's own lines, and the harness's wall clocks
    rows: List[Tuple[str, float]] = []
    report = None
    if report_path.exists():
        try:
            report = json.loads(report_path.read_text())
        except json.JSONDecodeError:
            report = None
    if report:
        meta["servekit_success"] = report.get("success")
        if status == "ok" and not report.get("success"):
            status, rc, reason = "failed", 1, "servekit saw no ready line"
        for p in report.get("phases", []):
            rows.append((p["name"], float(p["duration_s"])))
        if report.get("total_duration_s"):
            rows.append(("total", float(report["total_duration_s"])))
    log_phases, facts = parse_engine_log(log_path, args.engine)
    meta.update(facts)
    meta["log_phases"] = log_phases
    rows += [(f"{args.engine}:{k}", v) for k, v in log_phases.items()]
    if meta.get("ready_http_s") is not None:
        rows.append(("total_http", meta["ready_http_s"]))
    if staging_s is not None:
        rows.append(("staging", staging_s))
        total = report.get("total_duration_s") if report else meta.get("ready_http_s")
        if total:
            rows.append(("total_with_staging", round(staging_s + float(total), 3)))
    if meta.get("first_request_s") is not None:
        rows.append(("first_request", meta["first_request_s"]))

    meta.update({"status": status, "reason": reason, "phases": dict(rows)})
    with open(args.results / "runs.jsonl", "a") as f:
        f.write(json.dumps(meta) + "\n")
    if status == "ok" and not args.no_record:
        header = ["date", "model", "tp", label_col, "repeat", "phase", "seconds"]
        append_csv(args.csv, header, [[meta["date"], model_name, args.tp, label, args.repeat, n, v] for n, v in rows])
    got = dict(rows)
    summary = {}
    for keys in (("weight_loading", "vllm:model_loading", "sglang:load_weight"),
                 ("torch_compile", "vllm:torch_compile_total"),
                 ("cuda_graph_capture", "vllm:cuda_graph_capture", "sglang:decode_graphs"),
                 ("prefill_cuda_graph_capture", "sglang:prefill_graphs"), ("staging",), ("total", "total_http"),
                 ("total_with_staging",)):
        k = next((k for k in keys if k in got), None)
        if k:
            summary[k] = got[k]
    log(f"{status}: " + ", ".join(f"{k}={v:.1f}s" for k, v in summary.items()) + (f" ({reason})" if status != "ok" else ""))
    if status != "ok" and meta.get("errors"):
        for e in meta["errors"][:3]:
            log(f"  log: {e}")
    return rc if status == "ok" else 1


if __name__ == "__main__":
    raise SystemExit(main())
