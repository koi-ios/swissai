"""In-container driver for the presharding round-trip experiment (see MODAL.md).

For one model at TP=2 it runs, in this order (patches are applied to the
installed SGLang only in step 5, so steps 1-4 see SGLang exactly as shipped):

  1 default            default loader; records gold logprobs and a deep-state reference
  2 default_again      default loader again: the noise floor for both checks
  3 sharded_state_sglang   SGLang's own sharded_state save (CUDA graphs on, as in a
                       normal prepare) and reload; no servekit, no backports
  4 raw                servekit raw preshard (prepare records, serve replays), unpatched SGLang
  5 sharded_state_servekit [MLA only] servekit prepare after applying its backports
  6 timing             N cold-ish starts per correct arm: weight-load seconds and time to ready
  7 deepstate          every arm's deep-state dump against the reference, with arm 2 as noise filter

Every step records its outcome and keeps going, so one failure does not cost
the run. Results land in <results>/<model>/<run-id>/results.json.

    python modal_driver.py --model dsv2-lite --dry-run      # print the plan, run nothing
"""
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

PORT = 30123
HERE = Path(__file__).resolve().parent
SCRIPTS = HERE / "modal_scripts"
NO_GRAPHS = ["--disable-prefill-cuda-graph", "--disable-decode-cuda-graph"]
READY = "The server is fired up and ready to roll!"
LOAD_RE = re.compile(r"Load weight end\. elapsed=([\d.]+) s")

MODELS: Dict[str, Dict[str, Any]] = {
    # MLA (DeepseekV2AttentionMLA): servekit #19 / sglang #35702. BF16, 15.7B params, ~31 GB.
    "dsv2-lite": {
        "repo": "deepseek-ai/DeepSeek-V2-Lite",
        "ignore": [],
        "tp": 2,
        "args": ["--trust-remote-code", "--mem-fraction-static", "0.80"],
        "servekit_sharded_arm": True,
    },
    # MXFP4 experts on the triton_kernel MoE runner: servekit #9/#14, sglang #34448. ~13 GB.
    # No --moe-runner-backend: SGLang 0.5.20 picks triton_kernel itself below SM100.
    "gpt-oss-20b": {
        "repo": "openai/gpt-oss-20b",
        "ignore": ["original/*", "metal/*"],
        "tp": 2,
        "args": ["--mem-fraction-static", "0.80"],
        "servekit_sharded_arm": False,
    },
}


class Plan:
    """Runs commands, or prints them when dry."""

    def __init__(self, dry: bool, logs: Path):
        self.dry = dry
        self.logs = logs
        self.lines: List[str] = []

    def note(self, msg: str) -> None:
        print(f"[driver] {msg}", flush=True)
        self.lines.append(msg)

    def run(self, cmd: List[str], log: str, env: Optional[Dict[str, str]] = None, timeout: float = 5400) -> Dict[str, Any]:
        self.note(("DRY " if self.dry else "") + " ".join(map(str, cmd)) + (f"   [env {env}]" if env else ""))
        if self.dry:
            return {"rc": 0, "seconds": 0.0, "log": log}
        full_env = {**os.environ, **(env or {})}
        t0 = time.time()
        with open(self.logs / log, "w") as f:
            try:
                rc = subprocess.call([str(c) for c in cmd], stdout=f, stderr=subprocess.STDOUT, env=full_env, timeout=timeout)
            except subprocess.TimeoutExpired:
                rc = -9
        return {"rc": rc, "seconds": round(time.time() - t0, 2), "log": log}


def _tail(path: Path, n: int = 25, pattern: str = r"Error|error|Traceback|KeyError|Missing keys") -> List[str]:
    try:
        lines = path.read_text(errors="replace").splitlines()
    except OSError:
        return []
    hits = [l for l in lines if re.search(pattern, l)]
    return (hits[-n:] if hits else lines[-n:])


class Server:
    """sglang.launch_server in its own process group, up until stopped."""

    def __init__(self, plan: Plan, model_path: Path, tp: int, args: List[str], log: str,
                 env: Optional[Dict[str, str]] = None, timeout: float = 1800):
        self.plan, self.log = plan, plan.logs / log
        self.cmd = [sys.executable, "-m", "sglang.launch_server", "--model-path", str(model_path),
                    "--tensor-parallel-size", str(tp), "--host", "127.0.0.1", "--port", str(PORT), *NO_GRAPHS, *args]
        self.env, self.timeout = env or {}, timeout
        self.proc: Optional[subprocess.Popen] = None
        self.result: Dict[str, Any] = {"log": log}

    def __enter__(self) -> "Server":
        self.plan.note(("DRY " if self.plan.dry else "") + "serve: " + " ".join(self.cmd) + (f"   [env {self.env}]" if self.env else ""))
        if self.plan.dry:
            self.result.update(ready=True, ready_s=0.0, weight_load_s=0.0)
            return self
        f = open(self.log, "w")
        t0 = time.time()
        self.proc = subprocess.Popen(self.cmd, stdout=f, stderr=subprocess.STDOUT, env={**os.environ, **self.env},
                                     start_new_session=True)
        ready = False
        while time.time() - t0 < self.timeout:
            text = self.log.read_text(errors="replace")
            if READY in text:
                ready = True
                break
            if self.proc.poll() is not None:
                break
            time.sleep(1.0)
        text = self.log.read_text(errors="replace")
        loads = [float(x) for x in LOAD_RE.findall(text)]
        self.result.update(
            ready=ready,
            ready_s=round(time.time() - t0, 2) if ready else None,
            weight_load_s=max(loads) if loads else None,
            weight_load_s_per_rank=loads,
        )
        if not ready:
            self.result["error"] = _tail(self.log)
        return self

    def __exit__(self, *exc) -> None:
        if self.proc is None:
            return
        try:
            os.killpg(self.proc.pid, signal.SIGTERM)
            self.proc.wait(timeout=60)
        except (ProcessLookupError, subprocess.TimeoutExpired):
            try:
                os.killpg(self.proc.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
        time.sleep(5)  # let the GPUs drain before the next server


def verify(plan: Plan, mode: str, gold: Path, out: Path, log: str) -> Dict[str, Any]:
    flag = "--record" if mode == "record" else "--reference"
    cmd = ["servekit", "verify", "--url", f"http://127.0.0.1:{PORT}", flag, str(gold), "--wait-ready", "120"]
    if mode != "record":
        cmd += ["--out", str(out)]
    r = plan.run(cmd, log, timeout=1800)
    if not plan.dry and mode != "record" and out.is_file():
        data = json.loads(out.read_text())
        r["passed"] = data.get("passed")
        r["detail"] = {k: v for k, v in data.items() if k != "prompts"}
    elif mode != "record":
        r["passed"] = r["rc"] == 0
    return r


def dir_bytes(path: Path, suffix: str = ".safetensors") -> int:
    return sum(p.stat().st_size for p in path.rglob(f"*{suffix}")) if path.exists() else 0


def warm(path: Path, plan: Plan) -> float:
    """Read every file once so the next load finds it in the page cache (servekit stages to RAM)."""
    if plan.dry:
        return 0.0
    t0 = time.time()
    for p in sorted(path.rglob("*")):
        if p.is_file():
            with open(p, "rb") as f:
                while f.read(64 << 20):
                    pass
    return round(time.time() - t0, 2)


def run(model: str, hf_root: Path, work: Path, results_root: Path, runs: int = 3, dry: bool = False,
        steps: Optional[List[str]] = None) -> Dict[str, Any]:
    cfg = MODELS[model]
    run_id = time.strftime("%Y%m%d-%H%M%S")
    out = results_root / model / run_id
    logs = out / "logs"
    for d in (out, logs):
        d.mkdir(parents=True, exist_ok=True)
    plan = Plan(dry, logs)
    steps = steps or ["default", "default_again", "sharded_state_sglang", "raw", "sharded_state_servekit", "timing", "deepstate"]
    tp, args = cfg["tp"], cfg["args"]
    src = work / "src" / model
    art = work / "artifacts" / model
    ds = out / "deepstate"
    gold = out / "gold.json"
    res: Dict[str, Any] = {"model": model, "repo": cfg["repo"], "tp": tp, "args": args, "run_id": run_id,
                           "dry_run": dry, "arms": {}, "timing": {}, "deepstate": {}}

    def save() -> None:
        res["plan"] = plan.lines
        (out / "results.json").write_text(json.dumps(res, indent=2) + "\n")

    # ---------------------------------------------------------------- environment
    if not dry:
        env_out = subprocess.run(
            [sys.executable, "-c", "import torch, sglang; print(sglang.__version__, torch.__version__, torch.version.cuda)"],
            capture_output=True, text=True)
        smi = subprocess.run(["nvidia-smi", "--query-gpu=name,driver_version,memory.total", "--format=csv,noheader"],
                             capture_output=True, text=True)
        res["env"] = {"sglang_torch_cuda": env_out.stdout.strip(), "gpus": smi.stdout.strip().splitlines(),
                      "servekit_commit": os.environ.get("SERVEKIT_COMMIT", "unknown")}

    # ---------------------------------------------------------------- weights onto local disk
    snapshot = hf_root / model
    plan.note(f"copy {snapshot} -> {src}")
    if not dry and not src.exists():
        t0 = time.time()
        shutil.copytree(snapshot, src, symlinks=False, ignore=shutil.ignore_patterns(*cfg["ignore"], ".cache"))
        res["copy_s"] = round(time.time() - t0, 2)
    res["checkpoint_bytes"] = dir_bytes(src)
    save()

    def deep_env(arm: str) -> Dict[str, str]:
        return {"SERVEKIT_DEEPSTATE_DIR": str(ds / arm), "SERVEKIT_DEEPSTATE_DETERMINISTIC": "1"}

    def serve_and_check(arm: str, path: Path, extra: List[str], record: bool = False) -> Dict[str, Any]:
        r: Dict[str, Any] = {}
        with Server(plan, path, tp, args + extra, f"{arm}.serve.log", env=deep_env(arm)) as server:
            r["serve"] = server.result
            if server.result.get("ready"):
                r["verify"] = verify(plan, "record" if record else "reference", gold, out / f"verify-{arm}.json", f"{arm}.verify.log")
        r["status"] = ("ok" if (record or r.get("verify", {}).get("passed")) else "verify_failed") if r["serve"].get("ready") else "load_failed"
        return r

    # 1, 2 ----------------------------------------------------------- reference loads
    if "default" in steps:
        res["arms"]["default"] = serve_and_check("default", src, [], record=True)
        save()
    if "default_again" in steps:
        res["arms"]["default_again"] = serve_and_check("default_again", src, [])
        save()

    # 3 --------------------------------------------------------------- SGLang's own sharded_state
    if "sharded_state_sglang" in steps:
        target = art / "sharded_sglang"
        prep = plan.run([sys.executable, str(SCRIPTS / "sglang_save_sharded.py"), "--model-path", str(src),
                         "--tensor-parallel-size", str(tp), *args, "--output", str(target)], "sharded_state_sglang.prepare.log")
        arm = {"prepare": prep, "bytes": dir_bytes(target)}
        if prep["rc"] == 0:
            arm.update(serve_and_check("sharded_state_sglang", target, ["--load-format", "sharded_state"]))
        else:
            arm["status"] = "prepare_failed"
            arm["error"] = _tail(logs / prep["log"])
        res["arms"]["sharded_state_sglang"] = arm
        save()

    # 4 --------------------------------------------------------------- raw preshard, unpatched SGLang
    prepare_py = ("import sys; from pathlib import Path; from servekit.prepare import prepare; "
                  "sys.exit(prepare(Path(sys.argv[1]), Path(sys.argv[2]), int(sys.argv[3]), "
                  "engine_args=sys.argv[5:], fmt=sys.argv[4]))")
    if "raw" in steps:
        target = art / "raw"
        prep = plan.run([sys.executable, "-c", prepare_py, str(src), str(target), str(tp), "raw", *args], "raw.prepare.log")
        arm = {"prepare": prep, "bytes": dir_bytes(target)}
        if not dry and target.exists():
            arm["ranks"] = {p.name: json.loads(p.read_text())["stats"] for p in sorted(target.glob("servekit-raw-rank-*.json"))}
        if prep["rc"] == 0:
            arm.update(serve_and_check("raw", target, []))
        else:
            arm["status"] = "prepare_failed"
            arm["error"] = _tail(logs / prep["log"])
        res["arms"]["raw"] = arm
        save()

    # 5 --------------------------------------------------------------- servekit's sharded_state, backports applied
    if "sharded_state_servekit" in steps and cfg["servekit_sharded_arm"]:
        patch = plan.run([sys.executable, "-c", "from servekit._patches import apply_all; apply_all()"], "patches.log")
        target = art / "sharded_servekit"
        prep = plan.run([sys.executable, "-c", prepare_py, str(src), str(target), str(tp), "sharded_state", *args],
                        "sharded_state_servekit.prepare.log")
        arm = {"patches": patch, "prepare": prep, "bytes": dir_bytes(target)}
        if not dry:
            arm["patch_log"] = (logs / "patches.log").read_text(errors="replace")[-2000:]
        if prep["rc"] == 0:
            arm.update(serve_and_check("sharded_state_servekit", target, ["--load-format", "sharded_state"]))
        else:
            arm["status"] = "prepare_failed"
            arm["error"] = _tail(logs / prep["log"])
        res["arms"]["sharded_state_servekit"] = arm
        save()

    # 6 --------------------------------------------------------------- load time
    if "timing" in steps:
        candidates = [("default", src, [], {}),
                      ("raw", art / "raw", [], {}),
                      ("raw_noverify", art / "raw", [], {"SERVEKIT_RAW_VERIFY": "0"}),
                      ("sharded_state_servekit", art / "sharded_servekit", ["--load-format", "sharded_state"], {}),
                      ("sharded_state_sglang", art / "sharded_sglang", ["--load-format", "sharded_state"], {})]
        for name, path, extra, env in candidates:
            base = name.replace("_noverify", "")
            status = res["arms"].get(base, {}).get("status")
            if not dry and status != "ok":
                plan.note(f"timing: skip {name} (arm status {status!r}: a wrong or broken load is not worth timing)")
                continue
            samples = []
            for i in range(runs):
                warm_s = warm(path, plan)
                with Server(plan, path, tp, args + extra, f"timing-{name}-{i}.log", env=env) as server:
                    samples.append({**server.result, "warm_s": warm_s})
            res["timing"][name] = samples
            save()

    # 7 --------------------------------------------------------------- deep-state comparisons
    if "deepstate" in steps:
        for arm in ("default_again", "sharded_state_sglang", "raw", "sharded_state_servekit"):
            if not dry and not (ds / arm).is_dir():
                continue
            for label, extra in (("full", []), ("registered_only", ["--registered-only"])):
                report = out / f"deepstate-{arm}-{label}.json"
                r = plan.run(["servekit", "deepstate", str(ds / "default"), str(ds / arm),
                              "--again", str(ds / "default_again"), "--out", str(report), *extra],
                             f"deepstate-{arm}-{label}.log", timeout=1800)
                if not dry:
                    r["summary"] = (logs / r["log"]).read_text(errors="replace")[-4000:]
                res["deepstate"][f"{arm}/{label}"] = r
        save()

    save()
    return res


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--model", choices=sorted(MODELS), required=True)
    p.add_argument("--hf-root", type=Path, default=Path("/hf/models"))
    p.add_argument("--work", type=Path, default=Path("/tmp/exp"))
    p.add_argument("--results", type=Path, default=Path("/results"))
    p.add_argument("--runs", type=int, default=3)
    p.add_argument("--steps", default="", help="comma-separated subset of steps")
    p.add_argument("--dry-run", action="store_true")
    a = p.parse_args()
    if a.dry_run and str(a.results) == "/results":
        a.results = HERE / "results" / "dry-run"
    res = run(a.model, a.hf_root, a.work, a.results, a.runs, a.dry_run, [s for s in a.steps.split(",") if s] or None)
    print(json.dumps({k: v for k, v in res.items() if k != "plan"}, indent=1)[:4000])
    return 0


if __name__ == "__main__":
    sys.exit(main())
