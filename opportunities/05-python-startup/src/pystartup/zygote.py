"""Parent-side API of the forkserver zygote (see zygote_preload.py for the server side).

    z = zygote.start(modules, facts="probe")      # returns at once; preload runs in parallel
    zygote.install_parent_hooks()                  # children get the env as of Process.start()
    ...                                            # parent does its own imports / arg parsing
    if z.wait(timeout=180).get("ok"):
        multiprocessing.set_start_method("forkserver", force=True)

The forkserver is a fresh interpreter (fork+exec), so the parent's own CUDA state never
leaks into it. Workers are forked from it after the preload, so they skip the imports.

Calibration (`calibrate()`): import a candidate list under the guard in a throwaway process
and keep only modules that imported completely without touching CUDA. A calibrated list can
then be replayed in strict mode: any divergence disables the zygote rather than risking a
partially-initialised module in the workers.

CLI:
    python -m pystartup.zygote calibrate --candidates FILE|mod,mod --out preload.json
    python -m pystartup.zygote smoke [--modules ...]     # start, wait, fork two children
"""
from __future__ import annotations

import argparse
import json
import multiprocessing
import os
import subprocess
import sys
import tempfile
import time
from typing import Dict, List, Optional, Sequence

from pystartup._zygote_common import CONFIG_ENV, ENV_KEY

PRELOAD_MODULE = "pystartup.zygote_preload"
PKG_PARENT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Third-party modules that are CUDA-free at import and dominate the import time of both
# engines. Missing ones are skipped (ModuleNotFoundError on the module itself).
DEFAULT_SAFE_MODULES = [
    "torch", "torch.nn", "torch.nn.functional", "torch.distributed", "numpy", "safetensors",
    "safetensors.torch", "tokenizers", "huggingface_hub", "transformers",
    "transformers.configuration_utils", "transformers.tokenization_utils_base",
    "transformers.modeling_utils", "triton", "triton.language", "fastapi", "uvicorn",
    "pydantic", "zmq", "zmq.asyncio", "orjson", "msgspec", "uvloop", "psutil", "requests",
    "aiohttp", "jinja2", "einops", "PIL.Image", "sentencepiece", "openai", "partial_json_parser",
    "outlines_core", "xgrammar", "llguidance", "torchao", "compressed_tensors",
]
# Modules that read the environment at import time and must see the env the parent sets
# just before it starts workers (SGLang sets PROMETHEUS_MULTIPROC_DIR in _set_envs_and_config).
DEFAULT_DENY = ["prometheus_client"]


class Zygote:
    def __init__(self, status_path: str, config_path: str, t_start: float):
        self.status_path = status_path
        self.config_path = config_path
        self.t_start = t_start
        self._status: Optional[dict] = None

    def status(self) -> Optional[dict]:
        if self._status is None and os.path.exists(self.status_path):
            try:
                with open(self.status_path) as f:
                    self._status = json.load(f)
            except (OSError, ValueError):
                return None
        return self._status

    def wait(self, timeout: float = 180.0, poll: float = 0.01) -> dict:
        deadline = time.time() + timeout
        while time.time() < deadline:
            st = self.status()
            if st is not None:
                st.setdefault("waited_s", round(time.time() - self.t_start, 3))
                return st
            if not forkserver_alive():
                time.sleep(0.05)  # a status written just before exit still counts
                st = self.status()
                return st or {"ok": False, "reasons": ["the fork server exited during preload"]}
            time.sleep(poll)
        return {"ok": False, "reasons": [f"zygote not ready after {timeout:.0f} s"]}


def forkserver_alive() -> bool:
    """Is this process's multiprocessing fork server still running (it may die in preload)?"""
    from multiprocessing import forkserver

    pid = getattr(forkserver._forkserver, "_forkserver_pid", None)
    if pid is None:
        return False
    try:
        wpid, _ = os.waitpid(pid, os.WNOHANG)
    except ChildProcessError:
        return False
    return wpid == 0


def _ensure_importable_env():
    """The forkserver must be able to import pystartup: put its parent dir on PYTHONPATH."""
    pp = os.environ.get("PYTHONPATH", "")
    parts = [p for p in pp.split(os.pathsep) if p]
    if PKG_PARENT not in parts:
        os.environ["PYTHONPATH"] = os.pathsep.join([PKG_PARENT] + parts)


def start(modules: Sequence[str], *, facts="probe", strict: bool = False, calibrated: bool = False,
          origins: Optional[Dict[str, str]] = None, deny_modules: Sequence[str] = tuple(DEFAULT_DENY),
          allow_native_threads: bool = True, status_dir: Optional[str] = None,
          exit_if_not_ok: bool = True) -> Zygote:
    """Start the forkserver now and return immediately; the preload runs in the background."""
    from multiprocessing import forkserver

    d = status_dir or tempfile.mkdtemp(prefix="pystartup-zygote-")
    status_path = os.path.join(d, "status.json")
    config_path = os.path.join(d, "config.json")
    cfg = {"modules": list(modules), "facts": facts, "strict": strict, "calibrated": calibrated,
           "origins": origins or {}, "deny_modules": list(deny_modules),
           "allow_native_threads": allow_native_threads, "status_path": status_path,
           "exit_if_not_ok": exit_if_not_ok}
    with open(config_path, "w") as f:
        json.dump(cfg, f)
    _ensure_importable_env()
    os.environ[CONFIG_ENV] = config_path
    ctx = multiprocessing.get_context("forkserver")
    ctx.set_forkserver_preload([PRELOAD_MODULE])
    t = time.time()
    forkserver.ensure_running()
    return Zygote(status_path, config_path, t)


_PARENT_HOOKED = False


def install_parent_hooks():
    """Ship the parent's current environment with every process start (spawn semantics).

    Without this, forkserver children would see the environment as it was when the
    forkserver started, missing everything the engine sets afterwards (NCCL_*, CUDA_DEVICE_
    MAX_CONNECTIONS, PROMETHEUS_MULTIPROC_DIR, per-rank CUDA_VISIBLE_DEVICES...).
    """
    global _PARENT_HOOKED
    if _PARENT_HOOKED:
        return
    import multiprocessing.spawn as mpspawn

    orig = mpspawn.get_preparation_data

    def get_preparation_data(name):
        d = orig(name)
        d[ENV_KEY] = dict(os.environ)
        return d

    mpspawn.get_preparation_data = get_preparation_data
    _PARENT_HOOKED = True


def load_module_list(spec: str) -> List[str]:
    """`spec` is a comma list, a text file (one module per line) or a calibration JSON."""
    if os.path.exists(spec):
        with open(spec) as f:
            text = f.read()
        if spec.endswith(".json"):
            return list(json.loads(text)["modules"])
        return [l.strip() for l in text.splitlines() if l.strip() and not l.startswith("#")]
    return [m for m in spec.split(",") if m]


# --------------------------------------------------------------------------------------------
# calibration

_CALIBRATE_CHILD = r"""
import importlib, json, sys, time
sys.path.insert(0, %(pkg)r)
from pystartup.cudafacts import CudaFacts, CudaGuard, CudaTouch, probe_facts
cands = %(cands)r
facts = %(facts)r
f = probe_facts() if facts == "probe" else (CudaFacts(**facts) if facts else None)
before = set(sys.modules)
import torch
g = CudaGuard(f).arm()
res = {"facts": f.__dict__ if f else None, "targets": {}}
for name in cands:
    g.current_module = name
    n0 = len(g.trips); t = time.time()
    try:
        importlib.import_module(name)
        res["targets"][name] = "tainted" if len(g.trips) > n0 else "ok"
    except CudaTouch as e:
        res["targets"][name] = "tripped: " + str(e)
    except BaseException as e:
        res["targets"][name] = "failed: " + repr(e)[:300]
g.disarm()
mods, origins = [], {}
for name, m in list(sys.modules.items()):
    if name in before or m is None:
        continue
    spec = getattr(m, "__spec__", None)
    if spec is None or not getattr(spec, "has_location", False) or not spec.origin:
        continue
    mods.append(name); origins[name] = spec.origin
res["modules"] = mods
res["origins"] = origins
res["trips"] = g.trips
print("CALIBRATION_JSON " + json.dumps(res))
"""


def calibrate(candidates: Sequence[str], facts="probe", timeout: float = 900.0,
              python: Optional[str] = None, extra_env: Optional[Dict[str, str]] = None) -> dict:
    """Import `candidates` under the guard in a throwaway interpreter.

    Returns {"modules": [...insertion order of fully imported modules...], "origins": {...},
    "targets": {candidate: ok|tripped|failed|tainted}, "trips": [...], "facts": {...}}.
    """
    code = _CALIBRATE_CHILD % {"pkg": PKG_PARENT, "cands": list(candidates), "facts": facts}
    env = dict(os.environ)
    env.update(extra_env or {})
    t = time.time()
    r = subprocess.run([python or sys.executable, "-c", code], capture_output=True, text=True,
                       timeout=timeout, env=env)
    for line in r.stdout.splitlines():
        if line.startswith("CALIBRATION_JSON "):
            res = json.loads(line[len("CALIBRATION_JSON "):])
            res["calibration_s"] = round(time.time() - t, 2)
            return res
    raise RuntimeError(f"calibration failed (rc={r.returncode}):\n{r.stderr[-4000:]}")


# --------------------------------------------------------------------------------------------
# smoke test: start, wait, fork two children that report what they inherited

def _smoke_child(q, names):
    import sys as _s

    q.put({"pid": os.getpid(), "preloaded": [n for n in names if n in _s.modules],
           "env_marker": os.environ.get("PYSTARTUP_SMOKE_MARKER"),
           "zygote_child": os.environ.get("PYSTARTUP_ZYGOTE_CHILD")})


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("calibrate")
    c.add_argument("--candidates", required=True)
    c.add_argument("--facts", default="probe", help='"probe", "none" or a JSON object')
    c.add_argument("--out", required=True)
    s = sub.add_parser("smoke")
    s.add_argument("--modules", default="json,email.mime.text")
    s.add_argument("--facts", default="probe")
    a = ap.parse_args(argv)
    facts = a.facts
    if facts == "none":
        facts = None
    elif facts.startswith("{"):
        facts = json.loads(facts)
    if a.cmd == "calibrate":
        res = calibrate(load_module_list(a.candidates), facts=facts)
        with open(a.out, "w") as f:
            json.dump(res, f, indent=1)
        ok = sum(1 for v in res["targets"].values() if v == "ok")
        print(f"{ok}/{len(res['targets'])} candidates ok, {len(res['modules'])} modules preloadable, "
              f"{len(res['trips'])} trips, {res['calibration_s']} s -> {a.out}")
        return 0
    names = load_module_list(a.modules)
    z = start(names, facts=facts)
    install_parent_hooks()
    st = z.wait(120)
    print(json.dumps({k: st.get(k) for k in ("ok", "reasons", "preload_s", "waited_s", "invariants")}, indent=1))
    if not st.get("ok"):
        return 1
    os.environ["PYSTARTUP_SMOKE_MARKER"] = "set-after-zygote-start"
    ctx = multiprocessing.get_context("forkserver")
    q = ctx.Queue()
    ps = [ctx.Process(target=_smoke_child, args=(q, names)) for _ in range(2)]
    for p in ps:
        p.start()
    out = [q.get(timeout=60) for _ in ps]
    for p in ps:
        p.join()
    print(json.dumps(out, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
