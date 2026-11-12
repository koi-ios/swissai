"""Run SGLang 0.5.20 with the start-up fixes, without changing SGLang's source.

    python -m pystartup.sglang_launch [wrapper options] -- <sglang.launch_server arguments>

Wrapper options:
    --zygote / --no-zygote      fork TP schedulers and the detokenizer from a preloaded
                                forkserver instead of spawning fresh interpreters (default on)
    --zygote-modules SPEC       comma list, text file, or calibration JSON (`python -m
                                pystartup.zygote calibrate`); a calibration JSON is replayed in
                                strict mode. Default: CUDA-free third-party stack only.
    --zygote-wait S             how long to wait for the zygote before falling back to spawn (300)
    --prewarm-manifest FILE     read these files with --prewarm-threads threads in the
                                background from t=0 (page-cache prewarm, fix d)
    --pycache-prefix DIR        PYTHONPYCACHEPREFIX for this and every child process (fix c)
    --report FILE               write what the wrapper decided (zygote status, timings) as JSON

What it changes at run time, all in this process before SGLang is imported:
1. sets the environment defaults SGLang's `_set_envs_and_config` would set unconditionally
   (CUDA_DEVICE_MAX_CONNECTIONS, TRTLLM_ENABLE_PDL, CUTE_DSL_LOG_*), so the zygote imports
   with the same values the workers would see;
2. starts the zygote (a fresh interpreter: it inherits none of this process's CUDA state);
3. ships this process's environment with every worker start (spawn semantics);
4. turns SGLang's `mp.set_start_method("spawn", force=True)` into "forkserver" once the
   zygote reports ok, and keeps "spawn" otherwise;
5. if SGLang's NUMA binding wraps the interpreter in numactl for a worker (only possible
   with spawn), that one worker is spawned as before.
Unsupported paths fall back to plain spawn: --enable-memory-saver (LD_PRELOAD), Ray, and
data-parallel controllers (they start their own workers in a child).
"""
from __future__ import annotations

import json
import os
import runpy
import sys
import time
from contextlib import contextmanager
from typing import List, Optional

T0 = time.time()
STATE = {"t0": T0, "zygote": None, "decided": False, "use_forkserver": False, "events": []}


def _log(msg: str):
    STATE["events"].append([round(time.time() - T0, 3), msg])
    sys.stderr.write(f"[pystartup {time.time() - T0:7.2f}s] {msg}\n")
    sys.stderr.flush()


def split_args(argv: List[str]):
    if "--" in argv:
        i = argv.index("--")
        return argv[:i], argv[i + 1:]
    return [], argv


def parse_wrapper(argv: List[str]) -> dict:
    o = {"zygote": True, "modules": None, "wait": 300.0, "prewarm": None, "prewarm_threads": 32,
         "pycache_prefix": None, "report": None}
    it = iter(argv)
    for a in it:
        if a == "--zygote":
            o["zygote"] = True
        elif a == "--no-zygote":
            o["zygote"] = False
        elif a == "--zygote-modules":
            o["modules"] = next(it)
        elif a == "--zygote-wait":
            o["wait"] = float(next(it))
        elif a == "--prewarm-manifest":
            o["prewarm"] = next(it)
        elif a == "--prewarm-threads":
            o["prewarm_threads"] = int(next(it))
        elif a == "--pycache-prefix":
            o["pycache_prefix"] = next(it)
        elif a == "--report":
            o["report"] = next(it)
        else:
            raise SystemExit(f"pystartup.sglang_launch: unknown option {a}")
    return o


UNSUPPORTED_FLAGS = ("--enable-memory-saver", "--use-ray")


def unsupported_reason(sglang_args: List[str]) -> Optional[str]:
    for f in UNSUPPORTED_FLAGS:
        if f in sglang_args:
            return f"{f} needs spawn"
    for flag in ("--dp-size", "--data-parallel-size", "--dp"):
        if flag in sglang_args:
            i = sglang_args.index(flag)
            if i + 1 < len(sglang_args) and sglang_args[i + 1] not in ("1",):
                return "data parallelism starts workers from a controller child"
    return None


def apply_sglang_static_env():
    """The unconditional part of sglang.srt.entrypoints.engine._set_envs_and_config (0.5.20)."""
    os.environ["CUDA_DEVICE_MAX_CONNECTIONS"] = "8"
    if os.environ.get("TRTLLM_ENABLE_PDL", "1") != "0":
        os.environ["TRTLLM_ENABLE_PDL"] = "1"
    os.environ.setdefault("CUTE_DSL_LOG_LEVEL", "30")
    os.environ.setdefault("CUTE_DSL_LOG_TO_CONSOLE", "1")


def _patch_numa_fallback(orig_ssm):
    """SGLang's numactl binding swaps the spawn executable and asserts spawn: keep that worker on spawn."""
    nu = sys.modules.get("sglang.srt.utils.numa_utils")
    if nu is None or getattr(nu, "_pystartup_patched", False):
        return
    orig = nu._mp_set_executable

    @contextmanager
    def _mp_set_executable(executable, debug_str):
        orig_ssm("spawn", force=True)
        _log("NUMA binding requested for a worker: spawning that worker")
        try:
            with orig(executable=executable, debug_str=debug_str):
                yield
        finally:
            orig_ssm("forkserver", force=True)

    nu._mp_set_executable = _mp_set_executable
    nu._pystartup_patched = True


def install_start_method_switch(wait_s: float):
    import multiprocessing

    orig = multiprocessing.set_start_method

    def set_start_method(method, force=False):
        z = STATE["zygote"]
        if method == "spawn" and z is not None:
            if not STATE["decided"]:
                t = time.time()
                st = z.wait(wait_s)
                STATE["decided"] = True
                STATE["status"] = st
                STATE["blocked_on_zygote_s"] = round(time.time() - t, 3)
                if st.get("ok"):
                    STATE["use_forkserver"] = True
                    _log(f"zygote ready (preload {st.get('preload_s')} s, waited "
                         f"{STATE['blocked_on_zygote_s']} s): workers will be forked from it")
                else:
                    _log(f"zygote not usable, keeping spawn: {st.get('reasons')}")
                write_report(STATE.get("report_path"))
            if STATE["use_forkserver"]:
                _patch_numa_fallback(orig)
                return orig("forkserver", force=True)
        return orig(method, force=force)

    multiprocessing.set_start_method = set_start_method


def write_report(path: Optional[str]):
    if not path:
        return
    st = STATE.get("status") or {}
    rep = {"t0": T0, "t_report": time.time(), "decided": STATE["decided"], "use_forkserver": STATE["use_forkserver"],
           "blocked_on_zygote_s": STATE.get("blocked_on_zygote_s"), "events": STATE["events"],
           "zygote_status": st}
    with open(path, "w") as f:
        json.dump(rep, f, indent=1, default=str)


def main(argv: Optional[List[str]] = None) -> None:
    argv = sys.argv[1:] if argv is None else argv
    wargs, sargs = split_args(argv)
    o = parse_wrapper(wargs)
    if o["pycache_prefix"]:
        os.makedirs(o["pycache_prefix"], exist_ok=True)
        os.environ["PYTHONPYCACHEPREFIX"] = o["pycache_prefix"]
        sys.pycache_prefix = o["pycache_prefix"]
    if o["prewarm"]:
        from pystartup import prewarm

        paths = prewarm.load_manifest(o["prewarm"])
        prewarm.start_background(paths, threads=o["prewarm_threads"])
        _log(f"prewarming {len(paths)} files with {o['prewarm_threads']} threads")
    reason = unsupported_reason(sargs) if o["zygote"] else "disabled"
    if reason is None:
        from pystartup import zygote

        apply_sglang_static_env()
        if o["modules"] and o["modules"].endswith(".json"):
            with open(o["modules"]) as f:
                cal = json.load(f)
            z = zygote.start(cal["modules"], facts="probe", strict=True, calibrated=True,
                             origins=cal.get("origins"))
            _log(f"zygote started with a calibrated list of {len(cal['modules'])} modules")
        else:
            mods = zygote.load_module_list(o["modules"]) if o["modules"] else zygote.DEFAULT_SAFE_MODULES
            z = zygote.start(mods, facts="probe", strict=False)
            _log(f"zygote started with {len(mods)} modules (best effort)")
        STATE["zygote"] = z
        zygote.install_parent_hooks()
        install_start_method_switch(o["wait"])
    else:
        _log(f"zygote off: {reason}")
    if o["report"]:
        import atexit

        STATE["report_path"] = o["report"]
        atexit.register(write_report, o["report"])
    sys.argv = ["sglang.launch_server"] + sargs
    runpy.run_module("sglang.launch_server", run_name="__main__", alter_sys=True)


if __name__ == "__main__":
    main()
