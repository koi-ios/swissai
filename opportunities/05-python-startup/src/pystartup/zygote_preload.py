"""Runs *inside* the multiprocessing forkserver process, as its only preload module.

The parent (see zygote.py) starts the forkserver with
`set_forkserver_preload(["pystartup.zygote_preload"])` and points
PYSTARTUP_ZYGOTE_CONFIG at a JSON file:

    {"modules": [...],          # import order (calibrated list or a safe default list)
     "facts": "probe" | {...} | null,
     "strict": true,            # any failure/trip -> not ok
     "deny_modules": ["prometheus_client"],
     "allow_native_threads": true,
     "status_path": "/tmp/...json"}

What happens here, in order:
1. `import torch` (CUDA-free) and arm the CudaGuard with device facts from an NVML probe
   that runs in a separate short-lived process.
2. Import the configured modules one by one, attributing any CUDA touch to the module.
3. Disarm, then check the invariants that make forking safe:
   - torch.cuda.is_initialized() is False;
   - no /dev/nvidia* descriptor is open (no driver handle, so no cuInit, no NVML session);
   - exactly one Python thread (and report native threads);
   - no denied module was imported (e.g. prometheus_client, which reads
     PROMETHEUS_MULTIPROC_DIR at import and must see the env the parent sets later).
4. Install the child-side hooks: every forked child first takes the parent's environment as
   it was at Process.start() (spawn semantics, via multiprocessing.spawn.prepare), then,
   after its own first real CUDA init, checks the served facts against the device.
5. Write the status JSON atomically. If not ok, exit: the parent falls back to spawn.

Nothing here may raise: forkserver.main() would die with it. Everything is caught and
reported in the status file.
"""
from __future__ import annotations

import importlib
import json
import os
import sys
import threading
import time

from pystartup._zygote_common import CHILD_ENV, CONFIG_ENV, ENV_KEY, in_forkserver_process


def _write_status(path, status):
    if not path:
        return
    tmp = f"{path}.tmp.{os.getpid()}"
    with open(tmp, "w") as f:
        json.dump(status, f, indent=1, default=str)
    os.replace(tmp, path)


def _install_child_hooks(facts):
    import multiprocessing.spawn as mpspawn

    orig_prepare = mpspawn.prepare

    def prepare(data):
        env = data.get(ENV_KEY) if isinstance(data, dict) else None
        if env is not None:
            # spawn semantics: the child's environment is the parent's at Process.start()
            for k in list(os.environ):
                if k not in env:
                    del os.environ[k]
            os.environ.update(env)
        os.environ[CHILD_ENV] = str(os.getppid())
        return orig_prepare(data)

    mpspawn.prepare = prepare

    def after_in_child():
        tc = sys.modules.get("torch.cuda")
        if tc is None:
            return
        try:
            if tc.is_initialized():  # cannot happen if the invariants held; never continue if it does
                os.write(2, b"pystartup zygote: CUDA initialised in the zygote, child aborts\n")
                os._exit(70)
        except Exception:
            pass
        if facts is None:
            return
        from pystartup.cudafacts import verify_facts

        def _verify():
            problems = verify_facts(facts)
            if problems:
                raise RuntimeError("pystartup zygote: device facts served during preload differ "
                                   "from the device: " + "; ".join(problems))

        try:
            tc._lazy_call(_verify)
        except Exception:
            pass

    os.register_at_fork(after_in_child=after_in_child)


def _replay_ok(name, origin):
    """For calibrated lists: is `name` still the module calibration saw?"""
    import importlib.util

    try:
        spec = importlib.util.find_spec(name)
    except (ImportError, ValueError, AttributeError):
        return False
    if spec is None:
        return False
    return origin is None or spec.origin == origin


def run():
    t0 = time.time()
    status = {"ok": False, "pid": os.getpid(), "ppid": os.getppid(), "python": sys.version,
              "t_start": t0, "loaded": [], "failed": [], "tripped": [], "tainted": [],
              "skipped": [], "reasons": []}
    cfg_path = os.environ.get(CONFIG_ENV)
    cfg = {}
    try:
        with open(cfg_path) as f:
            cfg = json.load(f)
    except Exception as e:  # no config: nothing to do, report and serve plain forks
        status["reasons"].append(f"no config at {cfg_path!r}: {e!r}")
    status_path = cfg.get("status_path")
    strict = bool(cfg.get("strict", True))
    modules = cfg.get("modules", [])
    origins = cfg.get("origins", {})
    n_before = len(sys.modules)
    guard = None
    facts = None
    try:
        from pystartup import cudafacts
        from pystartup.cudafacts import CudaFacts, CudaGuard, CudaTouch, probe_facts

        if cudafacts.ACTIVE is not None and cudafacts.ACTIVE.armed:
            # armed earlier in this process by an early hook (e.g. the top of a package __init__)
            guard = cudafacts.ACTIVE
            facts = guard.facts
            status["guard"] = "armed by early hook"
        else:
            f = cfg.get("facts", "probe")
            t = time.time()
            if f == "probe":
                facts = probe_facts()
            elif isinstance(f, dict):
                facts = CudaFacts(**f)
            status["probe_s"] = round(time.time() - t, 3)
            try:
                t = time.time()
                import torch  # noqa: F401

                status["loaded"].append(["torch", round(time.time() - t, 3)])
                guard = CudaGuard(facts).arm()
            except ImportError as e:
                status["reasons"].append(f"torch not importable, no guard needed: {e!r}")
        status["facts"] = facts.__dict__ if facts else None
        for name in modules:
            if name in sys.modules:
                continue
            if cfg.get("calibrated") and not _replay_ok(name, origins.get(name)):
                status["skipped"].append(name)
                continue
            if guard is not None:
                guard.current_module = name
            trips_before = len(guard.trips) if guard else 0
            t = time.time()
            try:
                importlib.import_module(name)
            except CudaTouch as e:
                status["tripped"].append([name, str(e), guard.trips[-1] if guard and guard.trips else None])
                continue
            except ModuleNotFoundError as e:
                # The named module itself is absent: nothing of it executed.
                status["failed"].append([name, repr(e), e.name == name])
                continue
            except BaseException as e:  # noqa: BLE001 - must not kill the forkserver
                status["failed"].append([name, repr(e), False])
                continue
            dt = round(time.time() - t, 3)
            if guard is not None and len(guard.trips) > trips_before:
                status["tainted"].append([name, guard.trips[trips_before:]])
            else:
                status["loaded"].append([name, dt])
        if guard is not None:
            guard.disarm()
            status["trips_total"] = len(guard.trips)
    except BaseException as e:  # noqa: BLE001
        status["reasons"].append(f"preload crashed: {e!r}")
        if guard is not None:
            try:
                guard.disarm()
            except BaseException:
                pass

    # ---- invariants ---------------------------------------------------------------------
    try:
        from pystartup.cudafacts import (mapped_libraries, native_thread_count, nvidia_fds,
                                         torch_cuda_initialized)

        inv = {
            "torch_cuda_initialized": torch_cuda_initialized(),
            "nvidia_fds": nvidia_fds(),
            "python_threads": threading.active_count(),
            "native_threads": native_thread_count(),
            "mapped": mapped_libraries(),
            "denied_present": [m for m in cfg.get("deny_modules", []) if m in sys.modules],
        }
    except BaseException as e:  # noqa: BLE001
        inv = {"error": repr(e)}
    status["invariants"] = inv
    reasons = status["reasons"]
    if "error" in inv:
        reasons.append("invariant check failed to run")
    else:
        if inv["torch_cuda_initialized"] is not False:
            reasons.append("torch.cuda is initialised in the zygote")
        if inv["nvidia_fds"]:
            reasons.append(f"driver handles open in the zygote: {inv['nvidia_fds']}")
        if inv["python_threads"] != 1:
            reasons.append(f"{inv['python_threads']} Python threads in the zygote")
        if (not cfg.get("allow_native_threads", True)) and (inv["native_threads"] or 1) > 1:
            reasons.append(f"{inv['native_threads']} native threads in the zygote")
        if inv["denied_present"]:
            reasons.append(f"denied modules imported: {inv['denied_present']}")
    if status["tripped"]:
        reasons.append(f"{len(status['tripped'])} module(s) touched CUDA during preload; their partial "
                       f"side effects stay in the zygote")
    if status["tainted"]:
        reasons.append(f"a CUDA trip was swallowed inside {[t[0] for t in status['tainted']]}")
    if strict and status["failed"]:
        reasons.append(f"{len(status['failed'])} module(s) failed in strict mode")
    if not strict:
        bad = [f for f in status["failed"] if not f[2]]
        if bad:
            reasons.append(f"{len(bad)} module(s) failed after executing code: {[b[0] for b in bad][:5]}")
    status["ok"] = not reasons
    status["modules_before"] = n_before
    status["modules_after"] = len(sys.modules)
    status["preload_s"] = round(time.time() - t0, 3)
    status["t_ready"] = time.time()
    if status["ok"]:
        try:
            _install_child_hooks(facts)
        except BaseException as e:  # noqa: BLE001
            status["ok"] = False
            reasons.append(f"child hooks: {e!r}")
    try:
        _write_status(status_path, status)
    except BaseException:  # noqa: BLE001
        pass
    if not status["ok"] and cfg.get("exit_if_not_ok", True):
        os._exit(0)  # the parent will not use this server; free its memory now


# Only the fork server runs the preload; importing this module anywhere else is a no-op.
if in_forkserver_process(getattr(sys, "orig_argv", None)) and os.environ.get(CONFIG_ENV):
    run()
