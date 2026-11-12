"""Generate patches/sglang-v0.5.20-worker-zygote.patch from the pystartup sources.

The patch adds an opt-in worker zygote to SGLang 0.5.20 (SGLANG_WORKER_ZYGOTE=1):
  python/sglang/_worker_zygote/{__init__,common,cudafacts,preload}.py   new, stdlib-only at import
  python/sglang/__init__.py            3-line hook at the very top (before any heavy import)
  python/sglang/srt/entrypoints/engine.py   start method: forkserver if the zygote is ok, else spawn
  python/sglang/srt/utils/numa_utils.py     numactl-wrapped workers are still spawned
  python/sglang/srt/environ.py         registers SGLANG_WORKER_ZYGOTE / _MODULES / _WAIT

Run from the opportunity folder (needs the vendor clone at vendor/sglang-v0.5.20):
    python patches/make_sglang_patch.py
It works on a local branch `pystartup-worker-zygote` of the vendor clone; nothing is pushed.
"""
import os
import re
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
REPO = os.path.join(ROOT, "vendor", "sglang-v0.5.20")
SRC = os.path.join(ROOT, "src", "pystartup")
PKG = os.path.join(REPO, "python", "sglang", "_worker_zygote")
OUT = os.path.join(HERE, "sglang-v0.5.20-worker-zygote.patch")
BRANCH = "pystartup-worker-zygote"


def git(*args, check=True):
    return subprocess.run(["git", "-C", REPO, *args], check=check, capture_output=True, text=True)


def rewrite(text: str) -> str:
    text = text.replace("from pystartup._zygote_common import", "from sglang._worker_zygote.common import")
    text = text.replace("from pystartup.cudafacts import", "from sglang._worker_zygote.cudafacts import")
    text = text.replace("from pystartup import cudafacts", "from sglang._worker_zygote import cudafacts")
    text = text.replace('"PYSTARTUP_ZYGOTE_CONFIG"', '"SGLANG_WORKER_ZYGOTE_CONFIG"')
    text = text.replace('"PYSTARTUP_ZYGOTE_CHILD"', '"SGLANG_WORKER_ZYGOTE_CHILD"')
    text = text.replace('"pystartup_environ"', '"sglang_worker_zygote_environ"')
    text = text.replace('"PYSTARTUP_FAKE_CUDA"', '"SGLANG_WORKER_ZYGOTE_FAKE_CUDA"')
    text = text.replace("pystartup zygote", "sglang worker zygote")
    text = text.replace('"pystartup.zygote_preload"', '"sglang._worker_zygote.preload"')
    text = text.replace("The parent (see zygote.py)", "The launch process (see __init__.py)")
    assert "pystartup." not in text, [l for l in text.splitlines() if "pystartup." in l]
    return text


INIT = '''"""Opt-in worker zygote (SGLANG_WORKER_ZYGOTE=1): fork TP workers from a preloaded server.

With the default "spawn" start method every scheduler and the detokenizer is a fresh
interpreter that re-imports torch, transformers, triton and SGLang (11-16 s per launch on
the Swiss AI clusters). With the zygote, a multiprocessing fork server is started from the
very top of `import sglang` in the launch process, preloads the import set in parallel with
the launch process's own work, and the workers are forked from it.

Safety (see cudafacts.py and preload.py):
* the fork server is a fresh interpreter (fork+exec): none of the launch process's CUDA
  state reaches it;
* while it preloads, torch.cuda answers is_available/device_count/capability/name from an
  NVML probe run in a separate short-lived process, and every other way into the driver
  raises; after preload it checks that torch.cuda is not initialised, that no /dev/nvidia*
  descriptor is open and that it runs one Python thread, else the launch uses spawn;
* each worker gets the launch process's environment as of Process.start(), and checks the
  served device facts against the device after its own first CUDA init.

This module is standard library only at import.
"""
import json
import os
import sys
import tempfile
import time

from sglang._worker_zygote.common import CONFIG_ENV, ENV_KEY, in_forkserver_process

STARTED_ENV = "SGLANG_WORKER_ZYGOTE_STARTED"
PRELOAD_MODULE = "sglang._worker_zygote.preload"
DEFAULT_MODULES = [
    "torch", "torch.nn", "torch.nn.functional", "torch.distributed", "numpy", "safetensors",
    "safetensors.torch", "tokenizers", "huggingface_hub", "transformers",
    "transformers.configuration_utils", "transformers.tokenization_utils_base",
    "transformers.modeling_utils", "triton", "triton.language", "fastapi", "uvicorn",
    "pydantic", "zmq", "zmq.asyncio", "orjson", "msgspec", "uvloop", "psutil", "requests",
    "aiohttp", "jinja2", "einops", "PIL.Image", "sentencepiece", "openai", "partial_json_parser",
    "outlines_core", "xgrammar", "llguidance", "torchao", "compressed_tensors",
]
_STATE = {"status_path": None, "t_start": None, "method": None}


def enabled() -> bool:
    return os.environ.get("SGLANG_WORKER_ZYGOTE", "0") == "1"


def _is_launch_process() -> bool:
    argv = list(getattr(sys, "orig_argv", []) or [])
    joined = " ".join(argv)
    if "sglang.launch_server" in joined:
        return True
    return any(os.path.basename(a) == "sglang" for a in argv[:2]) and "serve" in argv


def _static_env():
    """The unconditional part of engine._set_envs_and_config, set before the zygote imports."""
    os.environ["CUDA_DEVICE_MAX_CONNECTIONS"] = "8"
    if os.environ.get("TRTLLM_ENABLE_PDL", "1") != "0":
        os.environ["TRTLLM_ENABLE_PDL"] = "1"
    os.environ.setdefault("CUTE_DSL_LOG_LEVEL", "30")
    os.environ.setdefault("CUTE_DSL_LOG_TO_CONSOLE", "1")


def _modules():
    spec = os.environ.get("SGLANG_WORKER_ZYGOTE_MODULES")
    if not spec:
        return DEFAULT_MODULES, False, {}
    with open(spec) as f:
        if spec.endswith(".json"):
            cal = json.load(f)
            return cal["modules"], True, cal.get("origins", {})
        return [l.strip() for l in f if l.strip() and not l.startswith("#")], False, {}


def _install_parent_hooks():
    import multiprocessing.spawn as mpspawn

    orig = mpspawn.get_preparation_data

    def get_preparation_data(name):
        d = orig(name)
        d[ENV_KEY] = dict(os.environ)
        return d

    mpspawn.get_preparation_data = get_preparation_data


def _start():
    import multiprocessing
    from multiprocessing import forkserver

    os.environ[STARTED_ENV] = str(os.getpid())
    _static_env()
    modules, calibrated, origins = _modules()
    d = tempfile.mkdtemp(prefix="sglang-worker-zygote-")
    status_path = os.path.join(d, "status.json")
    cfg = {"modules": modules, "facts": "probe", "strict": calibrated, "calibrated": calibrated,
           "origins": origins, "deny_modules": ["prometheus_client"], "allow_native_threads": True,
           "status_path": status_path, "exit_if_not_ok": True}
    cfg_path = os.path.join(d, "config.json")
    with open(cfg_path, "w") as f:
        json.dump(cfg, f)
    os.environ[CONFIG_ENV] = cfg_path
    multiprocessing.get_context("forkserver").set_forkserver_preload([PRELOAD_MODULE])
    _STATE["t_start"] = time.time()
    forkserver.ensure_running()
    _STATE["status_path"] = status_path
    _install_parent_hooks()


def on_sglang_import():
    """Called first thing in sglang/__init__.py."""
    if not enabled():
        return
    if in_forkserver_process(getattr(sys, "orig_argv", None)) and os.environ.get(CONFIG_ENV):
        # This is the zygote: arm the guard before the rest of `import sglang` runs.
        from sglang._worker_zygote import cudafacts

        facts = cudafacts.probe_facts()
        import torch  # noqa: F401  (CUDA-free)

        cudafacts.CudaGuard(facts).arm()
        return
    if os.environ.get(STARTED_ENV) or not _is_launch_process():
        return
    try:
        _start()
    except Exception as e:  # never break the launch: spawn stays the fallback
        sys.stderr.write(f"sglang worker zygote not started: {e!r}\\n")


def _forkserver_alive() -> bool:
    from multiprocessing import forkserver

    pid = getattr(forkserver._forkserver, "_forkserver_pid", None)
    if pid is None:
        return False
    try:
        return os.waitpid(pid, os.WNOHANG)[0] == 0
    except ChildProcessError:
        return False


def worker_start_method() -> str:
    """'forkserver' if the zygote is ready and safe, else 'spawn' (decided once)."""
    if _STATE["method"] is not None:
        return _STATE["method"]
    path = _STATE["status_path"]
    method, why = "spawn", "zygote not enabled"
    if path:
        deadline = time.time() + float(os.environ.get("SGLANG_WORKER_ZYGOTE_WAIT", "300"))
        st = None
        while time.time() < deadline:
            if os.path.exists(path):
                with open(path) as f:
                    st = json.load(f)
                break
            if not _forkserver_alive():
                time.sleep(0.05)
                if os.path.exists(path):
                    with open(path) as f:
                        st = json.load(f)
                break
            time.sleep(0.01)
        if st and st.get("ok"):
            method, why = "forkserver", f"preload {st.get('preload_s')} s"
        else:
            why = (st or {}).get("reasons") or "zygote did not report"
    _STATE["method"] = method
    import logging

    logging.getLogger(__name__).info(f"worker start method: {method} ({why})")
    return method
'''

HOOK = '''# Opt-in worker zygote (SGLANG_WORKER_ZYGOTE=1): must run before anything heavy is imported.
from sglang._worker_zygote import on_sglang_import as _wz_on_import

_wz_on_import()
'''


def edit(path, old, new, count=1):
    p = os.path.join(REPO, path)
    s = open(p).read()
    assert s.count(old) == count, (path, old, s.count(old))
    open(p, "w").write(s.replace(old, new))


def main():
    if not os.path.isdir(os.path.join(REPO, ".git")):
        sys.exit(f"no git clone at {REPO}")
    base = git("rev-parse", "v0.5.20^{commit}", check=False).stdout.strip() or git("rev-parse", "HEAD").stdout.strip()
    git("checkout", "-q", "-B", BRANCH, base)
    os.makedirs(PKG, exist_ok=True)
    open(os.path.join(PKG, "__init__.py"), "w").write(INIT)
    for src, dst in (("_zygote_common.py", "common.py"), ("cudafacts.py", "cudafacts.py"),
                     ("zygote_preload.py", "preload.py")):
        open(os.path.join(PKG, dst), "w").write(rewrite(open(os.path.join(SRC, src)).read()))
    edit("python/sglang/__init__.py", 'import platform as _platform\nimport sys as _sys\n',
         HOOK + '\nimport platform as _platform\nimport sys as _sys\n')
    edit("python/sglang/srt/entrypoints/engine.py",
         '    # Set mp start method\n    mp.set_start_method("spawn", force=True)\n',
         '    # Set mp start method ("forkserver" only with a ready, CUDA-clean worker zygote)\n'
         '    from sglang._worker_zygote import worker_start_method\n\n'
         '    mp.set_start_method(worker_start_method(), force=True)\n')
    edit("python/sglang/srt/utils/numa_utils.py",
         '    start_method = multiprocessing.get_start_method()\n    assert start_method == "spawn", f"{start_method=}"\n',
         '    start_method = multiprocessing.get_start_method()\n'
         '    if start_method == "forkserver":\n'
         '        # numactl wraps the interpreter, which only spawn executes: spawn this worker\n'
         '        multiprocessing.set_start_method("spawn", force=True)\n'
         '        try:\n'
         '            with _mp_set_executable(executable=executable, debug_str=debug_str):\n'
         '                yield\n'
         '        finally:\n'
         '            multiprocessing.set_start_method("forkserver", force=True)\n'
         '        return\n'
         '    assert start_method == "spawn", f"{start_method=}"\n')
    edit("python/sglang/srt/environ.py", "    SGLANG_NUMA_BIND_V2 = EnvBool(True)\n",
         "    SGLANG_NUMA_BIND_V2 = EnvBool(True)\n"
         "    # Fork TP workers from a preloaded, CUDA-clean fork server (sglang/_worker_zygote).\n"
         "    # Read directly from os.environ at the top of `import sglang`; listed here for discovery.\n"
         "    SGLANG_WORKER_ZYGOTE = EnvBool(False)\n"
         "    SGLANG_WORKER_ZYGOTE_MODULES = EnvStr(\"\")\n"
         "    SGLANG_WORKER_ZYGOTE_WAIT = EnvFloat(300.0)\n")
    git("add", "-A", "python/sglang")
    msg = ("[startup] opt-in worker zygote: fork TP workers from a preloaded fork server\n\n"
           "With SGLANG_WORKER_ZYGOTE=1 the launch process starts a multiprocessing fork server\n"
           "at the top of `import sglang`. It preloads the CUDA-free import set in parallel with\n"
           "the launch process, under a guard that serves device facts from an NVML probe and\n"
           "trips on any other way into the driver. Workers are forked from it only if torch.cuda\n"
           "is uninitialised, no /dev/nvidia* fd is open and it runs one thread; otherwise spawn\n"
           "is kept. Each worker gets the launch process's environment as of Process.start().\n\n"
           "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>\n")
    git("-c", "user.name=pystartup", "-c", "user.email=pystartup@localhost", "commit", "-q", "-m", msg)
    patch = git("format-patch", "-1", "--stdout").stdout
    open(OUT, "w").write(patch)
    stat = git("show", "--stat", "--format=", "HEAD").stdout
    print(stat)
    print(f"-> {OUT}")


if __name__ == "__main__":
    main()
