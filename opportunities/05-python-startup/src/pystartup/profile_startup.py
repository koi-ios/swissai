"""Profile the Python start-up of an engine image: what an SGLang / vLLM launch spends before
torch.distributed init, and what each fix removes.

    python -m pystartup.profile_startup env       [--engine sglang|vllm]
    python -m pystartup.profile_startup imports   --engine sglang --role scheduler --modes asis,nopyc,prefix --repeats 3
    python -m pystartup.profile_startup compileall --engine sglang
    python -m pystartup.profile_startup spawn     --engine sglang --role scheduler --workers 2
    python -m pystartup.profile_startup tree      --trace-dir DIR [--log engine.log]
    python -m pystartup.profile_startup all       --engine sglang --out results.json   (CPU study)

`--fake-cuda` (default on for `imports`/`spawn` when no GPU is visible) serves device facts from
PYSTARTUP_FAKE_CUDA (default: 2 x sm90) so the GPU import path can be measured in a CPU
container; anything that needs more than those facts is recorded as a trip, not faked.

Every timing is a fresh interpreter. Repeats are interleaved across modes. Results are JSON.
"""
from __future__ import annotations

import argparse
import glob
import importlib.metadata as md
import importlib.util
import json
import os
import platform
import shutil
import statistics as st
import sys
import sysconfig
import tempfile
import time
from typing import Dict, List, Optional

from pystartup import importprof, pycheck

ROLES = {
    "sglang": {
        # what `python -m sglang.launch_server` imports before it spawns anything
        "main": ["sglang.launch_server", "sglang.srt.entrypoints.http_server"],
        # what each TP scheduler child imports (target of the spawned process)
        "scheduler": ["sglang.launch_server", "sglang.srt.managers.scheduler"],
        "detokenizer": ["sglang.launch_server", "sglang.srt.managers.detokenizer_manager"],
    },
    "vllm": {
        "main": ["vllm.entrypoints.cli.main", "vllm.entrypoints.openai.api_server"],
        "engine_core": ["vllm.v1.engine.core"],
        "worker": ["vllm.v1.executor.multiproc_executor", "vllm.v1.worker.gpu_worker",
                   "vllm.v1.worker.gpu_model_runner"],
    },
    "proxy": {  # local stand-in when no engine is installed
        "main": ["torch", "transformers", "fastapi", "uvicorn"],
        "scheduler": ["torch", "transformers", "torch.distributed", "safetensors.torch"],
    },
}
DEFAULT_FAKE = {"count": 2, "major": 9, "minor": 0, "name": "NVIDIA H100 80GB HBM3"}
VERSIONS = ["torch", "sglang", "vllm", "transformers", "triton", "flashinfer-python", "sglang-kernel",
            "sgl-kernel", "torchao", "numpy", "fastapi", "pydantic"]


def gpu_visible() -> bool:
    return any(os.path.exists(p) for p in ("/dev/nvidia0", "/dev/nvidiactl"))


def cuda_stub_dir() -> Optional[str]:
    """A dir with libcuda.so.1 -> the CUDA toolkit stub, so CUDA extensions import without a driver.

    Only for CPU-container import studies: every driver call in the stub fails, so anything
    that really needs the GPU still fails (and is recorded).
    """
    cands = glob.glob("/usr/local/cuda*/lib64/stubs/libcuda.so") + glob.glob("/usr/local/cuda*/targets/*/lib/stubs/libcuda.so")
    if not cands:
        return None
    d = tempfile.mkdtemp(prefix="pystartup-cudastub-")
    os.symlink(cands[0], os.path.join(d, "libcuda.so.1"))
    os.symlink(cands[0], os.path.join(d, "libcuda.so"))
    return d


def fake_env() -> Dict[str, str]:
    env = {"PYSTARTUP_FAKE_CUDA": os.environ.get("PYSTARTUP_FAKE_CUDA", json.dumps(DEFAULT_FAKE))}
    stub = cuda_stub_dir()
    if stub:
        env["LD_LIBRARY_PATH"] = os.pathsep.join([stub] + [p for p in os.environ.get("LD_LIBRARY_PATH", "").split(os.pathsep) if p])
    return env


def package_roots(engine: Optional[str]) -> List[str]:
    roots = []
    for key in ("purelib", "platlib"):
        p = sysconfig.get_paths().get(key)
        if p and os.path.isdir(p) and p not in roots:
            roots.append(p)
    try:
        import site

        for p in site.getsitepackages():
            if os.path.isdir(p) and p not in roots:
                roots.append(p)
    except Exception:
        pass
    # editable / source-tree installs (upstream lmsysorg/sglang: /sgl-workspace/sglang/python)
    for pkg in ([engine] if engine and engine != "proxy" else []):
        try:
            spec = importlib.util.find_spec(pkg)
        except Exception:
            spec = None
        for loc in (spec.submodule_search_locations or []) if spec else []:
            if not any(os.path.abspath(loc).startswith(os.path.abspath(r)) for r in roots):
                roots.append(loc)
    return roots


def env_report(engine: Optional[str], audit: bool = True) -> dict:
    vers = {}
    for p in VERSIONS:
        try:
            vers[p] = md.version(p)
        except md.PackageNotFoundError:
            pass
    rep = {"python": sys.version, "executable": sys.executable, "platform": platform.platform(),
           "machine": platform.machine(), "cpus": os.cpu_count(), "versions": vers,
           "gpu_visible": gpu_visible(), "roots": package_roots(engine),
           "env": {k: os.environ.get(k) for k in ("PYTHONDONTWRITEBYTECODE", "PYTHONPYCACHEPREFIX",
                                                  "UV_COMPILE_BYTECODE", "PYTHONPATH")}}
    try:
        st_ = os.statvfs(rep["roots"][0])
        rep["root_fs_readonly"] = bool(st_.f_flag & os.ST_RDONLY)
    except (OSError, IndexError):
        pass
    if audit:
        t = time.perf_counter()
        a = pycheck.audit(rep["roots"], hash_sources=False)
        rep["pyc_audit"] = a.to_dict()
        rep["pyc_audit"]["examples"] = {k: v[:3] for k, v in a.examples.items()}
        rep["pyc_audit_s"] = round(time.perf_counter() - t, 2)
    return rep


def _med(xs):
    xs = [x for x in xs if x is not None]
    if not xs:
        return None
    return {"median": round(st.median(xs), 3), "min": round(min(xs), 3), "max": round(max(xs), 3), "n": len(xs)}


def imports_study(engine: str, role: str, modes: List[str], repeats: int, fake: bool,
                  first_cold: bool = True) -> dict:
    targets = ROLES[engine][role]
    env_extra = fake_env() if fake else {}
    work = tempfile.mkdtemp(prefix="pystartup-imports-")
    runs = []
    if first_cold:
        # the very first import in this container: page cache and (if writable) pycs as shipped
        r = importprof.run_probe(targets, "asis", fake_cuda=fake, env_extra=env_extra, audit_pyc=True,
                                 dump_files=True, dump_modules=True, workdir=work)
        r["label"] = "first-in-container"
        runs.append(r)
    for rep in range(repeats):
        for m in modes:
            if m == "prefix":
                pdir = tempfile.mkdtemp(prefix="pystartup-prefix-")
                for sub in ("prefix-cold", "prefix-warm"):
                    r = importprof.run_probe(targets, sub, prefix_dir=pdir, fake_cuda=fake, env_extra=env_extra,
                                             workdir=work)
                    r["label"], r["repeat"] = sub, rep
                    runs.append(r)
                shutil.rmtree(pdir, ignore_errors=True)
                continue
            if m == "unchecked":
                continue  # handled by compileall_study, which needs the module list
            r = importprof.run_probe(targets, m, fake_cuda=fake, env_extra=env_extra, workdir=work)
            r["label"], r["repeat"] = m, rep
            runs.append(r)
    summary = {}
    for lab in sorted({r["label"] for r in runs}):
        rs = [r for r in runs if r["label"] == lab and r.get("returncode") == 0]
        if not rs:
            summary[lab] = {"error": [r.get("stderr_tail", "")[-500:] for r in runs if r["label"] == lab][:1]}
            continue
        summary[lab] = {
            "process_wall_s": _med([r["wall_s"] for r in rs]),
            "import_s": _med([r.get("import_s") for r in rs]),
            "compile_s": _med([r.get("breakdown_s", {}).get("compile") for r in rs]),
            "compiled_files": _med([r.get("compiled_files") for r in rs]),
            "unmarshal_s": _med([r.get("breakdown_s", {}).get("unmarshal") for r in rs]),
            "read_s": _med([r.get("breakdown_s", {}).get("read") for r in rs]),
            "find_s": _med([r.get("breakdown_s", {}).get("find") for r in rs]),
            "ext_s": _med([r.get("breakdown_s", {}).get("ext") for r in rs]),
            "exec_and_other_s": _med([r.get("breakdown_s", {}).get("exec_and_other") for r in rs]),
            "modules_new": _med([r.get("modules_new") for r in rs]),
            "errors": rs[0].get("errors"),
        }
    first = next((r for r in runs if r.get("label") == "first-in-container"), {})
    return {"engine": engine, "role": role, "targets": targets, "fake_cuda": fake, "summary": summary,
            "first_run_detail": {k: first.get(k) for k in ("pyc_audit", "opened_files", "maps", "proc_io_delta",
                                                           "modules_by_top_package", "importtime", "cuda_trips",
                                                           "compiled_examples", "rss_kb", "errors")},
            "manifest": first.get("files_in_order", []), "module_order": first.get("module_order", []),
            "runs": [{k: v for k, v in r.items() if k not in ("files_in_order", "module_order")} for r in runs]}


def compileall_study(engine: str, role: str, module_files: List[str], repeats: int, fake: bool) -> dict:
    """Fix (a), measured out of tree: unchecked-hash pycs for exactly the import set, then import."""
    targets = ROLES[engine][role]
    env_extra = fake_env() if fake else {}
    out = {}
    for inv in ("timestamp", "unchecked-hash"):
        pdir = tempfile.mkdtemp(prefix=f"pystartup-ca-{inv}-")
        c = importprof.compileall_prefix([], pdir, invalidation=inv, files=module_files)
        runs = [importprof.run_probe(targets, f"prefix:{pdir}", fake_cuda=fake, env_extra=env_extra)
                for _ in range(repeats)]
        out[inv] = {"compileall": c, "import_s": _med([r.get("import_s") for r in runs]),
                    "process_wall_s": _med([r["wall_s"] for r in runs]),
                    "compiled_files": _med([r.get("compiled_files") for r in runs])}
        shutil.rmtree(pdir, ignore_errors=True)
    return out


def tree_summary(trace_dir: str) -> dict:
    """Process tree of a launch from tracer/sitecustomize.py records."""
    procs = []
    for p in glob.glob(os.path.join(trace_dir, "proc-*.json")):
        try:
            with open(p) as f:
                procs.append(json.load(f))
        except (OSError, ValueError):
            continue
    if not procs:
        return {"processes": 0}
    t0 = min((p.get("proc_start_wall") or p["t_site"]) for p in procs)
    rows = []
    for p in sorted(procs, key=lambda p: p.get("proc_start_wall") or p["t_site"]):
        cmd = " ".join(p.get("cmdline", []))[:120]
        role = "main"
        if "forkserver" in cmd:
            role = "zygote/forkserver"
        elif "resource_tracker" in cmd:
            role = "resource_tracker"
        elif "spawn_main" in cmd or p.get("forked_from"):
            role = "worker"
        rows.append({"pid": p["pid"], "ppid": p.get("ppid"), "role": role, "forked_from": p.get("forked_from"),
                     "start_s": round((p.get("proc_start_wall") or p["t_site"]) - t0, 3),
                     "modules_loaded": p.get("imports"),
                     "import_window_s": (round(p["t_last_import"] - p["t_first_import"], 3)
                                         if p.get("t_first_import") else None),
                     "top_packages": dict(sorted(p.get("by_package", {}).items(), key=lambda kv: -kv[1])[:6]),
                     "cmd": cmd})
    return {"processes": len(rows), "rows": rows}


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("cmd", choices=["env", "imports", "compileall", "spawn", "tree", "all"])
    ap.add_argument("--engine", default="sglang", choices=list(ROLES))
    ap.add_argument("--role", default="scheduler")
    ap.add_argument("--modes", default="asis,nopyc,prefix")
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--fake-cuda", choices=["auto", "on", "off"], default="auto")
    ap.add_argument("--trace-dir")
    ap.add_argument("--out")
    a = ap.parse_args(argv)
    fake = (a.fake_cuda == "on") or (a.fake_cuda == "auto" and not gpu_visible() and a.engine != "proxy")
    res: dict = {"cmd": a.cmd, "t": time.time()}
    if a.cmd in ("env", "all"):
        res["env"] = env_report(a.engine)
    if a.cmd in ("imports", "all"):
        roles = [a.role] if a.cmd == "imports" else list(ROLES[a.engine])
        res["imports"] = {r: imports_study(a.engine, r, a.modes.split(","), a.repeats, fake) for r in roles}
    if a.cmd in ("compileall", "all"):
        role = a.role if a.role in ROLES[a.engine] else list(ROLES[a.engine])[-1]
        study = (res.get("imports") or {}).get(role) or imports_study(a.engine, role, [], 0, fake)
        files = [f for f in study.get("manifest", []) if f.endswith(".py")]
        res["compileall"] = compileall_study(a.engine, role, files, a.repeats, fake)
    if a.cmd in ("spawn", "all"):
        from pystartup import spawnbench

        role = a.role if a.role in ROLES[a.engine] else "scheduler"
        facts = json.loads(fake_env()["PYSTARTUP_FAKE_CUDA"]) if fake else "probe"
        if fake:
            os.environ.update(fake_env())
        res["spawn"] = spawnbench.bench(ROLES[a.engine][role], a.workers, a.repeats, ["spawn", "zygote", "fork"],
                                        facts=facts)
    if a.cmd == "tree":
        res["tree"] = tree_summary(a.trace_dir)
    text = json.dumps(res, indent=1, default=str)
    if a.out:
        with open(a.out, "w") as f:
            f.write(text)
    print(text[:20000])
    return 0


if __name__ == "__main__":
    sys.exit(main())
