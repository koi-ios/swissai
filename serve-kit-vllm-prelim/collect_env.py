#!/usr/bin/env python3
"""Record the setup in ENV.md (for people) and results/env.json (for summarize.py).

    python collect_env.py

Versions (vLLM, torch, CUDA, driver, servekit commit, loader extras), the GPUs and
how they are linked, CPU, RAM, /dev/shm, and the file system under each model path.
Set PROVIDER and INSTANCE_TYPE in the environment to record where it ran. Nothing
secret is read: no environment dump, no tokens.
"""
from __future__ import annotations

import json
import os
import platform
import re
import shutil
import subprocess
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path

RESULTS = Path(os.environ.get("RESULTS", "results"))
PATHS = {k: os.environ.get(k) for k in ("MODEL_NET", "MODEL_LOCAL", "PRESHARD_NET", "SHM_DIR", "FRESH_CACHE_BASE")}


def sh(*cmd: str, timeout: int = 60) -> str:
    if not shutil.which(cmd[0]):
        return ""
    try:
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        return p.stdout.strip() if p.returncode == 0 else ""
    except (OSError, subprocess.TimeoutExpired):
        return ""


def version(dist: str) -> str:
    try:
        return metadata.version(dist)
    except metadata.PackageNotFoundError:
        return ""


def servekit_commit() -> str:
    try:
        direct = metadata.distribution("servekit").read_text("direct_url.json")
    except metadata.PackageNotFoundError:
        return ""
    if not direct:
        return ""
    info = json.loads(direct)
    vcs = info.get("vcs_info", {})
    return f"{info.get('url', '')}@{vcs.get('commit_id', '')}".strip("@")


def torch_info() -> dict:
    out = sh("python3", "-c", "import torch,json;print(json.dumps({'torch':torch.__version__,"
             "'cuda_runtime':torch.version.cuda,'cudnn':torch.backends.cudnn.version()}))", timeout=120)
    try:
        return json.loads(out) if out else {}
    except json.JSONDecodeError:
        return {}


def storage(path: str) -> dict:
    if not path:
        return {}
    probe = Path(path)
    while not probe.exists() and probe != probe.parent:
        probe = probe.parent
    out = {"path": path, "exists": Path(path).exists()}
    fm = sh("findmnt", "-n", "-o", "SOURCE,FSTYPE,OPTIONS", "-T", str(probe))
    if fm:
        parts = fm.split(None, 2)
        out.update({"source": parts[0], "fstype": parts[1] if len(parts) > 1 else "",
                    "options": parts[2][:160] if len(parts) > 2 else ""})
    try:
        du = shutil.disk_usage(probe)
        out.update({"size_gb": round(du.total / 1e9, 1), "free_gb": round(du.free / 1e9, 1)})
    except OSError:
        pass
    if Path(path).is_dir():
        out["data_gb"] = round(sum(p.stat().st_size for p in Path(path).rglob("*") if p.is_file()) / 1e9, 2)
    return out


def model_revision(path: str) -> str:
    """`hf download --local-dir` leaves .cache/huggingface/download/*.metadata whose first line is the commit."""
    if not path:
        return ""
    for meta in Path(path).glob(".cache/huggingface/download/*.metadata"):
        first = meta.read_text().splitlines()[:1]
        if first and re.fullmatch(r"[0-9a-f]{40}", first[0].strip()):
            return first[0].strip()
    return ""


def main() -> int:
    gpus = [ln.split(", ") for ln in sh("nvidia-smi", "--query-gpu=name,memory.total,driver_version",
                                        "--format=csv,noheader").splitlines() if ln]
    smi = sh("nvidia-smi")
    cuda_driver = (re.search(r"CUDA Version:\s*([\d.]+)", smi) or [None, ""])[1]
    lscpu = sh("lscpu")
    cpu_model = (re.search(r"Model name:\s*(.+)", lscpu) or [None, platform.processor()])[1].strip()
    cpus = (re.search(r"^CPU\(s\):\s*(\d+)", lscpu, re.M) or [None, str(os.cpu_count())])[1]
    numa = (re.search(r"NUMA node\(s\):\s*(\d+)", lscpu) or [None, ""])[1]
    mem_gb = ""
    if Path("/proc/meminfo").exists():
        kb = int(re.search(r"MemTotal:\s*(\d+)", Path("/proc/meminfo").read_text()).group(1))
        mem_gb = f"{kb / 1e6:.0f}"
    os_name = ""
    if Path("/etc/os-release").exists():
        os_name = (re.search(r'PRETTY_NAME="([^"]+)"', Path("/etc/os-release").read_text()) or [None, ""])[1]
    t = torch_info()
    stores = {k: storage(v) for k, v in PATHS.items() if v}
    net = stores.get("MODEL_NET", {})
    env = {
        "date": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
        "provider": os.environ.get("PROVIDER", ""),
        "instance_type": os.environ.get("INSTANCE_TYPE", ""),
        "gpus": [{"name": g[0], "memory": g[1] if len(g) > 1 else "", "driver": g[2] if len(g) > 2 else ""}
                 for g in gpus],
        "gpu_summary": f"{len(gpus)}x {gpus[0][0]}" if gpus else "",
        "driver": gpus[0][2] if gpus and len(gpus[0]) > 2 else "",
        "cuda_driver": cuda_driver,
        "cuda_runtime": t.get("cuda_runtime", ""),
        "torch": t.get("torch", version("torch")),
        "vllm": version("vllm"),
        "servekit": version("servekit"),
        "servekit_source": servekit_commit(),
        "extras": {d: version(d) for d in ("flashinfer-python", "triton", "safetensors", "transformers",
                                           "huggingface-hub", "runai-model-streamer", "fastsafetensors",
                                           "instanttensor") if version(d)},
        "python": platform.python_version(),
        "os": os_name or platform.platform(),
        "kernel": platform.release(),
        "cpu": f"{cpu_model}, {cpus} CPUs" + (f", {numa} NUMA nodes" if numa else ""),
        "ram_gb": mem_gb,
        "shm": storage("/dev/shm"),
        "storage": stores,
        "net_storage": f"checkpoint on {net.get('fstype', '?')} ({net.get('source', '?')})" if net.get("fstype") else "",
        "model_revision": model_revision(PATHS.get("MODEL_NET") or ""),
        "gpu_topology": sh("nvidia-smi", "topo", "-m"),
    }
    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / "env.json").write_text(json.dumps(env, indent=1))

    def row(k, v):
        return f"| {k} | {v if v not in (None, '') else '–'} |"

    lines = ["# Environment", "", f"Recorded by `collect_env.py` on {env['date']}.", "",
             "| | |", "|---|---|",
             row("Provider / instance", " / ".join(x for x in (env["provider"], env["instance_type"]) if x)),
             row("GPUs", ", ".join(f"{g['name']} ({g['memory']})" for g in env["gpus"])),
             row("NVIDIA driver / CUDA (driver)", f"{env['driver']} / {env['cuda_driver']}"),
             row("CUDA runtime (torch)", env["cuda_runtime"]),
             row("vLLM", env["vllm"]), row("torch", env["torch"]),
             row("servekit", f"{env['servekit']} {env['servekit_source']}".strip()),
             row("Loader extras and libraries", ", ".join(f"{k} {v}" for k, v in env["extras"].items())),
             row("Model revision", env["model_revision"]),
             row("Python / OS / kernel", f"{env['python']} / {env['os']} / {env['kernel']}"),
             row("CPU", env["cpu"]), row("RAM (GB)", env["ram_gb"]),
             row("/dev/shm", f"{env['shm'].get('size_gb', '?')} GB ({env['shm'].get('fstype', '?')})"),
             "", "## Storage", "", "| role | path | file system | source | size GB | free GB | data GB |",
             "|---|---|---|---|---|---|---|"]
    for role, s in stores.items():
        lines.append(f"| {role} | {s.get('path')} | {s.get('fstype', '–')} | {s.get('source', '–')} | "
                     f"{s.get('size_gb', '–')} | {s.get('free_gb', '–')} | {s.get('data_gb', '–')} |")
    if env["gpu_topology"]:
        lines += ["", "## GPU topology (`nvidia-smi topo -m`)", "", "```text", env["gpu_topology"], "```"]
    lines += ["", "No Lustre here: these numbers show the mechanism on a rented VM's network volume, "
              "not Alps or capstor speedups.", ""]
    (Path(__file__).resolve().parent / "ENV.md").write_text("\n".join(lines))
    print("\n".join(lines[:20]))
    print("... written to ENV.md and results/env.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
