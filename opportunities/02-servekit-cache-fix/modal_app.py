"""Does servekit's JIT cache cover SGLang 0.5.20, and what does the fix buy?

SGLang 0.5.20 (lmsysorg/sglang:v0.5.20), TP2 on 2x H100 SXM. NVLink matters:
sglang's custom all-reduce v2 only comes up on a full-NVLink group, and it is
what JIT-compiles during torch.distributed init. Three arms, interleaved, 3 runs
each:

  none     no caches: every cache empty in every run
  main     servekit cd8f0b8: what its prepare keeps (Triton, TVM-FFI, FlashInfer)
  patched  servekit + patches/: the keyed namespace (every variable)

Every run is a new job on a new node: a fresh $HOME, $TMPDIR and node-local cache
dir, with only the artifact carried over (tools/run_arm.py). MODAL.md has the
runbook and the cost.

    python modal_app.py --dry-run          # the plan and its cost; no Modal account needed
    modal run modal_app.py::download       # CPU: the model into the hf-cache Volume
    modal run modal_app.py::smoke          # CPU: image builds, both servekits, unit tests on Linux
    modal run modal_app.py::gpu_check      # 2x H100 for ~1 min: driver, NVLink, CUDA
    modal run modal_app.py::experiment     # the A/B: ~40 min on 2x H100
"""
from __future__ import annotations

import json
import os
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
IMAGE_TAG = os.environ.get("SGLANG_IMAGE", "lmsysorg/sglang:v0.5.20")
MODEL = os.environ.get("MODEL", "Qwen/Qwen2.5-1.5B-Instruct")
TP = 2
GPU = f"H100:{TP}"
REPS = 3
SERVEKIT_REPO = "https://github.com/eth-easl/servekit"
SERVEKIT_BASE = "cd8f0b84477f607a8eeb07579ca3089ec42c90ad"
ENGINE_ARGS = ["--context-length", "8192", "--mem-fraction-static", "0.8"]
SERVEKIT = {"main": "/opt/servekit-main", "patched": "/opt/servekit-patched", "none": "/opt/servekit-patched"}
TIMEOUT_S = 3600  # the experiment's hard stop, which also caps its cost

# For the estimate only: modal.com/pricing, check before running.
H100_PER_HOUR = 3.95
MINUTES = {"setup": 8.0, "prepare": 4.0, "launch": 2.5}


def plan(reps: int = REPS) -> list:
    """The jobs, in order: two prepares, one unrecorded priming launch, then the
    arms interleaved and rotated so no arm always runs first or last."""
    steps = [
        {"kind": "prepare", "arm": "main"},
        {"kind": "prepare", "arm": "patched"},
        {"kind": "launch", "arm": "patched", "prime": True},
    ]
    orders = [("none", "main", "patched"), ("main", "patched", "none"), ("patched", "none", "main")]
    for rep in range(reps):
        for arm in orders[rep % len(orders)]:
            steps.append({"kind": "launch", "arm": arm, "rep": rep + 1})
    return steps


def estimate(steps: list) -> tuple:
    minutes = MINUTES["setup"] + sum(MINUTES[s["kind"]] for s in steps)
    return minutes, minutes / 60 * H100_PER_HOUR * TP


def step_command(step: dict, idx: int, model_dir: str, run_dir: str) -> list:
    cmd = [
        sys.executable, "/opt/tools/run_arm.py",
        "--servekit", SERVEKIT[step["arm"]],
        "--artifact", f"/work/artifacts/{step['arm']}",
        "--model", model_dir, "--tp", str(TP), "--run-dir", run_dir,
        "--port", str(30000 + 10 * idx),
    ]
    if step["kind"] == "prepare":
        cmd.append("--prepare")
    return cmd + ["--"] + ENGINE_ARGS


def run_id(step: dict, idx: int) -> str:
    rid = f"{idx:02d}-{step['kind']}-{step['arm']}"
    if step.get("rep"):
        rid += f"-r{step['rep']}"
    return rid + ("-prime" if step.get("prime") else "")


if __name__ == "__main__" and "--dry-run" in sys.argv:
    steps = plan()
    for i, s in enumerate(steps):
        print(f"{run_id(s, i):28s} " + " ".join(step_command(s, i, f"/work/models/{MODEL.split('/')[-1]}", f"/work/runs/{run_id(s, i)}")))
    minutes, dollars = estimate(steps)
    print(f"\n{len(steps)} jobs on {GPU} ({IMAGE_TAG}, {MODEL}): about {minutes:.0f} min, ${dollars:.2f} "
          f"at ${H100_PER_HOUR}/GPU-h; hard stop after {TIMEOUT_S // 60} min (${TIMEOUT_S / 3600 * H100_PER_HOUR * TP:.2f}).")
    sys.exit(0)

import modal  # noqa: E402

image = (
    modal.Image.from_registry(IMAGE_TAG)
    .apt_install("git")
    .pip_install("pytest", "safetensors", "numpy")
    .add_local_dir(HERE / "patches", "/opt/patches", copy=True)
    .run_commands(
        f"git clone -q {SERVEKIT_REPO} /opt/servekit-main && git -C /opt/servekit-main checkout -q {SERVEKIT_BASE}",
        f"git clone -q {SERVEKIT_REPO} /opt/servekit-patched && git -C /opt/servekit-patched checkout -q {SERVEKIT_BASE}"
        " && git -C /opt/servekit-patched -c user.name=modal -c user.email=modal@localhost am -q /opt/patches/*.patch",
    )
    .add_local_dir(HERE / "tools", "/opt/tools", copy=True, ignore=["__pycache__"])
)

app = modal.App("servekit-jit-cache-0520", image=image)
hf_cache = modal.Volume.from_name("hf-cache", create_if_missing=True)
results = modal.Volume.from_name("servekit-jit-results", create_if_missing=True)
secrets = [modal.Secret.from_name("huggingface")] if os.environ.get("HF_SECRET") else []


def _sh(cmd: str) -> str:
    out = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    return (out.stdout + out.stderr).strip()


@app.function(volumes={"/vol": hf_cache}, secrets=secrets, cpu=4, timeout=1800)
def download(model: str = MODEL) -> str:
    from huggingface_hub import snapshot_download

    path = snapshot_download(model, cache_dir="/vol/hub", allow_patterns=["*.json", "*.safetensors", "*.txt", "*.model", "*.tiktoken"])
    hf_cache.commit()
    print(path)
    return path


@app.function(cpu=4, timeout=1800)
def smoke() -> str:
    """No GPU: both servekit trees import, the patches applied, servekit's tests pass on Linux."""
    report = [
        _sh("python -c 'import sglang; print(\"sglang\", sglang.__version__)'"),
        _sh("git -C /opt/servekit-main log --oneline -1"),
        _sh("git -C /opt/servekit-patched log --oneline -4"),
        _sh("cd /opt/servekit-patched && PYTHONPATH=src python -m pytest tests -q -p no:cacheprovider 2>&1 | tail -5"),
        _sh("cd /opt/servekit-main && PYTHONPATH=src python -c 'import servekit.launch; print(\"main imports\")'"),
    ]
    text = "\n\n".join(report)
    print(text)
    return text


@app.function(gpu=GPU, timeout=600)
def gpu_check() -> str:
    """What the experiment depends on: the driver runs the image's CUDA, both GPUs, NVLink between them."""
    probe = (
        "import torch; print('torch', torch.__version__, 'cuda', torch.version.cuda, 'available', torch.cuda.is_available());"
        "print('devices', torch.cuda.device_count(), [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]);"
        "print('p2p 0->1', torch.cuda.device_count() > 1 and torch.cuda.can_device_access_peer(0, 1))"
    )
    key = (
        "import sys; sys.path.insert(0, '/opt/servekit-patched/src'); from servekit import jit_cache;"
        "k = jit_cache.validity_key('sglang'); print('servekit key', k.name); print(k.summary())"
    )
    report = [
        _sh("nvidia-smi --query-gpu=name,driver_version,compute_cap,memory.total --format=csv"),
        _sh("nvidia-smi topo -m"),
        _sh(f"python -c \"{probe}\""),
        _sh(f"python -c \"{key}\""),
        _sh("df -h /dev/shm /tmp | tail -2"),
    ]
    text = "\n\n".join(report)
    print(text)
    return text


def _wait_for_free_gpus(limit_s: float = 120) -> None:
    deadline = time.time() + limit_s
    while time.time() < deadline:
        if not _sh("nvidia-smi --query-compute-apps=pid --format=csv,noheader"):
            return
        time.sleep(2)


def _model_dir(model: str) -> str:
    """The model on local disk: a Volume read per launch would add its own noise."""
    from huggingface_hub import snapshot_download

    try:
        src = snapshot_download(model, cache_dir="/vol/hub", local_files_only=True)
    except Exception:  # noqa: BLE001 - not downloaded yet: do it here, on GPU time
        src = download.local(model)
    dest = Path("/work/models") / model.split("/")[-1]
    if not dest.exists():
        shutil.copytree(src, dest, symlinks=False)
    return str(dest)


def _none_artifact() -> None:
    """The patched artifact without its caches: same shards, every launch cold."""
    src, dest = Path("/work/artifacts/patched"), Path("/work/artifacts/none")
    dest.mkdir(parents=True, exist_ok=True)
    for f in src.iterdir():
        if f.is_file() and not (dest / f.name).exists():
            os.link(f, dest / f.name)


@app.function(gpu=GPU, volumes={"/vol": hf_cache, "/results": results}, secrets=secrets, timeout=TIMEOUT_S)
def experiment(model: str = MODEL, reps: int = REPS) -> str:
    sys.path.insert(0, "/opt/tools")
    import runlog
    import summarize_modal

    stamp = time.strftime("%Y%m%d-%H%M%S")
    out = Path("/results") / stamp
    out.mkdir(parents=True)
    model_dir = _model_dir(model)
    meta = {
        "kind": "meta", "model": model, "tp": TP, "image": IMAGE_TAG,
        "gpus": _sh("nvidia-smi --query-gpu=name,driver_version --format=csv,noheader | sort | uniq -c"),
        "servekit_base": SERVEKIT_BASE, "patched": _sh("git -C /opt/servekit-patched log --oneline -3"),
    }
    (out / "topology.txt").write_text(_sh("nvidia-smi topo -m"))
    rows = [meta]
    for idx, step in enumerate(plan(reps)):
        if step["arm"] == "none":
            _none_artifact()
        rid = run_id(step, idx)
        run_dir = Path("/work/runs") / rid
        run_dir.mkdir(parents=True, exist_ok=True)
        t0 = time.time()
        with open(run_dir / "engine.log", "w") as log:
            proc = subprocess.Popen(step_command(step, idx, model_dir, str(run_dir)), stdout=log,
                                    stderr=subprocess.STDOUT, start_new_session=True)
            try:
                proc.wait(timeout=1800)
            except subprocess.TimeoutExpired:
                pass
        with_leftovers = proc.poll() is None
        try:
            os.killpg(proc.pid, signal.SIGKILL)  # the engine's schedulers, if any outlived it
        except ProcessLookupError:
            pass
        proc.wait()
        _wait_for_free_gpus()
        if not (run_dir / "result.json").exists():
            (run_dir / "result.json").write_text(json.dumps({"rc": proc.returncode, "left_behind": {}}))
        row = {**runlog.read_run(run_dir), "run_id": rid, "arm": step["arm"], "rep": step.get("rep"),
               "prime": bool(step.get("prime")), "prepare": step["kind"] == "prepare",
               "job_wall_s": round(time.time() - t0, 1), "killed": with_leftovers}
        rows.append(row)
        keep = out / "runs" / rid
        keep.mkdir(parents=True)
        for name in ("engine.log", "report.json", "result.json"):
            if (run_dir / name).exists():
                shutil.copy2(run_dir / name, keep / name)
        print(f"{rid}: rc={row.get('rc')} dist_init={row['dist_init_s']} first_prefill={row['first_prefill_step_s']} "
              f"first_request={row['first_request_s']} total={row['total_s']} custom_ar={row['custom_all_reduce']}", flush=True)
        with open(out / "runs.jsonl", "w") as f:
            f.writelines(json.dumps(r) + "\n" for r in rows)
        results.commit()

    summary = summarize_modal.table([r for r in rows if r.get("kind") != "meta"])
    leftovers = summarize_modal.left_behind([r for r in rows if r.get("kind") != "meta"])
    text = f"{json.dumps(meta, indent=1)}\n\n{summary}\n\n{leftovers}\n"
    (out / "summary.md").write_text(text)
    results.commit()
    print(text)
    print(f"\nresults: modal volume get servekit-jit-results {stamp} results/modal/")
    return text
