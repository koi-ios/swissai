"""Does presharding round-trip a model? SGLang v0.5.20 on two GPUs, on Modal.

Validates servekit's raw preshard and the deep-state check against SGLang's own
sharded_state on DeepSeek-V2-Lite (MLA) and gpt-oss-20b (MXFP4). Runbook,
costs and expected outcomes: MODAL.md. The per-arm logic is modal_driver.py,
which also has a free local --dry-run.

    modal run modal_app.py::smoke                       # ~5 min on one L4: image, CUDA, plugin, patch anchors
    modal run modal_app.py --model dsv2-lite            # download + experiment on 2x L40S
    modal run modal_app.py --model gpt-oss-20b          # download + experiment on 2x H100
"""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import modal

HERE = Path(__file__).resolve().parent
SGLANG_IMAGE = os.environ.get("SGLANG_IMAGE", "lmsysorg/sglang:v0.5.20")
DEFAULT_GPU = {"dsv2-lite": "l40s", "gpt-oss-20b": "h100"}


def _servekit_commit() -> str:
    try:
        out = subprocess.run(["git", "-C", str(HERE / "vendor" / "servekit"), "rev-parse", "--short", "HEAD"],
                             capture_output=True, text=True, timeout=10)
        return out.stdout.strip() or "unknown"
    except Exception:  # noqa: BLE001 - only a label
        return "unknown"


app = modal.App("servekit-preshard-roundtrip")
hf_volume = modal.Volume.from_name("hf-cache", create_if_missing=True)
results_volume = modal.Volume.from_name("preshard-results", create_if_missing=True)

image = (
    modal.Image.from_registry(SGLANG_IMAGE)
    .env({
        "PYTHONUNBUFFERED": "1",
        "SERVEKIT_COMMIT": _servekit_commit(),
        "SGLANG_IMAGE": SGLANG_IMAGE,
    })
    # servekit on the preshard-roundtrip branch, installed so SGLang discovers
    # its plugin entry point (servekit._shim:install) in every scheduler.
    .add_local_dir(HERE / "vendor" / "servekit", "/opt/servekit", copy=True,
                   ignore=[".git", "**/__pycache__", "**/.pytest_cache", "tests/fixtures/*.log"])
    .run_commands("python -m pip install --no-deps --no-build-isolation -e /opt/servekit")
    .add_local_file(HERE / "modal_driver.py", "/opt/exp/modal_driver.py")
    .add_local_dir(HERE / "modal_scripts", "/opt/exp/modal_scripts")
)

GPU_RUN = dict(
    image=image,
    volumes={"/hf": hf_volume, "/results": results_volume},
    timeout=4 * 3600,
    cpu=8,
    memory=96 * 1024,  # page cache for the checkpoint and one artifact
    ephemeral_disk=200 * 1024,  # source + three artifacts of DeepSeek-V2-Lite (~31 GB each)
)


def _driver():
    sys.path.insert(0, "/opt/exp")
    import modal_driver

    return modal_driver


@app.function(image=image, volumes={"/hf": hf_volume}, timeout=2 * 3600, cpu=4, memory=8 * 1024)
def download(model: str) -> dict:
    """Fetch the checkpoint into the hf-cache Volume (CPU only). Idempotent."""
    from huggingface_hub import snapshot_download

    cfg = _driver().MODELS[model]
    target = Path("/hf/models") / model
    snapshot_download(cfg["repo"], local_dir=str(target), ignore_patterns=cfg["ignore"] + [".gitattributes"])
    hf_volume.commit()
    files = {p.name: p.stat().st_size for p in target.glob("*.safetensors")}
    return {"model": model, "files": len(files), "gb": round(sum(files.values()) / 1e9, 2)}


@app.function(image=image, gpu="L4", timeout=20 * 60)
def smoke() -> dict:
    """Cheap pre-flight: the image runs on Modal's driver, SGLang sees servekit, the patch fits."""
    out: dict = {}

    def sh(cmd: str) -> str:
        r = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        return (r.stdout + r.stderr).strip()[-3000:]

    out["nvidia_smi"] = sh("nvidia-smi --query-gpu=name,driver_version --format=csv,noheader")
    out["torch"] = sh("python -c 'import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())'")
    out["sglang"] = sh("python -c 'import sglang; print(sglang.__version__)'")
    out["entry_points"] = sh("python -c \"from importlib.metadata import entry_points; "
                             "print([e.value for e in entry_points(group='sglang.srt.plugins')])\"")
    out["hooks"] = sh(
        "python -c \"from sglang.srt.plugins import load_plugins; load_plugins(); "
        "from sglang.srt.plugins.hook_registry import HookRegistry; print(sorted(HookRegistry._hooks))\""
    )
    out["hook_targets_exist"] = sh(
        "python -c \"from sglang.srt.model_loader.loader import DefaultModelLoader as D; "
        "from sglang.srt.model_executor.model_runner import ModelRunner as R; "
        "print(all(hasattr(D, n) for n in ('_get_weights_iterator','load_model','commit_model_weights')) "
        "and hasattr(R, 'load_model'))\""
    )
    out["mla_patch_anchors"] = sh(
        "python -c \"from servekit._patches import sglang_35715 as p; t = p.target().read_text(); "
        "print(p.target(), t.count(p.OLD_INIT), [t.count(f) for f in p.OLD_FORWARDS])\""
    )
    out["gpt_oss_backend_rule"] = sh(
        "grep -n 'triton_kernel' $(python -c 'import sglang, os; print(os.path.dirname(sglang.__file__))')"
        "/srt/arg_groups/model_overrides/gpt_oss.py | head -3"
    )
    out["driver_dry_run"] = sh("cd /opt/exp && python modal_driver.py --model gpt-oss-20b --dry-run --results /tmp/dry | tail -3")
    for k, v in out.items():
        print(f"--- {k}\n{v}")
    return out


def _experiment(model: str, runs: int, steps: str) -> dict:
    driver = _driver()
    try:
        res = driver.run(model, Path("/hf/models"), Path("/tmp/exp"), Path("/results"), runs=runs,
                         steps=[s for s in steps.split(",") if s] or None)
    finally:
        results_volume.commit()
    return {
        "results": f"/results/{model}/{res['run_id']}/results.json",
        "arms": {k: v.get("status") for k, v in res["arms"].items()},
        "timing_weight_load_s": {k: [s.get("weight_load_s") for s in v] for k, v in res["timing"].items()},
        "deepstate_rc": {k: v.get("rc") for k, v in res["deepstate"].items()},
    }


@app.function(gpu="L40S:2", **GPU_RUN)
def experiment_l40s(model: str, runs: int = 3, steps: str = "") -> dict:
    return _experiment(model, runs, steps)


@app.function(gpu="H100:2", **GPU_RUN)
def experiment_h100(model: str, runs: int = 3, steps: str = "") -> dict:
    return _experiment(model, runs, steps)


@app.function(gpu="A100-80GB:2", **GPU_RUN)
def experiment_a100(model: str, runs: int = 3, steps: str = "") -> dict:
    return _experiment(model, runs, steps)


@app.local_entrypoint()
def main(model: str = "dsv2-lite", gpu: str = "", runs: int = 3, steps: str = "", skip_download: bool = False):
    if model not in DEFAULT_GPU:
        raise SystemExit(f"--model must be one of {sorted(DEFAULT_GPU)}")
    gpu = gpu or DEFAULT_GPU[model]
    fn = {"l40s": experiment_l40s, "h100": experiment_h100, "a100": experiment_a100}[gpu]
    if not skip_download:
        print(download.remote(model))
    print(fn.remote(model, runs, steps))
