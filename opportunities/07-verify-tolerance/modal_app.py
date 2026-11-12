"""Modal app: the verify-tolerance study on real engines (SGLang 0.5.20, vLLM 0.33.0) on one H100.

For each model, one container runs `study_driver.Study` end to end: a clean
reference server with `servekit verify`'s calibration, held-out load, restarts,
other engine settings, the batch-invariant mode, and one fresh server per
injected fault. Results land in the `verify-tolerance-results` Volume in the
same format as the CPU study, so `python -I cpu/report.py <dir>` analyses them.

    python modal_app.py plan                        # local, no Modal account: plan, cost, fault-grid check
    modal run modal_app.py --mode download --model qwen3-1.7b
    modal run modal_app.py --mode smoke --model qwen3-1.7b
    modal run --detach modal_app.py --mode study --model qwen3-1.7b

See MODAL.md for the full runbook, costs and how to split them across accounts.
"""
from __future__ import annotations

import json
import shutil
import sys
import time
from pathlib import Path

import modal

HERE = Path(__file__).resolve().parent
SERVEKIT_SRC = HERE / "vendor" / "servekit" / "src"
sys.path.insert(0, str(SERVEKIT_SRC))
sys.path.insert(0, str(HERE))

APP_NAME = "verify-tolerance"
SGLANG_IMAGE = "lmsysorg/sglang:v0.5.20"
VLLM_IMAGE = "vllm/vllm-openai:v0.33.0"
GPU = "H100"
CPU_CORES = 8.0
MEMORY_MIB = 65536
TIMEOUT_S = 3 * 3600  # hard stop; the driver's --budget stops starting servers well before
# modal.com/pricing, 12 Nov 2026 (per second): H100 $0.001097, CPU $0.0000131/core, memory $0.00000222/GiB
PRICE_PER_S = 0.001097 + CPU_CORES * 0.0000131 + (MEMORY_MIB / 1024) * 0.00000222

# Per-model time budget in minutes: the driver starts no new server after it.
BUDGET_MIN = {"qwen3-1.7b": 60, "qwen3-1.7b-fp8": 50, "gpt-oss-20b": 90}

app = modal.App(APP_NAME)
hf_cache = modal.Volume.from_name("hf-cache", create_if_missing=True)
results = modal.Volume.from_name("verify-tolerance-results", create_if_missing=True)

_ENV = {"HF_HOME": "/hf/home", "PYTHONUNBUFFERED": "1", "TOKENIZERS_PARALLELISM": "false"}


def _with_code(image: modal.Image) -> modal.Image:
    return (image.env(_ENV)
            .add_local_dir(str(SERVEKIT_SRC), "/opt/servekit/src")
            .add_local_file(str(HERE / "study_driver.py"), "/opt/study/study_driver.py"))


download_image = modal.Image.debian_slim(python_version="3.12").pip_install("huggingface_hub==0.36.0").env(_ENV)
# Modal wants `python` and `pip` on PATH; engine images may ship only python3 (no-ops when both exist).
_PY_SETUP = [
    'RUN command -v python >/dev/null 2>&1 || ln -sf "$(command -v python3)" /usr/local/bin/python',
    "RUN python -m pip --version >/dev/null 2>&1 || python -m ensurepip --upgrade "
    "|| (apt-get update && apt-get install -y --no-install-recommends python3-pip)",
]
sglang_image = _with_code(modal.Image.from_registry(SGLANG_IMAGE, setup_dockerfile_commands=_PY_SETUP).entrypoint([]))
vllm_image = _with_code(modal.Image.from_registry(VLLM_IMAGE, setup_dockerfile_commands=_PY_SETUP).entrypoint([]))


def _models():
    import study_driver as sd

    return sd.MODELS


@app.function(image=download_image, volumes={"/hf": hf_cache}, timeout=3600, cpu=4.0, memory=8192)
def download(hf_id: str) -> str:
    """Weights into the shared `hf-cache` Volume (ungated models; add a `huggingface` Secret for gated ones)."""
    from huggingface_hub import snapshot_download

    # gpt-oss ships a second copy of its weights under original/ and metal/: the engines need neither.
    path = snapshot_download(hf_id, local_dir=f"/hf/models/{hf_id}",
                             allow_patterns=["*.safetensors", "*.json", "*.txt", "*.model", "tokenizer*", "*.jinja"],
                             ignore_patterns=["original/*", "metal/*"])
    hf_cache.commit()
    size = sum(p.stat().st_size for p in Path(path).rglob("*") if p.is_file())
    return f"{hf_id}: {size / 1e9:.1f} GB in {path}"


def _local_copy(hf_id: str, key: str) -> Path:
    """The checkpoint on the container's local disk, so corrupted copies can hard-link to it."""
    src = Path(f"/hf/models/{hf_id}")
    if not (src / "config.json").exists():
        raise RuntimeError(f"{hf_id} is not in the hf-cache Volume; run --mode download first")
    dst = Path("/tmp/models") / key / "clean"
    if not dst.exists():
        t0 = time.time()
        shutil.copytree(src, dst, ignore=shutil.ignore_patterns(".cache", "original", "metal"))
        print(f"[modal] copied {hf_id} to local disk in {time.time() - t0:.0f}s", flush=True)
    return dst


def _study(model: str, engine: str, quick: bool, budget_min: float, max_faults: int) -> str:
    sys.path.insert(0, "/opt/study")
    sys.path.insert(0, "/opt/servekit/src")
    import study_driver as sd

    spec = sd.MODELS[model]
    if max_faults < len(spec.faults):
        spec.faults = spec.faults[:max_faults]
    clean = _local_copy(spec.hf_id, model)
    out = Path("/results") / model / f"{model}-{engine}.jsonl.gz"
    plan = sd.QUICK if quick else sd.Plan()
    print(f"[modal] {model} on {engine}: {sd.estimate(spec, engine, plan)}", flush=True)
    t0 = time.time()
    sd.Study(spec, engine, clean, Path("/tmp/models") / model / "work", out, plan, budget_min,
             on_record=results.commit).run()
    results.commit()
    spent = time.time() - t0
    return f"{model} on {engine}: {spent / 60:.1f} min, about ${spent * PRICE_PER_S:.2f}; results in {out}"


_GPU_KW = dict(gpu=GPU, cpu=CPU_CORES, memory=MEMORY_MIB, timeout=TIMEOUT_S,
               volumes={"/hf": hf_cache, "/results": results})


@app.function(image=sglang_image, **_GPU_KW)
def study_sglang(model: str, quick: bool = False, budget_min: float = 60.0, max_faults: int = 99) -> str:
    return _study(model, "sglang", quick, budget_min, max_faults)


@app.function(image=vllm_image, **_GPU_KW)
def study_vllm(model: str, quick: bool = False, budget_min: float = 45.0, max_faults: int = 8) -> str:
    return _study(model, "vllm", quick, budget_min, max_faults)


@app.function(image=sglang_image, gpu=GPU, cpu=CPU_CORES, memory=MEMORY_MIB, timeout=1800,
              volumes={"/hf": hf_cache, "/results": results})
def smoke(model: str = "qwen3-1.7b") -> str:
    """One server, one capture, one corrupted checkpoint loaded and captured: ~5 min, checks image and flags."""
    sys.path.insert(0, "/opt/study")
    sys.path.insert(0, "/opt/servekit/src")
    import study_driver as sd
    from servekit import faults
    from servekit.verify import capture, check, load_prompt_set, weights_checksum

    spec = sd.MODELS[model]
    clean = _local_copy(spec.hf_id, model)
    prompt_set, prompts = load_prompt_set(None)
    lines = []
    srv = sd.Server("sglang", clean, [], {}, Path("/results/smoke") / f"{model}-clean.log", spec)
    t = srv.start(900)
    try:
        ref = capture(srv.url, "m", prompts, prompt_set)
        w = weights_checksum(srv.url)
        lines.append(f"clean server ready in {t:.0f}s; {sum(p['n_tokens'] for p in ref['prompts'])} tokens captured; "
                     f"/weights_checker {'present' if w else 'absent'}")
    finally:
        srv.stop()
    cls, mag, fspec = spec.faults[0]
    bad = Path("/tmp/models") / model / "smoke-fault"
    faults.inject(clean, bad, faults.FaultSpec.from_dict(fspec))
    srv = sd.Server("sglang", bad, [], {}, Path("/results/smoke") / f"{model}-fault.log", spec)
    t = srv.start(900)
    try:
        got = capture(srv.url, "m", prompts, prompt_set)
        res = check(ref, got, mode="exact")
        lines.append(f"faulty server ({cls}, {mag}) ready in {t:.0f}s; exact check passed={res.passed} "
                     f"(max |delta| {res.decision.stats.max_abs_delta:.3g})")
    finally:
        srv.stop()
    results.commit()
    return "\n".join(lines)


def plan(models=None) -> None:
    """Print the plan and its cost, and check every fault spec against the real checkpoint headers."""
    import study_driver as sd
    from servekit import faults

    try:
        from huggingface_hub import get_safetensors_metadata
    except ImportError:
        get_safetensors_metadata = None
    rows = [("qwen3-1.7b", "sglang"), ("qwen3-1.7b-fp8", "sglang"), ("gpt-oss-20b", "sglang"), ("qwen3-1.7b", "vllm")]
    if models:
        rows = [r for r in rows if r[0] in models]
    total = 0.0
    print(f"price per container-second: ${PRICE_PER_S:.6f} ({GPU} + {CPU_CORES:g} cores + {MEMORY_MIB // 1024} GiB)")
    for model, engine in rows:
        spec = sd.MODELS[model]
        if engine == "vllm":
            spec.faults = spec.faults[:8]
        est = sd.estimate(spec, engine, sd.Plan())
        budget = BUDGET_MIN.get(model, 60) if engine == "sglang" else 45
        minutes = min(est["minutes"], budget + spec.start_s / 60 + 5)
        cost = minutes * 60 * PRICE_PER_S
        total += cost
        print(f"{model:16s} {engine:6s} {est['server_starts']:3d} starts {est['captures']:4d} captures "
              f"~{est['minutes']:5.1f} min (budget {budget} min) -> ~${cost:.2f}")
        if get_safetensors_metadata is not None:
            md = get_safetensors_metadata(spec.hf_id)
            meta = {n: (ti.dtype, tuple(ti.shape)) for fm in md.files_metadata.values() for n, ti in fm.tensors.items()}
            for cls, mag, fspec in spec.faults:
                try:
                    touched = faults.validate(faults.FaultSpec.from_dict(fspec), meta)
                    print(f"    ok   {cls:22s} {mag:36s} -> {len(touched)} tensor(s), e.g. {touched[0]}")
                except ValueError as e:
                    print(f"    FAIL {cls:22s} {mag:36s} -> {e}")
    print(f"total, all four arms: ~${total:.2f} (estimate; the budgets cap it)")


@app.local_entrypoint()
def main(mode: str = "plan", model: str = "qwen3-1.7b", engine: str = "sglang", quick: bool = False,
         budget_min: float = 0.0, max_faults: int = 99):
    if mode == "plan":
        plan([model] if model != "all" else None)
    elif mode == "download":
        ids = sorted({m.hf_id for k, m in _models().items() if k != "qwen3-0.6b"}) if model == "all" \
            else [_models()[model].hf_id]
        for line in download.map(ids):
            print(line)
    elif mode == "smoke":
        print(smoke.remote(model))
    elif mode == "study":
        budget = budget_min or (BUDGET_MIN.get(model, 60) if engine == "sglang" else 45)
        fn = study_sglang if engine == "sglang" else study_vllm
        print(fn.remote(model, quick, budget, max_faults))
    else:
        raise SystemExit(f"unknown mode {mode!r}: plan | download | smoke | study")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "plan":
        plan(sys.argv[2:] or None)
    else:
        print(__doc__)
