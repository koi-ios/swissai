"""servekit for vLLM on Modal: Apertus-8B at TP 1 and TP 2, slow store vs RAM staging.

Runs the prelim harness (serve-kit-vllm-prelim/) inside the exact vLLM image,
with the patched servekit (vendor/servekit, branch vllm-fast-loading) installed
so that its `vllm.general_plugins` entry point is live in every vLLM worker.

    python modal_app.py --dry-run                     # local, no Modal account: check inputs, print plan + cost
    modal run modal_app.py::download                  # once per account: weights into the hf-cache Volume (CPU)
    modal run modal_app.py::selftest                  # the harness's mock-engine selftest on Linux (CPU)
    modal run modal_app.py::smoke                     # TP 1, one repeat of 3 arms (~20 min on 1x A10G)
    modal run modal_app.py::experiment_tp1            # every arm, 3 cold repeats, TP 1
    modal run modal_app.py::experiment_tp2            # every arm, 3 cold repeats, TP 2 (2 GPUs)
    modal volume get sk-vllm-results / results/modal  # then: RESULTS=results/modal/<run> python ../../serve-kit-vllm-prelim/summarize.py

What this can and cannot show (also in MODAL.md):
- Modal has no Lustre. The "slow store" is a Modal Volume (network storage);
  the control is the same checkpoint copied to the container's local disk.
  Read speeds, and how much a Volume caches on its own side, are Modal's, not
  capstor's: these runs test the mechanism and the code, not Alps speedups.
- `/dev/shm` in a Modal container may be small. If it cannot hold a staged copy
  plus the stager's 10 GiB headroom, /dev/shm/servekit and the harness's SHM_DIR
  point at local disk instead, and env.json records `shm_fallback: true`.
- The page cache is dropped per file with posix_fadvise (evict.py); the
  Volume's own cache cannot be cleared from inside the container.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
SERVEKIT_SRC = HERE / "vendor" / "servekit"
PRELIM_SRC = HERE.parents[1] / "serve-kit-vllm-prelim"

VLLM_VERSION = "0.33.0"
VLLM_IMAGE = f"vllm/vllm-openai:v{VLLM_VERSION}"
MODEL_ID = "swiss-ai/Apertus-8B-Instruct-2509"
GPU = os.environ.get("SK_GPU", "A10G")          # the smallest GPU that holds Apertus-8B at TP 1
CPU, MEMORY_MIB = 8, 48 * 1024                   # staging is CPU- and RAM-bound, not GPU-bound
EXPERIMENT_TIMEOUT_S = 5 * 3600
SMOKE_TIMEOUT_S = 3600

# Every arm of run_arms.sh; the harness skips one whose loader this image lacks.
ARMS = ["default_net", "default_local", "lazy_net", "eager_net", "prefetch8_net", "prefetch32_net",
        "instanttensor_net", "fastsafetensors_net", "runai_net", "sharded_net", "runai_sharded_net",
        "staged_shm", "servekit_serial", "servekit_overlap"]
SMOKE_ARMS = ["default_net", "servekit_serial", "servekit_overlap"]
# A10G: 24 GB. Short contexts keep TP 1 inside it; the same flags at TP 2 keep runs comparable.
VLLM_EXTRA_ARGS = "--max-model-len 4096 --gpu-memory-utilization 0.90"

# Approximate prices (modal.com/pricing, Nov 2026): GPU per hour, plus the CPU and memory reservation.
PRICE_GPU_H = {"A10G": 1.10, "L40S": 1.95, "A100-80GB": 2.50, "H100": 3.95}
PRICE_CPU_CORE_H, PRICE_MEM_GIB_H = 0.047, 0.008


# --------------------------------------------------------------------------- Modal objects

def _modal_objects():
    import modal

    app = modal.App("servekit-vllm")
    image = (
        modal.Image.from_registry(VLLM_IMAGE)
        .entrypoint([])
        .apt_install("procps", "util-linux", "numactl", "rsync")
        # The loaders behind the native arms; an extra that will not install only skips its arm.
        .run_commands(*[f"pip install 'vllm[{x}]=={VLLM_VERSION}' || echo 'no {x} extra'"
                        for x in ("runai", "fastsafetensors", "instanttensor")])
        .pip_install("huggingface_hub[cli]")
        .add_local_dir(SERVEKIT_SRC, "/opt/servekit", copy=True,
                       ignore=[".git", "**/__pycache__", "**/*.pyc", ".pytest_cache"])
        # A real install, not PYTHONPATH: vLLM finds the plugin through its entry point.
        .run_commands("pip install --no-deps /opt/servekit",
                      "python -c \"from importlib.metadata import entry_points as e; "
                      "assert any(p.name == 'servekit' for p in e(group='vllm.general_plugins'))\"")
        .add_local_dir(PRELIM_SRC, "/opt/prelim", ignore=["results", "**/__pycache__", "**/*.pyc"])
    )
    hf = modal.Volume.from_name("hf-cache", create_if_missing=True)
    results = modal.Volume.from_name("sk-vllm-results", create_if_missing=True)
    secrets = [modal.Secret.from_name("huggingface")] if os.environ.get("SK_HF_SECRET", "1") == "1" else []
    return modal, app, image, hf, results, secrets


# --------------------------------------------------------------------------- what runs in the container

def _sh(cmd: str, env: dict, check: bool = True) -> int:
    print(f"\n$ {cmd}", flush=True)
    rc = subprocess.call(["bash", "-c", cmd], cwd="/opt/prelim", env=env)
    if check and rc != 0:
        raise RuntimeError(f"`{cmd}` exited {rc}")
    return rc


def _dir_bytes(path: Path) -> int:
    return sum(p.stat().st_size for p in path.rglob("*") if p.is_file()) if path.is_dir() else 0


def _shm_or_disk(need_bytes: int) -> dict:
    """Point the staging targets at local disk when /dev/shm cannot take a staged copy."""
    shm_free = shutil.disk_usage("/dev/shm").free
    headroom = 11 * 1024**3  # servekit's stager keeps 10 GiB free
    fact = {"shm_free_gib": round(shm_free / 1024**3, 1), "need_gib": round((need_bytes + headroom) / 1024**3, 1)}
    if shm_free >= need_bytes + headroom:
        return {**fact, "shm_fallback": False, "shm_root": "/dev/shm"}
    fallback = Path("/local/shm-fallback")
    (fallback / "servekit").mkdir(parents=True, exist_ok=True)
    link = Path("/dev/shm/servekit")
    if link.is_symlink() or not link.exists():
        if link.is_symlink():
            link.unlink()
        link.symlink_to(fallback / "servekit")
    return {**fact, "shm_fallback": True, "shm_root": str(fallback)}


def _run(tp: int, repeats: int, arms: list, sweep: bool, baseline: bool, hf, results, label: str) -> str:
    stamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    run_dir = Path("/results") / f"{stamp}_{label}"
    model_net = Path("/vol/models/apertus-8b")
    if not (model_net / "config.json").is_file():
        raise SystemExit("no checkpoint in the hf-cache Volume: run `modal run modal_app.py::download` first")
    shm = _shm_or_disk(_dir_bytes(model_net) // max(tp, 1))
    shm_root = shm["shm_root"]
    env = {
        **os.environ,
        "MODEL_ID": MODEL_ID, "MODEL_NAME": "Apertus-8B-Instruct-2509", "SERVED_NAME": "apertus-8b",
        "NET_ROOT": "/vol/models", "LOCAL_ROOT": "/local/models",
        "MODEL_NET": str(model_net), "MODEL_LOCAL": "/local/models/apertus-8b",
        "PRESHARD_NET": f"/vol/models/apertus-8b-tp{tp}", "SK_ARTIFACT": f"/vol/models/apertus-8b-sk-tp{tp}",
        "SHM_DIR": f"{shm_root}/apertus-8b-tp{tp}",
        "TP_LIST": str(tp), "ARMS_TP": str(tp), "REPEATS": str(repeats), "RESULTS": str(run_dir),
        "VLLM_EXTRA_ARGS": VLLM_EXTRA_ARGS, "WARM_CACHE_ROOT": "/local/sk-warm", "FRESH_CACHE_BASE": "/local/sk-caches",
        "PROVIDER": "modal", "INSTANCE_TYPE": f"{GPU}:{tp}, {CPU} CPU, {MEMORY_MIB // 1024} GiB",
        "ARMS_ONLY": " ".join(arms), "HF_HUB_OFFLINE": "1",
    }
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "modal_run.json").write_text(json.dumps({
        "label": label, "tp": tp, "repeats": repeats, "arms": arms, "gpu": GPU, "vllm_image": VLLM_IMAGE,
        "started": stamp, **shm}, indent=1))
    try:
        _sh("python collect_env.py && python detect_options.py --refresh", env)
        if not Path(env["MODEL_LOCAL"], "config.json").is_file():
            _sh('mkdir -p "$LOCAL_ROOT" && cp -a "$MODEL_NET/." "$MODEL_LOCAL/"', env)
        _sh('python evict.py --verify --json "$MODEL_NET" > "$RESULTS/evict_verify.json"', env, check=False)
        if "staged_shm" in arms or "sharded_net" in arms or "runai_sharded_net" in arms:
            _sh("./run_preshard.sh", env)
            hf.commit()
        if any(a.startswith("servekit") for a in arms):
            _sh("./run_servekit_prepare.sh", env)
            hf.commit()
        if sweep:
            _sh("./run_staging_sweep.sh", env, check=False)
        if baseline:
            _sh("./run_baseline.sh", env, check=False)
        _sh("./run_arms.sh", env, check=False)
        _sh("python summarize.py", env, check=False)
    finally:
        results.commit()
    print(f"\nresults in the sk-vllm-results Volume under /{run_dir.name}", flush=True)
    return run_dir.name


# --------------------------------------------------------------------------- the app

if os.environ.get("SK_MODAL_DRY_RUN") != "1":
    try:
        modal, app, image, hf, results, secrets = _modal_objects()
    except ImportError:  # `python modal_app.py --dry-run` works without modal installed
        app = None

    if app is not None:
        common = dict(image=image, cpu=CPU, memory=MEMORY_MIB, volumes={"/vol": hf, "/results": results})

        @app.function(image=image, volumes={"/vol": hf}, secrets=secrets, timeout=3600, cpu=4, memory=8192)
        def download() -> None:
            from huggingface_hub import snapshot_download

            dest = Path("/vol/models/apertus-8b")
            snapshot_download(MODEL_ID, local_dir=str(dest), allow_patterns=["*.json", "*.safetensors", "*.txt",
                                                                            "*.model", "*.jinja", "*.py"])
            hf.commit()
            print(f"{MODEL_ID} -> {dest}: {_dir_bytes(dest) / 1e9:.2f} GB", flush=True)

        @app.function(image=image, timeout=1800, cpu=4, memory=16384)
        def selftest() -> None:
            env = dict(os.environ)
            _shm_or_disk(0)
            rc = subprocess.call(["./tests/selftest.sh"], cwd="/opt/prelim", env=env)
            if rc != 0:
                raise SystemExit(f"selftest failed (rc={rc})")

        @app.function(gpu=GPU, timeout=SMOKE_TIMEOUT_S, **common)
        def smoke() -> str:
            return _run(1, 1, SMOKE_ARMS, sweep=False, baseline=False, hf=hf, results=results, label="smoke-tp1")

        @app.function(gpu=GPU, timeout=EXPERIMENT_TIMEOUT_S, **common)
        def experiment_tp1() -> str:
            return _run(1, 3, ARMS, sweep=True, baseline=True, hf=hf, results=results, label="arms-tp1")

        @app.function(gpu=f"{GPU}:2", timeout=EXPERIMENT_TIMEOUT_S, **common)
        def experiment_tp2() -> str:
            return _run(2, 3, ARMS, sweep=False, baseline=False, hf=hf, results=results, label="arms-tp2")


# --------------------------------------------------------------------------- dry run (local, no Modal)

# Container-hours per function, estimated: a cold Apertus-8B start on an A10G is ~1.5-3 min
# (load + torch.compile + graphs), plus verify on repeat 1, plus shutdown; see MODAL.md.
PLAN = [
    # name, GPUs, est. hours, what
    ("download", 0, 0.25, "weights into the hf-cache Volume"),
    ("selftest", 0, 0.1, "mock-engine selftest on Linux"),
    ("smoke", 1, 0.35, "TP1, 1 repeat: default_net, servekit_serial, servekit_overlap"),
    ("experiment_tp1", 1, 3.0, f"TP1, 3 cold repeats x {len(ARMS)} arms + staging sweep + Day-1 baseline"),
    ("experiment_tp2", 2, 2.5, f"TP2, 3 cold repeats x {len(ARMS)} arms"),
]


def estimate(gpu: str = GPU) -> list:
    rows = []
    for name, gpus, hours, what in PLAN:
        cpu, mem = (CPU, MEMORY_MIB / 1024) if gpus else (4, 8 if name == "download" else 16)
        per_h = gpus * PRICE_GPU_H[gpu] + cpu * PRICE_CPU_CORE_H + mem * PRICE_MEM_GIB_H
        rows.append((name, gpus, hours, round(per_h * hours, 2), what))
    return rows


def dry_run(gpu: str) -> int:
    problems = []
    if gpu not in PRICE_GPU_H:
        problems.append(f"unknown GPU {gpu!r}; one of {sorted(PRICE_GPU_H)}")
    pyproject = SERVEKIT_SRC / "pyproject.toml"
    if not pyproject.is_file() or 'vllm.general_plugins' not in pyproject.read_text():
        problems.append(f"{SERVEKIT_SRC} is not the patched servekit (no vllm.general_plugins entry point)")
    for f in ("run_once.py", "run_arms.sh", "run_preshard.sh", "run_servekit_prepare.sh", "run_staging_sweep.sh",
              "run_baseline.sh", "evict.py", "stage.py", "preshard.py", "summarize.py", "collect_env.py",
              "detect_options.py", "config.env", "tests/selftest.sh", "tests/mock_vllm.py"):
        if not (PRELIM_SRC / f).is_file():
            problems.append(f"missing {PRELIM_SRC / f}")
    arms_sh = (PRELIM_SRC / "run_arms.sh").read_text() if (PRELIM_SRC / "run_arms.sh").is_file() else ""
    for arm in ARMS:
        if f'"{arm}|' not in arms_sh:
            problems.append(f"run_arms.sh has no arm {arm}")
    try:
        import modal  # noqa: F401

        names = [n for n, *_ in PLAN if isinstance(globals().get(n), modal.Function)]
        missing = [n for n, *_ in PLAN if n not in names]
        if missing:
            problems.append(f"the app does not define {', '.join(missing)}")
        modal_note = f"modal {modal.__version__}: the app builds locally with functions {', '.join(names)}"
    except ImportError:
        modal_note = "modal is not installed here: install it to also check the app objects"
        names = []
    print(f"servekit source : {SERVEKIT_SRC}")
    print(f"harness         : {PRELIM_SRC}")
    print(f"image           : {VLLM_IMAGE} + servekit (pip install, entry point checked at build) + harness")
    print(f"GPU             : {gpu}; container {CPU} CPU, {MEMORY_MIB // 1024} GiB; vLLM args: {VLLM_EXTRA_ARGS}")
    print(f"modal           : {modal_note}")
    print(f"arms            : {', '.join(ARMS)}")
    print("\nfunction          GPUs  est. h  est. USD  what")
    total = 0.0
    for name, gpus, hours, usd, what in estimate(gpu):
        total += usd
        print(f"{name:<17} {gpus:>4}  {hours:>6.2f}  {usd:>8.2f}  {what}")
    print(f"{'total':<17} {'':>4}  {'':>6}  {total:>8.2f}  (estimate; check modal.com/pricing)")
    print("\nsplit: account A runs download, selftest, smoke, experiment_tp1; account B runs download and "
          "experiment_tp2 (Volumes are per account, so each downloads once)")
    if total > 25:
        problems.append(f"estimated total ${total:.2f} is over the $25 budget")
    for p in problems:
        print(f"! {p}")
    print("dry run: " + ("OK" if not problems else f"{len(problems)} problem(s)"))
    return 0 if not problems else 1


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--dry-run", action="store_true", help="check inputs, print the plan and cost; no Modal calls")
    ap.add_argument("--gpu", default=GPU, help=f"GPU type for the estimate (default {GPU})")
    a = ap.parse_args()
    if not a.dry_run:
        print("use `modal run modal_app.py::<function>`, or `python modal_app.py --dry-run`", file=sys.stderr)
        raise SystemExit(2)
    raise SystemExit(dry_run(a.gpu))
