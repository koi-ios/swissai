"""Modal validation for swiss-ai/model-launch#270 (persistent JIT/compile caches).

Experiments (each one Modal function, explicit timeouts, one container):

  selftest    CPU   cache-audit test suite on Linux (strace + inotify tracers)
  audit       H100  cache-audit on a real one-request warmup inside lmsysorg/sglang:v0.5.20 and
                    vllm/vllm-openai:v0.32.0: the ground-truth cache inventory per image, the
                    mapping verified by re-run, relocation + read-only behaviour, and the gate
                    result for the patched env TOMLs (harness/envs/*.toml)
  ab          H100  cold vs servekit's 3 variables vs all variables (harness/ab.py), Apertus-8B
                    TP1/TP2 and a small FP8 MoE, both engines; servekit-profile phases per run
  tilelang    H100  TileLang CUDA JIT: compile seconds per kernel cold vs warm (vLLM image)
  gpu_stress  H100  N processes JIT the same Triton / FlashInfer kernels into one dir, then load all

Run (see MODAL.md for the per-account split and costs):

  python modal_app.py --dry-run                        # local: plan + cost, no Modal calls
  modal run modal_app.py::smoke                        # ~5 min: selftest + one tiny audit
  modal run modal_app.py::plan_a                       # SGLang suite      (account A)
  modal run modal_app.py::plan_b                       # vLLM suite        (account B)
  modal run modal_app.py::plan_c                       # audits + stress + TileLang (account C)
  modal volume get cache-270-results / results/modal/   # fetch, then:
  python harness/summarize.py results/modal/ab/*.json > results/ab_summary.md
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent

try:
    import modal
except ImportError:  # the dry run works without modal installed
    modal = None

SGLANG_IMAGE = "lmsysorg/sglang:v0.5.20"
VLLM_IMAGE = "vllm/vllm-openai:v0.32.0"
MODELS = {
    "apertus8b": "swiss-ai/Apertus-8B-Instruct-2509",
    "moe_fp8": "Qwen/Qwen3-30B-A3B-Instruct-2507-FP8",   # 30 GB, FP8 block-scaled: DeepGEMM + FlashInfer MoE paths
    "tiny": "Qwen/Qwen2.5-0.5B-Instruct",                  # audit warmups
}
H100_PER_H = 3.95  # USD, approximate (modal.com/pricing)

# (function, engine, model key, tp, est. GPU minutes) -- estimates, not measurements.
# TP2 runs only cold and full (servekit3 is compared at TP1), 3 measured runs per arm.
PLAN = {
    "a": [("ab", "sglang", "apertus8b", 1, 35), ("ab", "sglang", "apertus8b", 2, 25),
          ("ab", "sglang", "moe_fp8", 1, 45)],
    "b": [("ab", "vllm", "apertus8b", 1, 40), ("ab", "vllm", "apertus8b", 2, 28),
          ("ab", "vllm", "moe_fp8", 1, 50)],
    "c": [("audit", "sglang", "tiny", 1, 20), ("audit", "vllm", "tiny", 1, 25),
          ("tilelang", "vllm", "-", 1, 10), ("gpu_stress", "sglang", "-", 1, 10)],
}

REMOTE = "/root/opp"
IGNORE = ["vendor", ".venv", "**/__pycache__", "results", "research", ".pytest_cache", "cache-audit-runs"]


def dry_run() -> int:
    """Validate locally: harness files exist and import, plan + cost estimate."""
    sys.path.insert(0, str(HERE))
    sys.path.insert(0, str(HERE / "integration"))
    import importlib

    for mod in ("cache_audit.cli", "cache_audit.audit", "sml_jit_cache"):
        importlib.import_module(mod)
    for f in ("harness/ab.py", "harness/warmup_once.py", "harness/summarize.py", "harness/envs/vllm_base.toml",
              "harness/envs/sglang_base.toml", "stress/stress_writers.py", "kb/caches.toml"):
        assert (HERE / f).exists(), f
        if f.endswith(".py"):
            compile((HERE / f).read_text(), f, "exec")
    total = 0.0
    print(f"modal installed: {modal is not None}; images: {SGLANG_IMAGE}, {VLLM_IMAGE}")
    for plan, items in PLAN.items():
        cost = 0.0
        print(f"\nplan_{plan}:")
        for fn, engine, model, tp, minutes in items:
            gpus = tp
            c = minutes / 60 * H100_PER_H * gpus
            cost += c
            print(f"  {fn:10s} {engine:6s} {MODELS.get(model, model):40s} TP{tp}  ~{minutes} min x {gpus} H100  ~${c:.2f}")
        print(f"  estimate: ~${cost:.2f} (+ CPU for image pulls and weight downloads, < $1)")
        total += cost
    print(f"\ntotal estimate ~${total:.2f} (budget $25); every number above is an estimate, not a measurement")
    return 0


if modal is not None:
    app = modal.App("cache-270")
    hf_vol = modal.Volume.from_name("hf-cache", create_if_missing=True)
    res_vol = modal.Volume.from_name("cache-270-results", create_if_missing=True)
    shared_vol = modal.Volume.from_name("cache-270-shared", create_if_missing=True)
    VOLS = {"/hf": hf_vol, "/results": res_vol, "/shared": shared_vol}
    try:
        SECRETS = [modal.Secret.from_name("huggingface")]
    except Exception:  # optional: only gated models need it
        SECRETS = []

    def _engine_image(ref: str) -> "modal.Image":
        return (
            modal.Image.from_registry(ref)
            .entrypoint([])
            .run_commands("command -v python >/dev/null || ln -sf \"$(command -v python3)\" /usr/local/bin/python")
            .apt_install("strace")
            .add_local_dir(str(HERE / "vendor" / "servekit"), "/root/servekit", copy=True,
                           ignore=["**/__pycache__", ".git"])
            .run_commands("python -m pip install --no-deps /root/servekit pytest")
            .add_local_dir(str(HERE), REMOTE, ignore=IGNORE)
        )

    sglang_img = _engine_image(SGLANG_IMAGE)
    vllm_img = _engine_image(VLLM_IMAGE)
    cpu_img = (modal.Image.debian_slim(python_version="3.12").apt_install("strace")
               .pip_install("pytest", "huggingface_hub").add_local_dir(str(HERE), REMOTE, ignore=IGNORE))

    def _sh(cmd: list[str], env: dict | None = None, timeout: float | None = None) -> int:
        print("+", " ".join(cmd), flush=True)
        return subprocess.run(cmd, cwd=REMOTE, env={**os.environ, **(env or {})}, timeout=timeout).returncode

    @app.function(image=cpu_img, volumes={"/hf": hf_vol}, secrets=SECRETS, timeout=3600)
    def download(model_key: str) -> str:
        from huggingface_hub import snapshot_download
        repo = MODELS[model_key]
        path = snapshot_download(repo, local_dir=f"/hf/{repo}",
                                 allow_patterns=["*.json", "*.safetensors", "*.model", "*.txt", "*.py", "*.jinja"])
        hf_vol.commit()
        return path

    @app.function(image=cpu_img, timeout=1200)
    def selftest() -> int:
        return _sh([sys.executable, "-m", "pytest", "-q", "-m", "not slow", "tests/"])

    def _engine_cmd(engine: str, model_dir: str, tp: int) -> list[str]:
        if engine == "sglang":
            return ["python", "-m", "sglang.launch_server", "--model-path", model_dir, "--tp", str(tp),
                    "--port", "8080", "--served-model-name", "m", "--context-length", "4096",
                    "--mem-fraction-static", "0.8"]
        return ["vllm", "serve", model_dir, "--tensor-parallel-size", str(tp), "--port", "8080",
                "--served-model-name", "m", "--max-model-len", "4096", "--gpu-memory-utilization", "0.8"]

    def _audit(engine: str) -> dict:
        model_dir = f"/hf/{MODELS['tiny']}"
        out = f"/results/audit/{engine}"
        warm = ["python", f"{REMOTE}/harness/warmup_once.py", "--", *_engine_cmd(engine, model_dir, 1)]
        codes = {}
        codes["auto"] = _sh(["python", "-m", "cache_audit", "run", "--tracer", "auto", "--relocate", "--readonly",
                             "--timeout", "2400", "--workdir", f"/tmp/ca-{engine}", "--out", out, "-v", "--", *warm])
        codes["gate_patched_env"] = _sh(["python", "-m", "cache_audit", "run", "--tracer", "auto", "--no-warm",
                                         "--env-toml", f"{REMOTE}/harness/envs/{engine}_base.toml",
                                         "--timeout", "2400", "--workdir", f"/tmp/ca-gate-{engine}",
                                         "--out", f"{out}-gate", "--", *warm])
        _sh(["sh", "-c", f"python -m cache_audit static > {out}/static.md; echo exit $? >> {out}/static.md"])
        res_vol.commit()
        Path(f"{out}/codes.json").write_text(json.dumps(codes))
        res_vol.commit()
        return codes

    @app.function(image=sglang_img, gpu="H100", volumes=VOLS, timeout=3 * 3600)
    def audit_sglang() -> dict:
        return _audit("sglang")

    @app.function(image=vllm_img, gpu="H100", volumes=VOLS, timeout=3 * 3600)
    def audit_vllm() -> dict:
        return _audit("vllm")

    def _ab(engine: str, model_key: str, tp: int, repeats: int, arms: str) -> int:
        model_dir = f"/hf/{MODELS[model_key]}"
        # weights into the page cache first: this experiment measures compile caches, not weight I/O
        _sh(["sh", "-c", f"cat {model_dir}/*.safetensors > /dev/null"])
        out = f"/results/ab/{engine}-{model_key}-tp{tp}.json"
        code = _sh(["python", f"{REMOTE}/harness/ab.py", "--engine", engine, "--model-dir", model_dir, "--tp", str(tp),
                    "--arms", arms, "--repeats", str(repeats), "--shared-root", "/shared", "--out", out])
        shared_vol.commit()
        res_vol.commit()
        return code

    @app.function(image=sglang_img, gpu="H100", volumes=VOLS, timeout=6 * 3600)
    def ab_sglang(model_key: str, tp: int = 1, repeats: int = 3, arms: str = "cold,servekit3,full") -> int:
        return _ab("sglang", model_key, tp, repeats, arms)

    @app.function(image=vllm_img, gpu="H100", volumes=VOLS, timeout=6 * 3600)
    def ab_vllm(model_key: str, tp: int = 1, repeats: int = 3, arms: str = "cold,servekit3,full") -> int:
        return _ab("vllm", model_key, tp, repeats, arms)

    @app.function(image=vllm_img, gpu="H100", volumes=VOLS, timeout=3600)
    def tilelang() -> int:
        return _sh(["python", f"{REMOTE}/stress/tilelang_cuda_bench.py", "--out", "/results/tilelang_bench.json"])

    @app.function(image=sglang_img, gpu="H100", volumes=VOLS, timeout=3600)
    def gpu_stress() -> int:
        code = _sh(["python", f"{REMOTE}/stress/stress_writers.py", "--libs", "triton,flashinfer", "--n", "8",
                    "--repeats", "3", "--out", "/results/stress_gpu.json"])
        res_vol.commit()
        return code

    @app.local_entrypoint()
    def smoke() -> None:
        print("selftest:", selftest.remote())
        download.remote("tiny")
        print("audit vllm:", audit_vllm.remote())

    @app.local_entrypoint()
    def plan_a() -> None:
        for key in ("apertus8b", "moe_fp8"):
            download.remote(key)
        print(ab_sglang.remote("apertus8b", 1))
        print(ab_sglang.with_options(gpu="H100:2").remote("apertus8b", 2, arms="cold,full"))
        print(ab_sglang.remote("moe_fp8", 1))

    @app.local_entrypoint()
    def plan_b() -> None:
        for key in ("apertus8b", "moe_fp8"):
            download.remote(key)
        print(ab_vllm.remote("apertus8b", 1))
        print(ab_vllm.with_options(gpu="H100:2").remote("apertus8b", 2, arms="cold,full"))
        print(ab_vllm.remote("moe_fp8", 1))

    @app.local_entrypoint()
    def plan_c() -> None:
        print("selftest:", selftest.remote())
        download.remote("tiny")
        print("audit sglang:", audit_sglang.remote())
        print("audit vllm:", audit_vllm.remote())
        print("gpu stress:", gpu_stress.remote())
        print("tilelang:", tilelang.remote())


if __name__ == "__main__":
    if "--dry-run" in sys.argv or modal is None:
        sys.exit(dry_run())
    print(__doc__)
