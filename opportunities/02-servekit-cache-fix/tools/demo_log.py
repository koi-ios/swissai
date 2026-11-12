"""Print what servekit's log says in the three situations the keyed caches change.

No GPU: fake engines stand in for sglang, through the same seams the tests use.
The output is the source of the log excerpts in drafts/PR.md.

    PYTHONPATH=vendor/servekit/src python tools/demo_log.py > results/demo-log.txt
"""
import json
import os
import sys
import tempfile
from pathlib import Path
from unittest import mock

from servekit import jit_cache, launch as launch_mod, prepare as prepare_mod
from servekit.cli import main
from servekit.manifest import Manifest
from servekit.profile import ProfileReport
from servekit.stage import StageResult

WARMUP = '''
import os, sys
from pathlib import Path
leaf = Path(os.environ["SGLANG_JIT_CACHE_DIR"]) / "sm90" / "sgl_kernel_jit_communicator" / "build-0" / "deps-0"
leaf.mkdir(parents=True, exist_ok=True)
(leaf / "sgl_kernel_jit_communicator.so").write_bytes(b"x" * 2_400_000)
t = Path(os.environ["TRITON_CACHE_DIR"]) / "ABCD"
t.mkdir(parents=True, exist_ok=True)
(t / "rms_norm.cubin").write_bytes(b"x" * 90_000)
'''


def section(title):
    print(f"\n===== {title} =====", flush=True)


def fake_engine(write):
    def run_profile(command, ready_timeout=1800.0, on_ready=None, head=True, env=None):
        write({**os.environ, **(env or {})})
        report = ProfileReport(command="", started_at=0.0, ready_at=1.0, success=True, framework="sglang")
        on_ready(report)
        return report
    return run_profile


def serve(artifact, model, out, *flags):
    return main(["launch", "--servekit-artifact-path", str(artifact), "--out", str(out), *flags,
                 "--", "python", "-m", "sglang.launch_server", "--model-path", str(model), "--tp", "4"])


def main_demo():
    root = Path(tempfile.mkdtemp(prefix="servekit-demo-"))
    home = root / "home"
    home.mkdir()
    os.environ.update(HOME=str(home), USER="demo")
    tempfile.tempdir = str(root / "tmp")
    Path(tempfile.tempdir).mkdir()
    model = root / "Llama-3.1-70B-Instruct"
    model.mkdir()
    artifact = root / "llama70b-tp4"
    artifact.mkdir()
    (artifact / "config.json").write_text("{}")
    for r in range(4):
        (artifact / f"model-rank-{r}-part-0.safetensors").write_bytes(b"w")
    Manifest(format="sharded_state", engine="sglang", engine_version="0.5.16", tp_size=4, pp_size=1,
             dp_size=1, dtype="bfloat16", quantization=None, source=str(model)).write(artifact)

    versions = {"sglang": "0.5.20", "torch": "2.13.0", "flashinfer-python": "0.6.18", "apache-tvm-ffi": "0.1.11"}

    def version(name):
        if name in versions:
            return versions[name]
        raise jit_cache.metadata.PackageNotFoundError(name)

    old = jit_cache.CacheKey({
        "layout": 1, "engine": "sglang", "gpu": ["9.0"], "cpu": "aarch64", "python": "3.12", "cuda": "12.9.1",
        "image": {}, "packages": {"sglang": "0.5.16", "torch": "2.9.1", "flashinfer-python": "0.6.1",
                                  "apache-tvm-ffi": "0.1.8"},
    })
    build = root / "old-build"
    (build / "tvm-ffi").mkdir(parents=True)
    (build / "tvm-ffi" / "comm.so").write_bytes(b"x" * 1000)
    jit_cache.persist(build, artifact, old)

    script = root / "fake_warmup.py"
    script.write_text(WARMUP)
    patches = [
        mock.patch.object(jit_cache.metadata, "version", version),
        mock.patch.object(jit_cache, "gpu_capabilities", lambda: ("9.0",)),
        mock.patch.object(jit_cache.platform, "machine", lambda: "aarch64"),
        mock.patch.object(jit_cache, "_cuda_toolkit", lambda environ: "13.0.3"),
        mock.patch.object(launch_mod, "stage", lambda src, dest, slices=64, file_pattern="*": StageResult(dest, 0.1, 1.0, 1)),
        mock.patch.object(launch_mod, "DEFAULT_ROOT", root / "shm"),
        mock.patch.object(launch_mod, "DEFAULT_CACHE_ROOT", root / "node-local"),
        mock.patch.object(prepare_mod, "SAVE_SCRIPT", script),
    ]
    for p in patches:
        p.start()

    section("1. after an image upgrade: the artifact's caches are for another environment")
    with mock.patch.object(launch_mod, "run_profile", fake_engine(lambda env: None)):
        serve(artifact, model, root / "run1.json")

    section("2. servekit launch --only-prepare: builds this environment's caches, no re-shard")
    serve(artifact, model, root / "unused.json", "--only-prepare")

    section("3. the next launch stages them")
    with mock.patch.object(launch_mod, "run_profile", fake_engine(lambda env: None)):
        serve(artifact, model, root / "run3.json")

    section("4. an engine that compiles where servekit does not look (a renamed variable)")

    def renamed(env):
        jit = Path(env["HOME"]) / ".cache" / "sglang" / "jit" / "sm90"
        for i in range(3):
            (jit / f"mod{i}").mkdir(parents=True, exist_ok=True)
            (jit / f"mod{i}" / "m.so").write_bytes(b"x" * 1_500_000)
        sym = Path(tempfile.gettempdir()) / "symm_allocator"
        sym.mkdir(exist_ok=True)
        (sym / "nccl_allocator.so").write_bytes(b"x" * 300_000)

    with mock.patch.object(launch_mod, "run_profile", fake_engine(renamed)):
        rc = serve(artifact, model, root / "run4.json", "--jit-cache-strict")
    print(f"exit code {rc}")
    print("report jit_cache section:")
    print(json.dumps(json.loads((root / "run4.json").read_text())["jit_cache"], indent=2))


if __name__ == "__main__":
    sys.exit(main_demo())
