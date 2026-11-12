"""The upstream SGLang patch: applies to v0.5.20, compiles, and its zygote works end to end
in a stand-in `sglang` package (real SGLang needs a GPU)."""
import json
import os
import shutil
import subprocess
import sys
import textwrap

import pytest

from conftest import ROOT

REPO = os.path.join(ROOT, "vendor", "sglang-v0.5.20")
PATCH = os.path.join(ROOT, "patches", "sglang-v0.5.20-worker-zygote.patch")
PKG = os.path.join(REPO, "python", "sglang", "_worker_zygote")

pytestmark = pytest.mark.skipif(not (os.path.isdir(os.path.join(REPO, ".git")) and os.path.exists(PATCH)),
                                reason="vendor clone or patch missing (run patches/make_sglang_patch.py)")


def _git(*a):
    return subprocess.run(["git", "-C", REPO, *a], capture_output=True, text=True)


def test_patch_applies_to_v0_5_20(tmp_path):
    wt = tmp_path / "wt"
    r = _git("worktree", "add", "--detach", str(wt), "v0.5.20")
    if r.returncode != 0:
        pytest.skip(f"cannot create worktree: {r.stderr}")
    try:
        subprocess.run(["git", "-C", str(wt), "sparse-checkout", "set", "python"], capture_output=True)
        chk = subprocess.run(["git", "-C", str(wt), "apply", "--check", PATCH], capture_output=True, text=True)
        assert chk.returncode == 0, chk.stderr
    finally:
        _git("worktree", "remove", "--force", str(wt))


def test_patched_files_compile():
    if not os.path.isdir(PKG):
        pytest.skip("patched branch not checked out")
    for rel in ("python/sglang/__init__.py", "python/sglang/srt/entrypoints/engine.py",
                "python/sglang/srt/utils/numa_utils.py", "python/sglang/srt/environ.py"):
        path = os.path.join(REPO, rel)
        compile(open(path, "rb").read(), path, "exec")
    for fn in os.listdir(PKG):
        if fn.endswith(".py"):
            path = os.path.join(PKG, fn)
            compile(open(path, "rb").read(), path, "exec")


def test_standin_sglang_launch_forks_workers_from_zygote(tmp_path):
    pytest.importorskip("torch")
    if not os.path.isdir(PKG):
        pytest.skip("patched branch not checked out")
    pkg = tmp_path / "sglang"
    shutil.copytree(PKG, pkg / "_worker_zygote")
    hook = open(os.path.join(REPO, "python", "sglang", "__init__.py")).read().split("\nimport platform")[0]
    (pkg / "__init__.py").write_text(hook + "\nimport json as _heavy_stand_in  # the rest of `import sglang`\n")
    (pkg / "launch_server.py").write_text(textwrap.dedent("""
        import json, multiprocessing as mp, os, sys
        from sglang._worker_zygote import worker_start_method

        def worker(q):
            q.put({"pid": os.getpid(), "zygote_child": os.environ.get("SGLANG_WORKER_ZYGOTE_CHILD"),
                   "late_env": os.environ.get("SET_BY_LAUNCHER_LATER"),
                   "torch_preloaded": "torch" in sys.modules})

        if __name__ == "__main__":
            os.environ["SET_BY_LAUNCHER_LATER"] = "1"      # like _set_envs_and_config's NCCL_* etc.
            mp.set_start_method(worker_start_method(), force=True)
            q = mp.Queue()
            ps = [mp.Process(target=worker, args=(q,)) for _ in range(2)]
            [p.start() for p in ps]
            out = {"method": mp.get_start_method(), "workers": [q.get(timeout=60) for _ in ps]}
            [p.join() for p in ps]
            print("RESULT " + json.dumps(out))
    """))
    modfile = tmp_path / "mods.txt"
    modfile.write_text("torch\njson\n")
    env = dict(os.environ)
    env.update({"PYTHONPATH": str(tmp_path), "SGLANG_WORKER_ZYGOTE": "1",
                "SGLANG_WORKER_ZYGOTE_MODULES": str(modfile),
                "SGLANG_WORKER_ZYGOTE_FAKE_CUDA": json.dumps({"count": 0, "major": 0, "minor": 0, "name": "none"})})
    env.pop("SGLANG_WORKER_ZYGOTE_STARTED", None)
    r = subprocess.run([sys.executable, "-m", "sglang.launch_server"], capture_output=True, text=True,
                       env=env, cwd=str(tmp_path), timeout=180)
    line = next((l for l in r.stdout.splitlines() if l.startswith("RESULT ")), None)
    assert line, r.stderr[-4000:]
    out = json.loads(line[len("RESULT "):])
    assert out["method"] == "forkserver", r.stderr[-3000:]
    for w in out["workers"]:
        assert w["zygote_child"] is not None
        assert w["late_env"] == "1"
        assert w["torch_preloaded"] is True
