#!/usr/bin/env python3
"""Write-safety stress test: N processes compile the SAME kernels into ONE cache directory
at the same instant, then a fresh process loads every entry; count failures.

Real libraries at the images' versions (CPU): torch 2.13.0 Inductor, apache-tvm-ffi 0.1.11,
TileLang 0.1.12. The GPU libraries (Triton, FlashInfer, SGLang JIT, DeepGEMM, TileLang CUDA)
run the same protocol in modal_app.py::stress.

Per library and repeat:
  writers    N processes released together by a barrier file; failures = nonzero exits
  leftovers  temp/staging files left in the cache after all writers exit
  load       a fresh process with the same cache: wrong output or exception = failure;
             builds = entries it had to (re)build (should be 0)

    python stress/stress_writers.py --n 8 --repeats 3 --out results/stress_local.json
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import shutil
import subprocess
import sys
import tempfile
import textwrap
import time
from pathlib import Path

WAIT = textwrap.dedent('''
    import os, time
    go = os.environ["STRESS_GO"]
    while not os.path.exists(go):
        time.sleep(0.001)
''')

LIBS = {
    "inductor": {
        "env": "TORCHINDUCTOR_CACHE_DIR",
        "code": textwrap.dedent('''
            import torch
            from torch._dynamo.utils import counters
            def f(x, w):
                return torch.nn.functional.gelu(x @ w) + x.sum()
            g = torch.compile(f, dynamic=False)
            w = torch.ones(32, 32)
            for bs in (1, 4, 16):
                y = g(torch.ones(bs, 32), w)
                assert torch.allclose(y, f(torch.ones(bs, 32), w), atol=1e-4)
            print("BUILDS", counters["inductor"]["fxgraph_cache_miss"])
        '''),
    },
    "tvm_ffi": {
        "env": "TVM_FFI_CACHE_DIR",
        "code": textwrap.dedent('''
            import numpy as np, tvm_ffi, tvm_ffi.cpp, glob, os
            before = len(glob.glob(os.path.join(os.environ["TVM_FFI_CACHE_DIR"], "*", "*.so")))
            built = 0
            for i in (1, 2, 3):
                src = "void k%d(tvm::ffi::TensorView x, tvm::ffi::TensorView y) {" % i + \\
                      "for (int j = 0; j < x.size(0); ++j) static_cast<float*>(y.data_ptr())[j] = " + \\
                      "static_cast<float*>(x.data_ptr())[j] * %d; }" % i
                m = tvm_ffi.cpp.load_inline(name="k%d" % i, cpp_sources=src, functions="k%d" % i)
                x = np.arange(8, dtype=np.float32); y = np.empty_like(x)
                getattr(m, "k%d" % i)(tvm_ffi.from_dlpack(x), tvm_ffi.from_dlpack(y))
                assert np.allclose(y, x * i)
            print("BUILDS", 0)
        '''),
    },
    "tilelang": {
        "env": "TILELANG_CACHE_DIR",
        "code": textwrap.dedent('''
            import tilelang, tilelang.language as T, torch
            from tilelang.jit.kernel import JITKernel
            n_compiled = 0
            orig = JITKernel._compile_and_create_adapter
            def counting(self, *a, **k):
                global n_compiled
                n_compiled += 1
                return orig(self, *a, **k)
            JITKernel._compile_and_create_adapter = counting
            @tilelang.jit(target="c", out_idx=[1])
            def scale(N: int = 64):
                @T.prim_func
                def main(A: T.Tensor((N,), "float32"), B: T.Tensor((N,), "float32")):
                    with T.Kernel(1) as bx:
                        for i in T.serial(N):
                            B[i] = A[i] * 3.0
                return main
            for n in (64, 128, 256):
                a = torch.arange(n, dtype=torch.float32)
                assert torch.allclose(scale(n)(a), a * 3.0)
            print("BUILDS", n_compiled)
        '''),
    },
}

# GPU-only (Modal: modal_app.py::gpu_stress). Builds = cache entries this process wrote.
GPU_LIBS = {
    "triton": {
        "env": "TRITON_CACHE_DIR",
        "code": textwrap.dedent('''
            import glob, os, torch, triton, triton.language as tl
            root = os.environ["TRITON_CACHE_DIR"]
            before = set(glob.glob(os.path.join(root, "*", "__grp__*")))
            @triton.jit
            def add(x, y, out, n, BLOCK: tl.constexpr):
                offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
                m = offs < n
                tl.store(out + offs, tl.load(x + offs, mask=m) * 2 + tl.load(y + offs, mask=m), mask=m)
            x = torch.randn(4096, device="cuda"); y = torch.randn_like(x); o = torch.empty_like(x)
            for B in (64, 128, 256, 512, 1024):
                add[(triton.cdiv(4096, B),)](x, y, o, 4096, BLOCK=B)
                torch.cuda.synchronize()
                assert torch.allclose(o, x * 2 + y)
            print("BUILDS", len(set(glob.glob(os.path.join(root, "*", "__grp__*"))) - before))
        '''),
    },
    "flashinfer": {
        "env": "FLASHINFER_WORKSPACE_BASE",
        "code": textwrap.dedent('''
            import glob, os, torch
            root = os.environ["FLASHINFER_WORKSPACE_BASE"]
            pat = os.path.join(root, ".cache", "flashinfer", "**", "*.so")
            before = set(glob.glob(pat, recursive=True))
            import flashinfer
            x = torch.randn(16, 4096, device="cuda", dtype=torch.float16)
            w = torch.randn(4096, device="cuda", dtype=torch.float16)
            y = flashinfer.norm.rmsnorm(x, w)
            ref = x.float() * torch.rsqrt(x.float().pow(2).mean(-1, keepdim=True) + 1e-6) * w.float()
            assert torch.allclose(y.float(), ref, atol=2e-2, rtol=2e-2)
            print("BUILDS", len(set(glob.glob(pat, recursive=True)) - before))
        '''),
    },
}

TEMP_PATTERNS = (".tmp", "tmp.pid_", ".staging", ".partial")


def leftovers(cache: Path) -> list[str]:
    out = []
    for p in cache.rglob("*"):
        if p.is_file() and any(t in p.name or t in str(p.relative_to(cache)) for t in TEMP_PATTERNS):
            if "/tmp/" in "/" + str(p.relative_to(cache)) + "/" and p.stat().st_size == 0:
                continue
            out.append(str(p.relative_to(cache)))
    return out


def run_once(lib: str, n: int, work: Path, timeout: float) -> dict:
    spec = LIBS.get(lib) or GPU_LIBS[lib]
    cache = work / "cache"
    home = work / "home"
    cache.mkdir(parents=True)
    home.mkdir()
    go = work / "GO"
    env = dict(os.environ, HOME=str(home), TMPDIR=str(work), STRESS_GO=str(go), **{spec["env"]: str(cache)})
    env.pop("CACHE_AUDIT_PYLOG", None)
    writer = WAIT + spec["code"]
    script = work / "writer.py"           # a real file: TileLang's @jit reads the function source
    script.write_text(writer)
    procs = [subprocess.Popen([sys.executable, str(script)], env=env, stdout=subprocess.PIPE,
                              stderr=subprocess.PIPE, text=True) for _ in range(n)]
    time.sleep(2.0)                      # let every interpreter import and reach the barrier
    t0 = time.monotonic()
    go.touch()
    fails, errors = 0, []
    for p in procs:
        try:
            out, err = p.communicate(timeout=timeout)
        except subprocess.TimeoutExpired:
            p.kill()
            out, err = p.communicate()
            err += "\nTIMEOUT"
        if p.returncode != 0:
            fails += 1
            errors.append(err.strip().splitlines()[-1][:300] if err.strip() else f"exit {p.returncode}")
    wall = time.monotonic() - t0
    left = leftovers(cache)
    # load every entry from a fresh process (no barrier)
    lp = subprocess.run([sys.executable, str(script)], env=env, capture_output=True, text=True,
                        timeout=timeout)          # GO exists already: no wait
    builds = None
    for line in lp.stdout.splitlines():
        if line.startswith("BUILDS"):
            builds = int(line.split()[1])
    return {"writers": n, "writer_failures": fails, "writer_errors": sorted(set(errors))[:5],
            "writers_wall_s": round(wall, 2), "leftover_temp_files": left[:10], "n_leftovers": len(left),
            "load_ok": lp.returncode == 0, "load_error": lp.stderr.strip().splitlines()[-1][:300] if lp.returncode else "",
            "rebuilds_on_load": builds}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--libs", default=",".join(LIBS))
    ap.add_argument("--timeout", type=float, default=300)
    ap.add_argument("--out", default="results/stress_local.json")
    a = ap.parse_args()
    results = {"host": platform.platform(), "python": sys.version.split()[0], "n": a.n,
               "repeats": a.repeats, "libs": {}}
    for lib in a.libs.split(","):
        runs = []
        for r in range(a.repeats):
            work = Path(tempfile.mkdtemp(prefix=f"stress-{lib}-"))
            try:
                runs.append(run_once(lib, a.n, work, a.timeout))
            finally:
                shutil.rmtree(work, ignore_errors=True)
            print(lib, r, json.dumps(runs[-1]), flush=True)
        results["libs"][lib] = {
            "runs": runs,
            "writer_failures": sum(x["writer_failures"] for x in runs),
            "load_failures": sum(not x["load_ok"] for x in runs),
            "rebuilds_on_load": [x["rebuilds_on_load"] for x in runs],
            "leftovers": sum(x["n_leftovers"] for x in runs),
        }
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(results, indent=1))
    print(json.dumps({k: {kk: v[kk] for kk in ("writer_failures", "load_failures", "rebuilds_on_load", "leftovers")}
                      for k, v in results["libs"].items()}, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
