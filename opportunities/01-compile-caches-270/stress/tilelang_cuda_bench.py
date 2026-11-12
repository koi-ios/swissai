#!/usr/bin/env python3
"""TileLang CUDA JIT cost, cold vs warm cache, on a GPU (Modal: modal_app.py::tilelang).

Compiles the canonical TileLang GEMM at several static shapes and tile configs (each one a
separate cache entry, like the band variants vLLM's DeepSeek-V4 mHC kernels compile), first in
a fresh TILELANG_CACHE_DIR (cold), then in a NEW process with the same dir (warm, a new job).
Reports seconds per kernel for both, and checks every kernel's result. An estimate of what
this implies for DeepSeek-V4.1-Flash's ~40 min (model-launch #270) must be labelled as such.

    python stress/tilelang_cuda_bench.py --out results/tilelang_bench.json
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
import time

SHAPES = [(m, 4096, 4096, cfg) for m in (16, 64, 256, 1024) for cfg in ((64, 64, 32), (128, 128, 32))]


def phase() -> list[dict]:
    import tilelang
    import tilelang.language as T
    import torch

    @tilelang.jit(out_idx=[-1])
    def matmul(M, N, K, block_M, block_N, block_K, dtype="float16", accum_dtype="float"):
        @T.prim_func
        def main(A: T.Tensor((M, K), dtype), B: T.Tensor((K, N), dtype), C: T.Tensor((M, N), dtype)):
            with T.Kernel(T.ceildiv(N, block_N), T.ceildiv(M, block_M), threads=128) as (bx, by):
                A_shared = T.alloc_shared((block_M, block_K), dtype)
                B_shared = T.alloc_shared((block_K, block_N), dtype)
                C_local = T.alloc_fragment((block_M, block_N), accum_dtype)
                T.clear(C_local)
                for k in T.Pipelined(T.ceildiv(K, block_K), num_stages=3):
                    T.copy(A[by * block_M, k * block_K], A_shared)
                    T.copy(B[k * block_K, bx * block_N], B_shared)
                    T.gemm(A_shared, B_shared, C_local)
                T.copy(C_local, C[by * block_M, bx * block_N])
        return main

    out = []
    for (m, n, k, (bm, bn, bk)) in SHAPES:
        t0 = time.perf_counter()
        kern = matmul(m, n, k, bm, bn, bk)
        dt = time.perf_counter() - t0
        a = torch.randn(m, k, device="cuda", dtype=torch.float16)
        b = torch.randn(k, n, device="cuda", dtype=torch.float16)
        ok = torch.allclose(kern(a, b).float(), (a.float() @ b.float()), rtol=2e-2, atol=1e-1)
        out.append({"M": m, "N": n, "K": k, "block": [bm, bn, bk], "seconds": round(dt, 3), "correct": bool(ok)})
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="results/tilelang_bench.json")
    ap.add_argument("--phase", action="store_true", help=argparse.SUPPRESS)
    a = ap.parse_args()
    if a.phase:
        print("RESULT " + json.dumps(phase()))
        return 0
    cache = tempfile.mkdtemp(prefix="tl-cache-")
    env = dict(os.environ, TILELANG_CACHE_DIR=cache, TILELANG_TMP_DIR=os.path.join(cache, "tmp"))
    res = {"cache_dir": cache}
    for name in ("cold", "warm"):
        t0 = time.perf_counter()
        p = subprocess.run([sys.executable, __file__, "--phase"], env=env, capture_output=True, text=True)
        line = next((ln for ln in p.stdout.splitlines() if ln.startswith("RESULT ")), None)
        res[name] = {"process_s": round(time.perf_counter() - t0, 2), "returncode": p.returncode,
                     "kernels": json.loads(line[7:]) if line else [], "stderr_tail": p.stderr[-2000:]}
    try:
        import tilelang
        res["tilelang"] = tilelang.__version__
    except Exception:
        pass
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    with open(a.out, "w") as f:
        json.dump(res, f, indent=1)
    for name in ("cold", "warm"):
        ks = res[name]["kernels"]
        print(name, f"{sum(k['seconds'] for k in ks):.1f} s for {len(ks)} kernels,",
              f"all correct: {all(k['correct'] for k in ks)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
