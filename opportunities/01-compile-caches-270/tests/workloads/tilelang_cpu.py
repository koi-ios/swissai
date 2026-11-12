"""Real workload: TileLang 0.1.12 JIT (the kernel cache vLLM's DeepSeek-V4 mHC kernels use),
compiled for the C target so it runs on a CPU. One cache entry per static shape."""
import os
import sys
import time

import tilelang
import tilelang.language as T
import torch


@tilelang.jit(target="c", out_idx=[1])
def scale(N: int = 64, f: float = 2.0):
    @T.prim_func
    def main(A: T.Tensor((N,), "float32"), B: T.Tensor((N,), "float32")):
        with T.Kernel(1) as bx:
            for i in T.serial(N):
                B[i] = A[i] * f
    return main


def main() -> int:
    sizes = [64, 128, 256, 512][: int(os.environ.get("WORKLOAD_KERNELS", "3"))]
    t0 = time.perf_counter()
    for n in sizes:
        k = scale(n, 2.0)
        a = torch.arange(n, dtype=torch.float32)
        assert torch.allclose(k(a), a * 2.0)
    print(f"compile_s={time.perf_counter() - t0:.2f} kernels={len(sizes)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
