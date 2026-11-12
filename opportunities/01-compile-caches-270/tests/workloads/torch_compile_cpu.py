"""Real workload: torch.compile (Inductor, CPU) of a small MLP block.

Exercises the same Inductor code vLLM and SGLang use: FX graph cache,
AOTAutograd cache and the C++ codecache (.so). Prints compile seconds and the
Inductor cache counters, so a warm run shows hits.
"""
import os
import sys
import time

import torch
from torch._dynamo.utils import counters

torch.manual_seed(0)


def block(x, w1, w2):
    h = torch.nn.functional.gelu(x @ w1)
    return torch.nn.functional.layer_norm(h @ w2 + x, (x.shape[-1],))


def main() -> int:
    n = int(os.environ.get("WORKLOAD_SIZES", "2"))
    f = torch.compile(block, dynamic=False)
    w1, w2 = torch.randn(64, 256), torch.randn(256, 64)
    t0 = time.perf_counter()
    for bs in [1, 8, 32, 64][:n]:  # one compile per shape, like CUDA-graph capture sizes
        f(torch.randn(bs, 64), w1, w2)
    dt = time.perf_counter() - t0
    inductor = {k: v for k, v in counters["inductor"].items() if "cache" in k}
    aot = {k: v for k, v in counters["aot_autograd"].items() if "cache" in k or "autograd_cache" in k}
    print(f"compile_s={dt:.2f} inductor={inductor} aot_autograd={aot}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
