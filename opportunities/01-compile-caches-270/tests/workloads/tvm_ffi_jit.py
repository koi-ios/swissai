"""Real workload: apache-tvm-ffi 0.1.11 JIT (the builder under SGLang 0.5.16's load_jit
and under FlashInfer). Builds N small C++ kernels with load_inline and calls each."""
import os
import sys
import time

import numpy as np
import tvm_ffi
import tvm_ffi.cpp

SRC = r'''
void scale_add_{i}(tvm::ffi::TensorView x, tvm::ffi::TensorView y) {{
  for (int k = 0; k < x.size(0); ++k)
    static_cast<float*>(y.data_ptr())[k] = static_cast<float*>(x.data_ptr())[k] * {i} + 1;
}}
'''


def main() -> int:
    n = int(os.environ.get("WORKLOAD_KERNELS", "3"))
    t0 = time.perf_counter()
    for i in range(1, n + 1):
        mod = tvm_ffi.cpp.load_inline(name=f"scale_add_{i}", cpp_sources=SRC.format(i=i),
                                      functions=f"scale_add_{i}")
        x = np.arange(4, dtype=np.float32)
        y = np.empty_like(x)
        getattr(mod, f"scale_add_{i}")(tvm_ffi.from_dlpack(x), tvm_ffi.from_dlpack(y))
        assert np.allclose(y, x * i + 1), (i, y)
    print(f"build_s={time.perf_counter() - t0:.2f} kernels={n}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
