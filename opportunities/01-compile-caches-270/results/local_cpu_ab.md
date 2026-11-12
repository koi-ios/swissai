# Local CPU cold vs warm (macOS-15.6-arm64-arm-64bit, Python 3.12.13)

Median [min-max] over 3 runs; whole-process wall seconds incl. Python and library import (seconds, not the GPU compile times; the point is the hit/miss behaviour).

| workload (real library) | cold wall s | warm wall s | binaries built cold | rebuilt warm |
|---|---:|---:|---:|---:|
| torch_compile_cpu | 19.24 [18.17-19.32] | 14.93 [14.74-16.62] | 15 | 0 |
| tvm_ffi_jit | 5.31 [5.26-5.68] | 2.37 [2.34-2.48] | 3 | 0 |
| tilelang_cpu | 6.21 [5.72-7.39] | 3.21 [3.06-3.68] | 3 | 0 |
