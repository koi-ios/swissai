# MOCK RUN (format only, not a measurement): Cold vs warm vs servekit's three variables

Medians over the measured runs with [min-max]; seconds. Warm arms exclude their populate run.
Persisted arms seed a node-local overlay from a Modal Volume and promote back, as sml would.

## vllm · Apertus-8B · TP1

| arm | runs | ready | first request | seed | delta ready vs cold |
|---|---:|---:|---:|---:|---:|
| cold | 3 | 2.0 [1.9-2.3] | 0.0 [0.0-0.0] | - | +0.0 |
| servekit3 | 3 | 1.3 [1.3-1.3] | 0.0 [0.0-0.0] | 0.0 [0.0-0.0] | -0.7 |
| full | 3 | 0.1 [0.1-0.1] | 0.0 [0.0-0.0] | 0.0 [0.0-0.0] | -1.9 |

| phase | cold | servekit3 | full | full - cold |
|---|---:|---:|---:|---:|
| process_startup | 0.1 [0.1-0.4] | 0.1 [0.1-0.1] | 0.1 [0.1-0.1] | -0.0 |
| weight_loading | 0.4 [0.4-0.4] | 0.4 [0.4-0.4] | 0.4 [0.4-0.4] | +0.0 |
| torch_compile | 1.1 [1.1-1.1] | 0.5 [0.5-0.5] | 0.0 [0.0-0.0] | -1.1 |
| cuda_graph_capture | 0.8 [0.8-0.8] | 0.8 [0.8-0.8] | 0.0 [0.0-0.0] | -0.8 |

| cache component | arm | binaries built by populate | binaries rebuilt per warm run (median) | builds avoided |
|---|---|---:|---:|---:|
| deep_gemm | servekit3 | 2 | 2 | 0 |
| inductor.fxgraph | servekit3 | 3 | 3 | 0 |
| tilelang | servekit3 | 3 | 3 | 0 |
| triton | servekit3 | 4 | 0 | 4 |
| deep_gemm | full | 2 | 0 | 2 |
| inductor.fxgraph | full | 3 | 0 | 3 |
| tilelang | full | 3 | 0 | 3 |
| triton | full | 4 | 0 | 4 |

