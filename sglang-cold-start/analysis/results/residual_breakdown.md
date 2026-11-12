| cluster-model | arm | n | total s | process start + TP worker spawn (imports) | torch.distributed init | weight loading (+ stage when not overlapped) | KV cache allocation | prefill CUDA graph capture (breakable) | decode CUDA graph capture | first warmup request (JIT) | other gaps (unknown, http bind) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| bristen-apertus8b | servekit (overlap) | 1 | 114.4 | 29.4 | 20.1 | 2.1 | 5.9 | 37.5 | 4.9 | 8.8 | 5.6 |
| bristen-apertus8b | servekit (serial) | 1 | 115.4 | 28.9 | 20.4 | 5.3 | 3.7 | 37.5 | 4.8 | 8.8 | 5.5 |
| bristen-glm4.7 | servekit (overlap) | 1 | 89.5 | 33.7 | 13.3 | 16.2 | 5.1 | 0.0 | 12.8 | 2.7 | 5.8 |
| bristen-glm4.7 | servekit (serial) | 1 | 98.0 | 27.9 | 10.5 | 26.9 | 4.5 | 0.0 | 12.7 | 3.1 | 6.1 |
| bristen-llama70b | servekit (overlap) | 1 | 165.0 | 32.4 | 19.4 | 14.3 | 3.1 | 72.5 | 8.4 | 8.8 | 6.0 |
| bristen-llama70b | servekit (serial) | 1 | 173.8 | 29.4 | 19.4 | 23.5 | 3.1 | 72.6 | 8.4 | 8.9 | 6.0 |
| bristen-llama70b (JIT A/B) | servekit overlap, JIT no-cache | 1 | 217.6 | 79.8 | 19.5 | 14.2 | 3.1 | 73.4 | 12.9 | 8.9 | 5.8 |
| bristen-llama70b (JIT A/B) | servekit overlap, JIT with-cache | 1 | 146.2 | 63.4 | 2.7 | 14.1 | 3.2 | 43.3 | 13.1 | 1.5 | 4.8 |
| clariden-apertus8b | servekit (overlap) | 1 | 95.0 | 26.1 | 20.8 | 0.9 | 4.1 | 27.6 | 5.5 | 6.5 | 3.5 |
| clariden-apertus8b | servekit (serial) | 1 | 89.9 | 26.3 | 16.8 | 2.0 | 4.1 | 26.5 | 4.9 | 6.4 | 3.3 |
| clariden-glm4.7 | servekit (overlap) | 1 | 105.2 | 26.2 | 7.8 | 6.7 | 4.1 | 0.0 | 30.8 | 26.1 | 3.6 |
| clariden-glm4.7 | servekit (serial) | 1 | 205.0 | 25.9 | 57.8 | 47.4 | 4.0 | 0.0 | 34.2 | 22.6 | 3.8 |
| clariden-llama70b | servekit (overlap) | 1 | 116.8 | 24.7 | 19.0 | 6.0 | 3.2 | 46.3 | 8.0 | 6.3 | 3.2 |
| clariden-llama70b | servekit (serial) | 2 | 156.7 [138.7-174.8] | 26.3 [26.2-26.3] | 20.8 [19.4-22.2] | 32.2 [11.1-53.4] | 3.2 [3.2-3.2] | 51.4 [50.7-52.1] | 10.7 [10.0-11.3] | 6.4 [6.3-6.5] | 3.4 [3.4-3.5] |

Share of the median total (medians of each bucket / median total):

| cluster-model | arm | median total s | python_start_and_worker_spawn | dist_init | weights | kv_alloc | prefill_graphs | decode_graphs | warmup_jit | other |
|---|---|---|---|---|---|---|---|---|---|---|
| bristen-apertus8b | servekit (overlap) | 114.4 | 26% | 18% | 2% | 5% | 33% | 4% | 8% | 5% |
| bristen-apertus8b | servekit (serial) | 115.4 | 25% | 18% | 5% | 3% | 32% | 4% | 8% | 5% |
| bristen-glm4.7 | servekit (overlap) | 89.5 | 38% | 15% | 18% | 6% | 0% | 14% | 3% | 6% |
| bristen-glm4.7 | servekit (serial) | 98.0 | 28% | 11% | 27% | 5% | 0% | 13% | 3% | 6% |
| bristen-llama70b | servekit (overlap) | 165.0 | 20% | 12% | 9% | 2% | 44% | 5% | 5% | 4% |
| bristen-llama70b | servekit (serial) | 173.8 | 17% | 11% | 14% | 2% | 42% | 5% | 5% | 3% |
| bristen-llama70b (JIT A/B) | servekit overlap, JIT no-cache | 217.6 | 37% | 9% | 7% | 1% | 34% | 6% | 4% | 3% |
| bristen-llama70b (JIT A/B) | servekit overlap, JIT with-cache | 146.2 | 43% | 2% | 10% | 2% | 30% | 9% | 1% | 3% |
| clariden-apertus8b | servekit (overlap) | 95.0 | 27% | 22% | 1% | 4% | 29% | 6% | 7% | 4% |
| clariden-apertus8b | servekit (serial) | 89.9 | 29% | 19% | 2% | 5% | 29% | 5% | 7% | 4% |
| clariden-glm4.7 | servekit (overlap) | 105.2 | 25% | 7% | 6% | 4% | 0% | 29% | 25% | 3% |
| clariden-glm4.7 | servekit (serial) | 205.0 | 13% | 28% | 23% | 2% | 0% | 17% | 11% | 2% |
| clariden-llama70b | servekit (overlap) | 116.8 | 21% | 16% | 5% | 3% | 40% | 7% | 5% | 3% |
| clariden-llama70b | servekit (serial) | 156.7 | 17% | 13% | 21% | 2% | 33% | 7% | 4% | 2% |
