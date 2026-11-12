| cluster | experiment | image | torch pre-imported by a wrapper before SGLang starts | n | servekit `process_startup` (to first output line) | main process: spawn → `server_args=` | TP schedulers: `server_args=` → last `Init torch distributed begin` | total: spawn → dist init | SyntaxWarnings from wrapper / from SGLang (per log) |
|---|---|---|---|---|---|---|---|---|---|
| bristen | cpu-sweep | v0.5.10 | no | 4 | - | 27.8 [24.7-29.1] | 12.0 [11.0-13.0] | 39.3 [37.7-41.1] | 0.0 [0.0-0.0] / 0.0 [0.0-0.0] |
| bristen | jit-cache A/B | v0.5.16 | no | 2 | 40.3 [33.6-47.0] | 56.8 [49.1-64.4] | 14.5 [14.0-15.0] | 71.2 [63.1-79.4] | 0.0 [0.0-0.0] / 1.0 [1.0-1.0] |
| bristen | lustre-loading | v0.5.10 | no | 18 | 24.2 [18.2-31.8] | 29.9 [22.0-37.3] | 11.0 [10.0-12.0] | 40.7 [33.0-48.3] | 0.0 [0.0-0.0] / 0.0 [0.0-0.0] |
| bristen | servekit-eval | v0.5.16 | yes | 12 | 9.3 [9.0-12.8] | 13.8 [12.3-26.3] | 15.5 [14.0-16.0] | 29.1 [27.9-41.3] | 2.0 [2.0-2.0] / 0.0 [0.0-0.0] |
| clariden | servekit-eval | v0.5.16 | yes | 18 | 0.4 [0.4-0.8] | 11.3 [9.9-60.8] | 15.0 [14.0-16.0] | 26.5 [25.5-75.8] | 2.0 [1.0-2.0] / 0.0 [0.0-0.0] |

Median [min-max] over jobs, seconds. Log timestamps have 1 s resolution (taken at +0.5 s).
`spawn` is servekit's `started_at` (Popen of `python -m sglang.launch_server`).
