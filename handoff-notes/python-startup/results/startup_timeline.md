| experiment | cluster | image | torch pre-imported before Popen | n | job start -> Popen, minus serial staging (s) | Popen -> first line (s) | Popen -> server_args (s) | server_args -> last TP 'dist begin' (s) | dist begin -> nccl line (s) | nccl -> All Reduce config (s) | All Reduce config -> dist end (s) | pynvml FutureWarnings / log | SyntaxWarnings / log |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| cpu-sweep | bristen | v0.5.10 | no | 8 | 12.7 [10.8, 23.2] (n=8) | 26.0 [20.5, 53.2] (n=8) | 31.3 [24.7, 60.0] (n=8) | 12.0 [11.0, 20.0] (n=8) | 1.0 [1.0, 2.0] (n=8) | - | - | 0.0 [0.0, 0.0] (n=8) | 0.0 [0.0, 0.0] (n=8) |
| jit-cache-ab | bristen | v0.5.16 | no | 3 | - | 40.3 [33.6, 47.0] (n=2) | 56.8 [49.1, 64.4] (n=2) | 15.0 [14.0, 16.0] (n=3) | 1.0 [1.0, 2.0] (n=3) | 10.0 [2.0, 10.0] (n=3) | 7.0 [0.0, 8.0] (n=3) | 0.0 [0.0, 0.0] (n=3) | 1.0 [1.0, 1.0] (n=3) |
| lustre-loading | bristen | v0.5.10 | no | 18 | 13.2 [11.0, 28.5] (n=18) | 24.2 [18.2, 31.8] (n=18) | 29.9 [22.0, 37.3] (n=18) | 11.0 [10.0, 12.0] (n=18) | 1.0 [1.0, 2.0] (n=18) | - | - | 0.0 [0.0, 0.0] (n=18) | 0.0 [0.0, 0.0] (n=18) |
| servekit-eval | bristen | v0.5.16 | yes (sitecustomize) | 12 | 58.3 [49.9, 62.8] (n=12) | 9.3 [9.0, 12.8] (n=12) | 13.8 [12.3, 26.3] (n=12) | 15.5 [14.0, 16.0] (n=12) | 1.0 [1.0, 1.0] (n=12) | 11.0 [10.0, 13.0] (n=12) | 8.0 [8.0, 9.0] (n=12) | 0.0 [0.0, 0.0] (n=12) | 2.0 [2.0, 2.0] (n=12) |
| servekit-eval | clariden | v0.5.16 | yes (sitecustomize) | 18 | 37.9 [-10.6, 42.0] (n=18) | 0.4 [0.4, 0.8] (n=18) | 11.3 [9.9, 60.8] (n=18) | 15.0 [14.0, 16.0] (n=18) | 1.0 [0.0, 1.0] (n=18) | 11.0 [9.0, 13.0] (n=18) | 6.0 [6.0, 7.0] (n=18) | 10.0 [9.0, 10.0] (n=18) | 2.0 [1.0, 2.0] (n=18) |

Single-node launches only (head node). median [min, max] (n), seconds. SGLang log stamps have 1 s resolution.
`job start -> Popen` covers srun + container start + `pip install -e servekit` + (servekit-eval only) the
wrapper's own cold import of torch+sglang through sitecustomize; serial staging time is subtracted.
