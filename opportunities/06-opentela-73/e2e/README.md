# e2e: OpenTela #73 on localhost

`restart_repro.py` runs real `otela` binaries.

## What one trial does

1. Starts a head and a worker, with a mock LLM behind the worker.
2. Waits until the head's `/v1/dnt/table` lists the worker with `model=mock-model`.
3. Optionally writes `--history` records into the CRDT. Each record is `POST /v1/dnt/_node {}` on the head, which adds one DAG node.
4. Stops the worker with SIGTERM: it announces LEFT and wipes its store.
5. Waits `--delay` seconds, then restarts the worker with the same HOME. The worker gets the same peer ID and, because cleanslate defaults to true, an empty store.
6. Measures how long the head takes to list the worker again.

If the worker is still missing at `--timeout`, the trial applies the issue's workaround (`POST /v1/dnt/_node {}` on the worker) and records how long that takes (`after_post_s`).

## Columns

| column | meaning |
|---|---|
| `time_to_routable_s` | from worker relaunch to the head listing it with its llm service |
| `worker_self_status`, `worker_self_services` | what the restarted worker reports about itself (`/v1/self`) |
| `guard_s` | fixed build only: relaunch to the log line `Own row holds a record from an earlier run` |

## Scenarios

- **`pub`**: the worker gets `--public-addr 127.0.0.1`. It reserves no relay, so it misses the one re-announce a restarted worker otherwise gets about 10 s in, when its relay changes.
- **`relay`**: the default worker, which reserves a relay on the head. Its 10 s re-announce wins only if the history walk has already finished by then, so a long `--history` makes it lose.

## Isolation and local-only edits

- **Isolated state.** Each node runs with its own `HOME` under `runs/`, so keys, `cfg.yaml` and the badger store stay out of the real home directory. `OF_*` and `AXIOM_*` variables are scrubbed from the environment.
- **No public mesh.** The head runs with `--mode standalone`, so it never dials the public bootstrap list. The worker bootstraps only to its own head.
- **No router port mappings.** `bin/otela-base` (4b6e90f) and `bin/otela-fixed` (the patches) are built with one harness-only edit, the removal of `libp2p.NATPortMap()` in `internal/protocol/host.go`. This keeps the runs from asking the local router for port mappings. Both builds get the same edit.
- **Signing off.** `OF_SECURITY_REQUIRE_SIGNED_BINARY=false`, because local builds carry no build attestation. Without it, every node drops every peer's row.

## Run

```bash
./run_matrix.sh base pub 0 180 0 20 70     # binary, scenario, history, timeout, delays...
./run_matrix.sh fixed pub 0 180 0 20 70
./run_matrix.sh base relay 20000 180 0
```

Each command runs 3 trials per delay and appends rows to `../results/e2e_<scenario>_<binary>.csv`.
