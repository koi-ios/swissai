<!-- Draft for you to rewrite and own before sending. Delete this line. -->
# fix(protocol): let a restarted node's row outrank its earlier run's

Refs #73. Two commits on top of `main`; they also apply on top of #72 (checked at 32aeeb9 on 12 Nov 2026).

## Problem

After a worker restarts, the head can go minutes without the worker's node-table row. The worker lists itself as `ready` with its services, but `/v1/dnt/table` on the head omits it and nothing is routed to it. Waiting before the restart does not help, and `POST /v1/dnt/_node {}` on the worker brings the row back within seconds. That is #73.

## Root cause

A restart keeps the peer ID but not the CRDT store: the key file is persisted (`protocol/host.go`), `--cleanslate` defaults to true (`entry/cmd/root.go`), and shutdown wipes the store as well (`server/server.go`).

The rest of the mesh still holds the worker's previous row at the mesh's DAG height H: its `AnnounceLeave` LEFT record (`node_table.go`) or a head's eviction record (`liveness.go`). go-ds-crdt gives every new block the priority "max head height + 1" (`go-ds-crdt/crdt.go`), and `set.setValue` drops a value of lower priority without a log or a PutHook (`set.go`). So:

1. The restarted worker writes its row at heights 1, 2, … and every replica drops those writes. The head keeps the LEFT row, which is forced to `Connected=false`.
2. The worker's height catches up only after it has walked the whole history, because it writes before the first broadcast arrives and so never takes the "adopt heads when empty" shortcut in `handleNext`.
3. Even then, nothing writes the row again until something unrelated does: a new libp2p connection, a relay change (once, about 10 s after start, and only for workers that reserve a relay), a service-health transition, or the POST.

It is a regression. Until a8c1055 (#51), `clock.go` re-announced every 2 minutes, and its comment named this case. #51 removed that to stop DAG bloat. #49 had fixed a different cause of the same silent drop.

## Fix

1. **`crdt`: rank a new write above the values it replaces.** In `addDAGNode` the priority becomes `max(max head height, current priority of each key in the delta) + 1`, which is Lamport's receive rule. A replica that has merged a value outranks it with its next write, whether or not it has finished walking the history. When the replica is in sync this changes nothing, and convergence is unaffected: a block's priority is fixed when the block is created.
2. **`protocol`: republish our own row when it holds an earlier run's record.** Every record a node publishes carries `incarnation`, a random value per process, stamped in `applyLocalRuntimeMetadata`. When the PutHook sees our own key's winning value come from another incarnation, the node republishes once (1 s debounce). Records of this run are left alone, including a head's eviction of this run, so the guard never fights a live eviction. It is off after `AnnounceLeave`.

Neither change adds periodic writes. A restart costs one extra write, so the DAG-bloat fix in #51 stands.

Alternatives considered: adopting broadcast heads before walking their branch (changes what a head means for every key); re-announcing "after the initial sync" (there is no reliable signal that sync has finished); having the head treat a reconnecting LEFT peer as live (the next CRDT update would undo it).

The ideas are not new: height-priority last-writer-wins is go-ds-crdt's and the Merkle-CRDT paper's (Sanjuan et al., arXiv:2004.00107), the priority rule is Lamport's clock (1978), and the incarnation is SWIM's (Das et al., 2002).

## Evidence

**Regression test** (`go-ds-crdt/restart_priority_test.go`, deterministic). A head and a worker share 40 rows of history; the worker announces LEFT and restarts with an empty store; it merges the tip of the history while the rest of the walk is held back, then writes again. On `main` the write lands at height 2 against the LEFT record at height 42 and the head keeps LEFT: 10 of 10 runs fail. With the fix 10 of 10 pass, under `-race`.

**End to end on one machine.** Real `otela` binaries, a head and a worker serving a mock LLM, on localhost (macOS arm64, Go 1.26.11, 12 Nov 2026). The worker is stopped with SIGTERM and restarted with the same identity. Measured: seconds from the relaunch until the head's `/v1/dnt/table` lists the worker with its model, polled every 0.25 s, 180 s timeout. n = 3 runs per cell, so the times below describe what happened and are not a statistical comparison; the difference that matters is "never" against "always".

| Restarted worker | Restart delay | `main` (4b6e90f) | With this PR: median (min to max), n = 3 |
|---|---|---|---|
| public address | 0 s | not routable within 180 s, 3 of 3 runs | 15.5 s (15.3 to 15.6) |
| public address | 20 s | not routable within 180 s, 3 of 3 runs | 19.9 s (19.9 to 20.0) |
| public address | 70 s | not routable within 180 s, 3 of 3 runs | 20.0 s (19.9 to 20.1) |
| relay-reserving (default), 20,000-record history | 0 s | not routable within 180 s in 1 of 3 runs; 19.9 s in the other two | 19.8 s (19.8 to 19.9) |

In every baseline run the restarted worker reports itself `ready` while the head omits it, as in the issue, and the issue's POST workaround then gets it listed 5 to 20 s later. With the fix, the guard fires 3.3 to 13.0 s after the relaunch; the rest is the head waiting for its next ping before it marks a rejoined LEFT row connected.

Limits: this is localhost, not the mesh where the issue was seen. Both binaries were built with one harness-only edit, the removal of `libp2p.NATPortMap()`, so the test never asks the local router for port mappings; signed-binary enforcement was off. On a real mesh each block fetch costs a network round trip, so I expect the baseline to be worse there, but I have not measured it. I can share the harness.

## Testing

- New: `TestRestartedReplicaRowOutranksItsPreviousIncarnation` (go-ds-crdt), `TestObserveSelfRow_*` and `TestApplyLocalRuntimeMetadata_StampsIncarnation` (protocol).
- From `src/`: `go vet ./...` clean; `go test -race ./...` passes; `golangci-lint run` (v2.15) reports no issues.

Compatibility: `incarnation` is an optional JSON field that old nodes ignore. After an upgrade, a node whose row was written by an old binary republishes once. In a mixed-version mesh, an old head's eviction of a live new worker is answered with a republish, which is what `DeletePeerRow`'s comment already describes for live peers.

Not in this PR: a rejoined worker still waits for the head's next ping (up to 30 s) before it is listed. A small follow-up can mark it connected at once when the head holds a direct connection to it; I have it ready if you want it.

AI assistance: Claude Code was used to trace the root cause and to draft the patches, the tests and this description. I reviewed every changed line and ran the tests above.

🤖 Generated with [Claude Code](https://claude.com/claude-code)
