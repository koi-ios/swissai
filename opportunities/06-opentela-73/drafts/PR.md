# fix: a restarted node's row must outrank its earlier run's (#73)

Refs #73. Two commits (patches in `patches/`), based on main 4b6e90f; they apply cleanly on top of #72.

## Root cause

A restarted worker keeps its peer ID, but its CRDT store starts empty:

- the key file is persisted (seed 0, `protocol/host.go:107-149`);
- `--cleanslate` defaults to true (`entry/cmd/root.go:76`, `start.go:48-53`);
- shutdown wipes the store anyway (`server/server.go:359-368`).

The rest of the mesh still holds the worker's previous row at the mesh's DAG height H. That row is either its `AnnounceLeave` LEFT record (`node_table.go:353-379`) or a head's eviction record (`liveness.go:317-340`). On a long-lived mesh H is in the thousands.

go-ds-crdt gives every new block the priority "max head height + 1" (`go-ds-crdt/crdt.go:1398-1405`). `set.setValue` drops a value of lower priority than the current one without a log or a PutHook (`set.go:325-327`). As a result:

1. **The worker's restart writes are dropped everywhere.** The restarted worker writes its row from `InitializeMyself` and `provideService` at heights 1, 2, …, and every replica drops those writes. On the head the row stays LEFT, and a LEFT row is forced to `Connected=false` (`node_table.go:432-460`). So `/v1/dnt/table` (`GetConnectedPeers`) omits the worker. The worker's own table still shows itself as ready with its services.
2. **Its height catches up only after walking the whole history.** The worker writes before the first broadcast arrives, so the shortcut in `handleNext` that adopts broadcast heads when a node has none (`crdt.go:451-481`) never applies. A remote head becomes a local head only when the walk reaches the bottom of the DAG (`processNode`, `crdt.go:949-955`).
3. **Even after that, nothing writes the row again** until something unrelated triggers a write:
   - a new libp2p connection (`host.go:324-329`);
   - a change of relay (`host.go:805-813`), which happens once, about 10 s after start, and only for workers that reserve a relay;
   - a service-health transition;
   - `POST /v1/dnt/_node`.

This explains every observation in the issue: the worker shows itself as ready while the head omits it, the downtime before the restart makes no difference, and the POST fixes it within seconds.

**It is a regression.** Until a8c1055 (#51, 25 Mar 2026), `clock.go` re-announced every 2 minutes. Its comment named this exact case:

> Without this, a fresh node's initial low-priority Put is never superseded once the DAG catches up.

#51 removed that re-announce to stop DAG bloat. #49 had already fixed a different cause of the same silent drop: the head height going backwards.

## Fix

1. **`crdt`: rank a new write above the values it replaces.**
   - In `addDAGNode`, the priority becomes `max(max head height, current priority of each key in the delta) + 1`. This is Lamport's receive rule.
   - A replica that has merged a value now outranks it with its next write, whether or not it has finished walking the history.
   - When the replica is in sync, current priorities are at most the head height, so the change is a no-op.
   - Convergence is unaffected. A block's priority is fixed when the block is created, and the winning value is still a deterministic function of the set of merged blocks.
   - The current priorities are read under `putElemsMux`, so a merge in progress is either complete or not yet started.
2. **`protocol`: republish our own row when it holds an earlier run's record.**
   - Every record a node publishes now carries `incarnation`, a random value per process. It is stamped in `applyLocalRuntimeMetadata`, which every self-row writer already calls.
   - When the PutHook sees our own key's winning value come from another incarnation, the node republishes once via `ReannounceLocalServices` (1 s debounce, coalesced). That covers:
     - an earlier run's LEFT record;
     - a head's eviction of an earlier run;
     - the stale services left after a crash;
     - a record written by a binary that predates the field.
   - With fix 1, that republish outranks the record it answers.
   - Records of this run are left alone. That includes a head's eviction of this run, because the eviction copies our record and its incarnation. The guard therefore never fights a live eviction.
   - The guard is off after `AnnounceLeave`.

Neither change adds periodic writes. A restart costs one extra write, so the DAG-bloat fix in #51 stands.

Alternatives considered:
- **Adopt broadcast heads early, before walking their branch.** This changes what a head means for every key. Fix 1 only affects the priority of keys the replica itself writes.
- **Re-announce once "after the initial sync".** There is no reliable signal that sync has finished: a remote head is only adopted at the bottom of the walk, and a failed fetch can leave the walk unfinished indefinitely.
- **Have the head treat a reconnecting LEFT peer as live.** Every replica would still hold the LEFT record in the CRDT, and the next CRDT update would undo that in-memory override. Only the writer can outrank its old row.

## Evidence

**Unit regression test** (`go-ds-crdt/restart_priority_test.go`):
- A head and a worker share 40 rows of history. The worker announces LEFT and restarts with an empty store.
- The restarted worker merges the tip of the history while the rest of the walk is held back, then writes again.
- On main that write lands at height 2 against the LEFT record at height 42, and the head keeps LEFT: 10 of 10 runs fail.
- With the fix, 10 of 10 runs pass under `-race` (`results/crdt_test_*_x10.txt`).

**End to end on localhost** (`e2e/restart_repro.py`):
- Setup: real `otela` binaries, a head plus a worker serving a mock LLM.
- The worker is stopped with SIGTERM and restarted with the same identity after a delay.
- Measured: seconds until the head's `/v1/dnt/table` lists it with `model=mock-model`.
- Both binaries carry one harness-only edit, the removal of `libp2p.NATPortMap()`, so the test never asks the local router for port mappings.
- Signed-binary enforcement is off, because local builds carry no attestation.

Seconds from the worker's relaunch until the head's `/v1/dnt/table` lists it with `model=mock-model`. Each cell is the median (min-max) of 3 runs, polled every 0.25 s, with a 180 s timeout.

| restarted worker | restart delay | baseline 4b6e90f | fixed (patches 1+2) | + optional patch 3 |
|---|---|---|---|---|
| public address, no extra history | 0 s | not routable within 180 s (3/3) | 15.5 (15.3-15.6) | 8.4 (1.3-11.1) |
| public address, no extra history | 20 s | not routable within 180 s (3/3) | 19.9 (19.9-20.0) | 8.6 (2.8-9.9) |
| public address, no extra history | 70 s | not routable within 180 s (3/3) | 20.0 (19.9-20.1) | 6.5 (1.8-9.6) |
| relay-reserving (default), 20,000-record history | 0 s | not routable within 180 s in 1/3; the other 2 routable at 19.9 (19.8-19.9) | 19.8 (19.8-19.9) | not run |

- **Baseline.** In every baseline run the restarted worker reports itself `ready` with its service (`/v1/self`) while the head omits it, which matches the issue. The issue's workaround, `POST /v1/dnt/_node {}` on the worker, then gets it listed 15.5, 20.0 and 5.2 s later at the three delays. The spread is the phase of the head's next ping; the 5 s case matches the issue's report.
- **Fixed build.** The guard fires 3.3-13.0 s after the relaunch. The rest of the 15-20 s is the head waiting for its next ping before it marks a rejoined LEFT row connected. Optional patch 3 removes that wait whenever the head holds a direct connection to the worker.
- **Why the two rows differ.** A public-address worker reserves no relay, so it never gets the one re-announce a restarted worker otherwise makes about 10 s in. A default worker does make that re-announce, and whether it rescues the row is a race with the history walk. On localhost a 20,000-block walk mostly finishes first. The one stuck run was the calibration run: it ran at `nice -n 19` alongside other runs, so its walk was slower. With a 1,000-record history the worker recovered in 14.2 s.
- **Prediction, not measured.** On a real mesh every block fetch costs a WAN or relay round trip, so a long history makes the walk take minutes and the re-announce loses.
- **Harness.** See `e2e/README.md` for isolation, the one harness-only build edit and the signing setting. Raw rows are in `results/e2e_*.csv`.

## Tests

- `go-ds-crdt`: `TestRestartedReplicaRowOutranksItsPreviousIncarnation`.
- `protocol`: `TestObserveSelfRow_*` (republishes over an earlier run, once per burst; handles a record without an incarnation; ignores this run's records, including an eviction of this run; stays silent while leaving) and `TestApplyLocalRuntimeMetadata_StampsIncarnation`.
- `go vet ./...` is clean; golangci-lint v2.15 reports 0 issues on `internal/protocol/...`; `go test -race ./internal/protocol/...` passes, also with #72 applied underneath. The full `go test -race ./...` passes (17 packages ok, `results/full_suite_fix73.txt`).

## Compatibility

- `incarnation` is a new optional JSON field, and old nodes ignore it.
- After an upgrade, a node whose row was written by an old binary republishes once.
- In a mixed-version mesh, an eviction record from an old head carries no incarnation. A new worker that is evicted while alive therefore republishes instead of staying evicted. That is the outcome `DeletePeerRow`'s comment already describes for live peers: "its next registration rewrites the key and wins".

## Optional follow-up (not in this PR)

- **Connected on rejoin** (`patches/optional/0003`). A head marks a rejoined row `Connected` only after its next successful ping (30 s tick) or the worker's next pubsub ping (20 s), because a LEFT row is forced disconnected. A brand-new peer, by contrast, gets a connected minimal entry from `ConnectedF` at once. Most of the fixed build's remaining 15–20 s is this wait.
- **What patch 3 does.** When a non-LEFT record replaces a LEFT row and the head holds a direct (not relayed) connection to the peer, the head marks the row connected. That is the same proof of life `ConnectedF` accepts, and relayed peers still wait for the ping.
- **Measured.** With patch 3 the median drops to 6.5–8.6 s. It has its own unit test and can be reviewed separately.

🤖 Generated with [Claude Code](https://claude.com/claude-code)
