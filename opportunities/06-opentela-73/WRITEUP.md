# OpenTela #73: why a restarted worker stays unroutable, and a fix

## Problem

eth-easl/OpenTela#73 (Xiaozhe Yao, 6 Nov 2026, private mesh at f740781):

- After a worker restarts, the head can go minutes without its node-table row. The worker meanwhile lists itself as `ready` with its services, but requests are not routed to it.
- Waiting 20 s or 70 s before the restart does not help.
- `POST /v1/dnt/_node {}` on the worker makes the row appear within about 5 s.

**Reproduced on localhost with real `otela` binaries** (head + worker, mock LLM, SIGTERM then restart, 3 runs per cell):

- **Baseline (4b6e90f), public-address worker:** not routable within 180 s in any of the 9 runs (restart delays 0, 20 and 70 s). Meanwhile the worker reports itself `ready` with its service, exactly as in the issue.
- **Baseline, default relay worker with a 20,000-record history:** stuck in 1 of 3 runs; the other 2 recovered at about 20 s. On localhost this is a race against the worker's single re-announce.
- **Fixed:** routable in 15.5-20.0 s (medians).
- **With optional patch 3:** routable in 6.5-8.6 s.

The Results section below has the full table.

## Root cause

A restart keeps the peer ID (persisted key) but not the CRDT store:

- `cleanslate` defaults to true;
- shutdown wipes the store as well.

Every replica still holds the worker's previous row at the mesh's DAG height H: its LEFT record, or a head's eviction record. The forked go-ds-crdt gives each new block priority "max head height + 1", and `set.setValue` silently drops a lower-priority value. The restarted worker's writes therefore land at heights 1, 2, … and are dropped everywhere, so the head keeps the LEFT row, which it treats as disconnected.

**Why it lasts minutes:**

1. The worker writes before the first broadcast arrives. That disables go-ds-crdt's "adopt heads when empty" shortcut, so it must walk the whole history before its height passes H.
2. After that walk, only an unrelated event makes it write again: a new connection, a relay change, a health transition, or the POST.

The one re-announce at startup comes from the relay-reservation change about 10 s in. On small meshes it can land after the walk and rescue the row. Workers with a public address skip relay reservations, and a failing reservation (cf. #66) skips the re-announce too, so for those workers nothing rescues the row.

**This is a regression.** a8c1055 (#51, Mar 2026) removed a 2-minute re-announce, whose comment named exactly this case, to stop DAG bloat. #49 had fixed a different cause of the same silent drop.

## Related work, and what is new here

- **go-ds-crdt** (ipfs/go-ds-crdt, H. Sanjuan) and the Merkle-CRDT paper (Sanjuan et al., arXiv:2004.00107) define height-priority last-writer-wins. Upstream's empty-heads shortcut targets the same fresh-replica problem, but it only works before a replica's first write.
- **The fix borrows two classic ideas:**
  - Lamport's clock receive rule (CACM 1978), for write priority;
  - SWIM's incarnation numbers (Das et al., DSN 2002), with which a live member refutes stale state about itself.
- **OpenTela's own history:**
  - #49: head-height guard;
  - #51: DAG-bloat fix, which removed the periodic re-announce;
  - #65: never sweep this node's own row;
  - #67: connectivity watchdog.
- **Independent hypothesis.** A parallel session reached the same hypothesis from code reading (`handoff-notes/opentela-73/NOTES.md`). It built no test and no fix.
- **New here:**
  - tracing the regression to a8c1055;
  - the explanation of when a small mesh recovers (the relay-change race);
  - a deterministic regression test;
  - an end-to-end harness;
  - a fix that adds no periodic writes.

## What was built

1. **go-ds-crdt `addDAGNode`.** Priority becomes max(head height, current priority of each key written) + 1. A replica that has merged a value now outranks it on its next write, even mid-walk. It is a no-op when the replica is in sync, and convergence is unchanged because priorities stay fixed per block.
2. **protocol self-row guard.**
   - Every record carries a per-run `incarnation`, stamped in `applyLocalRuntimeMetadata`, which every self-row writer already calls.
   - When the PutHook sees our row holding another run's record, it republishes once (1 s debounce).
   - Records from this run are ignored, including a head's eviction of this run, so the guard never fights a live eviction.
   - The guard is off after AnnounceLeave.
3. **Tests.**
   - `TestRestartedReplicaRowOutranksItsPreviousIncarnation`: a fresh replica merges the tip of the history while the rest of the walk is held back; fails 10 of 10 runs on main, passes 10 of 10 fixed, `-race`.
   - 5 guard unit tests.
   - `e2e/restart_repro.py`, which runs real binaries and isolates HOME, ports and bootstrap.

## Results

| check | baseline 4b6e90f | fixed |
|---|---|---|
| CRDT regression test, 10 runs, `-race` | 10 fail | 10 pass |
| `go vet ./...`; `go test -race ./...` | n/a | vet clean; full `go test -race ./...` passes (17 pkgs) |
| on top of PR #72: `go test -race ./internal/protocol/...` | n/a | pass |

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

## How to run

```bash
cd opportunities/06-opentela-73 && source env.sh            # Go 1.26.11 in toolchain/
cd vendor/OpenTela-final/src && go test -race ./internal/protocol/... -run 'Restarted|ObserveSelfRow|StampsIncarnation' -v
cd ../../../e2e && ./run_matrix.sh fixed pub 0 180 0 20 70   # or base; CSV in ../results/
```

## On the private mesh / Alps, and risks

**Prediction, not measured.** With the fix, time-to-routable after a restart is about the CRDT start (3 s), plus one rebroadcast (5 s ± 30 %), plus fetching the top blocks, plus the 1 s debounce, plus the head's next ping (≤ 30 s). That stays under about 40 s regardless of history length. Without the fix it is the full history walk plus the wait for the next unrelated write: minutes, or unbounded.

**Risks:**

- One extra CRDT write per restart.
- In a mixed-version mesh, an old head's eviction of a live new worker is answered with a republish.
- The head still waits up to one ping interval before it re-marks a rejoined LEFT row connected; that is most of the fixed build's 15-20 s. Optional `patches/optional/0003` takes `Connected` from a live direct connection instead, which brings the median to 6.5-8.6 s. It is kept separate so the core fix stays minimal.

## Pitch (for Xiaozhe)

#73 is a priority regression in the forked go-ds-crdt, uncovered when #51 removed the periodic re-announce. A restarted worker writes at height 1 against its own LEFT row at height H, and every replica drops the write silently. Two small patches fix it: a Lamport rule in `addDAGNode` and an incarnation-based self-row guard. They come with a deterministic regression test and an e2e harness, and the worker comes back in seconds instead of minutes, with no periodic writes.
