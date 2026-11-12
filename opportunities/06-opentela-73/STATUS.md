# STATUS: OpenTela #73 (restarted worker unroutable for minutes)

Updated 12 Nov 2026.

## Done

- **Root cause found, with file:line references.** It is a CRDT priority regression, first introduced by a8c1055 (#51), which removed a periodic re-announce. Details: `drafts/PR.md`, `WRITEUP.md`.
- **Reproduced at two levels:**
  - **Deterministic Go test** `go-ds-crdt/restart_priority_test.go`: fails 10 of 10 runs on main 4b6e90f and passes 10 of 10 with the fix, under `-race` (`results/crdt_test_*_x10.txt`).
  - **End to end with real `otela` binaries on localhost** (`e2e/`, 3 runs per cell):
    - Baseline, public-address worker: never routable within 180 s in 9 of 9 runs, at restart delays 0, 20 and 70 s. The worker reports itself `ready` throughout.
    - Baseline, default relay worker with a 20,000-record history: stuck in 1 of 3 runs. The other 2 recovered at about 20 s, so on localhost this is a race between the history walk and the worker's single re-announce.
    - Fixed: routable in 15.5–20.0 s (medians).
    - Table: `drafts/PR.md` and `WRITEUP.md`; raw rows: `results/e2e_*.csv`.
- **Fix in two commits**, authored as snarayana:
  - `patches/0001`: go-ds-crdt ranks a new write above the values it replaces (Lamport rule).
  - `patches/0002`: protocol republishes its own row when that row holds an earlier run's record (per-run `incarnation`, 1 s debounce).
- **Optional `patches/optional/0003`**: a rejoined peer with a direct connection is marked connected at once. Median 6.5–8.6 s instead of 15.5–20.0 s.
- **Tests and checks:**
  - Regression and unit tests: 1 + 5 tests, plus 1 for the optional patch.
  - Full `go test -race ./...` passes (17 packages).
  - `go vet ./...` is clean.
  - golangci-lint v2.15: 0 issues on `internal/protocol/...`.
- **Compatible with PR #72**: the patches apply cleanly on top of its head (32aeeb9), and `go test -race ./internal/protocol/...` passes there.
- **Drafts**: `drafts/PR.md` (not posted). Hand-over note in the shared notes folder.

## Not done / blocked

- Not run on the private mesh where the issue was seen (no access). The prediction for it is in `WRITEUP.md`.
- Nothing has been pushed or posted.

## Next 3 actions

1. Review `patches/` and `drafts/PR.md`, then open the PR on eth-easl/OpenTela from your fork with `git am patches/*.patch` (the author is your git identity).
2. Ask Xiaozhe to restart one worker on the private mesh with the fixed build. The prediction is that it is routable within about 35 s whatever the history length, and that the log line `Own row holds a record from an earlier run` appears once.
3. Offer `patches/optional/0003` separately if he also wants the remaining ping wait removed.
