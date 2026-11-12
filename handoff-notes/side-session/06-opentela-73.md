# 06-opentela-73: hand-over note (session for eth-easl/OpenTela#73)

Folder: `opportunities/06-opentela-73/`. Start with `STATUS.md`, then `drafts/PR.md` and `WRITEUP.md`.

## Root cause (verified)

- **Priority regression in the forked go-ds-crdt.**
  - A restarted worker keeps its peer ID but its store is wiped (`cleanslate` defaults to true, and shutdown wipes it too).
  - Its new rows are written at DAG height 1, 2, … against its own LEFT row (or a head's eviction record) at mesh height H.
  - `set.setValue` drops them silently.
  - It writes again only on an unrelated event: a new connection, a relay change, or `POST /v1/dnt/_node`.
- **This is a regression from a8c1055 (#51, Mar 2026),** which removed the 2-minute `ReannounceLocalServices` written for exactly this case.

## Evidence

- **Go regression test:** 10 of 10 runs fail on 4b6e90f, 10 of 10 pass with the fix (`-race`).
- **E2E with real binaries** on localhost, 3 runs per cell:
  - Public-address worker, baseline: never routable in 180 s in 9 of 9 runs at restart delays 0, 20 and 70 s, while the worker reports itself `ready`.
  - Public-address worker, fixed: medians 15.5, 19.9 and 20.0 s. With the optional patch 3: 6.5–8.6 s.
  - The issue's POST workaround on the baseline: the worker is listed 5–20 s after the POST, depending on the head's ping phase.
  - Default relay worker with a 20,000-record history, baseline: stuck in 1 of 3 runs, about 20 s in the other 2. This is a race against the single re-announce about 10 s after start.
- CSVs are in `results/`.

## Fix

Two patches in `patches/`, authored as snarayana, not pushed:

1. go-ds-crdt `addDAGNode`: Lamport rule, priority = max(head height, current priority of the keys) + 1.
2. Per-run `incarnation` plus a PutHook guard that republishes our own row when it holds an earlier run's record.

Checks:
- applies cleanly on PR #72, and the protocol tests pass there;
- `go vet` clean;
- golangci-lint v2.15: 0 issues on `internal/protocol/...`.

## For the summary

Pitch line: "#73 is a CRDT priority regression; two small patches plus a deterministic test bring a restarted worker back in about 15–20 s instead of never/minutes, with no periodic writes."
