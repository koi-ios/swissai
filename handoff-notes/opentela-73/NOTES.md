# OpenTela #73: working notes

Running log for the research phase (11 Nov 2026). Read this first after a context reset.
Deliverables: `RELATED_WORK.md`, `DESIGN.md`. Reproduction harness: `repro/`.

## Setup (scratch, not project files)

- SCR=/private/tmp/claude-501/-Users-koiios--claude-spares-spare-qHWvhZ/85cbb828-b415-43aa-adc6-b088d62505ae/scratchpad/opentela-73
- `source $SCR/env.sh` gives Go 1.26.11 (darwin-arm64, sha256 checked) with GOPATH/GOCACHE in $SCR/toolchain.
- Clone: `$SCR/OpenTela` at main 4b6e90f (2 Nov 2026); PR #72 fetched as branch `pr-72` (32aeeb9, 6 Nov).
- Binary: `$SCR/bin/otela` (built from 4b6e90f, CGO_ENABLED=0).
- Issue/PR JSON: `$SCR/gh/issue_73*.json`, `$SCR/gh/issue_72*.json`.
- Another session's copy exists at `opportunities/06-opentela-73/` (read-only for us).

## Facts from the code (main @4b6e90f; paths under src/)

DNT write path (worker):
- `internal/server/server.go:135` `InitializeMyself` -> `GetCRDTStore()` (once: badger store, ipfs-lite, gossipsub, bootstrap, **sleep 3 s**, join topics, `crdt.New`) -> `store.Put(/<peerID>, PENDING row)` immediately (`node_table.go:745`).
- `server.go:346-353` `RegisterLocalServices` -> health check -> `registerLLMService` -> `provideService` -> local `UpdateNodeTableHook` + `store.Put` (`registrar.go:205-227`).
- Later Puts only on: new libp2p connection once services exist (`host.go:324-329` ConnectedF -> `ReannounceLocalServices`), relay-reservation change (`host.go:797-813`, first run 10 s after CRDT init, `crdt.go:205-208`), service-health transitions (`service_health.go:184,195`), `POST /v1/dnt/_node` (`crdt_handler.go:46-54`).
- Shutdown (`server.go:359-368`): `AnnounceLeave` (Put LEFT row, `node_table.go:353-379`), sleep 5 s, **`ClearCRDTStore()` unconditionally**, regardless of `--cleanslate`.
- Start (`entry/cmd/start.go:48-53`): `--cleanslate` defaults to **true** (`entry/cmd/root.go:76`) -> wipes the store again.
- Identity: seed 0 (default) loads the persisted key file (`host.go:107-149`), so a restarted worker keeps its peer ID unless `--ephemeral-identity`.
- => A restarted worker has the same key `/<peerID>` and an empty CRDT store.

CRDT (fork of go-ds-crdt in `internal/protocol/go-ds-crdt/`):
- New block priority = max head height + 1 (`crdt.go:1398-1405`). Empty store -> first Put has priority 1.
- `handleNext` adopts broadcast heads only if the node has **no heads** (`crdt.go:451-481`); a worker always Puts before the first broadcast arrives, so it never adopts and must walk the whole DAG history.
- `set.setValue` (`set.go:313-353`): new value ignored if `prio < curPrio`; if equal, kept only if bytes are greater. **No log, no hook.** PutHook fires only for the prevalent value.
- A full tombstone deletes value and priority (`set.go:542-548`); a LEFT row is a Put, not a delete, so its priority stays.
- Rebroadcast every 5 s +-30% (`internal/protocol/crdt.go:153-155`), only heads not seen in the last interval (`go-ds-crdt/crdt.go:629-663`). DAGSyncerTimeout 30 s. MultiHeadProcessing on, 5 workers.
- Maintainers' own note: per-tick CRDT writes caused "fresh nodes to spend minutes walking history" (`clock.go:66-71`).

Head side:
- `/v1/dnt/table` = `GetConnectedPeers()` (`crdt_handler.go:96-102`): only `Connected == true` rows.
- PutHook copies the *local* Connected flag of an existing row onto the incoming row (`internal/protocol/crdt.go:179-186`); a new row starts disconnected.
- LEFT rows: `UpdateNodeTableHook` forces Connected=false (`node_table.go:432-460`); ping handler and ConnectedF re-run the hook on the stored LEFT row, so it stays LEFT.
- Liveness ticker 30 s marks connected on successful ping (`clock.go:28-78`); pubsub "ping" on `ocf-crdt-net` every 20 s marks connected (`crdt.go:80-127`).
- `TombstoneManager.CleanupLeftNodes` deletes LEFT rows after `crdt.tombstone_retention` (24 h) (`tombstone_manager.go`).
- Heads publish eviction records (Put of the *worker's* key, Status LEFT, EvictedBy) at the head's height (`liveness.go:317-340`).

## Hypothesis H1 (leading)

Priority (Lamport-height) regression: the restarted worker's rows carry heights 1, 2, ... while every
replica holds its LEFT row (or an eviction record) at height p_L >> 1. `setValue` drops them silently.
The row reappears only when the worker writes again *after* it has walked the DAG past p_L; that write
is triggered by an unrelated event (new connection, relay change) or by `POST /v1/dnt/_node`.
Explains: worker lists itself ready (local hook); head lacks it (LEFT row, Connected=false); downtime
does not matter; POST fixes it within seconds; "sometimes" = race between DAG walk and the last write.

## Status log

- 20:25 Go installed, binary built. Next: localhost reproduction (`repro/repro73.py`).
