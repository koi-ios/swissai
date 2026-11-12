# Export report: 06-opentela-73 to eth-easl/OpenTela

Run 2026-11-12T14:10:51. Base: `origin/main` at `4b6e90f19734` (2026-11-02). Branch: `fix/restarted-worker-row`. Worktree: `opportunities/06-opentela-73/vendor/export/OpenTela`. Nothing was pushed.

## Commits

Author and committer: Soorya <soorya.narayananiv@gmail.com>.

- `4351dae` fix(crdt): rank a new write above the values it replaces
- `72d1d82` fix(protocol): republish our own row when it holds an earlier run's record

## The repository's own checks

- pass: go vet. `go vet ./...`. Log: `export/OpenTela/checks/01.log`
- pass: unit tests with the race detector (the flags of `make test`, without its download of gotestsum). `CGO_ENABLED=1 go test -race -count=1 -covermode=atomic -coverprofile="${TMPDIR:-/tmp}/opentela-coverage.txt" ./...`. Log: `export/OpenTela/checks/02.log`
- pass: lint: golangci-lint v2.15, as CI runs it. `golangci-lint run --timeout=5m`. Log: `export/OpenTela/checks/03.log`

## Leak check

2 commit(s), 7 file(s), +559 -1 lines.
- pass: no private strings, scaffolding files, placeholder identities or refused trailers

## PR description (`PR.md`)

- pass: every section the repository uses is present; no private strings

## Result

Ready for the user's review.

## Amended after the run (12 Nov 2026)

At the user's decision the second commit's message says `Refs #73` instead of `Fixes #73`; the commit was `0f36e78` during the run and is now `72d1d82`. Only the message changed: the tree is identical, so the checks above still apply. The leak check was rerun and passes, and `series/` was regenerated. The stored patch `patches/0002-...` carries the same wording.
