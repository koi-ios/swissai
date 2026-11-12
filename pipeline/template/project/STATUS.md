# STATUS: NN-slug

Updated YYYY-MM-DD. Stage: intake (see pipeline/DOCTRINE.md, D1). Rewrite this file only when the state changes.

## Done

- 

## Not done or blocked

- 

## Next 3 actions

1. 
2. 
3. 

## Gates passed

Tick a gate only with the evidence beside it (a file, a commit, a run ID).

- [ ] 0 Intake: card in `project.toml` (claim, cheapest falsifier, kill criterion, upstream repo); an active slot was free
- [ ] 1 Archaeology: every fact cites file:line at a pinned commit or a source I opened; `upstream/<repo>.md` exists
- [ ] 2 Hypotheses: `ASSUMPTIONS.md` filled; maintainer or supervisor asked about large or unrequested work
- [ ] 3 Harness: smoke run on the real target; baseline reproduced; A/A noise floor; oracle; `HARNESS.sha256` written
- [ ] 4 Sweeps: planned runs done or stop rule hit; every run in `results/runs.jsonl`; spend in `budget/ledger.csv`
- [ ] 5 Verify: `VERIFY.md` complete
- [ ] 6 Write-up and export: `WRITEUP.md`; `make export` passed; I read every changed line
- [ ] 7 Close: post-mortem below

## Post-mortem (five lines, when closed)

