# Verification: NN-slug

The three gates of D5. An agent's "done" or a green test run is not evidence; what is written here is.

## Gate 1: deterministic checks

| Check | Command | Result (date, commit) |
|---|---|---|
| The upstream repository's own tests and lint | from `upstream/<repo>.md` | |
| Correctness oracle (output equality at a pinned batch size, or this project's equivalent) | | |
| Harness unchanged since it was frozen | `shasum -a 256 -c HARNESS.sha256` | |
| Known-bad patch is rejected | see `harness/known_bad/` | |
| Statistics self-test | `python3 harness/measure.py selftest` | |

## Gate 2: review in a fresh context

A session that did not write the work, from another model family when one is available, with the repository and the right to run code. Its findings are advice.

- Reviewer (model, date):
- Brief given (no conclusions, only the claim and where the evidence is):
- Findings, and what was done about each:

## Gate 3: me

- [ ] I re-derived each headline number from `results/runs.jsonl`, without the agent
- [ ] I read every changed line that goes upstream
- [ ] I can explain the design and the most surprising number without notes
- [ ] Every number in `WRITEUP.md` is labelled MEASURED, PREDICTED or ESTIMATE
- [ ] Winners were re-measured on fresh runs; the held-out configuration agrees

Headline numbers, re-derived:

| Claim | Value in the write-up | My re-derivation | Raw records |
|---|---|---|---|
| | | | |
