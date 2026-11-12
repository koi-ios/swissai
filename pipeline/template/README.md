# Template

A skeleton for the next project, built on [../DOCTRINE.md](../DOCTRINE.md).

## Start a project in this workspace

```bash
cp -R pipeline/template/project opportunities/10-short-name    # then edit project.toml
```

Once the migration in [../MIGRATION.md](../MIGRATION.md) is applied, `make new-project SLUG=10-short-name` does the same and fills in the id. The other `make` targets the template names (`status`, `check`, `budget`, `spend`, `export`) are created by that migration too.

A new project starts parked. Make it active only when one of the two active slots is free (D2).

## What is in `project/`

| File | Stage | Who writes it |
|---|---|---|
| `project.toml` | 0 Intake | agent drafts, I decide; scripts read it |
| `STATUS.md` | every stage | the session working on the project, when the state changes |
| `ASSUMPTIONS.md` | 2 Hypotheses | agent proposes, I write the predictions and kill criteria |
| `harness/measure.py` | 3 Harness | copied as is: run records, interleaved schedule, small-n analysis |
| `harness/README.md`, `harness/known_bad/` | 3 Harness | how to freeze the harness, and the change it must reject |
| `MODAL.md` | 4 Sweeps | agent writes and dry-runs; I launch |
| `VERIFY.md` | 5 Verify | agent fills gates 1 and 2; I fill gate 3 |
| `WRITEUP.md` | 6 Write-up | I write the claim and the pitch; the agent fills tables from records |

Create these folders as they are needed: `research/` (facts with file:line at a pinned commit, sources I opened), `results/` (raw records, never typed by hand), `patches/<repo>/` (a `git format-patch` series against a named base commit), `drafts/` (PR text and messages, never sent by an agent), `vendor/` (clones and worktrees, ignored by git).

## A new upstream repository

Copy `upstream-card.md` to `upstream/<name>.md` and fill it from the repository's own files and history before writing any code for it.

## A new workspace

Copy the root files (`AGENTS.md`, `CLAUDE.md`, `Makefile`, `.gitignore`, `.claude/settings.json`), `scripts/` and `pipeline/` into an empty folder, run `git init`, edit `pipeline/workspace.toml`, and delete this workspace's `pipeline/research/` notes only if the next problem is unrelated. The doctrine and the scripts do not depend on this project's subject.
