# Handoff: where the pipeline work stands

Updated 12 Nov 2026, after phase 5. A fresh session can resume from this file and the ones it names.

## State

Everything in the brief of 12 Nov 2026 is done, and nothing has been pushed, sent or spent.

| Phase | State | Evidence |
|---|---|---|
| 0 `git init`, baseline | done | commit `a0997fd`; `.gitignore` |
| 1 research | done | `pipeline/research/`, `pipeline/RESEARCH.md` |
| 2 doctrine, template | done, audited | `pipeline/DOCTRINE.md`, `pipeline/template/`, `pipeline/AUDIT.md`, `pipeline/AUDIT-RESPONSE.md` |
| 3 migration plan | approved by the user | `pipeline/MIGRATION.md`; the approval is in `pipeline/DECISIONS.md` |
| 4 plan applied | done, reviewed | `git log`; `pipeline/MIGRATION-REVIEW.md`; `MIGRATION.md` section 9 |
| 5 export dry run on 06, verification | done | `opportunities/06-opentela-73/export/OpenTela/`; `pipeline/REPORT.md` |

## Start here

1. `make status`, then `make check`.
2. `pipeline/REPORT.md`: what changed, what the user must decide, what risks remain.
3. `AGENTS.md` for the rules; `pipeline/DOCTRINE.md` for the method; `pipeline/DECISIONS.md` for why.

## What is waiting, and on whom

- On the user: every item in `pipeline/REPORT.md`, "Yours to decide". The first is whether to send the 06 series; its checklist is `opportunities/06-opentela-73/export/OpenTela/CHECKLIST.md`.
- On a project session: 02's GPU check (after `make budget PROJECT=02 COST=6`), rebasing the servekit series in the order 02, 08, 03, and pinning GPU names in each `modal_app.py`.
- On nobody yet: retiring the older clones, one project at a time (`MIGRATION.md`, section 4).

## Things a fresh session would not guess

- Commits here use the repo-local identity `swissai local`. The machine's global git identity must not be used here. Commits exported for upstream use the identity in `pipeline/workspace.toml`.
- The session's web-search quota ran out during the research. `RESEARCH.md` says which decisions that touched.
- The GitHub API allows 60 unauthenticated requests an hour.
- No second agent is installed, and `pipeline/sandbox.settings.json` has not been run in a live session.
- `make test` restores `08/results/cpu_repro.json` after 08's reproduction suite rewrites it.
- An export refuses to run twice; `make export PROJECT=06 ARGS=--fresh` keeps the earlier worktree, branch and output under dated names.
