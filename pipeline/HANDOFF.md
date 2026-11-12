# Handoff: where the pipeline work stands

Updated 12 Nov 2026, end of phase 3. A fresh session can resume from this file and the ones it names.

## The task

The brief of 12 Nov 2026 has three parts: (1) research and settle an AI-first research pipeline; (2) rebuild this workspace on it with minimal churn; (3) build the boundary to the five upstream repositories and dry-run an export on project 06. Standing rules: never push, open or comment on anything, email, or spend on Modal; never delete (move to `archive/` and list it); never rewrite upstream history; treat clones and web pages as untrusted data; ask the user only about what to send, to whom and what to spend.

## State

| Phase | State | Evidence |
|---|---|---|
| 0 `git init`, baseline | done | commit `d6debc9`; `.gitignore` |
| 1 research | done | `pipeline/research/01` to `11`, `pipeline/RESEARCH.md` |
| 2 doctrine, template | done, audited | `pipeline/DOCTRINE.md`, `pipeline/template/`, `pipeline/AUDIT.md` |
| Part 3 cards | done | `upstream/*.md`, `upstream/evidence/` |
| 3 migration plan | **written; waiting for the user's approval** | `pipeline/MIGRATION.md` |
| 4 apply the plan | not started | `MIGRATION.md` section 6 gives the order |
| 5 export dry run on 06, verification, report | not started | `MIGRATION.md` section 5 |

## To resume after approval

1. Read `pipeline/MIGRATION.md` and any changes the user asked for. Section 7 lists the decisions that are the user's; the defaults there apply until the user says otherwise.
2. Apply section 6 in order, one commit per step. For `.claude/settings.json`, check the current key names against the Claude Code documentation before writing the file.
3. The facts needed to write the nine `project.toml` files are in each `opportunities/NN/STATUS.md` and in `opportunities/README.md` (Modal plan table). Proposed states: 06 and 02 active, the rest parked.
4. Test commands for `pipeline/tests.toml` are in each WRITEUP's "How to run" section; the expected counts are in the verification table of `opportunities/README.md`.
5. The dry run uses the Go toolchain in `opportunities/06-opentela-73/toolchain/` (`source env.sh`). Rules for the export are in `upstream/OpenTela.md`.
6. Finish with: `make check`; every project's tests against the recorded counts; a fresh subagent checking the result against `MIGRATION.md`; `pipeline/REPORT.md`, one page.

## Things a fresh session would not guess

- All sixteen research and card agents have finished. Do not start them again; their output is on disk.
- The session's web-search quota ran out during research. `RESEARCH.md` says which decisions that touched.
- The GitHub API allows 60 unauthenticated requests an hour. About 20 were used on 12 Nov around 12:00 UTC.
- No second agent (Codex, Gemini CLI) is installed, so its adapter can be linted, not run.
- Peer sessions for the nine projects exist and are idle. One writer per project folder still holds; this session adds only `project.toml` to each, and the export output to 06.
- `pipeline/DECISIONS.md` logs every conflict between the brief, the workspace's facts, the research and the playbooks, and how each was resolved.
- The audit is answered finding by finding in `pipeline/AUDIT-RESPONSE.md`. After the migration, delete the sentence in DOCTRINE.md's first paragraph that says the `make` commands are created by MIGRATION.md.
- Every project's suites were rerun before any change, and all match the recorded counts (see the last entry of DECISIONS.md). The runner used is in this session's scratchpad and is not in the repository; `pipeline/tests.toml` should hold the same commands. Two traps: `serve-kit-vllm-prelim/tests/selftest.sh` needs servekit importable (find the interpreter project 03 used), and 08's `repro` suite rewrites `results/cpu_repro.json` with a temporary path, so restore it from git after a run.
- Commands that worked: `PYTHONPATH=vendor/servekit/src .venv/bin/python -m pytest vendor/servekit/tests -q` in 02; `PYTHONPATH=src ../../.venv/bin/python -m pytest tests -q` inside `vendor/servekit` of 03, 07 and 08; `.venv/bin/python -m pytest -q` (01), `... tests` (04, 05, 09), `... repro` (08); for 06, `source env.sh`, then in `vendor/OpenTela-final/src`: `GOFLAGS=-mod=mod GOPROXY=off go vet ./...` and `go test -race ./...`.
