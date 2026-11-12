# Migration plan: put the workspace on the doctrine

Written 12 Nov 2026 by the coordinating session. **Nothing below has been applied.** Phases 0 to 2 are done: the repository exists (baseline `d6debc9`), and `pipeline/` and `upstream/` hold the research, the doctrine, the template and five repository cards. This plan is what remains, and it waits for approval.

Principles: no project folder moves, nothing is deleted, no project's code, tests, patches, results, STATUS or WRITEUP is edited. The plan is 2 renames, 1 move, about a dozen line edits in four index files, and about 50 new files, most of them small. Every step is one commit and can be reverted on its own.

## 1. What the audit found

Your four problems, checked:

| Problem | Finding |
|---|---|
| Not under version control | Fixed in phase 0: 769 files tracked, about 15 GB ignored, no remote. |
| About 30 clones and worktrees | 38 git checkouts: 23 clones and 12 worktrees, plus 3 submodule stubs inside an unpacked tarball. servekit is in 5 folders (6 checkouts), SGLang 0.5.20 in 6, model-launch in 3 (5 checkouts), OpenTela 9 times in 06. All 12 worktrees point at absolute paths in both directions. 04's change is uncommitted on a detached HEAD, and I checked that its diff is byte-identical to the tracked patch. |
| Index files disagree | vLLM: 01 targets 0.32.0 (inferred), the others 0.33.0, and nothing public pins what `vllm_base` runs. 08's issues are #19 and #14: `IDEAS.md` was right and `opportunities/README.md` wrong. The harness self-test has 57 checks, not 55. |
| No spend tracking | Confirmed. Also: every `modal_app.py` names GPUs unpinned (`"H100"`), which Modal may run on other hardware. |

Found in addition:

- **Hand-over notes live outside the workspace.** Four STATUS files point at a "shared notes folder". It is `~/Library/Application Support/Claude/side-session-notes/local_e5ab…/`, with nine notes, unversioned.
- **Two stale instruction files load themselves.** `serve-kit-vllm-prelim/CLAUDE.md` and `sglang-cold-start/CLAUDE.md` are chat handoffs from 11 Nov with superseded tasks. Claude Code loads them whenever it reads a file in those folders.
- **There is no root instruction file.** The rules live in `opportunities/SHARED_CONTEXT.md`, which no tool loads by itself.
- **06 holds state that no tracked file has:** a one-line edit to `host.go` used to build the e2e binaries. It is described in `e2e/README.md` but exists only in three worktrees.
- **06's clone also has three superseded branches** with authors `OpenTela73 Fix <noreply@example.invalid>` and committers `t <t@t>` and `check <check@invalid>`. The tracked patches come from the clean branch `fix/73`.
- **Every stored SGLang patch subject fails SGLang's PR title check** (enforced since 13 Oct 2026).
- **No outward credentials are reachable from an agent shell:** no credential helper, no SSH keys, no `gh`, Modal or Hugging Face tokens. Good, and worth keeping.
- **No second agent is installed** (no Codex, Gemini, Cursor or OpenCode), so adapters for one can be written and linted but not run here.

Patch problems, checked against the cards:

| Problem you listed | Finding |
|---|---|
| Placeholder authors | Confirmed: `cache-audit <noreply@example.invalid>` (01), `pystartup <pystartup@localhost>` (05), a folder path (04), "(to be filled by the sender)" (08's SGLang patches), no header at all (09). 02, 03, 06, 07 and 08's servekit patches carry your git identity. |
| No `Signed-off-by` | True, and right for four repositories, which neither require nor use it. Only vLLM requires it, and only a human may add it. |
| `Co-Authored-By` inconsistent | Confirmed: missing on 04, 08's SGLang patches and 09. The three lab repositories carry it on about a third of commits; SGLang on 71 of 200. |
| Branch names | 05's `pystartup-bytecode` breaks model-launch's rule; 01's `feat/shared-jit-caches` fits it. |
| servekit series conflict | Not re-tested here; the recorded order 02, 08, 03 stands. Rebasing them is real work for a project session, outside this migration. |

## 2. Every existing artifact and its role

| Artifact | Role in the doctrine | Action |
|---|---|---|
| `opportunities/README.md` | Program summary and the independent verification record | Keep. Edit 3 lines (section 3) |
| `opportunities/SHARED_CONTEXT.md` | Verified platform facts and Modal conventions, read on demand | Keep. Its rules move to `AGENTS.md`; 5 lines edited |
| `IDEAS.md` | Index of projects | Move to `archive/`. `PORTFOLIO.md`, generated, replaces it |
| `opportunities/NN/STATUS.md` | Project state (D9) | Keep, unchanged |
| `opportunities/NN/WRITEUP.md` | Stage 6 write-up | Keep, unchanged |
| `opportunities/NN/MODAL.md`, `modal_app.py`, 09's `RUNBOOK.md` | Stage 4 runbook | Keep, unchanged. `make check` warns about unpinned GPU names; each project fixes its own before its first paid run |
| `opportunities/NN/drafts/` | Outward text, never sent by an agent | Keep, unchanged |
| `opportunities/NN/patches/` | The tracked record of a change, and the input of the export | Keep, unchanged, placeholders included. The export fixes identity and trailers on the way out |
| `opportunities/NN/results/` | Evidence | Keep, unchanged |
| `opportunities/NN/research/`, 01's `kb/` | Stage 1 facts | Keep, unchanged |
| Project code and tests | The work | Keep, unchanged |
| `opportunities/NN/vendor/`, `.venv/`, 06's `toolchain/`, 07's `models/` | Disposable build inputs, ignored by git | Keep on disk; see the clone plan |
| `handoff-notes/` | Leads from the superseded parallel attempt; cited by two WRITEUPs | Keep in place. Gains a `side-session/` folder |
| `serve-kit-vllm-prelim/` | The shared launch harness, owned by project 03 | Keep. Its `CLAUDE.md` is renamed |
| `sglang-cold-start/` | The evaluation the nine projects came from | Keep. Its `CLAUDE.md` is renamed |
| `data/` | Shared read-only inputs | Keep. Gains a checksum file |
| `pipeline/`, `upstream/` | Method, evidence, template, repository cards | Keep |

## 3. File by file

**Renamed or moved (3).**

| From | To | Why |
|---|---|---|
| `serve-kit-vllm-prelim/CLAUDE.md` | `serve-kit-vllm-prelim/PLAN-2026-11-11.md` | A dated plan, not instructions. Stops it loading as instructions |
| `sglang-cold-start/CLAUDE.md` | `sglang-cold-start/PLAN-2026-11-11.md` | Same. The link in `EVALUATION.md` is updated |
| `IDEAS.md` | `archive/IDEAS.md` | Replaced by the generated portfolio |

**Edited (index files only).**

| File | Edit |
|---|---|
| `opportunities/README.md` | 08's row: "#19 and #9" becomes "#19 and #14". Under "Modal plan": one line pointing at `budget/` and at the open question of accounts. Top: one line pointing at `PORTFOLIO.md` |
| `opportunities/SHARED_CONTEXT.md` | "Rules for every session" becomes a pointer to `AGENTS.md`. The numbers rule points at D6. Modal conventions gain pinned GPU names and `retries=0`. Platform facts gain one line on vLLM versions and a corrected description of servekit #9 |
| `sglang-cold-start/EVALUATION.md` | One link: `CLAUDE.md` becomes `PLAN-2026-11-11.md` |
| `handoff-notes/README.md` | Two lines describing `side-session/` |

**Created.**

| Path | What it is |
|---|---|
| `AGENTS.md` | The one instruction source, about 50 lines: commands, rules, folder rules, where things are |
| `CLAUDE.md` | `@AGENTS.md` plus three Claude-only lines |
| `.claude/settings.json` | Excludes `vendor/` from instruction loading; denies `git push`, `gh` and `modal run` in agent sessions |
| `pipeline/sandbox.settings.json` | A settings file for sessions that work inside clones or run upstream code: the tool's sandbox on, network limited to the model API, package indexes and github.com (D11). Used on purpose, not by default, because ordinary sessions need the web |
| `.gemini/settings.json` | One setting so Gemini CLI reads `AGENTS.md`. Codex, Cursor, Copilot, OpenCode and Amp read it without an adapter |
| `Makefile` | A menu; every target is one line calling `scripts/` |
| `scripts/check.py` | `make check`: instruction-file paths and targets exist; every project has a valid `project.toml`, STATUS and WRITEUP; at most two active projects; generated files are fresh; the ledger is within its caps; no stray instruction files outside `vendor/`; no push, `gh`, Modal or Hugging Face credential is reachable. Warnings for placeholder patch authors and unpinned GPU names |
| `scripts/backup.sh` | `make backup DEST=…`: writes a `git bundle` of this repository to a place you choose. Run by you |
| `scripts/portfolio.py` | `make status` (terminal) and `make portfolio` (`PORTFOLIO.md`) |
| `scripts/budget.py` | `make budget` (the gate), `make spend` (append to the ledger) |
| `scripts/test.py`, `pipeline/tests.toml` | `make test PROJECT=NN`: reruns a project's suites and compares with the counts in `opportunities/README.md` |
| `scripts/clones.py` | `make clones`: every checkout, its role, branch, dirty state, and whether its branch equals the tracked patches |
| `scripts/export.py`, `scripts/leakcheck.py` | `make export PROJECT=NN` and `make leakcheck` (section 5) |
| `scripts/new_project.sh` | `make new-project SLUG=…` |
| `PORTFOLIO.md` | Generated: status, next action, stakeholder, upstream target and state, Modal dollars planned and spent, for the nine |
| `opportunities/NN/project.toml` (9) | The machine-read card. The only file this migration adds inside each project folder |
| `pipeline/workspace.toml` | Caps ($90 total, $25 a project), the identity for upstream commits, private strings for the leak check |
| `budget/plan.csv`, `budget/ledger.csv`, `budget/README.md` | The eight planned Modal steps as rows; an empty spend ledger |
| `upstream/<repo>.toml` (5), `upstream/README.md` | Each card's export rules in a form scripts can enforce |
| `handoff-notes/side-session/` (9 files) | Copies of the notes from the app folder. The originals stay where they are |
| `data/SHA256SUMS` | The hash of the ignored trace parquet |
| `archive/INDEX.md` | One line per archived item: what, from where, when, why |
| `opportunities/06-opentela-73/e2e/harness-only.diff` | The uncommitted one-line edit, captured |
| `opportunities/06-opentela-73/export/` | Output of the dry run: the exported series, `PR.md`, `CHECKLIST.md`, the leak-check report |
| `pipeline/HANDOFF.md`, `pipeline/REPORT.md` | Resume point; final report |

No nested instruction files. The two folder rules that an agent cannot infer (only project 03 edits the shared harness; `data/` is read-only) fit in two lines of the root file.

## 4. Clone plan

**Target.** One canonical clone per upstream repository at `vendor/<name>/` in the root, ignored by the existing `vendor/` rule: OpenTela, servekit, model-launch (full clones), sglang and vllm (`--filter=blob:none`, no checkout). Each sets `worktree.useRelativePaths` (Git 2.48 or later; 2.50.1 is installed), so the workspace can move without breaking links. Each gets agent files listed in `.git/info/exclude`. New work happens in worktrees of these clones under the project's own `vendor/`. Exports start from them.

**The 35 existing checkouts stay where they are for now.** Converting them all today would put the acceptance test at risk for little gain:

- Test commands, scripts and nine virtual environments refer to the present paths. The recorded test counts are the criterion for this migration.
- Twelve worktrees break if their folder moves.
- The disk saved is about 2 GB of 15.

Instead:

1. `make clones` proves, for each project, that the branch in its clone equals its tracked patch series. A clone that passes is disposable: it can be rebuilt from upstream plus `patches/`.
2. State that is in no tracked file is captured first (06's one-line edit).
3. A project's checkouts are retired when that project is next worked on: its session moves them to `archive/`, recreates them as worktrees of the canonical clone at the same paths, and reruns `make test PROJECT=NN`. `scripts/clones.py retire NN` does the move and the listing.
4. The three superseded branches in 06's clone stay there, unexported. Nothing is rewritten.

## 5. The upstream boundary

Each card in `upstream/` has a `.toml` beside it with the enforceable part: branch pattern, subject pattern, trailers to keep, add or refuse, whether a sign-off is required, the check commands, and the PR section headings.

`make export PROJECT=NN [REPO=name]`, run by you:

1. Fetch the canonical clone and add a fresh worktree from `origin/main` at `opportunities/NN/vendor/export/<repo>/`, on a branch named by the card.
2. Apply the series with `git am --3way`. If it does not apply, stop: rebasing is work for the project.
3. Set author and committer to the identity in `pipeline/workspace.toml`. Keep or drop trailers as the card says. Never add `Signed-off-by`: where a repository requires it, the checklist gives you the command.
4. Run the repository's own checks, with a scrubbed environment and no network where they allow it.
5. Run the leak check on the commits and the diff: private paths and strings, scaffolding files, placeholder identities, trailers against the card, subject pattern, secrets.
6. Check `export/PR.md` against the repository's section headings and for numbers without n and setup.
7. Write `export/CHECKLIST.md` for you: read every changed line (files and line counts listed), the questions the card leaves open, the sign-off command if needed, then the push and PR commands. The script runs none of them.

**Dry run on 06, in phase 5.** Fetch OpenTela, apply the two patches, run its checks (`go test -race ./...` and golangci-lint v2.15, with the folder's own Go toolchain), leak check, restructure `drafts/PR.md` into the headings its maintainers use (Problem, Root cause, Fix, Evidence, Testing), write the checklist. It stops there. One deviation: OpenTela's `make test` downloads a test runner with `curl`; the dry run runs the same `go test` flags without the download.

## 6. Order of work after approval

Small commits, in this order: (1) renames, the move and `archive/INDEX.md`; (2) index fixes and the side-session copies; (3) `AGENTS.md`, `CLAUDE.md` and tool settings; (4) nine `project.toml` and `workspace.toml`; (5) `Makefile`, scripts, `budget/`, `PORTFOLIO.md`; (6) `upstream/*.toml`, export and leak check; (7) canonical clones and the 06 capture; (8) the 06 dry run; (9) verification: `make check`, every project's tests against the recorded counts, a fresh subagent checking the result against this file, the final report.

Not done by this migration: pushing, opening or commenting on anything, sending anything, spending on Modal, rebasing the servekit series, fixing the stored patches in place, pinning GPU names in the nine Modal apps.

## 7. Yours to decide

None of these blocks the migration. Defaults are what the scripts will assume until you say otherwise.

1. **Which two projects are active.** Default: 06 (export-ready, no GPU) and 02 (one $6 run decides it). The other seven are parked.
2. **What goes out first, and to whom.** The doctrine says one small evidenced PR per repository, and a message to the maintainer before anything large. 06 answers Xiaozhe's own issue.
3. **The name on upstream commits.** Git is set to `snarayana <snarayana@student.ethz.ch>`. vLLM's DCO needs a real name; the others have no rule.
4. **AI disclosure where a repository is silent** (OpenTela, servekit, model-launch, SGLang). Default: keep the `Co-Authored-By: Claude Opus 5.5` trailer and a one-line statement in the PR body, as their maintainers do, and leave out session links.
5. **OpenTela #73:** on top of open PR #72 or independent of it; with or without the optional third patch.
6. **Modal:** one workspace or three accounts. The credit is monthly, the terms are silent on several accounts and bar third-party use. Asking Modal and applying for academic credits costs an email each.
7. **`make` or `just`.** `make` is installed. Say so if you want a `justfile` as well.
8. **A second agent.** Installing one (for example Codex CLI) would let the adapters be run, not only linted.
9. **A backup.** This repository has no remote, so one disk holds the only copy. Default: none until you choose a place for `make backup`, or a private remote that only you push to.
10. **A container runtime.** The doctrine allows unattended loops and unwatched auto-mode sessions only inside a container or microVM (D7, D11). Docker is installed here but its daemon is off. Until you start one, there are no such sessions.

## 8. Risks

- This session itself read web pages and cloned repositories next to the private workspace, with network access and automatic permissions. Nothing outward was possible, since no credentials exist here, but under D11 future web research belongs in sessions that see public inputs only.

- A script could disagree with a card. The `.toml` files are written from the cards and the leak check prints which rule fired, so a wrong rule is visible.
- Renaming the two `CLAUDE.md` files changes what a session in those folders loads. That is the intent; the content stays one `git mv` away.
- The dry run executes OpenTela's test suite, which is upstream code. It ran on this machine on 12 Nov already; this time it runs with a scrubbed environment.
- Canonical clones need the network once (read-only fetches from GitHub).
