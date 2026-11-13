# Decision and conflict log

Append-only. One entry per decision that someone might later question, and one per conflict between sources of authority. Precedence, from the brief of 12 Nov 2026: for anything going upstream, that repository's rules win; otherwise the brief, then workspace ground truth (for facts), then research (for methods), then the playbooks.

Format: `## [date] kind | title`, then what was decided, why, and what it overrides. Kinds: `decision`, `conflict`, `finding`.

## [2026-11-12] decision | The baseline ignores vendor trees, toolchains, models, run logs, built binaries and the trace parquet

About 15 GB stays on disk and out of history; 769 files (33 MB) are tracked. The trace parquet (213 MB) is rebuilt from Hugging Face and its SHA-256 is in the baseline commit message. Why: the brief asked for it, and a patch series is the durable record of a change, not a clone.

## [2026-11-12] decision | Research agents ran on smaller models than the coordinating session

Five topics on Opus and six on Sonnet; the five repository cards on Sonnet; the audit on Opus. Why: the plan's 5-hour usage window stood at 8% before the fan-out, and on 11 Nov sixteen parallel sessions had exhausted it in about 20 minutes. The cost is weaker reading on some topics, which the notes' "Not verified" lists and the audit make visible.

## [2026-11-12] conflict | The deep-research skill's layout against the brief's file names

The skill wants `research_notes/<title>/` and a narrative report from a writer agent. The brief names `pipeline/RESEARCH.md` (a ledger and a disagreement map) and `pipeline/DOCTRINE.md` (decisions). Resolved for the brief: notes go to `pipeline/research/`, the ledger is assembled by script from the notes so no row is retyped, and the coordinating session writes the doctrine because it needs the workspace's facts. The skill's researcher instructions were still given to every research agent.

## [2026-11-12] finding | The shared web-search quota ran out during research

Ten of eleven notes list searches they could not run. RESEARCH.md states which decisions were settled before that point and which were cut short (D2, D9, D10, D13).

## [2026-11-12] conflict | Brief: "median, min–max, n" for numbers; research: three runs cannot support a comparison

The brief asks for numbers as median, min to max, n and setup. The workspace rule is "median of at least 3 runs with min to max". The benchmarking note shows that three runs describe but do not compare (B28, B07). Resolved without contradiction: numbers are still reported as median, min to max, n and setup, as the brief says. D6 adds that a claim that one arm beats another needs an A/A pilot and six interleaved pairs, and that n = 3 is labelled descriptive. Existing write-ups keep their n = 3 tables; they are descriptive, and none claims significance.

## [2026-11-12] conflict | Brief: the export writes the PR description; research: maintainers want the author's own words

Several projects bar AI-written PR descriptions, and none of the five target repositories does (M26, M36; the five cards). Resolved for the brief, with a guard: the export produces a PR description in the repository's own section structure, filled with the evidence. The checklist then has the user rewrite or own it line by line before sending. Where a repository's maintainers end their PR bodies with a "Generated with Claude Code" line, the export keeps that line.

## [2026-11-12] conflict | Brief: `just`; research: no evidence for either runner, and `just` is not installed

The brief says "a command runner (the playbooks use `just`)" and accepts "`just check` (or equivalent)". Resolved: `make`, which is preinstalled, as a thin menu in which every target is one line calling a file in `scripts/`. A `justfile` with the same targets is a five-minute addition if the user installs `just`; installing software on the machine is the user's call.

## [2026-11-12] conflict | Brief: three $30 Modal accounts; research: one account

The current plan spreads about $88 over three accounts. The provider's terms have no clause on several accounts, bar third-party access, and describe the credit as monthly (X01, X02). Not resolved here, because spending and accounts are the user's decision. The doctrine plans on one workspace (D10); MIGRATION.md lists the question.

## [2026-11-12] conflict | Playbooks against research

Recorded in the last table of DOCTRINE.md: eight playbook rules kept, eight changed, four dropped. The largest changes: the instruction file is no longer treated as the main asset (E18), session rituals and the handoff rewrite are dropped (no evidence), the idea mechanisms are dropped (L14), and the measurement rule is replaced (B28).

## [2026-11-12] decision | The upstream cards were written before the migration was approved

They are new files in a new folder, they change nothing that existed, and the migration plan depends on what they found. Each card's key claims were re-checked against the clones by the coordinating session (trailer counts, the model-launch branch rule, vLLM's AGENTS.md lines, OpenTela's CI commands).

## [2026-11-12] finding | The user's observations about the upstream repositories, checked

- "OpenTela and model-launch commit their own CLAUDE.md/AGENTS.md": model-launch has both; OpenTela has CLAUDE.md only; servekit has neither.
- "Histories are full of `Co-Authored-By: Claude` trailers, including the maintainers' own": about a third of recent commits in each of the three lab repositories (33 of 100, 34 of 100, 7 of 19), mostly the maintainers'. Common, not universal.
- model-launch's branch rule: confirmed (AGENTS.md:7-14).
- vLLM: disclosure in the PR, trailers, DCO sign-off: confirmed. Duplicate checks: required by AGENTS.md for AI-assisted work, enforced by no workflow. Bans: "can result in", no mechanism found.
- SGLang's template: six sections, not four. It also enforces a PR title pattern since 13 Oct 2026, which every stored SGLang patch subject fails.
- "No `Signed-off-by` anywhere": true of the patches, and correct for four of the five repositories, which neither require nor use it. Only vLLM requires it, and there only the human may add it.

## [2026-11-12] finding | Index disagreements, resolved from evidence

- 08's servekit issues: #19 (MLA round trip, open) and #14 (MXFP4 expert weights dropped, closed by #16). `IDEAS.md` was right; `opportunities/README.md` said #9, which is a different open issue (its title says quantization, its body says pipeline-parallel presharding is missing).
- vLLM version: project 01 targets 0.32.0 because it infers that from release dates; projects 03, 04, 05, 07 and 08 target 0.33.0, the newest release on 11 Nov. Nothing public pins the version the platform's `vllm_base` image runs. It stays an open question for the maintainers.
- The harness self-test has 57 checks, not 55 (`IDEAS.md`, and the older `serve-kit-vllm-prelim/CLAUDE.md`).
- Four STATUS files point at a "shared notes folder". It is an app-managed folder outside the workspace holding nine hand-over notes.

## [2026-11-12] finding | The independent audit, and what changed because of it

A fresh Opus session audited DOCTRINE.md, RESEARCH.md and the template (`pipeline/AUDIT.md`): 2 high, 11 medium and 17 low findings. It fetched twelve primary sources and all matched their ledger rows. `pipeline/AUDIT-RESPONSE.md` lists the action taken on each finding. The two high findings:

- D11 had kept only the credentials rule and dropped network and isolation controls that the security note rates high confidence. D11 and D7 now require default-deny egress for sessions that work in clones, and a container for unattended loops.
- `measure.py` keyed records by arm and pair, so a main run overwrote the A/A pilot's baseline and the reported noise floor was wrong. Records now belong to a block, duplicates are refused, left-out runs are counted, and the verdict is computed. The self-test covers the failing case.

## [2026-11-12] finding | Correction to an earlier entry

The entry "The shared web-search quota ran out during research" says ten of eleven notes list searches they could not run. The count is nine; notes 06 and 08 list unreachable primary pages instead.

## [2026-11-12] finding | Pre-migration test baseline

Before any change to the workspace, every project's suites were rerun. All match the counts recorded in `opportunities/README.md`: 01: 53 passed, 2 skipped; 02: 187 passed with the 2 known macOS errors, plus 8; 03: 195 passed, 3 skipped; 04: 34; 05: 41; 06: `go vet` clean, 17 packages ok; 07: 183 passed, 2 known errors; 08: 176 passed, 2 known errors, plus 9; 09: 37. Two notes for the test runner: the harness self-test needs servekit importable and was not started correctly in this run, and 08's reproduction suite rewrites `results/cpu_repro.json` with a temporary path (restored from git).

## [2026-11-12] decision | The user approved the migration plan, and chose the identities

Asked in this session after a forked session relayed an approval (a relay is not an approval). The user's answers:

- Apply `pipeline/MIGRATION.md` as written, with the section 7 defaults, then dry-run the export on 06. Nothing pushed, sent or spent.
- Commits in this private repository carry a neutral local identity, `swissai local <local@swissai.invalid>`, set with `git config --local`. The machine's global identity is left alone and is not used here.
- The five commits that already existed were re-authored to that identity (author and committer; dates, messages and contents unchanged). The baseline commit is now `a0997fd`.
- Commits exported for upstream carry the identity recorded in `pipeline/workspace.toml`, which the user gave. The export's leak check treats the machine's global git identity as a private string.

## [2026-11-12] decision | The migration was applied; what differs from the plan

Applied in nine commits after the user's approval. Differences from `MIGRATION.md`, all small:

- `make clones-verify` proves seven series equal their clone's branch. It cannot compare 05's two clones (shallow, and nothing is fetched) or the two SGLang series that no clone holds (08's alias patch, 09's diff).
- The canonical `sglang` and `vllm` clones are single-branch, without tags or blobs. Enough for an export; not a full mirror.
- `pipeline/sandbox.settings.json` uses key names from the Claude Code documentation but has not been run in a live session. No second agent is installed, so `.gemini/settings.json` and the native `AGENTS.md` reading of other tools are linted, not exercised.
- The credentials check in `make check` is a warning, not a failure, so that the user's own later login does not break the lint. `make check STRICT=1` makes it fail.

## [2026-11-12] finding | The export dry run on 06

`make export PROJECT=06` on a fresh worktree of OpenTela `origin/main` (4b6e90f, fetched 12 Nov): both patches apply; author and committer are the upstream identity; the `Co-Authored-By` trailer is kept, as the card's observed practice has it; `go vet`, `go test -race` with the Makefile's flags (17 packages ok) and golangci-lint v2.15 pass; the leak check passes; the PR text has the headings OpenTela's contributors use. The leak check was also run on a superseded branch with a placeholder author and failed it, as it should. Nothing was pushed: the worktree has no remote but the read-only origin, and this machine holds no credentials. One deviation, stated in the card and the rule file: OpenTela's `make test` downloads a test runner with `curl`; the same `go test` flags were run without it.

## [2026-11-12] finding | The independent review of the migration, and corrections

A fresh session checked the result against the plan (`pipeline/MIGRATION-REVIEW.md`): no high findings, 6 medium, 11 low; nothing deleted, no project file other than the new cards touched, no sign of a push. Acted on:

- Thirteen third-party instruction files sit in 06's Go module cache, outside `vendor/`. The excludes now cover `toolchain/` and `.venv/`, `AGENTS.md` names them, and `make check` fails if the excludes are missing.
- `make clones-verify` miscounted its skips (it said 2; there are 4). Fixed. `make clones` no longer shows a meaningless dirty count for clones without a checkout.
- `make check` now warns about 02's unpinned GPU (an f-string) and 09's patch without an author.
- The export notes a commit message that closes an issue: 06's second commit says `Fixes #73` while its PR text says `Refs #73`. The user picks.
- `MIGRATION.md` and `HANDOFF.md` said the plan was waiting for approval. Both are updated; section 9 of the plan lists every difference.
- Correction to the entry "The migration was applied": it said nine commits; there were eight at that point.

Left as they are, on purpose: the rule files are slightly looser than the cards (servekit and model-launch subject prefixes); 06's STATUS.md still describes the state before the export, because only that project's session edits it.

## [2026-11-12] decision | 06 refers to issue #73 and does not close it

The user chose `Refs #73` over `Fixes #73` for the second commit, to match the PR text and OpenTela's own practice (no closing keywords in its PR bodies). Changed in the stored patch `opportunities/06-opentela-73/patches/0002-...` (one word in the message; the diff is untouched) and in the exported commit (now `72d1d82`, same tree). This is the one edit to a project's patch made by the coordinating session, at the user's instruction.

Follow-up, same day: the project's older draft `opportunities/06-opentela-73/drafts/PR.md` opened with `Fixes #73`. A closing keyword in a PR description closes the issue on merge just as one in a commit does, so that word is now `Refs` too (one word; nothing else in the draft changed). No text that could be sent closes the issue. This settles finding F11 in `MIGRATION-REVIEW.md` and the open item in the entry above ("The user picks"). `MIGRATION-REVIEW.md` is a reviewer's record of the state it saw and is left as written.

## [2026-11-13] decision | A private GitHub remote that only the user pushes to

The user asked to put this repository on GitHub so it can be used on more than one machine. This replaces "local-only: no remote, never pushed" in `AGENTS.md`, `.gitignore` and the DOCTRINE row on pushing. What stays the same: agents never push and never run `gh`, because `.claude/settings.json` still denies both, and D11 holds. The push credential is the user's. `make check` warns when an agent shell can reach it, as it was built to.

The repository must be private. A read-only audit of the tracked files, the two untracked `opportunities/EXPLAINED.*` files and `git log -p --all`, made before the first push, found:
- No secrets. The patterns searched were hf_, ghp_, github_pat_, sk-, Modal ak-/as-, AKIA, private keys and xox. The only hits were placeholders and the regex list in `scripts/leakcheck.py`.
- Private material: a copy of a third party's experiments repository with their cluster username in its paths (`data/cold-start-elimination-experiments-swissai/`), unsent drafts addressed to named people, references to a private swiss-ai repository, the owner's email addresses (in patches and `pipeline/workspace.toml`) and a machine hostname in 09's results. History holds about 4,400 lines with this machine's home path.

To make other machines work:
- `.gitignore` adds secrets (`.env*`, keys, `.netrc`, `.modal.toml`), per-machine agent files (`CLAUDE.local.md`, `.claude/settings.local.json`, `.claude/worktrees/`) and editor and packaging noise. No tracked file matches the new rules.
- A new `SETUP.md` covers the local identity, which a clone does not bring, and how to rebuild each ignored tree.

Left as found:
- Two paths would break on another machine: `opportunities/01-compile-caches-270/tools/fetch_sources.sh:10` hard-codes a miniconda Python, and `opportunities/08-preshard-roundtrip/scratch/probe_import.py:2` hard-codes a `sys.path`. Only those projects' sessions edit them.
- The records `MIGRATION.md`, `REPORT.md`, `AUDIT.md` and `AUDIT-RESPONSE.md` describe the state they saw, and are left as written.

## [2026-11-13] finding | `make check` does not see a GitHub login kept in the macOS keychain

This Mac's git uses the system-level `osxkeychain` credential helper, set in the Command Line Tools gitconfig, and the login keychain holds a github.com entry. `check_credentials()` in `scripts/check.py` reads only `git config --global` and a list of files, so it reported nothing. Any session that can run git over HTTPS can use this login, and the deny of `git push` in `.claude/settings.json` is the only guard. The `gh` CLI is not installed, and there are no SSH keys. Correction to the entry above: "`make check` warns when an agent shell can reach it" is not true for this credential. `scripts/check.py` is unchanged.
