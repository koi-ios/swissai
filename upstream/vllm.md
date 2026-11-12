# Upstream card: vllm-project/vllm

Evidence gathered 12 Nov 2026 from main at `3b43d2b01c2f` (scratch clone) and from the v0.33.0 clone; read-only; these rules govern only code written for this repository. Full listings with line citations: `/Users/koiios/Documents/Projects/swissai/upstream/evidence/vllm.md` (section numbers below refer to it).

## At a glance

| Item | Rule (source) |
|---|---|
| Branch name | No rule; contributors use forks. Observed: `type/topic` or kebab-case (ev. 3) |
| Commit subject | `[Tag][Tag] Description (#N)`; 297/300 recent commits tagged; `(#N)` added at squash (ev. 4) |
| Trailers | `Signed-off-by:` (295/300); AI trailer `Co-authored-by: <Model> <noreply@...>` (209/300) (`AGENTS.md:135-144`) |
| Sign-off | Required, by the human submitter, `git commit -s` (`docs/contributing/README.md:193-199`) |
| AI disclosure | Mandatory in PR body and checkbox (`AGENTS.md:29-33`, template `:25`) |
| PR title | Starts with one or more `[Tag]` then description (`README.md:233-261`; `pr-title.yml:34`) |
| PR body | Overview, Claims, Validation, Details, then the checklist (template `:3-31`) |
| Checks to run | `pre-commit run --all-files`; relevant pytest; docs build if docs touched |
| Reviewers | CODEOWNERS and area reviewers; committers merge (`README.md:327-334`) |

## Evidence

1. **Snapshot.** main `3b43d2b01c2fb667101746e1acd33b6a3c9d0863`, 2026-11-12; default branch `main`; Apache-2.0 (`LICENSE:1-3`); regular releases `vX.Y.Z` roughly every 2 weeks, minor bumps, RC tags (`docs/contributing/release_process.md:7-24`); squash merges (`docs/contributing/labels.md:34`; 0 merge commits in last 300). v0.33.0 (`4ba787e`) has the same template and nearly the same AGENTS.md (2 lines fewer).
2. **Agent files.** `AGENTS.md` plus 8 nested `AGENTS.md` (csrc, tests, rust, parser, reasoning, tool_parsers...), 9 skills in `.agents/skills/` symlinked from `.claude/skills/`; no CLAUDE.md, `.cursor/`, copilot file. Key rules: duplicate checks with `gh issue view` and two `gh pr list --search` (`AGENTS.md:8-19`); no tiny busywork PRs (`:21-23`); "Pure code-agent PRs are **not allowed**" (`:27`); PR text must give non-duplication reason, test commands and results, model evals, "Clear statement that AI assistance was used" (`:29-33`); docs in same PR (`:34-36`); run `pr-checklist` skill before any PR, drafts included (`:37`). Consequence: "can result in automatic banning" (`:4`); bot: "may result in an immediate ban" (`new_pr_bot.yml:93`). No ban mechanism found in the repo.
3. **Branches.** No stated rule (grep). Fork PRs assumed (`README.md:358-363`). Observed: `perf/sm90-fp8-megamoe`, `fix-sagemaker-invocations-validation`.
4. **Commits.** Last 300 (9-12 Nov): Bugfix 28%, ROCm 22%, CI 14%, Perf 14%, Rust Frontend 8%, Frontend 6%, XPU 5%, Model 5%; Feature 1, Doc 5. Body is trailers only (0/300 have prose). `Signed-off-by:` 295 (e.g. `3b43d2b01`, `9a77b4201`, `f55b08ec0`; missing: `54b29e254`, `9cb56aae7`, `499a3d1e1`). AI `Co-authored-by:` 209 (e.g. `3b43d2b01`, `94febce75`, `0a77deacd`, `3ea131ca2`); strings: `Claude Opus 5.5 <noreply@anthropic.com>` 130, `Claude <noreply@anthropic.com>` 34, `Cursor <cursoragent@cursor.com>` 12, `Codex <noreply@openai.com>` 9, `Claude Sonnet 5.5 <noreply@anthropic.com>` 5, `Copilot <223556219+Copilot@users.noreply.github.com>` 3. `Assisted-by:` and `Generated-by:`: 0. Sign-off survives squash in 98%.
5. **PRs.** Template as in the glance table plus `- [ ] I used vLLM's /pr-checklist skill. (Mandatory for agents, optional for humans).` and `- [ ] AI assistance was used during the creation of this PR.` and four bold checklist items. Prefixes: any tag; well-known list at `README.md:241-257`. Of 22 merged outside PRs in the latest 50 closed, 17 use the template headings and tick the AI box; prose pattern: "AI assistance (Claude Code) was used to prepare this change. I reviewed every changed line and the test results." (#64765, #66777).
6. **Size.** Last 150 non-merge commits: median 3 files (1-153), median 86.5 lines (1-8,792; p75 361). Large outside merges: #63383 (8,792 lines), #61869, #65848. Guide: >500 LOC architectural changes need an RFC issue, else `rfc-required` (`README.md:312-317`); keep changes concise.
7. **Checks.** `pre-commit` workflow runs all hooks (`pre-commit.yml:67-70`) but only for PRs with `ready`/`verified` or authors with 4+ merged PRs (`:22-45`); `dco` check; title check on `ready` PRs; docs preview only when docs files change or label `build-docs` (`README.md:141-142`); Buildkite only after a maintainer `/ci run` (`:346-352`). Hooks and pins: ev. 7 (ruff v0.17.2, typos v1.51.1, clang-format v23.1.3, markdownlint-cli2 v0.24.0, actionlint v1.7.12, shellcheck v0.11.0.1-1, uv 0.13.6, pre-commit-hooks v6.0.0, plus about 25 local hooks incl. mypy 3.11, SPDX header, signoff-commit).
8. **DCO.** `DCO` file (v1.1); `README.md:193-199`; check `dco` (`mergify.yml:55-64`); local `signoff-commit` hook appends the trailer from `git config` (`.pre-commit-config.yaml:251-262`). No CLA found.
9. **AI policy** lives in `AGENTS.md`, `README.md:209-231`, the template, and bot text; no separate policy page. `README.md:219`: "Always mention when a pull request includes AI-generated code."
10. **Review.** CODEOWNERS: compilation `:5` (@zou3519 @youkaichao @ProExpertProg @BoyuanFeng), cudagraph `:6`, model_loader `:15` (@22quinn), entrypoints `:38-55`, v1/engine `:114` (@njhill), v1/core `:106`; no owner for `docs/`. Committers merge; ping after 7 days (`README.md:339`); stale after 90 days, close 30 days later, exempt `keep-open` and drafts (`stale.yml:31-45`); max 6 open PRs (`README.md:392-397`).
11. **Future items.** #20098 RFC closed stale 2025-12-26; #23184 closed unmerged 2026-03-02 (stale, needs-rebase); #57899 OPEN, `needs-rebase`, 20 files +2,102, no human maintainer comment, and it covers sleep/wake graph discard, not general idle capture. RFC rule: `README.md:312-317` and `750-RFC.yml` (feedback period usually a week). Loader hook exists: `register_model_loader` (`vllm/model_executor/model_loader/__init__.py:69`); `vllm.general_plugins` is documented for models (`docs/design/plugin_system.md:42-48`), not explicitly for loaders.
12. **Traps.** Label gate, no automatic CI, zero-behind rule, SPDX header, test wiring (ev. 12).

## Claims checked

Claim: "vLLM requires stating AI use in the pull request, trailers on commits, a DCO sign-off and duplicate-PR checks, and bans violators."
- AI use stated in PR: CONFIRMED (`AGENTS.md:33`, `README.md:219`, template `:25`).
- Trailers on commits: CONFIRMED for AI-assisted commits; any attribution trailer is accepted ("such as", `AGENTS.md:137`); 209/300 comply.
- DCO sign-off: CONFIRMED (`README.md:195-196`; 295/300; `dco` check).
- Duplicate-PR checks: PARTLY. Required by AGENTS.md for AI-assisted work and the PR must say why it is not a duplicate (`:8-19, :30`); the human guide only says search issues (`README.md:182`); no workflow enforces it.
- Bans violators: PARTLY. Stated as "can result" (`AGENTS.md:4`; bot comment); no automated ban found; the code of conduct has its own ban ladder.
- Nothing REFUTED. The rules for AI apply to AI-assisted contributions, not all contributions.

## Export rules for this repository

1. Branch: push to a personal fork; use `fix/<topic>` or `perf/<topic>` kebab-case (observed practice, not a rule). Never push to `vllm-project/vllm`.
2. Author identity: a real name and an email the human accepts as public; the DCO records it indefinitely (`DCO` clause d). Set `git config user.name/email` before the first commit.
3. Run the duplicate check at export time, by the human or script, and paste the result: `gh issue view <N> --repo vllm-project/vllm --comments`; `gh pr list --repo vllm-project/vllm --state open --search "<N> in:body"`; `gh pr list --repo vllm-project/vllm --state open --search "<keywords>"`. Stop if an open PR covers it.
4. Commit: subject `[Type][Scope] Description`; body empty or short; trailers `Signed-off-by: Human Name <email>` (use `git commit -s`; the human signs, never the agent) and `Co-authored-by: <model name> <noreply@...>` (match the model that did the work, e.g. `Claude Sonnet 5.5 <noreply@anthropic.com>`). Keep the trailers when squashing.
5. Checks, from the repo root in a `uv` venv: `uv pip install -r requirements/lint.txt`; `pre-commit install`; `pre-commit run --all-files`; the touched tests with `.venv/bin/python -m pytest <path> -v`; `uv pip install -r requirements/docs.txt` and `API_AUTONAV_EXCLUDE=vllm mkdocs serve` if docs change. Add an SPDX header to new files; wire new tests into `.buildkite/test_areas/*.yaml`.
6. PR title: `[Type][Scope] Description` (regex `^(\[[^][]+\])+ +[^[:space:]].*$`).
7. PR body: copy `.github/PULL_REQUEST_TEMPLATE.md` (`gh pr create --template`), keep its sections `## Overview`, `## Claims`, `## Validation`, `## Details` and the checklist; tick the pr-checklist and AI boxes; include the non-duplication reason, test commands with results, and model evals if output changes.
8. Disclosure sentence: "AI assistance (Claude Code) was used to prepare this change. I reviewed every changed line and the test results." Only true if the human did review every line.
9. Do not request the `ready` label (agents must not, `pre-commit.yml:44`); allow maintainer edits; keep at most 6 open PRs.

## Silent or ambiguous

- Branch naming; PR size cap other than the 500 LOC RFC rule; whether a squash keeps a non-AI `Co-authored-by`.
- Which AI-trailer name string is preferred (22 variants observed); whether `Co-authored-by` can name a private-workspace agent.
- Whether a loader plugin should be proposed upstream or kept out-of-tree via `vllm.general_plugins`; ask in `#pr-reviews` or an RFC.
- Whether a design note for idle-time CUDA-graph capture should be a new RFC or a comment on #57899 (closed #20098 was marked stale).
- Real identity versus a pseudonym for the DCO sign-off; the human must decide.
- Branch protection and the `dco` app settings are not in the repo.

## Commands used

```bash
S=/private/tmp/claude-501/-Users-koiios--claude-spares-spare-uA32z4/bf8c79e0-2d81-4d6b-8a4d-1f23dcb48166/scratchpad/upstream-scratch
git clone --filter=blob:none --no-checkout --depth 400 --single-branch https://github.com/vllm-project/vllm.git
git log -1 origin/main; git symbolic-ref refs/remotes/origin/HEAD; git ls-tree -r --name-only origin/main
git show origin/main:AGENTS.md | cat -n   # likewise PR template, docs/contributing/README.md, CODEOWNERS, workflows
diff <(git show origin/main:<path>) <vendor clone>/<path>
git log -n300 origin/main --format='%H%x00%B%x01'     # parsed with python3 -I for tags and trailers
git log -n150 --no-merges --shortstat --format='C%h' origin/main
git ls-remote --tags origin
git grep -nI '<pattern>' origin/main -- <paths>
curl -s https://api.github.com/rate_limit   # then 5 API requests in total, raw JSON saved under $S:
#  pulls?state=closed&per_page=50&sort=updated&direction=desc ; pulls/57899 ; pulls/23184 ; issues/20098 ; issues/57899/comments
```
