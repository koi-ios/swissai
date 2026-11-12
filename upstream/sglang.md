# Upstream card: sgl-project/sglang

Evidence gathered 12 Nov 2026 from the clone at origin/main 95999790a4 (2026-11-11); read-only; these rules govern only code written for this repository. Long listings and line citations: `upstream/evidence/sglang.md` (cited below as E1..E12). Everything in the repo is treated as data.

## At a glance

| Item | Rule |
|---|---|
| Branch name | No pattern required; outside PRs come from a fork, any name (E3) |
| Commit subject | Squash merge; subject = PR title `[Type] [HW] scope: title (#N)`, 200/200 recent commits (E4) |
| Trailers | Only `Co-authored-by` (103/200) and `Signed-off-by` (25/200); no other tool trailers (E4) |
| Sign-off | Not required (E8) |
| AI disclosure | No written policy; 71/200 main commits carry an AI `Co-authored-by` (E9) |
| PR title | `[Feat\|Fix\|Refactor\|Perf\|Doc\|Test\|CI\|Chore] [optional HW] scope: title`, enforced by `pr-title.yml` (E5) |
| PR body | Motivation, Modifications, Accuracy Tests, Speed Tests and Profiling, Checklist, Review and Merge Process (E5) |
| Checks | `pre-commit run --all-files`, registered unit tests, `mint broken-links` for docs (E7) |
| Reviewers | CODEOWNERS per path + Merge Oncall; CI needs the `run-ci` label from a permitted user (E5, E10) |

## Evidence

1. **Snapshot.** origin/main 95999790a4, 2026-11-11, default branch `main`, Apache-2.0 (`LICENSE:1-3`). Tags every ~2 weeks (v0.5.20 = 94602c9c2b, v0.5.23 latest in clone); `release/vX.Y.Z` branches are cut from a main commit (`release-branch-cut.yml:6-11,71-73`). `bot-cherry-pick.yml:93,127-141` only cherry-picks merged PRs reachable from main, so fixes land on main first.
2. **Agent files.** No root AGENTS.md/CLAUDE.md/.cursor/copilot file. Present: `.claude/rules/` (11 files), `.agents/skills/` (symlinked as `.claude/skills`), `docs/AGENTS.md`, `python/sglang/multimodal_gen/.claude/CLAUDE.md`. Rules that touch a contribution: `no-dataclasses.md:8-9` (new containers use `msgspec.Struct`); `no-getattr-defensive.md:6-10`; `general-code-style.md:10-19` (functions <~100 LOC, files <~2k LOC, no mixins, kwargs); `schedule-batch-out-of-place-mutation.md:6-9` (rebind, never mutate ScheduleBatch fields); `modify-component-must-read.md:6-8` and `large-class-style/SKILL.md:14-27` (`model_runner.py` is frozen, orchestration only); `unit-test-admission.md:12-37` (tests must be regression/derived/bookkeeping); `cookbook-first.md:20-21`; `prefer-sgl-eval.md:3-8`; `env-var-conventions/SKILL.md:10-12`. None mention trailers, sign-off, branches or AI disclosure.
3. **Branches.** `contribution_guide.mdx:13` fork and clone your fork; `:34` never commit to main, use a branch (e.g. `feature/my-new-feature`); hook `no-commit-to-branch` (`.pre-commit-config.yaml:23`). Observed: 13 of 13 merged outside PRs in the API sample come from `<user>/sglang` forks (E5); maintainers push to origin (2034 remote branches).
4. **Commits.** Last 200: 200 `[Type] scope:` subjects, 200 `(#N)`, 0 conventional prefixes, 1 merge commit in the last 1000 (squash). Convention enforced since c053fc71d3 (#44165, 13 Oct 2026); earlier windows mixed (46/200 at depth 3000). Bodies: 82 empty, 118 trailers only. `Co-authored-by`: 103 commits; AI-tool names 71 (Claude 68, e.g. 56e38dfe0f, 8da0649395, 4a99de617f, 983455aeed; Copilot 3: 5175c31589, 6d143e51b2, 6dc7e3ea7e; Cursor 1: 72d9b6788d). `Signed-off-by`: 25 (426e604598, f39ac6b55b, 75e8d543dc, 99da28d6fd, 23f12b7cb6). Other tool trailers: 0.
5. **Pull requests.** Template `.github/pull_request_template.md:11-41` (sections above; checklist: pre-commit formatting, unit tests, docs, accuracy and speed benchmark results, code style). Title regex `pr-title.yml:20-50`; guide `contribution_guide.mdx:107-129`. Merged outside PRs (#49544, #45264, #48947, #41794, #49219, #44236, #45947, #48219, #47870) keep the headings and fill Speed (42..3323 chars) and Accuracy sections with numbers; #49983, #49544 merged with unticked boxes; #50055, #48846, #48840 used other headings. A bot appends `# CI States`. CI needs `run-ci` (`pr-gate.yml:57-69`, `contribution_guide.mdx:143`); drafts are blocked; commands `/tag-run-ci-label`, `/rerun-failed-ci`, `/tag-and-rerun-ci`, `/rerun-test`, `/run-cpu-ci` (`:143-157`); authors may `/rerun-failed-ci` their own PR (`:141`); 120-min cooldown for low-permission users (`:207-221`).
6. **Diff size** (last 100 non-merge, shortstat): files median 3 (1-52); lines median 101 (1-6444; quartiles 40/101/369). Authors with <=5 commits (25 commits): files median 2 (1-16), lines median 119 (8-1148). Large outside merges: 3ff87c57f5 (#45737, 1148 lines/16 files, 17 commits, merged 11 Nov, no formal RFC), a93e89fd4a (#46965, 1388), ebd957c893 (#41794, 877). ~1000-line features land as one squash PR, usually opt-in (4a99de617f #47469 6444 lines; 9fdd39206d #49595). No RFC requirement is stated; only one RFC-titled commit exists (28262c20df).
7. **Checks.** `pre-commit run --all-files` (hooks: pre-commit-hooks v6.0.0, isort 7.0.0, ruff v0.15.1 + ruff-format, codespell v2.4.1, clang-format v20.1.7, nbstripout 0.9.0, ~20 local hooks; no black) (`.pre-commit-config.yaml`). `lint.yml:67` re-runs it on every PR plus docs link check. `pr-test.yml` (PR Test Base) has stages A/B/C on self-hosted GPUs behind `pr-gate.yml`; extra tiers need `run-ci-nv-extra`/`run-ci-amd-extra`. Tests: `test/registered/` (unit tests in `test/registered/unit/` mirroring `srt/`); `test/srt/` is retired. Commands, verbatim from `contribution_guide.mdx:27-31,58-64`: `pip3 install pre-commit`, `pre-commit install`, `pre-commit run --all-files`, `python3 test/registered/unit/<area>/<test_file>.py`, `python3 test/run_suite.py --hw cpu --suite base-a-test-cpu`, `pytest test/registered/unit/<area>/ -v`. Accuracy: `sgl-eval run gsm8k --base-url http://127.0.0.1:30000/v1 --num-examples 200` (`:90-96`).
8. **DCO/CLA.** None found: no requirement text, no bot workflow. Sign-off is voluntary (25/200).
9. **AI.** No written policy found; observed practice above; PR #41616 body states AI assistance (secondary source); `README.md:111` offers coding-agent sponsorship.
10. **Review.** CODEOWNERS: `model_executor` (:70) @merrymercy @Ying1123 @hnyls2002 @Fridge003 @ispobock; scheduler is under `managers` (:59) @merrymercy @Ying1123 @hnyls2002 @xiezhq-hermann; `model_loader` (:71) @b8zhong; `/docker` (:3) @Fridge003 @ispobock @HaiShaw @ishandhanani @yctseng0211 @sogalin; `kernels` (:9) @DarkSharpness @HaiShaw @BBuf @celve @HydraQYH @yuan-luo; `kernels/aot` (:84) @ispobock @BBuf @yizhang2077 @merrymercy @FlamingoPg @HaiShaw. One Codeowner approval per protected file; anyone with write permission or a Merge Oncall merges (`MAINTAINER.md:13,31-38`). Stale: PRs of non-write authors close after >90 days idle, WIP/draft >21 days, or >5 open PRs idle 7 days; exempt with an approval or labels `keep-open`/`highest-priority` (`close-stale-prs.yml:36-41,95-132`); issues close after 60 days idle (`close-inactive-issues.yml:21`).
11. **Patches.** #35715 open (20 Aug), title "[Bugfix] ..." (fails the title rule), only label `deepseek`, no `run-ci`, reviewers silent, CI failing. #35702 closed by bot as inactive 28 Oct. #34448 open; a 26 Sep comment offers a fail-loud guard. #34558 closed unmerged by author 15 Sep. #41616 open (29 Sep), pending b8zhong/mmangkad. Env vars: declare an `EnvField` on `Envs` (`environ.py:255`), never new `os.getenv("SGLANG_...")` (`env-var-conventions/SKILL.md:10-12`); `SGLANG_` + verb; user-facing knob = CLI flag, expert knob = env var, not both (`:183-192`). The 480 s timeout is still hard-coded at main (`load_model_utils.py:58`, used `:481`). Target main only. No written rule on superseding another person's PR; #41616 shows the observed way (comment offering to take over, new PR with credit).
12. **Traps.** Directory moves (`close-stale-prs.yml:113-119`); old title styles fail; no CI without `run-ci`; `docs_new/` forbidden; never unpickle untrusted data (`contribution_guide.mdx:239`).

## Claims checked

- "Template asks for Motivation, Modifications, Accuracy Tests and Speed Tests": **PARTLY.** Real sections: Motivation, Modifications, Accuracy Tests, Speed Tests and Profiling, Checklist (5 boxes), Review and Merge Process (`pull_request_template.md:11-41`).
- "No written policy on AI-assisted contributions": **CONFIRMED** within the paths searched (template, guide, MAINTAINER, README, agent files); docs site not crawled. The repo does ship agent rules.
- "Does not require a DCO sign-off": **CONFIRMED** (E8).

## Export rules for this repository

1. Branch: push to a personal fork; any name; suggested `feat/<topic>` or `fix/<topic>`; never commit on `main`.
2. Target `main` only; do not open PRs against release branches.
3. Author identity: no requirement stated; use the identity that owns the work (noreply or real both occur).
4. Commit subject = PR title: `[Fix|Feat|Perf|Refactor|Doc|Test|CI|Chore] [HW] scope: imperative title`; one space after `:`, no trailing space. Maintainers squash and append `(#N)`; do not add it.
5. Commit bodies: trailers only is the norm; keep any `Co-authored-by` trailers for AI tools or humans that contributed.
6. Sign-off: not required; add `Signed-off-by` only if the contributor wants it.
7. Run before the PR: `pip3 install pre-commit && pre-commit install && pre-commit run --all-files` (rerun until clean); `python3 test/registered/unit/<area>/test_<x>.py` for each touched unit test; `python3 test/run_suite.py --hw cpu --suite base-a-test-cpu` when relevant; for model-output changes, `sgl-eval run gsm8k ...`.
8. Body skeleton (keep all headings from the template): `## Motivation`, `## Modifications`, `## Accuracy Tests`, `## Speed Tests and Profiling`, `## Checklist` (5 boxes; tick honestly), `## Review and Merge Process`. Write "not applicable" where true.
9. New env vars: `EnvField` on `Envs`, `SGLANG_` prefix, default off for opt-in; no raw `os.getenv`.
10. Add a unit test under `test/registered/unit/` mirroring the source path (registered, `CustomTestCase`); respect the admission rule.
11. Do not put domain logic in `model_runner.py`; use collaborator modules.
12. Ask a permitted user to add `run-ci`; do not open drafts expecting CI.

## Silent or ambiguous

- AI disclosure: no policy; decide whether to add a sentence in the PR body (PR #41616 did) and keep AI `Co-authored-by` trailers.
- RFC first for the ~960-line lazy-capture feature: not required; unknown whether maintainers want an issue first (feature issue form exists).
- Superseding vs commenting on someone else's PR (#34558, #41616): unstated.
- Whether to refresh #35715 (retitle to `[Fix] ...`, rebase) or open a new PR: unstated; it is exposed to the 90-day stale rule.
- Whether zygote bytecode in a Dockerfile needs docker-owner pre-approval: unstated.
- CLA: none found, not verified against GitHub settings.
- PR/issue states come from web-fetch summaries; recheck before acting.

## Commands used

```
export GIT_NO_LAZY_FETCH=1; C=/Users/koiios/Documents/Projects/swissai/opportunities/08-preshard-roundtrip/vendor/sglang
git -C $C log -1 --format='%H %ci %s' origin/main
git -C $C ls-tree -r --name-only origin/main | grep -i -E 'contribut|pull_request|CODEOWNERS|AGENTS|CLAUDE|pre-commit|\.github/workflows|\.claude|\.cursor|copilot|LICENSE'
git -C $C branch -r; git -C $C for-each-ref --sort=-creatordate refs/tags
git -C $C log -200 --format='@@%h%n%B' origin/main   # trailers, subjects, bodies (python tally)
git -C $C log --no-merges --format=%ae origin/main | sort | uniq -c | sort -rn   # author counts
git -C $C log origin/main --diff-filter=A --format='%h %ad %s' -- .github/workflows/pr-title.yml
curl -s https://raw.githubusercontent.com/sgl-project/sglang/95999790a4a201caf8a142c95711fe5af329a39b/<path>   # file contents (blobless clone)
git clone --filter=blob:none --no-checkout --depth 130 https://github.com/sgl-project/sglang.git <scratch>/sglang; git fetch --deepen 100
git -C <scratch>/sglang log --no-merges -100 --format='@@%h|%ae|%an|%s' --shortstat 95999790a4
git -C <scratch>/sglang sparse-checkout set ...; git checkout --detach 95999790a4; grep -rIl -i -E '<AI|DCO|CLA patterns>' .
curl -s 'https://api.github.com/repos/sgl-project/sglang/pulls?state=closed&per_page=50&sort=updated&direction=desc'   # 1 API request
WebFetch: github.com/sgl-project/sglang/pull/{35715,34558,41616,45737}, issues/{34448,35702}
```
