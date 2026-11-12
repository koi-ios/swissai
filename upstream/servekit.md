# Upstream card: eth-easl/servekit

Evidence gathered 12 Nov 2026 from the clone at origin/main cd8f0b8; read-only; these rules govern only code written for this repository. Long listings (per-commit table, every PR and issue): `evidence/servekit.md` (cited below as E-A to E-E).

## At a glance

| Item | Rule | Basis |
|---|---|---|
| Branch name | short kebab-case topic, e.g. `jit-caching`; optional `fix/` prefix | observed (18 of 19 PR heads have no prefix; `fix/sglang-model-flag` once) |
| Commit subject | `type: lowercase summary`, optional `(scope)`, `(#N)` added by squash | observed |
| Trailers | none required; maintainer commits often carry `Co-Authored-By: Claude <model> <noreply@anthropic.com>` and `Claude-Session:` | observed (7 of 19 commits) |
| Sign-off | not required, never used (0 of 19) | observed; no DCO file |
| AI disclosure | no policy; disclosure is by trailer or a "Generated with Claude Code" PR footer | observed (#11, #15, #26) |
| PR title | same as the future squash subject | observed |
| PR body | `## Summary` + `## Test plan`, or free headings | observed, no template |
| Checks | `PYTHONPATH=src python -m pytest tests -q` | README.md:108 |
| Reviewer | the single maintainer, youssef62; no review recorded on any PR | observed |

## Evidence

1. **Snapshot.** origin/main cd8f0b8, 26 Sep 2026; default branch `main` (`refs/remotes/origin/HEAD`); 19 commits. Apache-2.0 (LICENSE:189, "Copyright 2026 The Servekit Authors"). Layout: `src/servekit/` with cli.py, launch.py, prepare.py, verify.py, profile.py, bench.py, stage.py and others, plus `_patches/`, `_prepare/`, `_shim/`, `_stage/`, `_verify/`; `tests/`, `tests/e2e/`, `examples/`. Python `>=3.9` (pyproject.toml, `requires-python`). Tests: README.md:108 and :114 (E-E).
2. **Agent files.** None at origin/main (`git ls-tree -r` found no CLAUDE.md, AGENTS.md, .claude/, .cursor/, copilot-instructions, skills). History: a CLAUDE.md existed before and was removed by f01dca3 "remove useless claude.md" (on branch `youssef` only). `.gitignore:7-9` ignores `.agents/` and `.claude/`.
3. **Branches.** No stated rule. Remote branches: add-install-criu-script, multinode-pp, youssef, youssef-pre-merge. PR heads (E-C): kebab-case topics (`jit-caching`, `servekit-verify`, `overlap-barrier`); only `fix/sglang-model-flag` has a prefix. The outside PR #26 uses a fork branch `decomposition-schema`.
4. **Commits.** All 19 are authored by Youssef Boughizane; 17 are GitHub squash merges (committer GitHub, one parent, subject ends `(#N)`, 17 of 17); 2 are his own (init, revert 55a96c6). No merge commits, no rebases visible. Subjects: `feat:` 7, `fix:` 3, `doc:`/`docs:` 2, `test(stage):`, `dev(e2e):`, `feat(multinode):`, plus the first PR without a prefix (#1, #2). Lower case after the colon except "fix: Refuse ..." (#16). Bodies are GitHub's squash text: `* <inner subject>` bullets with prose explaining why and measured results. Trailers: 56 `Co-Authored-By: Claude ...` lines in 7 commits (Opus 4.8, Opus 5, Sonnet 5, Sonnet 4.6), 40 `Claude-Session:` lines in 3, 2 Copilot Autofix lines in 1, one human co-author (Xiaozhe Yao) in e15cfe3, `Signed-off-by` 0. Per-commit table: E-A.
5. **PRs.** 19 PRs, 18 by maintainer youssef62 (17 merged, #11 closed unmerged), #26 open by outsider youssef-lyceum. Titles equal the squash subject (exception: #18 reworded at merge). Headings used: `## Summary`/`## Test plan` (#3, #4, #5, #7, #15), `## Why`/`## What`, `## Problem`/`## Fix`, or none (#17, #26). Links: "Fixes #6" (#7), "Closes #12" (#18). Labels: none on any PR. #18 (d866545): +223 -11 in 10 files, "No reviews". #26: 7 commits, no reviews, no labels, size not visible, body "Tests: 31 new in tests/test_decompose.py; standard library only." and a Claude Code footer (E-C).
6. **Diff size.** 17 PR squash commits: median 12 files and 985 lines (added plus deleted); range 1 to 36 files, 3 to 6018 lines. Largest from a PR: #13 verify (21 files, 6018 lines). Series of 2 to 3 commits fit this size (E-B).
7. **Checks.** No CI, lint, format or type-check configuration anywhere (no ruff, black, mypy sections in pyproject.toml; no workflows). Local commands, README.md:108: `PYTHONPATH=src python -m pytest tests -q`; README.md:114: `PYTHONPATH=src python -m pytest tests/e2e -m e2e -q` (needs a live SLURM cluster with GPUs, pyproject marker `e2e`; default `addopts = "-m 'not e2e'"`). requirements-dev.txt: pytest, numpy, safetensors. Unit tests fake GPUs (test_prepare.py docstring); test_launch.py:243 uses `os.O_DIRECT` (Linux).
8. **DCO/CLA.** None found: no file, no mention (`git grep` for DCO, CLA, signed-off found only licence text), 0 sign-offs.
9. **AI rules.** No written policy. Observed: the maintainer's own commits carry AI co-author trailers in 7 of 19 commits, the newest being 0d54a36; cd8f0b8 has none. PR footers "Generated with Claude Code" on #11, #15, #26.
10. **Review/merge.** One active maintainer (youssef62) opens, merges and commits; `xzyao` was a requested reviewer on #2 and co-author of e15cfe3. No review recorded on #18 or #26. First PR #1 was merged, reverted (55a96c6) and reopened as #2.
11. **Series items.** Issues: #6, #8, #12, #14, #23 closed; #9 (Kimi-K3, open), #19 (MLA round trip, open). #22 and #26 are PRs. No issue concerns hardware drift in verify. README.md:103: `- [ ] Support vllm fast weight loading` (unchecked); README.md:67 already shows `servekit profile -- vllm serve`. Sizes: launch.py 387 lines, prepare.py 152, cli.py 351; largest past single-PR edits: launch.py +140 -43 (#20), prepare.py +47 -7 (#17), cli.py +71 -1 (#13) (E-B). Every feature PR touches these files.
12. **Pitfalls.** Backports live in `src/servekit/_patches/sglang_<issue>.py` with a `main()`; `apply_all()` globs them and drops failures with a log line (PR #25). Plugin entry point `sglang.srt.plugins` in pyproject. package-data lists only `_stage/*.sh`, `_prepare/*.py`, `_verify/*.json`. `*.log` is gitignored; log fixtures in tests/fixtures rely on a negation. e2e fixtures are recorded from a cluster baseline job (tests/e2e/test_e2e.py header). `.claude/` is gitignored.

## Claims checked

- "Full of Co-Authored-By: Claude trailers, including maintainer commits": **PARTLY.** 56 lines in 7 of 19 commits, all maintainer commits; but 12 commits have none, including the newest, and the final trailer block of only 4 commits holds any trailer.
- "No contribution guide, PR template or CI": **CONFIRMED** at origin/main (E-E). Org-level defaults on GitHub were not checked.
- "#19 is MLA round trip": **CONFIRMED** (title). "#9 is Kimi-K3 quantization": **PARTLY** (title says "quantization issue"; body says PP presharding is missing and loading takes 100 minutes).
- "#14 is gpt-oss MXFP4 weights loading as zeros": **CONFIRMED** in substance (expert weights silently dropped from the checkpoint, reload zero-initialised; upstream sgl-project/sglang#34448); closed by #16.
- "Hardware drift in verify" issue: **REFUTED** (no such issue among the 7 issues).

## Export rules for this repository

1. Branch: `<topic>` in kebab-case (observed); `fix/<topic>` allowed for bug fixes.
2. Author identity: a real name and email you control; no requirement is stated.
3. Commit subject: `feat:`, `fix:`, `test:`, `docs:` (observed), optional scope; lowercase; no trailing period. Do not add `(#N)`: GitHub adds it on squash.
4. Body: why, then measured results; no ticket links (observed).
5. Trailers: none required. Keep `Co-Authored-By:` for real co-authors; add an AI co-author trailer only if the contributor wants disclosure (observed, not required). Do not add `Claude-Session:` URLs from the private workspace.
6. Sign-off: no.
7. Checks before export: `PYTHONPATH=src python -m pytest tests -q` (README.md:108). The e2e suite is not runnable without a SLURM cluster; say so in the PR.
8. PR title: the intended squash subject. Body: `## Summary` then `## Test plan` (observed, most common), `Fixes #N` or `Closes #N` when an issue exists.
9. Keep each commit near the median (about 12 files, under 1,000 lines); one PR per series, as 2 to 3 commits are squashed anyway (observed).
10. Rebase the three conflicting series on each other in a fixed order before exporting; each PR is squashed to one commit.
11. Backports: add `src/servekit/_patches/sglang_<issue>.py` with `main()`; new package data needs a glob in pyproject.toml.

## Silent or ambiguous

- Whether the maintainer wants a PR per commit or per series; whether outside PRs are reviewed (#26 open since 7 Oct with no review).
- AI-assistance policy and whether trailers are wanted from outside contributors.
- Linter or formatter, Python style, type checks: none configured.
- Whether `servekit launch` changes need cluster e2e evidence.
- Vision for vLLM (roadmap unchecked; #26 is competing work), and whether sglang_* backports belong in this repo long term.
- Issue-first or PR-first (only the maintainer has opened issues).

## Commands used

```
export GIT_NO_LAZY_FETCH=1; C=/Users/koiios/Documents/Projects/swissai/opportunities/01-compile-caches-270/vendor/servekit
git -C $C log -1 --format='%H %ad %an' origin/main
git -C $C branch -r; git -C $C ls-tree -r --name-only origin/main
git -C $C log origin/main --format='%h|%an|%cn|%P|%ad|%s' --date=short --shortstat
git -C $C log -1 --format='%(trailers:only,unfold)' <h>; git -C $C log -1 --format=%B <h> | grep -ic ...
git -C $C show origin/main:{README.md,pyproject.toml,requirements-dev.txt,.gitignore}
git -C $C log origin/main --numstat -- src/servekit/{launch,prepare,cli}.py
git -C $C grep -n -iE 'ruff|DCO|signed-off|CLA|contribut' origin/main
git -C $C log --all --name-only -- '*CLAUDE*' '.github/*' ...
curl -s https://api.github.com/rate_limit        # 56 remaining
curl -s 'https://api.github.com/repos/eth-easl/servekit/pulls?state=all&per_page=50'
curl -s 'https://api.github.com/repos/eth-easl/servekit/issues?state=all&per_page=100'
git ls-remote https://github.com/eth-easl/servekit.git 'refs/pull/*/head' 'refs/heads/*'
web fetch: https://github.com/eth-easl/servekit/pull/26 and /pull/18
```
(3 API requests in total, including the rate-limit call; raw JSON in the scratch directory.)
