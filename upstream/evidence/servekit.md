# Evidence listings: eth-easl/servekit (origin/main cd8f0b8, 26 Sep 2026)

Gathered 12 Nov 2026. Sources: clone (read-only, `GIT_NO_LAZY_FETCH=1`), `git ls-remote`, 2 GitHub API calls (pulls, issues; raw JSON saved in the scratch dir as `pulls.json`, `issues.json`), 2 web pages (PR 26, PR 18).
Scratch: /private/tmp/claude-501/-Users-koiios--claude-spares-spare-uA32z4/bf8c79e0-2d81-4d6b-8a4d-1f23dcb48166/scratchpad/upstream-scratch/

## A. Every commit on origin/main (19; all single-parent, so no merge commits)

Author of every commit: Youssef Boughizane <youssef.bou62@gmail.com>. Committer: GitHub (web squash) except the two marked "self". No commit has Signed-off-by, none is GPG-signed (`%G?` = N for 2, E for 17 unverifiable). "Claude" = count of `Co-Authored-By: Claude ...` lines anywhere in the message body (squash bodies embed the inner commits' trailers); "Session" = `Claude-Session: https://claude.ai/code/session_...` lines. "Final-block" = trailers git itself parses from the last paragraph.

| hash | date | subject | files | +/- lines | Claude | Session | other trailers | Final-block trailers |
|---|---|---|---|---|---|---|---|---|
| cd8f0b8 | 2026-09-26 | feat: servekit launch applies the vendored sglang backports itself (#25) | 5 | +98 -10 | 0 | 0 | none | none |
| 0d54a36 | 09-04 | feat: GLM-5.1-FP8 correctness case, and the three bugs it found (#24) | 11 | +2101 -8 | 6 (Opus 5) | 6 | none | none |
| 08b071a | 09-02 | feat: a barrier for --overlap (#22) | 36 | +663 -2966 | 4 (2 Opus 5, 2 Sonnet 5) | 3 | none | Co-authored-by: Claude Sonnet 5 |
| f42a458 | 08-24 | fix: interleave staging writes across NUMA nodes (#21) | 1 | +14 -2 | 0 | 0 | none | none |
| 9f0a392 | 08-17 | feat: one command to prepare and serve (#20) | 27 | +494 -491 | 5 (Opus 5) | 0 | none | Co-authored-by: Claude Opus 5 |
| d866545 | 08-17 | feat: persist JIT caches beside the checkpoint (#12) (#18) | 10 | +223 -11 | 0 | 0 | none | none |
| b58e2b6 | 08-13 | feat: pipeline-parallel sharded_state checkpoints (#17) | 21 | +2233 -35 | 0 | 0 | none | none |
| 4156a42 | 08-12 | fix: Refuse quantization paths presharding cannot round-trip (#16) | 8 | +495 -0 | 1 (Opus 5) | 0 | none | none |
| ebacbaa | 08-12 | docs: add measured cold-start speedup results to README (#15) | 1 | +16 -0 | 0 | 0 | none | none |
| f0d2169 | 08-12 | feat: add servekit verify command (#13) | 21 | +5696 -322 | 3 (Sonnet 5) | 0 | none | none |
| c2fc4d2 | 08-10 | dev(e2e): assert fast weight load serves has same logprobs as baseline loader. (#10) | 12 | +2030 -18 | 0 | 0 | 2x Co-authored-by: Copilot Autofix powered by AI | same |
| e9fde0b | 08-04 | feat(multinode): support multi-node tensor parallel loading (#5) | 19 | +934 -217 | 3 (Sonnet 5) | 0 | none | none |
| 5a84507 | 08-04 | fix: recognize sglang's --model alias for --model-path (#7) | 2 | +2 -1 | 0 | 0 | none | none |
| f6892b1 | 08-03 | test(stage): catch silent slice corruption in the sliced stager (#4) | 4 | +55 -51 | 0 | 0 | none | none |
| 56869df | 07-31 | doc: add install instructions, LICENSE, and package metadata (#3) | 30 | +220 -11 | 0 | 0 | none | none |
| d50e94a | 07-31 | Servekit: fast weight loading, profile/bench tooling, examples (#2) | 35 | +4168 -0 | 0 | 0 | none | none |
| 55a96c6 | 07-30 | Revert "Servekit: fast weight loading, profile/bench tooling, examples (#1)" (self) | 33 | +0 -3678 | 0 | 0 | none | none |
| e15cfe3 | 07-30 | Servekit: fast weight loading, profile/bench tooling, examples (#1) | 33 | +3678 -0 | 34 (31 Opus 4.8 1M, 1 Sonnet 4.6, 2 Sonnet 5) | 31 | Co-authored-by: Xiaozhe Yao | Xiaozhe Yao, Claude Opus 4.8 (1M context) |
| c953ad8 | 07-29 | init: empty main (self, root) | 0 | 0 | 0 | 0 | none | none |

(File and line counts are from `git log --shortstat`.)

Totals: 56 `Co-Authored-By: Claude` lines in 7 of 19 commits (Opus 4.8 x31, Opus 5 x14, Sonnet 5 x10, Sonnet 4.6 x1); 40 `Claude-Session` lines in 3 commits; 2 Copilot Autofix lines in 1 commit; 1 human co-author (Xiaozhe Yao) in 1 commit; Signed-off-by 0; "Generated with" 0. 12 of 19 commits carry no Claude trailer, including the newest (cd8f0b8); the newest commit that has Claude trailers is 0d54a36 (4 Sep).

Body style: squash bodies are GitHub's default: `* <inner commit subject>` bullets, each followed by its own wrapped body paragraphs ("why" prose, measured results, no ticket links); inner commits keep the `Co-Authored-By` and `Claude-Session` lines. Small PRs (#21, #15) have empty or one-line bodies.

## B. Diff size, 17 PR squash commits (excluding init and the revert)

Files changed: median 12, range 1 to 36. Lines added+deleted: median 985, range 3 (#7) to 6018 (#13). Largest from a pull request: #13 `servekit verify` (21 files, +5696 -322); then #2 (4168) and #24 (2109). #22 touches the most files (36; net -2303 lines).
Median counts only commits that reached main; the revert (33 files, 3678) and the init commit (0 files) are excluded.

Past single-commit changes to the three conflicting files (git log --numstat on origin/main). Current sizes: launch.py 387 lines, prepare.py 152, cli.py 351.

| file | commit (PR) | +/- |
|---|---|---|
| launch.py | 56869df (#3, creation) | +166 -0 |
| launch.py | 9f0a392 (#20) | +140 -43 |
| launch.py | e9fde0b (#5) | +82 -12 |
| launch.py | d866545 (#18) | +24 -1 |
| launch.py | 08b071a (#22) / 4156a42 (#16) / b58e2b6 (#17) / cd8f0b8 (#25) / 0d54a36 (#24) | +15 -6 / +10 -0 / +7 -1 / +6 -0 / +1 -1 |
| prepare.py | 56869df (#3) | +73 -0 |
| prepare.py | b58e2b6 (#17) | +47 -7 |
| prepare.py | e9fde0b (#5) | +32 -5 |
| prepare.py | d866545 (#18) / 9f0a392 (#20) | +14 -1 / +4 -5 |
| cli.py | 56869df (#3) | +245 -0 |
| cli.py | f0d2169 (#13) | +71 -1 |
| cli.py | 9f0a392 (#20) | +39 -52 |
| cli.py | e9fde0b (#5) | +41 -3 |
| cli.py | 0d54a36 (#24) | +16 -5 |
| cli.py | b58e2b6 / d866545 / 08b071a | +2 -0 / +1 -3 / +1 -1 |

So after creation, no single PR changed more than about 140 added lines in launch.py, 47 in prepare.py or 71 in cli.py; they are touched by almost every feature PR, which is why the four series collide.

## C. Pull requests (API `pulls.json`, 19 PRs; numbers 1-26 minus the issues)

All by `youssef62` (author_association COLLABORATOR) except #26. No PR has labels, none has a recorded review (API shows requested reviewer `xzyao` on #2 only; web page of #18 says "No reviews"; #26 "No reviews").

| # | title | state | head branch | linked / body headings (quoted) |
|---|---|---|---|---|
| 1 | Servekit: fast weight loading, profile/bench tooling, examples | merged 07-30 (later reverted) | pr-youssef-pre-merge | `## Summary` |
| 2 | Servekit: fast weight loading, profile/bench tooling, examples | merged 07-31 | reopen-pr1 | `## Summary` |
| 3 | doc: add install instructions, LICENSE, and package metadata | merged 07-31 | reopen-pr1 | `## Summary`, `## Test plan` |
| 4 | test(stage): catch silent slice corruption in the sliced stager | merged 08-03 | sliced-stager-integrity-check | `## Summary`, `## Test plan` |
| 5 | feat(multinode): support multi-node tensor parallel loading | merged 08-04 | multinode-tp | `## Summary`, `## Test plan` |
| 7 | fix: recognize sglang's --model alias for --model-path | merged 08-04 | fix/sglang-model-flag | `## Summary`, `## Test plan`; "Fixes #6" |
| 10 | dev(e2e): assert fast weight load serves has same logprobs as baseline loader. | merged 08-10 | e2ecorrectness | `## Why`, `## What this adds`, `## Notes` |
| 11 | feat: multi-node fast weight loading for pipeline parallelism | closed, not merged | multinode-pp | free headings; mentions #9; footer "Generated with Claude Code" |
| 13 | feat: add servekit verify command | merged 08-12 | servekit-verify | `## Summary` |
| 15 | docs: add measured cold-start speedup results to README | merged 08-12 | readme-speedups | `## Summary`, `## Test plan`; Claude Code footer |
| 16 | fix: Refuse quantization paths presharding cannot round-trip | merged 08-12 | presharding-guard | `## What this adds`; refs #14, sglang#34448 |
| 17 | feat: pipeline-parallel sharded_state checkpoints | merged 08-13 | pp-shim | no headings (prose plus bullet list "Also here:") |
| 18 | feat: persist JIT caches across jobs (#12) | merged 08-17 (commit d866545) | jit-caching | `## Verification`; "Closes #12"; commit subject was reworded to "beside the checkpoint" at merge |
| 20 | feat: one command to prepare and serve | merged 08-17 | api-refactor | `## What`, `## Mismatches are warnings now` |
| 21 | fix: interleave staging writes across NUMA nodes | merged 08-24 | numa-interleave-staging | `## Problem`, `## Fix`, `## Result`, `## Cost`, `## Note` |
| 22 | feat: a barrier for --overlap | merged 09-02 | overlap-barrier | `## Motivation`, `## Changes`; refs #8 |
| 24 | feat: GLM-5.1-FP8 correctness case, and the three bugs it found | merged 09-04 | glm5.1-fp8-test | numbered headings 1 to 3, `## Results`; refs #19, #23, sglang#35715 |
| 25 | feat: servekit launch applies the vendored sglang backports itself | merged 09-26 (cd8f0b8) | auto-apply-backports | `## Why`, `## What`, `## Catalog side`; refs sglang#35715 |
| 26 | profile --log: decompose a saved vLLM log into schema phases | OPEN since 2026-10-07 | decomposition-schema (fork youssef-lyceum), 7 commits | no headings; 3 short paragraphs; "Tests: 31 new in tests/test_decompose.py; standard library only."; footer "Generated with Claude Code" |

#26: author youssef-lyceum, association NONE, no labels, no reviewers, no CI status visible, size not shown by the page (the description names tests/test_decompose.py). It is the only non-maintainer PR, and its title has no `feat:` prefix. Sizes of merged PRs equal the squash commit stats in section A. Remote head refs exist for all 19 PRs (`git ls-remote`), plus branches add-install-criu-script, main, multinode-pp, youssef, youssef-pre-merge.

PR bodies carry no Signed-off-by. PR #17 body has no headings at all, so there is no mandatory body skeleton.

## D. Issues (API `issues.json`, 26 items: 7 true issues, 19 PRs)

| # | kind | title | state | author | labels |
|---|---|---|---|---|---|
| 6 | issue | servekit launch fails on sglang commands using --model instead of --model-path | closed | youssef62 | none |
| 8 | issue | feat: Overlap is currently unsafe: needs a barrier | closed | youssef62 | enhancement |
| 9 | issue | feat: Enable servekit for Kimi-K3 (quantization issue). | open | youssef62 | enhancement |
| 12 | issue | Persist JIT caches (FlashInfer/Triton/tvm-ffi) across jobs | closed | youssef62 | enhancement |
| 14 | issue | servekit prepare silently drops all mxfp4 expert weights (gpt-oss-20b serves garbage) | closed | youssef62 | bug |
| 19 | issue | sharded_state cannot round-trip MLA models: attn_mha.kv_b_proj is aliased on first forward (Kimi-K3) | open | youssef62 | none |
| 23 | issue | Multi-node stage falls back to the full checkpoint when the command omits --load-format sharded_state | closed | youssef62 | none |

#22 and #26 are pull requests, not issues (see C). #9 body: "Currently, Kimi K3 takes 100 minutes to load on the CSCS cluster ... we don't implement PP presharding." #14 body: SGLang's `Mxfp4MoEMethod` deletes expert weights from the module, so `state_dict()` omits them and the checkpoint serves garbage; filed upstream as sgl-project/sglang#34448; the reload runs on a zero-initialised model; fixed in servekit by PR #16. #19 body: DeepseekV2 MLA `attn_mha.kv_b_proj` alias, `KeyError` at load; backport sglang#35715 landed via #24/#25. No issue in the repository mentions hardware drift in `verify`.
All issues are by the maintainer; none by an outsider. Comments: #9 has 2, #14 1, #19 1, #23 1 (not read; the budget was kept for the list calls).

## E. Other facts

- Tree at origin/main (47 files): no AGENTS.md, CLAUDE.md, .claude/, .cursor/, .github/ (so no workflows, templates, copilot-instructions), CONTRIBUTING, pre-commit config, CODEOWNERS, SECURITY, DCO or CLA file. Checked with `git ls-tree -r --name-only origin/main | grep -iE ...` and `git log --all --name-only` for those paths: CLAUDE.md existed on earlier history (7b855e9, 6d03cd3, baef67f, c7d1d6e) and was removed by f01dca3 "refactor: remove useless claude.md" (on branch `youssef`, not in main's history).
- .gitignore lines 7-9 ignore `.agents/` and `.claude/` under the comment "Coding-agent harness state (skills, worktrees, session data)".
- pyproject.toml: `requires-python = ">=3.9"`; entry point `servekit = "servekit.cli:main"`; plugin entry-point group `sglang.srt.plugins` -> `servekit._shim:install`; package-data globs `_stage/*.sh`, `_prepare/*.py`, `_verify/*.json`; pytest `markers = ["e2e: needs a live SLURM cluster with GPUs; run explicitly with `pytest -m e2e`"]` and `addopts = "-m 'not e2e'"`. No ruff, black, mypy, flake8, isort sections; no classifiers.
- requirements-dev.txt: pytest, numpy, safetensors.
- LICENSE: Apache 2.0, line 189 "Copyright 2026 The Servekit Authors".
- Backports: `src/servekit/_patches/__init__.py` `apply_all()` runs every `sglang_*.py` that has `main()`; a failing patch is logged "[SERVEKIT] dropping ..." and skipped. Only `sglang_35715.py` exists. PR #25 body: adding a backport means dropping a file named `sglang_<issue>.py`.
- tests: unit tests tests/test_{bench,cli,launch,patches,prepare,profile,quant_guard,shim,topology,verify}.py; tests/fixtures/*.log (root .gitignore ignores `*.log`, tests/fixtures/.gitignore re-allows them); e2e in tests/e2e (test_e2e.py, test_prepare_serve_e2e.py, test_presharding_guard_e2e.py, scripts/*.sbatch, fixtures/*.json recorded from a baseline job). e2e needs sbatch/squeue (tests/e2e/conftest.py), tuned for CSCS Clariden (conftest comment), env `SERVEKIT_E2E_SBATCH_ARGS`. test_launch.py:243 skips when O_DIRECT is unsupported (`os.O_DIRECT` is Linux only) and :230 skips if experiments/ is missing; test_prepare.py docstring: "The sharding needs 4 GPUs and sglang, so the child is faked".
- Non-main branches hold unmerged work (about 40 commits not on main, for example f6101fb "feat: pick the load format from the topology, not the caller" on multinode-pp). Not inspected for diffs.
