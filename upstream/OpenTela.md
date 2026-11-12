# Upstream card: eth-easl/OpenTela

Evidence gathered 12 Nov 2026 from the clone at origin/main 4b6e90f19734d46e107d503c8274f0cd3eb6aaa4 (2026-11-02), plus 7 GitHub API calls; read-only; these rules govern only code written for this repository. Long listings: `upstream/evidence/OpenTela.md`. "Maint." = Xiaozhe Yao (xzyaoi).

## At a glance

| Item | Rule (R = stated in a file, O = observed practice only) |
|---|---|
| Branch name | O: `<type>/<kebab-slug>` e.g. `fix/self-row-sweep` (#65), `fix/worker-connectivity-watchdog` (#67). No stated rule |
| Commit subject | O: `type(scope): summary`, lowercase type, scope often `protocol`; 76 of last 100 |
| Trailers | None required. O: `Co-Authored-By: Claude <Model> <noreply@anthropic.com>` on AI-assisted commits (33 of 100) |
| Sign-off | Not required; 0 of last 100 |
| AI disclosure | No policy. O: trailer in commits, "Generated with Claude Code" footer in PR body (#72, #67, #64, #63, #61, #60) |
| PR title | Same as commit subject; GitHub adds ` (#N)` on squash |
| PR body | `## Problem`, `## Change` (or `## Fix`), `## Testing` |
| Checks | From `src/`: `make test`, `make lint` (or `make check`) |
| Review | Maint. only; merges are squash merges |

## Evidence

1. **Snapshot.** origin/main 4b6e90f, Mon 2 Nov 2026 15:53 +0100, subject `fix(security): resolve Dependabot alerts ... (#70)`. Default branch `main` (`refs/remotes/origin/HEAD`). Apache-2.0 (`LICENSE:1-3`, `README.md:57`). Go module at `src/go.mod` (module `opentela`, `go 1.26.0`, line 3). Also Python (`contrib/`, `tools/`), TypeScript docs site (`docs/`), Terraform/Ansible (`deploy/`). 255 commits.
2. **Agent files** (none says anything about contributions). `CLAUDE.md` (added f88077d, 2025-11-03): build and test commands (`:11-31`), architecture, CI summary (`:97-101`). No AGENTS.md, `.cursor/`, copilot-instructions or skills (`git ls-tree`). `.claude/commands/deploy-test.md` (2c069a2): maintainer's deploy runbook to named hosts; irrelevant to patches, not followed. `.codex/hooks.json`, `.entire/settings.json:2` (enabled), `.pi/extensions/entire/index.ts`: Entire checkpoint hooks (added 61ccdb6); `.codex/hooks.json:9` exits silently when `entire` is not installed. `.gitignore:14` ignores `.claude/` although the file is tracked. No rule on branching, commits, PR text or AI disclosure in any of them.
3. **Branches.** No rule. Merged heads: `fix/self-row-sweep` (#65), `feat/configurable-build-pubkeys` (#64, from fork), `fix/bootstrap-explicit-precedence` (#63), `chore/log-noise` (#61), `feat/peer-metadata` (#60), `fix/dependabot-security-updates` (#70); bare names also occur (`auto-link-opt-in` #69, `ephemeral-identity-datadir` #68); `copilot/sub-pr-N`, `dependabot/...`. Merge commits 63b8421 "Merge feat/bench_overhead into main" and 664450d "Merge branch 'worktree-usage-tracking'".
4. **Commits.** Last 100: 76 conventional subjects (feat 36, fix 17, docs 10; scopes `protocol`, `cli`, `docs`, `benchmark_convergence`); median subject 59 characters, max 98; 68 have a body, written as prose or bullets without a template. 16 end in `(#N)`, 14 of them committed by GitHub (squash merges: 99 of 100 commits have one parent; PR commit lists appear in the squash body, e.g. 5c4bbdb, f0eca38, 4b6e90f). Caution: #1 to #5 on 2026-09 commits (b3d82d9, 61ccdb6, 7e28835, 3ba1806, b5fe983) are not this repository's PR numbers. 80 maintainer commits were pushed directly (committer = author).
   Trailers over last 100 (maint. 94, others 6):

   | Pattern | Maint. | Others | Example hashes |
   |---|---|---|---|
   | `Co-Authored-By` naming Claude | 29 | 4 | 9fac744, 49e39e6, 8d1eb88, f0eca38 / c7a4a96, 5c4bbdb, 119401e, 1111d5d |
   | `Signed-off-by` | 0 | 0 | none (6 in all history: dependabot, 3cffab9, f4d6a98) |
   | `Claude-Session:` URL | 8 | 2 | 9fac744, 49e39e6, ca50dd5 / c7a4a96, 5c4bbdb |
   | `Entire-Checkpoint:` | 5 | 2 | 4b6e90f, 42728f3, 778a8fa, 0f4d3de / c7a4a96, 5c4bbdb |

   Models named: Opus 4.7, 4.8, 5, 5.5; Sonnet 4.6, 5.5; address `noreply@anthropic.com`.
5. **Pull requests.** No template: `.github/` holds only `workflows/ci.yml`, `release.yml`. Titles are the commit subject convention. Recent bodies: #65 (maint.) `## Problem`, `## Root cause`, `## Fix`, `## Evidence`, `## Testing`; #67 (Rob) and #68 (Mel Zuercher) `## Problem`, `## Change`, `## Testing` (#68 adds `## For the maintainers to weigh`); #63, #64 (Aryan Ahadinia) `## Problem`, `## Change`, `## Tests`; #60, #61 `## What`, `## Why`/`## Notes`; #72 `## Problem`, `## Observed`, `## Fix`, `## Tests`. Older maintainer PRs have empty bodies (#69, #54, #51, #49). Outside PRs come from forks. No labels on human PRs; no closing keywords (`Fixes #N`) in any of the 50 PR bodies. Drafts only from bots (#23, #20).
6. **Diff size** (last 50 non-merge commits, lockfiles excluded): files median 4.5 (1 to 151), changed lines median 223 (0 to 11,340; the max is the 80cd709 refactor). Others (6 commits): files median 5.5 (2 to 15), lines median 170 (45 to 1,420). Largest recent outside merge: #67 5c4bbdb, 6 files, +1383/-37 (366-line integration test). All-time by files: Elia Palme #10 2c64ebc, 40 files.
7. **Checks.** `ci.yml`: on push and PR to `main`; job "Go tests" runs `make test VERBOSE=1` in `src` (`:27-28`, Go from `src/go.mod`), then a coverage PR comment skipped for fork PRs (`:45`); job "Lint" runs golangci-lint-action v9, `v2.15`, `--timeout=5m` (`:113-118`). `release.yml`: on `v*` tags, builds and publishes images. `src/Makefile`: `test` (`:168-174`) runs gotestsum 0.4.2 with `-race -coverprofile`, CGO on; `lint` (`:183-185`) golangci v2.15.0 (`:27`); `check` (`:157-159`) both. Config `src/.golangci.yml` (v2, presets comments, common-false-positives, legacy, std-error-handling; no formatters enabled). Integration tests need Docker: `make test TEST_PKGS="./tests/integration/..." GOARGS="-tags=integration -count=1"` (`CLAUDE.md:26-29`), not in CI. `.pre-commit-config.yaml` covers whitespace, YAML, black, nbstripout: no Go hook. No benchmark evidence is required (benchmarks exist: `loadbalancer_bench_test.go`). Go required: `src/go.mod:3` 1.26.0.
8. **DCO/CLA.** No DCO workflow, CONTRIBUTING, or CLA file. `README.md:51-53`: "follow the code of conduct and submit pull requests" (no code-of-conduct file exists).
9. **AI policy.** None stated anywhere. Observed: maint. adds Claude trailers on about a third of commits since 2026-03-08 (first 5ebb961/7356700); outside contributors do the same (#67, #68, #63, #64) and add a Claude Code footer to PR bodies; the maint. merged #67 and #68 (merger of the others not checked).
10. **Review.** No CODEOWNERS. Maint. is the requested reviewer (#68, #63, #61) and merged #67 and #68 (API `merged_by`). 94 of last 100 commits are his; robmsmt and theely are collaborators. Created-to-merged: maint. PRs median 3.6 h; outside 42.7 h median (0.4 h to 337 h; #68 took 10 days, #67 14 days).
11. **#73 context.** #73 open (xzyaoi, 2026-11-06 18:02 UTC, no labels, no comments): "Head can miss a restarted worker's node-table row for several minutes"; says untraced, workaround `POST /v1/dnt/_node` with `{}`, next steps reproduce, fix, add regression test. #72 open (JeniaJitsev fork, 2026-11-06 16:50, no requested reviewers): fixes llm identity groups on restore; touches only `src/internal/protocol/{registrar,registrar_test,service_health,service_health_test}.go` (+547/-63); its "Separate observation" section is where #73 came from. #66 open (AryanAhadinia, no comments): bootstrap resets new libp2p connections, likely per-IP limit in `host.go`; #67 does not reference it. History: a8c1055 = #51 "Infra Improvements" (59 files, maint., empty body); #49 = 2c069a2 "Fix/dev test bugs" (110 files); #65 = f740781 own-row sweep fix (2 files, `clock.go`, `liveness.go`, same area as #73); #67 = 5c4bbdb watchdog. go-ds-crdt: a forked copy lives in-tree at `src/internal/protocol/go-ds-crdt/` (`CLAUDE.md:63`), imported as `opentela/internal/protocol/go-ds-crdt` (`protocol/crdt.go:14`); `go.mod:16` still requires upstream `go-ds-crdt v0.6.8` (for `pb`, `go-ds-crdt/crdt.go:32`); no `replace`. No written convention for changing it; edits go through ordinary PRs (#61 touched it).
12. **Traps.** Run commands in `src/`, not the root. `CLAUDE.md:100` says golangci v2.6; CI and Makefile use v2.15. `Makefile:32` `GOLANG_VERSION ?= 1.14` and `.golangci.yml:3` `go: "1.25"` are stale against `go 1.26.0`. `make test` downloads an amd64 gotestsum by curl (`Makefile:165`). `Makefile:190` expects a `CHANGELOG.md` that does not exist. A squash merge turns a multi-commit series into one commit whose body lists the inner commits.

## Claims checked

- "Commits its own CLAUDE.md or AGENTS.md": **PARTLY**. CLAUDE.md yes (f88077d, `CLAUDE.md`); no AGENTS.md.
- "History is full of Co-Authored-By: Claude, including maintainer commits": **PARTLY**. True for the maintainer (29 of his 94 recent commits; 66 of 255 all-time, since 2026-03-08), but 67 of the last 100 have none; "full" overstates.
- "No rule requiring Signed-off-by": **CONFIRMED** (no CONTRIBUTING, DCO job or CLA; 0 in last 100). A DCO app configured outside the repository could not be seen.

## Export rules for this repository

1. Base branch `main` on a fork; branch `fix/<kebab-slug>` (`feat/`, `docs/`, `chore/` otherwise).
2. Author identity: your own name and email; no requirement found. Do not copy `Entire-Checkpoint` (tool-generated).
3. Commit subject `type(scope): summary`, e.g. `fix(protocol): ...`, no `(#N)`. Body: prose or bullets giving the why.
4. Trailers: no `Signed-off-by`. For AI-assisted work add `Co-Authored-By: Claude <Model> <noreply@anthropic.com>` (observed norm, not a rule). `Claude-Session:` is optional and private; decide before exporting.
5. Series: two commits, optional third; expect a squash.
6. From `src/`: `make test` and `make lint` (or `make check`); if the change touches only `internal/protocol`, `make test TEST_PKGS="./internal/protocol/..."` first, then the full `make test`. Needs Go 1.26+.
7. PR title = squash subject. Body: `## Problem`, `## Change`, `## Testing` (commands run and results), plus `## Observed` if reporting a field failure. Reference #73 in prose.
8. Put the AI disclosure footer in the PR body if used (as in #72, #67). Do not touch `go-ds-crdt/` unless required. Open ready, not draft.

## Silent or ambiguous

- Whether #73 should be fixed alongside #72 (same files) or after it merges; whether to rebase on #72.
- Closing keyword versus plain reference to #73.
- Whether two commits survive (squash), and whether the maintainer wants the optional third commit split out.
- Whether AI disclosure is wanted at all; the repo is silent.
- How to change `go-ds-crdt` if needed.
- Whether benchmark or hardware evidence is expected for a protocol fix.
- Whether `make lint` works with the local Go toolchain (v2.15 is built with Go 1.27).

## Commands used

```
export GIT_NO_LAZY_FETCH=1; C=<clone>
git -C $C log -1 --format='%H %ad %an' origin/main
git -C $C ls-tree -r --name-only origin/main
git -C $C show origin/main:{CLAUDE.md,.claude/commands/deploy-test.md,.codex/hooks.json,.entire/settings.json,.pre-commit-config.yaml,.github/workflows/ci.yml,.github/workflows/release.yml,src/Makefile,src/.golangci.yml,src/go.mod,README.md}
git -C $C log -100 --format='%h|%an|%cn|%P|%s' origin/main ; git -C $C log -100 --format=%B origin/main   # trailer counts via python -I
git -C $C log --no-merges -50 --numstat --format='@@%h|%ad|%an|%s' origin/main
git -C $C log origin/main -i --grep='^Signed-off-by' ; git -C $C log origin/main -i --grep='^Co-Authored-By:.*Claude'
git -C $C log --diff-filter=A --format='%h %ad %an' -- CLAUDE.md .claude .codex .entire .pi
git -C $C grep -n 'go-ds-crdt' origin/main -- src/go.mod src/internal ; git -C $C branch -r ; git -C $C shortlog -sne origin/main
git ls-remote https://github.com/eth-easl/OpenTela.git 'refs/pull/*/head'
curl api.github.com: /rate_limit; /pulls?state=all&per_page=50&sort=updated&direction=desc; /issues/73; /issues/66; /pulls/72/files; /issues/73/comments; /pulls/68; /pulls/67   (saved in scratch upstream-scratch/opentela/*.json)
```
