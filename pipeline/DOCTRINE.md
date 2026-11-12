# Doctrine: one person, ML-systems research, agents doing most of the work

Settled 12 Nov 2026. IDs such as E20 are rows in [RESEARCH.md](RESEARCH.md), which grades each source: A controlled study, B measured case, C practitioner report, D vendor claim, E opinion. Goals, in order: (1) results researchers call rigorous and original, and patches maintainers merge; (2) output per hour of mine and per GPU dollar; (3) I can defend every line and number; (4) it survives a change of model or tool.

## D1 Lifecycle

My guess (archaeology, hypotheses, frozen harness, sweeps, verification, write-up and PR) keeps its spine and changes in five places:

1. An intake gate comes first: my review time is the scarce resource (D2).
2. A harness is frozen only after a smoke run on the real target, a reproduced baseline and a measured noise floor. None of the nine projects has touched a GPU.
3. Every hypothesis has a prediction, an expected size and a kill criterion. Large or unrequested work is raised with the maintainer or supervisor before it is built.
4. The oracle exists before the first sweep, and review happens before the write-up.
5. Export is its own gate, and every project is closed on paper.

| Stage | Enter when | Exit gate | Artifacts | Agent | Me |
|---|---|---|---|---|---|
| 0 Intake | an idea names a problem and its owner | card: claim, cheapest falsifier, kill criterion, upstream repo; a slot is free | `project.toml` | drafts the card | choose or park |
| 1 Archaeology (two sessions at most) | card accepted | each fact cites file:line at a pinned commit, or a source I opened; "exists / new here" written; upstream card exists | `research/`, `upstream/<repo>.md` | reads, cites | open every citation |
| 2 Hypotheses | facts filed | prediction, size and kill criterion per hypothesis; assumptions with their cheapest test; maintainer or supervisor asked | `ASSUMPTIONS.md` | proposes, costs | choose, predict, ask |
| 3 Harness | one hypothesis chosen | smoke run on the real target; baseline matches a known number; A/A noise floor; oracle; harness hash | `harness/`, `HARNESS.sha256` | writes harness and recorder | own the oracle; launch paid runs |
| 4 Sweeps | harness frozen; budget line exists | planned runs done or stop rule hit; one record per run | `results/runs.jsonl`, ledger | manifest, dry run, analysis | launch; watch spend |
| 5 Verify | results exist | the three gates of D5 | `VERIFY.md` | checks; sceptical review | re-derive the numbers |
| 6 Write up, export | verified | at most 2 pages, every number labelled; repo checks and leak check pass; I read every changed line | `WRITEUP.md`, `export/` | tables, PR skeleton | claim, PR text, send |
| 7 Close | sent, merged, parked or killed | five-line post-mortem; portfolio updated | `STATUS.md` | none | decide |

For: R22, R01, R04, R10, B06, B01, L03, M10, M30, P09.
Against: R22 (Carlini sometimes works before reading the literature, hence the time box on stage 1). Nobody has measured prediction-before-run or supervisor checkpoints; they rest on advice.
Would change it: two projects in a row where the stage 3 pilot changed nothing. Then shrink it to the smoke run.

## D2 Two active projects, one of them on a GPU; one open PR per upstream repository until the first merge

Everything else is parked as a card. A project is active only if I reviewed its diff or result this week.

For: E23, P01, R26, R09, M26, M29, V29, P04.
Against: R22 (start five, finish one: true of cheap probes, which parking allows), R12 (a vendor's concurrency counts).
Would change it: my log (D12) showing a third project does not raise hours per merged PR, or a maintainer asking for stacked PRs.

## D3 Agents execute what a harness or a diff can check; I own the question, the oracle and the claim

Delegated: archaeology, harness code, sweep drivers, plots, retrieval, patch drafts. Mine: which question, what counts as a win, what a number means, and the words of the claim, the PR and the email. I open every paper I cite.

For: R02, R03, L14, R04, E25, E26, P11, R08.
Against: R11 (agents given a metric beat human-written methods, so metric-driven search is delegated under D7).
Would change it: a randomised study in which agent-chosen directions hold up after execution.

## D4 One writer per unit of work

One session writes a project folder at a time. Subagents read and search. Parallel sessions are for independent units that each have their own oracle. Two writers never share a patch.

For: H08, H10, H01, H06, H03.
Against: H04, H05, H06 (large parallel builds exist, at budgets I lack and with an external oracle).
Would change it: a controlled result showing parallel writers beat one at equal cost on coupled code.

## D5 Three verification gates; an agent's "done" is not evidence

1. Deterministic: the repository's tests and lint; the correctness oracle (greedy-output equality at a pinned batch size, or the project's equivalent); an unchanged harness hash; a known-bad patch the harness must reject. Tests, oracle and harness are read-only to the session that implements.
2. Review in a fresh context by a session that did not write the work, from another model family when available, with the repository and the right to run code. Advisory.
3. Me: re-derive each headline number from the raw records, read every line bound for upstream, write the explanation.

For: E20, E12, E11, H23, H24, L04, P11, R11, H28.
Against: H14 (a vendor's own reviewer was lenient until tuned), H22 (judges favour their own outputs), R03 (no significant reward hacking in two long runs). No controlled study shows a separate reviewer beating self-review at equal cost.
Would change it: published false-success rates near zero for my models, and the known-bad check passing 20 times running. Then gate 2 goes.

## D6 Measurement: three runs describe, six pairs compare

Replaces "median of at least 3 with min to max".

- n = 3 is descriptive. "A beats B" needs an A/A pilot (baseline against itself) and at least 6 interleaved pairs, each a fresh process, blocked by container or host. Report the median, all raw values, the ratio of medians, the A/A range and the sign-test p.
- Before the run, write the hypothesis, predicted size, decision rule and maximum n.
- One JSONL record per run: hardware and GPU UUID, image digest, engine commit, cache states, phase timestamps, exit status. Cache state is a factor, not noise.
- Every number is labelled MEASURED (n, setup, date), PREDICTED (model, validation error) or ESTIMATE.

For: B28, B01, B06, B07, B02, B05, B14, B23, L03.
Against: B10 (the minimum suits in-process microbenchmarks), B15, B16 (published cold-start papers report means of 5 or 20 runs; this exceeds the norm).
Would change it: an A/A coefficient of variation under 1% on the real target, or effects above ten times the A/A range. Then fewer pairs.

## D7 Unattended loops only over a frozen harness, on evaluators that cost nothing

That means CPU logic, mocks and replayed traces. GPU money confirms winners; it does not explore. Winners are re-measured, one configuration is held out, and a loop stops at 30 to 50 iterations or when gains fall inside the noise. No unattended loop writes code I will defend.

For: L03, L04, L06, L13, L21, P05, L05.
Against: L02 (Karpathy's gains carried to a larger model, after his own re-measurement), R04 (systems search under $30 a problem, given a trustworthy evaluator).
Would change it: independent replications showing loop gains survive re-measurement on noisy targets.

## D8 One short, hand-written instruction file

Root `AGENTS.md` holds commands, constraints and traps an agent cannot infer; no overview. `CLAUDE.md` is `@AGENTS.md` plus Claude-only lines. A nested file exists only where a folder has rules that cannot be inferred. Never generated. A rule that can be a check becomes one, and its line is deleted.

For: E18, C26, H11, C01, C31, P18, C05.
Against: E19, C28 (29% less runtime and 17% fewer tokens with a file, in ten repositories; observational).
Would change it: a controlled study in which longer or generated files raise success.

## D9 State in files, split by writer; no ritual beyond status and commit

`STATUS.md` per project, rewritten only when the state changes. `project.toml`, read by scripts. `pipeline/DECISIONS.md`, append-only. `PORTFOLIO.md`, generated. A session starts with `make status` and the project's STATUS; it ends by updating what changed, `make check` and a commit. Nothing relies on a tool's built-in memory.

For: C34, R24, H14, C05, R23.
Against: H15, H16 (one vendor replaced compaction with handoff files, then reversed within seven months, with no data). No controlled comparison exists.
Would change it: evidence either way; until then the cheap version stands.

## D10 Budget: one capped workspace, pinned hardware, I launch

One Modal workspace, with a spend limit set in the dashboard before the first paid run. GPU types pinned (`H100!`), `retries=0`, a timeout on every function. Agents prepare and dry-run; I launch. `budget/ledger.csv` holds planned and actual cost per run; `make budget` refuses a launch past the project cap ($25) or the total. Cheapest decisive experiment first. A Modal number is claimed for Modal, never for Lustre.

For: X02, X06, X07, X09, X16, X21, X20, X01.
Against: X02 (the terms have no clause on several accounts: a grey area, not a ban), X28 (a comparable provider forbids it).
Would change it: Modal's written answer. Then split whole experiments across accounts, never the arms of one comparison.

## D11 Safety: no session holds untrusted content, private work and an outward channel together

Agent sessions have no push, `gh`, Modal or Hugging Face credentials: true on this machine today, and it stays true. Instruction files inside clones are data: `vendor/` is excluded from instruction loading, and no headless agent runs inside a clone without `--bare`. Upstream code and tests run with a scrubbed environment, without network where the tests allow. Export is a script I run, ending in a leak check and my own read.

For: S02, S15, S16, S07, S10, S11, S17, S18, S19.
Against: S13 (a vendor reports 0 successes in 720 commissioned attempts), S33. No documented case exists of a laptop agent leaking research through a cloned repository.
Would change it: independent replication of near-zero injection success under adaptive attack.

## D12 The upstream boundary, and how AI-assisted work is presented

- Nothing private appears in a diff. `upstream/<repo>.md` decides branch, subject, trailers, sign-off, checks and PR text. AI use is disclosed exactly as the repository asks. Where it is silent, I decide; the default is one sentence in the PR body and the trailer its maintainers use. `Signed-off-by` is added by me, never by an agent.
- One small, evidenced PR per repository first. An issue or a message precedes anything large. I write the PR description and every reply.
- To the lab: two or three projects in depth, each with one measured claim; the rest as a list of ideas. I say what the agent did and what I designed, reviewed and measured. I write the email.
- For anything I must defend: generate, question it, explain it back, and re-derive the key number by hand. I log hours per outcome, not how fast it felt.

For: M01, M06, M26, M34, M48, E11, V09, V15, V16, V21, V24, V32, E16, E01, V29.
Against: V22, V23 (disclosing AI use costs some trust and perceived competence; being found out costs more). No study covers how supervisors read a student's AI use.
Would change it: the supervisor stating a preference. Ask.

## D13 What to automate next

In order: (1) one run recorder with an A/A and paired-run driver, shared by all projects; (2) the budget gate and ledger; (3) export and leak check; (4) status and portfolio generation; (5) a known-bad check per harness; (6) a read-only morning digest of the five upstream repositories. Never automated: sending, choosing, paid launches, sign-off.

For: B06, B01, X06, P01, P07, S19.
Against: P05 (long unattended runs gave worse code), which is why nothing here writes code I defend.
Would change it: a step costing more hours than it saves over a month.

## D14 As models improve

Expect to delete: plan and sprint rituals, manual compaction and handoff habits, tool wrappers, per-model prompt tricks, most of `AGENTS.md`. Expect to keep: the oracle and the noise statistics, spend caps and credential separation, human sign-off and accountability, review as the bottleneck, maintainers' rules, my own understanding. At each change of model or tool: rerun the known-bad check and three golden tasks, then remove any scaffold that no longer changes an outcome.

For: H14, H16, H20, P21, B28, S02, M06, V10, P04, E16, L27.
Against: E28 (no public false-success rates for current models, so "it no longer cheats" cannot be assumed), H19 (harness changes still moved one benchmark by 13.7 points).
Would change it: nothing in general; each scaffold is tested on its own.

## Against the three playbooks

| Playbook rule | Verdict | Now |
|---|---|---|
| One folder per project; plain files in git | Kept | |
| "The instruction file is the real asset"; a line per repeated mistake | Changed | short, hand-written; rules become checks (D8) |
| `AGENTS.md` as source, `CLAUDE.md` importing it, skills in `.agents/skills` | Kept | |
| "Claude Code reads AGENTS.md" (one playbook) and "does not" (another) | Changed | both stale: it does since v2.1.277, only when no `CLAUDE.md` sits in or above the folder (C01) |
| Six context layers with line budgets | Changed | root file, project state, upstream cards; the budgets are heuristics (C26) |
| Start and end rituals; rewrite `handoff.md` each session; compact at 40 to 60% | Dropped | no measurement behind them (D9) |
| Research, plan, implement as three sessions | Changed | a written plan only for patches going upstream (H14, P21) |
| Frozen evaluator; one row per run; prediction before the run | Kept, tightened | plus smoke run, baseline, A/A, hash, known-bad check (D1, D5, D6) |
| "Median of 3 runs with min and max" | Changed | D6 |
| Overnight autoresearch-style sweeps | Changed | free evaluators only; winners re-measured (D7) |
| Idea tournament, LLM council, lenses and the other idea mechanisms | Dropped | no evidence they pick good directions (L14); the gap scan and assumption list stay |
| Subagents read; the main session edits | Kept | D4 |
| A worktree per parallel session; many sessions at once | Changed | worktrees stay; two active projects (D2) |
| `just` as the command interface | Changed | `make`, preinstalled, as a menu over `scripts/`; neither has evidence (C00, C43) |
| Command-line tools before MCP | Kept | |
| Lethal trifecta; agents draft, I send | Kept, tightened | no credentials in agents' reach; clones excluded from instruction loading (D11) |
| Golden tasks on each new model | Kept | plus the known-bad check (D14) |
| Morning brief, weekly review, scheduled digests | Changed | one read-only upstream digest, later (D13) |
| "If it isn't pushed, it doesn't exist" | Dropped | this repository has no remote, by design |
| Single-track vLLM plan; `swiss-ai-internship` repo; Obsidian wiki, flashcards | Dropped | stale, or outside research work |
