# Doctrine: one person, ML-systems research, agents doing most of the work

Settled and audited 12 Nov 2026 ([AUDIT.md](AUDIT.md)). IDs such as E20 are rows in [RESEARCH.md](RESEARCH.md), graded A (controlled study) to E (opinion). Goals, in order: (1) rigorous, original results and patches maintainers merge; (2) output per hour of mine and per GPU dollar; (3) I can defend every line and number; (4) it survives a change of model or tool.

## D1 Lifecycle

The spine of my guess holds. Added: an intake gate; a smoke run, a reproduced baseline and a noise floor before a harness is frozen (none of the nine projects has touched a GPU); a prediction, size and kill criterion per hypothesis, and a question to the maintainer or supervisor before large work; the oracle before the first sweep; export as its own gate; a written close.

| Stage | Enter when | Exit gate | Artifacts | Agent | Me |
|---|---|---|---|---|---|
| 0 Intake | an idea names a problem and its owner | card: claim, cheapest falsifier, kill criterion, upstream repo; a slot is free | `project.toml` | drafts the card | choose or park |
| 1 Archaeology (two sessions at most) | card accepted | each fact cites file:line at a pinned commit or a source I opened; "exists / new here" written; upstream card exists | `research/`, `upstream/<repo>.md` | reads, cites | open every citation |
| 2 Hypotheses | facts filed | prediction, size, kill criterion per hypothesis; assumptions with their cheapest test; maintainer or supervisor asked | `ASSUMPTIONS.md` | proposes, costs | choose, predict, ask |
| 3 Harness | one hypothesis chosen | smoke run on the real target; baseline matches a known number; A/A noise floor; oracle; harness hash | `harness/`, `HARNESS.sha256` | writes harness and recorder | own the oracle; launch paid runs |
| 4 Sweeps | harness frozen; budget line exists | planned runs done or stop rule hit; one record per run | `results/runs.jsonl`, ledger | manifest, dry run, analysis | launch; watch spend |
| 5 Verify | results exist | the three gates of D5 | `VERIFY.md` | checks; sceptical review | re-derive the numbers |
| 6 Write up, export | verified | 2 pages at most, every number labelled; repo checks and leak check pass; I read every changed line | `WRITEUP.md`, `export/` | tables, PR skeleton | claim, PR text, send |
| 7 Close | sent, merged, parked or killed | five-line post-mortem; portfolio updated | `STATUS.md` | none | decide |

For: R22, R01, R04, R10, B06, B01, L03, M10, M30, P09.
Against: R22 (sometimes work before reading the literature: hence the time box on stage 1), R24 (ideas that start worse can win: a kill criterion names a result, not a first score). Predictions before runs and supervisor checkpoints rest on advice; nobody has measured them.
Would change it: two projects in a row where the stage 3 pilot changed nothing. Then keep only the smoke run.

## D2 A review budget: two active projects, one on a GPU; one open PR per upstream repository until the first merge

The rest are parked as cards. A project is active only if I reviewed its diff or result this week. This is a budget for my attention, not a measured optimum.

For: P01, R26, P04, R09, M26, M29, V29, E23.
Against: E23 (reports no drop in output up to about four projects a week; unverified), V35 (the lab publishes at volume), R22 (start five, finish one: true of cheap probes, which parking allows), R12.
Would change it: my log (D12) showing a third project does not raise hours per merged PR, or a maintainer asking for stacked PRs.

## D3 Agents execute what a harness or a diff can check; I own the question, the oracle and the claim

Delegated: archaeology, harness code, sweep drivers, plots, retrieval, patch drafts, candidate hypotheses. Mine: which question, what counts as a win, what a number means, and the words of the claim, the PR and the email. I open every paper I cite.

For: R02, R03, L14, R04, E25, E26, P11, R08.
Against: R11 (given a metric, agents beat human-written methods: that search is delegated under D7).
Would change it: a randomised study in which agent-chosen directions hold up after execution.

## D4 One writer per unit of work

One session writes a project folder at a time. Subagents read and search. Parallel sessions are for independent units with their own oracle. Two writers never share a patch.

For: H08, H10, H01, H06, H03.
Against: H04, H05, H06 (large parallel builds exist, at budgets I lack and with an external oracle).
Would change it: a controlled result showing parallel writers beat one at equal cost on coupled code.

## D5 Three verification gates; an agent's "done" is not evidence

1. Deterministic: the repository's tests and lint; the correctness oracle (greedy-output equality at a pinned batch size, or the project's equivalent); an unchanged harness hash; a known-bad patch the harness must reject. Tests, oracle and harness are read-only to the implementing session; the hash check enforces it.
2. Review in a fresh context by a session that did not write the work, from another model family when available, able to run code. Advisory.
3. Me: re-derive each headline number from the raw records, read every line bound for upstream, write the explanation.

For: E20, E12, E11, H23, H24, L04, P11, R11, H28.
Against: H14 (a vendor's reviewer was lenient until tuned), H22 (judges favour their own outputs), R03 (no significant reward hacking in two long runs). No controlled study shows a separate reviewer beating self-review at equal cost.
Would change it: false-success rates near zero for my models, and the known-bad check passing 20 times running. Then gate 2 goes.

## D6 Measurement: three runs describe, six pairs compare

Replaces "median of at least 3 with min to max".

- n = 3 is descriptive. "A beats B" needs an A/A pilot (baseline against itself) and 6 interleaved pairs, each a fresh process, blocked by container or host. Report the median, all raw values, the ratio of medians, the A/A range and the sign-test p. An effect within twice the A/A range gets 10 pairs, once; then it is reported as not distinguishable.
- Before the run: hypothesis, predicted size, decision rule, maximum n. A miss is investigated first.
- One JSONL record per run: hardware and GPU UUID, image digest, engine commit, harness hash, cache states, phase times, exit status. Cache state is a factor, not noise. Nothing is dropped silently.
- Every number is labelled MEASURED (n, setup, date), PREDICTED (model, validation error) or ESTIMATE.

For: B28, B01, B06, B07, B02, B05, B14, X21, B23, L03.
Against: B10 (the minimum suits in-process microbenchmarks), B15, B16 (published cold-start papers report means of 5 or 20 runs; this exceeds the norm).
Would change it: an A/A coefficient of variation under 1% on the real target. Then n = 3 may describe effects far above the noise; a comparison still needs 6 pairs, since 5 cannot reach p < 0.05.

## D7 Unattended loops: frozen harness, free evaluators, inside a container

Free evaluators are CPU logic, mocks and replayed traces. GPU money confirms winners; it does not explore. Winners are re-measured, one configuration is held out, and a loop stops at 30 to 50 iterations or when gains fall inside the noise. It runs in a container with no network, no host mounts and timeouts it cannot edit. No unattended loop writes code I will defend.

For: L03, L04, L06, L13, L21, P05, L05, S08.
Against: L02 (gains that carried to a larger model, after the author's own re-measurement), R04 (systems search under $30 a problem, given a trustworthy evaluator).
Would change it: independent replications of loop gains surviving re-measurement on noisy targets.

## D8 One hand-written instruction file, kept small

Root `AGENTS.md` holds commands, constraints and traps an agent cannot infer; no overview. `CLAUDE.md` is `@AGENTS.md` plus Claude-only lines. A nested file only where a folder has rules that cannot be inferred. Never generated. A rule that can be a check becomes one, and its line goes. Small is a cost rule (every line is paid for on every task); no length threshold has been measured.

For: E18, C26, H11, C01, C31, P18, C05.
Against: C26 (no relation between file length and success; with one Claude model, hand-written files lowered success slightly), E19, C28 (with a file, 29% less runtime and 17% fewer tokens on 124 pull requests: efficiency, not success).
Would change it: a controlled study in which longer or generated files raise success.

## D9 State in files, split by writer; one status file, no fixed ritual

`STATUS.md` per project is the handoff, updated when the state changes. `project.toml` is read by scripts. `pipeline/DECISIONS.md` is append-only. `PORTFOLIO.md` is generated. A session starts with `make status` and the project's STATUS; it ends by updating what changed, `make check` and a commit. Inside a task the tool's own compaction is fine. Nothing relies on a tool's memory.

For: C34, R24, H14, C05, R23.
Against: H15, H16 (one vendor swapped compaction for handoff files, then reversed within seven months, without data). No controlled comparison exists.
Would change it: evidence either way; until then the cheap version stands.

## D10 Budget: one capped workspace, pinned hardware, I launch

One Modal workspace. Its free credit is $30 a month, so $90 means three months or an academic grant: apply. `make budget` refuses a launch past the project cap ($25) or the month's total, from `budget/ledger.csv`; the dashboard spend limit, set before the first paid run, is the second stop. GPU types pinned (`H100!`), `retries=0`, a timeout on every function. Agents prepare and dry-run; I launch. Cheapest decisive experiment first. A Modal number is claimed for Modal, never for Lustre.

For: X01, X02, X03, X06, X07, X09, X16, X20.
Against: X02 (no clause on several accounts: a grey area, not a ban), X28 (a comparable provider forbids it), X06 (how fast the limit acts is undocumented).
Would change it: Modal's written answer. Then split whole experiments across accounts, never the arms of one comparison.

## D11 Safety: no session holds untrusted content, private work and an outward channel together

- **Credentials.** Agent sessions reach no push, `gh`, Modal or Hugging Face credentials. `make check` tests it. On 12 Nov none were found; the login keychain was not inspected.
- **Network.** Open egress is an outward channel. Sessions that work inside clones or run upstream code use the tool's sandbox with default-deny egress. Web research runs in sessions given public inputs only.
- **Unattended work.** Loops (D7) and auto or skip-permission sessions I am not watching run only in a container or microVM with no host mounts and no secrets. Until one is set up, there are none.
- **Clones.** Their instruction files are data, and `vendor/` is excluded from instruction loading. A repository's rules reach an agent through `upstream/<repo>.md`, which I have read; its own files load only in a patch session I start inside its worktree on purpose. No headless agent runs in a clone without `--bare`. Upstream code runs with a scrubbed environment.
- **Export** is a script I run, ending in a leak check and my own read.

For: S02, S15, S16, S07, S08, S10, S11, S17, S18, S19.
Against: S13, S33 (a vendor reports 0 successes in 720 commissioned attempts), M01, C26 (a repository's instruction file is also its rulebook). No documented case exists of a laptop agent leaking research through a cloned repository.
Would change it: independent replication of near-zero injection success under adaptive attack.

## D12 The upstream boundary, and how AI-assisted work is presented

- Nothing private appears in a diff. `upstream/<repo>.md` decides branch, subject, trailers, sign-off, checks and PR text. AI use is disclosed exactly as the repository asks. Where it is silent, I ask the maintainer; without an answer, one sentence in the PR body, and a trailer only if their own history shows one. `Signed-off-by` is mine to add, never an agent's.
- One small, evidenced PR per repository first. An issue or a message precedes anything large. I write the PR description and every reply.
- To the lab: two or three projects in depth, each with one measured claim; the rest as a list of ideas. I say what the agent did and what I designed, reviewed and measured. I write the email.
- For anything I must defend: generate, question it, explain it back, re-derive the key number by hand (suggestive evidence, small groups). I log hours per outcome, not how fast it felt.

For: M01, M06, M26, M34, M48, E11, V09, V15, V16, V21, V24, V32, E16, E01, V29.
Against: V22, V23 (disclosure costs some trust; being found out costs more), V35. No study ties depth or volume to being hired, or covers how supervisors read a student's AI use.
Would change it: the supervisor stating a preference. Ask.

## D13 What is automated, and what comes next

Built by the migration: status and portfolio, budget gate and ledger, export and leak check, credentials check. Next: (1) the run recorder wired into each harness; (2) a known-bad check per harness; (3) a container runner, without which D7 loops cannot exist; (4) a read-only morning digest of the five upstream repositories. Never automated: sending, choosing, paid launches, sign-off.

For: B06, B01, X06, P01, P07, S19, S08.
Against: P05 (long unattended runs gave worse code): nothing here writes code I defend.
Would change it: a step costing more hours than it saves over a month.

## D14 As models improve

Expect to delete: plan and sprint rituals, fixed compaction habits, tool wrappers, per-model prompt tricks, the lines of `AGENTS.md` that work around a model. Expect to keep: commands and constraints an agent cannot infer, the oracle and the noise statistics, spend caps and isolation, human sign-off, review as the bottleneck, maintainers' rules, my own understanding. At each change of model or tool: rerun the known-bad check and three golden tasks, then remove any scaffold that no longer changes an outcome.

For: H14, H16, H20, P21, B28, S02, M06, V10, P04, E16, L27.
Against: E28 (no public false-success rates for current models), H19 (harness changes still moved one benchmark by 13.7 points).
Would change it: nothing in general; each scaffold is tested on its own.

## Against the three playbooks

| Playbook rule | Verdict | Now |
|---|---|---|
| One folder per project; plain files in git; command-line tools before MCP; subagents read, the main session edits | Kept | D4 |
| "The instruction file is the real asset"; a line per repeated mistake, hardening into tests and hooks | Kept in part | the ladder stays; the file is a cost to keep small, not the main asset (E18, D8) |
| `AGENTS.md` as source, `CLAUDE.md` importing it, skills in `.agents/skills` | Kept | |
| "Claude Code reads AGENTS.md" (one playbook), "does not" (another) | Changed | the second is stale, the first incomplete: it does since v2.1.277, only when no `CLAUDE.md` sits in or above the folder (C01) |
| Six context layers with line budgets | Kept in part | fewer layers; the budgets stay rules of thumb (C26) |
| Start and end rituals; rewrite `handoff.md` each session; compact at 40 to 60% | Changed | one status file, updated when the state changes; no fixed-fill compaction (D9) |
| Research, plan, implement as three sessions; plan mode first | Changed | a written plan only for patches going upstream (H14, P21) |
| Frozen evaluator; one row per run; prediction before the run; surprise scan | Kept, tightened | plus smoke run, baseline, A/A, hash, known-bad check (D1, D5, D6) |
| "Median of 3 runs with min and max" | Changed | D6 |
| Overnight autoresearch-style sweeps | Changed | free evaluators, in a container; winners re-measured (D7) |
| Idea tournament and LLM council to choose what to work on | Dropped | executed AI ideas scored below expert ideas (L14); nothing shows ranking picks good directions |
| The other idea mechanisms (lenses, pre-mortem, contradiction lint, literature diff, dead-end revival, spikes) | Not evaluated | optional |
| A worktree per parallel session; many sessions at once | Changed | worktrees stay; two active projects (D2) |
| `just` as the command interface | Changed | `make`, preinstalled, over `scripts/`; neither has evidence (C00, C43) |
| Lethal trifecta; agents draft, I send; unattended agents only in a container | Kept, tightened | D11 |
| Agents open draft PRs with `gh`; agents get their own keys with spending caps | Dropped | agents hold no outward credential (D11, X16) |
| Golden tasks on each new model | Kept | plus the known-bad check (D14) |
| Morning brief, weekly review, scheduled digests | Changed | one read-only upstream digest, later (D13) |
| "If it isn't pushed, it doesn't exist" | Changed | agents never push; I keep a backup myself (MIGRATION.md) |
| Single-track vLLM plan; `swiss-ai-internship` repo; Obsidian wiki, flashcards | Dropped | stale, or outside research work |
