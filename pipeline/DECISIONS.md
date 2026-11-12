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
