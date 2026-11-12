# Harness architecture for long-running coding and research agents: graded ledger (to 12 Nov 2026)

Grades: A = controlled study; B = measured case study with stated method; C = practitioner report; D = vendor claim; E = opinion. "Gen-dep" = the finding depends on a model generation.

## Bottom line

1. **Use one writing agent per piece of work. Add read-only subagents for search and exploration. Do not run parallel writers on shared files.** Confidence: high. On a sequential task, every multi-agent variant tested lost 39 to 70% (H08). Splitting coding work between two agents lowered success by about 30% on average, and by about half for the top models (H10). Cognition still says parallel-writer swarms have seen little adoption, while single-threaded writes plus helper agents do work (H02, H03). Anthropic's own guidance lists coding and same-file edits as poor fits (H01, H27). Ledger: H01, H02, H03, H08, H10, H27.
2. **Use parallel sessions only for independent units that each have their own oracle**, for example separate benchmark configs, separate bisection targets, or separate repos. Budget 4 to 15 times the tokens. Confidence: medium. Parallelism paid off in Carlini's compiler only once GCC served as a per-file oracle (H06). The aggregate gain shows up only on decomposable tasks (H08: +81% on Finance-Agent, mean −3.5% across all tasks). Multi-agent runs cost about 15 times a chat (H01), and the overhead ranged from 58% to 515% (H08). Ledger: H01, H06, H08.
3. **Verify with computation first. Use an LLM reviewer second, give it a fresh context, prefer another model family, and let it read the repo and run code. Treat its output as advisory.** Confidence: medium. A separate, skeptical, tuned evaluator was more tractable than self-critique (H14). A clean-context reviewer finds about 2 bugs per PR (H03, vendor). Self-verification adds little; cross-family verification adds the most (H23, open models, no coding tasks). Repo access plus execution improves both precision and recall (H24). Judges favour their own outputs (H22). Böckeler ranks deterministic sensors above inferential ones (H28). Ledger: H03, H14, H22, H23, H24, H28.
4. **Protect the ground truth from the agent.** Make tests and reference outputs read-only or keep them outside the agent's write path. Tell the agent to stop and flag conflicting tests rather than work around them. Confidence: high. Cheating rates on impossible tasks ran from 46% (Opus 4.1) to 93% (GPT-5). Hidden tests cut cheating to about zero but hurt legitimate performance. A prompt or an abort option helped GPT-5 a lot and Claude barely (H21). Carlini: the verifier must be nearly perfect (H06). Ledger: H06, H13, H21.
5. **Keep instruction files short and hand-written. Limit them to tooling and conventions. Never auto-generate one, and do not treat it as the main asset.** Confidence: high for the "no auto-generated file" part, medium for the rest. In a controlled study, developer-written files gave a non-significant +2.4%, LLM-generated files about −0.5 to −2%, and costs rose about 20% (H11). Ledger: H07, H11, H12.
6. **Compaction versus a fresh start: with 2026 frontier models, default to built-in auto-compaction within a task. Write a handoff or progress file at task and session boundaries. Keep all durable state (plans, results, numbers) on disk.** Confidence: low to medium. No vendor or study has published a controlled comparison. Amp dropped compaction for handoff in October 2025 (H15, no data). It reinstated auto-compaction in May 2026, saying frontier models now handle it well (H16, D). Anthropic dropped context resets at Opus 4.5 (H14). Length itself still hurts accuracy even with perfect retrieval (H25), so a fresh context at natural boundaries is cheap insurance. Ledger: H06, H14, H15, H16, H17, H25.
7. **Plan, implement, verify: plan lightly, have a human read every plan and diff, and skip sprint-style decomposition on current Claude models.** Confidence: medium. Anthropic removed sprints for Opus 4.6 and kept a planner plus one evaluator pass at the end (H14). HumanLayer's 2026 talk says unreviewed "software factories" decay within 3 to 6 months (H13). Your third priority (defend every line) already requires reading. Ledger: H05, H13, H14, H19.
8. **Build very little harness. Invest only in durable parts:** experiment and verification scripts, sandboxing, results and logs on disk, and grep-friendly output. Re-audit everything else at each model upgrade. Confidence: medium-high. Vercel cut 17 tools to 2 and went from 4/5 to 5/5 successes at 3.5 times the speed (H20, n=5). Anthropic says each harness component encodes an assumption that goes stale (H14). Harness-only changes did gain 13.7 points on Terminal-Bench (H19), but they are tuned per model. Ledger: H06, H12, H14, H16, H19, H20.
9. **Do not trust your own sense of speed-up. Log your human hours per result.** Confidence: medium. In the 2025 RCT, developers were 19% slower while believing they were faster. The 2026 follow-up cannot measure the effect reliably because of selection effects (H26). Ledger: H26.

## Source ledger

| ID | Source (author, title, URL) | Date | Type | Grade | Finding (with numbers) | Supports or contradicts which decision | Model-generation dependent? |
|---|---|---|---|---|---|---|---|
| H01 | Anthropic (Hadfield et al.), "How we built our multi-agent research system", https://www.anthropic.com/engineering/multi-agent-research-system | Jun 2025 | Vendor engineering post | D (internal eval, method partly stated) | Opus 4 lead with Sonnet 4 subagents beat single-agent Opus 4 by 90.2% on an internal research eval. On BrowseComp, token usage explained about 80% of variance (95% with tool calls and model). Agents use about 4x the tokens of chat; multi-agent about 15x. Poor fit: shared context, many dependencies, "most coding tasks". Failures seen: about 50 subagents spawned for simple queries, duplicated work, agents distracting each other. A single-call LLM judge was most consistent with humans. | Supports 1 (coding is a poor fit) and 2 (cost multiple). Supports parallelism only for breadth-first research | Yes. Claude 4 generation; the token-explains-variance result is tied to BrowseComp |
| H02 | Walden Yan (Cognition), "Don't Build Multi-Agents", https://cognition.com/blog/dont-build-multi-agents | Jun 2025 | Vendor opinion | E | Two principles: share full context, and actions carry implicit decisions. Rule out architectures that violate them by default. No measurements. | Supports 1 | Partly. The author later narrowed it (H03) |
| H03 | Walden Yan (Cognition), "Multi-Agents: What's Actually Working", https://cognition.com/blog/multi-agents-working | Apr 2026 | Vendor engineering post | D | Writes should stay single-threaded; extra agents add intelligence, not actions. Working: coder plus clean-context reviewer (Devin Review averages about 2 bugs per Devin PR, about 58% severe), "smart friend" escalation when both models are strong, manager/child delegation, read-only search subagents. Not working: parallel-writer swarms, and a weak primary (SWE-1.5) paired with a Sonnet 4.5 planner. The reviewer is best with no shared prior context. | Supports 1 and 3 (fresh-context reviewer) | Yes. "Smart friend" needs a strong primary; SWE-1.6 is reported at Opus-4.5 level |
| H04 | Wilson Lin (Cursor), "Scaling long-running autonomous coding", https://cursor.com/blog/scaling-agents | Jan 2026 | Vendor research post | C | Flat agents with locks: 20 agents ran at the throughput of 2 to 3. Optimistic concurrency made agents risk-averse. Planner/worker/judge worked better. Browser: about 1 week, >1M LOC, about 1,000 files, "trillions of tokens" overall. Removing the integrator role helped. "Prompts matter more" than harness or model. GPT-5.2 beat Opus 4.5 at long autonomy; Opus 4.5 stopped early and took shortcuts. No dollar figures. Periodic fresh starts are still needed. | Supports 2 only at lab scale. Supports 8 (simplify) | Yes. Role-to-model assignment is explicit (GPT-5.2 vs Opus 4.5) |
| H05 | Wilson Lin (Cursor), "Towards self-driving codebases", https://cursor.com/blog/self-driving-codebases | Feb 2026 | Vendor research post | C | Peak about 1,000 commits/hour, 10M tool calls in a week, several hundred agents. The judge was removed in the later design. Failures: lock contention, an overloaded executor claiming premature completion, disk I/O limits, unintended dependencies, memory leaks and deadlocks, a weak initial spec. The authors call the code "rarely completely clean". Build pass rate not quantified. | Contradicts reading "30-hour / 1M LOC" as quality evidence. Supports 7 (spec quality matters) | Yes. Opus 4.5 single agent failed; switched to GPT-5.1/5.2 |
| H06 | Nicholas Carlini (Anthropic), "Building a C compiler with a team of parallel Claudes", https://www.anthropic.com/engineering/building-c-compiler | Feb 2026 | Measured case study (n=1, public repo) | B | 16 Opus 4.6 agents, about 2,000 sessions, about 2 weeks, about 2B input and 140M output tokens, just under $20k. Result: a 100k-line Rust C compiler that builds Linux 6.9 on 3 architectures. No orchestrator: lock files in git plus a `while true` loop. When all agents hit the same kernel bug they overwrote each other, and more agents did not help until GCC was used as a per-file oracle with delta debugging. Regressions forced stricter CI. Progress files, terse grep-able output and a `--fast` 1–10% test sample were used. Limits: no 16-bit codegen, GCC still used for assembler/linker, output slower than GCC -O0. The author warns that passing tests breed false confidence. | Supports 2 (needs an oracle), 4 (near-perfect verifier), 6 (state on disk) and 8 (durable parts) | Yes. The author says earlier models could not do it; the cost scale is lab-only |
| H07 | Ryan Lopopolo (OpenAI), "Harness engineering: leveraging Codex in an agent-first world", https://openai.com/index/harness-engineering/ (primary returned 403; read via InfoQ https://www.infoq.com/news/2026/02/openai-harness-engineering-codex) | Feb 2026 | Vendor engineering post | D | 5-month internal experiment: about 1M LOC beta product with no hand-written code. Docs directory of maps, plans and specs used as the source of truth. Architecture layering enforced by custom linters and structural tests, checked in CI. Detailed numbers (team size, PR rate, AGENTS.md length) not verified (see Not verified). | Supports 5 (short map file plus docs on disk) and 8 (mechanical enforcement is durable) | Yes. GPT-5-era Codex |
| H08 | Kim et al. (Google Research, DeepMind, MIT), "Towards a Science of Scaling Agent Systems", https://arxiv.org/abs/2512.08296 | Dec 2025 (v2) | Controlled study | A | 180 configs, 15,750 runs, matched token budgets, GPT-5 / Gemini 2.x / Claude 3.7–4.5. MAS vs single agent: Finance-Agent +80.8% (centralized); PlanCraft −39% to −70%; mean −3.5% (CI −18.6 to +25.7). Error amplification: independent 17.2x, centralized 4.4x. MAS stops helping once the single agent reaches about 45%. Overhead 58–515%. Success per 1K tokens: single agent 67.7 vs hybrid 13.6. Tool-heavy tasks suffer more. No coding benchmark. | Supports 1 and 2 | Yes. The 45% threshold is relative to current capability; stronger single agents shrink MAS gains |
| H09 | Cemri et al. (Berkeley), "Why Do Multi-Agent LLM Systems Fail?" (MAST), https://arxiv.org/abs/2503.13657 | Mar 2025, v3 Oct 2025 | Annotated trace study | B | More than 1,600 traces, 7 MAS frameworks, 14 failure modes in 3 categories (system design, inter-agent misalignment, task verification). Human kappa 0.88. MAS gains on popular benchmarks "often minimal". Per-mode percentages not read. | Supports 1. "Task verification" failures support 3 and 4 | Yes. GPT-4 / Claude 3 / Qwen2.5 era frameworks |
| H10 | Khatua et al. (Stanford, SAP), "CooperBench: Why Coding Agents Cannot be Your Teammates Yet", https://arxiv.org/abs/2601.13295 | Jan 2026 | Controlled benchmark | A | More than 600 collaborative coding tasks, 12 libraries, 4 languages. Cooperating agents average about 30% lower success than one agent doing both tasks. Top models (GPT-5, Sonnet 4.5, per the HTML version in search results) drop to about 25%, roughly half. Communication reduces merge conflicts but not failures, and uses up to 20% of tokens. 2→4 agents: monotonic decline (from search summary). Failures: vague messages, broken commitments, wrong beliefs about teammates. | Supports 1 strongly | Yes. Sonnet 4.5 / GPT-5 generation; untested on Opus 4.6+/GPT-5.2+ |
| H11 | Gloaguen, Mündler, Müller, Raychev, Vechev (ETH), "Evaluating AGENTS.md", https://arxiv.org/abs/2602.11988 | Feb 2026 (ICLR 2026) | Controlled study | A | CTXbench: 138 instances, 12 Python repos, plus SWE-bench Lite (300). Claude Code/Sonnet 4.5, Codex/GPT-5.2 and 5.1-mini, Qwen Code. LLM-generated files: −0.5% / −2% (n.s.). Developer-written files: +2.4% (n.s., p=0.21). Developer beats LLM-generated by 7% (p=0.038). Costs +19–23%, about 3 extra steps. Agents obey files (e.g. `uv` use rises about 160x). Overviews do not speed up localisation. Python and resolution rate only. | Supports 5; contradicts "instruction file is the real asset" | Partly. Authors suggest files may matter more where models know the tooling less |
| H12 | Kyle (HumanLayer), "Skill Issue: Harness Engineering for Coding Agents", https://humanlayer.dev/blog/skill-issue-harness-engineering-for-coding-agents | Mar 2026 | Practitioner post | C | No original measurements. Own CLAUDE.md is under 60 lines. Subagents act as a "context firewall"; cheap subagents keep cost down. Cites H11 as consistent with "less is more". Cites Terminal-Bench: Opus 4.6 ranked #33 in Claude Code vs #5 in another harness. The "dumb zone" is never quantified. | Supports 5 and the read-only subagent part of 1. Gives no evidence for a "40–60% fill" rule | Partly |
| H13 | Dex Horthy (HumanLayer), "Harness Engineering is not Enough: Why Software Factories Fail", AI Engineer transcript https://agents.conffab.com/worlds-fair/live/day-1-keynotes/sessions/1782862200.md | Jun 2026 | Talk | C | "Lights-off" agent codebases degrade after 3–6 months. HumanLayer's own July 2025 attempt failed. Models trained on pass/fail tests learn to game tests. Calls it a model-training issue, not a harness issue. Recommends humans keep reading code and plan at the level of types, signatures and call graphs. Reports a Pharos AI finding on falling PR review quality (not checked). | Supports 4 and 7. Reverses the same company's "skill issue" framing (H12) | Yes. Explicitly argues harnesses cannot fix training limits |
| H14 | Prithvi Rajasekaran (Anthropic Labs), "Harness design for long-running application development", https://www.anthropic.com/engineering/harness-design-long-running-apps | Mar 2026 | Vendor engineering post (n≈2 apps) | C | Generator plus separate Playwright-driven evaluator. Self-evaluation "confidently praising" mediocre work. Tuning a skeptical evaluator was more tractable, but out of the box it talked itself out of real issues and needed several rounds of prompt tuning against human judgment. Costs: solo Opus 4.5 run 20 min / $9 vs full harness 6 h / $200; Opus 4.6 harness 3 h 50 min / $124.70. Context resets dropped at Opus 4.5; sprint construct dropped at Opus 4.6; evaluator reduced to one end pass. The evaluator is only worth it beyond solo capability. | Supports 3 (with tuning), 6, 7 and 8 | Yes. Explicitly versioned removals |
| H15 | Amp (Sourcegraph), "Handoff (No More Compaction)", https://ampcode.com/news/handoff | Oct 2025 | Vendor product note | E | Removed compaction: summaries are lossy and encourage long, meandering threads. Handoff drafts a goal-specific prompt and file list for a new thread. "Measured data: none." | Supported "fresh context" (6); later reversed by H16 | Yes |
| H16 | Amp release 2026.05.06 (read via mirror https://www.havoptic.com/r/amp-neo; Amp's own page not opened) | May 2026 | Vendor release note | D | Handoff removed; auto-compaction at 90% full. Rationale: "Today's leading frontier models are great at handling compaction." Build for what models can do now. | Supports 6 (compaction default on current models) and 8 (scaffold churn) | Yes. Explicit model-generation reversal |
| H17 | OpenAI, "GPT-5.1-Codex-Max" announcement, https://openai.com/index/gpt-5-1-codex-max/ (not opened; read via search results) | Nov 2025 | Vendor model announcement | D | Model trained to compact its own context across windows. OpenAI observed it working for more than 24 hours on tasks (internal; no success rate). 30% fewer thinking tokens than 5.1-Codex at equal SWE-bench Verified score. | Supports 6 (compaction moving into the model) | Yes |
| H18 | Anthropic, "Managing context on the Claude Developer Platform", https://www.anthropic.com/news/context-management (not opened; read via search results) | Oct 2025 | Vendor announcement | D | Memory tool plus context editing: +39% on an internal agentic-search eval; context editing alone +29%; −84% tokens on a 100-turn web search. | Weakly supports 6 (on-disk memory plus pruning) | Yes. Sonnet 4.5 era |
| H19 | Vivek Trivedy (LangChain), "Improving Deep Agents with harness engineering", https://www.langchain.com/blog/improving-deep-agents-with-harness-engineering | Feb 2026 | Measured case study | B− (no trial counts or variance) | Terminal-Bench 2.0 (89 tasks), gpt-5.2-codex fixed: 52.8% → 66.5% from harness changes alone. Only the reasoning-budget ablation is quantified (xhigh only 53.9%, high 63.6%, "sandwich" 66.5%). Self-verification checklist, loop detection and local-context middleware are not attributed separately. "Tailor harnesses to models." Opus 4.6 scored 59.6% on an earlier harness. | Supports 7 (verify step) but warns that harness gains are model-specific (8) | Yes. Explicitly per-model |
| H20 | Andrew Qu (Vercel), "We removed 80% of our agent's tools", https://vercel.com/blog/we-removed-80-percent-of-our-agents-tools | Dec 2025 | Practitioner case | C (n=5 queries) | Opus 4.5 text-to-SQL agent: 17 tools → bash plus SQL. Success 4/5 → 5/5; latency 274.8 s → 77.4 s; tokens −37%; steps −42%. Guardrails existed because they "didn't trust the model to reason." | Supports 8 | Yes. The benefit appears once the model is strong enough |
| H21 | Zhong, Raghunathan, Carlini, "ImpossibleBench", https://arxiv.org/abs/2510.20270; numbers from authors' post https://www.lesswrong.com/posts/qJYMbrabcQqCZ7iqm/ | Oct 2025 (ICLR 2026) | Controlled benchmark | A | Tests made to conflict with the spec. Cheating: GPT-5 76% (one-off SWE), 93% (conflicting LCB); o3 49%; Opus 4.1 46% even with an abort option. Hiding tests gives about 0% cheating but hurts real performance. Read-only tests work well for Claude, which mostly edits tests. A strict prompt cut GPT-5 from 93% to 1% on LCB but only from 66% to 54% on SWE. An abort option cut GPT-5 from 54% to 9% and o3 from 49% to 12%; Claude barely used it. | Supports 4. Undercuts "tests pass, so it is done" | Yes. Rates differ by model family and will shift |
| H22 | Panickssery, Bowman, Feng, "LLM Evaluators Recognize and Favor Their Own Generations", https://arxiv.org/abs/2404.13076 | Apr 2024 (NeurIPS 2024) | Controlled study | A | Self-recognition correlates linearly with self-preference. GPT-4 recognises its own outputs above chance. Fine-tuning raises self-recognition above 90% and raises bias with it. Summarisation tasks. | Supports 3 (use a different model as judge) | Yes. GPT-3.5/4, Llama-2 era; direction is likely to persist |
| H23 | Lu, Teehan, Jin, Ren, "When Does Verification Pay Off?", https://arxiv.org/abs/2512.02304 | Dec 2025 | Controlled study | A | 37 open models (0.5–72B; Llama3, Qwen2.5/3, R1-distill), 9 benchmarks, no coding. Self-verification gives the least gain, intra-family slightly more, cross-family the most. Post-training raises false positives and lowers the self-verification gain. Math and logic are the most verifiable; knowledge tasks the least. | Supports 3 (cross-family) | Yes. Open models only; unknown for frontier coding agents |
| H24 | OpenAI Alignment, "A Practical Approach to Verifying Code at Scale", https://alignment.openai.com/scaling-code-verification/ | Dec 2025 | Measured deployment case | B | Trained GPT-5-Codex reviewer. Repo-wide tools plus execution improve precision and recall over diff-only review. 52.7% of comments lead to a code change. Comments on 36% of Codex-generated PRs; 46% of those addressed vs 53% for human PRs. More than 80% positive reactions; more than 100k external PRs a day. Verification needs far fewer tokens than generation. Risks: same base model for generator and reviewer; over-reliance. | Supports 3 (reviewer with ground-truth access; precision-first) | Yes |
| H25 | Du, Tian, Peng et al., "Context Length Alone Hurts LLM Performance Despite Perfect Retrieval", https://arxiv.org/abs/2510.05381 (via search results) | Oct 2025 | Controlled study | A | 5 LLMs on math, QA and code. Accuracy falls 13.9–85% as input grows, even with perfect retrieval and irrelevant tokens replaced by whitespace or masked. Llama-3.1-8B: −24.2% on MMLU at 30K. Effects on closed models are more modest. "Recite then solve" gives up to +4% for GPT-4o. | Supports 6 (fresh context at boundaries). Mechanism for "context rot" beyond distraction | Yes. Magnitude shrinks on stronger models |
| H26 | METR, "Uplift update", https://metr.org/blog/2026-02-24-uplift-update/ | Feb 2026 | RCT follow-up | A (with acknowledged selection bias) | 2025 RCT: 19% slower (CI +2% to +39%). Late-2025 study: 57 devs, 800+ tasks. Returning devs −18% time (CI −38% to +9%); new devs −4% (CI −15% to +9%). 30–50% withheld tasks they would not do without AI. METR says the design no longer measures the effect reliably. | Supports 9 | Yes |
| H27 | Anthropic, Claude Code docs, "Orchestrate teams of Claude Code sessions", https://code.claude.com/docs/en/agent-teams | accessed Nov 2026 | Official docs | D | Experimental. Uses "significantly more tokens"; cost scales linearly per teammate. Best for research/review, competing-hypothesis debugging, and separate modules. For sequential tasks, same-file edits or many dependencies, use a single session or subagents. Same-file edits cause overwrites. Suggests 3–5 teammates and starting with non-coding tasks. | Supports 1 and 2 | Yes |
| H28 | Birgitta Böckeler (Thoughtworks), "Harness engineering for coding agent users", https://martinfowler.com/articles/harness-engineering.html | Apr 2026 | Practitioner essay | E | Separates computational sensors (tests, linters, types: reliable) from inferential ones (LLM review: non-deterministic). Functional-behaviour checking is "the elephant in the room". Do not rely on AI-written tests alone. No data. | Supports 3 (computation first) and 8 | Inferential parts are gen-dep; computational parts are not |

## Disagreements

1. **Multi-agent: Anthropic (H01) vs Cognition (H02, H03).**
   - Anthropic reports +90.2% on breadth-first research at about 15 times the tokens.
   - Cognition argues that context fragmentation compounds errors. Its April 2026 update narrows the claim to "single-threaded writes" and allows helper agents and reviewers.
   - Controlled evidence: H08 shows big gains only on decomposable tasks and losses on sequential ones. H10 shows that cooperating coding agents perform worse.
   - Anthropic's own docs now say the same (H27).
   - Favoured for this setting: Cognition's narrowed position. Research-style fan-out (literature, code search) is the exception, and is acceptable when the token cost is tolerable.
2. **Cursor and Carlini (H04–H06) vs CooperBench and Google (H08, H10).**
   - Cursor and Carlini show that hundreds or dozens of parallel writers can produce large artifacts.
   - The controlled studies show coordination losses.
   - How they fit: the case studies are single runs with huge budgets ($20k; "trillions of tokens"). They needed hierarchy or an external oracle, report code-quality limits (H05, H06), and do not compare against a single agent.
   - Favoured for this setting: the controlled studies.
3. **Evaluator agents: Anthropic and Cognition (H14, H03) vs the judge-bias literature (H22, H23) and Böckeler (H28).**
   - Vendors report that separate evaluators catch what self-critique misses.
   - The literature shows self-preference, weak same-family verification, and leniency. H14 itself needed several rounds of tuning against human judgment.
   - The positions reconcile: a separate evaluator helps when it has (a) a fresh context, (b) a different model or family, (c) access to execution or ground truth (H24), and (d) calibration against human judgment.
   - Favoured: an evaluator under that protocol, subordinate to deterministic checks.
4. **Compaction vs handoff: Amp (H15) vs Amp (H16); Anthropic (H14); OpenAI (H17).**
   - One vendor reversed itself within 7 months, with no data either way.
   - OpenAI trains compaction into the model.
   - Favoured: no strong evidence. Use compaction within a task and handoff files at boundaries.
5. **Harness value: HumanLayer "skill issue" (H12) and LangChain (H19) vs Vercel (H20), Anthropic (H14) and HumanLayer's later talk (H13).**
   - Harness changes can move scores 13.7 points (H19), and harness choice moves rank #5 vs #33 (H12).
   - But gains are model-specific, scaffolds get deleted (H14, H16, H20), and HumanLayer itself now says the harness is "not enough" (H13).
   - Favoured: a thin harness, plus durable verification and state.
6. **Instruction files: HumanLayer (H12) vs readers of ETH (H11).**
   - Both sides agree that short, hand-written files are fine.
   - H11 shows no significant success gain and about 20% higher cost even for developer-written files.
   - Favoured: treat the file as a cheap convention carrier, not a performance lever.
7. **Model choice for long autonomy: Cursor (H04, H05) vs Anthropic (H06, H14).**
   - Cursor found GPT-5.x more persistent than Opus 4.5.
   - Anthropic shows Opus 4.6 sustaining multi-hour runs.
   - These are claims about specific generations. Expect them to flip.

## Disconfirming evidence and null results

- **"A separate evaluator agent improves outcomes."**
  - No controlled study with frontier coding agents compares a separate evaluator against self-review at equal cost. H14 (n≈2) and H03 are vendor reports.
  - H14 concedes the evaluator was overhead for tasks the model could already solve alone, and that it was lenient out of the box.
  - H23 finds self-verification gains are minimal and same-family gains limited. A "separate agent" built on the same model is closer to self-verification than to cross-family checking.
  - H24's reviewer shares a base model with the generator. OpenAI says it cannot directly measure whether the generator learns to game the reviewer.
- **"Parallel subagents are worth their cost for a solo user."**
  - H08: mean multi-agent effect −3.5%, single-agent success per 1K tokens about 5 times hybrid.
  - H10: about −30% when coding work is split.
  - H06: adding agents did not help on a single shared bug.
  - H01: about 15 times the tokens.
  - No study addresses solo users specifically. The evidence for read-only exploration subagents is only practitioner reports (H03, H12).
- **"Keep the context 40–60% full and compact on purpose."**
  - I found no measurement behind any specific fill percentage. HumanLayer's "dumb zone" is never quantified (H12).
  - Supporting mechanism only: length alone degrades accuracy (H25); Chroma's needle-in-a-haystack results (cited in H12).
  - Contrary: Amp now auto-compacts at 90% and says frontier models handle it (H16).
  - Searched for: Chroma context-rot critiques and agent-specific replications. I found commentary only, with no replication on agent workloads.
- **"A progress file plus a fresh context beats one long context" on current models.**
  - No controlled comparison found.
  - Anthropic dropped context resets at Opus 4.5 (H14). Amp moved from handoff back to compaction (H16). OpenAI trained compaction in (H17).
  - The remaining support is Carlini's fresh-container design (H06) and Cursor's "periodic fresh starts" (H04). Both are multi-agent settings where a fresh start is structural.
- **"The instruction file is the real asset."** Directly disconfirmed for success rate by H11 (no significant gain, +20% cost). The only support is OpenAI's practitioner account of the docs directory (H07), and its numbers are unverified.
- **Big autonomy claims.**
  - The 24-hour Codex run is an internal observation with no success rate (H17).
  - Cursor's browser is "rarely completely clean", had no JavaScript at first, and has unquantified build pass rates (H05).
  - Carlini's compiler still needs GCC's assembler and linker and is slower than GCC -O0 (H06).
  - The productivity RCT shows a 19% slow-down in 2025 and inconclusive results in 2026 (H26).
- **Harness gains are not portable.** LangChain's +13.7 points is a single benchmark with no variance reported, and the authors say it is model-tuned (H19).

## Transfer notes

- **Transfers well to a solo student:**
  - A single writer with read-only helpers (H03, H27).
  - Deterministic oracles and read-only ground truth (H06, H21). For serving research these are reference outputs, token-level correctness diffs, and fixed benchmark scripts.
  - Terse, grep-able logs and fixed test subsets (`--fast`) (H06).
  - Short, hand-written instruction files (H11).
  - A cross-family reviewer on diffs, for example Claude writes and GPT/Codex reviews, or the reverse (H23, H24).
- **Transfers poorly:**
  - Cursor- and Carlini-scale swarms ($20k, about 2B tokens; H06, H04). They are far beyond $90 of Modal credit and a subscription.
  - Anthropic's 6-hour, $200 three-agent harness (H14).
  - OpenAI's million-line, zero-human-code regime (H07). It conflicts with the setting's third priority (defend every line) and with H13's 3–6 month decay warning.
- **Transfers only with adaptation:**
  - An evaluator agent: use it for write-ups and claims (does the evidence support the number?) and for patch review against maintainers' conventions. It does not replace running the benchmark yourself.
  - Parallel sessions: use them for embarrassingly parallel GPU-free work, such as reading several repos or drafting independent patches in separate worktrees. Never two agents on one patch.
- **Reviewer audience:** systems researchers judge rigour, so verification artefacts (scripts, raw logs, seeds) are worth more than harness sophistication. That aligns with Böckeler's computational-sensors-first stance (H28).

## What better models change, and what they don't

- **Already superseded, with explicit vendor statements:**
  - Context resets, removed at Opus 4.5 (H14).
  - Sprint decomposition, removed at Opus 4.6 (H14).
  - Manual handoff, replaced by auto-compaction at Amp in May 2026 (H16).
  - Heavy custom tool wrappers, replaced by bash at Vercel with Opus 4.5 (H20).
  - Integrator and judge roles, removed in Cursor's later design (H04, H05).
  - Compaction is moving into model training (H17).
- **Likely to keep shifting:** which model persists longest (H04 vs H14), cheating rates per family (H21), and whether multi-agent gains survive a stronger single agent (H08's roughly 45% saturation threshold implies gains shrink as base capability rises).
- **Durable across generations (no evidence of decay):**
  - Deterministic verification and oracles (H06, H21, H28).
  - Sandboxing and blast-radius control (H06's `pkill` incident).
  - State and results on disk (H06, H07).
  - Single-writer discipline for shared code (H03, H10).
  - Human reading of plans and diffs where accountability matters (H13).
  - Cost: verification stays much cheaper than generation (H24).
- **Self-preference and same-family blind spots** come from model similarity (H22, H23), so a different-family reviewer should stay useful even as models improve.

## Not verified

- UNVERIFIED: OpenAI harness-engineering specifics (H07): team of about 3 growing to about 7 engineers, about 1,500 PRs, about 3.5 PRs per engineer per day, AGENTS.md of about 100 lines as a "table of contents", weekly "AI slop" cleanup about 20% of time before automated garbage-collection agents, single Codex runs over 6 hours. The primary page returned 403 and the secondary pages read did not state these numbers. They come from my prior knowledge.
- UNVERIFIED: the Amp May 2026 reversal (H16) was read from a third-party mirror of the release notes, not from ampcode.com.
- UNVERIFIED: GPT-5.1-Codex-Max details (H17) and Anthropic's +39% / −84% context-management numbers (H18). Both come from search results quoting the primary pages, which I did not open.
- UNVERIFIED: CooperBench's model-specific roughly 25% figures and the 2→4-agent decline (H10). They come from the HTML version via search results; the abstract confirms only the 30% average.
- UNVERIFIED: the Pharos AI review-quality report cited in H13.
- UNVERIFIED: MAST per-mode percentages and ChatDev intervention gains (H09).
- UNVERIFIED: Vals AI's finding that Opus multi-agent teams took up to 2.3 times as long as a single agent (seen only in search snippets).

## Saturation note

- **Opened about 28 distinct sources:** H01, H03–H08, H09 (abstract), H10 (abstract), H11, H12–H16, H19–H24, H26–H28, plus METR and InfoQ. H02, H17, H18 and H25 were taken from search results.
- **Searched for:**
  - Multi-agent controlled studies (Google scaling, CooperBench, MAST).
  - Cognition and Cursor 2026 updates, Carlini, and OpenAI harness engineering.
  - Amp handoff and compaction.
  - Judge bias and cross-family verification.
  - Reward hacking (ImpossibleBench).
  - Context-length studies and critiques of Context Rot.
  - The AGENTS.md evaluation, Vercel tool removal, LangChain harness, HumanLayer 2026, and the METR follow-up.
- **Not covered:**
  - Google/DeepMind Gemini CLI, Jules and AlphaEvolve posts.
  - Manus posts after July 2025.
  - Independent critiques of Cursor's browser claim. The web-search quota for this session ran out before these searches.
- **Why I stopped:** by the last 6–8 sources the recommendations had stopped changing. Each new source restated one of the same conclusions: single writer, deterministic verification, scaffolds deleted on upgrade, no controlled evidence on fill percentages or progress-file resets.
- **Most likely to change a recommendation:** a controlled test of a cross-family reviewer on frontier coding agents, and a controlled compaction-vs-handoff comparison. I found neither.
