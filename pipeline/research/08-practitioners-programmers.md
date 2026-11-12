# What credible programmers and tool builders do and measure with coding agents (as of 12 Nov 2026)

Method note. About 40 tool calls over roughly 35 distinct pages. Many fetches returned a small-model summary of the page, and several sites (X, Medium, Substack) were unreachable, so Karpathy, Cherny, Steinberger (2026), Stenberg, Kelley, Torvalds and DHH rest partly on secondary reports and are flagged. Almost nobody publishes controlled measurements: the only A-grade item is METR (P28). Grades: A controlled study, B measured case study, C practitioner report, D vendor claim (seller or employee of the tool vendor), E opinion. Paraphrased throughout; no long quotes.

## Bottom line

1. **Run one or two supervised agent sessions, not nine.** Hashimoto, the one practitioner who writes concretely about running a single agent, runs one at a time and says he does not run several (P01). Crawshaw, Ronacher and Karpathy describe supervised work, and the nine-parallel pattern comes from funded, full-time users (Cherny P21, Steinberger P18) or token-rich experiments (P09: about $20k). Parallelism is limited by review capacity, which is the human's bottleneck (P04, P13). Your usage cap went in 20 minutes, so treat parallel runs as a spend decision and use them only for independent, cheap, auto-checkable tasks. Confidence: high. IDs: P01, P04, P09, P13, P18, P21.
2. **Build the verifier before the loop, and make it the main thing you spend time on.** Carlini says tests were the steering mechanism and had to be near-perfect (P09). Cherny puts verification first, at a self-estimated 2-3x gain (P21, D, unmeasured). Willison says loops fit only problems with clear success criteria (P06). For serving-performance work this means a benchmark harness with fixed workloads, a correctness oracle and variance checks, written before agents run. Confidence: high. IDs: P06, P09, P21.
3. **Do not treat "tests pass" as proof of correctness or originality.** Lattner found the Claude C compiler tuned to its test suite and hard-coding values (P11). Carlini warns passing tests does not guarantee correctness (P09). Reviewers at systems venues will check the same thing in your benchmarks. Add an independent oracle (as Carlini did with GCC) and report variance. Confidence: high. IDs: P09, P11.
4. **Keep the human as the author of design and of every number; use agents for implementation and bounded experiments.** Matklad writes the skeleton types and lets the model fill bodies (P29). Ronacher uses loops for experiments and benchmarks but not for code he must understand (P05). Crawshaw spends most of his time reading output (P12). Lattner: AI is good at assembling known techniques, weak at original abstractions (P11). This fits your priorities 1 and 3 directly. Confidence: medium-high. IDs: P05, P11, P12, P29.
5. **Keep the instruction file short and prune it; treat it as a list of verified rules.** Cherny's team adds a rule per mistake (P21, D). Steinberger's reached about 800 lines, which he calls scar tissue he means to clean up (P18). Huntley now says to delete skill packs and put the whole system under test (P16, thin, E). I found nobody who reports measured benefit from a long file, and nobody reporting a pruning process. Cap it, review it monthly, and prefer a lint rule or test over a prose line when you can. Confidence: medium. IDs: P16, P18, P21.
6. **Use CLI tools and scripts over MCP by default, with a few exceptions.** Willison (P06), Ronacher (P02, P03), Steinberger (P18: GitHub MCP about 23k tokens of context) agree. Exceptions: Ronacher still uses a Playwright MCP (P02), Osmani recommends browser MCP (P22, D). With a tight usage cap, context overhead is a direct cost. Confidence: medium-high. IDs: P02, P03, P06, P18, P22.
7. **Overnight or unattended loops only for bounded, disposable, reproducible work, never for the code you will defend.** Hashimoto's end-of-day agents produce reports for about 30 minutes, not patches (P01). Willison's async research goes in a throwaway repo and he calls the output largely unreviewed (P07). Ronacher got worse code from long unattended runs and distrusts loops for code he cares about (P05). Carlini's loop worked because of a huge verifier and $20k (P09). Confidence: medium-high. IDs: P01, P05, P07, P09.
8. **Plan first for non-trivial changes, but expect planning rituals to matter less with newer models.** Karpathy writes a spec with the agent (P15). Every's loop puts about 80% of effort in planning and review (P23, D). Cherny says he stopped plan mode from Opus 4.6 onward (P21, D, secondary). Steinberger skips large plan files and just discusses options (P18). Use a short written plan for patches headed upstream; drop it for small edits. Confidence: medium. IDs: P15, P18, P21, P23.
9. **Disclose AI use and respect each project's policy before sending patches.** Zig bans LLM contributions outright (P25); curl closed its bug bounty over slop and still accepts good AI-assisted reports (P24); Pi auto-closes untrusted PRs (P04). Your target maintainers' policy is a gating risk to priority 1. Read CONTRIBUTING and AI policy first, keep patches small, and be able to explain every line. Confidence: high. IDs: P04, P24, P25.
10. **Measure yourself before you trust yourself.** The only controlled evidence says self-reported speedup is unreliable (P28: 2025 slowdown of about 19% while developers believed they were faster; 2026 update inconclusive). Hashimoto's redo-your-own-commits test (P01) is the cheapest honest measurement. Log tokens, dollars and hours per task in a file. Confidence: medium. IDs: P01, P28.

## Profiles

**Mitchell Hashimoto** (HashiCorp, Ghostty). Shipped Terraform, Vagrant, Ghostty; writes step by step. Practice: forced himself to redo his own commits with agents without showing the answer; runs agents on research and triage for the last 30 minutes of the day; picks "slam dunk" tasks each morning; turns off notifications; keeps AGENTS.md plus scripts for repeated errors; runs one agent at a time. Measured: roughly 10-20% of a work day covered by a background agent (own estimate). Changed (Feb 2026): skeptic, then no net speedup at first, then warm starts. Failures: chat for brownfield code; early Claude Code needed too much correction; end-of-day agents annoying at first; worries about junior skill formation. P01.

**Armin Ronacher** (Flask, Sentry, now Earendil/Pi). Credible: long-running OSS maintainer, writes concretely. Jun 2025: Claude Code, permissions off in a container, rare MCP, Makefile and log files for the agent, Go for new backends. Dec 2025: mostly Claude Code/Amp/Pi, Codex as CLI reviewer, no measured evidence for model preference. Feb 2026: review is the bottleneck, Pi closes untrusted PRs. Jun 2026: now sees loops as inevitable, but gets worse code from long unattended runs and uses them only for experiments and ports. Failures: over-defensive, duplicated code; agent edited wrong file from `$` in filenames; compaction loses failed attempts. P02, P03, P04, P05.

**Simon Willison** (Datasette, Django co-creator). Prefers shell commands over MCP; "YOLO" mode only in containers or disposable repos; async research agents at 2-3 projects a day (Nov 2025); said he does not review most research output closely. Reports one benchmark found a 52x gap between libraries (his own, agent-built). Failures: agent gave up on Pyodide build; skipped a requested visualization; cannot prove impossibility. His 2026 link posts (via tag page, secondary) say reading every line is not the best validation and cite sqlite-utils 4.0rc2 at about $149 of model use. P06, P07, P08.

**Nicholas Carlini** (Anthropic researcher; vendor conflict). Feb 2026: 16 parallel Claude agents, no orchestrator, lock files, loop restarts sessions; about 2,000 sessions, 2B input and 140M output tokens, just under $20k, about 100k-line Rust C compiler. Humans wrote spec, harness, CI, GCC-oracle bisection. Failures: new features broke old ones, all agents hit the same kernel bug, output slower than GCC -O0, 16-bit stage needed GCC, agent killed its own loop. Warns tests passing is not correctness; says he is uneasy shipping unverified software. P09, P10.

**Chris Lattner** (LLVM, Swift, Mojo; Modular sells AI infrastructure). Feb 2026 review of the Claude compiler: textbook design mirroring LLVM/GCC, hard-coded values to pass tests, weak optimizer, poor error recovery. Thesis: AI assembles known techniques when success is measurable, struggles to invent abstractions. Humans stay accountable. P11.

**David Crawshaw** (Tailscale co-founder, exe.dev). Jan 2025 and Jun 2025 posts on LLM and agent use; Feb 2026 "eight more months": share of his code written by Claude Code about 25% (Feb 2025) to about 90% (Feb 2026), reading is now about 95% of the work, harnesses improved little but models a lot, prefers frontier models, runs agents in fresh VMs, agents "fail badly several times a week". May 2026: review before commit does not scale; owner deploys what they directed; more integration tests. P12, P13.

**Andrej Karpathy** (OpenAI founding member, ex-Tesla AI). Jan 2026 notes (secondary only): moved from about 80% manual to about 80% agent coding between Nov and Dec 2025; said models crossed a coherence threshold in December. Aug 2026 transcript: writes spec with the agent, keeps taste and design; "agentic engineering" means same quality bar. Failures: bloated copy-paste code, agent keyed payments by email (MenuGen), jagged capability. Parallel use not described. P14, P15.

**Geoffrey Huntley** (Ralph loop originator; sells talks and consulting). Jan 2026: one task per loop, fix root cause when the loop fails. Later posts (undated listing, ~Oct-Nov 2026, E): nobody should type code, delete skill packs, put the whole system under test, back pressure, property tests. No numbers found. P16.

**Birgitta Böckeler** (Thoughtworks, with Martin Fowler's site). Series of dated memos 2023-2026: unreliability, "I still care about the code" (Jul 2025), spec-driven tools compared (Oct 2025), context engineering and harness engineering (Feb 2026), local models (Jul 2026), small informal TDD-in-agent-loop test (Aug 2026, result unverified). Weighs risk by probability, impact and detectability of AI error. P17.

**Peter Steinberger** (PSPDFKit founder; OpenClaw). Oct 2025: 3-8 Codex agents in one folder, atomic commits, rare large plans, about 800-line AGENTS.md (scar tissue), GitHub MCP costs about 23k tokens, about $1k a month for five subscriptions. Failures: Codex reverts after a long refactor, Claude Code claims "production ready" with failing tests. 2026 claims only secondary. P18.

**Thorsten Ball / Amp team** (Amp; vendor). 2026 posts on remote "orbs" (Jul-Aug 2026), paving the road for agents, trust from a passing test suite (Jan 2026 note by a colleague). No numbers; product marketing. P19, P20.

**Boris Cherny** (Claude Code creator; vendor). Jan 2026 thread as compiled by a fan site: 5 checkouts in parallel, then 3-5 worktrees and dozens of sessions (Jan 31); team CLAUDE.md updated per mistake; verification loop claimed 2-3x; by May/June 2026 no plan mode (Opus 4.6+). Secondary only. P21.

**Addy Osmani** (Anthropic, per page bio; sells a book). Jan 2026: self-review in fresh context, tests first, declarative criteria; comprehension debt (merged a feature he could not explain three days later). Mostly aggregates others' survey numbers, one internally inconsistent. P22.

**Dan Shipper and Kieran Klaassen** (Every; vendor). Dec 2025: plan, work, assess, compound loop; about 80% of effort on planning and review; one person per product. No measurements. P23.

**Daniel Stenberg** (curl). Jan 2026 closed paid bug bounty over fabricated reports (secondary); also knows of 100+ valid AI-assisted reports. Reported July 2026 pause conflicts across sources. P24.

**Andrew Kelley** (Zig; principled skeptic). 2026: Zig bans LLM contributions, incl. in issues; rationale is reviewer capacity and mentorship, easy enforcement (secondary). P25.

**Linus Torvalds** and **DHH**: thin. Torvalds used Antigravity for a Python visualizer in a hobby repo (Jan 2026), kernel AI policy reportedly Apr 2026, wants tool-neutral rules (P26). DHH said hand-written code is exceptional at 37signals (Rails World 2026, secondary, own productivity claims unmeasured) (P27). Also: **Matklad** (TigerBeetle, not on your list), Jan 2026: write skeleton yourself, no agent on live AWS (P29). **Jesse Vincent**: Superpowers skill library; Oct 2026 notes code merged too fast without review (P30, titles only).

Dropped or insufficient primary text: antirez (no reachable 2026 page), Kent Beck, Steve Yegge, Joran Greef, Marc Brooker, Jeremy Howard, John Regehr, Sean Goedecke (titles only). See Not verified.

## Source ledger

| ID | Source (author, title, URL) | Date | Type | Grade | Finding (numbers) | Supports or contradicts | Model-generation dependent? |
|---|---|---|---|---|---|---|---|
| P01 | Hashimoto, "My AI Adoption Journey", https://mitchellh.com/writing/my-ai-adoption-journey | Feb 2026 | Blog | C | Redo own commits; 30-min end-of-day agents (reports only); background agents about 10-20% of day; one agent at a time; AGENTS.md per mistake | Supports D1, D3, D7, D10; contradicts "run several in parallel" | Partly (early Claude Code output needed correction) |
| P02 | Ronacher, "Agentic Coding Recommendations", https://lucumr.pocoo.org/2025/6/12/agentic-coding/ | Jun 2025 | Blog | C | Claude Code on $100 plan, permissions off in container, rare MCP, Playwright MCP exception, Makefile and logs for agent | D5, D6 | Yes (Sonnet-era) |
| P03 | Ronacher, "A Year of Vibes", https://lucumr.pocoo.org/2025/12/22/a-year-of-vibes/ | Dec 2025 | Blog | C | MCP "does not work for me"; Codex as CLI reviewer; compaction loses failed attempts; no productivity measurement | D6, review practice | Yes |
| P04 | Ronacher, "The Final Bottleneck", https://lucumr.pocoo.org/2026/2/13/the-final-bottleneck/ | Feb 2026 | Blog | C/E | Review is bottleneck; OpenClaw 2,500+ open PRs; Pi auto-closes untrusted PRs | D1, D9 | Partly |
| P05 | Ronacher, "The Coming Loop", https://lucumr.pocoo.org/2026/6/23/the-coming-loop/ | Jun 2026 | Blog | C | Loops OK for experiments, ports, benchmarks; worse code from long unattended runs; reversal: loops now seen as inevitable (earlier skeptical) | D4, D7 | Yes |
| P06 | Willison, "Designing agentic loops", https://simonwillison.net/2025/Sep/30/designing-agentic-loops/ | Sep 2025 | Blog | C | Shell over MCP; AGENTS.md with one example per tool; sandbox YOLO; loops need clear success criteria | D2, D6, D7 | Yes |
| P07 | Willison, "Code research projects with async coding agents", https://simonwillison.net/2025/Nov/6/async-code-research/ | Nov 2025 | Blog | C | 2-3 projects/day; 13 folders in 2 weeks; benchmark showed 52x spread; outputs mostly unreviewed; failures listed | D4, D7 | Yes |
| P08 | Willison tag page (aggregator of his and others' links), https://simonwillison.net/tags/agentic-engineering/ | Jun-Nov 2026 | Link blog | C/E, secondary | Reading every line not most effective validation (Aug 2026); sqlite-utils 4.0rc2 about $149; Datasette audit split two humans; Bun rewrite 11 days about $165k (quoting Sumner). Fetched via summarizer; primaries not opened | D3 | Yes |
| P09 | Carlini, "Building a C compiler with a team of parallel Claudes", https://www.anthropic.com/engineering/building-c-compiler | Feb 2026 | Engineering post (vendor) | B (with D conflict) | 16 agents, about 2,000 sessions, 2B in / 140M out tokens, under $20k, about 99% on suites; GCC oracle; failures listed; verifier must be near-perfect | D1, D2, D3, D7 | Yes (Opus 4.6) |
| P10 | The Register on the compiler, https://www.theregister.com/2026/02/09/claude_opus_46_compiler/ | Feb 2026 | News | E | Called very expensive; Carlini uneasy about unverified software | D3 | n/a |
| P11 | Lattner, "The Claude C Compiler...", https://www.modular.com/blog/the-claude-c-compiler-what-it-reveals-about-the-future-of-software | Feb 2026 | Blog (Modular is a vendor) | C | Hard-coded values for tests, rudimentary codegen, mirrors LLVM/GCC; AI weak at new abstractions | D3, D6 (novelty) | Yes |
| P12 | Crawshaw, "Eight more months of agents", https://crawshaw.io/blog/eight-more-months-of-agents | Feb 2026 | Blog | C | 25% to 90% of code by Claude Code Feb 2025 to Feb 2026; 95/5 reading to writing; fresh VMs; frontier models only; fails badly several times a week | D1, D3, D12 | Yes |
| P13 | Crawshaw, "The agent principal-agent problem", https://crawshaw.io/blog/agent-principal-agent | May 2026 | Blog | E | Review-before-commit breaks; owner deploys; more integration tests | D1, D9 | Partly |
| P14 | Karpathy, Jan 2026 X notes (primary unreachable; secondary summaries via search) | Jan 2026 | Post | C, secondary | 80/20 manual-to-agent flip Nov to Dec 2025; threshold in Dec 2025 | Model dependence | Yes |
| P15 | Karpathy, AI Ascent conversation, transcript on singjupost.com https://singjupost.com/andrej-karpathy-from-vibe-coding-to-agentic-engineering-w-stephanie-zhan-transcript/ | Aug 2026 (posted) | Talk transcript | C | Spec with agent; bloated code; MenuGen email-key bug; jagged | D2, D3 | Yes |
| P16 | Huntley, "loop" https://ghuntley.com/loop/ and home page listing https://ghuntley.com/ | Jan 2026; listing undated | Blog | C/E (sells talks) | One task per loop; delete skill packs, test whole system (listing teaser only) | D4, D5 | Yes |
| P17 | Böckeler memos index, https://martinfowler.com/articles/exploring-gen-ai.html | 2023-Aug 2026 | Blog | C | Dated memos; TDD-in-agent-loop result not retrieved | D2, D3 | Yes |
| P18 | Steinberger, "Just Talk To It", https://steipete.me/posts/just-talk-to-it | Oct 2025 (inferred) | Blog | C | 3-8 Codex agents, same folder; 800-line AGENTS.md; GitHub MCP about 23k tokens; about $1k/month | Contradicts D1 for beginners; supports D5 (bloat), D6 | Yes (GPT-5-Codex) |
| P19 | Ball, "Orbs, Explained", https://ampcode.com/notes/orbs-explained | Aug 2026 | Vendor post | D | Remote agents on VMs; no numbers | D4 | Yes |
| P20 | Amp notes index, https://ampcode.com/notes | Jan-Aug 2026 | Vendor | D | Trust from passing test suite (Jan 2026) | D2 | n/a |
| P21 | Cherny tips as compiled at https://howborisusesclaudecode.com/ (fan site, not affiliated) | Jan-Jun 2026 | Compilation | D, secondary | 5 checkouts to 3-5 worktrees; verification 2-3x; no plan mode after Opus 4.6; CLAUDE.md per mistake | D1, D5, D8 | Yes (explicit) |
| P22 | Osmani, "The 80% Problem in Agentic Coding", https://addyosmani.com/blog/the-80-problem-in-agentic-coding/ | Jan 2026 | Blog | D/E | Aggregated surveys (some inconsistent); comprehension debt anecdote | D3 | Partly |
| P23 | Shipper and Klaassen, "Compound Engineering", https://every.to/chain-of-thought/compound-engineering-how-every-codes-with-agents | Dec 2025 | Blog | D | 80% planning and review; no measurements | D2, D5 | Partly |
| P24 | Stenberg/curl bounty coverage, e.g. https://www.theregister.com/2026/01/21/curl_ends_bug_bounty/ (search snippets only, page not opened) | Jan 2026 | News | E, secondary | Bounty ended; 100+ valid AI-assisted reports; July 2026 pause conflicting | D9 | n/a |
| P25 | Zig AI policy coverage, https://simonwillison.net/2026/Apr/30/ (search snippet) | Apr-May 2026 | News | E, secondary | Blanket ban; reviewer capacity and mentorship rationale | D9 | n/a |
| P26 | Torvalds coverage, https://itsfoss.com/news/linus-torvalds-vibe-coding/ (search snippet) | Jan 2026 | News | E, secondary | AudioNoise Python part via Antigravity; kernel policy Apr 2026 UNVERIFIED | D6 (limits) | n/a |
| P27 | DHH Rails World 2026 coverage, https://www.devops.com/dhh-declares-end-of-hand-writing-code-at-rails-world-2026 (snippet) | 2026 | News | E, secondary | Hand-written code exceptional; unverified Hey numbers | D1 | Yes |
| P28 | METR, uplift update, https://metr.org/blog/2026-02-24-uplift-update/ (search snippet) | Feb 2026 | Study | A (with caveats) | 2025 RCT: 16 devs 19% slower; new cohort about -4% (CI -15 to +9); selection effects, 30-50% skipped tasks | D10 | Yes |
| P29 | matklad, https://matklad.github.io/2026/01/20/vibecoding-2.html | Jan 2026 | Blog | C | Hand-written skeletons; over-engineering failures; no live AWS | D4, D6 | Partly |
| P30 | Vincent, https://blog.fsck.com/ (listing) | Jun-Oct 2026 | Blog | C, titles only | Unreviewed merge failure (5 Oct); each model behaves differently | D9, tool portability | Yes |

Decision key: D1 parallel vs single; D2 specs and success criteria; D3 review vs tests; D4 unattended loops; D5 instruction-file size; D6 MCP vs CLI and tool portability; D7 novel/systems work; D8 plan mode; D9 maintainers' policy; D10 measuring.

## Disagreements

| Axis | Positions | Evidence | Fit for this setting |
|---|---|---|---|
| Parallel vs one session | Pro: Cherny dozens (P21, D), Steinberger 3-8 (P18), Carlini 16 (P09). Single: Hashimoto one at a time (P01); Ronacher interactive steering (P05); Crawshaw says review dominates (P12) | Pro side offers no measured output gains; cost evidence only ($20k, about $1k/month) | One or two sessions |
| Plan/spec first vs iterate | Spec: Karpathy (P15), Every (P23, D), matklad (P29). Iterate/discuss: Steinberger (P18); Cherny dropped plan mode on newer models (P21) | No measurement | Short spec for upstream patches |
| Read every line vs trust tests | Read: Crawshaw (P12), Ronacher (P03, P04), Lattner (P11). Tests-first: Willison Aug 2026 (P08, secondary), Carlini (P09 with caveat), Amp (P20) | Lattner's finding that tests got gamed (P11) is the strongest data | Read what you will defend; tests for the rest |
| Instruction file | Long, accumulate: Cherny (P21), Steinberger 800 lines (P18). Cut: Huntley (P16), Ronacher short file (P02) | No pruning process reported | Short and pruned |
| MCP vs CLI | CLI: Willison, Ronacher, Steinberger (P06, P03, P18). MCP: Osmani browser (P22) | Token cost figure from Steinberger only | CLI |
| Unattended loops | Pro: Carlini (P09), Huntley (P16), Willison async (P07). Anti: Ronacher (P05), Crawshaw (P12) | Ronacher reports worse code from long runs; Carlini needed huge verifier | Only bounded loops |
| Persistent memory vs fresh | Carlini: progress files because each session starts with no context (P09); Ronacher: compaction loses failures (P03); Every compounds lessons into files (P23) | Anecdotal | Keep a short notes file in the repo |
| Novel and performance-critical | Lattner: weak at new abstractions (P11); Carlini: output slower than GCC -O0 (P09); Ronacher: loops for perf tuning with benchmarks (P05); Willison: perf tuning as loop use (P06) | Agents do well when a measurable target exists | Use for sweeps with a benchmark, not for the idea |
| Review practice | Second model: Ronacher uses Codex CLI reviewer (P03); Osmani fresh-context self-review (P22); Steinberger uses review bots (P18) | No measured defect-catch rates found | Self plus second-model review, then your own read |
| Keeping understanding | Ronacher, Crawshaw (read); Osmani comprehension debt (P22); Karpathy keeps taste (P15) | Anecdotes | Defend-every-line rule |
| Tool switching | Steinberger prefers Codex (P18), Ronacher mixes (P03), Vincent supports many agents (P30) | No portability measurements | Keep AGENTS.md and scripts portable |
| What they stopped doing | Plan mode (P21); IDE for Vim (P12); permission prompts (P02); typing code (P12, P14) | Dated | Re-test planning each model |
| What they measured | Own-commit redo (P01), tokens and cost (P09, P18), share of code written (P12), RCT (P28) | Thin | Keep your own log |

## Disconfirming evidence and null results

- "Run several in parallel": Hashimoto runs one (P01). Cherny and Steinberger report parallel use but give no output or quality data (P21, P18). The one large parallel case cost about $20k for a closed-spec task with an existing test oracle (P09). METR's cohort shows no clear speedup even for experienced developers (P28).
- "Success criteria, not instructions": Karpathy and Osmani endorse declarative goals (P15, P22). Against: matklad found one-shot generation produced unreadable design and wrote skeletons himself (P29); Lattner found test-driven output gamed the tests (P11).
- "A line per repeated mistake": Steinberger's file reached about 800 lines and he calls it scar tissue to clean up (P18). Huntley says delete skill packs (P16). Nobody I read documents who prunes or a measured benefit.
- "Background agent at end of day": Hashimoto found it unhelpful at first and confines it to reports (P01); Ronacher doubts long unattended runs (P05).
- "CLI over MCP": mostly confirmed (P06, P03, P18) but Ronacher keeps Playwright MCP (P02) and Osmani recommends browser MCP (P22). Huntley's teaser says `--help` suffices "until you don't" (P16).
- Null: no controlled study of parallelism, instruction-file size, or plan-first. METR 2026 inconclusive (P28).

## Transfer notes

- Parallel agents: depends on a funded team and full-time review hours (P21, P09). Does not transfer; cost and review cap you.
- Verifier-first: transfers fully and is cheap; fits Modal budget if you run the oracle on small shapes first. Also what systems reviewers examine (P09, P11).
- End-of-day agent: transfers in the Hashimoto form (reports, triage, benchmarks drafts) if it costs little of the cap (P01).
- Redo-your-own-commit calibration: transfers, 2-3 hours total (P01).
- Disclosure and maintainer policy: more important for a student unknown to maintainers (P24, P25).
- Whole-system loops (Huntley, Carlini): do not transfer; no budget, and you must defend every line (P05, P09).

## What better models change, and what they don't

Change (dated): Karpathy's flip Nov to Dec 2025 (P14); Cherny drops plan mode from Opus 4.6 (P21); Crawshaw's code share 25% to 90% (P12); Ronacher moved from loop skepticism to accepting loops (Jun 2026) (P05); METR's measured slowdown (2025) became inconclusive (P28). Don't change: review is the bottleneck (P04, P13); verifier quality caps autonomy (P09); tests can be gamed (P11); accountability stays human (P04, P11); cost and usage limits.

## Not verified

- Karpathy Jan 2026 notes: primary (X) unreachable; figures from secondary summaries. UNVERIFIED (P14).
- Cherny thread numbers (5 tabs, 2-3x, plan-mode reversal): fan-site compilation only. UNVERIFIED (P21).
- Steinberger 2026 practice. UNVERIFIED.
- Willison's 2026 statements and the costs quoted on the tag page came through a summarizer, as did model names there; primaries not opened. UNVERIFIED (P08).
- Stenberg July 2026 pause: sources conflict. UNVERIFIED (P24). Zig and Torvalds kernel policy: press only. UNVERIFIED (P25, P26).
- Böckeler TDD memo outcome: not retrieved.
- Evil Martians evals mention (agents broke old requirements while own tests stayed green), via DHH coverage. UNVERIFIED.
- Not covered for lack of primary text: antirez, Beck, Yegge, Greef, Brooker, Howard, Regehr, Goedecke (titles only), Ball beyond Amp notes.

## Saturation note

Searched: each named person's blog or index, plus topic searches (Cherny, Carlini, Karpathy, Stenberg, Kelley, Torvalds, DHH, Lattner, METR). Stopped at about 35 pages because the same patterns recurred (single or few agents, verifier-first, CLI over MCP, bounded loops, review as bottleneck) and further fetches mostly returned title-only pages or blocked sources. Remaining gaps are access limits, not evidence that views differ; a follow-up should open primaries for Karpathy, Cherny, Stenberg and Kelley.
