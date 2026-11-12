# 05: What systems evaluators count as rigorous, original and honest (state as of 12 Nov 2026)

Scope: artifact-evaluation norms, how systems papers are reviewed, academic AI policies, how to disclose AI use, what faculty want from prospective students, and how to stay able to defend AI-assisted work. The aim is to apply all of this to one MSc student who has nine agent-built LLM-serving projects, none yet run on a GPU, and who is applying for a research-assistant (RA) post at a systems lab.

Grades: A = controlled study; B = measured case study; C = practitioner report; D = vendor claim; E = opinion. Rows typed "policy" are graded by how binding they are for this student: **binding** (applies directly), **norm** (applies only by analogy), or **reference**.

## Bottom line

1. **Present 2 or 3 projects in depth, not 9 in breadth. Each one should rest on a single claim measured on real hardware. Move the other projects to a short "open ideas" list.** Confidence: medium. Basis: reviewers judge whether an idea is significant, not how much work went into it (V29). vLLM's guide asks contributors to avoid one-off busywork PRs and to bundle mechanical changes (V21). Faculty reward specific, verifiable engagement over volume (V32, V33, V31). Rejection modes seen only in agent-written PRs include "too large" and "no added value" (V25). arXiv shut its CS category to unreviewed surveys after a flood of LLM-made papers (V14). No controlled study ties volume to hiring outcomes (see Disconfirming).
2. **Package each showcased project as a "mini-artifact" at the SOSP/EuroSys 2026 Functional and Reproduced level. Do not aim for Reusable.** The package contains:
   - a README that maps each claim to the script that tests it;
   - pinned environments, with no `latest` tags;
   - a kick-the-tires step that runs in a few minutes;
   - a time and resource estimate for each experiment, including GPU-hours and dollars;
   - one script per experiment and one script per figure or table;
   - the raw logs.

   Confidence: high that this is the current norm (V02, V03, V04, V05, V06). Medium that it raises RA hiring odds: no direct evidence was found, but the target lab publishes at the venues that use this process (V35).
3. **Report no number that was not measured. Mark predictions as predictions. Build every evaluation against Heiser's list of benchmarking crimes.** That means:
   - the baseline is current upstream, not an old version or your own earlier code;
   - variance is reported;
   - absolute numbers appear next to ratios;
   - the platform is specified in full;
   - overhead is reported as latency or CPU cost, not only as lost throughput;
   - cases where the change makes things worse are shown.

   Confidence: high (V28, V29, V03). Evaluators check that the code does what the write-up claims, not only that the numbers match (V03).
4. **Disclose AI use in every deliverable: one specific, process-based line saying which tool did what and what the human verified.** Use a methods or acknowledgement line in write-ups, a note in each PR description, and a commit trailer. Do not use a prominent banner saying "AI-generated". Confidence: high that disclosure is required (V15, V16, V19, V21, V12). Medium on the framing:
   - a label read as "AI drafted, human refined" had no significant penalty, whereas "AI did everything" did (V24, A);
   - disclosure still costs some trust, but being found out costs more (V22, A);
   - AI users are rated as less competent in hiring tasks (V23, A).
5. **Open-source patches must follow the target repository's AI policy word for word.** For vLLM that means:
   - no "pure agent" PRs;
   - the human reviews every changed line, validates behaviour end to end and runs the relevant tests;
   - the PR description discloses AI use;
   - commits carry a trailer such as `Co-authored-by:` or `Assisted-by:`;
   - the student is personally ready to answer review comments.

   Confidence: high (V21, V19, V20).
6. **Be able to defend every line and every number.** Practices:
   - write the write-up and the outreach email yourself;
   - re-derive each headline number from the raw logs by hand;
   - rehearse explaining the design and one surprising number without notes;
   - be ready to re-run kick-the-tires live.

   Confidence: medium. The rule itself is explicit: the kernel says submitters must be able to defend everything they submit (V19), and USENIX makes every author accountable for the whole work (V10). No study was found showing which practice works best.
7. **ETH rules: if any project becomes graded work or a thesis, agree the AI use with the supervisor in advance and tick the second declaration option.** That option reads: GenAI use as a tool and its source declarations "agreed with the supervisor". Confidence: high (V16, V17). An RA application is not itself an assessed piece of work, but ETH's stated principle is transparency, and failing to disclose AI use in assessed work can lead to disciplinary action (V15, V17).
8. **Write the outreach personally. Make it short, about one specific paper from the lab, and link to one artifact.** Confidence: high. Faculty say they ignore AI-written emails; one reports about 20 a day (V32). Others ask for one specific question or extension of a named paper (V33), and call generic summaries counterproductive (V34, V31).
9. **Frame the contribution honestly as measurement or engineering, not as research novelty.** Validating, extending or refuting an earlier design through implementation is a valued contribution. A small improvement on an established technique belongs in a short note, not a paper-sized claim (V29). Confidence: medium.

## Source ledger

| ID | Source (author, title, URL) | Date | Type | Grade | Finding (with numbers or exact requirement) | Supports or contradicts which decision | Model-generation dependent? |
|---|---|---|---|---|---|---|---|
| V01 | sysartifacts maintainers, "sysartifacts" index, https://sysartifacts.github.io/ | read Nov 2026 | policy (index) | reference | Lists AE pages for ATC 2022–26, EuroSys 2021–27, OSDI 2020–25, SOSP 2019–26, FAST 2024–25 and SC 2021. **NSDI is not listed.** There is no OSDI 2026 page. | Supports 2 (where the norms live) | No |
| V02 | sysartifacts, "Packaging guide", https://sysartifacts.github.io/packaging-guide.html | read Nov 2026 | policy (guide) | norm | Container or VM preferred for non-trivial dependencies, with the Dockerfile included. A Zenodo image does not replace the build recipe. Common mistakes: `latest` tags or unpinned `apt-get`; instructions never tested on a clean machine; manual rather than automated experiments; misleading logs. Recommends measuring resource use. | Supports 2 | No |
| V03 | sysartifacts, "Evaluator guide", https://sysartifacts.github.io/evaluator-guide.html | read Nov 2026 | policy (guide) | norm | Kick-the-tires plan Q1–Q8 covers the claims, the mapping from artifact to claim, code comprehension, hardware, installation, a simple run and the expected results. Evaluators check that the artifact "must actually do what it claims to do", so matching numbers is not enough. A badge may be denied when reproducing a claim needs unreasonable manual effort. | Supports 2, 3 | No |
| V04 | SOSP 2026 AEC (Voulimeneas, Pirelli), "SOSP 2026 Artifact Evaluation", https://sysartifacts.github.io/sosp2026/ | 2026 (deadlines Jul–Aug 2026) | policy (CfA) | norm | Badges: Available, Functional, Reproduced. **No Reusable badge.** Process: register, submit, kick-the-tires (13–27 Jul 2026), main evaluation, archive with DOI. Single-blind and cooperative. Artifacts with all badges can win the Distinguished Artifact award. No AI policy. | Supports 2; contradicts "aim for Reusable" | No |
| V05 | SOSP 2026 AEC, "Badges" checklist, https://sysartifacts.github.io/sosp2026/badges | 2026 | policy (CfA) | norm | Available: permanent archive such as Zenodo ("not GitHub"), an open licence, a README that references the paper. Functional: the README describes each component and how it links to the paper, the exact environment, a minimal working example, and expected resources per experiment (example: "5 minutes, 10 GB of disk space"); also a precise dependency list, an example input for each experiment type, and no value entered in more than one place. Reproduced: one script per experiment, one script that prints results close to the paper's format, and active time of "a few minutes" apart from debugging. | Supports 2 (exact checklist) | No |
| V06 | EuroSys 2026 AEC, "Badges", https://sysartifacts.github.io/eurosys2026/badges | 2026 | policy (CfA) | norm | Same three combinations: A+F+R, A+F, F+R. **No Reusable badge.** Same requirements as V05. | Supports 2 | No |
| V07 | EuroSys AE chairs 2021–25 (curated by T. D. Doudali), "Lessons from Five Years of Artifact Evaluation at EuroSys", https://www.sigops.org/2025/lessons-from-five-years-of-artifact-evaluation-at-eurosys/ | Aug 2025 | case report | B | Over 2021–25: 161 Available, 136 Functional and **75 Reproduced** badges. About 58% of accepted papers took part each year, and the AEC peaked at 98 members. Problems: AE is separate from paper review, specialised hardware is needed, badge definitions are imprecise. Recommends that authors state at submission which results are reproducible and how. | Supports 2 (being reproducible stands out); also disconfirms "AE is universal" | No |
| V08 | SC25 Reproducibility Initiative, https://sc25.supercomputing.org/program/papers/reproducibility-initiative/ | 2025 | policy (CfP) | norm | An Artifact Description (AD) appendix is **mandatory** for every SC technical paper (since 2019) and is now generated from a form. The Artifact Evaluation (AE) appendix is optional but strongly encouraged. Automated reproduction is explicitly rewarded. Chameleon Cloud supports artifact sharing. | Supports 2 (the HPC norm: always describe the environment) | No |
| V09 | USENIX, "OSDI '26 Call for Papers", https://www.usenix.org/conference/osdi26/call-for-papers | ~Dec 2025 (PDF filename suggests 15 Dec 2025) | policy (CfP) | norm | Submitting work "wholly or largely generated by AI" counts as fraud. AI editing tools are acceptable. Papers are judged on novelty, significance, interest, clarity, relevance and correctness, and should "Draw appropriate conclusions". | Supports 4, 6; warns against agent-written write-ups | Yes (new AI clause, 2025–26) |
| V10 | USENIX Board, "Authorship Policy", https://www.usenix.org/node/321108 | 29 Jun 2026 | policy | norm | AI cannot be an author. There is no organisation-wide AI-use policy; each CfP decides. Every author must be "accountable for the work in its entirety" and must answer questions about its integrity. Sanctions: rejection or retraction, or an erratum for minor issues. | Supports 6 | Yes (new in 2026) |
| V11 | ACM, "Policy on Authorship" and FAQ, https://www.acm.org/publications/policy-on-authorship (403 to the fetcher; content from search snippets of ACM FAQ, RESPECT and SIGCHI pages) | FAQ current 2025–26 | policy | norm | GenAI may be used if disclosed in the **Acknowledgements**. Tools cannot be authors. Basic spelling and grammar checking is exempt. Authors are responsible for the "veracity and correctness" of all material. The SIGCSE TS 2027 page claims ACM no longer requires disclosure of AI help with writing (see Disagreements). PARTLY UNVERIFIED. | Supports 4 | Yes (changed 2023 to 2026) |
| V12 | ICLR PCs, "Policies on LLM usage at ICLR 2026", https://blog.iclr.cc/2025/08/26/policies-on-large-language-model-usage-at-iclr-2026/ | Aug 2025 | policy | norm | **Any** LLM use must be disclosed in the paper and in the submission form. Authors must verify any research contribution an LLM makes. A falsehood produced by an LLM is a Code of Ethics violation by the authors, and desk rejection is possible. Hidden prompt injections aimed at LLM reviewers count as collusion. | Supports 4, 6 | Yes (stricter than 2024) |
| V13 | NeurIPS, "LLM Policy" (2025), https://neurips.cc/Conferences/2025/LLM | 2025 | policy | norm | Disclose LLM use only when it is an "important, original, or non-standard" part of the method. Writing or editing use needs no declaration. Authors are responsible for all text, figures and references. Publication status can be revoked after the fact, for example over unverified references. | Contradicts strict "disclose all"; supports 4 (be specific) | Yes |
| V14 | arXiv CS moderators, "Updated practice for review articles and position papers in arXiv CS", https://blog.arxiv.org/2025/10/31/attention-authors-updated-practice-for-review-articles-and-position-papers-in-arxiv-cs-category/ | 31 Oct 2025 | policy | norm | CS review and position papers now need **completed peer review** at a journal or conference; workshop review does not count. The stated cause is an "unmanageable influx" made easier by LLMs, with most reviews "little more than annotated bibliographies". | Supports 1, 9 (volume of LLM-made text is now penalised) | Yes (2025 change) |
| V15 | ETH Zurich, "AI in Teaching and Learning", https://ethz.ch/en/the-eth-zurich/education/ai-in-education.html | updated Dec 2024 ("clarifications… regarding the declaration of GenAI") | policy | binding (ETH studies) | Exact wording: "You are responsible for the content of work you submit." Students should agree with the lecturer whether and how GenAI may be used and how to declare it, and should state clearly which tools were used for which part. Assessments must be done "independently and personally". Failing to disclose can lead to disciplinary action. | Supports 4, 7 | Yes (Dec 2024 update) |
| V16 | ETH Zurich, "Academic integrity" (AI in education), https://ethz.ch/en/the-eth-zurich/education/ai-in-education/academic-integrity.html | page © 2026; declaration PDFs undated | policy | binding (theses, graded work) | The declaration of originality has two AI options: (1) "Generative AI technologies were *not* used"; (2) "The use of generative AI as a tool and its source declarations has been agreed with the supervisor." Also: "The creator of content is always responsible for its correctness and quality." Undisclosed use is subject to discipline. | Supports 7, 4 | Yes (options added 2023–24) |
| V17 | ETH Rectorate (Unit for Teaching and Learning), "Generative AI in Teaching & Learning: Guidelines" (PDF), https://ethz.ch/content/dam/ethz/main/eth-zurich/education/ai_in_education/Generative%20AI%20in%20Teaching%20and%20Learning%20-%20Guidelines%20ETH.pdf | Dec 2024 | policy | binding | Principles: responsibility, transparency, fairness. "Clearly declare which tools you have used for which part of your work". GenAI may only play a supplementary role. Students must not put confidential information into commercial GenAI tools unless expressly permitted. Legal basis: the existing assessment rules and the declaration of originality. | Supports 4, 7 (and: no lab-confidential code into commercial tools without permission) | Yes |
| V18 | SNSF, "The SNSF's approach to the use of AI in funding applications", https://snf.ch/en/Gz1VrJlkLyay4H7z/news/the-snsfs-approach-to-the-use-of-artificial-intelligence-in-funding-applications | 28 Feb 2024 | policy | reference | Applicants are the sole authors and carry "full responsibility for the content". The page states no disclosure duty. Putting whole dossiers or new ideas into GenAI may break confidentiality. A separate 2025 SNSF reviewer AI guideline exists but was not opened. | Supports 6; weak on 4 | Yes |
| V19 | Linux kernel docs, "Generated content" (process doc), https://docs.kernel.org/process/generated-content.html | kernel 7.4-rc2 (late 2026) | policy | norm | Covers any meaningful tool-generated content. Disclose tools, inputs, a summary of prompts, which parts the tool affected, and how it was tested. "If in doubt, choose transparency." Submitters must be able "to defend everything you submit". Maintainers may reject undefended series without review or give them extra scrutiny. | Supports 4, 5, 6 | Yes (2025–26) |
| V20 | Linux kernel docs, "AI Coding Assistants", https://docs.kernel.org/process/coding-assistants.html | kernel 7.4-rc2 | policy | norm | AI agents "MUST NOT add Signed-off-by tags"; only a human can certify the DCO. The human reviews all AI code, checks licensing and takes full responsibility. No specific tag name was visible on the fetched page. | Supports 5 | Yes |
| V21 | vLLM project, "Contributing to vLLM", https://docs.vllm.ai/en/latest/contributing/ | footer 5 Nov 2026 | policy | **binding** for vLLM patches | AI section: no "pure agent" PRs; review all changed lines, validate end to end, run the relevant tests; mention AI-generated code in the PR description; add commit trailers (`Co-authored-by:`; other projects use `Assisted-by:` or `Generated-by:`); same quality bar as other code. Avoid one-off "busywork" PRs and bundle mechanical cleanups. Contributors without a GPU may rely on CI for tests. DCO `Signed-off-by` required. | Supports 1, 4, 5 | Yes |
| V22 | Schilke & Reimann, "The transparency dilemma: How AI disclosure erodes trust", OBHDP vol. 188 (via University of Arizona news, https://news.arizona.edu/employee-news/being-honest-about-using-ai-work-makes-people-trust-you-less-research-finds) | May 2025 | controlled experiments | A | 13 experiments, more than 5,000 participants (students, legal analysts, hiring managers, investors). Disclosing AI use lowers trust. The penalty held whether disclosure was voluntary or mandatory. Being **exposed** by a third party caused the steepest drop. A favourable view of technology only "reduced the effect slightly". Effect sizes were not available to me; the journal page was blocked. | Contradicts "disclose prominently"; supports disclosing over hiding (4) | Possibly (authors say it is unclear whether the penalty fades as use becomes normal) |
| V23 | Reif, Larrick & Soll, "Evidence of a social evaluation penalty for using AI", PNAS 122 (doi 10.1073/pnas.2426766122), via search summaries (PNAS and PMC blocked) | May 2025 | controlled experiments (4 preregistered) | A | N = 4,439. People who use AI expect, and receive, lower ratings of competence and motivation. In an incentive-compatible hiring task, managers who do not use AI themselves acted on that penalty. A press report says the penalty shrank when AI clearly fit the task (UNVERIFIED). | Contradicts "disclose prominently"; supports 4 (frame by task fit and human verification) and 6 | Possibly |
| V24 | Altay & Gilardi, "People are skeptical of headlines labeled as AI-generated, even if true or human-made…", PNAS Nexus 3(10) pgae403, https://academic.oup.com/pnasnexus/article/3/10/pgae403/7795946 | Oct 2024 | controlled experiments (2 preregistered) | A | N = 4,976 (US and UK). An "AI-generated" label cut perceived accuracy by 2.66 percentage points, about a third of the 9.33-point effect of a "False" label. The effect appeared when the label was undefined or meant AI wrote everything, and was **not significant** when the label meant AI improved clarity or wrote a first draft. Mechanism: readers assume no human was involved. | Supports 4 (process-specific framing reduces the penalty) | Possibly |
| V25 | Nakashima, Ishimoto, Kondo, McIntosh, Kamei, "Why Agentic-PRs Get Rejected", arXiv 2602.04226, https://arxiv.org/html/2602.04226v1 | Feb 2026 | measured case study | B | AIDev-pop: about 37.4k PRs; 654 rejected PRs coded by hand. Merge rates (as summarised in search results): Codex 85.8%, Cursor 74.6%, Claude Code 71.3%, humans 82.6%. 67.9% of rejections came with no feedback. Rejection modes seen only in agent PRs: too large (1.5%), no added value (1.1%), more complex than warranted, distrust of AI code (0.2%). | Supports 1, 5; partly contradicts "AI PRs are distrusted" (explicit distrust is rare) | **Yes** (rates are per agent and per period) |
| V26 | Li, Zhang & Hassan, "The Rise of AI Teammates in SE 3.0" (AIDev), arXiv 2507.15003, https://arxiv.org/abs/2507.15003 | Jul 2025 | measured case study | B | More than 456k PRs from five agents across 61k repositories. The abstract says agent PRs are "accepted less frequently" than human PRs. Per-agent rates (Codex 64%, Devin 49%, Copilot 35% vs human 76.8%) come from search snippets only. | Supports 5 | **Yes** |
| V27 | GPTZero, "NeurIPS hallucinated citations" (https://gptzero.me/news/neurips/) and arXiv 2602.05930 (independent re-analysis), via search summaries | Jan–Feb 2026 | vendor report plus case study | D (GPTZero), B (2602.05930) | At least 100 hallucinated citations in about 51–53 of 4,841 accepted NeurIPS 2025 papers (about 1%). About 50 in ICLR 2026 submissions under review. NeurIPS and ICLR treat these as grounds for rejection or retraction. The vendor sells a detection tool. | Supports 3, 6 (unverified AI output is a public cost) | **Yes** |
| V28 | Gernot Heiser, "Systems Benchmarking Crimes", https://gernot-heiser.org/benchmarking-crimes.html | updated after Apr 2025 | practitioner guide | C | Lists crimes in five groups: selective benchmarking, improper handling of results, wrong benchmarks, improper comparison, missing information. Examples: no proper baseline; evaluating only against yourself; an outdated baseline; ratios without absolute numbers; no variance; a missing platform spec; equating throughput loss with overhead (seen in "probably 10%" of papers he reviews). | Supports 3 | No |
| V29 | Roy Levin & David Redell, "An evaluation of the ninth SOSP submissions, or How (and how not) to write a good systems paper", https://www.usenix.org/legacy/events/samples/submit/advice_old.html | 1983 (still circulated by USENIX) | opinion (SOSP PC chairs) | E | Asks "Can you state the new idea concisely?" Significance is judged, not effort. Validating, extending or refuting earlier designs through implementation is valuable. Small improvements suit a short note. Comparisons must be explicit. State assumptions and do not generalise from one example. Ideas that failed are instructive. | Supports 1, 3, 9 | No |
| V30 | Timothy Roscoe (ETH Zurich), "Writing reviews for systems conferences", https://people.inf.ethz.ch/troscoe/pubs/review-writing.pdf | Mar 2007 | opinion | E | Reviewers summarise, then state the contributions. They check novelty against related work, technical flaws (phrased as questions), gaps, and fit with the venue. Conference papers report "complete, mature work"; workshop papers report early work or a position. | Supports 9 (match claims to maturity) | No |
| V31 | Phil Koopman (CMU), "SinceritySlop and the death of personalized communication", https://philkoopman.substack.com/p/sincerityslop-and-the-death-of-personalized | 10 May 2025 | practitioner report | C | An email from a prospective student cited his work well but contained a hallucinated URL. He now treats such mail as probable spam, so genuine mail gets lost with it. | Supports 8, 3 | Yes |
| V32 | Ran Blekhman (UChicago), "Why I am not reading your AI-generated email", https://blekhman.substack.com/p/why-i-am-not-reading-your-ai-generated | 4 Oct 2026 | practitioner report | C | Receives about 20 lab-application emails a day, "almost all" AI-generated and alike. Wants the applicant's own voice, specific reasons for fit, and verifiable engagement with the lab's work. He is not anti-AI: the line is polishing your own writing versus avoiding the thinking. Cites a study in which cover letters predicted callbacks 51% less well after an AI writing tool appeared (secondary; UNVERIFIED). | Supports 8, 4 | Yes |
| V33 | David Evans (UVA), "Advice on Contacting Faculty", https://uvasrg.github.io/prospective/ | undated | opinion | E | Read at least one recent paper first; raise a specific question or extension; about one screen of text; no attachments, no flattery, no generic interest. Keep a personal website that shows your work. | Supports 8, 1 | No |
| V34 | Saúl Blanco (Indiana), "Advice for Prospective Research Students", https://homes.luddy.indiana.edu/sblancor/prospective_students.html | undated (post-2023, mentions GenAI) | opinion | E | GenAI makes summaries of a professor's work "fairly easy". Mass emails with "esteemed" or "respected" are counterproductive. "DON'T MASS EMAIL PEOPLE." | Supports 8 | Yes |
| V35 | Ana Klimovic (ETH, EASL), lab page, https://anakli.inf.ethz.ch/ | read Nov 2026 | public example | C | Contact: email with a CV. Recent serving and serverless work at the target venues: Dirigent (SOSP 2024), Dandelion and Sailor (SOSP 2025), DeltaZip (EuroSys 2025), ThunderServe (MLSys 2025), OpenTela (OSDI 2026). The group includes RAs and MSc students. | Supports 2 (the lab lives in the AE-badging venues), 8 | No |
| V36 | arXiv 2512.00651, "LLMs for Software Engineering: A Reproducibility Crisis" (search snippet only) | Dec 2025 | measured case study | B | Share of LLM4SE papers with badges: under 8% before 2023, 9.8% in 2024, 20.1% in 2025. 41.7% of badged 2024 papers still had versioning problems. | Supports 2 (pin versions; a badge alone is not enough) | Yes |
| V37 | arXiv 2601.17036, quantitative evaluation of the arXiv CS review-paper policy (search snippet only) | Jan 2026 | measured case study | B | Review papers carry more LLM-generated content, but flagged non-review papers are far more numerous. The policy would remove about 50% of cs.CY papers but about 3% of CV papers. | Nuances V14 | Yes |

Opened directly: V01–V10, V12–V22, V24, V25 (abstract and body), V26 (abstract), V28–V35. Search snippets only: V11, V23, V27, V36, V37.

## Disagreements

- **How much AI use must be disclosed.**
  - ICLR 2026 requires disclosure of *any* LLM use (V12). The kernel says "if in doubt, choose transparency" (V19). ETH asks students to state which tool was used for which part (V15, V17).
  - Against that: NeurIPS 2025 exempts writing and editing use (V13), OSDI '26 explicitly allows AI editing tools (V09), ACM exempts grammar checking (V11), and the SIGCSE 2027 page claims ACM dropped disclosure for writing help (V11, unverified).
  - **For this setting the stricter side wins.** The binding rules (ETH V15–V17, vLLM V21) both require disclosure, and agent-written code is "methodology", not editing, under every one of these policies.
- **Disclosure helps trust (policy view) versus disclosure costs trust (experimental view).**
  - Three A-grade studies show a penalty: on trust (V22), on perceived competence and in hiring (V23), and on perceived accuracy (V24).
  - The same evidence shows that being exposed is worse than self-disclosing (V22), and that a label describing a human-led process carries no significant penalty (V24).
  - **The evidence favours disclosure that is specific and process-based** ("Claude Code drafted X; I designed Y, reviewed every diff, measured Z"), not a prominent "AI-generated" label. Leaving it out is not an option: it violates V15, V16 and V21, and exposure is the worst outcome (V22).
- **Are agent-written PRs merged less often?**
  - AIDev says yes (V26).
  - Nakashima et al. find Codex PRs merged *more* often than human PRs (85.8% vs 82.6%) in popular repositories, and explicit "distrust of AI" is only 0.2% of rejections (V25).
  - Rates depend on the subset and the agent. **What looks robust:** size, lack of added value and unnecessary complexity are failure modes specific to agents (V25), which is consistent with V21's "busywork" rule.
- **Should artifact evaluation be mandatory?**
  - The EuroSys chairs propose making AE mandatory and tying it to acceptance (V07).
  - In practice AE is opt-in at SOSP and EuroSys (V04, V06), only about 58% of papers take part, and fewer than half of the Available badges were matched by a Reproduced badge (75 of 161) (V07).
  - SC already makes the artifact description mandatory (V08).
  - So reproducibility still distinguishes a submission rather than being the baseline.
- **Do AI-drafted emails fail because they are AI or because they are generic?**
  - Blekhman refuses AI-generated emails as such (V32).
  - Koopman (V31) and Blanco (V34) object mainly to generic text and hallucinated details. Commercial advice pages argue that tailoring is what matters, not the tool (not ledgered, D/E).
  - For a student who must also show their own understanding (decision 6), writing it personally costs little and removes the risk.

## Disconfirming evidence and null results

- **"Disclose all AI use prominently."** Partly disconfirmed. A-grade evidence shows a real trust and competence penalty for disclosure (V22, V23, V24). The penalty survived voluntary or mandatory framing and positive attitudes to technology (V22). The only framing found to remove the penalty was defining the label as AI assisting a human-led process (V24). This comes from a headline-credibility setting, so transferring it to hiring is an inference. The Reif et al. task-fit moderator is UNVERIFIED (V23). Net result: disclose, but specifically and without a banner.
- **"More projects and patches are more impressive."** No controlled study of RA or PhD hiring was found either way. The search budget ran out before a dedicated search could be made. The indirect evidence all points against volume:
  - significance over effort (V29);
  - the busywork rule (V21);
  - agent-specific "no added value" and "too large" rejections (V25);
  - arXiv's reaction to LLM-driven volume (V14);
  - faculty valuing one specific, verifiable engagement (V32, V33).

  Against that: V35 shows the target lab publishes many papers a year, so output volume is valued at the *lab* level. The question is per-person depth.
- **"Artifact-style packaging matters for an RA application."** No direct evidence was found; no faculty source in this set mentions artifacts in RA hiring. Further cautions:
  - even accepted papers at top venues often lack a Reproduced badge (V07);
  - **Reusable is not offered** at SOSP 2026 or EuroSys 2026 (V04, V06), so polishing for "Reusable" is wasted effort;
  - badges do not guarantee clean versioning (V36).

  The case for a *light* package rests on two transfer arguments: the target lab's venues use the checklist (V35), and evaluators value code that "does what it claims" (V03). It also helps the student defend their numbers (V19).
- **Null result.** ETH has no separate RA-specific AI rule. The binding ETH text covers teaching, assessment and theses (V15–V17). An ETH research-integrity page specific to GenAI was not found as a primary source; secondary sources (UZH, PSI and Lib4RI summaries) say ETH research guidance requires disclosure. UNVERIFIED.

## Transfer notes

- **Carries over from full papers to short write-ups and patches:**
  - claim-to-evidence mapping and one script per number (V05);
  - the benchmarking-crime checklist (V28);
  - explicit comparison with current upstream (V29, V28);
  - stated assumptions and limits (V29);
  - environment specification, which SC makes mandatory (V08);
  - personal accountability for everything submitted (V10, V19).
- **Carries over in reduced form:**
  - kick-the-tires becomes a CPU or mock smoke test of under 5 minutes;
  - "Available" becomes a tagged commit, or a Zenodo DOI if cheap (V05);
  - expected resources become the GPU type, minutes and dollars per experiment, which matters on a $90 Modal budget.
- **Does not carry over:**
  - full Reproduced status across every figure;
  - Reusable polish;
  - the novelty bar of a full paper. A measurement or engineering note is judged as a short note (V29, V30).
- **Patches.** The repository's own policy (V21) outranks academic norms. Maintainers review the code itself, not artifacts.
- **Assessed work.** If the work becomes an MSc thesis or project at ETH, V16's declaration options apply formally and the use must be agreed in advance.

## What better models change, and what they don't

- **Changes with model generation:**
  - agent PR merge rates, which already differ widely by agent (V25, V26);
  - how easy AI-written emails and text are to spot (V31, V32, V34);
  - the rate of hallucinated citations and URLs (V27, V31);
  - possibly the size of the disclosure penalty as AI use becomes normal (V22 calls this an open question).
- **Policies changed between 2024 and 2026, so check again before each submission:**
  - ETH update, Dec 2024 (V15);
  - ICLR 2026, stricter (V12);
  - arXiv CS moderation, Oct 2025 (V14);
  - OSDI '26 AI-fraud clause (V09);
  - USENIX Authorship Policy, Jun 2026 (V10);
  - kernel generated-content docs (V19);
  - vLLM AI section (V21).
- **Does not change:**
  - human accountability for every line and number (V10, V15, V16, V19, V21);
  - the requirement that a number be measured on stated hardware against a fair baseline (V28, V03);
  - evaluators checking that code does what is claimed (V03);
  - the need to explain the design in person (V19).

  These are properties of the human and of the evidence, not of the tool. They are what keeps a portfolio valid through switches of model or tool (priority 4 in the setting).

## Not verified

- UNVERIFIED: the exact text of ACM Artifact Review and Badging **v1.1** and its "Reusable" definition. acm.org returned 403. From memory, Reusable means documented and structured well beyond minimal function so that reuse and repurposing are easy. That memory was not checked.
- UNVERIFIED: the exact ACM Policy on Authorship and FAQ wording on GenAI (V11), and the SIGCSE claim that ACM dropped disclosure for writing help.
- UNVERIFIED: the Reif et al. effect sizes and the task-fit moderator (V23). PNAS and PMC were blocked.
- UNVERIFIED: the Schilke & Reimann effect sizes (V22). The journal page was blocked; the numbers come from the authors' university article.
- UNVERIFIED: per-agent merge rates in AIDev (V26) and the AIDev-pop figures (V25), which come from search summaries.
- UNVERIFIED: the GPTZero counts (V27; vendor claim). The re-analysis in arXiv 2602.05930 was not opened.
- UNVERIFIED: the OSDI 2026, NSDI 2025/26 and ATC 2026 calls for artifacts. They were not on sysartifacts or not opened, and NSDI is absent from sysartifacts.
- UNVERIFIED: ICML 2026 LLM policy (URL returned 404); IEEE author AI policy (not opened); the SNSF 2025 reviewer guideline and handbook (not opened); ETH research-integrity guidance specific to GenAI (secondary sources only).
- Not opened: S. Keshav's writing advice, MLSys reviewer guidelines, and studies of how labs probe understanding (whiteboard, live reproduction). Decision 6 therefore rests on policy text (V10, V19), not on studies.

## Saturation note

**Searched or opened:**
- sysartifacts: index, packaging and evaluator guides, SOSP 2026 and EuroSys 2026 badges, ATC 2025;
- the EuroSys AE retrospective and SC25;
- OSDI '26 CfP, the USENIX authorship policy, ACM (blocked), ICLR 2026, NeurIPS 2025, ICML 2026 (404) and the arXiv blog;
- ETH: the AI-in-education page, the academic-integrity page and the guidelines PDF;
- SNSF;
- the kernel and vLLM contribution policies;
- three A-grade disclosure studies;
- two agentic-PR studies;
- Heiser, Levin & Redell, and Roscoe;
- four faculty pages or posts on outreach;
- the EASL lab page.

That is about 37 sources, 32 of them opened directly.

**Why I stopped:** The last six sources (V24–V27, V32–V34) refined the framing of decisions 4 and 8 but did not change any decision. The tool's per-turn web-search budget was also exhausted; later sources were fetched only from URLs already known. Remaining gaps that could change confidence:
- a direct study of how volume versus depth affects RA or PhD hiring;
- primary text of ACM v1.1;
- the OSDI/NSDI 2026 calls for artifacts.
