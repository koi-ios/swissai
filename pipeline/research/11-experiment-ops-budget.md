# Unattended GPU experiment sweeps on about $90 (Modal-centred), as of 12 Nov 2026

All pages below were read on 12 Nov 2026 unless the Date column says otherwise. Pages were read through a summarising fetch tool, so numbers are as that tool reported them; spot-check anything you will rely on for money. Web search budget ran out partway through, so a few planned checks were not done (see Not verified).

## Bottom line

1. **Do not make "three free accounts" the plan's backbone. Use one primary account, ask Modal in writing, and apply to the academic programme.** (confidence: medium)
   - Modal's terms (May 2026) have no explicit multi-account clause. They do bar letting a third party use the service, and they say free use can carry separate limits that I could not find. Using someone else's account is the clearly risky case. Using your own several accounts is a grey area, not a documented permission. [X02]
   - The $30 credit is described as monthly on the pricing page. One account therefore gets $30 every month, so $90 can also be reached by time-spreading over three months instead of three accounts. [X01]
   - Modal advertises academic credits of up to $10k. Apply with a one-paragraph research plan. Eligibility and duration were not visible to me. [X01, X03]
   - A comparable provider (Colab) explicitly prohibits multiple accounts used to evade limits, which shows the usual reading of this practice. [X28]
2. **Cap spend in the Modal dashboard before the first paid run, and log planned against estimated cost yourself.** (high on the gating fact, medium on the workflow)
   - Workspace budgets and a net-charge "spend limit" exist and are hard caps set by an Owner or Manager. [X06]
   - The scriptable billing report (`modal billing`, `Workspace.billing`) is documented as offered to Team and Enterprise only ($250 per month), not Starter. [X04, X05, X01]
   - On Starter, compute cost per run from logged GPU type and container seconds times the current rates, then reconcile by hand against the dashboard. Reports show cost before credits, so even on Team they do not equal the invoice. [X04]
3. **Pin hardware explicitly and record it per run.** (high)
   - `gpu="H100"` may silently run on an H200 at the same price, and `A100` may become 80 GB. Use `H100!`, `A100-40GB` or `A100-80GB` for anything you measure. [X09]
   - Within one GPU class, devices differ: a large study found 32% mean and 72% maximum variation across same-SKU GPUs, with outliers up to 1.5x slower than the median (version figures differ, see Not verified). [X21]
4. **Treat Modal numbers as Modal numbers. Do not claim they predict Lustre results.** (high that the claim is unsupported, low on how large the gap is)
   - Modal runs on gVisor, a user-space kernel whose documented overhead sits in syscalls and file I/O. It does not expose a Lustre client or page-cache control. [X20, X08, X18]
   - Euler's Lustre scratch is a shared filesystem with a 15-day purge. GPU nodes there are for shareholder groups only. [X30, X31]
   - I found no study comparing Modal I/O against Lustre. State the platform in every claim. Run the Lustre-decisive step on the lab cluster, even if small.
5. **Design for variance, not for 3 runs.** (medium)
   - Run at least 3 per configuration as a floor. Use a pilot (5 to 10 runs of the noisiest configuration) to measure the coefficient of variation, then size repetitions from that. [X22, X23]
   - Interleave or randomise run order across configurations and across days, so drift in hardware or Modal's storage shows up as noise and not as a fake effect. Report medians with nonparametric intervals. [X22, X23]
   - In a large bare-metal study, configurations up to about 4% CoV usually needed "tens" of repetitions for a 1% interval, with some needing hundreds. [X22] That is a different platform; it tells you 3 runs supports only large effects.
6. **A human launches paid runs. The agent prepares and dry-runs.** (medium-high)
   - On Starter, a token acts as a user with full workspace access. Service users and restricted environments, the documented way to scope a token, are Team and Enterprise features. [X16, X17]
   - A Member-role seat cannot touch billing or workspace settings, and Starter allows up to 3 seats, so a separate Member seat for the agent may be possible. This is my inference and untested. [X01, X17]
   - Protections that work on any plan: a human-run launcher with a budget gate, `timeout`, `max_containers` and `retries=0` in every function, and a dashboard spend limit that the agent's token cannot be trusted to leave alone. [X06, X07, X14]
7. **Make runs idempotent and cheap to abandon.** (high, mostly engineering practice)
   - Modal timeouts apply per attempt, so retries multiply cost: a 100 s timeout with 3 retries can bill 400 s. Set retries to 0 for measurement runs. [X07]
   - All functions are preemptible and a preempted function restarts on the same input. `nonpreemptible` is not supported for GPU functions. [X13] Log a run record per attempt so a restart is visible.
   - Use `modal run --detach` only when the launcher will also record the app ID, so the human can `modal app stop` it. [X15]
8. **Split work across accounts only by independent experiment blocks, never within a comparison.** (medium)
   - Volumes live in one workspace environment, so each account re-downloads weights. Public weights from the Hugging Face Hub are the portable source. [X08, X11]
   - Two accounts also mean two Volumes, possibly different regions and hardware pools. Keep every comparison arm in one account and one pinned configuration. [X08, X12]
9. **Use cheaper venues for smoke tests and dev, Modal for the pinned measurement set.** (medium)
   - Community or marketplace GPUs run roughly 0.3 to 0.8 of Modal's price for similar classes, but hardware, disks and neighbours vary more. [X24, X26]
   - A T4 costs about $0.59 per hour on Modal. Smoke tests on cheap GPU classes cost cents, if the code path under test does not depend on the GPU model. [X01]
10. **Order experiments by decision value per dollar and stop early.** (medium; method, not evidence-backed)
    - Rank each step by (probability it changes a decision) times (value of that decision) divided by expected cost. Put cheap, high-information steps first, with a pre-written stop rule per step (see Transfer notes). [X22, X23]

## Price and limits table

Modal per-second prices read 12 Nov 2026 [X01]. Hourly values are my multiplication of the per-second price by 3600, rounded; they exclude CPU and memory.

| Item | Modal price | Hourly equivalent | Notes |
|---|---|---|---|
| T4 | $0.000164 per s | $0.59 | [X01] |
| L4 | $0.000222 per s | $0.80 | [X01] |
| A10 | $0.000306 per s | $1.10 | Pricing and docs list "A10", not A10G [X01, X09] |
| L40S | $0.000542 per s | $1.95 | [X01] |
| A100 40 GB | $0.000583 per s | $2.10 | [X01] |
| A100 80 GB | $0.000694 per s | $2.50 | [X01] |
| RTX PRO 6000 | $0.000842 per s | $3.03 | [X01] |
| H100 SXM5 | $0.001097 per s | $3.95 | May run as H200 unless `H100!` [X01, X09] |
| H200 SXM | $0.001261 per s | $4.54 | [X01] |
| B200 | $0.001736 per s | $6.25 | "B200+" may land on B300, billed as B200 [X01, X09] |
| B300 | $0.001972 per s | $7.10 | Needs CUDA 13.1+ [X09] |
| CPU | $0.0000131 per physical core per s | $0.047 per core-hour | 1 core = 2 vCPU; minimum 0.125 core [X01] |
| Memory | $0.00000222 per GiB per s | $0.008 per GiB-hour | [X01] |
| Volumes | $0.09 per GiB-month | | Page also mentions 1 TiB free; unclear whether that is storage or egress (see Not verified) [X01] |
| Egress | $0.04 per GiB | | [X01] |
| Region multiplier | 1.15x (broad, e.g. `us`) to 1.75x (narrow, e.g. `us-west`) | | Applies when a region is set [X01, X12] |
| Non-preemptible | 3x CPU and memory | | Not supported for GPU functions [X13, X01] |
| Starter plan | $0 plus compute; $30 free monthly credit | | 3 seats, 100 containers, 10 GPU concurrency [X01] |
| Team plan | $250 plus compute; $100 free monthly credit | | 50 GPU concurrency; billing API, service users, RBAC [X01, X04, X16, X17] |
| Academic credits | "up to $10k" | | Amount only confirmed; duration and eligibility not [X01, X03] |
| Spend controls | Workspace budget, spend limit, (Team) environment budget | | [X06] |

Rule of thumb (my arithmetic): $90 buys about 22 H100-hours or about 36 A100-80GB-hours or about 46 L40S-hours at GPU price alone. Add CPU and memory: a 4-core, 64 GiB container adds about $0.70 per hour. Add a 1.15x region multiplier if you set a region. Subtract failed runs and idle time before runs.

Alternatives (read 12 Nov 2026):

| Provider | Prices | Terms to know | IDs |
|---|---|---|---|
| RunPod pods | H100 SXM $3.99 secure, $2.69 community; A100 $1.79 / $1.19 to $1.39; L40S $1.09 / $0.79; RTX 4090 $0.89 / $0.34 per GPU-hour | Per-second billing; no spending limit mentioned on page; disk $0.10 per GB-month | [X24] |
| RunPod serverless | H100 $5.39, A100 $2.72, L40S $1.75 per hour | | [X24] |
| Lambda on-demand | A10 $1.29; A100 40 GB $1.99; A100 80 GB $3.19; H100 $3.79 to $4.89; B200 $7.69 to $7.99 per GPU-hour | Page says these prices take effect 16 Nov 2026; billing granularity not stated | [X25] |
| Vast.ai | Prices set by market; no figures visible on pages I read | Per-second billing; interruptible often "50%+ cheaper"; storage billed while stopped; bandwidth charged | [X26] |
| Lightning AI | T4 $0.55; L4 $0.79; A100 40 GB $2.19; L40S $3.54; H100 $5.45 per hour | Free plan up to 30 credits (about 50 interruptible GPU-hours per its FAQ); free sessions limited to 4 hours on A100/H100/H200; academic Pro about $20 per month | [X27] |
| Colab | No published limits; GPU type varies | Multiple accounts to dodge limits prohibited; 12 h maximum free runtime | [X28] |
| Kaggle | UNVERIFIED (page did not load) | | |
| Azure for Students | $100 credit for 12 months, one account per person | GPU eligibility not stated | [X35] |
| ETH Euler public share | Free for ETH members; up to 48 cores and 128 GiB; no GPUs | GPU nodes only for shareholder groups; Lustre scratch 2.5 TB, 15-day purge | [X29, X30, X31] |
| Euler GPU models | RTX 2080 Ti, 3090, 4090, A100 40/80 GB, RTX PRO 6000 (96 GB), TITAN RTX, Quadro RTX 6000; MI300A nodes | Shareholder access via your group | [X31] |
| CSCS / Alps | Small projects up to 8,000 node-hours; PhD students need supervisor as PI; production calls start 1 Apr and 1 Oct | Applications via projectoffice.cscs.ch; read via search summary only | [X32] |
| Swiss AI Initiative | Compute grants; 6-month small grants; 4th call deadline 14 Sep 2026 (passed) | No student-specific programme found | [X33] |

## Source ledger

Type: docs / terms / paper / blog / forum. Grade: A controlled study, B measured case study, C practitioner report, D vendor claim, E opinion. Provider documentation, prices and terms are marked in Type and graded D only where they assert a benefit; otherwise the grade is shown as n/a (docs are authoritative for what the product does, not for performance).

| ID | Source (author, title, URL) | Date | Type | Grade | Finding (with numbers or exact terms) | Supports or contradicts which decision | Model-generation dependent? |
|---|---|---|---|---|---|---|---|
| X01 | Modal Labs, Pricing, https://modal.com/pricing | read 12 Nov 2026 | docs (prices) | n/a | GPU per-second prices as in the table (T4 $0.000164 to B300 $0.001972); CPU $0.0000131 per core-s; memory $0.00000222 per GiB-s; Starter $0 plus compute with $30 free monthly credit, 3 seats, 100 containers, 10 GPU concurrency; Team $250, $100 credit, 50 GPU concurrency; academics "up to $10k"; regional 1.15 to 1.75x; non-preemptible 3x | 1, 2, 6, 8, table | Yes (prices change; Lambda changes 16 Nov) |
| X02 | Modal Labs, Software as a Service Agreement, https://modal.com/legal/terms | "Effective May 2026"; read 12 Nov 2026 | terms | n/a | Business agreement: use "solely for Customer's internal business". Restrictions bar letting third parties "rent, resell or otherwise allow any third party direct access" (s1.3). No-fee use is "as-is" and may carry separate limits (s1.2). No clause found on multiple accounts, credential sharing or free-credit abuse. Modal may suspend access with notice (s1.6). A separate "No-Fee Use Limitations" document is referenced but not found | 1 | Terms revise (an Oct 2025 version exists at /legal/terms-effective-date-october-2025) |
| X03 | Modal Labs, Modal for Academics, https://frontend.modal.com/academics (also modal.com/lp/academics, which returned empty) | read 12 Nov 2026 | docs (programme page) | D | "Up to $10k" in credits; page invites sign-up for grant programmes; eligibility, duration, restrictions not stated on the page I read. A search summary of the main page said credits are spread over 1 year, do not roll over and are granted once per research project (not confirmed by my fetch) | 1 | Yes |
| X04 | Modal Labs, Billing guide, https://modal.com/docs/guide/billing.md | read 12 Nov 2026 | docs | n/a | Spend reports by App or resource via `Workspace.billing` and `modal billing`, offered to Team and Enterprise; reports exclude credits, reservations and the egress allowance; data appears within minutes but may lag; App tags (`modal.App(tags=...)`) can be included in reports | 2 | Possible (plan gating may change) |
| X05 | Modal Labs, `modal billing` CLI, https://modal.com/docs/cli/latest/billing | read 12 Nov 2026 | docs | n/a | Subcommands `rates`, `report`, `summary`; report options `--start --end --for -r d/h --tag-names --show-resources --json/--csv`; summaries are single calendar months; plan gating not stated on this page | 2 | Possible |
| X06 | Modal Labs, Budgets, https://modal.com/docs/guide/budgets | read 12 Nov 2026 | docs | n/a | Workspace budget is a monthly "hard outer cap", measured before credits. Spend limit caps net charges after credits and stops workloads that would add out-of-pocket charges. Environment budgets need Team or Enterprise. Only Owners and Managers can set them. Enforcement delay not stated | 2, 6 | Possible |
| X07 | Modal Labs, Timeouts, https://modal.com/docs/guide/timeouts | read 12 Nov 2026 | docs | n/a | Default 300 s, range 1 s to 24 h; `startup_timeout` separate; timeout applies to each attempt, so 100 s with 3 retries runs at least 400 s; functions may run a few seconds past the timeout | 2, 7 | Possible |
| X08 | Modal Labs, Volumes, https://modal.com/docs/guide/volumes | read 12 Nov 2026 | docs | D (throughput claim) | Designed for up to 2.5 GB/s, "not guaranteed"; v1 about 50,000 files and 500,000 inode limit; v2 no file-count limit; changes need commit/reload; last-write-wins on the same file; created in an environment | 4, 8 | Yes (Volume v2 is new) |
| X09 | Modal Labs, GPU guide, https://modal.com/docs/guide/gpu | read 12 Nov 2026 | docs | n/a | `H100` may run on H200 at the same cost, use `H100!`; `A100` may be upgraded to 80 GB, use `A100-40GB` or `A100-80GB`; fallback lists; up to 8 GPUs per container for most types, A10 up to 4; more than 2 GPUs usually means longer waits; payment method required | 3, table | Yes |
| X10 | Modal Labs, Cold start guide, https://modal.com/docs/guide/cold-start | read 12 Nov 2026 | docs | D | Containers "boot in about one second"; default scale-down window 60 s (2 s to 20 min); pre-downloading weights can cut boot from minutes to seconds; no variance section | 4 | Yes |
| X11 | Modal Labs, Storing model weights, https://modal.com/docs/guide/model-weights.md | read 12 Nov 2026 | docs | D | "Performance is similar for the two methods" (Volume vs Image); no load-time numbers; Image rebuilds cause extra downloads | 4, 8 | Yes |
| X12 | Modal Labs, Region selection, https://modal.com/docs/guide/region-selection | read 12 Nov 2026 | docs | n/a | Broad region 1.15x, narrow 1.75x, smaller multiplier applies for mixed sets; unconstrained containers run "globally across multiple different clouds"; inputs route through us-east; hardware variation by region not addressed | 3, 8 | Possible |
| X13 | Modal Labs, Preemption, https://modal.com/docs/guide/preemption | read 12 Nov 2026 | docs | n/a | All functions preemptible, restart on the same input; `nonpreemptible=True` is 3x CPU and memory price and not supported for GPU functions | 7 | Possible |
| X14 | Modal Labs, Scaling out, https://modal.com/docs/guide/scale | read 12 Nov 2026 | docs | n/a | `max_containers` caps parallelism; plan-level limits on containers and GPUs in use; 4,000 per-function hard limit | 6 | No |
| X15 | Modal Labs, Apps, https://modal.com/docs/guide/apps | read 12 Nov 2026 | docs | n/a | Ephemeral apps stop when the client disconnects unless `--detach`; deployed apps persist until stopped with `modal app stop` | 7 | No |
| X16 | Modal Labs, Service users, https://modal.com/docs/guide/service-users.md | read 12 Nov 2026 | docs | n/a | Available on Team and Enterprise only; new service users have no environment access until granted; `--expires-in` for tokens; deletion revokes immediately | 6 | Possible |
| X17 | Modal Labs, RBAC, https://modal.com/docs/guide/rbac.md | read 12 Nov 2026 | docs | n/a | RBAC and restricted environments are Team and Enterprise; Member role cannot do billing or workspace settings; Owner and Manager have full access | 6 | Possible |
| X18 | Modal Labs, Cloud bucket mounts, https://modal.com/docs/guide/cloud-bucket-mounts | read 12 Nov 2026 | docs | n/a | S3, R2, GCS via AWS Mountpoint; "optimized for reading large files sequentially"; no throughput figures; no append or rename; `read_only=True` available | 4, 8 | No |
| X19 | Modal Labs, GPU memory snapshots blog, https://modal.com/blog/gpu-mem-snapshots | Jul 2025 (per listing); read 12 Nov 2026 | blog (vendor) | D | One vLLM data point: startup 45 s to 5 s with snapshots (Qwen2.5-0.5B); no hardware, run count or variance given | 4 (cold-start numbers are vendor-only) | Yes |
| X20 | gVisor project, Performance guide, https://gvisor.dev/docs/architecture_guide/performance/ | n/a (undated); read 12 Nov 2026 | docs (vendor) | D | Overhead is mainly in syscalls and file operations crossing the sandbox (through the gofer); raw disk I/O no significant fundamental overhead; no percentages in text; tests on GCE n1-standard-4 | 4 | No (design property) |
| X21 | Sinha et al., Not All GPUs Are Created Equal, https://arxiv.org/pdf/2208.11035 | 2022 (SC22); search summary read 12 Nov 2026 | paper | B | Same-SKU GPUs on 5 HPC clusters: mean 32%, max 72% variation in the SC22 abstract (arXiv text cites up to 22%); outliers up to 1.5x slower than median; causes manufacturing and power management | 3, 5 | Partly (older GPUs; cloud fleets differ) |
| X22 | Maricq et al., Taming Performance Variability, OSDI 2018, https://usenix.org/system/files/osdi18-maricq.pdf | 2018; PDF text read 12 Nov 2026 | paper | B | 835 servers, 10,400 runs; configurations up to about 4% CoV need tens of repetitions for 1% intervals, some need hundreds; 10 measurements gave medians outside the interval from 58 to 68 repetitions in one case; CONFIRM estimates repetitions from past data; high-variance hardware needs more runs | 5, 10 | No (method); numbers are bare-metal CloudLab, not Modal |
| X23 | Hoefler and Belli, Scientific Benchmarking of Parallel Computing Systems, SC15, https://htor.inf.ethz.ch/publications/img/hoefler-scientific-benchmarking.pdf | 2015; PDF text read 12 Nov 2026 | paper (guideline) | E | 12 rules: do not cherry-pick; do not assume normality; compare nondeterministic data with statistics; document all varying factors; report measurement and summarisation techniques; plot enough information to interpret | 5, 10 | No |
| X24 | RunPod, Pricing, https://www.runpod.io/pricing | read 12 Nov 2026 | docs (prices) | n/a | Pod prices per GPU-hour as in table; per-second billing; storage $0.10 per GB-month; no spending limit mentioned | 9, table | Yes |
| X25 | Lambda, Pricing, https://lambda.com/pricing | read 12 Nov 2026 | docs (prices) | n/a | A10 $1.29, A100 $1.99 to $3.19, H100 $3.79 to $4.89, B200 $7.69 to $7.99 per GPU-hour; "take effect November 16, 2026" | 9, table | Yes |
| X26 | Vast.ai, Pricing and instance pricing docs, https://vast.ai/pricing and https://docs.vast.ai/documentation/instances/pricing | read 12 Nov 2026 | docs | D ("50%+ cheaper") | Market-set prices; per-second billing; interruptible "50%+ cheaper"; storage billed even when stopped; bandwidth charged per byte; no price figures visible | 9 | Yes |
| X27 | Lightning AI, Pricing, https://lightning.ai/pricing | read 12 Nov 2026 | docs (prices) | n/a | T4 $0.55, L4 $0.79, L40S $3.54, A100 $2.19, H100 $5.45 per hour; free plan up to 30 credits; 4 h session limit on A100/H100/H200; academic Pro about $20 per month | 9, table | Yes |
| X28 | Google, Colab FAQ, https://research.google.com/colaboratory/faq.html | read 12 Nov 2026 | terms/docs | n/a | Limits unpublished and fluctuating; GPU types vary; free runtime up to 12 h; prohibits multiple accounts "to work around access or resource usage restrictions" | 1, 9 | Yes |
| X29 | ETH Zurich Scientific IT, Euler user account, https://docs.hpc.ethz.ch/users/user_account/ | read 12 Nov 2026 | docs | n/a | Open to all ETH members; public share free for small jobs, up to 48 cores and 128 GiB, no GPUs; account created on first SSH login with ETH email verification; groups ("shareholders") manage their own membership | table, 4 | No |
| X30 | ETH Zurich Scientific IT, Euler storage, https://docs.hpc.ethz.ch/hardware/storage/ | read 12 Nov 2026 | docs | n/a | Scratch and Work are Lustre; Home and Project NFS v3; scratch 2.5 TB and 1,000,000 files per user; files older than 15 days deleted without notice (table says 2 weeks); no snapshots or backup; node-local `$TMPDIR` on XFS | 4 | No |
| X31 | ETH Zurich Scientific IT, Euler GPU nodes, https://docs.hpc.ethz.ch/hardware/gpu_nodes/ | read 12 Nov 2026 | docs | n/a | GPU nodes "only available to shareholders"; models RTX 2080 Ti, 3090, 4090, TITAN RTX, Quadro RTX 6000, A100 40/80 GB, RTX PRO 6000; V100 decommissioned 2026 | table, 4 | Yes |
| X32 | CSCS, allocation schemes and deadlines (via search summary of https://www.cscs.ch/user-lab/allocation-schemes/ pages), plus news https://www.cscs.ch/publications/news/2026/cscs-upcoming-deadlines-for-submission-of-proposals-on-alps-and-lumi | May 2026; summary read 12 Nov 2026 | docs (secondary read) | n/a | Small projects up to 8,000 node-hours, PhD students need their supervisor as PI; production calls start 1 Apr and 1 Oct; portal projectoffice.cscs.ch; preparatory access by email | table, 4 | Yes |
| X33 | Swiss AI Initiative / EPFL, Large Grants, https://www.epfl.ch/research/funding/epfl-programmes/swiss-ai-initiative-large-grants | 2026; summary read 12 Nov 2026 | docs (secondary read) | n/a | Compute grants plus up to 200k CHF; 6-month small grants; 4th call deadline 14 Sep 2026; no student-specific programme found | table | Yes |
| X34 | Serverless GPU cold-start comparison blogs, e.g. https://hostfleet.pages.dev/cold-start-latency-showdown/ , https://dev.to/heckno/i-tested-9-serverless-gpu-providers-for-ai-inference-in-2026-heres-what-id-actually-use-4cf4 | 2026; snippets read via search 12 Nov 2026 | blog | C/E | Claim of median cold starts 2.4 s RunPod, 3.1 s Modal, 7.8 s Replicate for Llama-3 8B; other pages give RunPod 30 s; methods undocumented; no transparent same-model comparison found | 4 (no usable external Modal cold-start data) | Yes |
| X35 | Microsoft, Azure for Students, https://azure.microsoft.com/en-us/free/students | read 12 Nov 2026 | docs/terms | n/a | $100 credit for 12 months; one account per person; renewable while a student; GPU use not stated | table | Yes |
| X36 | Aggregators (costbench, freetier etc.) on Modal Starter, e.g. https://costbench.com/software/ai-gpu-cloud/modal/free-plan/ | 2026; snippet via search | blog | E | Say $30 resets monthly and does not roll over; consistent with X01 but secondary | 1 | Yes |
| X37 | Other providers' multi-account rules, e.g. Anyscale AUP https://anyscale.com/aup ; Fly.io forum https://community.fly.io/t/free-usage-quotas-and-violating-terms-of-service/8580/2 | snippets via search 12 Nov 2026 | terms/forum | C | Anyscale lists multiple accounts to extend a free trial as prohibited; Fly staff said spreading orgs to reuse the free tier violates terms while good-faith use is fine | 1 (analogy only; says nothing about Modal) | No |

## Disagreements

- **Modal (docs) vs a cautious reading of its own behaviour on weights storage.** Modal says Volumes and Image-baked weights perform "similarly" and Volumes are more flexible [X11]; its cold-start guide says baking weights into the Image avoids slow boots [X10]. Neither gives numbers. For a cold-start study this is the thing you are measuring, so do not adopt either claim; measure both arms and label them. Fits: measure.
- **Modal vs gVisor documentation on overhead.** gVisor says raw disk I/O has no significant fundamental overhead but file operations crossing the sandbox boundary pay a cost [X20]. For loading multi-GB safetensors the relevant path is large sequential reads, probably cheap, while many-small-file loads (JIT caches, Triton and torch.compile cache directories) are exactly the syscall-heavy case. This is my inference, not a measurement. Fits: assume file-count-heavy cache experiments may be most affected.
- **Statistical sufficiency of 3 runs.** The plan's 3 runs per configuration vs Maricq (tens of repetitions to pin a 1% interval at about 4% CoV) [X22] and Hoefler and Belli (report variation, no normality assumption) [X23]. Both sides are about different effect sizes: 3 runs is defensible for effects of several times the run-to-run spread (for example 2x cold-start gains), not for 5% effects. Fits: keep 3 as a floor, use pilot-based sizing.
- **Vendors vs independent blogs on cold-start numbers.** Modal's 45 s to 5 s [X19] vs blog medians of 3.1 s [X34]; the blogs disagree with each other (RunPod 2.4 s vs 30 s). None has a stated method. Fits: ignore all for design; measure yourself.
- **Is multi-account use fine?** Modal's terms are silent on it [X02], while Colab and Anyscale prohibit it explicitly [X28, X37]. Silence is not permission, particularly with a "separate limits" clause I could not read. Fits: treat as unresolved, ask Modal.

## Disconfirming evidence and null results

- **"Split the plan across three free accounts."**
  - Terms: no explicit multi-account clause (a null result, searched in the terms page and by search for an acceptable use policy), but a clause barring third-party access, and a reference to separate no-fee limits I could not find [X02]. Other providers prohibit it [X28, X37].
  - The $30 is monthly [X01], so one account over three months gives the same credit without the question. If the deadline binds, that is the cost; if not, it is the cleanest fix.
  - Reproducibility: Volumes belong to one workspace environment, so each account re-downloads weights and builds its own caches [X08]. Hardware pools may differ by account and time, and Modal says unconstrained containers run across multiple clouds [X12]. Splitting inside a comparison therefore adds an uncontrolled factor.
  - Duplicated downloads cost time and any egress; public weights re-download is the cheapest workaround [X11].
- **"Results on Modal predict results on a Lustre-backed cluster."** No source tests this. Evidence against: a different kernel layer (gVisor) with documented file-I/O costs [X20]; Volume throughput stated as "not guaranteed" [X08]; Lustre scratch has its own caching, striping and 15-day purge behaviour [X30]. I searched for Modal-versus-Lustre or object-storage-versus-Lustre model-loading comparisons; the last search was blocked by the search budget, so this is a null result with a gap.
- **"Agents can safely run paid sweeps unattended."** No documented agent-specific incident found (I searched only Modal docs, not incident reports). Structural evidence against safety on Starter: tokens carry user-level authority, and token scoping is a Team feature [X16, X17]; budgets are the only hard stop, and they are set by the Owner [X06]; retries multiply billable time [X07]; detached apps keep running after the client exits [X15]; enforcement delay of the spend limit is undocumented [X06].
- **"A scriptable spend log is available."** On Starter, the billing report API is documented only for Team and Enterprise [X04]. I did not test the command on a Starter account, and the CLI page does not state gating [X05], so treat this as likely but untested.
- **Null results:** no independent measurement of Modal Volume read variance; no independent Modal-versus-bare-metal I/O comparison; no controlled study (grade A) on any question here.

## Transfer notes

Practices that transfer from funded labs to a student with $90:

- **Transfer as is:** pinned hardware names [X09]; per-run environment capture; a manifest of planned runs; interleaved run order; medians with nonparametric intervals; a stated stop rule [X22, X23]; hard dashboard cap [X06]; retries at zero for measured runs [X07].
- **Transfer in reduced form:** CONFIRM-style repetition planning, using 5 to 10 pilot runs instead of a large past data set [X22]; blocking by day and by GPU instance instead of a fleet-wide design.
- **Do not transfer:** Team-plan tooling (billing API, service users, RBAC) at $250 per month, which alone costs about three times the $90 [X01, X04, X16]; multi-tenant experiment trackers that cost money.
- **Per-run record (my design, consistent with Hoefler and Belli rule 9 [X23]):** run ID, config hash, git commit, image digest, requested GPU string, `nvidia-smi` name, UUID, driver, clock and power state, CPU model, kernel, region if known, container start and end times, whether it was a cold or reused container, Volume reload state, planned and measured cost.
- **Cost log:** estimated cost = container seconds x (GPU rate + cores x CPU rate + GiB x memory rate) x region multiplier, taken from `modal billing rates` or the pricing page [X01, X05]. Reconcile weekly against the dashboard.
- **Result writing:** write to a temp file, fsync, then rename to `results/<run_id>.json`, so "result exists" is a valid skip test. Failed runs go to `quarantine/` with the log, not retried automatically. Keep the manifest and result files as CSV or JSON in git; for this size a tracker adds an account and a provider dependency without helping rigour (my judgement, no controlled evidence; priority 4 survives switching providers).
- **Smoke test:** run one tiny configuration on the cheapest class (T4, about $0.59 per hour) or a marketplace GPU, with a 5-minute timeout, to catch image, path and import errors before paid pinned runs [X01, X07]. Add fail-fast checks (assert GPU name, free disk, model path exists, expected token count) before any timed section.
- **Budget gate:** a launcher script, run by the human, that reads planned cost for pending runs from the manifest, subtracts logged spend, and refuses when remaining budget minus a reserve (suggest 15%) is below the batch's estimate. This is a courtesy check; the dashboard cap is the real stop [X06].
- **Planning method:** score steps by decision value per dollar; run a pilot of the noisiest step first; pre-commit stop rules such as "stop adding repetitions when the 95% interval on the ratio is narrower than half the smallest effect that would change the decision, or when the step has used 1.5x its estimate" [X22, X23]. These thresholds are suggestions, not from a source.

## What better models change, and what they don't

- **Change:** agents write launchers, manifests, analysis scripts and dry-run cost estimators faster and with fewer errors; this lowers human hours per experiment. Prices and GPU lists (B300, RTX PRO 6000 now on Modal) change with hardware generations [X01, X09].
- **Don't change:** billing gating by plan [X04, X16, X17]; whether the token an agent holds can exceed its brief [X16]; run-to-run and device-to-device variance [X21, X22]; the fact that Modal has no Lustre [X08, X18]; platform terms. A better model cannot read a "No-Fee Use Limitations" document that is not published to it, and it can invent a plausible reading of the terms; keep a human reading of terms in the loop.
- Newer inference engines (SGLang and vLLM releases) change cold-start behaviour and flags, so a stored result is tied to engine version; record it [X19].

## Not verified

- UNVERIFIED: whether `modal billing report` works on a Starter workspace. Docs say the APIs are offered to Team and Enterprise [X04]; the CLI page is silent [X05].
- UNVERIFIED: Modal's "No-Fee Use Limitations" document; any Modal acceptable-use policy on multiple accounts. Not found [X02].
- UNVERIFIED: academic programme eligibility, duration, one-per-project rule, and a contact email (a search summary named academics@modal.com for teaching; my page fetch did not show it) [X03].
- UNVERIFIED: whether the pricing-page line "Volumes $0.09 per GiB per month, 1 TiB free" means free storage or free egress [X01].
- UNVERIFIED: when the spend limit stops workloads (immediate or delayed), and whether a container already running is killed [X06].
- UNVERIFIED: that Starter credits expire monthly with no rollover (aggregators say so [X36]; Modal's billing guide does not cover it [X04]).
- UNVERIFIED: whether Modal's A10 is the same hardware that other docs and blogs call A10G [X01, X09].
- UNVERIFIED: Kaggle GPU quotas and current types (the page did not load usably).
- UNVERIFIED: the Sinha et al. headline (32% or 22%) differs between abstract and body in the summaries I saw [X21].
- UNVERIFIED: a USENIX gVisor syscall-cost figure (2.2x) quoted second-hand; not used.
- UNVERIFIED: CSCS allocation details beyond search summaries; one CSCS page returned 404 [X32].
- UNVERIFIED: Lambda prices before 16 Nov 2026, and the configuration behind each price range [X25]; Vast.ai marketplace prices (not visible).
- Not searched (search quota exhausted): Lustre versus other storage in LLM loading; any incident reports about unattended agents and cloud bills; RunPod and Vast.ai terms on multiple accounts; Hugging Face download and egress terms.

## Saturation note

I read 30 primary pages (Modal docs, pricing and terms; provider price pages; ETH Euler docs; two papers' full text) and used search summaries for CSCS, Swiss AI, Modal token scoping, cold-start blogs and multi-account analogies. New Modal documentation pages after about the 15th source (billing, budgets, service users, RBAC, preemption) confirmed one pattern: hard controls exist but advanced controls are gated to Team. Later sources did not change the recommendations. I stopped because (a) the web search quota ran out, and (b) remaining open questions (Starter report gating, no-fee limits, academic eligibility) can only be settled by Modal, not by more public pages. Total distinct sources in the ledger: 37, a little over the 20 to 35 target because several are secondary (X32, X33, X34, X36, X37). No grade A sources exist for this topic; the strongest evidence is grade B measurement papers on other platforms.
