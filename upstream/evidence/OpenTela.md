# Evidence listing: eth-easl/OpenTela (origin/main 4b6e90f, gathered 12 Nov 2026)

Read-only. Trailer columns are matched anywhere in the message body, so squash merges also show trailers of the squashed inner commits.

## Last 100 commits on origin/main: author, committer, trailers

| hash | date | author | committer | AI Co-Authored-By | Claude-Session | Entire-Checkpoint | Signed-off-by | (#N) | subject |
|---|---|---|---|---|---|---|---|---|---|
| 4b6e90f | 2026-11-02 | Xiaozhe Yao | GitHub | - | - | Y | - | Y | fix(security): resolve Dependabot alerts across Go, npm and Python dep |
| f0eca38 | 2026-11-02 | Xiaozhe Yao | GitHub | Y: Claude Opus 5.5/Claude Sonnet 5.5 | Y | Y | - | Y | Auto link opt in (#69) |
| c7a4a96 | 2026-11-02 | Mel Zuercher | GitHub | Y: Claude Opus 5.5 | Y | Y | - | Y | start: --ephemeral-identity and --datadir for nodes sharing a $HOME (# |
| 5c4bbdb | 2026-11-02 | Rob | GitHub | Y: Claude Opus 5.5/Claude Sonnet 5.5 | Y | Y | - | Y | fix(protocol): worker connectivity watchdog retries bootstrap + relay  |
| f740781 | 2026-10-15 | Xiaozhe Yao | GitHub | - | - | - | - | Y | fix(protocol): never sweep or evict this node's own table row (#65) |
| 307ed15 | 2026-09-30 | Xiaozhe Yao | GitHub | - | - | - | - | - | Update README with OpenTela adoption links |
| 00bb756 | 2026-09-24 | Xiaozhe Yao | Xiaozhe Yao | - | - | - | - | - | feat(cli): persist deploy keys; auto-link at 'otela start' |
| b5fe983 | 2026-09-24 | Xiaozhe Yao | GitHub | - | - | - | - | Y | feat(cli): 'otela instance link' via scoped deploy keys (#5) |
| 3ba1806 | 2026-09-23 | Xiaozhe Yao | GitHub | - | - | - | - | Y | feat(acl): whitelist 'api_key' and 'no_match' reasons in the v2 evalua |
| 783409c | 2026-09-19 | Xiaozhe Yao | GitHub | - | - | - | - | - | Update README with adoption examples for OpenTela |
| 42728f3 | 2026-09-19 | Xiaozhe Yao | Xiaozhe Yao | - | - | Y | - | - | feat(routing): session affinity — sticky peers for KV-cache warmth |
| 6f56f30 | 2026-09-18 | Xiaozhe Yao | Xiaozhe Yao | - | - | - | - | - | feat(home): live network stats strip on the frontpage |
| 073cd3c | 2026-09-18 | Xiaozhe Yao | Xiaozhe Yao | - | - | - | - | - | docs(cloud): add network statistics page with live snapshot |
| 778a8fa | 2026-09-17 | Xiaozhe Yao | Xiaozhe Yao | - | - | Y | - | - | fix(lint): expand math.Pow in allowed_peers_test (QF1005) |
| 0f4d3de | 2026-09-17 | Xiaozhe Yao | Xiaozhe Yao | - | - | Y | - | - | fix(protocol): make workers relayed by the node itself evictable |
| 7e28835 | 2026-09-14 | Xiaozhe Yao | GitHub | - | - | - | - | Y | boot: reconcile the node with its cloud account at startup (#3) |
| 61ccdb6 | 2026-09-14 | Xiaozhe Yao | GitHub | - | - | - | - | Y | billing: retire the legacy settlement rail — the API market is the sin |
| b3d82d9 | 2026-09-14 | Xiaozhe Yao | GitHub | - | - | - | - | Y | Price-aware routing in the mesh head (Phase 1 market signal) (#1) |
| 119401e | 2026-09-06 | Aryan Ahadinia | GitHub | Y: Claude Opus 5 (1M context) | - | - | - | Y | feat(attestation): make the build-attestation trust anchor configurabl |
| 1111d5d | 2026-09-06 | Aryan Ahadinia | GitHub | Y: Claude Opus 5 (1M context) | - | - | - | Y | fix(bootstrap): explicit bootstrap peers must override the static list |
| 6c8574d | 2026-08-25 | Aryan Ahadinia | GitHub | - | - | - | - | Y | logging: reduce hot-path log noise, configurable stdout, harden two fa |
| b04d592 | 2026-08-25 | Xiaozhe Yao | Xiaozhe Yao | - | - | - | - | - | chore: strip output from Jupyter notebooks |
| 80cd709 | 2026-08-25 | Xiaozhe Yao | Xiaozhe Yao | - | - | - | - | - | Refactor code structure for improved readability and maintainability |
| 935a875 | 2026-08-25 | Xiaozhe Yao | Xiaozhe Yao | - | - | - | - | - | ci: make coverage-comment step non-fatal for fork PRs |
| 19ae659 | 2026-08-25 | Aryan Ahadinia | GitHub | - | - | - | - | Y | feat: peer metadata (hostname, version, labels, pending/ready status)  |
| c32ee69 | 2026-08-22 | Xiaozhe Yao | Xiaozhe Yao | - | - | - | - | - | docs: remove account console, extracted to cloud.opentela.ai |
| f1b645d | 2026-08-22 | Xiaozhe Yao | Xiaozhe Yao | - | - | - | - | - | feat(start): add `otela start --wallet.address` to run as a chosen man |
| a94f2f6 | 2026-08-22 | Xiaozhe Yao | Xiaozhe Yao | - | - | - | - | - | feat(cli): add `otela wallet link` to bind a wallet to a cloud account |
| 44e2db9 | 2026-08-17 | Xiaozhe Yao | Xiaozhe Yao | - | - | - | - | - | docs: hide proposals from sidebar and search |
| a358872 | 2026-08-16 | Xiaozhe Yao | Xiaozhe Yao | - | - | - | - | - | docs(site): a11y + UX audit fixes for the documentation site |
| f29c6b7 | 2026-08-16 | Xiaozhe Yao | Xiaozhe Yao | - | - | - | - | - | docs(account): email-verification step for Neon Auth sign-up |
| 1628c10 | 2026-08-16 | Xiaozhe Yao | Xiaozhe Yao | - | - | - | - | - | docs(account): graceful empty state when billing mode is off |
| ab922e8 | 2026-08-16 | Xiaozhe Yao | Xiaozhe Yao | - | - | - | - | - | docs(env.example): update Neon Auth URL to the live instance |
| ce5611f | 2026-08-16 | Xiaozhe Yao | Xiaozhe Yao | - | - | - | - | - | billing: mesh peer filtering by billing affordability + account billin |
| a5e6529 | 2026-08-14 | Xiaozhe Yao | Xiaozhe Yao | - | - | - | - | - | update docs |
| e059134 | 2026-08-06 | Xiaozhe Yao | Xiaozhe Yao | - | - | - | - | - | docs(cloud): add Public Cloud setup guide for the api.opentela.ai stac |
| 7928586 | 2026-08-05 | Xiaozhe Yao | Xiaozhe Yao | - | - | - | - | - | docs(cf): strip unused @vercel/og from Worker bundle to fit free-tier  |
| a612337 | 2026-08-05 | Xiaozhe Yao | Xiaozhe Yao | - | - | - | - | - | feat: mesh observatory docs page + protocol liveness hardening |
| 1c54345 | 2026-07-30 | Xiaozhe Yao | GitHub | - | - | - | - | - | Update README.md |
| e9d1696 | 2026-07-29 | Xiaozhe Yao | Xiaozhe Yao | - | - | - | - | - | fix(ci): synchronize autorelay host publication |
| 9fac744 | 2026-07-29 | Xiaozhe Yao | Xiaozhe Yao | Y: Claude Opus 5 (1M context) | Y | - | - | - | fix(shutdown): drain the LEFT announcement before wiping the CRDT stor |
| 7f42183 | 2026-07-29 | Xiaozhe Yao | Xiaozhe Yao | Y: Claude Opus 5 (1M context) | Y | - | - | - | fix(p2p): dial the target peer, not the incoming Host header |
| 43faf54 | 2026-07-29 | Xiaozhe Yao | Xiaozhe Yao | - | - | - | - | - | docs: explain and align trusted region workflow |
| f6a49c9 | 2026-07-29 | Xiaozhe Yao | Xiaozhe Yao | - | - | - | - | - | feat: enforce trusted-region service policies |
| 5a95896 | 2026-07-29 | Xiaozhe Yao | Xiaozhe Yao | - | - | - | - | - | chore: snapshot ACL client baseline |
| 0c88314 | 2026-07-27 | Xiaozhe Yao | Xiaozhe Yao | - | - | - | - | - | update docs versions |
| 49e39e6 | 2026-07-26 | Xiaozhe Yao | Xiaozhe Yao | Y: Claude Opus 5 (1M context) | Y | - | - | - | feat(docs): standalone /account console |
| 8d1eb88 | 2026-07-25 | Xiaozhe Yao | Xiaozhe Yao | Y: Claude Opus 4.8 (1M context) | Y | - | - | - | fix(docs): suppressHydrationWarning on <body> for extension-injected a |
| 7c8f72c | 2026-07-24 | Xiaozhe Yao | Xiaozhe Yao | - | - | - | - | - | fix(docs): clear sk- key on sign-out, reactive session + redirect, rev |
| 52a1f2c | 2026-07-24 | Xiaozhe Yao | Xiaozhe Yao | - | - | - | - | - | chore(docs): verify build + document /account page |
| be53b78 | 2026-07-24 | Xiaozhe Yao | Xiaozhe Yao | - | - | - | - | - | feat(docs): client-side wallet/account soft-link |
| b4c488c | 2026-07-24 | Xiaozhe Yao | Xiaozhe Yao | - | - | - | - | - | feat(docs): live /v1/models services panel |
| 866082e | 2026-07-24 | Xiaozhe Yao | Xiaozhe Yao | - | - | - | - | - | feat(docs): sk- key management via /manage/keys |
| abe6247 | 2026-07-24 | Xiaozhe Yao | Xiaozhe Yao | - | - | - | - | - | feat(docs): neon auth sign-in panel on /account |
| c1856d0 | 2026-07-24 | Xiaozhe Yao | Xiaozhe Yao | - | - | - | - | - | fix(docs): only construct neon auth client in the browser (SSR-safe) |
| 4a10434 | 2026-07-24 | Xiaozhe Yao | Xiaozhe Yao | - | - | - | - | - | feat(docs): neon auth client + getAuthJwt |
| 0c74f23 | 2026-07-24 | Xiaozhe Yao | Xiaozhe Yao | - | - | - | - | - | chore(docs): stop tracking generated tsconfig.tsbuildinfo |
| 8384f83 | 2026-07-24 | Xiaozhe Yao | Xiaozhe Yao | - | - | - | - | - | feat(docs): add api base + neon auth url config |
| bde06a4 | 2026-07-24 | Xiaozhe Yao | Xiaozhe Yao | - | - | - | - | - | feat(docs): /manage/keys client with status mapping |
| a8851a6 | 2026-07-24 | Xiaozhe Yao | Xiaozhe Yao | - | - | - | - | - | test(docs): spec-correct in-memory localStorage shim for jsdom |
| 7b4ad66 | 2026-07-24 | Xiaozhe Yao | Xiaozhe Yao | Y: Claude Opus 4.8 (1M context) | Y | - | - | - | fix(docs): SSR guard writeLink, cover missing-field case |
| a53e9f9 | 2026-07-24 | Xiaozhe Yao | Xiaozhe Yao | - | - | - | - | - | feat(docs): soft-link localStorage helpers |
| 42d35db | 2026-07-24 | Xiaozhe Yao | Xiaozhe Yao | - | - | - | - | - | test(docs): add vitest for account modules |
| 055b004 | 2026-07-24 | Xiaozhe Yao | Xiaozhe Yao | Y: Claude Opus 4.8 (1M context) | Y | - | - | - | refactor(docs): move wallet page to /account, add redirects, fix nav |
| d39e57a | 2026-07-14 | Xiaozhe Yao | Xiaozhe Yao | - | - | - | - | - | update package-lock.json |
| c1d67b6 | 2026-07-14 | Xiaozhe Yao | Xiaozhe Yao | - | - | - | - | - | feat: improve owner integration and message signing functionality |
| ca50dd5 | 2026-06-24 | Xiaozhe Yao | Xiaozhe Yao | Y: Claude Opus 4.8 (1M context) | Y | - | - | - | feat(analytics): head-node performance analytics to PostHog |
| 78c3097 | 2026-06-24 | Xiaozhe Yao | Xiaozhe Yao | - | - | - | - | - | update docs |
| b00dc4b | 2026-06-05 | Xiaozhe Yao | Xiaozhe Yao | - | - | - | - | - | minor |
| 63b8421 | 2026-06-05 | Xiaozhe Yao | Xiaozhe Yao | Y: Claude Opus 4.8 (1M context) | - | - | - | - | Merge feat/bench_overhead into main |
| ca58f99 | 2026-06-05 | Xiaozhe Yao | Xiaozhe Yao | - | - | - | - | - | feat: add scheduling documentation and update meta.json |
| c05fd79 | 2026-04-12 | Xiaozhe Yao | Xiaozhe Yao | Y: Claude Opus 4.8 (1M context) | - | - | - | - | Add additional load balancing policies |
| 2599f89 | 2026-05-17 | Xiaozhe Yao | Xiaozhe Yao | - | - | - | - | Y | Fix/autoban orphan cid (#7) |
| 5caeeb6 | 2026-05-15 | Xiaozhe Yao | Xiaozhe Yao | - | - | - | - | Y | feat: crdt sync goes async now (#6) |
| f3fe012 | 2026-06-05 | Xiaozhe Yao | Xiaozhe Yao | - | - | - | - | - | Add benchmarking tools and tests for OpenTela scalability |
| b306501 | 2026-05-26 | Xiaozhe Yao | Xiaozhe Yao | - | - | - | - | - | update readme |
| 536a96f | 2026-05-26 | Xiaozhe Yao | Xiaozhe Yao | - | - | - | - | - | update readme |
| 8d55e66 | 2026-05-26 | Xiaozhe Yao | Xiaozhe Yao | - | - | - | - | - | update doc paths |
| 0d90147 | 2026-05-25 | Xiaozhe Yao | Xiaozhe Yao | - | - | - | - | - | use smaller models for eval |
| 7117c93 | 2026-05-22 | Xiaozhe Yao | Xiaozhe Yao | - | - | - | - | - | refactor notebooks |
| cb1f6ac | 2026-05-21 | Xiaozhe Yao | Xiaozhe Yao | - | - | - | - | - | improve contrib/benchmark |
| 3ae588d | 2026-05-14 | Xiaozhe Yao | Xiaozhe Yao | Y: Claude Sonnet 4.6 | - | - | - | - | chore(benchmark_convergence): gitignore runs/ directory |
| ac20065 | 2026-05-13 | Xiaozhe Yao | Xiaozhe Yao | Y: Claude Opus 4.7 (1M context) | - | - | - | - | feat(benchmark_convergence): uniform-random writer + 5ms poll |
| 9595576 | 2026-05-13 | Xiaozhe Yao | Xiaozhe Yao | Y: Claude Opus 4.7 (1M context) | - | - | - | - | fix(benchmark_convergence): handle libp2p TIME_WAIT between sweep cell |
| 62bbcb6 | 2026-05-13 | Xiaozhe Yao | Xiaozhe Yao | Y: Claude Opus 4.7 (1M context) | - | - | - | - | fix(benchmark_convergence): isolate mesh from prod + fix protocol/sock |
| 77c6d00 | 2026-05-13 | Xiaozhe Yao | Xiaozhe Yao | Y: Claude Opus 4.7 (1M context) | - | - | - | - | feat(benchmark_convergence): SSH-based driver as alternative to SLURM |
| f0c9755 | 2026-05-13 | Xiaozhe Yao | Xiaozhe Yao | Y: Claude Opus 4.7 (1M context) | - | - | - | - | fix(benchmark_convergence): populate started_at/finished_at; widen pos |
| 8f9e67b | 2026-05-13 | Xiaozhe Yao | Xiaozhe Yao | Y: Claude Opus 4.7 (1M context) | - | - | - | - | feat(benchmark_convergence): add doctor.py preflight |
| bd51051 | 2026-05-13 | Xiaozhe Yao | Xiaozhe Yao | Y: Claude Opus 4.7 (1M context) | - | - | - | - | feat(benchmark_convergence): add submit.py and run.py |
| bed0efe | 2026-05-13 | Xiaozhe Yao | Xiaozhe Yao | Y: Claude Opus 4.7 (1M context) | - | - | - | - | feat(benchmark_convergence): add task_entry.py + slurm/convergence.sba |
| 0133e4f | 2026-05-13 | Xiaozhe Yao | Xiaozhe Yao | Y: Claude Opus 4.7 (1M context) | - | - | - | - | feat(benchmark_convergence): implement coordinator, observer, protocol |
| 61a98ad | 2026-05-13 | Xiaozhe Yao | Xiaozhe Yao | Y: Claude Opus 4.7 (1M context) | - | - | - | - | feat(benchmark_convergence): implement text-only report.py |
| f50a74a | 2026-05-13 | Xiaozhe Yao | Xiaozhe Yao | Y: Claude Opus 4.7 (1M context) | - | - | - | - | feat(benchmark_convergence): implement aggregate.py (cell + run) |
| 9f04caf | 2026-05-13 | Xiaozhe Yao | Xiaozhe Yao | Y: Claude Opus 4.7 (1M context) | - | - | - | - | feat(benchmark_convergence): scaffold CLI, config, and tests |
| e90f136 | 2026-05-13 | Xiaozhe Yao | Xiaozhe Yao | Y: Claude Opus 4.7 (1M context) | - | - | - | - | test(integration): verify admin-route writes propagate over CRDT |
| 3ec75a9 | 2026-05-13 | Xiaozhe Yao | Xiaozhe Yao | Y: Claude Opus 4.7 (1M context) | - | - | - | - | feat(admin): add localhost POST /v1/_admin/register route |
| 910443e | 2026-05-13 | Xiaozhe Yao | Xiaozhe Yao | Y: Claude Opus 4.7 (1M context) | - | - | - | - | refactor(protocol): extract RegisterAdHocService helper from registrar |
| 07df974 | 2026-05-13 | Xiaozhe Yao | Xiaozhe Yao | Y: Claude Opus 4.7 (1M context) | - | - | - | - | feat(admin): add admin.enabled / admin.port config defaults and flags |
| 9968686 | 2026-05-13 | Xiaozhe Yao | Xiaozhe Yao | Y: Claude Opus 4.7 (1M context) | - | - | - | - | feat(benchmark_overhead): switch worker to vLLM/Pyxis, add fixed workl |
| 766c7e1 | 2026-05-11 | Xiaozhe Yao | Xiaozhe Yao | Y: Claude Opus 4.7 (1M context) | - | - | - | - | fix(libp2p): remove rcmgr stream caps that 502 head→worker forwarding |

Totals over these 100 (maintainer = Xiaozhe Yao / xzyao, 94 commits; other = Aryan Ahadinia 4, Rob 1, Mel Zuercher 1, 6 commits):
- AI Co-Authored-By naming Claude: maintainer 29, other 4 (119401e, 1111d5d, c7a4a96, 5c4bbdb). Models seen as trailer lines: Opus 5.5 x20, Opus 4.7 (1M context) x18, Opus 5 (1M context) x7, Opus 4.8 (1M context) x6, Sonnet 5.5 x4, Sonnet 4.6 x1 (lines, so squashed inner commits count each).
- Claude-Session: maintainer 8, other 2 (c7a4a96, 5c4bbdb). Entire-Checkpoint: maintainer 5, other 2 (c7a4a96, 5c4bbdb).
- Signed-off-by: 0. 'Generated with Claude Code' text in a commit message: 0.
- Human Co-authored-by: f740781 (maintainer's own noreply identity), 6c8574d and 19ae659 (Rob, credited by Aryan Ahadinia).
- Subject starts with a lowercase conventional type (optionally scoped): 76/100. Ends with (#N): 16/100 (14 committed by GitHub = squash merges; 2 by the maintainer himself: 2599f89 (#7), 5caeeb6 (#6)). 17 commits have committer GitHub, 3 of them web edits without a number (README).
- Parents: 99 single-parent, 1 merge (63b8421 Merge feat/bench_overhead into main). All history: 249 non-merge, 6 merge commits (63b8421, 44cb544, 182e2ea, 664450d, 8d4e1b5, 1a884b1).
- Bodies: 68 have a body, 32 are subject-only.

## Last 50 non-merge commits: size (files, add, del; lock files, go.sum and .ipynb excluded from add/del, see lockish)

| hash | date | author | files | add | del | lockish | subject |
|---|---|---|---|---|---|---|---|
| 4b6e90f | 2026-11-02 | Xiaozhe Yao | 12 | 149 | 149 | 8366 | fix(security): resolve Dependabot alerts across Go, npm and Python dep |
| f0eca38 | 2026-11-02 | Xiaozhe Yao | 18 | 399 | 33 | 0 | Auto link opt in (#69) |
| c7a4a96 | 2026-11-02 | Mel Zuercher | 15 | 519 | 37 | 0 | start: --ephemeral-identity and --datadir for nodes sharing a $HOME (# |
| 5c4bbdb | 2026-11-02 | Rob | 6 | 1383 | 37 | 0 | fix(protocol): worker connectivity watchdog retries bootstrap + relay  |
| f740781 | 2026-10-15 | Xiaozhe Yao | 2 | 9 | 0 | 14 | fix(protocol): never sweep or evict this node's own table row (#65) |
| 307ed15 | 2026-09-30 | Xiaozhe Yao | 1 | 1 | 0 | 0 | Update README with OpenTela adoption links |
| 00bb756 | 2026-09-24 | Xiaozhe Yao | 5 | 403 | 17 | 0 | feat(cli): persist deploy keys; auto-link at 'otela start' |
| b5fe983 | 2026-09-24 | Xiaozhe Yao | 3 | 410 | 1 | 0 | feat(cli): 'otela instance link' via scoped deploy keys (#5) |
| 3ba1806 | 2026-09-23 | Xiaozhe Yao | 2 | 29 | 1 | 0 | feat(acl): whitelist 'api_key' and 'no_match' reasons in the v2 evalua |
| 783409c | 2026-09-19 | Xiaozhe Yao | 1 | 1 | 0 | 0 | Update README with adoption examples for OpenTela |
| 42728f3 | 2026-09-19 | Xiaozhe Yao | 7 | 393 | 3 | 0 | feat(routing): session affinity — sticky peers for KV-cache warmth |
| 6f56f30 | 2026-09-18 | Xiaozhe Yao | 4 | 230 | 0 | 0 | feat(home): live network stats strip on the frontpage |
| 073cd3c | 2026-09-18 | Xiaozhe Yao | 4 | 61 | 3 | 0 | docs(cloud): add network statistics page with live snapshot |
| 778a8fa | 2026-09-17 | Xiaozhe Yao | 1 | 2 | 3 | 0 | fix(lint): expand math.Pow in allowed_peers_test (QF1005) |
| 0f4d3de | 2026-09-17 | Xiaozhe Yao | 9 | 408 | 65 | 0 | fix(protocol): make workers relayed by the node itself evictable |
| 7e28835 | 2026-09-14 | Xiaozhe Yao | 6 | 590 | 9 | 0 | boot: reconcile the node with its cloud account at startup (#3) |
| 61ccdb6 | 2026-09-14 | Xiaozhe Yao | 28 | 411 | 2838 | 0 | billing: retire the legacy settlement rail — the API market is the sin |
| b3d82d9 | 2026-09-14 | Xiaozhe Yao | 7 | 437 | 24 | 0 | Price-aware routing in the mesh head (Phase 1 market signal) (#1) |
| 119401e | 2026-09-06 | Aryan Ahadinia | 5 | 189 | 28 | 0 | feat(attestation): make the build-attestation trust anchor configurabl |
| 1111d5d | 2026-09-06 | Aryan Ahadinia | 2 | 68 | 4 | 0 | fix(bootstrap): explicit bootstrap peers must override the static list |
| 6c8574d | 2026-08-25 | Aryan Ahadinia | 12 | 86 | 36 | 2 | logging: reduce hot-path log noise, configurable stdout, harden two fa |
| b04d592 | 2026-08-25 | Xiaozhe Yao | 2 | 0 | 0 | 237 | chore: strip output from Jupyter notebooks |
| 80cd709 | 2026-08-25 | Xiaozhe Yao | 151 | 22 | 11318 | 7973 | Refactor code structure for improved readability and maintainability |
| 935a875 | 2026-08-25 | Xiaozhe Yao | 1 | 9 | 1 | 0 | ci: make coverage-comment step non-fatal for fork PRs |
| 19ae659 | 2026-08-25 | Aryan Ahadinia | 3 | 45 | 0 | 0 | feat: peer metadata (hostname, version, labels, pending/ready status)  |
| c32ee69 | 2026-08-22 | Xiaozhe Yao | 48 | 44 | 9805 | 5648 | docs: remove account console, extracted to cloud.opentela.ai |
| f1b645d | 2026-08-22 | Xiaozhe Yao | 6 | 153 | 7 | 0 | feat(start): add `otela start --wallet.address` to run as a chosen man |
| a94f2f6 | 2026-08-22 | Xiaozhe Yao | 6 | 652 | 8 | 6 | feat(cli): add `otela wallet link` to bind a wallet to a cloud account |
| 44e2db9 | 2026-08-17 | Xiaozhe Yao | 2 | 36 | 3 | 0 | docs: hide proposals from sidebar and search |
| a358872 | 2026-08-16 | Xiaozhe Yao | 17 | 357 | 64 | 0 | docs(site): a11y + UX audit fixes for the documentation site |
| f29c6b7 | 2026-08-16 | Xiaozhe Yao | 10 | 556 | 6 | 0 | docs(account): email-verification step for Neon Auth sign-up |
| 1628c10 | 2026-08-16 | Xiaozhe Yao | 2 | 53 | 0 | 0 | docs(account): graceful empty state when billing mode is off |
| ab922e8 | 2026-08-16 | Xiaozhe Yao | 1 | 1 | 1 | 0 | docs(env.example): update Neon Auth URL to the live instance |
| ce5611f | 2026-08-16 | Xiaozhe Yao | 11 | 2055 | 5 | 0 | billing: mesh peer filtering by billing affordability + account billin |
| a5e6529 | 2026-08-14 | Xiaozhe Yao | 10 | 473 | 3 | 0 | update docs |
| e059134 | 2026-08-06 | Xiaozhe Yao | 4 | 288 | 0 | 0 | docs(cloud): add Public Cloud setup guide for the api.opentela.ai stac |
| 7928586 | 2026-08-05 | Xiaozhe Yao | 2 | 43 | 3 | 0 | docs(cf): strip unused @vercel/og from Worker bundle to fit free-tier  |
| a612337 | 2026-08-05 | Xiaozhe Yao | 22 | 2090 | 70 | 58 | feat: mesh observatory docs page + protocol liveness hardening |
| 1c54345 | 2026-07-30 | Xiaozhe Yao | 1 | 19 | 0 | 0 | Update README.md |
| e9d1696 | 2026-07-29 | Xiaozhe Yao | 2 | 11 | 21 | 0 | fix(ci): synchronize autorelay host publication |
| 9fac744 | 2026-07-29 | Xiaozhe Yao | 1 | 5 | 1 | 0 | fix(shutdown): drain the LEFT announcement before wiping the CRDT stor |
| 7f42183 | 2026-07-29 | Xiaozhe Yao | 1 | 10 | 2 | 0 | fix(p2p): dial the target peer, not the incoming Host header |
| 43faf54 | 2026-07-29 | Xiaozhe Yao | 7 | 400 | 84 | 0 | docs: explain and align trusted region workflow |
| f6a49c9 | 2026-07-29 | Xiaozhe Yao | 31 | 4654 | 331 | 0 | feat: enforce trusted-region service policies |
| 5a95896 | 2026-07-29 | Xiaozhe Yao | 20 | 3545 | 112 | 0 | chore: snapshot ACL client baseline |
| 0c88314 | 2026-07-27 | Xiaozhe Yao | 3 | 20 | 0 | 304 | update docs versions |
| 49e39e6 | 2026-07-26 | Xiaozhe Yao | 40 | 2880 | 1507 | 0 | feat(docs): standalone /account console |
| 8d1eb88 | 2026-07-25 | Xiaozhe Yao | 1 | 4 | 1 | 0 | fix(docs): suppressHydrationWarning on <body> for extension-injected a |
| 7c8f72c | 2026-07-24 | Xiaozhe Yao | 2 | 29 | 16 | 0 | fix(docs): clear sk- key on sign-out, reactive session + redirect, rev |
| 52a1f2c | 2026-07-24 | Xiaozhe Yao | 1 | 3 | 0 | 0 | chore(docs): verify build + document /account page |

Summary: all 50: files median 4.5 (1-151), lines median 223.5 (0-11340). Maintainer 44: files median 4 (1-151), lines median 259. Other authors 6: files median 5.5 (2-15), lines median 169.5 (45-1420). Last 30: files median 5.5, lines median 264. The 151-file / 11340-line maximum is 80cd709 (maintainer refactor, mostly deletions); 4b6e90f (#70) is 12 files, 3659+/5005- including lock files (about 8.4k lines of lockfile churn).

## Outside (non-maintainer) merged commits, all history, by size (files / add / del / lockish)

- 5c4bbdb Rob #67 (2026-11-02): 6 / 1383 / 37 / 0 (largest by lines; 366-line integration test included)
- 2c64ebc Elia Palme #10 (2025-08-13): 40 / 1191 / 194 / 169 (largest by files; Elia is a COLLABORATOR)
- c7a4a96 Mel Zuercher #68: 15 / 519 / 37; 1864ce2 Rob #3 (2026-04-08): 3 / 267 / 96; 27d17d1 Elia #6: 10 / 291 / 2; 119401e Aryan #64: 5 / 189 / 28; 6c8574d Aryan #61: 12 / 86 / 36; 1111d5d Aryan #63: 2 / 68 / 4; 19ae659 Aryan #60: 3 / 45 / 0.

## Pull requests (GitHub API, newest 50, saved at scratch pulls.json)

| # | state | author (assoc) | head repo:branch | title | body headings | created to merged |
|---|---|---|---|---|---|---|
| 72 | open | JeniaJitsev (NONE) | JeniaJitsev/OpenTela:fix/llm-restore-identity-groups | fix(protocol): keep model identity groups when a withdrawn llm service | ## Problem; ## Observed; ## Fix; ## Tests; ## Separate observation, not addressed here | open |
| 70 | closed | xzyaoi (COLLABORATOR) | eth-easl/OpenTela:fix/dependabot-security-updates | fix(security): resolve Dependabot alerts across Go, npm and Python dep | ## Summary; ### `src/go.mod` (Go); ### `docs/` (npm); ### `contrib/` (Python); ## Validation | 0.2 h |
| 69 | closed | xzyaoi (COLLABORATOR) | eth-easl/OpenTela:auto-link-opt-in | Auto link opt in |  | 0.1 h |
| 68 | closed | mzueri (CONTRIBUTOR) | mzueri/OpenTela:ephemeral-identity-datadir | start: --ephemeral-identity and --datadir for nodes sharing a $HOME | ## Problem; ## Change; ## For the maintainers to weigh; ## Testing | 245.1 h |
| 67 | closed | robmsmt (COLLABORATOR) | robmsmt/OpenTela:fix/worker-connectivity-watchdog | fix(protocol): worker connectivity watchdog retries bootstrap + relay  | ## Problem; ## Change; ## Testing | 337.3 h |
| 65 | closed | xzyaoi (COLLABORATOR) | eth-easl/OpenTela:fix/self-row-sweep | fix(protocol): never sweep or evict this node's own node-table row | ## Problem; ## Root cause; ## Fix; ## Evidence (production, ocf-1 head @ 140.238.223.116); ## Testing | 3.7 h |
| 64 | closed | AryanAhadinia (CONTRIBUTOR) | swiss-ai/OpenTela:feat/configurable-build-pubkeys | feat(attestation): make the build-attestation trust anchor configurabl | ## Problem; ## Change; ## Details worth reviewing; ## Tests; ## Two things I noticed while working here — happy to file separately | 42.7 h |
| 63 | closed | AryanAhadinia (CONTRIBUTOR) | swiss-ai/OpenTela:fix/bootstrap-explicit-precedence | fix(bootstrap): explicit bootstrap peers must override the static disc | ## Problem; ## The intent already exists in the code; ## Change; ## Tests | 42.6 h |
| 61 | closed | AryanAhadinia (CONTRIBUTOR) | swiss-ai/OpenTela:chore/log-noise | logging: reduce hot-path log noise, configurable stdout, harden two fa | ## What; ## Notes | 5.0 h |
| 60 | closed | AryanAhadinia (CONTRIBUTOR) | swiss-ai/OpenTela:feat/peer-metadata | feat: peer metadata (hostname, version, labels, pending/ready status) | ## What; ## Why; ## Design notes | 1.8 h |
| 57 | closed | robmsmt (COLLABORATOR) | swiss-ai/OpenTela:feat/peer-metadata | Feat/peer metadata |  | closed unmerged |
| 56 | closed | xzyaoi (COLLABORATOR) | eth-easl/OpenTela:net-profiler | Implement mesh-bench |  | 166.1 h |
| 53 | closed | xzyaoi (COLLABORATOR) | eth-easl/OpenTela:fix/code-review-findings | Refactor usage record handling |  | 0.0 h |
| 54 | closed | xzyaoi (COLLABORATOR) | eth-easl/OpenTela:feat/fleet-manager | Implement otela-fleet project with CLI, templates, and deployment |  | 78.8 h |
| 51 | closed | xzyaoi (COLLABORATOR) | eth-easl/OpenTela:fix/security | Infra Improvements |  | 71.9 h |
| 52 | closed | dependabot[bot] (CONTRIBUTOR) | eth-easl/OpenTela:dependabot/go_modules/src/google.golang.org/grpc-1.79.3 | Bump google.golang.org/grpc from 1.79.1 to 1.79.3 in /src |  | 0.7 h |
| 50 | closed | xzyaoi (COLLABORATOR) | ResearchComputer/OpenTela:main | Add Cloudflare Workers configuration (#1) |  | 0.0 h |
| 49 | closed | xzyaoi (COLLABORATOR) | eth-easl/OpenTela:fix/dev_test_bugs | Fix/dev test bugs |  | 47.9 h |
| 48 | closed | xzyaoi (COLLABORATOR) | eth-easl/OpenTela:feat/scalability | Feat/scalability |  | 1.3 h |
| 47 | closed | xzyaoi (COLLABORATOR) | eth-easl/OpenTela:feat/general | Overall improvements |  | 4.8 h |
| 26 | closed | xzyaoi (COLLABORATOR) | eth-easl/OpenTela:feat/grpc | feat: Implement gRPC forwarding over P2P using HTTP/2 Cleartext (h2c)… |  | closed unmerged |
| 41 | closed | Copilot (CONTRIBUTOR) | eth-easl/OpenTela:copilot/sub-pr-40 | Enforce challenge freshness, wallet binding, and replay prevention in  | ## Changes; # Valid challenge format; ## Model addition; ## Tests | 8.0 h |
| 42 | closed | Copilot (CONTRIBUTOR) | eth-easl/OpenTela:copilot/sub-pr-40-again | Fix verifyBearerToken: bound auth latency with timeout and context pro | ## Changes | 8.0 h |
| 40 | closed | xzyaoi (COLLABORATOR) | eth-easl/OpenTela:docs/init | Initialize documents, security improvements; |  | 8.7 h |
| 43 | closed | Copilot (CONTRIBUTOR) | eth-easl/OpenTela:copilot/sub-pr-40-another-one | Split Peer.Owner (wallet pubkey) from Peer.ProviderID to fix access-co | ## Changes | 8.0 h |
| 44 | closed | Copilot (CONTRIBUTOR) | eth-easl/OpenTela:copilot/sub-pr-40-yet-again | Exclude locally-computed Peer fields from CRDT serialization | ## Changes | 7.9 h |
| 45 | closed | Copilot (CONTRIBUTOR) | eth-easl/OpenTela:copilot/sub-pr-40-one-more-time | Remove X-Otela-Client-Wallet from CORS allowed headers | ## Changes | 0.5 h |
| 39 | closed | xzyaoi (COLLABORATOR) | eth-easl/OpenTela:feat/usage_settlement | Feat/usage settlement |  | 0.1 h |
| 36 | closed | xzyaoi (COLLABORATOR) | eth-easl/OpenTela:feat/owner_and_peer_authing | wip: ownership model |  | 48.7 h |
| 35 | closed | xzyaoi (COLLABORATOR) | eth-easl/OpenTela:feat/fleet_manager | Initial version of fleet manager |  | 51.2 h |
| 33 | closed | Copilot (CONTRIBUTOR) | eth-easl/OpenTela:copilot/add-arbitrary-annotations-feature | Add --annotation flag for arbitrary node annotations in DNT |  | closed unmerged |
| 34 | closed | xzyaoi (COLLABORATOR) | eth-easl/OpenTela:fix/put-requests-not-forwarded | forward put requests as well |  | 0.0 h |
| 30 | closed | xzyaoi (COLLABORATOR) | eth-easl/OpenTela:feat/#29 | Improve routing logic to be more flexible |  | 26.6 h |
| 31 | closed | Copilot (CONTRIBUTOR) | eth-easl/OpenTela:copilot/sub-pr-30 | Extract routing helpers and add unit tests for identity-group matching | ## Changes | 3.5 h |
| 28 | closed | xzyaoi (COLLABORATOR) | eth-easl/OpenTela:chore/renaming | Renaming from OpenFabric to OpenTela to avoid confusion |  | 0.3 h |
| 27 | closed | xzyaoi (COLLABORATOR) | eth-easl/OpenTela:feat/update-docs | Update documents & README |  | 3.6 h |
| 25 | closed | xzyaoi (COLLABORATOR) | eth-easl/OpenTela:feat/perf | Performance Improvements |  | 11.9 h |
| 22 | closed | xzyaoi (COLLABORATOR) | eth-easl/OpenTela:fix/#21 | fix #21 |  | 2.8 h |
| 24 | closed | Copilot (CONTRIBUTOR) | eth-easl/OpenTela:copilot/sub-pr-22-again | Fix potential panic from using host before error check in GetP2PNode |  | 2.5 h |
| 23 | closed draft | Copilot (CONTRIBUTOR) | eth-easl/OpenTela:copilot/sub-pr-22 | Add configuration and logging improvements | ## Changes; ## Example; # Now supports environment variables; ## Dependencies | closed unmerged |
| 20 | closed draft | google-labs-jules[bot] (NONE) | eth-easl/OpenTela:docs-improvement-1799446929310627961 | Improve documentation structure and content |  | closed unmerged |
| 19 | closed | xzyaoi (COLLABORATOR) | eth-easl/OpenTela:feat/wallet | Feat/wallet |  | 0.0 h |
| 18 | closed | theely (COLLABORATOR) | eth-easl/OpenTela:Add-image-release | Add image release |  | 0.4 h |
| 17 | closed | theely (COLLABORATOR) | eth-easl/OpenTela:fix-internal-state-update | Fixes the internal state update |  | 12.3 h |
| 15 | closed | theely (COLLABORATOR) | eth-easl/OpenTela:Ghost-peers-handling | Ghost peers handling |  | 129.3 h |
| 14 | closed | theely (COLLABORATOR) | eth-easl/OpenTela:Fix-race-condition | fix race condition bug |  | 129.3 h |
| 16 | closed | theely (COLLABORATOR) | eth-easl/OpenTela:Compacted-logs | Compacted Logs |  | 125.4 h |
| 13 | closed | theely (COLLABORATOR) | theely/OpenComputeFramework:Compacted-logs | Compacted logs |  | closed unmerged |
| 12 | closed | dependabot[bot] (CONTRIBUTOR) | eth-easl/OpenTela:dependabot/go_modules/src/golang.org/x/net-0.38.0 | Bump golang.org/x/net from 0.35.0 to 0.38.0 in /src |  | 0.1 h |
| 11 | closed | dependabot[bot] (CONTRIBUTOR) | eth-easl/OpenTela:dependabot/go_modules/src/github.com/pion/interceptor-0.1.39 | Bump github.com/pion/interceptor from 0.1.37 to 0.1.39 in /src |  | 0.1 h |

Labels: only dependabot PRs (#11, #12, #52) carry labels (dependencies, go). Drafts: #23 (Copilot) and #20 (google-labs-jules[bot]); no maintainer or human outside draft in these 50.
PR merged_by (single-PR API calls): #67 and #68 both xzyaoi, merged 5 s apart on 2026-11-02 14:10 UTC. Median created-to-merged: maintainer PRs 3.6 h (n=22); outside/collaborator PRs (mzueri, robmsmt, AryanAhadinia, theely) 42.7 h, range 0.4 h to 337 h (n=11).

## Remote branches in the clone (git branch -r)
origin/main, origin/legacy, origin/auto-link-opt-in, origin/copilot/sub-pr-22, origin/fix/dependabot-security-updates, origin/fix/ghost-peer-liveness, origin/fix/self-row-sweep. Tags v0.2.3 to v0.2.7 seen.
