# Response to the audit

12 Nov 2026. [AUDIT.md](AUDIT.md) is left as the auditor wrote it. This table says what was done about each finding. "Accepted" means the fix the auditor suggested, or an equivalent, is in the files now.

| Finding | Severity | Action |
|---|---|---|
| A1 Isolation dropped from D11 and D7 | high | Accepted. D11 now requires default-deny egress for sessions that work in clones or run upstream code, public inputs only for web research, and a container or microVM for unattended loops and unwatched auto-mode sessions. D7 requires the container. The playbook row now reads "Kept, tightened" with the container rule named. MIGRATION.md adds a sandbox settings file and lists the container runtime as the user's decision. |
| A2 `measure.py` overwrote the pilot's baseline | high | Accepted. Records carry a block; pairs are numbered inside a block; a duplicate block, arm and pair is refused; left-out runs are counted and printed. The self-test replays the failing case (16 checks pass). |
| A3 "True on this machine today" uncited | medium | Accepted. D11 states what was checked on 12 Nov and what was not (the login keychain), and `make check` will repeat the check. |
| A4 "46 to 93%" rests on secondary figures | medium | Accepted. The map uses E20's verified figures. RESEARCH.md has a corrections section for H21 and L27, and lists them as secondary reading. |
| A5 "Fewer pairs" contradicts the sign test | medium | Accepted. D6 keeps 6 pairs for any comparison, and adds the escalation to 10 pairs and the stop. |
| A6 "Short" presented as evidence-based | medium | Accepted. D8 calls it a cost rule, and the length null and the per-model result are under Against. |
| A7 "Rituals dropped" contradicts D9 | medium | Accepted. Verdict changed to "Changed"; D9 says STATUS.md is the handoff. |
| A8 All idea mechanisms dropped on thin evidence | medium | Accepted. Only tournaments and councils are dropped; the rest are "not evaluated"; the surprise scan stays with predictions. |
| A9 D2 stated as settled | medium | Accepted. D2 is framed as a review budget, E23's contrary number and V35 are under Against. |
| A10 Notes disagree on upstream instruction files | medium | Accepted. A map row records it, and D11 states the trade-off and the patch-session exception. |
| A11 Doctrine and template name tools that do not exist | medium | Accepted in part. The tools are what MIGRATION.md builds. The doctrine and the template README now say so, and D13 separates what the migration builds from what comes next. After the migration the note in the doctrine's first paragraph can go. |
| A12 Saturation summary overstated | medium | Accepted. Rewritten per decision from the notes' own statements. |
| A13 Template lacked decision rule, maximum n, escalation and record fields | medium | Accepted. ASSUMPTIONS.md and MODAL.md have them; `record` stores phases, exit code and the harness digest. |
| A14 Monthly credit, academic credits | low | Accepted in D10. |
| A15 X21 and X20 cited for the wrong thing | low | Accepted. X21 moved to D6; X20 stays for the Lustre sentence. |
| A16 "I decide" against "ask" | low | Accepted. D12 says ask; without an answer, a sentence, and a trailer only if the history shows one. |
| A17 V35 and the null missing from D12 | low | Accepted. |
| A18 E16's pattern is suggestive | low | Accepted; D12 says so. |
| A19 "Most of AGENTS.md" | low | Accepted; D14 names only the lines that work around a model. |
| A20 Ten notes, not nine | low | Accepted in RESEARCH.md; DECISIONS.md has a correcting entry. |
| A21 Secondary-reading list incomplete | low | Accepted: S02, B14, H21, L27 and the X rows added. |
| A22 "15 times the tokens" | low | Accepted: of a chat; about 4 times a single agent. |
| A23 Lulla called observational | low | Accepted: described as a paired comparison measuring efficiency. The ledger's two grades for one paper are noted under corrections; the notes are not edited. |
| A24 Two disagreements missing from the map | low | Accepted: kill early against a bad start, and compaction inside a task. |
| A25 H08 row describes an older version | low | Accepted as a correction in RESEARCH.md. The direction, which is all D4 uses, is unchanged. |
| A26 "Both stale" | low | Accepted: one stale, one incomplete. |
| A27 Playbook rows unfair | low | Accepted: "Kept in part" for the instruction file and the context layers. |
| A28 No remote leaves one copy | low | Accepted. The verdict is "Changed", and a backup is in MIGRATION.md as the user's decision, with `make backup`. |
| A29 Playbook rules missing from the table | low | Accepted: `gh` draft PRs and agent keys (dropped), plan mode (with the three-session row), containers (with the trifecta row). |
| A30 Verdict printed regardless of p | low | Accepted. The verdict is computed: DESCRIPTIVE, NO-PILOT, NOT DISTINGUISHABLE or SUPPORTED. |

Not changed: the eleven topic notes. They are the agents' record, and corrections to them live in RESEARCH.md.

Still open after the audit: V22, V23, S13, S16 and R12 were not fetched by anyone (blocked or relayed). About 40 of the 113 cited rows were not checked against their notes by the auditor.
