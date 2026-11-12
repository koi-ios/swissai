#!/usr/bin/env python3
"""Deterministic leak check for a series that is about to go upstream.

  leakcheck.py WORKTREE --repo NAME [--base origin/main]

Looks at every commit in base..HEAD of the worktree and at the lines the series adds. Fails on:
private paths and strings, scaffolding files, an identity that is not the one in
pipeline/workspace.toml, placeholder identities, trailers the repository's rules refuse or require,
a subject or branch name that breaks the repository's pattern, and anything that looks like a secret.
Rules: upstream/NAME.toml and pipeline/workspace.toml. It reads; it never writes into the worktree.
"""
from __future__ import annotations

import argparse
import pathlib
import re
import sys

from lib import ROOT, git, load_toml, workspace

AI = re.compile(r"claude|copilot|cursor|codex|gpt|gemini|anthropic|openai|devin", re.I)
SECRETS = [r"hf_[A-Za-z0-9]{20,}", r"sk-[A-Za-z0-9_-]{20,}", r"ghp_[A-Za-z0-9]{20,}", r"github_pat_[A-Za-z0-9_]{20,}",
           r"AKIA[0-9A-Z]{16}", r"-----BEGIN [A-Z ]*PRIVATE KEY-----", r"xox[baprs]-[A-Za-z0-9-]{10,}"]


def trailers(message: str) -> list[tuple[str, str]]:
    """Key and value of each trailer line in the last paragraph of a commit message."""
    block = message.rstrip().split("\n\n")[-1]
    found = []
    for line in block.splitlines():
        m = re.match(r"^([A-Za-z][A-Za-z-]*): (.+)$", line)
        if m:
            found.append((m.group(1), m.group(2).strip()))
    return found


def run(worktree: pathlib.Path, base: str, rules: dict, ws: dict) -> tuple[list[str], list[str], dict]:
    fails: list[str] = []
    warns: list[str] = []
    ident = ws["identity"]
    leak = ws["leak"]
    who = f"{ident['name']} <{ident['email']}>"
    private = leak["private_strings"]

    def private_hits(text: str) -> list[str]:
        low = text.lower()
        return [s for s in private if s.lower() in low]

    commits = git("rev-list", "--reverse", f"{base}..HEAD", cwd=worktree).split()
    if not commits:
        fails.append(f"no commits between {base} and HEAD")
    ai_seen = 0
    for c in commits:
        an, ae, cn, ce, subject = git("show", "-s", "--format=%an%x00%ae%x00%cn%x00%ce%x00%s", c, cwd=worktree).split("\0")
        body = git("show", "-s", "--format=%B", c, cwd=worktree)
        tag = c[:9]
        for label, name, email in (("author", an, ae), ("committer", cn, ce)):
            shown = f"{name} <{email}>"
            if any(p.lower() in shown.lower() for p in leak["placeholder_identities"]):
                fails.append(f"{tag}: {label} is a placeholder: {shown}")
            elif shown != who:
                fails.append(f"{tag}: {label} is {shown}, expected {who}")
        if not re.search(rules["subject_pattern"], subject):
            fails.append(f"{tag}: subject does not match the repository's pattern: {subject!r}")
        if len(subject) > rules.get("subject_max", 100):
            fails.append(f"{tag}: subject is {len(subject)} characters, over {rules.get('subject_max', 100)}")
        if rules.get("no_pr_suffix") and re.search(r"\(#\d+\)\s*$", subject):
            fails.append(f"{tag}: subject ends in a PR number; the squash merge adds it")
        tr = trailers(body)
        signoffs = [v for k, v in tr if k.lower() == "signed-off-by"]
        policy = rules.get("signoff", "absent")
        if policy == "required" and who not in signoffs:
            fails.append(f"{tag}: Signed-off-by: {who} is required here and missing. Add it yourself: "
                         f"git rebase --signoff {base}")
        if policy == "absent" and signoffs:
            fails.append(f"{tag}: has Signed-off-by, which this repository does not use")
        if signoffs and any(s != who for s in signoffs):
            fails.append(f"{tag}: a Signed-off-by names someone other than {who}")
        ai = [v for k, v in tr if k.lower() == "co-authored-by" and AI.search(v)]
        ai_seen += bool(ai)
        if rules.get("ai_trailer") == "forbid" and ai:
            fails.append(f"{tag}: AI co-author trailer present, which this repository forbids: {ai[0]}")
        for k, v in tr:
            if k in rules.get("drop_trailers", []):
                fails.append(f"{tag}: trailer {k} must not go upstream")
        for s in private_hits(body):
            fails.append(f"{tag}: commit message contains a private string: {s!r}")
    if rules.get("ai_trailer") == "keep" and commits and ai_seen < len(commits):
        warns.append(f"{len(commits) - ai_seen} of {len(commits)} commits carry no AI co-author trailer; this repository's "
                     "maintainers use one on AI-assisted commits. Decide per the card.")

    path, line_no = "", 0
    for line in git("diff", "--unified=0", f"{base}..HEAD", cwd=worktree).splitlines():
        if line.startswith("+++ "):
            path = line[6:] if line.startswith("+++ b/") else line[4:]
        elif line.startswith("@@"):
            m = re.search(r"\+(\d+)", line)
            line_no = int(m.group(1)) - 1 if m else 0
        elif line.startswith("+"):
            line_no += 1
            for s in private_hits(line):
                fails.append(f"{path}:{line_no}: added line contains a private string: {s!r}")
            for pat in SECRETS:
                if re.search(pat, line):
                    fails.append(f"{path}:{line_no}: added line looks like a secret ({pat[:12]}...)")
    for row in git("diff", "--name-status", f"{base}..HEAD", cwd=worktree).splitlines():
        status, _, name = row.partition("\t")
        base_name = name.rsplit("/", 1)[-1]
        if status.startswith("A") and (base_name in leak["scaffold_files"] or base_name in ("AGENTS.md", "CLAUDE.md")
                                       or any(d in name + "/" for d in leak["scaffold_dirs"])):
            fails.append(f"{name}: a scaffolding file is added by the series")
        elif base_name in ("AGENTS.md", "CLAUDE.md") or any(d in name for d in leak["scaffold_dirs"]):
            warns.append(f"{name}: the series changes an agent file of the upstream repository")
    files = added = deleted = 0
    for row in git("diff", "--numstat", f"{base}..HEAD", cwd=worktree).splitlines():
        a, d, name = row.split("\t", 2)
        files += 1
        if a == "-":
            warns.append(f"{name}: binary file in the series")
        else:
            added += int(a)
            deleted += int(d)
    if added + deleted > rules.get("max_lines_warn", 1000):
        warns.append(f"the series changes {added + deleted} lines; this repository's usual size is below "
                     f"{rules.get('max_lines_warn', 1000)}. Consider an issue or a message first.")
    branch = git("symbolic-ref", "--short", "-q", "HEAD", cwd=worktree, check=False)
    if not branch or not re.search(rules["branch_pattern"], branch):
        fails.append(f"branch name {branch or '(detached)'} does not match the repository's pattern")
    if git("status", "--porcelain", cwd=worktree):
        warns.append("the worktree has uncommitted changes; they are not part of the series")
    stats = {"commits": len(commits), "files": files, "added": added, "deleted": deleted, "branch": branch,
             "ai_trailers": ai_seen}
    return fails, warns, stats


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("worktree")
    ap.add_argument("--repo", required=True, help="card name, e.g. OpenTela")
    ap.add_argument("--base", default="origin/main")
    a = ap.parse_args()
    fails, warns, stats = run(pathlib.Path(a.worktree).resolve(), a.base, load_toml(ROOT / f"upstream/{a.repo}.toml"), workspace())
    for w in warns:
        print("warn  " + w)
    for f in fails:
        print("FAIL  " + f)
    print(f"leakcheck: {stats['commits']} commit(s), {stats['files']} file(s), +{stats['added']} -{stats['deleted']}; "
          f"{len(fails)} failure(s), {len(warns)} warning(s)")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
