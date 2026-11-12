#!/usr/bin/env python3
"""make export PROJECT=NN [REPO=name]: prepare a patch series for an upstream repository. Never pushes.

  export.py 06                     the project's only upstream block
  export.py 05 --repo sglang       pick one block
  export.py 06 --offline           do not fetch the canonical clone first
  export.py 06 --fresh             keep the previous export aside and start again

Steps: (1) a fresh worktree from the canonical clone's origin/main, on a branch the card allows;
(2) apply the tracked series; (3) set author and committer to the identity in pipeline/workspace.toml and
drop trailers the card refuses (a sign-off is never added: that is the user's act); (4) run the
repository's own checks with credentials removed from the environment; (5) the leak check;
(6) check export/<repo>/PR.md against the repository's headings; (7) write REPORT.md and CHECKLIST.md.
Everything is written under opportunities/NN/export/<repo>/ and opportunities/NN/vendor/export/<repo>/.
"""
from __future__ import annotations

import argparse
import datetime as dt
import glob
import os
import pathlib
import re
import subprocess
import sys

import leakcheck
from lib import ROOT, card_name, clean_env, die, find_project, git, load_toml, workspace


def sh(cmd: list[str], cwd: pathlib.Path, env: dict | None = None) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, cwd=cwd, env=env, capture_output=True, text=True)


def patch_header(path: str) -> tuple[str, str]:
    """Subject and body of a mail-format patch, or ('', '') for a plain diff."""
    text = pathlib.Path(path).read_text(encoding="utf-8", errors="replace")
    head = text.split("\n---\n", 1)[0] if "\ndiff --git" in text else ""
    m = re.search(r"^Subject: (?:\[PATCH[^\]]*\] )?(.*(?:\n .*)*)", head, re.M)
    if not m:
        return "", ""
    subject = re.sub(r"\n ", " ", m.group(1)).strip()
    body = head.split("\n\n", 1)[1].strip() if "\n\n" in head else ""
    return subject, body


def clean_message(message: str, rules: dict, subject: str | None) -> str:
    lines = message.rstrip().splitlines()
    if subject:
        lines[0] = subject
    drop = tuple(f"{t}:" for t in rules.get("drop_trailers", []))
    kept = [ln for ln in lines if not ln.startswith(drop)]
    if rules.get("ai_trailer") == "forbid":
        kept = [ln for ln in kept if not (ln.lower().startswith("co-authored-by:") and leakcheck.AI.search(ln))]
    kept = [ln for ln in kept if not ln.lower().startswith("signed-off-by:")]   # only the user signs off
    while kept and not kept[-1].strip():
        kept.pop()
    return "\n".join(kept) + "\n"


def check_pr(path: pathlib.Path, rules: dict, ws: dict) -> tuple[list[str], list[str]]:
    fails, warns = [], []
    if not path.exists():
        heads = "\n\n".join(f"## {group[0]}\n\nTODO" for group in rules.get("pr_sections", []))
        path.write_text(f"# TODO: the PR title (the commit subject)\n\n{heads}\n", encoding="utf-8")
        return [f"{path.name} did not exist; a skeleton was written. Fill it in and run the export again."], warns
    text = path.read_text(encoding="utf-8")
    headings = [h.strip().lower() for h in re.findall(r"^#{2,3} +(.+)$", text, re.M)]
    for group in rules.get("pr_sections", []):
        if not any(h.startswith(g.lower()) for g in group for h in headings):
            fails.append(f"PR.md lacks a section the repository uses: {' or '.join(group)}")
    if "TODO" in text:
        fails.append("PR.md still contains TODO")
    for s in ws["leak"]["private_strings"]:
        if s.lower() in text.lower():
            fails.append(f"PR.md contains a private string: {s!r}")
    for must in rules.get("pr_must_contain", []):
        if must.lower() not in text.lower():
            fails.append(f"PR.md must contain {must!r} (the repository's rule)")
    footer = "generated with [claude code]" in text.lower()
    if rules.get("pr_footer") == "forbid" and footer:
        fails.append("PR.md carries a 'Generated with Claude Code' line, which this repository does not want")
    if rules.get("pr_footer") == "keep" and not footer and "ai assist" not in text.lower():
        warns.append("PR.md has no AI-use statement; outside contributors to this repository add one")
    for para in re.split(r"\n\s*\n", text):
        if re.search(r"\bmedian", para, re.I) and not re.search(r"\bn ?= ?\d|\b\d+ runs\b|\bof \d+ runs|\bruns per\b", para, re.I):
            warns.append("a paragraph in PR.md gives a median without the number of runs: " + para.strip()[:70])
    return fails, warns


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("project")
    ap.add_argument("--repo")
    ap.add_argument("--offline", action="store_true")
    ap.add_argument("--fresh", action="store_true")
    ap.add_argument("--skip-checks", action="store_true")
    a = ap.parse_args()

    ws = workspace()
    proj = find_project(a.project)
    blocks = [u for u in proj["card"].get("upstream", []) if not a.repo or card_name(u["repo"]) == a.repo]
    if len(blocks) != 1:
        die(f"{proj['slug']}: pick one upstream block with --repo; it has {[card_name(u['repo']) for u in proj['card'].get('upstream', [])]}")
    up = blocks[0]
    name = card_name(up["repo"])
    rules = load_toml(ROOT / f"upstream/{name}.toml")
    ident = ws["identity"]
    who = f"{ident['name']} <{ident['email']}>"
    canon = ROOT / "vendor" / name
    if not (canon / ".git").exists():
        die(f"vendor/{name} is missing; create it with: python3 scripts/clones.py ensure {name}")
    branch = up.get("export_branch", "")
    if not re.search(rules["branch_pattern"], branch):
        die(f"export_branch {branch!r} does not match {name}'s branch pattern {rules['branch_pattern']}")
    patches = sorted(glob.glob(str(proj["dir"] / up["series"])))
    if not patches:
        die(f"series {up['series']!r} matches nothing in {proj['slug']}")
    subjects = up.get("subjects", [])

    base_ref = f"origin/{rules.get('default_branch', 'main')}"
    if not a.offline:
        print(f"export: fetching {up['repo']} (read-only)")
        git("fetch", "--prune", "origin", cwd=canon)
    base = git("rev-parse", base_ref, cwd=canon)
    work = proj["dir"] / "vendor/export" / name
    out = proj["dir"] / "export" / name
    stamp = dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    if work.exists() or out.exists() or git("rev-parse", "-q", "--verify", f"refs/heads/{branch}", cwd=canon, check=False):
        if not a.fresh:
            die(f"an earlier export exists ({work.relative_to(ROOT)} or branch {branch}). Rerun with --fresh: "
                "the old worktree, branch and output are kept under a dated name, not deleted.")
        if work.exists():
            git("worktree", "move", str(work), f"{work}-{stamp}", cwd=canon)
        if git("rev-parse", "-q", "--verify", f"refs/heads/{branch}", cwd=canon, check=False):
            git("branch", "-m", branch, f"{branch}-old-{stamp}", cwd=canon)
        if out.exists():
            keep = out / "PR.md"
            pr_text = keep.read_text(encoding="utf-8") if keep.exists() else None
            out.rename(out.with_name(f"{out.name}-{stamp}"))
            out.mkdir(parents=True)
            if pr_text is not None:
                keep.write_text(pr_text, encoding="utf-8")
    work.parent.mkdir(parents=True, exist_ok=True)
    (out / "checks").mkdir(parents=True, exist_ok=True)

    # 1. fresh worktree
    git("worktree", "add", "-b", branch, str(work), base, cwd=canon)
    report = [f"# Export report: {proj['slug']} to {up['repo']}", "",
              f"Run {dt.datetime.now().isoformat(timespec='seconds')}. Base: `{base_ref}` at `{base[:12]}` "
              f"({git('show', '-s', '--format=%cd', '--date=short', base, cwd=canon)}). Branch: `{branch}`. "
              f"Worktree: `{work.relative_to(ROOT)}`. Nothing was pushed.", ""]

    # 2 and 3. apply, then set identity and trailers
    env = dict(os.environ, GIT_COMMITTER_NAME=ident["name"], GIT_COMMITTER_EMAIL=ident["email"])
    for i, patch in enumerate(patches):
        rel = pathlib.Path(patch).relative_to(proj["dir"])
        override = subjects[i] if i < len(subjects) else None
        p = sh(["git", "am", "--3way", patch], work, env)
        if p.returncode != 0:
            sh(["git", "am", "--abort"], work, env)
            subject, body = patch_header(patch)
            subject = override or subject
            if not subject:
                die(f"{rel} is a plain diff: give its commit subject in project.toml (subjects = [...])")
            q = sh(["git", "apply", "--index", "--3way", patch], work, env)
            if q.returncode != 0:
                die(f"{rel} does not apply to {base_ref} at {base[:12]}:\n{(p.stderr + q.stderr).strip()[-600:]}\n"
                    "Rebasing a series is work for the project, not for the export.")
            sh(["git", "commit", "-q", "-m", subject + ("\n\n" + body if body else ""), "--author", who], work, env)
        date = git("show", "-s", "--format=%aI", "HEAD", cwd=work)
        message = clean_message(git("show", "-s", "--format=%B", "HEAD", cwd=work), rules, override)
        q = subprocess.run(["git", "commit", "-q", "--amend", "--author", who, "--date", date, "-F", "-"],
                           cwd=work, env=env, input=message, capture_output=True, text=True)
        if q.returncode != 0:
            die(f"could not rewrite the commit for {rel}: {q.stderr.strip()}")
        print(f"export: applied {rel}")
    log = git("log", "--reverse", "--format=- `%h` %s", f"{base}..HEAD", cwd=work)
    report += ["## Commits", "", f"Author and committer: {who}.", "", log, ""]

    # 4. the repository's own checks
    results = []
    if not a.skip_checks:
        extra = dict(rules.get("env", {}))
        path = os.environ.get("PATH", "")
        if (proj["dir"] / ".venv/bin").exists():
            path = f"{proj['dir'] / '.venv/bin'}:{path}"
        script = (proj["card"].get("export") or {}).get("env_script")
        source = f"source '{proj['dir'] / script}' && " if script else ""
        for n, chk in enumerate(rules.get("checks", []), 1):
            logf = out / "checks" / f"{n:02d}.log"
            cmdline = f"{source}cd '{work / chk.get('cwd', '.')}' && {chk['cmd']}"
            p = subprocess.run(["bash", "-c", cmdline], env=clean_env({**extra, "PATH": path}), capture_output=True, text=True)
            logf.write_text(f"$ {chk['cmd']}   (in {chk.get('cwd', '.')})\n\n{p.stdout}\n{p.stderr}\nexit code {p.returncode}\n", encoding="utf-8")
            results.append((chk["name"], chk["cmd"], p.returncode, logf))
            print(f"export: check {'ok  ' if p.returncode == 0 else 'FAIL'} {chk['name']}")
    report += ["## The repository's own checks", ""]
    report += [f"- {'pass' if rc == 0 else '**FAIL**'}: {nm}. `{cmd}`. Log: `{lf.relative_to(proj['dir'])}`" for nm, cmd, rc, lf in results] or ["- skipped"]
    report.append("")

    # 5. leak check
    lfails, lwarns, stats = leakcheck.run(work, base, rules, ws)
    report += ["## Leak check", "", f"{stats['commits']} commit(s), {stats['files']} file(s), +{stats['added']} -{stats['deleted']} lines."]
    report += [f"- **FAIL**: {f}" for f in lfails] + [f"- warning: {w}" for w in lwarns] or []
    report += ["- pass: no private strings, scaffolding files, placeholder identities or refused trailers"] if not lfails else []
    report.append("")

    # 6. PR text
    pfails, pwarns = check_pr(out / "PR.md", rules, ws)
    report += ["## PR description (`PR.md`)", ""] + [f"- **FAIL**: {f}" for f in pfails] + [f"- warning: {w}" for w in pwarns]
    report += ["- pass: every section the repository uses is present; no private strings"] if not pfails else []
    report.append("")

    # the series as it would be sent
    series = out / "series"
    series.mkdir(exist_ok=True)
    git("format-patch", "-q", "-o", str(series), f"{base}..HEAD", cwd=work)

    # 7. checklist for the user
    numstat = git("diff", "--numstat", f"{base}..HEAD", cwd=work).splitlines()
    files = "\n".join(f"  - [ ] `{row.split(chr(9))[2]}` (+{row.split(chr(9))[0]} -{row.split(chr(9))[1]})" for row in numstat)
    signoff = rules.get("signoff", "absent")
    sign_line = {"required": f"- [ ] Sign off every commit yourself (the repository requires it; a script must not): "
                             f"`git -C {work.relative_to(ROOT)} rebase --signoff {base[:12]}`",
                 "optional": "- [ ] Sign-off is optional here. Add it yourself if you want it; a script must not.",
                 "absent": "- [x] No sign-off: this repository neither requires nor uses one."}[signoff]
    checks_failed = [nm for nm, _, rc, _ in results if rc != 0]
    ok = not (lfails or pfails or checks_failed)
    checklist = f"""# Checklist before anything leaves this machine: {proj['slug']} to {up['repo']}

Generated by `make export`. The script pushed nothing and will not. Tick each box yourself.
Rules: `upstream/{name}.md`. Result of this run: {'ready for your review' if ok else 'NOT ready (see REPORT.md)'}.

## Read

- [ ] I read every changed line ({stats['files']} files, +{stats['added']} -{stats['deleted']}): `git -C {work.relative_to(ROOT)} diff {base[:12]}..HEAD`
{files}
- [ ] I can explain the change and its most surprising number without notes.

## Identity and trailers

- [ ] Author and committer are right: {who}
- [ ] AI disclosure is what I want for this repository. {stats['ai_trailers']} of {stats['commits']} commits carry an AI co-author trailer; the card's "Silent or ambiguous" section says what the repository leaves open.
{sign_line}

## Evidence

- [ ] The repository's checks passed here ({len(results) - len(checks_failed)} of {len(results)}); logs in `export/{name}/checks/`. Rerun any of them in the worktree if in doubt.
- [ ] The leak check passed: `make leakcheck DIR={work.relative_to(ROOT)} REPO={name} BASE={base[:12]}`
- [ ] `export/{name}/PR.md` is in my own words, has the repository's sections, and every number in it gives n and the setup.

## Ask first

- [ ] If the change is large or unrequested, I asked the maintainer in an issue or a message.
- [ ] I checked for an open pull request that already does this.

## Send (commands for you; the script runs none of them)

```bash
git -C {work.relative_to(ROOT)} remote add fork <the URL of your fork of {up['repo']}>
git -C {work.relative_to(ROOT)} push fork {branch}
# then open a pull request against {up['repo']} {rules.get('default_branch', 'main')} with PR.md as the description
```

## After sending

- [ ] Set `state = "sent"` in this project's `project.toml`, run `make portfolio`, and note the PR number in STATUS.md.
"""
    (out / "CHECKLIST.md").write_text(checklist, encoding="utf-8")
    report += ["## Result", "", "Ready for the user's review." if ok else "Not ready: fix the failures above and run the export again with `--fresh`."]
    (out / "REPORT.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    print(f"export: {'READY for your review' if ok else 'NOT READY'}; see {(out / 'REPORT.md').relative_to(ROOT)} "
          f"and {(out / 'CHECKLIST.md').relative_to(ROOT)}. Nothing was pushed.")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
