#!/usr/bin/env python3
"""make check: deterministic lint of the control plane. Exit 1 on any failure; warnings do not fail.

  check.py            run every check
  check.py --strict   warnings fail too

What it checks: the instruction files (paths and make targets exist, the Claude adapter imports
AGENTS.md, no stray instruction files outside vendor/), every project's card and files, the limit on
active projects, generated files, the budget ledger, the upstream cards, and whether an agent shell
could reach outward credentials. Warnings: placeholder patch authors, unpinned GPU names, drifted
copies of measure.py.
"""
from __future__ import annotations

import glob
import os
import pathlib
import re
import subprocess
import sys

from lib import ROOT, STAGES, STATES, UPSTREAM_STATES, card_name, git, money, projects, read_csv, workspace

fails: list[str] = []
warns: list[str] = []
SKIP_DIRS = {"vendor", ".venv", "toolchain", "models", ".git", "node_modules", "__pycache__"}
INSTRUCTION_NAMES = {"AGENTS.md", "CLAUDE.md", "CLAUDE.local.md", "GEMINI.md"}


def fail(msg: str) -> None:
    fails.append(msg)


def warn(msg: str) -> None:
    warns.append(msg)


def make_targets() -> set[str]:
    mk = ROOT / "Makefile"
    if not mk.exists():
        fail("Makefile is missing")
        return set()
    return set(re.findall(r"^([A-Za-z][\w-]*):", mk.read_text(encoding="utf-8"), re.M))


def check_instructions() -> None:
    agents, claude = ROOT / "AGENTS.md", ROOT / "CLAUDE.md"
    if not agents.exists():
        return fail("AGENTS.md is missing")
    lines = agents.read_text(encoding="utf-8").splitlines()
    if len(lines) > 80:
        fail(f"AGENTS.md has {len(lines)} lines; the budget is 80 (D8: move rules into checks)")
    first = next((ln.strip() for ln in claude.read_text(encoding="utf-8").splitlines() if ln.strip()), "") if claude.exists() else ""
    if first != "@AGENTS.md":
        fail("CLAUDE.md must exist and start with the line @AGENTS.md")
    targets = make_targets()
    for path in (agents, claude):
        if not path.exists():
            continue
        text = path.read_text(encoding="utf-8")
        for token in re.findall(r"`([^`\n]+)`", text):
            if token.startswith("make "):
                name = token.split()[1]
                if name not in targets:
                    fail(f"{path.name} names `make {name}`, which the Makefile does not have")
                continue
            if not re.fullmatch(r"[\w./@-]+", token) or token.startswith(("-", "@")) or re.fullmatch(r"\.\w+", token):
                continue
            if "/" not in token and not re.search(r"\.(md|toml|json|py|sh|csv)$", token):
                continue
            hits = [ROOT / token] + [p["dir"] / token for p in projects()]
            if "/" not in token:
                hits += list(ROOT.glob(f"*/{token}"))
            if not any(h.exists() for h in hits):
                fail(f"{path.name} names `{token}`, which does not exist")
    for name in (".claude/settings.json", ".gemini/settings.json"):
        if not (ROOT / name).exists():
            fail(f"{name} is missing (adapter)")
    for dirpath, dirnames, filenames in os.walk(ROOT):
        dirnames[:] = [d for d in dirnames if d not in SKIP_DIRS]
        for f in filenames:
            p = pathlib.Path(dirpath) / f
            if f in INSTRUCTION_NAMES and p.parent != ROOT:
                fail(f"stray instruction file outside vendor/: {p.relative_to(ROOT)} (D8: one source)")


def check_projects() -> None:
    ws = workspace()
    active = []
    for p in projects():
        slug, c = p["slug"], p["card"]
        if c is None:
            fail(f"{slug}: no project.toml")
            continue
        for key in ("id", "title", "stage", "state", "stakeholder", "claim", "kill", "next_action", "needs_gpu"):
            if key not in c or c[key] == "":
                fail(f"{slug}/project.toml: missing {key}")
        if c.get("id") != slug[:2]:
            fail(f"{slug}/project.toml: id {c.get('id')!r} does not match the folder")
        if c.get("stage") not in STAGES:
            fail(f"{slug}/project.toml: stage {c.get('stage')!r} is not one of {', '.join(STAGES)}")
        if c.get("state") not in STATES:
            fail(f"{slug}/project.toml: state {c.get('state')!r} is not one of {', '.join(STATES)}")
        if len(c.get("next_action", "")) > 110:
            fail(f"{slug}/project.toml: next_action is longer than 110 characters")
        if c.get("state") == "active":
            active.append(p)
        for name in ("STATUS.md", "WRITEUP.md"):
            if not (p["dir"] / name).exists():
                fail(f"{slug}: {name} is missing")
        for u in c.get("upstream", []):
            name = card_name(u.get("repo", ""))
            if u.get("state") not in UPSTREAM_STATES:
                fail(f"{slug}: upstream state {u.get('state')!r} is not one of {', '.join(UPSTREAM_STATES)}")
            for ext in ("md", "toml"):
                if not (ROOT / f"upstream/{name}.{ext}").exists():
                    fail(f"{slug}: upstream/{name}.{ext} is missing for {u.get('repo')}")
            if u.get("state") != "none" and not glob.glob(str(p["dir"] / u.get("series", "nothing-here"))):
                fail(f"{slug}: series {u.get('series')!r} matches no file")
        mine = p["dir"] / "harness/measure.py"
        if mine.exists() and mine.read_bytes() != (ROOT / "pipeline/template/project/harness/measure.py").read_bytes():
            warn(f"{slug}: harness/measure.py differs from the template's")
    limit = ws["projects"]["max_active"]
    if len(active) > limit:
        fail(f"{len(active)} active projects ({', '.join(p['slug'] for p in active)}); the limit is {limit} (D2)")
    if sum(bool(p["card"].get("needs_gpu")) for p in active) > 1:
        warn("more than one active project needs a GPU (D2: one of them on a GPU)")


def check_generated() -> None:
    for cmd, label in (([sys.executable, "scripts/portfolio.py", "--check"], "PORTFOLIO.md"),
                       ([sys.executable, "pipeline/build_research.py", "--check"], "pipeline/RESEARCH.md")):
        p = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
        if p.returncode != 0:
            fail(f"{label}: {(p.stdout + p.stderr).strip().splitlines()[-1]}")
    p = subprocess.run([sys.executable, "pipeline/template/project/harness/measure.py", "selftest"],
                       cwd=ROOT, capture_output=True, text=True)
    if p.returncode != 0:
        fail("pipeline/template/project/harness/measure.py selftest fails")


def check_budget() -> None:
    b = workspace()["budget"]
    ledger = read_csv(ROOT / "budget/ledger.csv")
    if not (ROOT / "budget/plan.csv").exists():
        fail("budget/plan.csv is missing")
    if money(ledger, "actual_usd") > b["total_usd"]:
        fail(f"budget: spent ${money(ledger, 'actual_usd'):.2f}, over the total of ${b['total_usd']:.2f}")
    for p in projects():
        cap = ((p["card"] or {}).get("budget") or {}).get("cap_usd", b["project_cap_usd"])
        spent = money(ledger, "actual_usd", p["slug"])
        if spent > cap:
            fail(f"budget: {p['slug']} spent ${spent:.2f}, over its cap of ${cap:.2f}")


def check_upstream() -> None:
    for name in workspace()["upstream"]:
        for ext in ("md", "toml"):
            if not (ROOT / f"upstream/{name}.{ext}").exists():
                fail(f"upstream/{name}.{ext} is missing")


def check_credentials() -> None:
    """D11: an agent shell should reach no outward credential. A finding is a warning: the user decides."""
    home = pathlib.Path.home()
    found = [str(p).replace(str(home), "~") for p in (
        home / ".config/gh/hosts.yml", home / ".modal.toml", home / ".netrc", home / ".git-credentials",
        home / ".cache/huggingface/token", home / ".huggingface/token") if p.exists()]
    found += [f"~/.ssh/{p.name}" for p in (home / ".ssh").glob("id_*") if not p.name.endswith(".pub")]
    helper = subprocess.run(["git", "config", "--global", "--get-all", "credential.helper"], capture_output=True, text=True).stdout.strip()
    if helper:
        found.append(f"git credential.helper={helper}")
    found += [f"${k}" for k in os.environ if re.search(r"^(GH_|GITHUB_|MODAL_|HF_|HUGGING).*TOKEN", k)]
    if found:
        warn("outward credentials are reachable from this shell, so agent sessions can use them (D11): " + ", ".join(found))


def check_warnings() -> None:
    placeholders = workspace()["leak"]["placeholder_identities"]
    for p in projects():
        bad = set()
        for patch in glob.glob(str(p["dir"] / "patches/**/*.patch"), recursive=True):
            head = pathlib.Path(patch).read_text(encoding="utf-8", errors="replace")[:600]
            m = re.search(r"^From: (.*)$", head, re.M)
            if m and any(x in m.group(1) for x in placeholders):
                bad.add(m.group(1).strip())
        if bad:
            warn(f"{p['slug']}: stored patches have placeholder authors ({'; '.join(sorted(bad))}); the export sets the identity")
        app = p["dir"] / "modal_app.py"
        if app.exists():
            loose = sorted(set(re.findall(r'"((?:H100|H200|A100|B200)(?::\d+)?)"', app.read_text(encoding="utf-8"))))
            if loose:
                warn(f"{p['slug']}/modal_app.py names GPUs unpinned ({', '.join(loose)}); pin before the first paid run (D10)")
    tracked = git("ls-files", "-z").split("\0")
    if any(re.search(r"(^|/)vendor/", f) for f in tracked):
        fail("a file under vendor/ is tracked")
    big = [f for f in tracked if f and (ROOT / f).exists() and (ROOT / f).stat().st_size > 10 * 2 ** 20]
    if big:
        warn("tracked files over 10 MB: " + ", ".join(big))


def main() -> int:
    for fn in (check_instructions, check_projects, check_generated, check_budget, check_upstream,
               check_credentials, check_warnings):
        fn()
    for w in warns:
        print("warn  " + w)
    for f in fails:
        print("FAIL  " + f)
    strict = "--strict" in sys.argv
    bad = len(fails) + (len(warns) if strict else 0)
    print(f"check: {len(fails)} failure(s), {len(warns)} warning(s)" + (" (strict)" if strict else ""))
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
