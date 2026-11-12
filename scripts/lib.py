"""Shared helpers for the control-plane scripts. Standard library only (Python 3.11 or later, for tomllib)."""
from __future__ import annotations

import csv
import os
import pathlib
import subprocess
import sys
import tomllib

ROOT = pathlib.Path(__file__).resolve().parent.parent
STAGES = ("intake", "archaeology", "hypotheses", "harness", "sweeps", "verify", "export", "closed")
STATES = ("active", "parked", "sent", "merged", "killed")
UPSTREAM_STATES = ("none", "drafted", "exported", "sent", "merged", "rejected")
SECRET_WORDS = ("TOKEN", "SECRET", "PASSW", "API_KEY", "CREDENTIAL")


def die(msg: str) -> "NoReturn":
    print(msg, file=sys.stderr)
    sys.exit(1)


def load_toml(path: pathlib.Path) -> dict:
    with open(path, "rb") as fh:
        return tomllib.load(fh)


def workspace() -> dict:
    return load_toml(ROOT / "pipeline/workspace.toml")


def projects() -> list[dict]:
    """Every folder under the projects directory whose name starts with two digits."""
    base = ROOT / workspace()["projects"]["dir"]
    out = []
    for d in sorted(base.iterdir()):
        if d.is_dir() and d.name[:2].isdigit():
            card = d / "project.toml"
            out.append({"dir": d, "slug": d.name, "card": load_toml(card) if card.exists() else None})
    return out


def find_project(key: str) -> dict:
    hits = [p for p in projects() if p["slug"] == key or p["slug"].startswith(key + "-") or p["slug"][:2] == key]
    if len(hits) != 1:
        die(f"project {key!r}: expected one match, found {[p['slug'] for p in hits]}")
    if hits[0]["card"] is None:
        die(f"{hits[0]['slug']} has no project.toml")
    return hits[0]


def card_name(repo: str) -> str:
    """'eth-easl/servekit' -> 'servekit': the name of its card and of its canonical clone."""
    return repo.rsplit("/", 1)[-1]


def git(*args: str, cwd: pathlib.Path | str = ROOT, check: bool = True, env: dict | None = None) -> str:
    p = subprocess.run(["git", *args], cwd=cwd, capture_output=True, text=True, env=env)
    if check and p.returncode != 0:
        die(f"git {' '.join(args)} failed in {cwd}:\n{p.stderr.strip()}")
    return p.stdout.strip()


def clean_env(extra: dict | None = None) -> dict:
    """The current environment without anything that looks like a credential."""
    env = {k: v for k, v in os.environ.items() if not any(w in k.upper() for w in SECRET_WORDS)}
    env.update(extra or {})
    return env


def read_csv(path: pathlib.Path) -> list[dict]:
    if not path.exists():
        return []
    with open(path, newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def money(rows: list[dict], key: str, project: str | None = None, month: str | None = None) -> float:
    total = 0.0
    for r in rows:
        if project and r.get("project", "")[:2] != project[:2]:
            continue
        if month and not r.get("date", "").startswith(month):
            continue
        try:
            total += float(r.get(key) or 0)
        except ValueError:
            die(f"budget: {key}={r.get(key)!r} is not a number in row {r}")
    return total
