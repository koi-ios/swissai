"""Mechanical check of source citations: every KB claim must point at a real line.

A citation is ``{"id", "claim", "path", "line", "quote"}``; ``path`` is relative
to the project folder and ``quote`` must be a substring of that exact line of
the vendored source (the version the image ships). ``check()`` returns the
failures; ``tests/test_kb_citations.py`` asserts there are none, and that every
``evidence`` id in ``kb/caches.toml`` resolves to a citation.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


@dataclass
class Citation:
    id: str
    claim: str
    path: str
    line: int
    quote: str
    source: str = ""


def load(root: Path = ROOT) -> list[Citation]:
    out: list[Citation] = []
    for f in sorted((root / "research").glob("*.citations.json")):
        for c in json.loads(f.read_text()):
            out.append(Citation(c["id"], c.get("claim", ""), c["path"], int(c["line"]), c["quote"], f.name))
    for f in sorted((root / "kb").glob("*.citations.json")):
        for c in json.loads(f.read_text()):
            out.append(Citation(c["id"], c.get("claim", ""), c["path"], int(c["line"]), c["quote"], f.name))
    return out


def check(cites: list[Citation], root: Path = ROOT) -> list[str]:
    problems: list[str] = []
    cache: dict[str, list[str]] = {}
    ids: dict[str, Citation] = {}
    for c in cites:
        prev = ids.get(c.id)
        # The same claim cited identically by two research files is fine; a clash is not.
        if prev is not None and (prev.path, prev.line, prev.quote) != (c.path, c.line, c.quote):
            problems.append(f"{c.id}: id reused for a different line in {prev.source} and {c.source}")
        ids.setdefault(c.id, c)
        p = root / c.path
        if c.path not in cache:
            try:
                cache[c.path] = p.read_text(errors="replace").splitlines()
            except OSError:
                cache[c.path] = []
        lines = cache[c.path]
        if not lines:
            problems.append(f"{c.id}: missing file {c.path}")
            continue
        if not (1 <= c.line <= len(lines)):
            problems.append(f"{c.id}: line {c.line} out of range in {c.path}")
            continue
        if c.quote.strip() not in lines[c.line - 1]:
            near = [i + 1 for i, l in enumerate(lines) if c.quote.strip() in l]
            hint = f" (found at {near[:3]})" if near else " (quote not in file)"
            problems.append(f"{c.id}: quote not on {c.path}:{c.line}{hint}")
    return problems
