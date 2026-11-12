#!/usr/bin/env python3
"""Rebuild the generated parts of pipeline/RESEARCH.md and check DOCTRINE.md's citations.

Reads   pipeline/research/NN-*.md   (one source ledger per topic, written by research agents)
        pipeline/DOCTRINE.md        (decisions D1..Dn, each with "For:" and "Against:" lines)
Writes  the blocks between <!-- generated:NAME:start --> and <!-- generated:NAME:end -->
        in pipeline/RESEARCH.md (NAME = stats, decision-index, ledger).

Fails (exit 1) when the doctrine cites an ID no ledger has, or a decision has no "For:" line.
`--check` writes nothing and fails if RESEARCH.md is not what a rebuild would produce.

Standard library only. Usage: python3 pipeline/build_research.py [--check]
"""
from __future__ import annotations

import pathlib
import re
import sys

HERE = pathlib.Path(__file__).resolve().parent
NOTES = HERE / "research"
DOCTRINE = HERE / "DOCTRINE.md"
RESEARCH = HERE / "RESEARCH.md"

TOPICS = {
    "01": "Harness architecture (H)",
    "02": "Empirical studies (E)",
    "03": "Autonomous research loops (L)",
    "04": "Benchmarking methodology (B)",
    "05": "Evaluator and academic norms (V)",
    "06": "Agent security (S)",
    "07": "Maintainer policies and PR acceptance (M)",
    "08": "Practitioners: programmers (P)",
    "09": "Practitioners: researchers (R)",
    "10": "Context files and portability (C)",
    "11": "Experiment operations and budget (X)",
}
ID_RE = re.compile(r"\b([BCEHLMPRSVX]\d{2})\b")
ROW_RE = re.compile(r"^\|\s*([BCEHLMPRSVX]\d{2})\s*\|")
PIPE = "\x00"


def cells(line: str) -> list[str]:
    parts = line.replace("\\|", PIPE).strip().strip("|").split("|")
    return [p.strip().replace(PIPE, "\\|") for p in parts]


def read_ledgers() -> tuple[dict[str, dict], list[tuple[str, list[str]]]]:
    rows: dict[str, dict] = {}
    order: list[tuple[str, list[str]]] = []
    for path in sorted(NOTES.glob("[0-9][0-9]-*.md")):
        topic = TOPICS.get(path.name[:2], path.stem)
        ids: list[str] = []
        in_ledger = False
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.startswith("## "):
                in_ledger = line.lower().startswith("## source ledger")
                continue
            if not in_ledger or not ROW_RE.match(line):
                continue
            c = cells(line)
            if len(c) != 8:
                sys.exit(f"{path.name}: ledger row {c[0]} has {len(c)} cells, expected 8")
            if c[0] in rows:
                sys.exit(f"duplicate ledger ID {c[0]} in {path.name} and {rows[c[0]]['file']}")
            rows[c[0]] = {"file": path.name, "topic": topic, "cells": c}
            ids.append(c[0])
        order.append((path.name, ids))
    return rows, order


def read_doctrine(rows: dict[str, dict]) -> tuple[list[dict], list[str]]:
    decisions: list[dict] = []
    problems: list[str] = []
    current = None
    for line in DOCTRINE.read_text(encoding="utf-8").splitlines():
        m = re.match(r"^##\s+(D\d+)\s+(.*)$", line)
        if m:
            current = {"id": m.group(1), "title": m.group(2).strip(), "for": [], "against": [], "other": []}
            decisions.append(current)
            continue
        if line.startswith("## "):
            current = None
            continue
        if current is None:
            continue
        plain = line.replace("*", "").strip()
        ids = ID_RE.findall(plain)
        if plain.startswith("For:"):
            current["for"] += ids
        elif plain.startswith("Against:"):
            current["against"] += ids
        else:
            current["other"] += ids
    for d in decisions:
        for bucket in ("for", "against", "other"):
            seen: list[str] = []
            for i in d[bucket]:
                if i not in rows:
                    problems.append(f"{d['id']} cites {i}, which is in no ledger")
                elif i not in seen:
                    seen.append(i)
            d[bucket] = seen
        if not d["for"]:
            problems.append(f"{d['id']} has no 'For:' line with a ledger ID")
    if not decisions:
        problems.append("DOCTRINE.md has no '## D<n> ...' headings")
    return decisions, problems


def grade(row: dict) -> str:
    g = row["cells"][4].replace("*", "").strip()
    m = re.match(r"^([A-E])(?![a-z])", g)
    return m.group(1) if m else "other"


def tag(i: str, rows: dict[str, dict]) -> str:
    g = grade(rows[i])
    return f"{i} ({g})" if g != "other" else i


def build(rows, order, decisions) -> dict[str, str]:
    cited: dict[str, list[str]] = {}
    for d in decisions:
        for i in d["for"] + d["other"]:
            cited.setdefault(i, []).append(d["id"] + "+" if i in d["for"] else d["id"])
        for i in d["against"]:
            cited.setdefault(i, []).append(d["id"] + "−")

    counts: dict[str, int] = {}
    for r in rows.values():
        counts[grade(r)] = counts.get(grade(r), 0) + 1
    stats = (
        f"{len(rows)} ledger rows in {len(order)} topic files. By grade: "
        + ", ".join(f"{k} {counts.get(k, 0)}" for k in ("A", "B", "C", "D", "E"))
        + f", other (documentation, policy text, terms, local checks) {counts.get('other', 0)}. "
        + f"The doctrine cites {len(cited)} of them across {len(decisions)} decisions."
    )

    idx = ["| Decision | Evidence for | Evidence against or qualifying |", "|---|---|---|"]
    for d in decisions:
        extra = [i for i in d["other"] if i not in d["for"] and i not in d["against"]]
        f = ", ".join(tag(i, rows) for i in d["for"] + extra) or "none"
        a = ", ".join(tag(i, rows) for i in d["against"]) or "none recorded"
        idx.append(f"| **{d['id']}** {d['title']} | {f} | {a} |")

    led: list[str] = []
    for fname, ids in order:
        topic = TOPICS.get(fname[:2], fname)
        led.append(f"### {topic}")
        led.append("")
        led.append(f"Notes, disagreements and unverified items for this topic: [research/{fname}](research/{fname}).")
        led.append("")
        led.append("| Doctrine | ID | Source | Date | Type | Grade | Finding | Bears on (the note's own wording) | Depends on model generation? |")
        led.append("|---|---|---|---|---|---|---|---|---|")
        for i in ids:
            c = rows[i]["cells"]
            led.append("| " + " | ".join([" ".join(cited.get(i, [])) or "", *c]) + " |")
        led.append("")
    return {"stats": stats, "decision-index": "\n".join(idx), "ledger": "\n".join(led).rstrip()}


def splice(text: str, blocks: dict[str, str]) -> str:
    for name, body in blocks.items():
        start, end = f"<!-- generated:{name}:start -->", f"<!-- generated:{name}:end -->"
        if start not in text or end not in text:
            sys.exit(f"RESEARCH.md lacks the markers for '{name}'")
        head, rest = text.split(start, 1)
        _, tail = rest.split(end, 1)
        text = f"{head}{start}\n{body}\n{end}{tail}"
    return text


def main() -> int:
    rows, order = read_ledgers()
    decisions, problems = read_doctrine(rows)
    if problems:
        print("\n".join("research: " + p for p in problems))
        return 1
    old = RESEARCH.read_text(encoding="utf-8")
    new = splice(old, build(rows, order, decisions))
    if "--check" in sys.argv:
        if new != old:
            print("research: RESEARCH.md is stale; run python3 pipeline/build_research.py")
            return 1
        print(f"research: ok ({len(rows)} rows, {len(decisions)} decisions, every cited ID exists)")
        return 0
    RESEARCH.write_text(new, encoding="utf-8")
    print(f"research: rebuilt ({len(rows)} rows, {len(decisions)} decisions)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
