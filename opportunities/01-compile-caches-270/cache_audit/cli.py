"""Command line: ``python -m cache_audit {run,static,kb-verify,match}``."""

from __future__ import annotations

import argparse
import os
import sys

from . import kb as kbmod
from . import report as reportmod
from .audit import run_audit


def _env_toml_vars(path: str) -> dict[str, str]:
    data = kbmod._toml.loads(open(path, "rb").read().decode())
    return {k: str(v) for k, v in data.get("env", {}).items()}


def cmd_run(a: argparse.Namespace) -> int:
    if not a.cmd:
        print("cache-audit run: give the command after --", file=sys.stderr)
        return 2
    kb = kbmod.load(a.kb)
    sets = dict(s.split("=", 1) for s in a.set)
    workdir = os.path.abspath(a.workdir)
    # Runs execute in a fresh work dir: make caller-relative paths absolute.
    a.cmd = [os.path.abspath(x) if ("/" in x and not os.path.isabs(x) and os.path.exists(x)) else x
             for x in a.cmd]
    res = run_audit(
        a.cmd, kb, dict(os.environ), workdir,
        env_toml_vars=_env_toml_vars(a.env_toml) if a.env_toml else None,
        warm=not a.no_warm, relocate=a.relocate, readonly=a.readonly, allow_unknown=a.allow_unknown,
        opts={"tracer": a.tracer, "audit_hook": not a.no_audit_hook, "set": sets, "keep": a.keep,
              "timeout": a.timeout, "watch": a.watch, "ignore": a.ignore, "xdg": a.xdg, "cwd": a.cwd})
    out = a.out or os.path.join(workdir, "report")
    paths = reportmod.write_all(res, out)
    print(reportmod.markdown(res) if a.verbose else reportmod.toml_snippet(res))
    print(f"report: {paths['md']}\njson:   {paths['json']}\ntoml:   {paths['toml']}")
    print(f"gate:   {reportmod.GATE_TEXT.get(res.exit_code)} (exit {res.exit_code})")
    return res.exit_code


def cmd_static(a: argparse.Namespace) -> int:
    from . import static
    kb = kbmod.load(a.kb)
    rep = static.scan(kb, a.site or None, a.scope or None)
    print(static.render(rep))
    return rep.exit_code


def cmd_kb_verify(a: argparse.Namespace) -> int:
    from . import citations
    cites = citations.load()
    probs = citations.check(cites)
    kb = kbmod.load(a.kb)
    known = {c.id for c in cites}
    for e in kb.entries:
        for ev in e.evidence:
            if ev not in known:
                probs.append(f"kb {e.id}: evidence id {ev!r} has no citation")
    print(f"{len(cites)} citations, {len(kb.entries)} KB entries, {len(probs)} problems")
    for p in probs:
        print("  " + p)
    return 1 if probs else 0


def cmd_match(a: argparse.Namespace) -> int:
    kb = kbmod.load(a.kb)
    for p in a.paths:
        m = kb.match(os.path.abspath(p), dict(os.environ))
        print(f"{p}: {m[0].id + ' -> ' + m[1] if m else 'UNKNOWN'}")
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="cache-audit", description=__doc__)
    ap.add_argument("--kb", default=None, help="knowledge base TOML (default kb/caches.toml)")
    sub = ap.add_subparsers(dest="sub", required=True)

    r = sub.add_parser("run", help="audit a command: cold, redirected, warm (+relocate, readonly)")
    r.add_argument("--workdir", default="cache-audit-runs")
    r.add_argument("--out", default=None)
    r.add_argument("--tracer", default="auto", choices=["auto", "strace", "inotify", "snapshot"])
    r.add_argument("--no-audit-hook", action="store_true")
    r.add_argument("--env-toml", default=None, help="gate mode: redirect with this TOML's [env] ({jit_cache} placeholder)")
    r.add_argument("--set", action="append", default=[], metavar="K=V", help="extra env for every run")
    r.add_argument("--keep", action="append", default=[], metavar="VAR", help="do not scrub this cache variable")
    r.add_argument("--no-warm", action="store_true")
    r.add_argument("--relocate", action="store_true", help="also test moving the cache to another path")
    r.add_argument("--readonly", action="store_true", help="also test a read-only cache")
    r.add_argument("--allow-unknown", action="store_true")
    r.add_argument("--timeout", type=float, default=None)
    r.add_argument("--watch", action="append", default=[], help="extra writable root to snapshot")
    r.add_argument("--ignore", action="append", default=[], help="path prefix to ignore")
    r.add_argument("--xdg", action="store_true", help="also give the sandbox its own XDG_* dirs")
    r.add_argument("--cwd", default=None)
    r.add_argument("-v", "--verbose", action="store_true")
    r.add_argument("cmd", nargs=argparse.REMAINDER)
    r.set_defaults(fn=cmd_run)

    s = sub.add_parser("static", help="GPU-free drift scan of installed packages")
    s.add_argument("--site", action="append", default=[], help="site-packages dir (default: this interpreter)")
    s.add_argument("--scope", action="append", default=[], help="distribution to scan for new variables")
    s.set_defaults(fn=cmd_static)

    k = sub.add_parser("kb-verify", help="check every citation against the vendored source")
    k.set_defaults(fn=cmd_kb_verify)

    m = sub.add_parser("match", help="which KB entry claims a path")
    m.add_argument("paths", nargs="+")
    m.set_defaults(fn=cmd_match)

    a = ap.parse_args(argv)
    if getattr(a, "cmd", None) and a.cmd[:1] == ["--"]:
        a.cmd = a.cmd[1:]
    return a.fn(a)


if __name__ == "__main__":
    sys.exit(main())
