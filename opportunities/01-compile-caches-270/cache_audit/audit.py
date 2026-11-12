"""Orchestrate the audit: cold run, redirected run, warm run, and optional
relocation and read-only runs; then decide the CI-gate verdict.

1. **cold**: KB cache variables scrubbed, fresh HOME/TMPDIR. Every persisted
   file is clustered into a root and mapped to a KB entry (or UNKNOWN).
2. **redirect**: each mapped variable set to ``<cache>/<component>`` in a new
   sandbox (or the variables from a deployed env TOML, gate mode). A KB entry
   is *verified* when its files now land under ``<cache>`` and none remain at
   the default; anything persisted outside ``<cache>`` is *unredirected*.
3. **warm**: same ``<cache>``, fresh HOME/TMPDIR (a new job). Binaries written
   again are rebuilds; the rest are builds avoided.
4. **relocate** (optional): the warm cache copied to another absolute path.
   Rebuilds here mean absolute paths are baked in (ninja), so a per-job
   overlay must live at a fixed path.
5. **readonly** (optional): the cache made read-only. Failed or attempted
   writes and the exit code show whether a library can use a shared
   read-only cache without a writable overlay.
"""

from __future__ import annotations

import os
import shutil
from collections import Counter
from dataclasses import dataclass, field
from typing import Mapping

from . import cluster as clustering
from .kb import KB, CacheEntry, scrub_env
from .runner import Sandbox, record, set_tree_writable, tail
from .tracers import Recording

JIT = "{jit_cache}"

EXIT_OK = 0
EXIT_RUN_FAILED = 2
EXIT_UNREDIRECTED = 10
EXIT_UNKNOWN = 11
EXIT_DRIFT = 12


@dataclass
class RunResult:
    name: str
    rec: Recording
    roots: list[clustering.Root]
    env_set: dict[str, str]
    cache_dir: str | None = None
    log_tail: str = ""
    sb_root: str | None = None

    @property
    def ok(self) -> bool:
        return self.rec.returncode == 0


@dataclass
class AuditResult:
    argv: list[str]
    runs: dict[str, RunResult] = field(default_factory=dict)
    proposal: dict[str, str] = field(default_factory=dict)        # VAR -> "{jit_cache}/x"
    proposal_entries: dict[str, list[str]] = field(default_factory=dict)  # VAR -> entry ids
    verification: dict[str, str] = field(default_factory=dict)    # entry id -> verdict
    outside: list[clustering.Root] = field(default_factory=list)  # persisted outside cache in redirect run
    temporaries: list[clustering.Root] = field(default_factory=list)  # leaked temp files (random names), not gated
    builds: dict[str, dict[str, int]] = field(default_factory=dict)
    relocation: dict[str, int] = field(default_factory=dict)
    readonly: dict[str, object] = field(default_factory=dict)
    exit_code: int = EXIT_OK
    reasons: list[str] = field(default_factory=list)


def _under(p: str, d: str | None) -> bool:
    if not d:
        return False
    for a in (d, os.path.realpath(d)):
        if p == a or p.startswith(a.rstrip("/") + "/"):
            return True
    return False


def propose(roots: list[clustering.Root]) -> tuple[dict[str, str], dict[str, list[str]]]:
    """VAR -> '{jit_cache}/<component>' for every persistable KB root seen."""
    prop: dict[str, str] = {}
    owners: dict[str, list[str]] = {}
    for r in roots:
        e = r.entry
        if e is None or not e.persist or not e.primary_var:
            continue
        var = e.primary_var
        prop.setdefault(var, f"{JIT}/{e.redirect or e.id}")
        owners.setdefault(var, [])
        if e.id not in owners[var]:
            owners[var].append(e.id)
    return dict(sorted(prop.items())), owners


def _binaries(roots: list[clustering.Root], cache_dir: str | None, inside: bool = True) -> Counter:
    c: Counter = Counter()
    for r in roots:
        if _under(r.path, cache_dir) != inside:
            continue
        c[r.label] += r.contents.by_class.get("binary", 0)
    return c


def _files(roots: list[clustering.Root], cache_dir: str | None) -> Counter:
    c: Counter = Counter()
    for r in roots:
        if _under(r.path, cache_dir):
            c[r.label] += r.contents.files
    return c


def _do_run(name: str, argv: list[str], kb: KB, base_env: Mapping[str, str], sb: Sandbox,
            env_set: dict[str, str], cache_dir: str | None, opts: dict) -> RunResult:
    env = sb.env(scrub_env(base_env, kb, keep=opts.get("keep", ())))
    env.update(env_set)
    cwd = opts.get("cwd") or os.path.join(sb.root, "work")
    os.makedirs(cwd, exist_ok=True)
    rec = record(argv, env, sb, cwd=cwd, tracer=opts.get("tracer", "auto"),
                 audit_hook=opts.get("audit_hook", True), watch=opts.get("watch"),
                 timeout=opts.get("timeout"), log_name=name)
    persisted = _drop_shared_noise(rec, sb, strict=opts.get("strict_shared", False))
    roots = clustering.cluster(persisted, kb, env, cwd, rec.events, needles=sb.needles(),
                               ignore=opts.get("ignore", ()))
    return RunResult(name, rec, roots, env_set, cache_dir, tail(sb, name), sb.root)


_SHARED = ("/tmp/", "/private/tmp/", "/var/tmp/", "/private/var/tmp/", "/dev/shm/")


def _drop_shared_noise(rec: Recording, sb: Sandbox, strict: bool) -> set[str]:
    """Snapshot changes under machine-wide dirs (/tmp, /dev/shm) may come from other
    processes. Keep one only if strace or the audit hook saw our process tree write it;
    with strace attribution is complete, without it hardcoded C-level /tmp writes are
    reported as unattributed rather than gated."""
    persisted = rec.persisted()
    if strict:
        return persisted
    ours = {e.path for e in rec.events if e.source in ("strace", "audithook")}
    ours |= {os.path.realpath(p) for p in ours}
    mine = (sb.root.rstrip("/") + "/", os.path.realpath(sb.root).rstrip("/") + "/")
    keep, dropped = set(), 0
    for p in persisted:
        if p.startswith(mine) or not p.startswith(_SHARED) or p in ours or os.path.realpath(p) in ours:
            keep.add(p)
        else:
            dropped += 1
    if dropped:
        how = "complete (strace)" if "strace" in rec.tracers else "Python-only (no strace)"
        rec.notes.append(f"ignored {dropped} unattributed change(s) under /tmp,/var/tmp,/dev/shm; "
                         f"attribution is {how}")
    return keep


def run_audit(argv: list[str], kb: KB, base_env: Mapping[str, str], workdir: str, *,
              env_toml_vars: Mapping[str, str] | None = None, warm: bool = True,
              relocate: bool = False, readonly: bool = False, allow_unknown: bool = False,
              opts: dict | None = None) -> AuditResult:
    opts = dict(opts or {})
    res = AuditResult(argv=list(argv))
    os.makedirs(workdir, exist_ok=True)

    # 1. cold, defaults
    sa = Sandbox.create(workdir, "cold", xdg=opts.get("xdg", False))
    cold = _do_run("cold", argv, kb, base_env, sa, dict(opts.get("set", {})), None, opts)
    res.runs["cold"] = cold
    if not cold.ok:
        res.exit_code = EXIT_RUN_FAILED
        res.reasons.append(f"cold run exited {cold.rec.returncode}")
        return res

    # 2. redirect: proposal from the cold run, or the deployed env TOML (gate mode)
    if env_toml_vars is not None:
        res.proposal = {k: v for k, v in env_toml_vars.items()
                        if JIT in v or k in kb.all_vars()}
        _, res.proposal_entries = propose(cold.roots)
    else:
        res.proposal, res.proposal_entries = propose(cold.roots)
    sb = Sandbox.create(workdir, "redirect", xdg=opts.get("xdg", False))
    cache_dir = os.path.join(sb.root, "cache")
    os.makedirs(cache_dir, exist_ok=True)
    env_set = dict(opts.get("set", {}))
    env_set.update({k: v.replace(JIT, cache_dir) for k, v in res.proposal.items()})
    red = _do_run("redirect", argv, kb, base_env, sb, env_set, cache_dir, opts)
    res.runs["redirect"] = red
    if not red.ok:
        res.exit_code = EXIT_RUN_FAILED
        res.reasons.append(f"redirect run exited {red.rec.returncode}")
        return res

    # verification per KB entry seen cold
    seen_cold = {r.entry.id for r in cold.roots if r.entry}
    for eid in sorted(seen_cold):
        e = kb.by_id(eid)
        if not e.persist:
            continue
        ins = [r for r in red.roots if r.entry and r.entry.id == eid and _under(r.path, cache_dir)]
        outs = [r for r in red.roots if r.entry and r.entry.id == eid and not _under(r.path, cache_dir)]
        var_set = e.primary_var in env_set
        if not var_set:
            res.verification[eid] = "not redirected (variable not set)"
        elif ins and not outs:
            res.verification[eid] = "verified"
        elif ins and outs:
            res.verification[eid] = "partial"
        elif outs:
            res.verification[eid] = "NOT MOVED"
        else:
            # The entry's files may now be claimed by the redirected entry of the same var.
            moved = any(r.entry and e.primary_var in r.entry.vars and _under(r.path, cache_dir)
                        for r in red.roots)
            res.verification[eid] = "verified (under shared var)" if moved else "not seen"

    res.outside = [r for r in red.roots if not _under(r.path, cache_dir)
                   and (r.entry is None or r.entry.persist)]

    # 3. warm: fresh HOME/TMPDIR, same cache
    if warm:
        sw = sb.fresh_home("warm")
        wr = _do_run("warm", argv, kb, base_env, sw, env_set, cache_dir, opts)
        res.runs["warm"] = wr
        b_cold, b_warm = _binaries(red.roots, cache_dir), _binaries(wr.roots, cache_dir)
        f_cold, f_warm = _files(red.roots, cache_dir), _files(wr.roots, cache_dir)
        for label in sorted(set(b_cold) | set(f_cold)):
            res.builds[label] = {"binaries_cold": b_cold[label], "binaries_warm": b_warm[label],
                                 "files_cold": f_cold[label], "files_warm": f_warm[label]}

    # 4. relocation
    if relocate:
        moved = os.path.join(sb.root, "cache-moved")
        shutil.copytree(cache_dir, moved, symlinks=True, dirs_exist_ok=True)
        env_m = {k: v.replace(cache_dir, moved) for k, v in env_set.items()}
        sm = sb.fresh_home("relocate")
        # Hide the original while the copy is used: a cache whose index stores absolute
        # paths (Triton's group files) would otherwise silently read the original.
        hidden = cache_dir + ".hidden"
        os.rename(cache_dir, hidden)
        try:
            rr = _do_run("relocate", argv, kb, base_env, sm, env_m, moved, opts)
        finally:
            os.rename(hidden, cache_dir)
        res.runs["relocate"] = rr
        res.relocation = dict(_files(rr.roots, moved))

    # 5. read-only
    if readonly:
        ro = os.path.join(sb.root, "cache-ro")
        shutil.copytree(cache_dir, ro, symlinks=True, dirs_exist_ok=True)
        set_tree_writable(ro, False)
        env_r = {k: v.replace(cache_dir, ro) for k, v in env_set.items()}
        sr = sb.fresh_home("readonly")
        try:
            rro = _do_run("readonly", argv, kb, base_env, sr, env_r, ro, opts)
        finally:
            set_tree_writable(ro, True)
        res.runs["readonly"] = rro
        attempts = Counter()
        for e in rro.rec.events:
            if e.source in ("audithook", "strace") and _under(e.path, ro) and e.op not in ("rename_from",):
                m = kb.match(e.path, rro.env_set | {"HOME": sr.home})
                attempts[m[0].id if m else "UNKNOWN"] += 1
        res.readonly = {"returncode": rro.rec.returncode,
                        "write_attempts_into_ro_cache": dict(attempts),
                        "failed_writes": sum(1 for e in rro.rec.events if e.op == "write_failed")}

    # verdict. Unknown roots made only of randomly named temp files (tmpab12cd34.so) are leaked
    # temporaries, not caches: nothing reuses them. Reported, not gated.
    res.temporaries = [r for r in res.outside if r.entry is None and set(r.contents.by_class) == {"temp"}]
    res.outside = [r for r in res.outside if r not in res.temporaries]
    drift = [k for k, v in res.verification.items() if v in ("NOT MOVED", "partial")]
    unknown = [r for r in res.outside if r.entry is None]
    known = [r for r in res.outside if r.entry is not None]
    if drift:
        res.exit_code = EXIT_DRIFT
        res.reasons.append("variable did not relocate: " + ", ".join(drift))
    elif unknown and not allow_unknown:
        res.exit_code = EXIT_UNKNOWN
        res.reasons.append("unknown cache roots outside the cache dir: " + ", ".join(r.path for r in unknown))
    elif known:
        res.exit_code = EXIT_UNREDIRECTED
        res.reasons.append("known caches not redirected: " + ", ".join(f"{r.label} ({r.path})" for r in known))
    for name in ("warm", "relocate", "readonly"):
        if name in res.runs and not res.runs[name].ok and name != "readonly":
            res.exit_code = EXIT_RUN_FAILED
            res.reasons.append(f"{name} run exited {res.runs[name].rec.returncode}")
    return res


def entries_seen(res: AuditResult) -> list[CacheEntry]:
    out: dict[str, CacheEntry] = {}
    for run in res.runs.values():
        for r in run.roots:
            if r.entry:
                out[r.entry.id] = r.entry
    return list(out.values())
