"""Knowledge base of known cache roots: which variable relocates which directory.

The KB (``kb/caches.toml``) lists one ``[[cache]]`` table per cache. Each has
path *patterns* with ``{PLACEHOLDER}`` variables and ``*`` wildcards per path
segment, e.g. ``"{TRITON_HOME}/.triton/cache"`` or
``"{VLLM_CACHE_ROOT}/torch_compile_cache/*/triton_cache"``. A file belongs to
the entry whose resolved pattern matches the most leading path segments.

Placeholders resolve from the run's environment, falling back to the library
default (``XDG_CACHE_HOME`` -> ``$HOME/.cache`` and so on). A pattern that is
exactly one of the entry's own ``vars`` (``"{TRITON_CACHE_DIR}"``) only applies
when that variable is set: its *unset* behaviour is spelled out by the other
patterns.
"""

from __future__ import annotations

import fnmatch
import getpass
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Mapping

try:  # Python >= 3.11
    import tomllib as _toml
except ModuleNotFoundError:  # pragma: no cover - 3.10 fallback
    import tomli as _toml  # type: ignore[no-redef]

DEFAULT_KB = Path(__file__).resolve().parent.parent / "kb" / "caches.toml"

_PLACEHOLDER = re.compile(r"\{([A-Z_][A-Z0-9_]*)\}")


def _user(env: Mapping[str, str]) -> str:
    for k in ("USER", "LOGNAME", "LNAME", "USERNAME"):
        if env.get(k):
            return env[k]
    try:
        return getpass.getuser()
    except Exception:  # pragma: no cover
        return "unknown"


def _sanitized_user(env: Mapping[str, str]) -> str:
    # torch/_inductor/runtime/cache_dir_utils.py: torchinductor_<sanitized user>
    return re.sub(r'[\\/:*?"<>|]', "_", _user(env))


# Defaults used when a placeholder is not set in the environment. Order matters:
# later entries may refer to earlier ones. Values are templates themselves.
_FALLBACKS: list[tuple[str, str]] = [
    ("HOME", "__HOME__"),
    ("TMPDIR", "/tmp"),
    ("XDG_CACHE_HOME", "{HOME}/.cache"),
    ("XDG_CONFIG_HOME", "{HOME}/.config"),
    ("XDG_DATA_HOME", "{HOME}/.local/share"),
    ("TRITON_HOME", "{HOME}"),
    ("FLASHINFER_WORKSPACE_BASE", "{HOME}"),
    ("SGLANG_CACHE_DIR", "{HOME}/.cache/sglang"),
    ("VLLM_CACHE_ROOT", "{XDG_CACHE_HOME}/vllm"),
    ("HF_HOME", "{XDG_CACHE_HOME}/huggingface"),
    ("CUDA_CACHE_PATH", "{HOME}/.nv/ComputeCache"),
]


def resolve_vars(env: Mapping[str, str]) -> dict[str, str]:
    """Concrete values for every placeholder the KB may use, under ``env``."""
    out: dict[str, str] = {}
    for name, tmpl in _FALLBACKS:
        if env.get(name):
            out[name] = env[name]
        elif tmpl == "__HOME__":
            out[name] = os.path.expanduser("~")
        else:
            out[name] = _PLACEHOLDER.sub(lambda m: out.get(m.group(1), m.group(0)), tmpl)
    out["USER"] = _user(env)
    out["SANITIZED_USER"] = _sanitized_user(env)
    for k, v in env.items():  # any other variable can be used verbatim
        out.setdefault(k, v)
    return out


def expand(template: str, values: Mapping[str, str]) -> str | None:
    """Expand ``{VAR}`` placeholders; ``None`` if one is unknown."""
    missing = False

    def sub(m: re.Match[str]) -> str:
        nonlocal missing
        v = values.get(m.group(1))
        if v is None:
            missing = True
            return m.group(0)
        return v

    s = _PLACEHOLDER.sub(sub, template)
    if missing:
        return None
    return os.path.normpath(os.path.expanduser(s))


@dataclass
class CacheEntry:
    id: str
    name: str
    vars: list[str]
    patterns: list[str]
    redirect: str = ""            # snippet value, relative to {root}
    var_is_base: bool = False     # the variable names a base dir; library appends `subpath`
    subpath: str = ""
    libraries: list[str] = field(default_factory=list)
    engines: list[str] = field(default_factory=list)
    kind: str = "compile"         # compile | autotune | download | runtime-state | bytecode
    persist: bool = True          # worth persisting across jobs
    key: list[str] = field(default_factory=list)
    write: str = ""
    lock: str = ""
    relocatable: str = "unknown"  # yes | no | partial | unknown
    readonly_ok: str = "unknown"
    notes: str = ""
    evidence: list[str] = field(default_factory=list)
    suffixes: list[str] = field(default_factory=list)  # match by file suffix anywhere (bytecode)

    @property
    def primary_var(self) -> str | None:
        return self.vars[0] if self.vars else None


@dataclass
class KB:
    entries: list[CacheEntry]
    path: Path | None = None

    def by_id(self, cid: str) -> CacheEntry:
        for e in self.entries:
            if e.id == cid:
                return e
        raise KeyError(cid)

    def all_vars(self) -> list[str]:
        seen: list[str] = []
        for e in self.entries:
            for v in e.vars:
                if v not in seen:
                    seen.append(v)
        return seen

    def resolved_patterns(self, env: Mapping[str, str]) -> list[tuple[CacheEntry, list[str]]]:
        """Each entry with its patterns resolved to absolute segment lists."""
        values = resolve_vars(env)
        with_default = {name for name, _ in _FALLBACKS}
        out: list[tuple[CacheEntry, list[str]]] = []
        for e in self.entries:
            for pat in e.patterns:
                m = _PLACEHOLDER.fullmatch(pat)
                own = m.group(1) if m else None
                if own in e.vars and not env.get(own) and own not in with_default:
                    continue  # bare "{VAR}" without a known default: only when VAR is set
                p = expand(pat, values)
                if p is None:
                    continue
                if e.var_is_base and own in e.vars and e.subpath:
                    p = os.path.normpath(os.path.join(p, e.subpath))
                out.append((e, _segments(p)))
        return out

    def match(self, path: str, env: Mapping[str, str], _cache: dict | None = None) -> tuple[CacheEntry, str] | None:
        """Best KB entry for ``path`` and the concrete root it matched."""
        for e in self.entries:
            if e.suffixes and path.endswith(tuple(e.suffixes)):
                return e, "**/*" + e.suffixes[0]
        pats = self.resolved_patterns(env) if _cache is None else _cache.setdefault(
            "pats", self.resolved_patterns(env))
        segs = _segments(path)
        best: tuple[int, CacheEntry, list[str]] | None = None
        for e, pseg in pats:
            if len(pseg) > len(segs):
                continue
            if all(fnmatch.fnmatchcase(s, p) for s, p in zip(segs, pseg)):
                score = len(pseg) * 2 - sum("*" in p for p in pseg)  # prefer literal segments
                if best is None or score > best[0]:
                    best = (score, e, segs[: len(pseg)])
        if best is None:
            return None
        return best[1], "/" + "/".join(best[2])


def _segments(p: str) -> list[str]:
    return [s for s in os.path.normpath(p).split("/") if s]


def load(path: str | os.PathLike[str] | None = None) -> KB:
    p = Path(path) if path else DEFAULT_KB
    data = _toml.loads(p.read_text())
    entries = []
    for raw in data.get("cache", []):
        entries.append(CacheEntry(
            id=raw["id"], name=raw.get("name", raw["id"]), vars=list(raw.get("vars", [])),
            patterns=list(raw.get("patterns", [])), redirect=raw.get("redirect", ""),
            var_is_base=bool(raw.get("var_is_base", False)), subpath=raw.get("subpath", ""),
            libraries=list(raw.get("libraries", [])), engines=list(raw.get("engines", [])),
            kind=raw.get("kind", "compile"), persist=bool(raw.get("persist", True)),
            key=list(raw.get("key", [])), write=raw.get("write", ""), lock=raw.get("lock", ""),
            relocatable=raw.get("relocatable", "unknown"), readonly_ok=raw.get("readonly_ok", "unknown"),
            notes=raw.get("notes", ""), evidence=list(raw.get("evidence", [])),
            suffixes=list(raw.get("suffixes", [])),
        ))
    ids = [e.id for e in entries]
    dupes = {i for i in ids if ids.count(i) > 1}
    if dupes:
        raise ValueError(f"duplicate KB ids: {sorted(dupes)}")
    return KB(entries=entries, path=p)


def scrub_env(env: Mapping[str, str], kb: KB, keep: Iterable[str] = ()) -> dict[str, str]:
    """Copy of ``env`` without any KB cache variable (so defaults get exercised)."""
    keep = set(keep)
    drop = set(kb.all_vars()) | {"XDG_CACHE_HOME", "XDG_CONFIG_HOME", "XDG_DATA_HOME", "XDG_STATE_HOME"}
    return {k: v for k, v in env.items() if k not in drop or k in keep}
