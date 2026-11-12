"""Static drift scan: no GPU, no engine launch, seconds per image.

Two checks over an installed environment (an image's site-packages):

1. **KB variables still exist.** For each KB entry whose library is installed,
   the entry's variable names must still appear in that library's source. A
   rename (SGLang 0.5.16 ``TVM_FFI_CACHE_DIR`` -> 0.5.20 ``SGLANG_JIT_CACHE_DIR``)
   shows up as a variable that vanished.
2. **New cache-like variables.** Every string literal in the installed source
   that looks like a cache-location variable (``*_CACHE_DIR``, ``*_CACHE_ROOT``,
   ``*_CACHE_PATH``, ``*_WORKSPACE_BASE``, ``*_TMP_DIR``...) and every
   ``~/.cache/...`` / ``~/.<x>`` default, minus those the KB knows. New ones
   mean a library added a cache the platform does not persist yet.

This is the cheap CI gate (runs on any CPU runner right after an image
build); ``cache-audit run`` on a GPU node is the ground truth.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from importlib import metadata
from typing import Iterable

from .kb import KB

_VAR = re.compile(rb"""["']([A-Z][A-Z0-9_]*(?:_CACHE_DIR|_CACHE_ROOT|_CACHE_PATH|_CACHE_HOME|_CACHE|_WORKSPACE_BASE|_WORKSPACE_DIR|_JIT_DIR|_TMP_DIR|_HOME|_EXTENSIONS_DIR))["']""")
_HOME_DEFAULT = re.compile(rb"""["'](~/\.[A-Za-z0-9_.\-/]+)["']""")
_SKIP_VARS = {"HF_HOME", "CUDA_HOME", "ROCM_HOME", "JAVA_HOME", "XDG_CACHE_HOME", "XDG_CONFIG_HOME",
              "PYTHONHOME", "CONDA_HOME", "HOME", "MUSA_HOME", "NEURON_HOME", "ASCEND_HOME",
              "TORCH_HOME", "LD_LIBRARY_PATH", "CUDA_PATH", "OPAL_PREFIX", "XPU_HOME", "TPU_HOME",
              "MOONCAKE_HOME", "CUTLASS_HOME", "NVSHMEM_HOME", "CUDNN_HOME", "MPI_HOME"}
_EXTS = (".py", ".pyi", ".sh")


@dataclass
class StaticReport:
    installed: dict[str, str] = field(default_factory=dict)           # dist -> version
    missing_vars: list[tuple[str, str, str]] = field(default_factory=list)  # (entry, var, library)
    present_vars: list[tuple[str, str, str]] = field(default_factory=list)
    new_vars: dict[str, list[str]] = field(default_factory=dict)       # var -> files
    new_home_defaults: dict[str, list[str]] = field(default_factory=dict)

    @property
    def exit_code(self) -> int:
        if self.missing_vars:
            return 12
        if self.new_vars:
            return 11
        return 0


def _files_of(dist: metadata.Distribution) -> list[str]:
    out = []
    for f in dist.files or []:
        if str(f).endswith(_EXTS):
            p = dist.locate_file(f)
            out.append(str(p))
    return out


def _dist_index(paths: Iterable[str] | None) -> dict[str, metadata.Distribution]:
    dists = metadata.distributions(path=list(paths)) if paths else metadata.distributions()
    idx: dict[str, metadata.Distribution] = {}
    for d in dists:
        name = (d.metadata.get("Name") or "").lower().replace("_", "-")
        if name and name not in idx:
            idx[name] = d
    return idx


def scan(kb: KB, site_paths: Iterable[str] | None = None, scope: Iterable[str] | None = None) -> StaticReport:
    """``scope``: distribution names to scan for new variables (default: KB libraries + engines)."""
    rep = StaticReport()
    idx = _dist_index(site_paths)
    libs = set()
    for e in kb.entries:
        libs.update(x.lower() for x in e.libraries)
    scope = {s.lower() for s in scope} if scope else (libs | {"sglang", "vllm"})
    texts: dict[str, list[tuple[str, bytes]]] = {}
    for name in sorted(set(scope) | libs):
        d = idx.get(name)
        if d is None:
            continue
        rep.installed[name] = d.version
        blobs = []
        for p in _files_of(d):
            try:
                with open(p, "rb") as f:
                    blobs.append((p, f.read()))
            except OSError:
                pass
        texts[name] = blobs

    known_vars = set(kb.all_vars())
    for e in kb.entries:
        for lib in (x.lower() for x in e.libraries):
            if lib not in texts:
                continue
            for var in e.vars:
                hit = any(var.encode() in b for _, b in texts[lib])
                (rep.present_vars if hit else rep.missing_vars).append((e.id, var, lib))
    # A variable is only "missing" if no installed KB library mentions it at all.
    present = {(eid, v) for eid, v, _ in rep.present_vars}
    rep.missing_vars = [m for m in rep.missing_vars if (m[0], m[1]) not in present]

    for name, blobs in texts.items():
        if name not in scope:
            continue
        for path, b in blobs:
            for m in _VAR.finditer(b):
                v = m.group(1).decode()
                if v in known_vars or v in _SKIP_VARS or v.startswith(("CMAKE_", "SETUPTOOLS_")):
                    continue
                rep.new_vars.setdefault(v, []).append(os.path.relpath(path, os.path.dirname(os.path.dirname(path))))
            for m in _HOME_DEFAULT.finditer(b):
                rep.new_home_defaults.setdefault(m.group(1).decode(), []).append(os.path.basename(path))
    return rep


def render(rep: StaticReport) -> str:
    out = ["# cache-audit static scan", "", "Installed: " + ", ".join(f"{k} {v}" for k, v in sorted(rep.installed.items())), ""]
    out.append(f"KB variables found in source: {len(rep.present_vars)}")
    if rep.missing_vars:
        out.append("**KB variables no longer in the installed source (drift):**")
        out += [f"- {v} ({eid}, expected in {lib})" for eid, v, lib in rep.missing_vars]
    if rep.new_vars:
        out.append("")
        out.append("**Cache-like variables the KB does not know:**")
        for v, files in sorted(rep.new_vars.items()):
            out.append(f"- `{v}`: {', '.join(sorted(set(files))[:3])}")
    if rep.new_home_defaults:
        out.append("")
        out.append("Home-directory defaults seen (review): " +
                   ", ".join(f"`{k}`" for k in sorted(rep.new_home_defaults)[:40]))
    out.append("")
    out.append(f"exit {rep.exit_code}")
    return "\n".join(out) + "\n"
