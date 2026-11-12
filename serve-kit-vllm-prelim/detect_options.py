#!/usr/bin/env python3
"""Which of vLLM's own weight-loading options does the installed version have?

    python detect_options.py              # print a table, write results/vllm_options.json
    python detect_options.py --has flag:--safetensors-load-strategy format:instanttensor pkg:instanttensor
                                          # exit 0 only if all are present (run_arms.sh uses this)

An option counts only if `vllm serve --help` lists it AND the package it needs
imports: the newest loaders are missing from older releases, and the optional
ones (Run:ai, fastsafetensors, InstantTensor) are pip extras.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Dict, List

FLAGS = ["--safetensors-load-strategy", "--safetensors-prefetch-num-threads", "--safetensors-prefetch-block-size",
         "--load-format", "--model-loader-extra-config"]
FORMATS = ["sharded_state", "runai_streamer", "runai_streamer_sharded", "instanttensor", "fastsafetensors",
           "ipc_cache", "tensorizer"]
PACKAGES = ["runai_model_streamer", "instanttensor", "fastsafetensors", "tensorizer"]


def help_text(vllm: str, cache: Path) -> str:
    if cache.exists() and cache.stat().st_size > 0:
        return cache.read_text(errors="replace")
    text = ""
    for flag in ("--help=all", "--help"):  # newer vLLM groups its help; =all prints every option
        try:
            p = subprocess.run([vllm, "serve", flag], capture_output=True, text=True, timeout=300,
                               env={**os.environ, "HF_HUB_OFFLINE": "1"})
        except (OSError, subprocess.TimeoutExpired) as e:
            raise SystemExit(f"cannot run `{vllm} serve {flag}`: {e}")
        text = p.stdout + p.stderr
        if p.returncode == 0 and "--load-format" in text:
            break
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_text(text)
    return text


def present(kind: str, name: str, text: str) -> bool:
    if kind == "flag":
        return bool(re.search(rf"(?<![\w-]){re.escape(name)}(?![\w-])", text))
    if kind == "format":
        return bool(re.search(rf"(?<!\w){re.escape(name)}(?!\w)", text))
    if kind == "pkg":
        return importlib.util.find_spec(name) is not None
    raise SystemExit(f"bad requirement kind {kind!r}: use flag:, format: or pkg:")


def detect(text: str) -> Dict[str, Dict[str, bool]]:
    return {
        "flag": {f: present("flag", f, text) for f in FLAGS},
        "format": {f: present("format", f, text) for f in FORMATS},
        "pkg": {p: present("pkg", p, text) for p in PACKAGES},
    }


def missing(needs: List[str], text: str) -> List[str]:
    out = []
    for need in needs:
        kind, _, name = need.partition(":")
        if not present(kind, name, text):
            out.append(need)
    return out


def _cache_path() -> Path:
    return Path(os.environ.get("RESULTS", "results")) / "vllm_serve_help.txt"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--has", nargs="*", metavar="KIND:NAME", help="exit 1 unless all of these are present")
    ap.add_argument("--vllm-bin", default=os.environ.get("VLLM_BIN", "vllm"))
    ap.add_argument("--refresh", action="store_true", help="re-read `vllm serve --help` instead of the cached copy")
    args = ap.parse_args(argv)
    cache = _cache_path()
    if args.refresh and cache.exists():
        cache.unlink()
    text = help_text(args.vllm_bin, cache)
    if args.has is not None:
        return 1 if missing(args.has, text) else 0
    found = detect(text)
    try:
        from importlib.metadata import version
        vllm_version = version("vllm")
    except Exception:  # noqa: BLE001 - absent, or a mock on a laptop
        vllm_version = None
    out = {"vllm_version": vllm_version, **found}
    (cache.parent / "vllm_options.json").write_text(json.dumps(out, indent=1))
    print(f"vLLM {vllm_version or '(version unknown)'}")
    for kind in ("flag", "format", "pkg"):
        for name, ok in found[kind].items():
            print(f"  {kind:6s} {name:36s} {'yes' if ok else 'no'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
