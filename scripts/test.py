#!/usr/bin/env python3
"""make test: rerun project test suites and compare with the counts recorded in opportunities/README.md.

  test.py            every suite in pipeline/tests.toml
  test.py 06 02      only these projects

Each suite has a command, a folder, and a pattern its output must contain. A suite that rewrites a
tracked file names it under `restore`, and the file is put back from git afterwards. Commands run
with credentials removed from the environment.
"""
from __future__ import annotations

import re
import subprocess
import sys
import time

from lib import ROOT, clean_env, git, load_toml


def main() -> int:
    suites = load_toml(ROOT / "pipeline/tests.toml")["suite"]
    only = sys.argv[1:]
    failed = 0
    ran = 0
    for s in suites:
        if only and s["project"] not in only:
            continue
        ran += 1
        t0 = time.time()
        try:
            p = subprocess.run(["bash", "-c", s["cmd"]], cwd=ROOT / s["cwd"], env=clean_env(), capture_output=True,
                               text=True, timeout=s.get("timeout", 900))
            out = p.stdout + "\n" + p.stderr
        except subprocess.TimeoutExpired:
            out = "TIMEOUT"
        for path in s.get("restore", []):
            git("checkout", "--", path)
        ok = re.search(s["expect"], out) is not None
        failed += not ok
        line = next((ln for ln in reversed(out.splitlines()) if re.search(s["expect"], ln)), "")
        if not ok:
            line = " | ".join(ln.strip() for ln in out.strip().splitlines()[-2:])
        print(f"{'ok  ' if ok else 'FAIL'} {s['project']} {s['name']:<18} {time.time() - t0:5.0f}s  {line.strip()[:110]}")
        if not ok:
            print(f"     expected /{s['expect']}/")
    print(f"test: {ran - failed} of {ran} suites match the recorded counts")
    return 1 if failed or not ran else 0


if __name__ == "__main__":
    sys.exit(main())
