"""Apply the lazy-capture patch to the SGLang installed in the current image.

Run at image build time:  python3 apply_patch.py <patch> [--expect-version 0.5.20]

Patch paths look like a/python/sglang/srt/...; the installed package may be an
editable checkout (<repo>/python/sglang) or site-packages/sglang, so the patch
is applied relative to the directory that contains `sglang/`, stripping the
two leading components (a/ and python/).
"""

from __future__ import annotations

import argparse
import importlib.util
import os
import re
import shutil
import subprocess
import sys


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("patch")
    ap.add_argument("--expect-version", default="0.5.20")
    a = ap.parse_args()
    spec = importlib.util.find_spec("sglang")
    if spec is None or spec.origin is None:
        print("sglang is not installed in this image", file=sys.stderr)
        return 2
    pkg = os.path.dirname(spec.origin)
    root = os.path.dirname(pkg)
    version_file = os.path.join(pkg, "version.py")
    if a.expect_version and os.path.exists(version_file):
        text = open(version_file).read()
        m = re.search(r"__version__\s*=\s*[\"']([^\"']+)", text)
        found = m.group(1) if m else "?"
        if found != a.expect_version:
            print(f"expected sglang {a.expect_version}, image has {found}", file=sys.stderr)
            return 3
    if shutil.which("patch"):
        cmd = ["patch", "-p2", "--forward", "--batch", "-d", root, "-i", a.patch]
    else:
        cmd = ["git", "apply", "-p2", "--unsafe-paths", f"--directory={root}", a.patch]
    print("+", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)
    marker = os.path.join(pkg, "srt", "model_executor", "runner", "lazy_capture.py")
    if not os.path.exists(marker):
        print(f"patch did not create {marker}", file=sys.stderr)
        return 4
    print(f"patched sglang at {pkg}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
