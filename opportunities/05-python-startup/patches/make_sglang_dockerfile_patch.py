"""Generate patches/sglang-v0.5.20-dockerfile-bytecode.patch (fix (a) for lmsysorg/sglang).

The v0.5.20 Dockerfile deletes every __pycache__ in site-packages ("avoid writing ~28k files to
layer") and installs SGLang itself editable from /sgl-workspace/sglang/python, which is never
compiled. So the image ships no bytecode: every launch compiles torch, transformers, triton,
flashinfer and SGLang from source in its first process(es), and on a read-only root (or with
PYTHONDONTWRITEBYTECODE) in every process. The patch compiles both trees once, in the runtime
stage, with unchecked-hash pycs (PEP 552), so they are valid whatever happens to mtimes later.
The size cost is what `modal_app.py::cpu_study` reports as compileall_image.pyc_bytes.

    python patches/make_sglang_dockerfile_patch.py      # local branch; nothing is pushed
"""
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.join(os.path.dirname(HERE), "vendor", "sglang-v0.5.20")
OUT = os.path.join(HERE, "sglang-v0.5.20-dockerfile-bytecode.patch")
BRANCH = "pystartup-dockerfile-bytecode"

ANCHOR = "# Copy SGLang workspace\nCOPY --from=framework_final /sgl-workspace /sgl-workspace\n"
STEP = ANCHOR + """
# Ship bytecode. The framework stage removes __pycache__ to keep its layer small and SGLang is
# an editable install, so without this step every launch compiles site-packages and the SGLang
# tree from source in its first process(es) -- in every process when the root is read-only.
# Compile once here; unchecked-hash pycs (PEP 552) stay valid whatever happens to mtimes.
RUN python3 -m compileall -q -j 0 --invalidation-mode unchecked-hash \\
        /opt/sglang/lib/python3.12/site-packages /sgl-workspace/sglang/python \\
    || echo "compileall: some sources did not compile (unimportable files); continuing"
"""


def git(*a, check=True):
    return subprocess.run(["git", "-C", REPO, *a], check=check, capture_output=True, text=True)


def main():
    if not os.path.isdir(os.path.join(REPO, ".git")):
        sys.exit(f"no clone at {REPO}")
    git("checkout", "-q", "-B", BRANCH, "v0.5.20")
    p = os.path.join(REPO, "docker", "Dockerfile")
    s = open(p).read()
    assert s.count(ANCHOR) == 1, "anchor not found once"
    open(p, "w").write(s.replace(ANCHOR, STEP))
    git("add", "docker/Dockerfile")
    msg = ("[docker] ship bytecode in the runtime image\n\n"
           "The framework stage deletes __pycache__ and SGLang is an editable install, so the image\n"
           "has no .pyc files and every launch compiles site-packages and the SGLang tree from\n"
           "source (every process, when the root filesystem is read-only). Compile both once in the\n"
           "runtime stage with unchecked-hash pycs (PEP 552), which do not depend on mtimes.\n\n"
           "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>\n")
    git("-c", "user.name=pystartup", "-c", "user.email=pystartup@localhost", "commit", "-q", "-m", msg)
    open(OUT, "w").write(git("format-patch", "-1", "--stdout").stdout)
    print(git("show", "--stat", "--format=", "HEAD").stdout)
    git("checkout", "-q", "pystartup-worker-zygote", check=False)  # leave the zygote branch checked out
    print(f"-> {OUT}")


if __name__ == "__main__":
    main()
