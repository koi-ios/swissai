"""Generate patches/model-launch-python-startup.patch (fix (a) in the images + a wrapper hook).

Changes, on a local branch of vendor/model-launch (nothing is pushed):
* images/{sglang_cuda13,sglang_0.5.20,vllm_cuda13,vllm_0.30.0,vllm_cuda13_v2,
  vllm_apertus_1.5_release}/Dockerfile: a final, gap-filling `compileall` step with
  hash-based pycs that are never checked against the source (PEP 552 unchecked-hash). The uv-built
  vLLM images ship no bytecode at all today; in the pip-built SGLang image the step only fills
  gaps (existing valid pycs are kept: no -f), so the layer stays small.
* src/swiss_ai_model_launch/launchers/framework.py: the SGLang entrypoint becomes
  `${SML_SGLANG_ENTRYPOINT:-python3 -m sglang.launch_server}` so a deployment can switch to
  `python3 -m pystartup.sglang_launch --` (worker zygote) through the environment, without a
  code change. Default behaviour and tests/unit/test_rank_script_content.py are unchanged.

    python patches/make_model_launch_patch.py
"""
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.join(os.path.dirname(HERE), "vendor", "model-launch")
OUT = os.path.join(HERE, "model-launch-python-startup.patch")
BRANCH = "pystartup-bytecode"

STEP = '''
# Ship bytecode with the image. Without it the first process of every launch compiles torch,
# transformers and the engine from source (uv never compiles; pip does), and on a read-only
# root every process does. Hash-based pycs that are never checked against the source
# (PEP 552, unchecked-hash) suit an immutable image and survive mtime changes on the way to
# the .sqsh. No -f: valid pycs that pip already wrote are kept.
RUN python -m compileall -q -j 0 --invalidation-mode unchecked-hash {paths} \\
    || echo "compileall: some sources did not compile (unimportable files); continuing"
'''
PURELIB = r'''"$(python -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])')"'''

DOCKERFILES = {
    "images/sglang_cuda13/Dockerfile": PURELIB,
    "images/sglang_0.5.20/Dockerfile": PURELIB,
    "images/vllm_cuda13/Dockerfile": PURELIB,
    "images/vllm_0.30.0/Dockerfile": PURELIB,
    "images/vllm_cuda13_v2/Dockerfile": PURELIB,
    "images/vllm_apertus_1.5_release/Dockerfile": PURELIB + " /workspace/vllm",
}


def git(*a, check=True):
    return subprocess.run(["git", "-C", REPO, *a], check=check, capture_output=True, text=True)


def add_step(rel: str, paths: str):
    p = os.path.join(REPO, rel)
    s = open(p).read().rstrip("\n")
    step = STEP.format(paths=paths)
    if s.endswith("WORKDIR /opt"):
        s = s[: -len("WORKDIR /opt")].rstrip("\n") + "\n" + step + "\nWORKDIR /opt\n"
    else:
        s = s + "\n" + step
    open(p, "w").write(s)


def main():
    if not os.path.isdir(os.path.join(REPO, ".git")):
        sys.exit(f"no clone at {REPO}")
    base = git("rev-parse", "origin/HEAD", check=False).stdout.strip() or git("rev-parse", "HEAD").stdout.strip()
    git("checkout", "-q", "-B", BRANCH, base)
    for rel, paths in DOCKERFILES.items():
        add_step(rel, paths)
    fw = os.path.join(REPO, "src/swiss_ai_model_launch/launchers/framework.py")
    s = open(fw).read()
    old = '    entrypoint = "python3 -m sglang.launch_server"\n'
    assert s.count(old) == 1
    s = s.replace(old, '    # Overridable at job runtime (bash expansion in the rank scripts), e.g.\n'
                       '    # SML_SGLANG_ENTRYPOINT="python3 -m pystartup.sglang_launch --" to fork the TP\n'
                       '    # workers from a preloaded zygote instead of spawning fresh interpreters.\n'
                       '    entrypoint = "${SML_SGLANG_ENTRYPOINT:-python3 -m sglang.launch_server}"\n')
    open(fw, "w").write(s)
    git("add", "-A", "images", "src/swiss_ai_model_launch/launchers/framework.py")
    msg = ("images: ship bytecode; sglang: overridable entrypoint\n\n"
           "The uv-built vLLM images contain no .pyc files, so the first process of every launch\n"
           "compiles torch, transformers and vLLM from source (and every process does on a\n"
           "read-only root). Add a gap-filling compileall step with unchecked-hash pycs to the\n"
           "SGLang and vLLM Dockerfiles. Make the SGLang entrypoint overridable through\n"
           "SML_SGLANG_ENTRYPOINT so a deployment can start SGLang through a launcher such as the\n"
           "worker-zygote wrapper without a code change; the default is unchanged.\n\n"
           "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>\n")
    git("-c", "user.name=pystartup", "-c", "user.email=pystartup@localhost", "commit", "-q", "-m", msg)
    open(OUT, "w").write(git("format-patch", "-1", "--stdout").stdout)
    print(git("show", "--stat", "--format=", "HEAD").stdout)
    print(f"-> {OUT}")


if __name__ == "__main__":
    main()
