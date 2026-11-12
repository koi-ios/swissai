#!/usr/bin/env python3
"""make clones: every git checkout in the workspace, its role and state.

  clones.py                 list every clone and worktree
  clones.py verify          for each project, does the branch in its clone equal its tracked patches?
  clones.py ensure NAME     create the canonical clone vendor/NAME if it is missing (needs the network)

Roles: canonical (vendor/<name> at the root), clone or worktree (inside a project's vendor/),
export (a worktree made by `make export`), stub (a submodule placeholder inside unpacked sources).
`verify` never writes into a clone: it builds trees in a temporary object store.
"""
from __future__ import annotations

import glob
import os
import pathlib
import subprocess
import sys
import tempfile

from lib import ROOT, card_name, die, git, projects, workspace

BIG = {"sglang", "vllm"}          # cloned without blobs and without a checkout
EXCLUDES = ["CLAUDE.local.md", ".claude/settings.local.json", ".agents/", ".cursor/", ".entire/"]


def checkouts() -> list[pathlib.Path]:
    out = []
    for dirpath, dirnames, filenames in os.walk(ROOT):
        if dirpath == str(ROOT):
            dirnames[:] = [d for d in dirnames if d != ".git"]
        dirnames[:] = [d for d in dirnames if d not in (".venv", "toolchain", "models", "node_modules", "__pycache__")]
        if ".git" in dirnames or ".git" in filenames:
            out.append(pathlib.Path(dirpath))
            if ".git" in dirnames:
                dirnames.remove(".git")
    return sorted(out)


def role(path: pathlib.Path) -> str:
    rel = path.relative_to(ROOT).parts
    if len(rel) == 2 and rel[0] == "vendor":
        return "canonical"
    if "export" in rel:
        return "export"
    if (path / ".git").is_file():
        return "worktree" if git("rev-parse", "--git-dir", cwd=path, check=False) else "stub"
    return "clone"


def cmd_list() -> int:
    counts: dict[str, int] = {}
    for path in checkouts():
        r = role(path)
        counts[r] = counts.get(r, 0) + 1
        if r == "stub":
            continue
        head = git("rev-parse", "--short", "HEAD", cwd=path, check=False) or "?"
        branch = git("symbolic-ref", "--short", "-q", "HEAD", cwd=path, check=False) or "detached"
        dirty = len([ln for ln in git("status", "--porcelain", cwd=path, check=False).splitlines() if ln])
        url = git("config", "--get", "remote.origin.url", cwd=path, check=False).removeprefix("https://github.com/").removesuffix(".git")
        shallow = " shallow" if git("rev-parse", "--is-shallow-repository", cwd=path, check=False) == "true" else ""
        print(f"{r:<9} {str(path.relative_to(ROOT)):<62} {url:<26} {branch:<32.32} {head} dirty={dirty}{shallow}")
    print("clones: " + ", ".join(f"{v} {k}" for k, v in sorted(counts.items())))
    return 0


def _scratch_env(clone: pathlib.Path, tmp: str) -> dict:
    common = pathlib.Path(git("rev-parse", "--path-format=absolute", "--git-common-dir", cwd=clone))
    os.makedirs(f"{tmp}/objects")
    return dict(os.environ, GIT_INDEX_FILE=f"{tmp}/index", GIT_OBJECT_DIRECTORY=f"{tmp}/objects",
                GIT_ALTERNATE_OBJECT_DIRECTORIES=str(common / "objects"), GIT_NO_LAZY_FETCH="1")


def _try(args: list[str], clone: pathlib.Path, env: dict) -> tuple[bool, str]:
    p = subprocess.run(["git", *args], cwd=clone, env=env, capture_output=True, text=True)
    return p.returncode == 0, (p.stdout if p.returncode == 0 else p.stderr).strip()


def tree_of_series(clone: pathlib.Path, base: str, patches: list[str]) -> tuple[str, str]:
    """(tree, problem): the tree that results from applying the patches to base, built in a temporary
    object store so nothing is written into the clone."""
    with tempfile.TemporaryDirectory() as tmp:
        env = _scratch_env(clone, tmp)
        for args in (["read-tree", base], *(["apply", "--cached", p] for p in patches), ["write-tree"]):
            ok, out = _try(list(args), clone, env)
            if not ok:
                partial = "promisor" in out or "lazy fetching" in out
                return "", "partial clone: objects are missing and nothing is fetched" if partial else out.splitlines()[-1][:90]
        return out, ""


def tree_of_worktree(clone: pathlib.Path) -> tuple[str, str]:
    with tempfile.TemporaryDirectory() as tmp:
        env = _scratch_env(clone, tmp)
        for args in (["read-tree", "HEAD"], ["add", "-A"], ["write-tree"]):
            ok, out = _try(args, clone, env)
            if not ok:
                return "", out.splitlines()[-1][:90]
        return out, ""


def cmd_verify() -> int:
    bad = skipped = 0
    for p in projects():
        for u in (p["card"] or {}).get("upstream", []):
            legacy = u.get("legacy", "")
            label = f"{p['slug'][:2]} {card_name(u['repo']):<13}"
            patches = sorted(glob.glob(str(p["dir"] / u.get("series", "none"))))
            if not legacy:
                print(f"skip  {label} no clone holds this series ({len(patches)} file(s) in patches/)")
                continue
            rel, ref = legacy.split("@", 1)
            clone = p["dir"] / rel
            base = u.get("legacy_base", u.get("base", ""))
            if not clone.exists():
                print(f"gone  {label} {rel} is not on disk; the patches are the record")
                continue
            if ref == "WORKTREE":
                want, why_want = tree_of_worktree(clone)
            else:
                want = git("rev-parse", "-q", "--verify", f"{ref}^{{tree}}", cwd=clone, check=False)
                why_want = "" if want else f"{ref} not found"
            got, why_got = tree_of_series(clone, base, patches)
            if not want or not got:
                skipped += 1
                print(f"skip  {label} {rel}@{ref}: cannot compare ({why_want or why_got})")
                continue
            same = want == got
            bad += not same
            print(f"{'same' if same else 'DIFF'}  {label} {rel}@{ref} against {len(patches)} patch(es) on {base}"
                  + ("" if same else f"  (clone {want[:12]}, patches {got[:12]})"))
    print(f"clones: {bad} clone(s) hold something the patches do not" if bad
          else f"clones: every clone that could be compared equals its tracked patches ({skipped} could not be compared)")
    return 1 if bad else 0


def cmd_ensure(name: str) -> int:
    urls = workspace()["upstream"]
    if name not in urls:
        die(f"{name} is not in pipeline/workspace.toml [upstream]: {', '.join(urls)}")
    dest = ROOT / "vendor" / name
    if not (dest / ".git").exists():
        dest.parent.mkdir(exist_ok=True)
        args = ["clone", "--origin", "origin"]
        if name in BIG:
            args += ["--filter=blob:none", "--no-checkout", "--single-branch", "--no-tags"]
        git(*args, urls[name], str(dest))
        print(f"clones: created vendor/{name}")
    git("config", "worktree.useRelativePaths", "true", cwd=dest)
    exclude = dest / ".git/info/exclude"
    have = exclude.read_text() if exclude.exists() else ""
    add = [e for e in EXCLUDES if e not in have.split()]
    if add:
        exclude.write_text(have.rstrip("\n") + "\n# agent files of the private workspace: never commit them here\n" + "\n".join(add) + "\n")
    print(f"clones: vendor/{name} ready at {git('rev-parse', '--short', 'origin/HEAD', cwd=dest, check=False) or 'origin/HEAD unknown'}")
    return 0


def main() -> int:
    args = sys.argv[1:]
    if not args:
        return cmd_list()
    if args[0] == "verify":
        return cmd_verify()
    if args[0] == "ensure" and len(args) == 2:
        return cmd_ensure(args[1])
    die(__doc__)


if __name__ == "__main__":
    sys.exit(main())
