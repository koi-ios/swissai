# A new machine

Git carries the control plane, the notes, every project's patches and its results: about 870 files and 26 MB of history. It does not carry about 15 GB of clones, environments, toolchains, models and run logs (see `.gitignore`). Each project's `patches/` folder is the record of its change, and everything below can be rebuilt from it or downloaded again.

The GitHub repository is private and only you push to it. Agents never push (`AGENTS.md`, decision D11 in `pipeline/DOCTRINE.md`).

## 1. Clone and set the local identity

```bash
git clone <the URL of your private repository> swissai
cd swissai
git config --local user.name "swissai local"
git config --local user.email local@swissai.invalid
```

The local identity lives in `.git/config`, so a clone does not bring it. Commits in this repository use it rather than the machine's global identity (`pipeline/DECISIONS.md`, 12 Nov 2026). Exports to upstream set their own identity from `pipeline/workspace.toml`.

## 2. Check

Needs `make`, `git` and Python 3.11 or later (the scripts use `tomllib` and nothing outside the standard library).

```bash
make check
make status
```

`make check` warns if a push, `gh`, Modal or Hugging Face credential can be reached from the shell (D11). The warning is expected on a machine where you push; it says agent sessions there could reach the credential too. Prefer an SSH key with a passphrase that is not cached, or a fine-grained token limited to this one repository, so that an agent session cannot push silently.

## 3. Rebuild only what a task needs

| What | Where | How |
|---|---|---|
| Canonical upstream clones | `vendor/<name>` | `python3 scripts/clones.py ensure NAME`, for NAME in OpenTela, servekit, model-launch, sglang, vllm. URLs are in `pipeline/workspace.toml`. `make clones` lists what exists. |
| A project's own clone | `opportunities/<NN>/vendor/` | No script. Clone the upstream named in the project's `project.toml`, check out its `base` commit, and apply `patches/*.patch` with `git am`. `make clones-verify` compares a clone with its patches. |
| Python environments | `opportunities/<NN>/.venv/` | No requirements files are tracked. See the project's `WRITEUP.md` or `STATUS.md`. |
| 06 Go toolchain | `opportunities/06-opentela-73/toolchain/` | Go 1.26.11, unpacked there; `env.sh` points at `toolchain/go`. |
| 07 test models | `opportunities/07-verify-tolerance/models/` | `Qwen3-0.6B` and `Qwen3-0.6B-FP8` from Hugging Face; `Qwen3-0.6B-MXFP4-mlp` is made by `cpu/mxfp4.py`. |
| Trace parquet | `data/swissai-trace.parquet` | Follow `data/swissai-trace.README.md`, then check it against `data/SHA256SUMS`. |
| 06 run logs and builds | `opportunities/06-opentela-73/e2e/runs/`, `e2e/bin/` | Written again by 06's scripts. |

## 4. Moving between machines

Commit before you leave one machine and `git pull` when you arrive on the next. Untracked files stay behind, and so does anything under an ignored folder. `make backup DEST=dir` still writes a git bundle if you want a copy that is not on GitHub.
