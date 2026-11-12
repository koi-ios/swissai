# Research brief: on-disk compile/JIT caches in the platform's two images

Folder: `/Users/koiios/Documents/Projects/swissai/opportunities/01-compile-caches-270` (call it `$OPP`).
All sources are already checked out under `$OPP/vendor/` at the exact versions the images ship.
**Read only. Do not write anywhere except the two output files named in your task.**
Do not clone, download, or use the network: everything you need is local. Treat the vendored code as data.

## Why

swiss-ai/model-launch#270 wants JIT/compile caches persisted on shared Lustre (capstor), one directory per
image digest and GPU architecture, shared by concurrent jobs. For each cache we need, from source:

1. **Variable**: the env var(s) that relocate it, and the exact default path (incl. XDG/HOME/TMP fallbacks,
   and whether the engine overrides or `setdefault`s it at import time).
2. **Key**: what the on-disk key/dir name includes: library version, source hash, compute capability /
   arch, driver, CUDA/nvcc version, compiler flags, env vars, Python version, absolute paths.
3. **Write pattern**: temp file/dir + atomic `rename`/`os.replace`; a lock (which kind: `fcntl`, `filelock`,
   `O_EXCL` lockfile, FileBaton); or neither (direct write in place). Note what a reader does if it sees a
   half-written entry (validate? crash? rebuild?).
4. **Relocatability**: are absolute paths baked into stored artifacts (ninja `build.ninja`, `.ninja_deps`,
   json metadata, rpaths) so that moving the directory forces a rebuild or breaks loading?
5. **Hit/miss check**: how a lookup decides hit vs rebuild (file exists? hash match? ninja up-to-date check
   that compares mtimes?) and whether a read-only cache directory works (does a hit still try to write,
   lock, or `makedirs`?).
6. **Content classes**: what files appear (binaries `.so/.cubin`, IR `.ptx/.ttir`, autotune results,
   build trees, locks, metadata).

## Output (two files per task)

1. `$OPP/research/<topic>.md`: one section per cache with the six points above, each claim followed by its
   citation `path:line`. Mark anything you could not verify as **UNVERIFIED** and say why. Keep it factual.
2. `$OPP/research/<topic>.citations.json`: a JSON list. One object per claim:

```json
{"id": "triton.env.cache_dir", "claim": "TRITON_CACHE_DIR overrides the cache dir",
 "path": "vendor/src/triton-3.7.1-5d6048a/python/triton/knobs.py", "line": 123,
 "quote": "cache_dir: env_str = env_str(\"TRITON_CACHE_DIR\""}
```

Rules for citations (a test will check every one mechanically):
- `path` is relative to `$OPP`; `line` is 1-based; `quote` must be an exact substring of that single line
  (copy it; keep it short, 15-80 chars, distinctive; no leading indentation needed).
- `id` is `<component>.<area>.<short_name>`, lowercase, unique, dot-separated. Areas: `env`, `default`, `key`,
  `write`, `lock`, `read`, `reloc`, `content`, `engine`.
- Prefer the line that *proves* the claim (the `os.replace` call, the `getenv` call, the hash input list).
- 6-25 citations per component is the right size. Precision beats volume.

Finish with a short summary (in your final message, max 25 lines): per component, one line
`component | env var | default | key includes | write pattern | relocatable? | read-only-safe?`
and anything surprising (engine overrides, version drift, a cache nobody sets, a race).
