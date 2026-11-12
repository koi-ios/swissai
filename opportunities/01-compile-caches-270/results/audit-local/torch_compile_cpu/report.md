# cache-audit report

Command: `/Users/koiios/Documents/Projects/swissai/opportunities/01-compile-caches-270/.venv/bin/python /Users/koiios/Documents/Projects/swissai/opportunities/01-compile-caches-270/tests/workloads/torch_compile_cpu.py`

**Gate: PASS** (exit 0)

## Run `cold`: exit 0, 22.6 s, tracers snapshot+audithook

| root | KB entry | variable | files | size | contents | abs paths baked | how written | writers |
|---|---|---|---:|---:|---|---:|---|---|
| `<sandbox>/home/Library/Caches/Homebrew` | macos.homebrew | `HOMEBREW_CACHE` | 637 | 35.7 MB | other 633, metadata 4 | 0 | no live trace | - |
| `<sandbox>/tmp/torchinductor_koiios/precompiled_headers` | inductor.pch | `-` | 3 | 29.6 MB | source 1, other 1, lock 1 | 0 | mixed (1 renames/links, 2 direct opens) | /private/tmp/claude-501/-Users-koiios-Documents-Projects-swissai/a94ca039-9dd7-4f72-9781-19ec1d4e1034/scratchpad/audit-torch_compile_cpu/cold-702029/tmp/torchinductor_koiios/3m/c3mo5soouw4mtdok44bxqzfsw7mir5ydqmbcfsy2lbsqjqndml2p.py:33 (7) |
| `<sandbox>/tmp/torchinductor_koiios` | inductor | `TORCHINDUCTOR_CACHE_DIR` | 32 | 460.9 KB | binary 10, other 10, source 7, lock 5 | 15 | mixed (7 renames/links, 28 direct opens) | <string>:13 (8), /Users/koiios/Documents/Projects/swissai/opportunities/01-compile-caches-270/tests/workloads/torch_compile_cpu.py:28 (8) |
| `<sandbox>/tmp/torchinductor_koiios/fxgraph` | inductor.fxgraph | `TORCHINDUCTOR_CACHE_DIR` | 2 | 176.4 KB | other 2 | 2 | atomic publish (2 renames/links, 2 direct opens) | /Users/koiios/Documents/Projects/swissai/opportunities/01-compile-caches-270/tests/workloads/torch_compile_cpu.py:28 (10) |
| `<sandbox>/tmp/torchinductor_koiios/aotautograd` | inductor.aotautograd | `TORCHINDUCTOR_CACHE_DIR` | 2 | 151.9 KB | other 2 | 0 | atomic publish (2 renames/links, 2 direct opens) | /Users/koiios/Documents/Projects/swissai/opportunities/01-compile-caches-270/tests/workloads/torch_compile_cpu.py:28 (6) |

## Run `redirect`: exit 0, 19.1 s, tracers snapshot+audithook
Variables set: `TORCHINDUCTOR_CACHE_DIR`

| root | KB entry | variable | files | size | contents | abs paths baked | how written | writers |
|---|---|---|---:|---:|---|---:|---|---|
| `<sandbox>/home/Library/Caches/Homebrew` | macos.homebrew | `HOMEBREW_CACHE` | 637 | 35.7 MB | other 633, metadata 4 | 0 | no live trace | - |
| `<sandbox>/tmp/torchinductor_koiios/precompiled_headers` | inductor.pch | `-` | 3 | 29.6 MB | source 1, other 1, lock 1 | 0 | mixed (1 renames/links, 2 direct opens) | /private/tmp/claude-501/-Users-koiios-Documents-Projects-swissai/a94ca039-9dd7-4f72-9781-19ec1d4e1034/scratchpad/audit-torch_compile_cpu/redirect-20814f/cache/inductor/3m/c3mo5soouw4mtdok44bxqzfsw7mir5ydqmbcfsy2lbsqjqndml2p.py:33 (7) |
| `<sandbox>/cache/inductor` | inductor | `TORCHINDUCTOR_CACHE_DIR` | 32 | 460.7 KB | binary 10, other 10, source 7, lock 5 | 15 | mixed (7 renames/links, 28 direct opens) | <string>:13 (8), /Users/koiios/Documents/Projects/swissai/opportunities/01-compile-caches-270/tests/workloads/torch_compile_cpu.py:28 (8) |
| `<sandbox>/cache/inductor/fxgraph` | inductor.fxgraph | `TORCHINDUCTOR_CACHE_DIR` | 2 | 176.3 KB | other 2 | 2 | atomic publish (2 renames/links, 2 direct opens) | /Users/koiios/Documents/Projects/swissai/opportunities/01-compile-caches-270/tests/workloads/torch_compile_cpu.py:28 (10) |
| `<sandbox>/cache/inductor/aotautograd` | inductor.aotautograd | `TORCHINDUCTOR_CACHE_DIR` | 2 | 151.9 KB | other 2 | 0 | atomic publish (2 renames/links, 2 direct opens) | /Users/koiios/Documents/Projects/swissai/opportunities/01-compile-caches-270/tests/workloads/torch_compile_cpu.py:28 (6) |

## Run `warm`: exit 0, 13.3 s, tracers snapshot+audithook
Variables set: `TORCHINDUCTOR_CACHE_DIR`

| root | KB entry | variable | files | size | contents | abs paths baked | how written | writers |
|---|---|---|---:|---:|---|---:|---|---|
| `<sandbox>/home-warm/Library/Caches/Homebrew` | macos.homebrew | `HOMEBREW_CACHE` | 637 | 35.7 MB | other 633, metadata 4 | 0 | no live trace | - |
| `<sandbox>/tmp-warm/torchinductor_koiios/precompiled_headers` | inductor.pch | `-` | 3 | 29.6 MB | source 1, other 1, lock 1 | 0 | mixed (1 renames/links, 2 direct opens) | /private/tmp/claude-501/-Users-koiios-Documents-Projects-swissai/a94ca039-9dd7-4f72-9781-19ec1d4e1034/scratchpad/audit-torch_compile_cpu/redirect-20814f/cache/inductor/3m/c3mo5soouw4mtdok44bxqzfsw7mir5ydqmbcfsy2lbsqjqndml2p.py:33 (7) |

## Run `relocate`: exit 0, 14.7 s, tracers snapshot+audithook
Variables set: `TORCHINDUCTOR_CACHE_DIR`

| root | KB entry | variable | files | size | contents | abs paths baked | how written | writers |
|---|---|---|---:|---:|---|---:|---|---|
| `<sandbox>/home-relocate/Library/Caches/Homebrew` | macos.homebrew | `HOMEBREW_CACHE` | 637 | 35.7 MB | other 633, metadata 4 | 0 | no live trace | - |
| `<sandbox>/tmp-relocate/torchinductor_koiios/precompiled_headers` | inductor.pch | `-` | 3 | 29.6 MB | source 1, other 1, lock 1 | 0 | mixed (1 renames/links, 2 direct opens) | /private/tmp/claude-501/-Users-koiios-Documents-Projects-swissai/a94ca039-9dd7-4f72-9781-19ec1d4e1034/scratchpad/audit-torch_compile_cpu/redirect-20814f/cache-moved/inductor/3m/c3mo5soouw4mtdok44bxqzfsw7mir5ydqmbcfsy2lbsqjqndml2p.py:33 (7) |

## Run `readonly`: exit 1, 14.2 s, tracers snapshot+audithook
Variables set: `TORCHINDUCTOR_CACHE_DIR`

| root | KB entry | variable | files | size | contents | abs paths baked | how written | writers |
|---|---|---|---:|---:|---|---:|---|---|
| `<sandbox>/home-readonly/Library/Caches/Homebrew` | macos.homebrew | `HOMEBREW_CACHE` | 637 | 35.7 MB | other 633, metadata 4 | 0 | no live trace | - |
| `<sandbox>/tmp-readonly/torchinductor_koiios/precompiled_headers` | inductor.pch | `-` | 3 | 29.6 MB | source 1, other 1, lock 1 | 0 | mixed (1 renames/links, 2 direct opens) | /private/tmp/claude-501/-Users-koiios-Documents-Projects-swissai/a94ca039-9dd7-4f72-9781-19ec1d4e1034/scratchpad/audit-torch_compile_cpu/redirect-20814f/cache-ro/inductor/3m/c3mo5soouw4mtdok44bxqzfsw7mir5ydqmbcfsy2lbsqjqndml2p.py:33 (7) |

## Mapping verified by re-run

| KB entry | verdict |
|---|---|
| inductor | verified |
| inductor.aotautograd | verified |
| inductor.fxgraph | verified |

## Warm run: builds avoided

| KB entry | binaries cold | binaries rebuilt warm | files cold | files written warm |
|---|---:|---:|---:|---:|
| inductor | 10 | 0 | 32 | 0 |
| inductor.aotautograd | 0 | 0 | 2 | 0 |
| inductor.fxgraph | 0 | 0 | 2 | 0 |

Wall time: cold 19.1 s, warm 13.3 s

## Relocation (cache copied to a new absolute path)

Files rewritten after the move, per entry: none (relocatable)

## Read-only cache

```
{
  "returncode": 1,
  "write_attempts_into_ro_cache": {
    "inductor": 26
  },
  "failed_writes": 0
}
```

## Env snippet

```toml
# Generated by cache-audit. {jit_cache} is resolved by sml per image digest and GPU arch.
# Paste into the env TOML's [env] table.
TORCHINDUCTOR_CACHE_DIR = "{jit_cache}/inductor"  # inductor, inductor.fxgraph, inductor.aotautograd
```
