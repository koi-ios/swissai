# cache-audit report

Command: `/Users/koiios/Documents/Projects/swissai/opportunities/01-compile-caches-270/.venv/bin/python /Users/koiios/Documents/Projects/swissai/opportunities/01-compile-caches-270/tests/workloads/tilelang_cpu.py`

**Gate: PASS** (exit 0)

## Run `cold`: exit 0, 7.7 s, tracers snapshot+audithook

| root | KB entry | variable | files | size | contents | abs paths baked | how written | writers |
|---|---|---|---:|---:|---|---:|---|---|
| `<sandbox>/home/.tilelang/cache` | tilelang | `TILELANG_CACHE_DIR` | 12 | 53.8 KB | source 6, binary 3, serialized 3 | 3 | atomic publish (15 renames/links, 12 direct opens) | /Users/koiios/Documents/Projects/swissai/opportunities/01-compile-caches-270/tests/workloads/tilelang_cpu.py:26 (42), /Users/koiios/Documents/Projects/swissai/opportunities/01-compile-caches-270/tests/workloads/tilelang_cpu.py:7 (21) |
| `<sandbox>/tmp/tmp8vd7syqp.so` | UNKNOWN | `-` | 1 | 16.6 KB | temp 1 | 0 | no live trace | - |
| `<sandbox>/tmp/tmpb0x65cnd.so` | UNKNOWN | `-` | 1 | 16.6 KB | temp 1 | 0 | no live trace | - |
| `<sandbox>/tmp/tmpfi3rci9c.so` | UNKNOWN | `-` | 1 | 16.6 KB | temp 1 | 0 | no live trace | - |
| `<sandbox>/tmp/tmp8vd7syqp.cpp` | UNKNOWN | `-` | 1 | 759 B | temp 1 | 0 | direct writes (1 opens, no rename) | /Users/koiios/Documents/Projects/swissai/opportunities/01-compile-caches-270/tests/workloads/tilelang_cpu.py:26 (2) |
| `<sandbox>/tmp/tmpb0x65cnd.cpp` | UNKNOWN | `-` | 1 | 759 B | temp 1 | 0 | direct writes (1 opens, no rename) | /Users/koiios/Documents/Projects/swissai/opportunities/01-compile-caches-270/tests/workloads/tilelang_cpu.py:26 (2) |
| `<sandbox>/tmp/tmpfi3rci9c.cpp` | UNKNOWN | `-` | 1 | 758 B | temp 1 | 0 | direct writes (1 opens, no rename) | /Users/koiios/Documents/Projects/swissai/opportunities/01-compile-caches-270/tests/workloads/tilelang_cpu.py:26 (2) |

## Run `redirect`: exit 0, 8.8 s, tracers snapshot+audithook
Variables set: `TILELANG_CACHE_DIR`

| root | KB entry | variable | files | size | contents | abs paths baked | how written | writers |
|---|---|---|---:|---:|---|---:|---|---|
| `<sandbox>/cache/tilelang` | tilelang | `TILELANG_CACHE_DIR` | 12 | 53.8 KB | source 6, binary 3, serialized 3 | 3 | atomic publish (15 renames/links, 12 direct opens) | /Users/koiios/Documents/Projects/swissai/opportunities/01-compile-caches-270/tests/workloads/tilelang_cpu.py:26 (42), /Users/koiios/Documents/Projects/swissai/opportunities/01-compile-caches-270/tests/workloads/tilelang_cpu.py:7 (21) |
| `<sandbox>/tmp/tmp0mzd07ym.so` | UNKNOWN | `-` | 1 | 16.6 KB | temp 1 | 0 | no live trace | - |
| `<sandbox>/tmp/tmp8l2th3gg.so` | UNKNOWN | `-` | 1 | 16.6 KB | temp 1 | 0 | no live trace | - |
| `<sandbox>/tmp/tmpcofz0w3p.so` | UNKNOWN | `-` | 1 | 16.6 KB | temp 1 | 0 | no live trace | - |
| `<sandbox>/tmp/tmp0mzd07ym.cpp` | UNKNOWN | `-` | 1 | 759 B | temp 1 | 0 | direct writes (1 opens, no rename) | /Users/koiios/Documents/Projects/swissai/opportunities/01-compile-caches-270/tests/workloads/tilelang_cpu.py:26 (2) |
| `<sandbox>/tmp/tmp8l2th3gg.cpp` | UNKNOWN | `-` | 1 | 759 B | temp 1 | 0 | direct writes (1 opens, no rename) | /Users/koiios/Documents/Projects/swissai/opportunities/01-compile-caches-270/tests/workloads/tilelang_cpu.py:26 (2) |
| `<sandbox>/tmp/tmpcofz0w3p.cpp` | UNKNOWN | `-` | 1 | 758 B | temp 1 | 0 | direct writes (1 opens, no rename) | /Users/koiios/Documents/Projects/swissai/opportunities/01-compile-caches-270/tests/workloads/tilelang_cpu.py:26 (2) |

## Run `warm`: exit 0, 4.8 s, tracers snapshot+audithook
Variables set: `TILELANG_CACHE_DIR`

| root | KB entry | variable | files | size | contents | abs paths baked | how written | writers |
|---|---|---|---:|---:|---|---:|---|---|

## Run `relocate`: exit 0, 4.4 s, tracers snapshot+audithook
Variables set: `TILELANG_CACHE_DIR`

| root | KB entry | variable | files | size | contents | abs paths baked | how written | writers |
|---|---|---|---:|---:|---|---:|---|---|

## Run `readonly`: exit 0, 4.1 s, tracers snapshot+audithook
Variables set: `TILELANG_CACHE_DIR`

| root | KB entry | variable | files | size | contents | abs paths baked | how written | writers |
|---|---|---|---:|---:|---|---:|---|---|

## Mapping verified by re-run

| KB entry | verdict |
|---|---|
| tilelang | verified |

## Leaked temporary files (randomly named, never reused; not gated)

- `/private/tmp/claude-501/-Users-koiios-Documents-Projects-swissai/a94ca039-9dd7-4f72-9781-19ec1d4e1034/scratchpad/audit-tilelang3/redirect-0e1041/tmp/tmp0mzd07ym.so` (16.6 KB; writer: no Python frame (C/Cython))
- `/private/tmp/claude-501/-Users-koiios-Documents-Projects-swissai/a94ca039-9dd7-4f72-9781-19ec1d4e1034/scratchpad/audit-tilelang3/redirect-0e1041/tmp/tmp8l2th3gg.so` (16.6 KB; writer: no Python frame (C/Cython))
- `/private/tmp/claude-501/-Users-koiios-Documents-Projects-swissai/a94ca039-9dd7-4f72-9781-19ec1d4e1034/scratchpad/audit-tilelang3/redirect-0e1041/tmp/tmpcofz0w3p.so` (16.6 KB; writer: no Python frame (C/Cython))
- `/private/tmp/claude-501/-Users-koiios-Documents-Projects-swissai/a94ca039-9dd7-4f72-9781-19ec1d4e1034/scratchpad/audit-tilelang3/redirect-0e1041/tmp/tmp0mzd07ym.cpp` (759 B; writer: /Users/koiios/Documents/Projects/swissai/opportunities/01-compile-caches-270/tests/workloads/tilelang_cpu.py:26)
- `/private/tmp/claude-501/-Users-koiios-Documents-Projects-swissai/a94ca039-9dd7-4f72-9781-19ec1d4e1034/scratchpad/audit-tilelang3/redirect-0e1041/tmp/tmp8l2th3gg.cpp` (759 B; writer: /Users/koiios/Documents/Projects/swissai/opportunities/01-compile-caches-270/tests/workloads/tilelang_cpu.py:26)
- `/private/tmp/claude-501/-Users-koiios-Documents-Projects-swissai/a94ca039-9dd7-4f72-9781-19ec1d4e1034/scratchpad/audit-tilelang3/redirect-0e1041/tmp/tmpcofz0w3p.cpp` (758 B; writer: /Users/koiios/Documents/Projects/swissai/opportunities/01-compile-caches-270/tests/workloads/tilelang_cpu.py:26)

## Warm run: builds avoided

| KB entry | binaries cold | binaries rebuilt warm | files cold | files written warm |
|---|---:|---:|---:|---:|
| tilelang | 3 | 0 | 12 | 0 |

Wall time: cold 8.8 s, warm 4.8 s

## Relocation (cache copied to a new absolute path)

Files rewritten after the move, per entry: none (relocatable)

## Read-only cache

```
{
  "returncode": 0,
  "write_attempts_into_ro_cache": {
    "tilelang": 25
  },
  "failed_writes": 0
}
```

## Env snippet

```toml
# Generated by cache-audit. {jit_cache} is resolved by sml per image digest and GPU arch.
# Paste into the env TOML's [env] table.
TILELANG_CACHE_DIR = "{jit_cache}/tilelang"  # tilelang
```
