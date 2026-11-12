# caps-exp: minimal criu file capabilities, run unprivileged

**Question**: what's the smallest set of Linux capabilities `criu` needs to checkpoint
and restore a warmed SGLang server, when invoked the way criu is actually designed to be
invoked unprivileged?

**Method (current)**: run as a genuinely non-root OS user (`user`, uid 1000), never as
root with capabilities dropped. `checkpoint.sh`/`restore.sh` always pass criu's
`--unprivileged` flag. The only thing that varies between tests is which Linux **file**
capabilities (`setcap`) are set on the copy of the `criu` binary that user executes —
see `scripts/caps-exp/run.sh`. Root's only role is `setcap`-ing that binary copy; it
never touches the actual dump/restore, which always runs via `sudo -u user`.

An earlier version of this experiment used `capsh --drop` on a **root** shell instead,
dropping down from Docker's default capability set. That approach is retired — see
"Why the earlier root+capsh approach was dropped" below for what it got wrong.

## Result

| test | criu file caps | checkpoint | restore | verdict |
|---|---|---|---|---|
| test1 | *(none)* | **FAIL** — refuses to run at all | — (no image) | criu needs at least one capability just to start |
| test2 | `cap_checkpoint_restore=ep` | **FAIL** — can't seize the process tree | — (no image) | the capability's name is not sufficient by itself |
| test3 | `cap_checkpoint_restore,cap_sys_ptrace=ep` | **PASS** — 25.6s, 15G snapshot | **PASS** — 22.4s to serving, 4.91x vs cold | minimal working set |

**`cap_checkpoint_restore` + `cap_sys_ptrace`, as a file capability on `criu`, granted
to a genuinely non-root user, is sufficient for the full round trip.** Nothing else
(`net_admin`, `sys_admin`, etc.) is required — a materially narrower ask than what the
root+capsh approach kept hitting a wall on (see below).

## test1 — no file caps at all

```
CRIU needs to have the CAP_SYS_ADMIN or the CAP_CHECKPOINT_RESTORE capability:
setcap cap_checkpoint_restore+eip <criu-binary>
```

criu prints this and exits before doing anything — a clean, expected failure, not a
crash. Confirms the floor: some capability is required, and criu itself points at the
fix.

## test2 — `cap_checkpoint_restore` only

```
Warn  (compel/src/lib/infect.c:132): Unable to interrupt task: <pid> (Operation not permitted)
Error (compel/src/lib/infect.c:417): Unable to detach from <pid>: No such process
Error (criu/cr-dump.c:1975): Dumping FAILED.
```

Gets further than test1 (criu starts), but can't `ptrace`-seize the target process tree.
**`CAP_CHECKPOINT_RESTORE` alone does not imply seize/ptrace rights**, despite the name
— this is the same failure signature seen on the old root+capsh `test1` (docker-default
+ `checkpoint_restore`, no `sys_ptrace`), which is reassuring: the two independent
methods agree on where the wall is.

## test3 — `cap_checkpoint_restore` + `cap_sys_ptrace`

Full round trip, no workarounds:

| step | time |
|---|---|
| checkpoint total | 34.9s (cuda-checkpoint evict 9.4s + criu dump 25.6s) |
| snapshot size | 15G |
| criu restore | 16.0s |
| cuda reattach | 6.3s |
| first served request | 108ms |
| **restore-to-served** | **22.4s** |
| cold-launch baseline | 109.8s |
| **speedup vs cold launch** | **4.91x** |
| post-restore throughput | 253.5 tok/s (64/64 reqs, 0 errors) |
| correctness | greedy-decode outputs coherent, matches prior full-privilege runs |
| GATE | **PASS** |

Full run dir: `results/caps-exp/test3/run-20260821T130648Z/` (`meta.txt`,
`server.log`, bench JSON).

No `CAP_NET_ADMIN` needed for either the `SIOCUNIXFILE` dump-time ioctl or the
`SO_SNDBUFFORCE`/`SO_RCVBUFFORCE` restore-time calls that blocked the old root+capsh
`test2`. `--unprivileged` mode evidently takes different code paths for those
operations rather than needing that capability at all — it isn't that the non-root user
secretly has `CAP_NET_ADMIN` (it doesn't; `id user` shows no capabilities).

## Why the earlier root+capsh approach was dropped

The original version of this dir (`cap_sets.sh` + `run.sh`, since rewritten) ran as
**root**, using `capsh --drop` to remove everything outside a keep-list starting from
Docker's default 14 capabilities. That produced a different, worse-behaved picture:

- Old `test1` (docker-default + `checkpoint_restore`): same ptrace-seize failure as the
  new `test2` above — consistent, good.
- Old `test2` (+ `sys_ptrace`): dump failed on `criu/sk-unix.c: Unable to get a socket
  file descriptor with SIOCUNIXFILE ioctl (Operation not permitted)`; restore (from an
  existing full-caps snapshot) failed on `criu/sockets.c: Unable to set socket
  SO_SNDBUFFORCE/SO_RCVBUFFORCE (Operation not permitted)`. Both need `CAP_NET_ADMIN`.
- Adding `--unprivileged` on top of that dropped-root-caps shell did **not** cleanly
  fix this: it got past the socket-buffer error, then hung in a retry loop inside
  criu's in-process "pie" restorer stub (`Unable to restore capabilities`, `BUG at
  restorer.c:820`), spawning and re-failing indefinitely without ever actually serving.
  Killed by hand after ~20 minutes.

Old `test1`/`test2` run dirs from that methodology are still present under
`results/caps-exp/test1/` and `results/caps-exp/test2/` (timestamps `10:2x`) alongside
the new non-root runs (`13:0x`) — same directory names, different methodology. The
takeaway that matters: **root-with-`capsh --drop` and non-root-with-file-caps are not
equivalent even when the "kept" capability list looks the same on paper** (old test2's
keep-list == new test3's file caps), and only the genuinely-unprivileged path actually
works cleanly here.

## Deployment takeaway

The deployable answer isn't "which capabilities does a root container need
`--cap-add`ed" — that path kept hitting `CAP_NET_ADMIN`, which most container
platforms are reluctant to grant broadly. It's: **run the server as a non-root user,
and give the `criu` binary `cap_checkpoint_restore` + `cap_sys_ptrace` as file
capabilities via `setcap`.** That's a narrow, single-binary, `setcap`-time grant a
security-conscious platform is far more likely to accept than a broad root capability
set.
