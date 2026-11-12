#!/usr/bin/env bash
# Gate 0 — environment probe. Records the facts that determine whether CRIU can run
# here at all: capability bounding set, seccomp state, criu file caps, kernel knobs.
# Pure read-only; safe to re-run.
#
# Usage: scripts/utils/probe_env.sh [output_file]
set -u
OUT="${1:-/dev/stdout}"
exec > >(tee "$OUT") 2>&1

echo "==== checkpoint-restore-exp :: environment probe ===="
echo "date(host clock): $(date -u '+%Y-%m-%dT%H:%M:%SZ')"
echo "hostname: $(hostname)"
echo

echo "== container check =="
[ -f /.dockerenv ] && echo "/.dockerenv present (container)" || echo "/.dockerenv absent"
echo "cgroup: $(cat /proc/1/cgroup 2>/dev/null | tr '\n' ' ')"
echo

echo "== user / caps =="
id
grep -E 'Cap(Inh|Prm|Eff|Bnd|Amb)|Seccomp' /proc/self/status
echo

echo "== GPU / driver =="
nvidia-smi --query-gpu=index,name,memory.total,driver_version --format=csv,noheader 2>/dev/null \
  || echo "(nvidia-smi unavailable)"
echo

echo "== RAM / disk =="
free -h | awk 'NR==1 || /Mem:/'
df -h / | awk 'NR==1 || NR==2'
echo

echo "== criu =="
CRIU="$(command -v criu || echo /usr/sbin/criu)"
echo "path: $CRIU"
"$CRIU" --version 2>&1 | head -1
echo -n "file caps: "; getcap "$CRIU" 2>/dev/null || echo "(none)"
echo "-- criu check (feature probe; non-fatal) --"
"$CRIU" check 2>&1 | tail -5
echo

echo "== kernel knobs =="
echo "ptrace_scope        = $(cat /proc/sys/kernel/yama/ptrace_scope 2>/dev/null)"
echo "unprivileged_userns = $(cat /proc/sys/kernel/unprivileged_userns_clone 2>/dev/null || sysctl -n kernel.unprivileged_userns_clone 2>/dev/null)"
echo "kernel              = $(uname -r)"
echo

echo "== interpretation =="
echo "The known blocker pattern (Bristen, and the first vast.ai container instance on"
echo "this same plan) is: CapBnd excludes cap_checkpoint_restore/cap_sys_ptrace/"
echo "cap_net_admin (unfixable from inside a container), and/or Seccomp != 0. If CapBnd"
echo "here is the full set and Seccomp is 0, this box is not container-restricted."
echo "==== end probe ===="
