#!/usr/bin/env bash
# Shared config + helpers for every script in this experiment. Source, don't execute:
#
#     source "$(dirname "$0")/utils/common.sh"
#
# Ported from checkpoint-restore-local-exp/scripts/lib/ckpt_common.sh, plus:
#   - GPU pinning (CUDA_VISIBLE_DEVICES=$GPU_IDS, --tensor-parallel-size $TP_SIZE) in
#     launch_and_warm, since this box has 2 GPUs and nothing may assume GPU 0 by default.
#   - report_line(), to replace the hand-written printf blocks that were repeated in
#     restore_server.sh / thin_restore.sh.

# HERE = the experiment root (<root>/scripts/utils/common.sh -> up two).
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
source "$HERE/config.sh"

PYBIN="$VENV/bin/python"
SERVEKIT="$VENV/bin/servekit"
# Overridable so caps-exp/run.sh can point at a criu binary carrying a specific file
# capability set, to test what the minimal set actually is.
CRIU="${CRIU:-$(command -v criu || echo /usr/sbin/criu)}"
CCKPT="$HERE/bin/cuda-checkpoint"
# Add the venv's bin folder to PATH so SGLang can find the "ninja" build tool it needs.
# We deliberately do NOT add our own bin folder here: if CRIU could find our
# cuda-checkpoint tool by name, it would try to run it automatically during the snapshot
# step, conflicting with us running it ourselves first. Every place that needs
# cuda-checkpoint uses its full path ($CCKPT) instead of relying on PATH.
export PATH="$VENV/bin:$PATH"
export XDG_RUNTIME_DIR="${XDG_RUNTIME_DIR:-/tmp/criu-xdg}"; mkdir -p "$XDG_RUNTIME_DIR"
export CUDA_VISIBLE_DEVICES="$GPU_IDS"

# Make sure the CUDA runtime library bundled inside our venv is found before any other
# copy on the system. We also have a separate, system-wide CUDA toolkit installed (needed
# for its compiler, since the OS's own compiler is too old to build SGLang's kernels --
# see NOTES.md), and it registers its own copy of the same library in the system's
# default search path. If a kernel compiled at runtime picks up that other copy instead
# of the venv's, the process ends up running two different CUDA runtimes at once, which
# don't understand each other's handles -- causing a confusing "invalid resource handle"
# crash the first time the server launches.
_NV_RT_LIB="$("$PYBIN" -c 'import os,nvidia; print(os.path.join(os.path.dirname(nvidia.__file__), "cuda_runtime", "lib"))' 2>/dev/null)"
[ -n "$_NV_RT_LIB" ] && export LD_LIBRARY_PATH="$_NV_RT_LIB${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

# If CUDA_HOME isn't set, torch tries to guess it from where the "nvcc" compiler lives.
# Our nvcc is a small wrapper script kept outside the real toolkit folder, so that guess
# lands on the wrong directory and every runtime-compiled kernel then fails to link with
# "cannot find -lcudart". Setting CUDA_HOME explicitly avoids the bad guess.
export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"

# The CRIU flags that make it possible to snapshot the whole SGLang process tree. Each
# one fixes a real error we hit without it; see NOTES.md for the details.
#   --tcp-close          close the network connection torch's internal coordination
#                        service holds open (the "proper" fix needs a permission we
#                        don't have on this box)
#   --network-lock skip  skip a firewall lock that also needs a permission we don't have
# No --link-remap: that flag would keep around files deleted before the snapshot (like
# the /dev/shm files below) by hard-linking them live, but that's exactly what makes a
# snapshot single-use. Paired with drop_tree_shm() below (which embeds a copy of each
# tree-held /dev/shm file directly into the snapshot instead), omitting it lets one
# snapshot be restored more than once. Ported from an earlier version of this experiment
# (commit 50db0ba); see NOTES.md for the two bugs this closes.
#
# CRIU_TCP_ARGS is overridable: the tp2-exp keep-tcp test sets
# "--tcp-established --network-lock skip" together with CRIU=<a criu binary carrying
# cap_net_admin> so the inter-rank TCPStore connection is snapshotted via TCP_REPAIR and
# restored intact, instead of --tcp-close force-closing it (which kills the TP=2 restore
# -- see results/tp2-exp/results.md).
CRIU_TCP_ARGS="${CRIU_TCP_ARGS:---tcp-close --network-lock skip}"
REUSABLE_DUMP_ARGS="--shell-job $CRIU_TCP_ARGS --file-locks --ext-unix-sk"
# --ghost-limit: criu's default (1M) is too small for NCCL's /dev/shm ring-buffer file
# under TP>1 (tens of MB) -- overridable per-caller (see tp2-exp/checkpoint.sh) without
# changing the TP=1 default.
[ -n "${GHOST_LIMIT:-}" ] && REUSABLE_DUMP_ARGS="$REUSABLE_DUMP_ARGS --ghost-limit $GHOST_LIMIT"

now_ms(){ date +%s%3N; }

tree_pids(){ local r=$1; echo "$r"; for k in $(pgrep -P "$r" 2>/dev/null); do tree_pids "$k"; done; }

# Finds every process that's directly touching the GPU, not just the one nvidia-smi
# calls the "compute app". Every one of them has to be handed to cuda-checkpoint first,
# or CRIU fails when it hits GPU memory it doesn't know how to handle.
gpu_vma_pids(){ local root=$1 p; for p in $(tree_pids "$root" | sort -un); do
    [ "$(grep -c /dev/nvidia "/proc/$p/maps" 2>/dev/null)" -gt 0 ] && echo "$p"; done; }

# Deletes the shared-memory files the server tree holds open under /dev/shm -- POSIX
# semaphores (SGLang, all TP sizes) and libfabric/PSM segments (only present at TP>1) --
# so CRIU saves a full copy of each inside the snapshot instead of just remembering where
# it lives on disk. This is what makes the snapshot reusable -- otherwise the first
# restore would consume the one shared file on disk (its process unlinks it on exit) and
# every later restore would fail looking for it. Prints how many files were removed.
# Ported from an earlier version of this experiment (commit 50db0ba); widened from
# semaphores-only to also cover /dev/shm/psm_* (see results/tp2-exp/results.md).
#
# Why deleting it works: CRIU only saves a full copy of a file in the snapshot when that
# file has already been deleted from disk (it survives only because a process still has
# it open). SGLang renames its lock file right after creating it, so the process's memory
# listing shows an old, no-longer-valid name -- so we can't reliably find these by name.
# Instead we collect the inode numbers the tree actually references (via both its memory
# maps and its open fds), then unlink any matching /dev/shm entry by inode.
drop_tree_shm(){
  local root=$1 p f fd i t inos n=0
  inos=" $(for p in $(tree_pids "$root" | sort -un); do
             awk '/\/dev\/shm\/(sem\.|psm_)/ {print $5}' "/proc/$p/maps" 2>/dev/null
             for fd in /proc/$p/fd/*; do
               t="$(readlink "$fd" 2>/dev/null)" || continue
               case "$t" in /dev/shm/sem.*|/dev/shm/psm_*) stat -Lc%i "$fd" 2>/dev/null;; esac
             done
           done | sort -u | tr '\n' ' ')"
  for f in /dev/shm/sem.* /dev/shm/psm_*; do
    [ -e "$f" ] || continue
    i="$(stat -c%i "$f" 2>/dev/null)" || continue
    case "$inos" in *" $i "*) rm -f "$f" 2>/dev/null && n=$((n+1));; esac
  done
  echo "$n"
}

# Only kill processes whose actual program name is python/sglang -- never match on the
# full command-line text ("pkill -f sglang.launch_server"), because that same text can
# also appear in this very harness's own command line and would kill the wrong thing.
kill_servers(){ local p c; for p in $(pgrep -f "sglang.launch_server" 2>/dev/null); do
    c=$(cat "/proc/$p/comm" 2>/dev/null); case "$c" in python|sglang::*) kill -9 "$p" 2>/dev/null;; esac; done; }

gpu_used_mib(){ nvidia-smi --query-gpu=memory.used --format=csv,noheader -i "$GPU_IDS"; }

# post_json <path> <json-body> -> prints "<status> <body>" on one line.
post_json(){
  local path=$1 body=$2
  "$PYBIN" - "$path" "$body" <<'PY' 2>&1
import json, sys, urllib.error, urllib.request
path, body = sys.argv[1], sys.argv[2]
url = "http://127.0.0.1:%s%s" % (__import__("os").environ.get("PORT", "8080"), path)
req = urllib.request.Request(url, data=body.encode(), headers={"Content-Type": "application/json"})
try:
    with urllib.request.urlopen(req, timeout=1800) as r:
        status, raw = r.status, r.read().decode()
except urllib.error.HTTPError as e:
    status, raw = e.code, e.read().decode()
except Exception as e:
    status, raw = -1, repr(e)
print(status, " ".join(raw.split())[:400])
PY
}

# Adds up how much real memory (RAM), in kilobytes, every process in the tree is using.
tree_vmrss_kb(){ local root=$1 p t=0 v; for p in $(tree_pids "$root" | sort -un); do
    v=$(awk '/^VmRSS:/{print $2}' "/proc/$p/status" 2>/dev/null); t=$((t + ${v:-0})); done; echo "$t"; }

# Repeatedly checks the server's address until it responds, or gives up after a timeout.
# Prints how long that took, in milliseconds.
# Checks the /model_info page (which just returns basic info) rather than /health,
# because by default /health actually runs a real test generation -- and between a thin
# restore and reloading the model weights, the memory that should hold the weights is
# garbage, so a real generation there crashes the worker instead of failing cleanly.
wait_http_alive(){
  local timeout_s=${1:-120} t0 t1
  t0=$(now_ms)
  "$PYBIN" - "$timeout_s" <<'PY'
import sys, time, urllib.request
deadline = time.time() + float(sys.argv[1])
url = "http://127.0.0.1:%s/model_info" % __import__("os").environ.get("PORT", "8080")
while time.time() < deadline:
    try:
        with urllib.request.urlopen(url, timeout=5) as r:
            if r.status == 200:
                sys.exit(0)
    except Exception:
        time.sleep(0.05)
sys.exit(1)
PY
  local rc=$?; t1=$(now_ms); echo $((t1-t0)); return $rc
}

# Installs a small compatibility patch that makes Python's networking library (uvloop)
# use an older, simpler method (epoll) instead of a newer one (io_uring) that CRIU can't
# snapshot. It's installed as a ".pth" file so it's automatically loaded in every process
# in this venv, including SGLang's background worker process (which normally starts with
# a stripped-down environment). Also turns off a similar feature in torch's own
# networking code, for the same reason. Safe to run more than once.
ensure_shim(){
  local sp pth
  sp="$("$PYBIN" -c 'import site;print(site.getsitepackages()[0])')"
  pth="$sp/zz_uvloop_epoll_shim.pth"
  printf "import os; os.environ.setdefault('USE_LIBUV','0'); import sys; sys.path.insert(0, '%s/shim')\n" "$HERE" > "$pth"
  env -u PYTHONPATH "$PYBIN" -c "import uvloop,sys; assert 'checkpoint-restore-exp/shim' in uvloop.__file__, uvloop.__file__" \
    || { echo "shim install FAILED"; return 1; }
  echo "shim active: $pth"
}

# Starts the server, waits until it's ready, then sends it one test request to warm it
# up. Prints only the server's process ID to normal output (callers capture it with
# `ROOT=$(launch_and_warm)`), so any other messages go to the error stream instead. Needs
# $SERVER_LOG set; $EXTRA_SERVER_ARGS is optional extra flags for the server.
launch_and_warm(){
  kill_servers; sleep 2
  TORCHINDUCTOR_COMPILE_THREADS=1 HF_HUB_OFFLINE=1 GLOO_SOCKET_IFNAME=lo \
  SGLANG_ENABLE_HEALTH_ENDPOINT_GENERATION=0 \
  nohup "$PYBIN" -m sglang.launch_server \
    --model-path "$MODEL_DIR" --served-model-name "$SERVED_NAME" \
    --host 127.0.0.1 --port "$PORT" --tensor-parallel-size "$TP_SIZE" --trust-remote-code \
    --context-length "$CONTEXT_LEN" --mem-fraction-static "$MEM_FRACTION" \
    --max-running-requests "$MAX_RUNNING" \
    $([ "${DISABLE_CUDA_GRAPH:-0}" = 1 ] && echo --disable-cuda-graph) \
    ${EXTRA_SERVER_ARGS:-} \
    > "$SERVER_LOG" 2>&1 &
  local lp=$! i=0
  while [ $i -lt 180 ]; do grep -q "fired up and ready" "$SERVER_LOG" 2>/dev/null && break; sleep 1; i=$((i+1)); done
  grep -q "fired up and ready" "$SERVER_LOG" 2>/dev/null || { echo "server never became ready" >&2; tail -8 "$SERVER_LOG" >&2; return 1; }
  "$PYBIN" -c "import json,urllib.request as u; u.urlopen(u.Request('http://127.0.0.1:$PORT/generate',data=json.dumps({'text':'Hello','sampling_params':{'temperature':0,'max_new_tokens':4}}).encode(),headers={'Content-Type':'application/json'}),timeout=30).read()" 2>/dev/null
  echo "$lp"
}

# report_line "<label>" <ms> -- one formatted timing line, used by restore.sh's report block.
report_line(){ printf "  %-22s: %6d ms\n" "$1" "$2"; }
