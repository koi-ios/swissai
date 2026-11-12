#!/usr/bin/env bash
# Checkpoint a warmed SGLang server (cuda-checkpoint + CRIU) for restore.sh. See NOTES.md.
# Usage: scripts/checkpoint.sh [--thin] [run-dir]   (--thin: release weights+kv_cache first)
set -u
source "$(dirname "$0")/utils/common.sh"

parse_args(){
  THIN=0; [ "${1:-}" = "--thin" ] && { THIN=1; shift; }
  MODE_DIR="$([ $THIN = 1 ] && echo thin_checkpoint || echo naive_checkpoint)"
  RUN="${1:-$HERE/results/$MODE_DIR/run-$(date -u +%Y%m%dT%H%M%SZ)}"
  IMG="$RUN/img"
  # Each run gets its own log file (not shared): it's the exact file the server was
  # writing to when we take the snapshot, and CRIU won't restore onto a log file whose
  # size has changed since the snapshot was taken.
  SERVER_LOG="$RUN/server.log"
  RELEASE_TAGS="${RELEASE_TAGS:-weights,kv_cache}"
}

check_prereqs(){
  ls "$MODEL_DIR"/*.safetensors >/dev/null 2>&1 || { echo "weights missing -- setup_env.sh download"; exit 2; }
  FREE_GB=$(df -BG --output=avail "$HERE" | tail -1 | tr -dc '0-9')
  [ "${FREE_GB:-0}" -ge 15 ] || { echo "only ${FREE_GB} GiB free -- need >=15 GiB for the snapshot"; exit 2; }
}

# Turns on the memory-saver flag and fixes a library path so the server can actually use
# it. Does nothing in naive mode.
configure_thin_env(){
  [ $THIN = 1 ] || return 0
  export EXTRA_SERVER_ARGS="--enable-memory-saver"
  # The memory saver needs a small helper library loaded before the scheduler process
  # even starts Python, and that helper itself needs the CUDA runtime library -- but
  # only the copy bundled inside our venv, not the system one. Because the helper loads
  # before Python's own library-path setup runs, we have to point it at the right place
  # ourselves, or the scheduler crashes on startup.
  NV_LIB="$("$PYBIN" -c 'import os,nvidia; print(os.path.join(os.path.dirname(nvidia.__file__), "cuda_runtime", "lib"))')"
  export LD_LIBRARY_PATH="$NV_LIB${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
}

evict_page_cache(){
  echo "-- evict model from page cache (true cold start for the launch below) --"
  "$PYBIN" "$HERE/scripts/cache_tools.py" evict --verify "$MODEL_DIR" | sed 's/^/   /'
}

# Records the server's process ID and how long launching + warming up took; both get
# written to meta.txt later.
launch_warm_phase(){
  echo "-- launch + warm --"
  tl0=$(now_ms)
  # launch_and_warm is in utils/common.sh; 
  # launches the server, waits for ready, and gives its pid.
  ROOT=$(launch_and_warm) || exit 1
  tl1=$(now_ms)
  echo "  launcher pid=$ROOT | launch->warmed: $((tl1-tl0)) ms"

  if [ $THIN = 1 ] && grep -q "will not save memory because torch_memory_saver is not enabled" "$SERVER_LOG"; then
    echo "  FATAL: torch_memory_saver is a Noop -- --enable-memory-saver did not take effect"; exit 1
  fi
}

# Finds every process that's directly using the GPU -- every later step needs this list.
# Also checks for a networking feature (io_uring) that CRIU can't snapshot; this only
# warns here, but it would make the snapshot step below fail.
discover_gpu_pids(){
  GPIDS="$(gpu_vma_pids "$ROOT" | tr '\n' ' ')"
  echo "  GPU-VMA pids: $GPIDS"

  # io_uring warning. 
  iou=0; for p in $(tree_pids "$ROOT"); do iou=$((iou + $(ls -la /proc/$p/fd 2>/dev/null | grep -c io_uring))); done
  echo "  io_uring fds in tree: $iou $([ $iou -ne 0 ] && echo '(WARNING: criu will fail -- shim not active?)')"
}


# Records how long freeing GPU memory took (0 in naive mode, since nothing is freed).
# Sends an HTTP JSON to the server /release_memory_occupation endpoint. 
# All the other code is just for logging and sanity checks. 
release_memory_phase(){
  TR_MS=0
  [ $THIN = 1 ] || return 0
  S1_GPU="$(gpu_used_mib)"; S1_RSS="$(tree_vmrss_kb "$ROOT")"
  echo "  S1 pre-release : gpu=$S1_GPU | tree VmRSS=$((S1_RSS/1024)) MiB"
  sleep 2   # give the warmup request time to fully finish -- the next call fails unless
            # the server is completely idle
  echo "-- release_memory_occupation tags=[$RELEASE_TAGS] --"
  body="{\"tags\": [$(echo "$RELEASE_TAGS" | sed 's/[^,]*/"&"/g')]}"
  
  # The REQUEST
  tr0=$(now_ms); rel="$(post_json /release_memory_occupation "$body")"; tr1=$(now_ms)
  
  TR_MS=$((tr1-tr0))
  echo "   -> $rel  (${TR_MS} ms)"
  case "$rel" in 200*) ;; *) echo "  FATAL: release failed"; kill_servers; exit 1;; esac
  S2_GPU="$(gpu_used_mib)"; S2_RSS="$(tree_vmrss_kb "$ROOT")"
  echo "  S2 post-release: gpu=$S2_GPU | tree VmRSS=$((S2_RSS/1024)) MiB"
  echo "  freed on GPU: $(( ${S1_GPU%% *} - ${S2_GPU%% *} )) MiB | VmRSS delta: $(( (S2_RSS-S1_RSS)/1024 )) MiB (must be ~0)"
}

drop_shm_phase(){
  echo "-- /dev/shm files (sems + psm) unlinked pre-dump (for N-times-restorable snapshot) --"
  NSHM="$(drop_tree_shm "$ROOT")"
  echo "  unlinked: $NSHM"
}

# Records how long each step took -- moving GPU memory out, writing the snapshot, and
# the total -- plus the snapshot's size on disk. All of this is saved to meta.txt.
checkpoint_dump_phase(){
  echo "-- cuda-checkpoint (GPU evict) + criu dump --"
  t0=$(now_ms)
  for p in $GPIDS; do "$CCKPT" --action lock       --pid "$p" >/dev/null || { echo "lock $p FAILED"; exit 1; }; done
  for p in $GPIDS; do "$CCKPT" --action checkpoint --pid "$p" >/dev/null || { echo "checkpoint $p FAILED"; exit 1; }; done
  t1=$(now_ms)
  echo "  cuda-checkpoint (GPU evict): $((t1-t0)) ms | gpu: $(gpu_used_mib)"

  rm -rf "$IMG"; mkdir -p "$IMG"
  "$CRIU" --unprivileged dump -t "$ROOT" $REUSABLE_DUMP_ARGS -D "$IMG" -o dump.log
  drc=$?; t2=$(now_ms)
  if [ $drc -ne 0 ]; then echo "  criu dump FAILED (rc=$drc):"; grep -iE "Error" "$IMG/dump.log" | tail -12; exit 1; fi
  IMG_SZ="$(du -sh "$IMG" | cut -f1)"
  echo "  criu dump: $((t2-t1)) ms | snapshot: $IMG_SZ | total checkpoint: $((t2-t0)) ms"
  echo "  ghost files in image: $(ls "$IMG" | grep -c '^ghost-file')"
}

# Remembers the log file's exact size right now, and keeps a backup copy of it. CRIU
# recorded this file's path and size when it took the snapshot, and refuses to restore if
# the file's size has since changed -- even from the server's own writes. restore.sh uses
# this size and backup to reset the log file before every restore.
snapshot_log_size(){
  LOG_SZ="$(stat -c%s "$SERVER_LOG")"
  cp "$SERVER_LOG" "$IMG/server.log.orig"
}

write_meta(){
  cat > "$RUN/meta.txt" <<EOF
gpu_pids=$GPIDS
root=$ROOT
port=$PORT
released_tags=$([ $THIN = 1 ] && echo "$RELEASE_TAGS")
model_path=$MODEL_DIR
img_size=$IMG_SZ
log=$SERVER_LOG
log_size=$LOG_SZ
launch_to_warmed_ms=$((tl1-tl0))
release_ms=$TR_MS
gpu_evict_ms=$((t1-t0))
criu_dump_ms=$((t2-t1))
checkpoint_total_ms=$((t2-t0))
EOF
  echo "  meta: $RUN/meta.txt"
  echo "==== done -- restore N times with: scripts/restore.sh --cold $RUN ===="
}

main(){
  parse_args "$@"
  check_prereqs
  mkdir -p "$IMG"
  ensure_shim || exit 1
  configure_thin_env

  echo "==== checkpoint ($([ $THIN = 1 ] && echo thin || echo naive)) :: Apertus-8B server ===="
  echo "  run dir: $RUN"

  evict_page_cache
  launch_warm_phase
  discover_gpu_pids
  release_memory_phase
  drop_shm_phase
  checkpoint_dump_phase
  snapshot_log_size
  write_meta
}

main "$@"
