#!/usr/bin/env bash
# Restore a checkpointed SGLang server from a CRIU snapshot, and verify it serves.
# Pairs with checkpoint.sh (naive/thin auto-detected from its meta.txt). See NOTES.md.
# Usage: scripts/restore.sh [--cold] <run_dir>   (--cold: honest cold-node timing)
set -u
source "$(dirname "$0")/utils/common.sh"

parse_args(){
  COLD=0; [ "${1:-}" = "--cold" ] && { COLD=1; shift; }
  RUN="${1:-}"
  [ -n "$RUN" ] && [ -d "$RUN" ] || { echo "usage: $0 [--cold] <run_dir>  (from checkpoint.sh)"; exit 2; }
  IMG="$RUN/img"
  META="$RUN/meta.txt"
  [ -f "$META" ] || { echo "no meta.txt in $RUN"; exit 2; }
}

# Reads back everything checkpoint.sh recorded about this snapshot -- process IDs, which
# memory was released, the model path, log file info -- so every later step knows what
# it's working with.
read_meta(){
  meta(){ grep -E "^$1=" "$META" | cut -d= -f2-; }
  GPIDS="$(meta gpu_pids)"; TAGS="$(meta released_tags)"; RELOAD_MODEL="$(meta model_path)"
  LOGF="$(meta log)"; LOGSZ="$(meta log_size)"
  [ -n "$GPIDS" ] || { echo "meta.txt has no gpu_pids"; exit 2; }
  THIN=0; [ -n "$TAGS" ] && THIN=1
  LOAD_FORMAT="${LOAD_FORMAT:-safetensors}"
}

evict_page_cache(){
  [ "$COLD" = 1 ] || return 0
  echo "-- evict snapshot$([ $THIN = 1 ] && echo " + model") from page cache (honest cold restore) --"
  if [ $THIN = 1 ]; then
    "$PYBIN" "$HERE/scripts/cache_tools.py" evict --verify "$IMG" "$RELOAD_MODEL" | sed 's/^/   /'
  else
    "$PYBIN" "$HERE/scripts/cache_tools.py" evict --verify "$IMG" | sed 's/^/   /'
  fi
}

# CRIU restores a process onto its exact original process ID, so if an earlier restore of
# this same snapshot (or any other server) is still running, the restore fails with a
# "file exists" error. Killing it isn't enough by itself -- a killed process can briefly
# linger as a zombie, still holding its ID, until the system cleans it up, and CRIU can
# lose that race. So we wait here until the exact process IDs we need are actually free,
# instead of guessing with a fixed delay.
clear_stale_pids(){
  kill_servers
  i=0
  while [ $i -lt 100 ]; do
    busy=0
    for p in $GPIDS; do [ -e "/proc/$p" ] && busy=1; done
    [ $busy = 0 ] && break
    sleep 0.1; i=$((i+1))
  done
  [ $busy = 0 ] || echo "   WARNING: pid(s) among $GPIDS still present after 10s wait -- restore may collide"
}

# Reset the log file to the exact size it was when the snapshot was taken -- required
# before every restore, and cheap/safe to do every time. Since the server only ever
# appends to this file, its first log_size bytes are always still the true original
# content, so trimming it back to that size loses nothing. If the file was deleted
# instead of just appended to, restore the backup copy first -- otherwise trimming a
# missing file would just create an empty one of the right size, which would pass CRIU's
# size check but leave a garbage log.
reset_log_size(){
  [ -n "$LOGF" ] && [ -n "$LOGSZ" ] || return 0
  [ -f "$LOGF" ] || cp "$IMG/server.log.orig" "$LOGF" 2>/dev/null
  truncate -s "$LOGSZ" "$LOGF" 2>/dev/null && echo "-- fd-1 log reset to dump-time size (${LOGSZ} B) --"
}

# Records how long the CRIU restore step itself took.
# RESTORE_TCP_ARGS is overridable to match how the snapshot was dumped: default
# --tcp-close (connections were force-closed at dump), or "--tcp-established" for the
# tp2-exp keep-tcp path (connections snapshotted via TCP_REPAIR, restored intact).
criu_restore_phase(){
  echo "-- criu restore --"
  t0=$(now_ms)
  "$CRIU" --unprivileged restore -D "$IMG" --shell-job ${RESTORE_TCP_ARGS:---tcp-close} -d -o restore.log
  rrc=$?; t1=$(now_ms)
  if [ $rrc -ne 0 ]; then
    echo "RESTORE FAILED (criu rc=$rrc):"; grep -iE "Error" "$IMG/restore.log" | tail -10; exit 1
  fi
  echo "   criu restore: $((t1-t0)) ms"
}

# Records how long reattaching the process to the GPU took.
gpu_reattach_phase(){
  echo "-- cuda-checkpoint restore + unlock (GPU reattach) --"
  for p in $GPIDS; do "$CCKPT" --action restore --pid "$p" >/dev/null 2>&1 || echo "   [$p] gpu-restore FAILED"; done
  for p in $GPIDS; do "$CCKPT" --action unlock  --pid "$p" >/dev/null 2>&1 || echo "   [$p] unlock FAILED"; done
  t2=$(now_ms)
  echo "   cuda-checkpoint restore: $((t2-t1)) ms | gpu: $(gpu_used_mib)"
}

# Records how long each of these took: waiting for the server to respond, resuming the
# freed GPU memory, and reloading the model weights. Naive mode does none of this, so
# these timestamps just get copied from t2.
reload_weights_phase(){
  if [ $THIN = 1 ]; then
    HTTP_WAIT_MS="$(wait_http_alive 120)" || { echo "   server never answered /model_info after restore"; exit 1; }
    t3=$(now_ms)
    echo "   http alive: ${HTTP_WAIT_MS} ms"

    echo "-- resume_memory_occupation tags=[$TAGS] --"
    body="{\"tags\": [$(echo "$TAGS" | sed 's/[^,]*/"&"/g')]}"
    res="$(post_json /resume_memory_occupation "$body")"; t4=$(now_ms)
    echo "   -> $res  ($((t4-t3)) ms) | gpu: $(gpu_used_mib)"
    case "$res" in 200*) ;; *) echo "   FATAL: resume failed"; exit 1;; esac

    echo "-- update_weights_from_disk ($LOAD_FORMAT) --"
    upd="$(post_json /update_weights_from_disk "{\"model_path\": \"$RELOAD_MODEL\", \"load_format\": \"$LOAD_FORMAT\"}")"
    t5=$(now_ms)
    echo "   -> $upd  ($((t5-t4)) ms) | gpu: $(gpu_used_mib)"
    case "$upd" in 200*) ;; *) echo "   FATAL: weight reload failed"; exit 1;; esac
  else
    t3=$t2; t4=$t2; t5=$t2
  fi
}

# Runs a benchmark against the restored server and records the results: output file,
# pass/fail, time until it was ready, and throughput.
verify_bench_phase(){
  echo "-- verify: servekit bench on the restored server --"
  BENCH_OUT="$RUN/restore-$(date -u +%Y%m%dT%H%M%SZ)-bench.json"
  tb0=$(now_ms)
  "$SERVEKIT" bench --url "http://127.0.0.1:$PORT" --out "$BENCH_OUT" \
    --wait-ready 120 --requests 64 --concurrency 16 --input-len 512 --output-len 128
  vrc=$?

  read -r READY_MS TPUT <<<"$("$PYBIN" -c "
import json
try:
    r = json.load(open('$BENCH_OUT'))
    print(int(r['ready_wait_s'] * 1000), (r.get('throughput') or {}).get('output_tok_per_s', -1))
except Exception:
    print(-1, -1)
" 2>/dev/null)"
  if [ "${READY_MS:--1}" -ge 0 ]; then t6=$((tb0 + READY_MS)); else t6=$(now_ms); fi
}

print_report(){
  echo
  echo "==== RESULT ($([ $THIN = 1 ] && echo thin || echo naive)) ===="
  report_line "criu restore" $((t1-t0))
  report_line "cuda reattach" $((t2-t1))
  if [ $THIN = 1 ]; then
    report_line "http alive" $((t3-t2))
    report_line "resume_memory_occup." $((t4-t3))
    report_line "update_weights_disk" $((t5-t4))
  fi
  report_line "first served request" $((t6-t5))
  printf "  ---------------------------------\n"
  report_line "restore-to-served" $((t6-t0))
  if [ -n "$BASELINE_COLD_S" ]; then
    report_line "cold-launch baseline" "$(awk -v s="$BASELINE_COLD_S" 'BEGIN{printf "%d", s*1000}')"
    awk -v r=$((t6-t0)) -v b="$BASELINE_COLD_S" 'BEGIN{printf "  vs cold launch       : %.2fx\n", (b*1000)/r}'
  else
    echo "  cold-launch baseline : (BASELINE_COLD_S not set in config.sh -- run baseline.sh --cold first)"
  fi
  if [ $THIN = 1 ] && [ -n "$NAIVE_RESTORE_S" ]; then
    report_line "naive ckpt reference" "$(awk -v s="$NAIVE_RESTORE_S" 'BEGIN{printf "%d", s*1000}')"
    awk -v r=$((t6-t0)) -v n="$NAIVE_RESTORE_S" 'BEGIN{printf "  vs naive checkpoint  : %.2fx\n", (n*1000)/r}'
  fi
  printf "  snapshot size        : %6s\n" "$(du -sh "$IMG"|cut -f1)"
  printf "  post-restore tput    : %6s tok/s\n" "${TPUT:--1}"
  echo "  bench report         : $BENCH_OUT"
  [ $vrc -eq 0 ] && echo "  GATE: PASS (restored server serves at full rate, correct output)" || echo "  GATE: FAIL (see bench report)"
  echo "==== end ===="
}

append_meta(){
  cat >> "$META" <<EOF
criu_restore_ms=$((t1-t0))
cuda_reattach_ms=$((t2-t1))
http_alive_ms=$([ $THIN = 1 ] && echo $((t3-t2)) || echo 0)
resume_ms=$([ $THIN = 1 ] && echo $((t4-t3)) || echo 0)
reload_ms=$([ $THIN = 1 ] && echo $((t5-t4)) || echo 0)
first_request_ms=$((t6-t5))
restore_to_served_ms=$((t6-t0))
throughput_tok_s=${TPUT:--1}
cold_mode=$COLD
EOF
}

main(){
  parse_args "$@"
  read_meta

  echo "==== restore ($([ $THIN = 1 ] && echo thin || echo naive)) :: SGLang server from CRIU snapshot ===="
  echo "snapshot: $IMG ($(du -sh "$IMG"|cut -f1)) | gpu pids: $GPIDS | mode: $([ $COLD = 1 ] && echo COLD || echo warm)"

  evict_page_cache
  clear_stale_pids
  reset_log_size
  criu_restore_phase
  gpu_reattach_phase
  reload_weights_phase
  verify_bench_phase
  print_report
  append_meta
  exit $vrc
}

main "$@"
