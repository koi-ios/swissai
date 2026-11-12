#!/usr/bin/env python3
"""Start a server, wait for /health, send one completion, stop it, exit 0.

Turns a long-running `vllm serve` / `sglang.launch_server` into a finite warmup command that
`cache-audit run` can trace from cold start through the first request (where the last JIT
kernels compile):

    python -m cache_audit run --tracer strace -- python harness/warmup_once.py -- vllm serve <model> ...
"""

from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import time
import urllib.request

PORT = int(os.environ.get("WARMUP_PORT", "8080"))


def main() -> int:
    argv = sys.argv[1:]
    if argv[:1] == ["--"]:
        argv = argv[1:]
    timeout = float(os.environ.get("WARMUP_TIMEOUT", "2400"))
    proc = subprocess.Popen(argv, start_new_session=True)
    t0 = time.monotonic()
    try:
        while True:
            if proc.poll() is not None:
                print(f"server exited {proc.returncode} before ready", file=sys.stderr)
                return 1
            try:
                with urllib.request.urlopen(f"http://127.0.0.1:{PORT}/health", timeout=5) as r:
                    if r.status == 200:
                        break
            except OSError:
                pass
            if time.monotonic() - t0 > timeout:
                print("not ready before timeout", file=sys.stderr)
                return 1
            time.sleep(2)
        ready = time.monotonic() - t0
        body = json.dumps({"model": "m", "prompt": "Hello", "max_tokens": 8, "temperature": 0}).encode()
        req = urllib.request.Request(f"http://127.0.0.1:{PORT}/v1/completions", data=body,
                                     headers={"Content-Type": "application/json"})
        t1 = time.monotonic()
        with urllib.request.urlopen(req, timeout=600) as r:
            r.read()
        print(json.dumps({"ready_s": round(ready, 2), "first_request_s": round(time.monotonic() - t1, 2)}))
        return 0
    finally:
        try:
            os.killpg(proc.pid, signal.SIGTERM)
            proc.wait(timeout=60)
        except (ProcessLookupError, subprocess.TimeoutExpired):
            try:
                os.killpg(proc.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass


if __name__ == "__main__":
    sys.exit(main())
