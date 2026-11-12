"""CPU stand-in for `python -m sglang.launch_server` (dry runs only).

Accepts the subset of launch_server flags the experiment uses, prints log lines
in the formats bench/logparse.py expects, serves /health and streaming
/generate, and emulates lazy capture: when idle for SGLANG_LAZY_CUDA_GRAPH_MIN_IDLE_S
it "captures" one deferred bucket at a time; requests arriving during a step
wait for its end; buckets not captured yet are slower (eager). All durations
are multiplied by --mock-time-scale so a dry run finishes in seconds.

Nothing here is a measurement; it only exercises the harness code paths.
"""

from __future__ import annotations

import argparse
import bisect
import json
import os
import random
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

PREFILL = (
    list(range(4, 33, 4)) + list(range(48, 257, 16)) + list(range(288, 513, 32))
    + list(range(576, 1025, 64)) + list(range(1280, 4097, 256)) + list(range(4608, 8193, 512))
)
DECODE = [1, 2, 4, 8, 12] + list(range(16, 257, 8))


def log(msg: str) -> None:
    print(f"[mock TP0] {msg}", flush=True)


class State:
    def __init__(self, args):
        self.scale = args.mock_time_scale
        cfg = json.loads(args.cuda_graph_config) if args.cuda_graph_config else {}
        self.disabled = args.disable_cuda_graph
        self.planned = {
            "decode": sorted(cfg.get("decode", {}).get("bs") or DECODE),
            "prefill": sorted(cfg.get("prefill", {}).get("bs") or PREFILL),
        }
        self.captured = {"decode": set(), "prefill": set()}
        self.pending = []  # (priority, phase, size)
        for phase in ("decode", "prefill"):
            p = cfg.get(phase, {})
            if self.disabled:
                continue
            if p.get("initial_bs") is None:
                self.captured[phase] = set(self.planned[phase])
                continue
            self.captured[phase] = set(p["initial_bs"])
            order = list(p.get("deferred_bs") or [])
            prio = list(p.get("deferred_priority") or [len(order) - i for i in range(len(order))])
            rest = [s for s in self.planned[phase] if s not in self.captured[phase] and s not in order]
            order += rest
            prio += [min(prio, default=0) - 1 - i for i in range(len(rest))]
            self.pending += [(pr, phase, s) for pr, s in zip(prio, order)]
        self.pending.sort(key=lambda x: -x[0])
        self.lock = threading.Condition()
        self.inflight = 0
        self.capturing = False
        self.idle_since = time.monotonic()
        self.ready = False
        self.min_idle = float(os.environ.get("SGLANG_LAZY_CUDA_GRAPH_MIN_IDLE_S", "1.0"))
        self.step = 0

    def cost(self, phase: str, size: int) -> float:
        # PREDICTED 8B TP1 H100 profile (lazygraph/costs.py), scaled.
        return (0.13 if phase == "decode" else 0.18 + 8e-5 * size) * self.scale

    def bucket(self, phase: str, n: int):
        b = self.planned[phase]
        i = bisect.bisect_left(b, n)
        return b[i] if i < len(b) else None

    def is_graph(self, phase: str, n: int) -> bool:
        b = self.bucket(phase, n)
        return b is not None and b in self.captured[phase]


def startup(st: State, args) -> None:
    log(f"Load weight begin. model={args.model_path} tp={args.tp}")
    time.sleep(2.0 * st.scale)
    for phase in ("prefill", "decode"):
        if st.disabled:
            continue
        sizes = sorted(st.captured[phase])
        if len(sizes) < len(st.planned[phase]):
            log(f"Lazy CUDA graph capture [{phase}]: {len(sizes)} of "
                f"{len(st.planned[phase])} buckets at startup {sizes}, "
                f"{len(st.planned[phase]) - len(sizes)} deferred.")
        log(f"Capture target {phase} CUDA graph begin. sizes={sizes}")
        t = 0.0
        for s in sorted(sizes, reverse=True):
            c = st.cost(phase, s)
            time.sleep(c)
            t += c
        mem = 0.002 * len(sizes)
        log(f"Capture target {phase} CUDA graph end. elapsed={t / st.scale:.2f} s, "
            f"mem usage={mem:.2f} GB, avail mem=10.00 GB.")


def lazy_loop(st: State) -> None:
    while True:
        time.sleep(0.01)
        with st.lock:
            if not st.ready or not st.pending:
                if st.ready and not st.pending:
                    return
                continue
            idle = st.inflight == 0 and time.monotonic() - st.idle_since >= st.min_idle * st.scale
            if not idle:
                continue
            _, phase, size = st.pending.pop(0)
            st.capturing = True
        c = st.cost(phase, size)
        time.sleep(c)
        with st.lock:
            st.captured[phase].add(size)
            st.capturing = False
            log(f"Lazy CUDA graph capture: {phase} bucket {size} captured in "
                f"{c / st.scale:.3f} s (step {st.step}, {len(st.pending)} buckets pending).")
            st.step += 1
            if not st.pending:
                log("Lazy CUDA graph capture: all planned buckets captured.")
            st.lock.notify_all()


def make_handler(st: State):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *a):  # quiet
            pass

        def do_GET(self):
            code = 200 if (self.path == "/health" and st.ready) else 503
            self.send_response(code)
            self.end_headers()

        def do_POST(self):
            if self.path != "/generate":
                self.send_response(404)
                self.end_headers()
                return
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            ids = body.get("input_ids") or [1]
            sp = body.get("sampling_params", {})
            n_out = int(sp.get("max_new_tokens", 16))
            with st.lock:
                while st.capturing:  # a step in progress delays the request
                    st.lock.wait()
                st.inflight += 1
            try:
                rng = random.Random(hash(tuple(ids)) & 0xFFFFFFFF)
                out = [rng.randrange(1000, 30000) for _ in range(n_out)]
                prefill = 0.0002 * len(ids) + 0.004
                if not st.is_graph("prefill", len(ids)):
                    prefill += 0.008
                time.sleep(prefill * st.scale)
                if body.get("stream"):
                    self.send_response(200)
                    self.send_header("Content-Type", "text/event-stream")
                    self.end_headers()
                    for k in range(1, n_out + 1):
                        with st.lock:
                            bs = st.inflight
                        step = 0.006 + 0.00003 * bs
                        if not st.is_graph("decode", bs):
                            step += 0.006
                        time.sleep(step * st.scale)
                        chunk = {"text": "x", "output_ids": out[:k],
                                 "meta_info": {"completion_tokens": k, "prompt_tokens": len(ids)}}
                        self.wfile.write(f"data: {json.dumps(chunk)}\n\n".encode())
                        self.wfile.flush()
                    self.wfile.write(b"data: [DONE]\n\n")
                else:
                    time.sleep(0.006 * n_out * st.scale)
                    payload = json.dumps({"text": "x", "output_ids": out,
                                          "meta_info": {"completion_tokens": n_out}}).encode()
                    self.send_response(200)
                    self.send_header("Content-Type", "application/json")
                    self.send_header("Content-Length", str(len(payload)))
                    self.end_headers()
                    self.wfile.write(payload)
            finally:
                with st.lock:
                    st.inflight -= 1
                    if st.inflight == 0:
                        st.idle_since = time.monotonic()

    return Handler


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-path", required=True)
    ap.add_argument("--tp", type=int, default=1)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=30000)
    ap.add_argument("--cuda-graph-config", default=None)
    ap.add_argument("--disable-cuda-graph", action="store_true")
    ap.add_argument("--mock-time-scale", type=float, default=0.05)
    args, _unknown = ap.parse_known_args(argv)
    st = State(args)
    startup(st, args)
    server = ThreadingHTTPServer((args.host, args.port), make_handler(st))
    threading.Thread(target=server.serve_forever, daemon=True).start()
    with st.lock:
        st.ready = True
        st.idle_since = time.monotonic()
    log("The server is fired up and ready to roll!")
    lazy = threading.Thread(target=lazy_loop, args=(st,), daemon=True)
    lazy.start()
    try:
        while True:
            time.sleep(3600)
    except KeyboardInterrupt:
        return 0


if __name__ == "__main__":
    sys.exit(main())
