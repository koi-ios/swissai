#!/usr/bin/env python3
"""A stand-in for the `vllm` CLI and for SGLang's server, for testing the harness
on a machine with no GPU.

`mock_vllm.py serve MODEL ...` prints startup lines in vLLM 0.33's formats;
`mock_vllm.py sglang --model-path MODEL ...` prints SGLang 0.5.x's (per TP rank,
with the real prefixes, the self-issued warmup request and the ready line). Both
serve /health, /v1/models and /v1/completions (echo + logprobs, as `servekit
verify` uses), and exit on SIGTERM. Timings are fake and short. A second launch
with the same cache directories reports cache hits, as the real engines do.

    MOCK_SCALE=0.2      seconds per fake phase unit
    MOCK_PERTURB=1      shift logprobs by 1e-3 (verify must fail)
    MOCK_ORPHAN=1       leave a child that ignores SIGTERM (the harness must kill the group)
    MOCK_FAIL=1         die during weight loading with a traceback
    MOCK_ECHO_ENV=A,B   print "mock env A=..." for these variables (tests of what the engine sees)
"""
from __future__ import annotations

import glob
import hashlib
import json
import os
import signal
import subprocess
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

VERSION = "0.33.0"
SCALE = float(os.environ.get("MOCK_SCALE", "0.2"))
HELP = """usage: vllm serve [model_tag] [options]
  --load-format LOAD_FORMAT
      "auto", "pt", "safetensors", "instanttensor", "ipc_cache", "npcache", "dummy", "tensorizer",
      "runai_streamer", "runai_streamer_sharded", "sharded_state", "mistral", "modelexpress"
  --safetensors-load-strategy {lazy,eager,prefetch,torchao}
  --safetensors-prefetch-num-threads SAFETENSORS_PREFETCH_NUM_THREADS
  --tensor-parallel-size TENSOR_PARALLEL_SIZE, -tp TENSOR_PARALLEL_SIZE
"""


def say(src: str, msg: str, level: str = "INFO") -> None:
    stamp = time.strftime("%m-%d %H:%M:%S")
    print(f"({src} pid={os.getpid()}) {level} {stamp} [mock.py:1] {msg}", flush=True)


def opt(args, name, default=None):
    if name in args:
        return args[args.index(name) + 1]
    for a in args:
        if a.startswith(name + "="):
            return a.split("=", 1)[1]
    return default


def lp(token: str) -> float:
    v = -int(hashlib.sha256(token.encode()).hexdigest()[:6], 16) / 0xFFFFFF * 5.0
    return round(v + (1e-3 if os.environ.get("MOCK_PERTURB") == "1" else 0.0), 6)


class Handler(BaseHTTPRequestHandler):
    served = "model"

    def log_message(self, *a):  # keep the log clean
        pass

    def _send(self, code, obj):
        body = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path == "/health":
            self._send(200, {})
        elif self.path == "/v1/models":
            self._send(200, {"data": [{"id": Handler.served}]})
        else:
            self._send(404, {"error": "not found"})

    def do_POST(self):
        if self.path != "/v1/completions":
            return self._send(404, {"error": "not found"})
        req = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))))
        prompt_toks = req["prompt"].split()
        n = int(req.get("max_tokens", 16))
        seed = int(hashlib.sha256(req["prompt"].encode()).hexdigest()[:4], 16)  # hash() differs per process
        gen = [f"g{seed % 97}_{i}" for i in range(n)]
        if req.get("echo"):
            toks = prompt_toks + gen
            lps = [None] + [lp(t) for t in prompt_toks[1:]] + [lp(t) for t in gen]
        else:
            toks, lps = gen, [lp(t) for t in gen]
        self._send(200, {"choices": [{"text": " ".join(gen), "logprobs": {"tokens": toks, "token_logprobs": lps}}]})


def serve(argv):
    model = argv[0]
    tp = int(opt(argv, "--tensor-parallel-size", opt(argv, "-tp", "1")))
    host, port = opt(argv, "--host", "127.0.0.1"), int(opt(argv, "--port", "8000"))
    Handler.served = opt(argv, "--served-model-name", model)
    fmt = opt(argv, "--load-format", "auto")
    srcs = [f"Worker_TP{r}" for r in range(tp)] if tp > 1 else ["EngineCore"]
    children = [subprocess.Popen([sys.executable, "-c", "import time; time.sleep(3600)"]) for _ in range(tp)]
    if os.environ.get("MOCK_ORPHAN") == "1":
        children.append(subprocess.Popen(
            [sys.executable, "-c", "import signal,time; signal.signal(signal.SIGTERM, signal.SIG_IGN); time.sleep(3600)"]))
    stop = threading.Event()
    signal.signal(signal.SIGTERM, lambda *a: stop.set())
    signal.signal(signal.SIGINT, lambda *a: stop.set())

    say("APIServer", f" mock banner  version {VERSION}")
    say("APIServer", f"non-default args: {{'model': '{model}', 'load_format': '{fmt}', 'tensor_parallel_size': {tp}}}")
    echo_env(lambda m: say("APIServer", m))
    time.sleep(SCALE)
    say("EngineCore", f"Initializing a V1 LLM engine (v{VERSION}) with config: model='{model}'")
    time.sleep(SCALE)
    if os.environ.get("MOCK_FAIL") == "1":
        say("EngineCore", "Traceback (most recent call last):", "ERROR")
        say("EngineCore", "RuntimeError: mock failure while loading weights", "ERROR")
        sys.exit(1)
    if fmt == "sharded_state":
        load_sharded(model, tp, srcs)
    else:
        for r, s in enumerate(srcs):
            say(s, "Filesystem type for checkpoints: NFS. Checkpoint size: 14.96 GiB. Available RAM: 880.00 GiB.")
    time.sleep(SCALE * 2)
    # vLLM 0.33 logs these with info_once(scope="local"): local rank 0 only
    say(srcs[0], "Loading weights took 1.20 seconds")
    say(srcs[0], "Model loading took 7.48 GiB memory and 1.400000 seconds")
    cache = os.environ.get("VLLM_CACHE_ROOT", "/tmp/mock-vllm-cache")
    marker = os.path.join(cache, "torch_compile_cache", "mock-hash", "done")
    hit = os.path.exists(marker)
    time.sleep(SCALE * (1 if hit else 3))
    for r, s in enumerate(srcs):
        say(s, f"Using cache directory: {os.path.dirname(marker)}/rank_{r}_0/backbone for vLLM's torch.compile")
        if hit:
            say(s, "Directly load the compiled graph(s) for compile range (1, 2048) from the cache, took 0.412 s")
            say(s, "torch.compile took 0.95 s in total")
        else:
            say(s, "Dynamo bytecode transform time: 2.10 s")
            say(s, f"Compiling a graph for compile range (1, 2048) takes {3.0 + 0.2 * r:.2f} s")
            say(s, f"torch.compile took {5.6 + 0.2 * r:.2f} s in total")
    if not hit:
        os.makedirs(os.path.dirname(marker), exist_ok=True)
        open(marker, "w").close()
    for s in srcs:
        say(s, "Initial profiling/warmup run took 0.48 s")
    say("EngineCore", "GPU KV cache size: 401,232 tokens")
    time.sleep(SCALE)
    say(srcs[0], "Graph capturing finished in 2 secs, took 0.31 GiB")
    say("EngineCore", "init engine (profile, create kv cache, warmup model) took 6.10 s (compilation: 5.60 s)")
    httpd = ThreadingHTTPServer((host, port), Handler)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    say("APIServer", f"Starting vLLM server on http://{host}:{port}")
    print(f"(APIServer pid={os.getpid()}) INFO:     Application startup complete.", flush=True)
    while not stop.wait(0.2):
        pass
    say("APIServer", "Shutting down")
    httpd.shutdown()
    for c in children:
        c.terminate()
    return 0


def load_sharded(model, tp, srcs):
    """What vLLM's ShardedStateLoader needs: this rank's files, or it raises.

    Launched by `servekit launch` (SERVEKIT_VLLM_LOADER=1), each rank first waits
    on servekit's stage barrier and afterwards records its load, as servekit's
    vLLM plugin does inside the real loader."""
    plugin = os.environ.get("SERVEKIT_VLLM_LOADER") == "1"
    for r, s in enumerate(srcs):
        waited = 0.0
        if plugin:
            from servekit import _shim
            t0 = time.monotonic()
            _shim.wait_for_stage(model, timeout_s=120, poll_s=0.05)
            waited = time.monotonic() - t0
        pattern = os.path.join(model, f"model-rank-{r}-part-*.safetensors")
        if not glob.glob(pattern):
            say(s, f"ValueError: Could not find checkpoint files '{pattern}', "
                   "only pre-sharded checkpoints are currently supported!", "ERROR")
            sys.exit(1)
        if plugin:
            from servekit import _vllm_plugin
            _vllm_plugin.record_loaded(model, _vllm_plugin.rank_key(0, r),
                                       {"load_s": round(1.2 + 0.1 * r, 3), "barrier_wait_s": round(waited, 3)})


def echo_env(say_fn) -> None:
    for name in filter(None, os.environ.get("MOCK_ECHO_ENV", "").split(",")):
        say_fn(f"mock env {name}={os.environ.get(name, '<unset>')}")


def sglang_serve(argv):
    """SGLang's launch_server, as far as the harness and `servekit profile` can tell."""
    model = opt(argv, "--model-path", opt(argv, "--model", "model"))
    tp = int(opt(argv, "--tp-size", opt(argv, "--tp", "1")))
    host, port = opt(argv, "--host", "127.0.0.1"), int(opt(argv, "--port", "30000"))
    Handler.served = opt(argv, "--served-model-name", model)
    max_bs = int(opt(argv, "--cuda-graph-max-bs", "256"))
    children = [subprocess.Popen([sys.executable, "-c", "import time; time.sleep(3600)"]) for _ in range(tp)]
    stop = threading.Event()
    signal.signal(signal.SIGTERM, lambda *a: stop.set())
    signal.signal(signal.SIGINT, lambda *a: stop.set())

    def main_say(msg):
        print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}", flush=True)

    def rank_say(r, msg):
        print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')} TP{r}] {msg}", flush=True)

    main_say(f"server_args=ServerArgs(model_path='{model}', tp_size={tp}, port={port}, cuda_graph_max_bs={max_bs})")
    echo_env(main_say)
    main_say("Using default HuggingFace chat template with detected content format: string")
    time.sleep(SCALE)
    # SGLang's custom all-reduce JIT-compiles during distributed init: a kept cache skips it
    jit = os.environ.get("SGLANG_JIT_CACHE_DIR") or os.path.expanduser("~/.cache/sglang/jit")
    marker = os.path.join(jit, "mock_all_reduce", "kernel.so")
    hit = os.path.exists(marker)
    for r in range(tp):
        rank_say(r, "Init torch distributed begin.")
    time.sleep(SCALE * (1 if hit else 3))
    for r in range(tp):
        rank_say(r, f"Init torch distributed ends. elapsed={(0.4 if hit else 3.1) + 0.05 * r:.2f} s, mem usage=0.10 GB")
    if not hit:
        os.makedirs(os.path.dirname(marker), exist_ok=True)
        open(marker, "w").close()
    if os.environ.get("MOCK_FAIL") == "1":
        rank_say(0, "Traceback (most recent call last):")
        rank_say(0, "RuntimeError: mock failure while loading weights")
        sys.exit(1)
    for r in range(tp):
        rank_say(r, "Load weight begin. avail mem=78.50 GB")
    time.sleep(SCALE * 2)
    for r in range(tp):
        rank_say(r, f"Load weight end. elapsed={1.3 + 0.1 * r:.2f} s, type=ApertusForCausalLM, dtype=torch.bfloat16, "
                    "avail mem=71.00 GB, mem usage=7.50 GB.")
    for r in range(tp):
        rank_say(r, "KV Cache is allocated. #tokens: 400000, K size: 12.00 GB, V size: 12.00 GB")
        rank_say(r, "Memory pool end. avail mem=46.00 GB")
    sizes = [1, 2, 4, 8] + list(range(16, max_bs + 1, 8))
    for r in range(tp):
        rank_say(r, f"Capture cuda graph begin. This can take up to several minutes. avail mem=46.00 GB")
        rank_say(r, f"Capture cuda graph bs {sizes}")
    time.sleep(SCALE * (1 + max_bs / 128))
    for r in range(tp):
        rank_say(r, f"Capture cuda graph end. Time elapsed: {0.02 * max_bs + 0.1 * r:.2f} s. mem usage=0.40 GB. "
                    "avail mem=45.60 GB.")
    httpd = ThreadingHTTPServer((host, port), Handler)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    main_say(f"INFO:     Uvicorn running on http://{host}:{port} (Press CTRL+C to quit)")
    time.sleep(SCALE)
    main_say(f'INFO:     127.0.0.1:41234 - "POST /generate HTTP/1.1" 200 OK')
    main_say("The server is fired up and ready to roll!")
    while not stop.wait(0.2):
        pass
    main_say("Shutting down")
    httpd.shutdown()
    for c in children:
        c.terminate()
    return 0


def bench_startup(argv):
    out = opt(argv, "--output-json")
    cold, warm = int(opt(argv, "--num-iters-cold", 3)), int(opt(argv, "--num-iters-warm", 3))
    res = {"avg_cold_startup_time": 30.0, "cold_startup_times": [29.0 + i for i in range(cold)],
           "cold_compilation_times": [9.0 + i for i in range(cold)],
           "avg_warm_startup_time": 12.0, "warm_startup_times": [11.5 + i for i in range(warm)],
           "warm_compilation_times": [1.0] * warm}
    print("Measuring cold startup time...")
    if out:
        with open(out, "w") as f:
            json.dump(res, f)
    return 0


def main():
    argv = sys.argv[1:]
    if argv[:1] == ["--version"]:
        print(VERSION)
        return 0
    if argv[:1] == ["serve"] and any(a.startswith("--help") for a in argv):
        print(HELP)
        return 0
    if argv[:1] == ["serve"]:
        return serve(argv[1:])
    if argv[:1] == ["sglang"]:
        return sglang_serve(argv[1:])
    if argv[:2] == ["bench", "startup"]:
        return bench_startup(argv[2:])
    print(f"mock vllm: unsupported {argv}", file=sys.stderr)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
