#!/usr/bin/env python3
"""Faithful mock of `vllm serve` for testing harness/ab.py without a GPU.

- Cache wiring as in vLLM 0.32.0: torch.compile under VLLM_CACHE_ROOT/torch_compile_cache/<hash
  incl. model path>; Triton under that hash's inductor_cache unless TRITON_CACHE_DIR is set;
  DeepGEMM under VLLM_CACHE_ROOT/deep_gemm unless DG_JIT_CACHE_DIR is set; TileLang in
  TILELANG_CACHE_DIR or ~/.tilelang/cache. A missing kernel "compiles" (sleep + ELF file).
- Log lines servekit's vLLM parser keys on (Model loading took ..., torch.compile took ...,
  Graph capturing finished in ..., Application startup complete).
- /health and /v1/completions until SIGTERM.
"""
import hashlib
import http.server
import json
import os
import sys
import threading
import time

COMPILE_S = float(os.environ.get("MOCK_COMPILE_S", "0.15"))


def kernel(root: str, name: str) -> bool:
    d = os.path.join(root, name)
    so = os.path.join(d, "k.so")
    if os.path.exists(so):
        return False
    os.makedirs(d, exist_ok=True)
    time.sleep(COMPILE_S)
    tmp = so + f".tmp{os.getpid()}"
    with open(tmp, "wb") as f:
        f.write(b"\x7fELF" + bytes(256))
    os.replace(tmp, so)
    return True


def main() -> int:
    args = sys.argv[1:]
    assert args[0] == "serve"
    model = args[1]
    port = int(args[args.index("--port") + 1]) if "--port" in args else 8000
    home = os.path.expanduser("~")
    root = os.environ.get("VLLM_CACHE_ROOT") or os.path.join(os.environ.get("XDG_CACHE_HOME", f"{home}/.cache"), "vllm")
    h = hashlib.sha256(model.encode()).hexdigest()[:10]
    tc = os.path.join(root, "torch_compile_cache", "torch_aot_compile", h)
    triton = os.environ.get("TRITON_CACHE_DIR") or os.path.join(tc, "inductor_cache", "triton", "0")
    dg = os.environ.get("DG_JIT_CACHE_DIR") or os.path.join(root, "deep_gemm")
    tl = os.environ.get("TILELANG_CACHE_DIR") or os.path.join(home, ".tilelang", "cache")
    print("INFO Model loading took 15.0 GiB memory and 0.40 seconds", flush=True)
    t0 = time.time()
    built = sum(kernel(os.path.join(tc, "inductor_cache", "fxgraph"), f"g{i}") for i in range(3))
    built += sum(kernel(triton, f"t{i}") for i in range(4))
    print(f"INFO torch.compile took {time.time() - t0:.2f} s in total", flush=True)
    t1 = time.time()
    built += sum(kernel(os.path.join(dg, "cache"), f"dg{i}") for i in range(2))
    built += sum(kernel(os.path.join(tl, "0.1.12", "kernels"), f"mhc{i}") for i in range(3))
    print(f"INFO Graph capturing finished in {time.time() - t1:.2f} secs", flush=True)
    print(f"INFO built {built} kernels", flush=True)

    class H(http.server.BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def do_GET(self):
            self.send_response(200 if self.path == "/health" else 404)
            self.end_headers()

        def do_POST(self):
            n = int(self.headers.get("Content-Length", 0))
            self.rfile.read(n)
            body = json.dumps({"choices": [{"text": " Bern"}]}).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(body)

    srv = http.server.ThreadingHTTPServer(("127.0.0.1", port), H)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    print("INFO Application startup complete.", flush=True)
    try:
        while True:
            time.sleep(0.5)
    except KeyboardInterrupt:
        pass
    return 0


if __name__ == "__main__":
    sys.exit(main())
