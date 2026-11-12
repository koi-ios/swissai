"""An OpenAI-compatible server over the CPU engine, for testing `servekit verify` and the study driver end to end.

It answers what verify asks of SGLang or vLLM -- GET /v1/models, POST
/v1/completions with echo, logprobs (top-k) and greedy max_tokens -- and SGLang's
POST /weights_checker {"action": "checksum"}. Like a real engine, its numerics
depend on load: a request's split-K, split-KV and chunked-prefill choices are
picked from how many requests are in flight when it starts, so concurrency and
filler traffic perturb logprobs the way they do on a GPU, while one request at
a time reproduces bitwise.

`--autotune-seed S` additionally fixes, per process, the kernel choices of the
idle schedule at random, the way an autotuner can pick different kernels on
each start: restarts then stop reproducing bitwise.

    python -I cpu/server.py --model-path models/Qwen3-0.6B --port 8080
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("HF_HUB_OFFLINE", "1")

import torch  # noqa: E402

from cpu.engine import Engine, Numerics, load_checkpoint  # noqa: E402


class State:
    def __init__(self, model_path: str, tokenizer_path: str, served_name: str, autotune_seed, threads: int):
        from transformers import AutoTokenizer

        torch.set_num_threads(threads)
        self.ckpt = load_checkpoint(Path(model_path))
        self.engine = Engine(self.ckpt)
        self.tok = AutoTokenizer.from_pretrained(tokenizer_path)
        self.name = served_name
        self.lock = threading.Lock()
        self.in_flight = 0
        if autotune_seed is None:
            self.idle = Numerics()
        else:
            rng = random.Random(autotune_seed)
            self.idle = Numerics(split_k=rng.choice([1, 2, 4]), attn_splits=rng.choice([1, 2]))

    def numerics(self, load: int) -> Numerics:
        """What a loaded engine picks: kernels keyed by batch size, chunked prefill under contention."""
        if load <= 1:
            return self.idle
        return Numerics(split_k={2: 2, 3: 4, 4: 4}.get(load, 1), attn_splits=2 if load >= 3 else 1,
                        chunk=128 if load >= 2 else None)

    def checksum(self) -> dict:
        sums = {}
        for name in sorted(self.ckpt.weights):
            t = self.ckpt.weights[name].contiguous()
            sums[name] = hashlib.sha256(t.numpy().tobytes()).hexdigest()[:16]
        h = hashlib.sha256()
        for name in sorted(sums):
            h.update(name.encode())
            h.update(sums[name].encode())
        rank = {"checksums": sums, "per_gpu_checksum": h.hexdigest(),
                "parallelism_info": {"tp_rank": 0, "tp_size": 1, "dp_rank": 0, "dp_size": 1, "pp_rank": 0,
                                     "pp_size": 1, "rank": 0, "size": 1}}
        return {"success": True, "message": "", "ranks": [rank],
                "per_engine_checksum": hashlib.sha256(rank["per_gpu_checksum"].encode()).hexdigest()}

    def complete(self, body: dict) -> dict:
        prompt = body.get("prompt", "")
        max_tokens = int(body.get("max_tokens", 16))
        k = body.get("logprobs")
        echo = bool(body.get("echo"))
        with self.lock:
            self.in_flight += 1
            load = self.in_flight
        try:
            num = self.numerics(load)
            ids = self.tok(prompt).input_ids
            dec = lambda i: self.tok.decode([i])  # noqa: E731
            tokens, lps, tops = [], [], []
            if echo:
                logits, _ = self.engine.prefill(ids, num)
                lp = torch.log_softmax(logits.float(), -1)
                tokens.append(dec(ids[0]))
                lps.append(None)
                tops.append(None)
                for i in range(1, len(ids)):
                    tokens.append(dec(ids[i]))
                    lps.append(float(lp[i - 1, ids[i]]))
                    if k:
                        v, ix = lp[i - 1].topk(int(k))
                        tops.append({dec(int(a)): float(b) for a, b in zip(ix, v)})
                    else:
                        tops.append(None)
            gen, gtops = self.engine.greedy(ids, max_tokens, num, top_k=max(1, int(k or 1)))
            for t, (ix, v) in zip(gen, gtops):
                tokens.append(dec(t))
                lps.append(float(v[0]))
                tops.append({dec(int(a)): float(b) for a, b in zip(ix, v)} if k else None)
            text = self.tok.decode(gen)
            choice = {"index": 0, "text": (prompt + text) if echo else text, "finish_reason": "length"}
            if k is not None:
                choice["logprobs"] = {"tokens": tokens, "token_logprobs": lps, "top_logprobs": tops}
            return {"object": "text_completion", "model": self.name, "choices": [choice],
                    "usage": {"prompt_tokens": len(ids), "completion_tokens": len(gen)}}
        finally:
            with self.lock:
                self.in_flight -= 1


def make_handler(state: State):
    class Handler(BaseHTTPRequestHandler):
        def _reply(self, code, payload):
            data = json.dumps(payload).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self):
            if self.path == "/v1/models":
                self._reply(200, {"data": [{"id": state.name}]})
            elif self.path in ("/health", "/health_generate"):
                self._reply(200, {})
            else:
                self._reply(404, {"error": "not found"})

        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))) or b"{}")
            try:
                if self.path == "/v1/completions":
                    self._reply(200, state.complete(body))
                elif self.path == "/weights_checker" and body.get("action", "checksum") == "checksum":
                    self._reply(200, state.checksum())
                else:
                    self._reply(404, {"error": "not found"})
            except Exception as e:  # noqa: BLE001
                self._reply(500, {"error": repr(e)})

        def log_message(self, *a):
            pass

    return Handler


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-path", required=True)
    ap.add_argument("--tokenizer", default=str(ROOT / "models" / "Qwen3-0.6B"))
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8080)
    ap.add_argument("--served-model-name", default="m")
    ap.add_argument("--autotune-seed", type=int, default=None)
    ap.add_argument("--threads", type=int, default=4)
    args = ap.parse_args(argv)
    state = State(args.model_path, args.tokenizer, args.served_model_name, args.autotune_seed, args.threads)
    srv = ThreadingHTTPServer((args.host, args.port), make_handler(state))
    print(f"ready on http://{args.host}:{args.port}", flush=True)
    srv.serve_forever()


if __name__ == "__main__":
    main()
