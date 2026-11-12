"""HTTP client for SGLang's native /generate endpoint (stdlib only).

Requests send token ids (`input_ids`) so prompt lengths are exact without a
tokenizer, use greedy decoding and `ignore_eos` so output lengths are fixed.
Times are time.monotonic() seconds; callers pass the experiment's t0.
"""

from __future__ import annotations

import json
import random
import threading
import time
import urllib.error
import urllib.request
from dataclasses import asdict, dataclass, field
from typing import Dict, List, Optional, Sequence

VOCAB_LOW, VOCAB_HIGH = 1000, 30000  # inside every Llama/Apertus vocabulary


def make_prompt(n_tokens: int, seed: int) -> List[int]:
    rng = random.Random(seed)
    return [rng.randrange(VOCAB_LOW, VOCAB_HIGH) for _ in range(n_tokens)]


def wait_ready(base_url: str, *, timeout_s: float, poll_s: float = 0.25) -> float:
    """Block until GET /health returns 200; returns time.monotonic() then."""
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        try:
            with urllib.request.urlopen(f"{base_url}/health", timeout=2) as r:
                if r.status == 200:
                    return time.monotonic()
        except (urllib.error.URLError, ConnectionError, TimeoutError, OSError):
            pass
        time.sleep(poll_s)
    raise TimeoutError(f"server at {base_url} not ready after {timeout_s} s")


@dataclass
class RequestResult:
    rid: int
    t_intended: float  # when the trace says the request arrives (since t0)
    t_sent: float  # when the HTTP request was accepted (since t0)
    prompt_tokens: int
    output_tokens: int = 0
    ttft_s: Optional[float] = None  # first token - t_intended
    tpot_s: Optional[float] = None  # mean inter-token time after the first
    e2e_s: Optional[float] = None
    output_ids: List[int] = field(default_factory=list)
    error: Optional[str] = None


def stream_generate(
    base_url: str,
    *,
    input_ids: Sequence[int],
    max_new_tokens: int,
    t0: float,
    t_intended: float,
    rid: int,
    connect_timeout_s: float = 900.0,
    return_ids: bool = False,
) -> RequestResult:
    """POST /generate with stream=true; retries the connection until accepted.

    A request that arrives before the server listens keeps retrying, so its
    TTFT includes the cold start, as it would behind a load balancer.
    """
    body = json.dumps(
        {
            "input_ids": list(input_ids),
            "sampling_params": {
                "max_new_tokens": max_new_tokens,
                "temperature": 0.0,
                "ignore_eos": True,
            },
            "stream": True,
            "return_logprob": False,
        }
    ).encode()
    res = RequestResult(
        rid=rid, t_intended=t_intended, t_sent=-1.0, prompt_tokens=len(input_ids)
    )
    deadline = time.monotonic() + connect_timeout_s
    while True:
        req = urllib.request.Request(
            f"{base_url}/generate",
            data=body,
            headers={"Content-Type": "application/json"},
        )
        try:
            resp = urllib.request.urlopen(req, timeout=connect_timeout_s)
            break
        except (urllib.error.URLError, ConnectionError, OSError) as e:
            if time.monotonic() > deadline:
                res.error = f"connect: {e}"
                return res
            time.sleep(0.25)
    res.t_sent = time.monotonic() - t0
    first = last = None
    n_first = n_last = 0
    try:
        with resp:
            for raw in resp:
                line = raw.decode().strip()
                if not line.startswith("data:"):
                    continue
                payload = line[5:].strip()
                if payload == "[DONE]":
                    break
                chunk = json.loads(payload)
                now = time.monotonic() - t0
                n = int(chunk.get("meta_info", {}).get("completion_tokens", 0))
                if n <= 0:
                    continue
                if first is None:
                    first, n_first = now, n
                last, n_last = now, n
                if return_ids and "output_ids" in chunk:
                    res.output_ids = list(chunk["output_ids"])
    except Exception as e:  # noqa: BLE001 - recorded, not raised
        res.error = f"stream: {e}"
        return res
    if first is None:
        res.error = "no tokens"
        return res
    res.output_tokens = n_last
    res.ttft_s = first - t_intended
    res.e2e_s = last - t_intended
    if n_last > n_first:
        res.tpot_s = (last - first) / (n_last - n_first)
    return res


def generate_ids(base_url: str, *, input_ids: Sequence[int], max_new_tokens: int) -> List[int]:
    """Non-streaming greedy generation; returns output token ids (correctness)."""
    body = json.dumps(
        {
            "input_ids": list(input_ids),
            "sampling_params": {
                "max_new_tokens": max_new_tokens,
                "temperature": 0.0,
                "ignore_eos": True,
            },
        }
    ).encode()
    req = urllib.request.Request(
        f"{base_url}/generate", data=body, headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(req, timeout=600) as r:
        out = json.loads(r.read())
    if isinstance(out, list):
        out = out[0]
    return list(out.get("output_ids") or [])


def replay(
    base_url: str,
    *,
    arrivals_s: Sequence[float],
    prompt_tokens: Sequence[int],
    output_tokens: Sequence[int],
    t0: float,
    max_prompt: int,
    max_output: int,
    seed: int = 0,
) -> List[RequestResult]:
    """Send requests at their trace offsets from t0 (threads; open loop)."""
    results: List[Optional[RequestResult]] = [None] * len(arrivals_s)

    def worker(i: int) -> None:
        delay = t0 + arrivals_s[i] - time.monotonic()
        if delay > 0:
            time.sleep(delay)
        p = int(min(max(prompt_tokens[i], 16), max_prompt))
        o = int(min(max(output_tokens[i], 8), max_output))
        results[i] = stream_generate(
            base_url,
            input_ids=make_prompt(p, seed * 100003 + i),
            max_new_tokens=o,
            t0=t0,
            t_intended=arrivals_s[i],
            rid=i,
        )

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(len(arrivals_s))]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    return [r for r in results if r is not None]


def throughput(
    base_url: str, *, n: int, prompt_len: int, output_len: int, concurrency: int, seed: int = 7
) -> Dict[str, float]:
    """Closed-loop throughput: `concurrency` workers, `n` requests in total."""
    lock = threading.Lock()
    todo = list(range(n))
    done: List[RequestResult] = []
    t0 = time.monotonic()

    def worker() -> None:
        while True:
            with lock:
                if not todo:
                    return
                i = todo.pop()
            r = stream_generate(
                base_url,
                input_ids=make_prompt(prompt_len, seed * 7919 + i),
                max_new_tokens=output_len,
                t0=t0,
                t_intended=time.monotonic() - t0,
                rid=i,
            )
            with lock:
                done.append(r)

    threads = [threading.Thread(target=worker) for _ in range(concurrency)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    elapsed = time.monotonic() - t0
    ok = [r for r in done if r.error is None]
    out_tokens = sum(r.output_tokens for r in ok)
    return {
        "requests": len(ok),
        "errors": len(done) - len(ok),
        "elapsed_s": elapsed,
        "output_tok_per_s": out_tokens / elapsed if elapsed > 0 else 0.0,
    }


def result_dicts(results: Sequence[RequestResult]) -> List[dict]:
    return [asdict(r) for r in results]
