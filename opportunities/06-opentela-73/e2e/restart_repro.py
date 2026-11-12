#!/usr/bin/env python3
"""End-to-end reproduction of eth-easl/OpenTela#73 on localhost.

One trial runs real otela binaries:

  1. Start a head (role=head, public-addr 127.0.0.1, mode=standalone so it never
     dials the public bootstrap list) and a worker that bootstraps only to it
     and serves a mock LLM (/health, /v1/models -> "mock-model").
  2. Wait until the head's /v1/dnt/table lists the worker with its llm service.
  3. Write --history records into the CRDT (POST /v1/dnt/_node {} on the head:
     one DAG node each), standing in for the history of a long-lived mesh.
  4. Stop the worker with SIGTERM, as an operator or SLURM would: otela
     announces LEFT and wipes its store.
  5. Wait --delay seconds, then restart the worker with the same HOME, so the
     same peer ID, and (cleanslate=true, the default) an empty store.
  6. Poll the head's /v1/dnt/table every 0.25 s and report how long it takes for
     the worker to be listed again with its llm service ("routable"), or a
     timeout.

Every node gets its own HOME under --workdir, so keys, cfg.yaml and the
badger store never touch the real home directory. AXIOM_* and OF_* variables
are scrubbed from the environment. Results are appended to --csv as one row
per trial.
"""

import argparse
import csv
import http.server
import json
import os
import shutil
import signal
import socket
import subprocess
import threading
import time
import urllib.error
import urllib.request
from datetime import datetime
from pathlib import Path

MODEL = "mock-model"


def free_port():
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


class MockLLM(http.server.BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path.startswith("/health"):
            body = b'{"status":"ok"}'
        elif self.path.startswith("/v1/models"):
            body = json.dumps({"object": "list", "data": [{"id": MODEL}]}).encode()
        else:
            self.send_response(404)
            self.end_headers()
            return
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *args):
        pass


def start_mock_llm():
    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), MockLLM)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server


def http_json(url, body=None, timeout=3.0):
    data = None if body is None else json.dumps(body).encode()
    req = urllib.request.Request(url, data=data, method="GET" if body is None else "POST")
    if body is not None:
        req.add_header("Content-Type", "application/json")
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        raw = resp.read()
        return json.loads(raw) if raw else None


class Node:
    def __init__(self, binary, home, args, log_path):
        self.binary, self.home, self.args, self.log_path = binary, Path(home), args, log_path
        self.http_port, self.tcp_port, self.udp_port = free_port(), free_port(), free_port()
        self.proc = None

    def url(self, path):
        return f"http://127.0.0.1:{self.http_port}{path}"

    def start(self):
        self.home.mkdir(parents=True, exist_ok=True)
        env = {k: v for k, v in os.environ.items() if not k.startswith(("OF_", "AXIOM_"))}
        env["HOME"] = str(self.home)
        env["OF_PORT"] = str(self.http_port)
        # Local builds carry no build attestation, and require_signed_binary
        # defaults to true, which makes every peer drop every other peer's
        # row. Production runs signed binaries that pass the same check.
        env["OF_SECURITY_REQUIRE_SIGNED_BINARY"] = "false"
        cmd = [self.binary, "start", "--tcpport", str(self.tcp_port), "--udpport", str(self.udp_port), *self.args]
        log = open(self.log_path, "ab")
        log.write(f"\n===== {time.strftime('%H:%M:%S')} start: {' '.join(cmd)}\n".encode())
        log.flush()
        self.proc = subprocess.Popen(cmd, env=env, cwd=self.home, stdout=log, stderr=subprocess.STDOUT,
                                     start_new_session=True)
        log.close()

    def self_id(self, timeout=60):
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if self.proc.poll() is not None:
                raise RuntimeError(f"{self.home.name} exited early (code {self.proc.returncode}); see {self.log_path}")
            try:
                me = http_json(self.url("/v1/self"))
                if me and me.get("id"):
                    return me["id"]
            except (urllib.error.URLError, ConnectionError, TimeoutError, json.JSONDecodeError):
                pass
            time.sleep(0.25)
        raise TimeoutError(f"{self.home.name} never answered /v1/self")

    def stop(self, sig=signal.SIGTERM, timeout=40):
        if not self.proc or self.proc.poll() is not None:
            return
        self.proc.send_signal(sig)
        try:
            self.proc.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            self.proc.kill()
            self.proc.wait()


def routable(head, worker_id):
    """True if the head's routing table lists the worker with its llm model."""
    try:
        table = http_json(head.url("/v1/dnt/table"))
    except (urllib.error.URLError, ConnectionError, TimeoutError, json.JSONDecodeError):
        return False
    row = (table or {}).get("/" + worker_id)
    if not row:
        return False
    return any(s.get("name") == "llm" and f"model={MODEL}" in (s.get("identity_group") or [])
               for s in row.get("service") or [])


def guard_seconds(log_path, since_wall):
    """Seconds from the restart to the fixed build's first self-row republish."""
    try:
        lines = Path(log_path).read_text(errors="replace").splitlines()
    except OSError:
        return ""
    for line in lines:
        if "Own row holds a record from an earlier run" not in line:
            continue
        try:
            stamp = datetime.strptime(line.split()[0], "%Y-%m-%dT%H:%M:%S.%f%z").timestamp()
        except ValueError:
            continue
        if stamp >= since_wall:
            return round(stamp - since_wall, 2)
    return ""


def wait_until(pred, timeout, step=0.25):
    start = time.monotonic()
    while time.monotonic() - start < timeout:
        if pred():
            return time.monotonic() - start
        time.sleep(step)
    return None


def trial(binary, label, history, delay, timeout, workdir, run, worker_public):
    tag = "-pub" if worker_public else ""
    root = Path(workdir).resolve() / f"{label}{tag}-h{history}-d{delay}-r{run}"
    shutil.rmtree(root, ignore_errors=True)
    root.mkdir(parents=True)
    mock = start_mock_llm()
    head = Node(binary, root / "head", ["--mode", "standalone", "--role", "head", "--public-addr", "127.0.0.1"],
                root / "head.log")
    worker = None
    result = dict(binary=label, worker_public_addr=worker_public, history=history, delay_s=delay, run=run,
                  initial_join_s="", left_hidden_s="", time_to_routable_s="", outcome="",
                  worker_self_status="", worker_self_services="", after_post_s="", guard_s="")
    try:
        head.start()
        head_id = head.self_id()
        boot = f"/ip4/127.0.0.1/tcp/{head.tcp_port}/p2p/{head_id}"
        worker_args = ["--role", "worker", "--bootstrap.addr", boot,
                       "--service.name", "llm", "--service.port", str(mock.server_address[1])]
        if worker_public:
            # A worker with a routable address skips relay reservations
            # (MakeRelayReservations returns early), and with them the one
            # re-announce a restarted worker otherwise gets ~10 s after start.
            worker_args += ["--public-addr", "127.0.0.1"]
        worker = Node(binary, root / "worker", worker_args, root / "worker.log")
        worker.start()
        worker_id = worker.self_id()
        joined = wait_until(lambda: routable(head, worker_id), 180)
        if joined is None:
            result["outcome"] = "setup: worker never became routable"
            return result
        result["initial_join_s"] = round(joined, 2)

        for _ in range(history):
            http_json(head.url("/v1/dnt/_node"), {}, timeout=10)
        time.sleep(3)  # let the worker merge the tail of the history

        worker.stop(signal.SIGTERM)  # AnnounceLeave, 5 s drain, store wipe
        hidden = wait_until(lambda: not routable(head, worker_id), 30)
        result["left_hidden_s"] = "" if hidden is None else round(hidden, 2)

        time.sleep(delay)
        restart_wall = time.time()
        worker.start()  # same HOME: same peer ID; cleanslate wipes the store
        if worker.self_id() != worker_id:
            result["outcome"] = "setup: peer ID changed across restart"
            return result
        ttr = wait_until(lambda: routable(head, worker_id), timeout)
        # What the restarted worker says about itself (the issue: "ready", with
        # its services, while the head does not list it).
        me = http_json(worker.url("/v1/self"))
        result["worker_self_status"] = me.get("status", "")
        result["worker_self_services"] = len(me.get("service") or [])
        if ttr is None:
            result["outcome"] = f"not routable after {timeout}s"
            # The workaround from the issue: force a fresh Put on the worker.
            post_wall = time.monotonic()
            http_json(worker.url("/v1/dnt/_node"), {}, timeout=10)
            after = wait_until(lambda: routable(head, worker_id), 60)
            result["after_post_s"] = "" if after is None else round(time.monotonic() - post_wall, 2)
        else:
            result["time_to_routable_s"] = round(ttr, 2)
            result["outcome"] = "routable"
        result["guard_s"] = guard_seconds(root / "worker.log", restart_wall)
        return result
    finally:
        if worker:
            worker.stop(signal.SIGKILL, timeout=10)
        head.stop(signal.SIGKILL, timeout=10)
        mock.shutdown()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--binary", required=True)
    ap.add_argument("--label", required=True)
    ap.add_argument("--history", type=int, default=0)
    ap.add_argument("--delay", type=int, default=0)
    ap.add_argument("--timeout", type=int, default=240)
    ap.add_argument("--runs", type=int, default=1)
    ap.add_argument("--worker-public-addr", action="store_true",
                    help="give the worker --public-addr 127.0.0.1 (no relay reservation)")
    ap.add_argument("--workdir", required=True)
    ap.add_argument("--csv", required=True)
    args = ap.parse_args()

    csv_path = Path(args.csv)
    new = not csv_path.exists()
    with open(csv_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["binary", "worker_public_addr", "history", "delay_s", "run",
                                               "initial_join_s", "left_hidden_s", "time_to_routable_s",
                                               "outcome", "worker_self_status", "worker_self_services",
                                               "after_post_s", "guard_s"])
        if new:
            writer.writeheader()
        for run in range(1, args.runs + 1):
            res = trial(os.path.abspath(args.binary), args.label, args.history, args.delay, args.timeout,
                        args.workdir, run, args.worker_public_addr)
            print(json.dumps(res), flush=True)
            writer.writerow(res)
            f.flush()


if __name__ == "__main__":
    main()
