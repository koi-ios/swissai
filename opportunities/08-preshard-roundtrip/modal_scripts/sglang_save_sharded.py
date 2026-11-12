"""SGLang's own sharded_state save, with nothing of servekit in the way.

Mirrors SGLang's examples/runtime/engine/save_sharded_state.py: start an
Engine (the default load, CUDA graph capture included), then ask every rank to
`save_sharded_model`. No quantization guard and no backports, so this is what
an upstream user gets. Non-weight files are copied next to the shards.
"""
import dataclasses
import inspect
import os
import shutil
import sys
from argparse import ArgumentParser

from sglang import Engine, ServerArgs


def main() -> int:
    parser = ArgumentParser()
    ServerArgs.add_cli_args(parser)
    parser.add_argument("--output", required=True)
    parser.add_argument("--max-file-size", type=int, default=5 * 1024**3)
    args = parser.parse_args()
    server_args = ServerArgs.from_cli_args(args)
    os.makedirs(args.output, exist_ok=True)

    llm = Engine(**dataclasses.asdict(server_args))
    payload = {"path": args.output, "pattern": None, "max_size": args.max_file_size}
    from sglang.srt.managers.scheduler import Scheduler

    if "params" in inspect.signature(Scheduler.save_sharded_model).parameters:
        payload = {"params": payload}
    llm.collective_rpc("save_sharded_model", **payload)

    for name in os.listdir(server_args.model_path):
        src = os.path.join(server_args.model_path, name)
        if os.path.isdir(src) or name.endswith((".safetensors", ".bin", ".pt", ".index.json")):
            continue
        shutil.copy(src, os.path.join(args.output, name))
    print(f"saved sharded_state to {args.output}", flush=True)
    llm.shutdown()
    return 0


if __name__ == "__main__":
    sys.exit(main())
