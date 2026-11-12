#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Adapted from vLLM v0.33.0 examples/features/sharded_state/save_sharded_state_offline.py
# (SPDX-FileCopyrightText: Copyright contributors to the vLLM project). Changes: fallbacks
# for older vLLM APIs, a refusal to write into a non-empty directory, timings, and a manifest.
"""Save each tensor-parallel rank's weights as its own files (vLLM's sharded_state format).

    python preshard.py --model /net/models/apertus-8b --tensor-parallel-size 2 \
        --enforce-eager --output /net/models/apertus-8b-tp2

Then each rank reads only its own files:

    vllm serve /net/models/apertus-8b-tp2 --load-format sharded_state --tensor-parallel-size 2

The source must be a local directory. --enforce-eager skips torch.compile and CUDA
graphs, which saving does not need. The saved tensors are post-processing state, so
load them with the same vLLM version: preshard_manifest.json records which.
"""
import dataclasses
import json
import os
import shutil
import sys
import time
from pathlib import Path


def _parser():
    try:
        from vllm.utils.argparse_utils import FlexibleArgumentParser
    except ImportError:  # vLLM < 0.11
        from vllm.utils import FlexibleArgumentParser
    from vllm import EngineArgs

    parser = FlexibleArgumentParser()
    EngineArgs.add_cli_args(parser)
    parser.add_argument("--output", "-o", required=True, type=str, help="path to output checkpoint")
    parser.add_argument("--file-pattern", type=str, default=None,
                        help="saved file names; default model-rank-{rank}-part-{part}.safetensors")
    parser.add_argument("--max-file-size", type=int, default=5 * 1024**3,
                        help="max size (in bytes) of each safetensors file")
    return parser


def _save(llm, path: str, pattern, max_size: int) -> None:
    engine = llm.llm_engine
    if hasattr(engine, "engine_core"):  # V1 engine (what the 0.33 example calls)
        engine.engine_core.save_sharded_state(path=path, pattern=pattern, max_size=max_size)
    elif hasattr(engine, "model_executor"):  # V0 engine
        engine.model_executor.save_sharded_state(path=path, pattern=pattern, max_size=max_size)
    else:
        llm.collective_rpc("save_sharded_state", kwargs={"path": path, "pattern": pattern, "max_size": max_size})


def main() -> int:
    args = _parser().parse_args()
    import vllm
    from vllm import LLM, EngineArgs

    engine_args = EngineArgs.from_cli_args(args)
    if engine_args.enable_lora:
        raise ValueError("Saving with enable_lora=True is not supported!")
    model_path = engine_args.model
    if not Path(model_path).is_dir():
        raise ValueError("model path must be a local directory")
    out = Path(args.output)
    if out.exists() and any(out.iterdir()):
        raise SystemExit(f"{out} exists and is not empty; delete it first to re-shard")

    t0 = time.time()
    if hasattr(LLM, "from_engine_args"):
        llm = LLM.from_engine_args(engine_args)
    else:
        llm = LLM(**dataclasses.asdict(engine_args))
    t_init = time.time() - t0

    out.mkdir(parents=True, exist_ok=True)
    t1 = time.time()
    _save(llm, str(out), args.file_pattern, args.max_file_size)
    t_save = time.time() - t1

    # Copy metadata files (config, tokenizer, generation config) next to the shards, as the example does
    for file in os.listdir(model_path):
        if os.path.splitext(file)[1] not in (".bin", ".pt", ".safetensors"):
            src = os.path.join(model_path, file)
            if os.path.isdir(src):
                shutil.copytree(src, os.path.join(out, file))
            else:
                shutil.copy(src, out)

    files = sorted(p for p in out.rglob("*") if p.is_file())
    manifest = {
        "source": str(Path(model_path).resolve()),
        "vllm_version": getattr(vllm, "__version__", None),
        "tensor_parallel_size": engine_args.tensor_parallel_size,
        "pipeline_parallel_size": engine_args.pipeline_parallel_size,
        "dtype": str(engine_args.dtype),
        "quantization": engine_args.quantization,
        "file_pattern": args.file_pattern,
        "created": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "engine_init_s": round(t_init, 1),
        "save_s": round(t_save, 1),
        "files": {str(p.relative_to(out)): p.stat().st_size for p in files},
    }
    (out / "preshard_manifest.json").write_text(json.dumps(manifest, indent=1))
    shards = [p for p in files if p.name.startswith("model-") and p.suffix == ".safetensors"]
    print(f"presharded {model_path} -> {out}: {len(shards)} shard files, "
          f"{sum(p.stat().st_size for p in shards) / 1e9:.2f} GB; engine init {t_init:.1f} s, save {t_save:.1f} s",
          flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
