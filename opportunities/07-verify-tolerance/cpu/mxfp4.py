"""Make an MXFP4 variant of a BF16 checkpoint, for corrupting real E8M0 scales on CPU.

OCP Microscaling (MX) v1.0: blocks of 32 elements share one E8M0 scale 2^e;
each element is E2M1 FP4 (+-{0, 0.5, 1, 1.5, 2, 3, 4, 6}). The shared exponent
is floor(log2(amax)) - 2 (2 is E2M1's largest exponent), elements beyond 6
saturate. Storage follows gpt-oss: `<name>_blocks` [out, in/32, 16] uint8 with
two elements per byte (low nibble first) and `<name>_scales` [out, in/32] uint8
holding e + 127.

    python -m cpu.mxfp4 models/Qwen3-0.6B models/Qwen3-0.6B-MXFP4-mlp
"""
from __future__ import annotations

import json
import re
import shutil
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "vendor" / "servekit" / "src"))
from servekit import faults  # noqa: E402

E2M1 = np.array([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0], dtype=np.float32)
MATCH = r"mlp\.(gate|up|down)_proj\.weight$"


def quantize(w: np.ndarray):
    """[out, in] float32 -> (blocks [out, in/32, 16] uint8, scales [out, in/32] uint8)."""
    out_f, in_f = w.shape
    if in_f % 32:
        raise ValueError(f"in_features {in_f} is not a multiple of 32")
    g = w.reshape(out_f, in_f // 32, 32)
    amax = np.abs(g).max(-1)
    with np.errstate(divide="ignore"):
        e = np.where(amax > 0, np.floor(np.log2(np.maximum(amax, 1e-38))) - 2, -127).astype(np.int32)
    e = np.clip(e, -127, 127)
    scaled = g / np.exp2(e.astype(np.float32))[..., None]
    mag = np.minimum(np.abs(scaled), 6.0)
    # nearest E2M1 magnitude (ties go to the lower code, i.e. the even mantissa here)
    idx = np.abs(mag[..., None] - E2M1).argmin(-1).astype(np.uint8)
    code = idx | ((scaled < 0) & (idx > 0)).astype(np.uint8) << 3
    packed = (code[..., 0::2] | (code[..., 1::2] << 4)).astype(np.uint8)
    return packed, (e + 127).astype(np.uint8)


def dequantize(blocks: np.ndarray, scales: np.ndarray) -> np.ndarray:
    lut = np.concatenate([E2M1, -E2M1])
    lo, hi = blocks & 0x0F, blocks >> 4
    vals = np.stack([lut[lo], lut[hi]], -1).reshape(*blocks.shape[:-1], 32)
    return (vals * np.exp2(scales.astype(np.float32) - 127)[..., None]).reshape(blocks.shape[0], -1)


def convert(src: Path, dst: Path, match: str = MATCH) -> int:
    src, dst = Path(src), Path(dst)
    dst.mkdir(parents=True, exist_ok=False)
    pattern = re.compile(match)
    n = 0
    for f in faults.checkpoint_files(src):
        tensors, meta = faults.read_file(f)
        out = {}
        for name, t in tensors.items():
            if pattern.search(name) and t.dtype == "BF16" and len(t.shape) == 2:
                w = faults.to_float32(np.asarray(t.data), "BF16").reshape(t.shape)
                b, s = quantize(w)
                out[name + "_blocks"] = faults.Tensor("U8", tuple(b.shape), b)
                out[name + "_scales"] = faults.Tensor("U8", tuple(s.shape), s)
                n += 1
            else:
                out[name] = t
        faults.write_file(dst / f.name, out, meta)
    for item in src.iterdir():
        if item.is_file() and not item.name.endswith(".safetensors"):
            shutil.copy2(item, dst / item.name)
    cfg = json.loads((dst / "config.json").read_text())
    cfg["quantization_config"] = {"quant_method": "mxfp4", "note": "MLP weights only, made by cpu/mxfp4.py"}
    (dst / "config.json").write_text(json.dumps(cfg, indent=2))
    return n


if __name__ == "__main__":
    print(f"quantized {convert(Path(sys.argv[1]), Path(sys.argv[2]))} tensors to MXFP4")
