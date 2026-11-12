"""Test setup.

The SGLang patch adds two GPU-free modules:
  - python/sglang/srt/model_executor/runner/lazy_capture.py
  - python/sglang/srt/managers/scheduler_components/lazy_graph_capture.py

Importing them through the real `sglang` package would execute its
`__init__.py` files, which import torch and CUDA extensions. Instead we register
empty package objects whose `__path__` points into the patched vendor tree, so
only the two modules (stdlib + msgspec) are executed.

Set LAZYGRAPH_SGLANG_ROOT to test another checkout (e.g. a clean v0.5.20 tree
with the patch applied by `git apply`).
"""

from __future__ import annotations

import importlib
import os
import sys
import types
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))

SGLANG_ROOT = Path(
    os.environ.get("LAZYGRAPH_SGLANG_ROOT", ROOT / "vendor" / "sglang-v0.5.20")
)
_PKG_DIR = SGLANG_ROOT / "python" / "sglang"


def _stub_package(name: str, path: Path) -> None:
    if name in sys.modules:
        return
    module = types.ModuleType(name)
    module.__path__ = [str(path)]  # type: ignore[attr-defined]
    sys.modules[name] = module


for _name, _rel in (
    ("sglang", ""),
    ("sglang.srt", "srt"),
    ("sglang.srt.model_executor", "srt/model_executor"),
    ("sglang.srt.model_executor.runner", "srt/model_executor/runner"),
    ("sglang.srt.managers", "srt/managers"),
    ("sglang.srt.managers.scheduler_components", "srt/managers/scheduler_components"),
):
    _stub_package(_name, _PKG_DIR / _rel if _rel else _PKG_DIR)


def load_patch_module(dotted: str):
    return importlib.import_module(dotted)
