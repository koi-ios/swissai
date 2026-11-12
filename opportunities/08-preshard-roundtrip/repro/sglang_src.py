"""Run selected pieces of the vendored SGLang source without importing SGLang.

Importing `sglang.srt.models.deepseek_v2` pulls in hundreds of modules and
GPU-only packages. The reproductions need only a few functions and statements,
so this module finds them in the vendored v0.5.20 tree with `ast`, prints where
they came from, and compiles exactly that source into a namespace of stand-ins
for the handful of globals they touch. What runs is SGLang's code, byte for
byte; what is mocked is named at each call site.
"""
from __future__ import annotations

import ast
import contextlib
import os
import textwrap
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

ROOT = Path(__file__).resolve().parents[1]
SGLANG = ROOT / "vendor" / "sglang" / "python" / "sglang" / "srt"
LOADER = SGLANG / "model_loader" / "loader.py"
DEEPSEEK = SGLANG / "models" / "deepseek_v2.py"
MXFP4 = SGLANG / "layers" / "quantization" / "mxfp4.py"
LAYERNORM = SGLANG / "layers" / "layernorm.py"

EXTRACTED: List[str] = []  # "file:first-last  qualname", for the record


def _tree(path: Path) -> Tuple[str, ast.Module]:
    text = path.read_text()
    return text, ast.parse(text)


def _find_class(tree: ast.Module, name: str) -> ast.ClassDef:
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name == name:
            return node
    raise LookupError(name)


def _find_def(body: Iterable[ast.stmt], name: str) -> ast.FunctionDef:
    for node in body:
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return node
    raise LookupError(name)


def _segment(path: Path, text: str, node: ast.AST, what: str) -> str:
    src = ast.get_source_segment(text, node, padded=True)
    EXTRACTED.append(f"{path.relative_to(SGLANG.parents[2])}:{node.lineno}-{node.end_lineno}  {what}")
    return textwrap.dedent(src)


def function(path: Path, qualname: str) -> Tuple[str, List[str]]:
    """(dedented source of `Class.method` or `function`, its decorator names)."""
    text, tree = _tree(path)
    parts = qualname.split(".")
    body = tree.body
    if len(parts) == 2:
        body = _find_class(tree, parts[0]).body
    node = _find_def(body, parts[-1])
    decorators = [d.id for d in node.decorator_list if isinstance(d, ast.Name)]
    return _segment(path, text, node, qualname), decorators


def statements(path: Path, qualname: str, predicate) -> str:
    """Source of the statements in `Class.method` (searched recursively) for which predicate(source) holds."""
    text, tree = _tree(path)
    cls, meth = qualname.split(".")
    fn = _find_def(_find_class(tree, cls).body, meth)
    out = []
    for node in ast.walk(fn):
        if isinstance(node, ast.stmt) and node is not fn:
            seg = ast.get_source_segment(text, node)
            if seg is not None and predicate(seg):
                out.append(_segment(path, text, node, f"{qualname} statement"))
    if not out:
        raise LookupError(f"no statement in {qualname} matched")
    return "\n".join(out)


def if_body(path: Path, qualname: str, test_source: str) -> str:
    """The body of the first `if <test_source>:` inside `Class.method`, dedented."""
    text, tree = _tree(path)
    cls, meth = qualname.split(".")
    fn = _find_def(_find_class(tree, cls).body, meth)
    for node in ast.walk(fn):
        if isinstance(node, ast.If) and ast.get_source_segment(text, node.test) == test_source:
            first, last = node.body[0], node.body[-1]
            lines = text.splitlines()[first.lineno - 1 : last.end_lineno]
            EXTRACTED.append(
                f"{path.relative_to(SGLANG.parents[2])}:{first.lineno}-{last.end_lineno}  "
                f"{qualname}: body of `if {test_source}:`"
            )
            return textwrap.dedent("\n".join(lines))
    raise LookupError(test_source)


def compile_into(source: str, namespace: Dict[str, Any], name: str) -> Any:
    exec(compile(source, f"<sglang v0.5.20: {name}>", "exec"), namespace)  # noqa: S102 - vendored, reviewed source
    return namespace[name]


# ----------------------------------------------------------------- the loader


class _Parallel:
    def __init__(self, tp_rank: int = 0):
        self.tp_rank = tp_rank


@contextlib.contextmanager
def _default_dtype(dtype):
    import torch

    old = torch.get_default_dtype()
    if dtype is not None and dtype.is_floating_point:
        torch.set_default_dtype(dtype)
    try:
        yield
    finally:
        torch.set_default_dtype(old)


def sharded_state_loader(make_model, tp_rank: int = 0):
    """SGLang v0.5.20's ShardedStateLoader (loader.py:1662-1845), its methods unmodified.

    Stand-ins: get_parallel() (the rank), _initialize_model (calls make_model),
    _get_quantization_config (None), set_default_torch_dtype, and
    download_weights_from_hf (never reached: the path is a directory).
    """
    import collections
    import glob
    import logging
    from typing import Any as _Any, Dict as _Dict, List as _List, Optional as _Optional, Tuple as _Tuple

    import torch
    from torch import nn

    ns: Dict[str, Any] = {
        "collections": collections, "glob": glob, "os": os, "torch": torch, "nn": nn,
        "Any": _Any, "Dict": _Dict, "List": _List, "Optional": _Optional, "Tuple": _Tuple,
        "logger": logging.getLogger("sglang.repro"),
        "get_parallel": lambda: _Parallel(tp_rank),
        "_initialize_model": lambda model_config, load_config, quant_config: make_model(),
        "_get_quantization_config": lambda model_config, load_config: None,
        "set_default_torch_dtype": _default_dtype,
        "download_weights_from_hf": None,
        "ModelConfig": object, "DeviceConfig": object,
    }
    post_load, _ = function(LOADER, "_post_load_weights")
    compile_into(post_load, ns, "_post_load_weights")

    methods = {}
    for name in ("_filter_subtensors", "_prepare_weights", "load_model", "save_model"):
        src, decorators = function(LOADER, f"ShardedStateLoader.{name}")
        fn = compile_into(src, ns, name)
        methods[name] = staticmethod(fn) if "staticmethod" in decorators else fn
    cls = type("ShardedStateLoader", (), {
        **methods,
        "DEFAULT_PATTERN": "model-rank-{rank}-part-{part}.safetensors",
        "__init__": lambda self: setattr(self, "pattern", "model-rank-{rank}-part-{part}.safetensors"),
    })
    ns["ShardedStateLoader"] = cls  # save_model names its own class
    return cls


def load_with(loader_cls, path: str, dtype=None):
    from types import SimpleNamespace

    import torch

    loader = loader_cls()
    loader.load_config = SimpleNamespace(download_dir=None, ignore_patterns=None)
    config = SimpleNamespace(model_path=path, revision=None, dtype=dtype or torch.float32)
    return loader.load_model(model_config=config, device_config=SimpleNamespace(device="cpu"))
