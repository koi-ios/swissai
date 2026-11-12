"""Make SGLang's post-load barrier timeout configurable without rebuilding the image.

SGLang hard-codes ``UNBALANCED_MODEL_LOADING_TIMEOUT_S = 480`` (0.5.10.post1:
``sglang/srt/model_executor/model_runner.py``; 0.5.16-0.5.23 and main:
``.../model_runner_components/load_model_utils.py``) and no release reads an
environment variable for it. This file reads ``SGLANG_UNBALANCED_MODEL_LOADING_TIMEOUT_S``
(the name model-launch-factory already sets, and the name proposed upstream in
``sglang-barrier-timeout-env.diff``) and patches the constant right after SGLang
imports the module.

Install: mount this directory and put it on PYTHONPATH in the env TOML, e.g.
    [env]
    PYTHONPATH = "/capstor/.../sml-patches"
    SGLANG_UNBALANCED_MODEL_LOADING_TIMEOUT_S = "3600"

Python imports ``sitecustomize`` at interpreter start, so this reaches the TP
worker subprocesses too. Unlike an eager ``import sglang...`` here (the approach
in Youssef Boughizane's servekit-eval/patches/sitecustomize.py, which this
builds on), the hook below imports nothing: processes that never import SGLang
pay nothing, and import order inside SGLang is unchanged.
"""
import importlib.abc
import importlib.machinery
import os
import sys

ENV = "SGLANG_UNBALANCED_MODEL_LOADING_TIMEOUT_S"
ATTR = "UNBALANCED_MODEL_LOADING_TIMEOUT_S"
TARGETS = {
    "sglang.srt.model_executor.model_runner_components.load_model_utils",  # >= 0.5.16
    "sglang.srt.model_executor.model_runner",  # <= 0.5.10.post1 (harmless later)
}


def _patch(module) -> None:
    if hasattr(module, ATTR):
        setattr(module, ATTR, int(os.environ[ENV]))


class _PostImportPatcher(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path, target=None):
        if fullname not in TARGETS:
            return None
        spec = importlib.machinery.PathFinder.find_spec(fullname, path)
        if spec is None or spec.loader is None:
            return None
        inner_exec = spec.loader.exec_module

        def exec_module(module):
            inner_exec(module)
            _patch(module)

        spec.loader.exec_module = exec_module
        return spec


def install() -> bool:
    raw = os.environ.get(ENV, "")
    if not raw.strip().isdigit():
        return False
    if not any(isinstance(f, _PostImportPatcher) for f in sys.meta_path):
        sys.meta_path.insert(0, _PostImportPatcher())
    for name in TARGETS & set(sys.modules):  # already imported: patch now
        _patch(sys.modules[name])
    return True


install()
