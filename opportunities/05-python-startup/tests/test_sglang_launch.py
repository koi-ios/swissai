import multiprocessing
import os
import sys
import types
from contextlib import contextmanager

import pytest

from pystartup import sglang_launch as sl


def test_split_and_parse():
    w, s = sl.split_args(["--no-zygote", "--report", "r.json", "--", "--model-path", "m", "--tp", "2"])
    assert s == ["--model-path", "m", "--tp", "2"]
    o = sl.parse_wrapper(w)
    assert o["zygote"] is False and o["report"] == "r.json"
    assert sl.split_args(["--model-path", "m"]) == ([], ["--model-path", "m"])
    with pytest.raises(SystemExit):
        sl.parse_wrapper(["--bogus"])


def test_unsupported_paths_fall_back():
    assert sl.unsupported_reason(["--model-path", "m", "--tp", "2"]) is None
    assert "memory-saver" in sl.unsupported_reason(["--enable-memory-saver"])
    assert sl.unsupported_reason(["--dp-size", "2"]) is not None
    assert sl.unsupported_reason(["--dp-size", "1"]) is None


def test_static_env_matches_sglang_logic(monkeypatch):
    monkeypatch.setenv("TRTLLM_ENABLE_PDL", "0")
    monkeypatch.delenv("CUTE_DSL_LOG_LEVEL", raising=False)
    monkeypatch.setenv("CUTE_DSL_LOG_TO_CONSOLE", "0")
    sl.apply_sglang_static_env()
    assert os.environ["TRTLLM_ENABLE_PDL"] == "0"        # user opt-out kept, as in SGLang
    assert os.environ["CUTE_DSL_LOG_LEVEL"] == "30"
    assert os.environ["CUTE_DSL_LOG_TO_CONSOLE"] == "0"  # setdefault semantics
    assert os.environ["CUDA_DEVICE_MAX_CONNECTIONS"] == "8"


class _FakeZygote:
    def __init__(self, ok):
        self.ok = ok

    def wait(self, timeout):
        return {"ok": self.ok, "reasons": [] if self.ok else ["test: not ok"], "preload_s": 0.1}


@pytest.fixture
def restore_mp():
    orig_fn = multiprocessing.set_start_method
    orig_method = multiprocessing.get_start_method(allow_none=True)
    state = dict(sl.STATE)
    yield
    multiprocessing.set_start_method = orig_fn
    if orig_method:
        orig_fn(orig_method, force=True)
    sl.STATE.clear()
    sl.STATE.update(state)
    sys.modules.pop("sglang.srt.utils.numa_utils", None)


@pytest.mark.parametrize("ok,expected", [(True, "forkserver"), (False, "spawn")])
def test_spawn_becomes_forkserver_only_when_zygote_ok(restore_mp, ok, expected):
    sl.STATE.update({"zygote": _FakeZygote(ok), "decided": False, "use_forkserver": False})
    sl.install_start_method_switch(1.0)
    multiprocessing.set_start_method("spawn", force=True)  # what SGLang's _set_envs_and_config does
    assert multiprocessing.get_start_method() == expected
    multiprocessing.set_start_method("spawn", force=True)  # later calls keep the decision
    assert multiprocessing.get_start_method() == expected


def test_numa_binding_worker_is_spawned(restore_mp):
    seen = []

    @contextmanager
    def _mp_set_executable(executable, debug_str):
        seen.append(multiprocessing.get_start_method())
        assert multiprocessing.get_start_method() == "spawn"  # SGLang asserts this
        yield

    mod = types.ModuleType("sglang.srt.utils.numa_utils")
    mod._mp_set_executable = _mp_set_executable
    sys.modules["sglang.srt.utils.numa_utils"] = mod
    sl.STATE.update({"zygote": _FakeZygote(True), "decided": False, "use_forkserver": False})
    sl.install_start_method_switch(1.0)
    multiprocessing.set_start_method("spawn", force=True)
    assert multiprocessing.get_start_method() == "forkserver"
    with mod._mp_set_executable(executable="/tmp/x.sh", debug_str=""):
        assert multiprocessing.get_start_method() == "spawn"
    assert multiprocessing.get_start_method() == "forkserver"
    assert seen == ["spawn"]
