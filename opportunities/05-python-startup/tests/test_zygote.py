"""Zygote scenarios, each in a fresh interpreter (a process has one forkserver)."""
import pytest

from conftest import H100_FACTS, NO_GPU_FACTS, run_json

pytest.importorskip("torch")

SCENARIO = """
import json, multiprocessing, os
from pystartup import zygote
z = zygote.start({mods!r}, facts={facts!r}, strict={strict!r}, deny_modules={deny!r})
zygote.install_parent_hooks()
st = z.wait(120)
out = {{"status": st}}
if st.get("ok") and {fork!r}:
    from zt_child import child
    os.environ["SET_AFTER_START"] = "yes"
    ctx = multiprocessing.get_context("forkserver")
    q = ctx.Queue()
    ps = [ctx.Process(target=child, args=(q,)) for _ in range(2)]
    for p in ps:
        p.start()
    out["children"] = [q.get(timeout=60) for _ in ps]
    for p in ps:
        p.join()
print("RESULT " + json.dumps(out, default=str))
"""

CHILD = """import importlib, os, sys
MODS = {mods!r}
def child(q):
    pre = {{m: (m in sys.modules) for m in MODS}}
    for m in MODS:
        importlib.import_module(m)
    q.put({{"pre": pre, "env": os.environ.get("SET_AFTER_START"),
           "child_marker": os.environ.get("PYSTARTUP_ZYGOTE_CHILD")}})
"""


def scenario(modsdir, mods, facts=NO_GPU_FACTS, strict=False, deny=("zt_denied_target",), fork=True):
    import os

    with open(os.path.join(modsdir["dir"], "zt_child.py"), "w") as f:
        f.write(CHILD.format(mods=list(mods)))
    code = SCENARIO.format(mods=list(mods), facts=facts, strict=strict, deny=list(deny), fork=fork)
    return run_json(code, cwd=modsdir["dir"])


def exec_count(modsdir):
    try:
        return open(modsdir["marker"]).read().splitlines()
    except FileNotFoundError:
        return []


def test_children_inherit_preload_and_env(modsdir):
    out = scenario(modsdir, ["json", "zt_counted"])
    st = out["status"]
    assert st["ok"], st["reasons"]
    assert st["invariants"]["torch_cuda_initialized"] is False
    assert st["invariants"]["nvidia_fds"] == []
    assert st["invariants"]["python_threads"] == 1
    for c in out["children"]:
        assert c["pre"] == {"json": True, "zt_counted": True}   # no re-import in the worker
        assert c["env"] == "yes"                               # env set after zygote start arrives
        assert c["child_marker"] is not None
    assert exec_count(modsdir) == ["x"]                         # module body ran once, in the zygote


def test_facts_are_served_during_preload(modsdir):
    out = scenario(modsdir, ["zt_uses_facts"], facts=H100_FACTS, fork=False)
    st = out["status"]
    assert st["ok"], st["reasons"]
    assert ["zt_uses_facts"] == [m for m, _ in st["loaded"] if m == "zt_uses_facts"]


def test_cuda_touch_disables_zygote(modsdir):
    out = scenario(modsdir, ["json", "zt_trips"], facts=H100_FACTS, fork=False)
    st = out["status"]
    assert not st["ok"]
    assert st["tripped"][0][0] == "zt_trips"
    assert any("touched CUDA" in r for r in st["reasons"])


def test_trip_after_side_effect_is_still_refused(modsdir):
    out = scenario(modsdir, ["zt_partial"], facts=H100_FACTS, fork=False)
    assert not out["status"]["ok"]
    assert exec_count(modsdir) == ["partial"]  # the side effect happened: that is why we refuse


def test_swallowed_trip_is_detected(modsdir):
    out = scenario(modsdir, ["zt_swallows"], facts=H100_FACTS, fork=False)
    st = out["status"]
    assert not st["ok"]
    assert st["tainted"][0][0] == "zt_swallows"


def test_denied_module_disables_zygote(modsdir):
    out = scenario(modsdir, ["zt_denied_dep"], fork=False)
    st = out["status"]
    assert not st["ok"]
    assert st["invariants"]["denied_present"] == ["zt_denied_target"]


def test_python_thread_disables_zygote(modsdir):
    out = scenario(modsdir, ["zt_thread"], fork=False)
    st = out["status"]
    assert not st["ok"]
    assert st["invariants"]["python_threads"] == 2


def test_missing_module_is_tolerated_in_best_effort_mode(modsdir):
    out = scenario(modsdir, ["json", "definitely_not_installed_xyz"], fork=False)
    assert out["status"]["ok"], out["status"]["reasons"]
    out = scenario(modsdir, ["json", "definitely_not_installed_xyz"], strict=True, fork=False)
    assert not out["status"]["ok"]


def test_calibration_keeps_only_clean_modules(modsdir):
    code = f"""
import json
from pystartup import zygote
res = zygote.calibrate(["json", "zt_trips", "zt_uses_facts", "zt_swallows"], facts={H100_FACTS!r})
print("RESULT " + json.dumps(res))
"""
    res = run_json(code, cwd=modsdir["dir"])
    t = res["targets"]
    assert t["json"] == "ok" and t["zt_uses_facts"] == "ok"
    assert t["zt_trips"].startswith("tripped")
    assert t["zt_swallows"] == "tainted"
    assert "zt_uses_facts" in res["modules"]
    assert "zt_trips" not in res["modules"]
