"""importprof, prewarm, tracer and the log timeline parser."""
import json
import os
import sys

import pytest

from conftest import ROOT, SRC, run_py
from pystartup import importprof, prewarm

sys.path.insert(0, os.path.join(ROOT, "analysis"))


def test_parse_importtime():
    err = ("import time: self [us] | cumulative | imported package\n"
           "import time:       120 |        120 |   _io\n"
           "import time:      3000 |       5000 | json\n"
           "import time:      2000 |       2000 |   json.decoder\n")
    rows = importprof.parse_importtime(err)
    assert [(r["module"], r["depth"]) for r in rows] == [("_io", 1), ("json", 0), ("json.decoder", 1)]
    s = importprof.summarize_importtime(rows)
    assert s["by_top_package_s"]["json"] == 0.005
    assert s["slowest_roots"][0]["module"] == "json"


def test_probe_counts_compiles_only_without_pycs(tmp_path):
    pkg = tmp_path / "zz_probe_pkg"
    pkg.mkdir()
    (pkg / "__init__.py").write_text("from . import a, b\n")
    (pkg / "a.py").write_text("A = [i * i for i in range(100)]\n")
    (pkg / "b.py").write_text("B = {'x': 1}\n")
    env = {"PYTHONPATH": os.pathsep.join([str(tmp_path), SRC])}
    first = importprof.run_probe(["zz_probe_pkg"], "asis", env_extra=env, workdir=str(tmp_path), audit_pyc=True)
    assert first["returncode"] == 0, first.get("stderr_tail")
    assert first["compiled_files"] >= 3            # no pycs yet: compiled (and written)
    warm = importprof.run_probe(["zz_probe_pkg"], "asis", env_extra=env, workdir=str(tmp_path))
    assert warm["compiled_files"] == 0             # pycs written by the first run are used
    cold = importprof.run_probe(["zz_probe_pkg"], "nopyc", env_extra=env, workdir=str(tmp_path))
    assert cold["compiled_files"] >= 3             # read-only image without pycs: compiled every time
    assert cold["dont_write_bytecode"] is True
    assert "importtime" in warm


def test_prefix_cold_then_warm(tmp_path):
    pkg = tmp_path / "zz_prefix_pkg.py"
    pkg.write_text("X = 1\n")
    env = {"PYTHONPATH": os.pathsep.join([str(tmp_path), SRC])}
    pdir = str(tmp_path / "prefix")
    c = importprof.run_probe(["zz_prefix_pkg"], "prefix-cold", prefix_dir=pdir, env_extra=env, workdir=str(tmp_path))
    w = importprof.run_probe(["zz_prefix_pkg"], "prefix-warm", prefix_dir=pdir, env_extra=env, workdir=str(tmp_path))
    assert c["compiled_files"] >= 1 and w["compiled_files"] == 0
    assert not os.path.exists(tmp_path / "__pycache__")  # nothing written next to the source


def test_compileall_unchecked_hash_prefix(tmp_path):
    src = tmp_path / "zz_ca.py"
    src.write_text("Y = 2\n")
    pdir = str(tmp_path / "pfx")
    r = importprof.compileall_prefix([], pdir, invalidation="unchecked-hash", files=[str(src)])
    assert r["returncode"] == 0 and r["requested"] == 1 and r["requested_compiled"] == 1
    from pystartup import pycheck

    assert pycheck.classify(str(src), pycheck.pyc_path_for(str(src), pdir)) == pycheck.VALID_UNCHECKED_HASH


def test_prewarm_reads_manifest(tmp_path):
    files = []
    for i in range(5):
        p = tmp_path / f"f{i}.bin"
        p.write_bytes(os.urandom(100_000 + i))
        files.append(str(p))
    man = str(tmp_path / "m.txt")
    assert prewarm.write_manifest(files + [files[0], str(tmp_path / "missing")], man) == 5
    paths = prewarm.load_manifest(man)
    assert paths == files
    r = prewarm.prewarm(paths, threads=4)
    assert r["files"] == 5 and r["bytes"] == sum(os.path.getsize(f) for f in files)
    th = prewarm.start_background(paths, threads=2)
    th.join(10)
    assert not th.is_alive()
    res = prewarm.residency(paths)
    if sys.platform.startswith("linux"):
        assert res["files"] == 5
    else:
        assert "error" in res


def test_tracer_records_processes_and_forks(tmp_path):
    trace = tmp_path / "trace"
    code = """
import json, os, email.mime.text
pid = os.fork()
if pid == 0:
    import xml.dom.minidom
    os._exit(0)
os.waitpid(pid, 0)
"""
    r = run_py(code, env_extra={"PYTHONPATH": os.pathsep.join([os.path.join(SRC, "pystartup", "tracer"), SRC]),
                                "PYSTARTUP_TRACE_DIR": str(trace), "PYSTARTUP_TRACE_MODULES": "1"})
    assert r.returncode == 0, r.stderr
    recs = [json.load(open(trace / f)) for f in os.listdir(trace) if f.endswith(".json")]
    parents = [x for x in recs if not x["forked_from"]]
    kids = [x for x in recs if x["forked_from"]]
    assert len(parents) == 1 and len(kids) == 1
    assert parents[0]["imports"] > 0
    assert kids[0]["forked_from"] == parents[0]["pid"]
    assert any(m[1].startswith("email") for m in parents[0]["modules"])
    from pystartup.profile_startup import tree_summary

    t = tree_summary(str(trace))
    assert t["processes"] == 2


def test_log_timeline_on_synthetic_log(tmp_path):
    import log_timeline as lt

    d = tmp_path / "results" / "clariden-apertus8b"
    d.mkdir(parents=True)
    run = d / "apertus8b-mmap-1"
    run.mkdir()
    # spawn at 12:00:00 CEST
    import datetime as dt

    t_spawn = dt.datetime(2026, 8, 25, 12, 0, 0, tzinfo=lt.CEST).timestamp()
    json.dump({"started_at": t_spawn, "phases": [{"name": "process_startup", "duration_s": 0.5},
                                                  {"name": "tp_worker_spawn", "duration_s": 25.0}]},
              open(run / "run.json", "w"))
    log = "\n".join([
        "[SERVEKIT] profiling cold start of sglang",
        "/x/torchao/quantization/quant_api.py:1731: SyntaxWarning: invalid escape sequence '\\.'",
        "[2026-08-25 12:00:11] server_args=ServerArgs(model_path='m')",
        "[2026-08-25 12:00:25 TP0] Tokenizer ...",
        "[2026-08-25 12:00:26 TP0] Init torch distributed begin.",
        "[2026-08-25 12:00:26 TP1] Init torch distributed begin.",
    ])
    p = tmp_path / "experiments" / "servekit-eval" / "results" / "clariden-apertus8b"
    p.mkdir(parents=True)
    (p / "apertus8b-mmap-1.out").write_text(log)
    import shutil

    shutil.copytree(run, p / "apertus8b-mmap-1")
    row = lt.analyse(str(p / "apertus8b-mmap-1.out"))
    assert row["main_to_server_args_s"] == 11.5
    assert row["children_to_dist_begin_s"] == 15.0
    assert row["syntaxwarn_after_handover"] == 1 and row["syntaxwarn_before_handover"] == 0
    assert row["cluster"] == "clariden"
