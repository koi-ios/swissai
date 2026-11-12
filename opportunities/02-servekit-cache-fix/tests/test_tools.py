"""Tests for the Modal experiment's tooling, without a GPU or a Modal account.

    .venv/bin/python -m pytest tests -q
"""
import json
import os
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE / "tools"))
import runlog  # noqa: E402
import summarize_modal  # noqa: E402

LAB = HERE.parents[1] / "data/cold-start-elimination-experiments-swissai/experiments/servekit-jit-caching-eval/results"
SERVEKIT = {"patched": HERE / "vendor/servekit", "main": HERE / "vendor/servekit-main"}


# --- the parser, on the lab's own 0.5.16 A/B (Llama-70B TP4, Bristen) -------------------------

@pytest.mark.parametrize(
    "log, dist, first",
    [("servekit-jit-no-cache-81673.out", 19.54, (8192, 28.49)), ("servekit-jit-with-cache-81671.out", 2.74, (8192, 5.69))],
)
def test_the_parser_reads_the_lab_ab(log, dist, first):
    if not (LAB / log).is_file():
        pytest.skip("the experiments repo copy is not here")
    text = (LAB / log).read_text(errors="replace")
    assert runlog.dist_init(text) == dist
    assert runlog.first_prefill_step(text) == first
    assert runlog.custom_all_reduce(text) is True


def test_a_fast_first_step_is_read_from_its_rate():
    line = "Capturing num tokens (num_tokens=4096 avail_mem=60.1 GB):   2%|▏ | 1/52 [00:00<00:20,  2.50it/s]"
    assert runlog.first_prefill_step(line) == (4096, 0.4)


def test_the_inventory_names_the_all_reduce_cache_apart_from_the_rest(tmp_path):
    home, tmp = tmp_path / "home", tmp_path / "tmp"
    for rel, n in {".cache/sglang/jit/sm90/comm.so": 300, ".cache/sglang/triton/k.cubin": 20,
                   ".cache/huggingface/token": 5, ".nv/ComputeCache/0/1": 7}.items():
        (home / rel).parent.mkdir(parents=True, exist_ok=True)
        (home / rel).write_bytes(b"x" * n)
    (tmp / "symm_allocator").mkdir(parents=True)
    (tmp / "symm_allocator" / "a.so").write_bytes(b"x" * 9)

    inv = runlog.inventory(home, tmp)

    assert inv["~/.cache/sglang/jit"] == {"files": 1, "bytes": 300}
    assert inv["~/.cache/sglang/triton"] == {"files": 1, "bytes": 20}
    assert inv["~/.nv/ComputeCache"] == {"files": 1, "bytes": 7}
    assert inv["$TMPDIR/symm_allocator"] == {"files": 1, "bytes": 9}


# --- the runner, end to end with a fake sglang -------------------------------------------------

FAKE_SERVER = textwrap.dedent('''
    import os, sys, time
    from pathlib import Path
    print("[TP0] Init torch distributed begin.", flush=True)
    # sglang 0.5.20: load_jit caches under SGLANG_JIT_CACHE_DIR, else ~/.cache/sglang/jit.
    root = Path(os.environ.get("SGLANG_JIT_CACHE_DIR") or os.path.expanduser("~/.cache/sglang/jit"))
    hit = (root / "sm90" / "comm" / "comm.so").is_file()
    if not hit:
        (root / "sm90" / "comm").mkdir(parents=True, exist_ok=True)
        (root / "sm90" / "comm" / "comm.so").write_bytes(b"x" * 1000)
    print("[TP0] All Reduce config: symmetric_memory = 24.01 MB, local_buffer = 4.00 MB", flush=True)
    print(f"[TP0] Init torch distributed ends. elapsed={0.5 if hit else 17.0:.2f} s, mem usage=1.00 GB", flush=True)
    print("Capturing num tokens (num_tokens=8192 avail_mem=10.49 GB):   2%|  | 1/58 [00:05<05:24,  5.69s/it]", flush=True)
    print("[TP0] Capture target prefill CUDA graph end. elapsed=9.00 s, mem usage=1.0 GB", flush=True)
    print("INFO:     Uvicorn running on http://127.0.0.1:30000", flush=True)
    time.sleep(0.6)
    print('INFO:     127.0.0.1:5 - "POST /generate HTTP/1.1" 200 OK', flush=True)
    print("The server is fired up and ready to roll!", flush=True)
    time.sleep(120)
''')


@pytest.fixture
def fake_sglang(tmp_path):
    pkg = tmp_path / "fake" / "sglang"
    pkg.mkdir(parents=True)
    (pkg / "__init__.py").write_text("")
    (pkg / "launch_server.py").write_text(FAKE_SERVER)
    return tmp_path / "fake"


@pytest.fixture
def artifact(tmp_path):
    model = tmp_path / "Qwen2.5-1.5B-Instruct"
    model.mkdir()
    (model / "config.json").write_text("{}")
    art = tmp_path / "artifacts" / "main"
    art.mkdir(parents=True)
    (art / "config.json").write_text("{}")
    for r in range(2):
        (art / f"model-rank-{r}-part-0.safetensors").write_bytes(b"w")
    (art / "servekit.json").write_text(json.dumps({
        "format": "sharded_state", "engine": "sglang", "engine_version": "0.5.20", "tp_size": 2, "pp_size": 1,
        "dp_size": 1, "dtype": "bfloat16", "quantization": None, "source": str(model)}))
    return model, art


@pytest.mark.parametrize("tree", ["main", "patched"])
def test_one_job_stops_at_ready_and_reports_what_it_left_behind(tmp_path, fake_sglang, artifact, tree):
    if not (SERVEKIT[tree] / "src" / "servekit").is_dir():
        pytest.skip(f"no servekit {tree} checkout")
    model, art = artifact
    run = tmp_path / "runs" / "r1"
    run.mkdir(parents=True)
    env = {**os.environ, "PYTHONPATH": str(fake_sglang)}
    with open(run / "engine.log", "w") as log:
        proc = subprocess.run(
            [sys.executable, str(HERE / "tools" / "run_arm.py"), "--servekit", str(SERVEKIT[tree]),
             "--artifact", str(art), "--model", str(model), "--tp", "2", "--run-dir", str(run),
             "--stage-root", str(tmp_path / "stage"), "--node-local", str(tmp_path / "node-local"),
             "--timeout", "60"],
            stdout=log, stderr=subprocess.STDOUT, env=env, timeout=120,
        )
    row = runlog.read_run(run)

    assert proc.returncode == 0, (run / "engine.log").read_text()[-2000:]
    assert row["rc"] == 0 and row["custom_all_reduce"] is True
    assert row["first_prefill_step_s"] == 5.69 and row["first_request_s"] is not None
    # No artifact caches either way, so the kernel is built; where it goes is the difference.
    assert row["dist_init_s"] == 17.0
    if tree == "main":
        assert "~/.cache/sglang/jit" in row["left_behind"]
    else:
        assert row["left_behind"] == {} and row["jit_cache"]["compiled"]["sglang-jit"]["bytes"] == 1000


def test_the_summary_tables_median_and_range_per_arm():
    rows = [
        {"arm": arm, "rep": i, "rc": 0, "total_s": total + 10, "phases": {"stage": 10.0, "prefill_cuda_graph_capture": 30.0},
         "dist_init_s": dist, "first_prefill_step_s": 5.0, "first_request_s": 1.0, "custom_all_reduce": True,
         "left_behind": {"~/.cache/sglang/jit": {"files": 3, "bytes": 4_000_000}} if arm == "main" else {}}
        for arm, dist, total in (("none", 17.0, 100), ("main", 16.0, 90), ("patched", 2.5, 60))
        for i in (1, 2, 3)
    ]
    rows[0]["dist_init_s"] = 19.0

    table = summarize_modal.table(rows)

    assert "| torch.distributed init (custom all-reduce JIT) | 17.0 [17.0-19.0] | 16.0 [16.0-16.0] | 2.5 [2.5-2.5] |" in table
    assert "| total to ready (excl. weight stage) | 100.0 [100.0-100.0] | 90.0 [90.0-90.0] | 60.0 [60.0-60.0] |" in table
    assert "Custom all-reduce v2 enabled in every run: True" in table
    assert "| main | `~/.cache/sglang/jit` | 4.0 |" in summarize_modal.left_behind(rows)


def test_the_modal_plan_interleaves_three_runs_per_arm():
    sys.path.insert(0, str(HERE))
    import modal_app

    steps = modal_app.plan()
    measured = [s for s in steps if s["kind"] == "launch" and not s.get("prime")]
    assert [s["kind"] for s in steps[:2]] == ["prepare", "prepare"]
    assert sorted(s["arm"] for s in measured) == ["main"] * 3 + ["none"] * 3 + ["patched"] * 3
    assert [s["arm"] for s in measured[:3]] != [s["arm"] for s in measured[3:6]]
    minutes, dollars = modal_app.estimate(steps)
    assert dollars < 15
