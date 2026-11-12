"""sml_jit_cache: image key, seed/promote round trip, exclusions, modes, concurrent promotion, GC."""

import importlib.util
import json
import os
import stat
import subprocess
import sys
import time
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parent.parent / "integration" / "sml_jit_cache.py"
spec = importlib.util.spec_from_file_location("sml_jit_cache", SCRIPT)
sjc = importlib.util.module_from_spec(spec)
sys.modules["sml_jit_cache"] = sjc  # dataclasses resolve their module
spec.loader.exec_module(sjc)


def fake_sqsh(p: Path, salt: bytes = b"") -> Path:
    p.write_bytes(b"hsqs" + (b"\x01" * 4) + salt.ljust(88, b"\0") + b"payload" * 100)
    return p


def test_image_key_superblock_sidecar_and_ref(tmp_path):
    a = fake_sqsh(tmp_path / "a.sqsh")
    k1 = sjc.image_key(str(a))
    assert k1.startswith("sqsh-") and k1 == sjc.image_key(str(a))
    fake_sqsh(a, b"rebuilt")                        # rebuilt in place, same path
    assert sjc.image_key(str(a)) != k1
    (tmp_path / "a.sqsh.digest").write_text("sha256:0123456789abcdef0123456789abcdef\n")
    assert sjc.image_key(str(a)) == "sha256-0123456789abcdef"   # CI sidecar wins
    assert sjc.image_key("lmsysorg/sglang:v0.5.20").startswith("ref-")
    assert sjc.image_key("x/y@sha256:abcdef0123456789ff") == "sha256-abcdef0123456789"


def make_overlay(local: Path, n_dirs=5, n_files=3, tag=""):
    for i in range(n_dirs):                          # Triton-like: one dir per kernel
        d = local / "triton" / f"H{i}"
        d.mkdir(parents=True, exist_ok=True)
        (d / "k.cubin").write_bytes(b"\x7fELF" + bytes(4000))
        (d / f"__grp__k.json").write_text(json.dumps({"child_paths": {"k.cubin": str(d / "k.cubin")}}))
    (local / "inductor" / "locks").mkdir(parents=True, exist_ok=True)
    (local / "inductor" / "locks" / "x.lock").write_text("")
    for j in range(n_files):
        (local / "inductor" / f"m{j}.so").write_bytes(b"\x7fELF" + bytes(100))
    (local / "triton" / "H0" / "tmp.pid_1_abc").mkdir(exist_ok=True)
    (local / "vllm").mkdir(exist_ok=True)
    (local / "vllm" / "gpu_p2p_access_cache_for_0,1.json").write_text("{}")
    autotune = local / "sglang" / "flashinfer" / "autotune" / "rank_0.json"
    autotune.parent.mkdir(parents=True, exist_ok=True)
    autotune.write_text("{}")
    os.chmod(autotune, 0o600)                        # FlashInfer's mkstemp mode


def test_promote_then_seed_round_trip(tmp_path):
    shared, local, local2 = tmp_path / "shared", tmp_path / "o1", tmp_path / "o2"
    shared.mkdir()
    assert sjc.seed(shared, local)["files"] == 0
    make_overlay(local)
    r = sjc.promote(shared, local)
    assert r["errors"] == 0 and r["lost_race"] == 0 and r["promoted"] > 0
    assert (shared / "triton" / "H3" / "k.cubin").exists()
    assert not list(shared.rglob("*.lock")) and not list(shared.rglob("tmp.pid_*"))
    assert not list(shared.rglob("gpu_p2p_access_cache_for_*"))
    assert not (shared / ".staging").exists() or not any((shared / ".staging").iterdir())
    m = stat.S_IMODE((shared / "sglang/flashinfer/autotune/rank_0.json").stat().st_mode)
    assert m & 0o044 == 0o044                        # 0600 source made group/other readable
    # a new job seeds everything back at the same relative paths
    s = sjc.seed(shared, local2)
    assert s["errors"] == 0 and (local2 / "triton/H4/__grp__k.json").exists()
    assert (shared / ".meta" / "last_used").exists()
    # nothing new: promote is a no-op; a modified seeded file is never pushed
    (local2 / "inductor" / "m0.so").write_bytes(b"changed")
    r2 = sjc.promote(shared, local2)
    assert r2["promoted"] == 0 and r2["skipped_modified"] == 1


def test_concurrent_promoters_first_writer_wins(tmp_path):
    shared = tmp_path / "shared"
    shared.mkdir()
    n = 16
    overlays = []
    for i in range(n):
        o = tmp_path / f"o{i}"
        make_overlay(o, n_dirs=40, n_files=20)
        overlays.append(o)
    results = run_promoters(shared, overlays)
    assert all(r["errors"] == 0 for r in results)
    # into an empty cache each component dir is one atomic unit: published once, the rest lose
    winners = sum(r["promoted"] for r in results)
    assert winners == 3                              # triton/, inductor/, sglang/ (vllm/ holds only P2P state)
    assert sum(r["lost_race"] for r in results) <= (n - 1) * winners   # late starters see it exists
    # load every entry: all complete, no partial or temp files
    bad = 0
    for i in range(40):
        d = shared / "triton" / f"H{i}"
        grp = json.loads((d / "__grp__k.json").read_text())
        bad += (d / "k.cubin").stat().st_size != 4004 or "k.cubin" not in grp["child_paths"]
    for j in range(20):
        bad += (shared / "inductor" / f"m{j}.so").read_bytes() != b"\x7fELF" + bytes(100)
    assert bad == 0
    assert not [p for p in shared.rglob("*") if ".staging" in p.parts and p.is_file()]


def run_promoters(shared, overlays, settle="0"):
    procs = [subprocess.Popen([sys.executable, str(SCRIPT), "promote", "--shared", str(shared),
                               "--local", str(o), "--settle", settle], stdout=subprocess.PIPE, text=True)
             for o in overlays]
    return [json.loads(p.communicate()[0])["promote"] for p in procs]


def test_concurrent_promoters_with_disjoint_entries_all_land(tmp_path):
    """8 jobs each built different kernels into the same component dir: every entry must land,
    including those of jobs whose whole-directory rename lost the race (retried one level down)."""
    shared = tmp_path / "shared"
    shared.mkdir()
    overlays = []
    for i in range(8):
        o = tmp_path / f"o{i}"
        for k in range(10):
            d = o / "triton" / f"J{i}K{k}"
            d.mkdir(parents=True)
            (d / "k.cubin").write_bytes(bytes([i, k]) * 500)
        overlays.append(o)
    results = run_promoters(shared, overlays)
    assert all(r["errors"] == 0 for r in results)
    landed = sorted(p.name for p in (shared / "triton").iterdir())
    assert len(landed) == 80
    for i in range(8):
        for k in range(10):
            assert (shared / "triton" / f"J{i}K{k}" / "k.cubin").read_bytes() == bytes([i, k]) * 500


def test_settle_window_defers_fresh_builds(tmp_path):
    shared, local = tmp_path / "shared", tmp_path / "o"
    shared.mkdir()
    (local / "flashinfer" / "op").mkdir(parents=True)
    (local / "flashinfer" / "op" / "op.so").write_bytes(b"partial")   # just written: maybe in progress
    r = sjc.promote(shared, local, settle=60)
    assert r["promoted"] == 0 and r["unsettled"] == 1
    old = time.time() - 120
    os.utime(local / "flashinfer" / "op" / "op.so", (old, old))
    assert sjc.promote(shared, local, settle=60)["promoted"] == 1


def test_gc_lru_and_age(tmp_path):
    root = tmp_path / "root"
    for key, age_days, size in [("old", 40, 1000), ("mid", 5, 3000), ("new", 0, 3000)]:
        d = root / key / "sm90" / "triton"
        d.mkdir(parents=True)
        (d / "x").write_bytes(bytes(size))
        lu = root / key / "sm90" / ".meta" / "last_used"
        lu.parent.mkdir(parents=True)
        lu.touch()
        t = time.time() - age_days * 86400
        os.utime(lu, (t, t))
    dry = sjc.gc(root, max_bytes=10**9, max_age_days=30, dry_run=True)
    assert [Path(e["path"]).parent.name for e in dry] == ["old"] and (root / "old").exists()
    ev = sjc.gc(root, max_bytes=3500, max_age_days=30)
    assert [Path(e["path"]).parent.name for e in ev] == ["old", "mid"]   # age first, then LRU to budget
    assert not (root / "mid" / "sm90").exists() and (root / "new" / "sm90").exists()
