"""Unit and smoke tests for bench_storage.py (small files in a temp dir, ~15 s)."""
import json
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import bench_storage as bs  # noqa: E402

MB = 1 << 20


@pytest.fixture()
def files(tmp_path):
    out = []
    for i, size in enumerate((5 * MB + 123, 3 * MB, 7 * MB)):
        p = tmp_path / f"f{i}.bin"
        p.write_bytes(os.urandom(size))
        out.append((str(p), size))
    return out


def test_parse_size():
    assert bs.parse_size("16M") == 16 * MB
    assert bs.parse_size("1G") == 1 << 30
    assert bs.parse_size("4KiB") == 4096
    assert bs.parse_size("512") == 512
    assert bs.fmt_size(16 * MB) == "16M"


@pytest.mark.parametrize("readers", [1, 3, 8])
@pytest.mark.parametrize("block", [MB, 4 * MB])
def test_plan_covers_bytes_once(files, readers, block):
    total = sum(s for _, s in files)
    for cursor in (0, 4 * MB + 17, total - 1, total + 2 * MB):
        groups, got = bs.plan_run(files, cursor, 9 * MB, block, readers)
        segs = [s for g in groups for s in g]
        assert got == sum(s[2] for s in segs) == 9 * MB
        assert 1 <= len(groups) <= readers
        seen = set()
        for path, off, length in segs:
            assert off % block == 0 and length > 0
            assert off + length <= dict(files)[path]
            key = (path, off)
            assert key not in seen
            seen.add(key)


def test_plan_caps_at_total(files):
    total = sum(s for _, s in files)
    _, got = bs.plan_run(files, 0, 10 * total, MB, 4)
    assert got == total


@pytest.mark.parametrize("method", ["mmap", "pread", "odirect", "mmap-stride"])
@pytest.mark.parametrize("readers", [1, 2])
def test_methods_read_everything(files, method, readers):
    if method == "mmap-stride":
        pytest.importorskip("numpy")
    groups, got = bs.plan_run(files, 0, 8 * MB, MB, readers)
    t = bs.run_once(method, groups, MB, {"madvise": None, "stride": 4096, "chunk": 128})
    if method == "mmap-stride":  # whole rows only, counted per segment
        assert t["bytes_read"] == sum(s[2] // 4096 for g in groups for s in g) * 128
    else:
        assert t["bytes_read"] == got
    assert t["seconds"] > 0 and t["reader_min_s"] <= t["reader_max_s"]


def test_resident_fraction_in_range(files):
    segs = [(p, 0, s) for p, s in files]
    r = bs.resident_fraction(segs)
    assert r is None or 0.0 <= r <= 1.0


def test_drop_cache_does_not_fail(files):
    for p, _ in files:
        assert bs.drop_cache(p) in (True, False)


def test_cli_end_to_end(files, tmp_path):
    out = tmp_path / "res.jsonl"
    rc = bs.main([os.path.dirname(files[0][0]), "--glob", "f*.bin", "--readers", "1,2",
                  "--block-sizes", "1M", "--bytes-per-run", "4M", "--repeats", "3",
                  "--methods", "mmap,pread", "--out", str(out)])
    assert rc == 0
    lines = [json.loads(x) for x in out.read_text().splitlines()]
    assert "env" in lines[0]
    rows = lines[1:]
    assert len(rows) == 2 * 2 * 3
    assert all(r["gbps"] > 0 for r in rows)
    assert (tmp_path / "res.jsonl.summary.md").exists()


def test_dry_run_reads_nothing(files, capsys):
    rc = bs.main([files[0][0], "--dry-run"])
    assert rc == 0
    assert "runs" in capsys.readouterr().out


def test_create_refuses_nonempty_dir(tmp_path):
    (tmp_path / "x").write_text("keep me")
    with pytest.raises(SystemExit):
        bs.create_files(str(tmp_path), 2 * MB, 2)
    assert (tmp_path / "x").read_text() == "keep me"
