import os
import py_compile

from pystartup import pycheck


def _mk(tmp_path, body="X = 1\n"):
    src = tmp_path / "m.py"
    src.write_text(body)
    return str(src)


def test_missing(tmp_path):
    assert pycheck.classify(_mk(tmp_path)) == pycheck.MISSING


def test_timestamp_valid_then_stale_on_mtime_change(tmp_path):
    src = _mk(tmp_path)
    py_compile.compile(src, invalidation_mode=py_compile.PycInvalidationMode.TIMESTAMP)
    assert pycheck.classify(src) == pycheck.VALID_TIMESTAMP
    st = os.stat(src)
    os.utime(src, (st.st_atime, st.st_mtime + 10))  # e.g. a converter that rewrites mtimes
    assert pycheck.classify(src) == pycheck.STALE


def test_unchecked_hash_survives_mtime_and_content_change(tmp_path):
    src = _mk(tmp_path)
    py_compile.compile(src, invalidation_mode=py_compile.PycInvalidationMode.UNCHECKED_HASH)
    st = os.stat(src)
    os.utime(src, (st.st_atime, st.st_mtime + 1000))
    assert pycheck.classify(src) == pycheck.VALID_UNCHECKED_HASH


def test_checked_hash_tracks_content(tmp_path):
    src = _mk(tmp_path)
    py_compile.compile(src, invalidation_mode=py_compile.PycInvalidationMode.CHECKED_HASH)
    st = os.stat(src)
    os.utime(src, (st.st_atime, st.st_mtime + 1000))
    assert pycheck.classify(src) == pycheck.VALID_CHECKED_HASH  # mtime alone does not matter
    with open(src, "a") as f:
        f.write("Y = 2\n")
    assert pycheck.classify(src) == pycheck.STALE


def test_bad_magic(tmp_path):
    src = _mk(tmp_path)
    pyc = py_compile.compile(src)
    data = bytearray(open(pyc, "rb").read())
    data[0] ^= 0xFF
    open(pyc, "wb").write(bytes(data))
    assert pycheck.classify(src) == pycheck.BAD_MAGIC


def test_prefix_layout(tmp_path):
    src = _mk(tmp_path)
    p = pycheck.pyc_path_for(src, prefix="/pfx")
    assert p.startswith("/pfx" + os.sep)
    assert p.endswith(f"m.{pycheck.CACHE_TAG}.pyc")
    assert os.path.abspath(str(tmp_path)).lstrip(os.sep) in p


def test_audit_counts(tmp_path):
    a = tmp_path / "a.py"
    b = tmp_path / "b.py"
    a.write_text("A = 1\n")
    b.write_text("B = 2\n")
    py_compile.compile(str(a))
    res = pycheck.audit([str(tmp_path)])
    assert res.py_files == 2
    assert res.counts[pycheck.VALID_TIMESTAMP] == 1
    assert res.counts[pycheck.MISSING] == 1
    assert res.needs_compile == 1
