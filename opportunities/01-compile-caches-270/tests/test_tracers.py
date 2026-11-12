"""Tracer unit tests: strace log parsing, snapshot diff, audit-hook attribution, inotify (Linux)."""

import os
import subprocess
import sys
import textwrap
import time

import pytest

from cache_audit.tracers import audithook, snapshot
from cache_audit.tracers import inotify as inotify_tracer
from cache_audit.tracers import strace as strace_tracer

STRACE_LOG = textwrap.dedent('''\
    4100 execve("/usr/bin/python3", ["python3", "x.py"], 0x7ffd5bd8 /* 30 vars */) = 0
    4100 openat(AT_FDCWD</work>, "/usr/lib/python3/os.py", O_RDONLY|O_CLOEXEC) = 3</usr/lib/python3/os.py>
    4100 mkdirat(AT_FDCWD</work>, "/home/u/.triton/cache/AB/tmp.pid_4100_x", 0777) = 0
    4100 openat(AT_FDCWD</work>, "/home/u/.triton/cache/AB/tmp.pid_4100_x/k.cubin", O_WRONLY|O_CREAT|O_TRUNC|O_CLOEXEC, 0666) = 3</home/u/.triton/cache/AB/tmp.pid_4100_x/k.cubin>
    4100 renameat2(AT_FDCWD</work>, "/home/u/.triton/cache/AB/tmp.pid_4100_x/k.cubin", AT_FDCWD</work>, "/home/u/.triton/cache/AB/k.cubin", 0) = 0
    4100 clone3({flags=CLONE_VM|CLONE_VFORK, exit_signal=SIGCHLD, stack=0xffff, stack_size=0x9000}, 88) = 4101
    4101 execve("/usr/bin/ninja", ["ninja", "-v"], 0xffffd3a0 /* 31 vars */) = 0
    4101 chdir("/home/u/.cache/tvm-ffi/m_1") = 0
    4101 openat(AT_FDCWD, "main.o", O_WRONLY|O_CREAT|O_TRUNC, 0666 <unfinished ...>
    4100 openat(AT_FDCWD, "/ro/cache/x.lock", O_RDWR|O_CREAT, 0644) = -1 EROFS (Read-only file system)
    4101 <... openat resumed>) = 4</home/u/.cache/tvm-ffi/m_1/main.o>
    4101 unlinkat(AT_FDCWD, "stale.tmp", 0) = 0
    4101 renameat(AT_FDCWD, "m.so.tmp", AT_FDCWD, "m.so") = 0
    4101 +++ exited with 0 +++
''')


def test_strace_parse_writes_failures_and_cwd():
    ev, procs = strace_tracer.parse(STRACE_LOG, "/work")
    by = {(e.op, e.path) for e in ev}
    assert ("open_w", "/home/u/.triton/cache/AB/tmp.pid_4100_x/k.cubin") in by
    assert ("rename_to", "/home/u/.triton/cache/AB/k.cubin") in by
    assert ("mkdir", "/home/u/.triton/cache/AB/tmp.pid_4100_x") in by
    assert ("open_w", "/home/u/.cache/tvm-ffi/m_1/main.o") in by      # stitched unfinished/resumed
    assert ("delete", "/home/u/.cache/tvm-ffi/m_1/stale.tmp") in by    # relative path via tracked chdir
    assert ("rename_to", "/home/u/.cache/tvm-ffi/m_1/m.so") in by
    assert ("write_failed", "/ro/cache/x.lock") in by
    assert not any(e.path.endswith("os.py") for e in ev)               # read-only opens ignored
    ninja = [e for e in ev if e.path.endswith("main.o")][0]
    assert ninja.pid == 4101 and "/usr/bin/ninja" in ninja.writer
    assert procs[4101].startswith("/usr/bin/ninja")
    failed = [e for e in ev if e.op == "write_failed"][0]
    assert failed.detail == "EROFS"


def test_strace_cstring_escapes():
    assert strace_tracer._cstring(r'"a\nb\x41\101\""') == 'a\nbAA"'
    assert strace_tracer._split_args('AT_FDCWD</a, b>, "x, y", {a=1, b=2}, 3') == [
        "AT_FDCWD</a, b>", '"x, y"', "{a=1, b=2}", "3"]


def test_snapshot_diff(tmp_path):
    (tmp_path / "keep").write_text("a")
    (tmp_path / "mod").write_text("a")
    (tmp_path / "gone").write_text("a")
    before = snapshot.take([str(tmp_path)])
    time.sleep(0.01)
    (tmp_path / "mod").write_text("bb")
    (tmp_path / "gone").unlink()
    (tmp_path / "sub").mkdir()
    (tmp_path / "sub" / "new").write_text("c")
    ev = {(e.op, os.path.basename(e.path)) for e in snapshot.diff(before, snapshot.take([str(tmp_path)]))}
    assert ev == {("modify", "mod"), ("delete", "gone"), ("create", "new")}


def test_audithook_attributes_python_writes(tmp_path):
    hook = audithook.install(tmp_path / "hook")
    log = tmp_path / "log"
    script = tmp_path / "writer.py"
    target = tmp_path / "cache" / "entry.bin"
    script.write_text(textwrap.dedent(f'''
        import os
        os.makedirs({str(target.parent)!r}, exist_ok=True)
        tmp = {str(target)!r} + ".tmp"
        with open(tmp, "wb") as f:
            f.write(b"x")
        os.replace(tmp, {str(target)!r})
        import subprocess, sys
        subprocess.run([sys.executable, "-c", "open({str(target)!r} + '.child', 'w').write('c')"])
    '''))
    env = audithook.env_for(dict(os.environ), hook, str(log))
    subprocess.run([sys.executable, str(script)], env=env, check=True)
    events, procs = audithook.collect(str(log))
    ops = {(e.op, os.path.basename(e.path)) for e in events}
    assert ("open_w", "entry.bin.tmp") in ops
    assert ("rename_to", "entry.bin") in ops
    assert ("open_w", "entry.bin.child") in ops                  # child interpreter recorded too
    w = [e for e in events if e.op == "rename_to"][0]
    assert w.writer.startswith(str(script)) and w.detail.endswith("entry.bin.tmp")
    assert any(sys.executable in " ".join(p["args"]) for p in procs)


@pytest.mark.skipif(not inotify_tracer.available(), reason="inotify is Linux-only")
def test_inotify_recursive(tmp_path):
    w = inotify_tracer.Watcher([str(tmp_path)]).start()
    (tmp_path / "a" / "b").mkdir(parents=True)
    (tmp_path / "a" / "b" / "f").write_text("x")
    os.replace(tmp_path / "a" / "b" / "f", tmp_path / "a" / "g")
    time.sleep(0.5)
    ev = w.stop()
    paths = {(e.op, os.path.relpath(e.path, tmp_path)) for e in ev}
    assert ("mkdir", "a") in paths
    assert any(p == "a/b/f" for _, p in paths)
    assert ("rename_to", "a/g") in paths


@pytest.mark.skipif(not strace_tracer.available(), reason="strace not installed or ptrace blocked")
def test_strace_live(tmp_path):
    log = tmp_path / "s.log"
    target = tmp_path / "x" / "y.bin"
    code = f"import os; os.makedirs({str(target.parent)!r}); open({str(target)!r},'wb').write(b'1')"
    subprocess.run(strace_tracer.command(str(log), [sys.executable, "-c", code]), check=True)
    ev, _ = strace_tracer.parse(log.read_text(), str(tmp_path))
    assert ("open_w", str(target)) in {(e.op, e.path) for e in ev}


def test_classify_temp_names_and_binaries(tmp_path):
    from cache_audit.classify import classify
    assert classify("/t/tmpgwx_29sb.so") == "temp"        # tempfile name with an underscore
    assert classify("/t/tmp3fbb1dkg.cpp") == "temp"
    assert classify("/t/tmp_dir/k.so") == "binary"        # a dir called tmp_dir is not a temp name
    assert classify("/c/triton/AB/__grp__k.json") == "metadata"
    assert classify("/c/inductor/xx/c.best_config") == "autotune"
    assert classify("/c/build.ninja") == "build"
    elf = tmp_path / "kernel"
    elf.write_bytes(b"\x7fELF" + bytes(10))
    assert classify(str(elf)) == "binary"
