"""Process-tree import tracer for an engine launch (opt-in, standard library only).

Put this directory first on PYTHONPATH and set PYSTARTUP_TRACE_DIR. Every Python process of
the launch (main, schedulers, detokenizer, resource tracker, forkserver, its children) then
writes <dir>/proc-<pid>.json with: pid, ppid, cmdline, wall time at interpreter start
(from /proc), when site ran, how many modules it loaded and when the first/last load
happened, imports per top-level package and, if PYSTARTUP_TRACE_MODULES=1, the full ordered
module list with load times in ms after site ran (the input for zygote calibration).

It uses an audit hook on the "import" event (raised once per module actually loaded), keeps
no thread (a thread in a forkserver would break the zygote's single-thread invariant) and
rewrites its file at most every 0.5 s plus at exit. A forked child detects the pid change,
starts its own record and notes the parent it was forked from.

This file deliberately does nothing when PYSTARTUP_TRACE_DIR is unset.
"""
import os
import sys
import time

_DIR = os.environ.get("PYSTARTUP_TRACE_DIR")


def _proc_start_wall(pid):
    """Process start time as epoch seconds (Linux), from /proc/<pid>/stat and btime."""
    try:
        with open(f"/proc/{pid}/stat") as f:
            stat = f.read()
        fields = stat[stat.rindex(")") + 2:].split()
        start_ticks = int(fields[19])
        with open("/proc/stat") as f:
            btime = next(int(l.split()[1]) for l in f if l.startswith("btime"))
        return btime + start_ticks / os.sysconf("SC_CLK_TCK")
    except Exception:
        return None


def _cmdline(pid):
    try:
        with open(f"/proc/{pid}/cmdline", "rb") as f:
            return [a.decode(errors="replace") for a in f.read().split(b"\0") if a]
    except OSError:
        return list(getattr(sys, "orig_argv", []) or [])


if _DIR:
    import json

    _KEEP_MODULES = os.environ.get("PYSTARTUP_TRACE_MODULES") == "1"

    class _State:
        def __init__(self, forked_from=None):
            self.pid = os.getpid()
            self.rec = {
                "pid": self.pid, "ppid": os.getppid(), "cmdline": _cmdline(self.pid),
                "proc_start_wall": _proc_start_wall(self.pid), "t_site": time.time(),
                "forked_from": forked_from, "imports": 0, "t_first_import": None,
                "t_last_import": None, "modules_at_start": len(sys.modules), "by_package": {},
            }
            if _KEEP_MODULES:
                self.rec["modules"] = []
            self.last_write = 0.0

        def write(self, force=False):
            now = time.time()
            if not force and now - self.last_write < 0.5:
                return
            self.last_write = now
            self.rec["t_written"] = now
            self.rec["modules_now"] = len(sys.modules)
            path = os.path.join(_DIR, f"proc-{self.pid}.json")
            tmp = f"{path}.tmp"
            try:
                with open(tmp, "w") as f:
                    json.dump(self.rec, f)
                os.replace(tmp, path)
            except OSError:
                pass

    _S = [_State()]
    _IN_HOOK = [False]
    _FINALIZER = [False]

    def _hook(event, args):
        if event != "import" or _IN_HOOK[0]:
            return
        _IN_HOOK[0] = True
        try:
            st = _S[0]
            if os.getpid() != st.pid:  # forked: start a fresh record for this process
                st = _State(forked_from=st.pid)
                _S[0] = st
                _FINALIZER[0] = False
            now = time.time()
            r = st.rec
            r["imports"] += 1
            if r["t_first_import"] is None:
                r["t_first_import"] = now
            r["t_last_import"] = now
            name = args[0] if args else "?"
            top = str(name).split(".")[0]
            r["by_package"][top] = r["by_package"].get(top, 0) + 1
            if _KEEP_MODULES:
                r["modules"].append([round((now - r["t_site"]) * 1000, 1), name])
            if not _FINALIZER[0] and "multiprocessing.util" in sys.modules:
                # forkserver/fork children leave through os._exit(): atexit never runs there,
                # multiprocessing's own exit function does.
                _FINALIZER[0] = True
                try:
                    sys.modules["multiprocessing.util"].Finalize(None, _at_exit, exitpriority=100)
                except Exception:
                    pass
            st.write()
        finally:
            _IN_HOOK[0] = False

    def _at_exit():
        st = _S[0]
        if os.getpid() == st.pid:
            st.write(force=True)

    def _after_fork_in_child():
        parent = _S[0].pid
        _S[0] = _State(forked_from=parent)
        _FINALIZER[0] = False
        mpu = sys.modules.get("multiprocessing.util")
        if mpu is not None:
            try:
                mpu.Finalize(None, _at_exit, exitpriority=100)
                _FINALIZER[0] = True
            except Exception:
                pass
        _S[0].write(force=True)

    try:
        os.makedirs(_DIR, exist_ok=True)
        import atexit

        atexit.register(_at_exit)
        sys.addaudithook(_hook)
        os.register_at_fork(after_in_child=_after_fork_in_child)
        _S[0].write(force=True)
    except Exception:
        pass
