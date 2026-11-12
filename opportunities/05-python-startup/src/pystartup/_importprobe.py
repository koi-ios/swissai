"""Child side of importprof: import targets in this fresh interpreter and report what it cost.

Run as:  python [-X importtime] -m pystartup._importprobe --out OUT.json --targets a,b,c
         [--fake-cuda] [--audit-pyc]

Instruments the frozen importlib (the loaders the import system really uses) to split the
import time into: path search (FileFinder.find_spec), file reads (FileLoader.get_data),
unmarshal of pycs (_compile_bytecode), compiles from source (SourceLoader.source_to_code),
extension-module loading (ExtensionFileLoader.create_module, i.e. dlopen + PyInit) and the
remainder (executing module bodies). Every compile is also an audit event, so the count of
modules compiled at import is exact.
"""
import sys
import time

T0 = time.perf_counter()
T0_WALL = time.time()

import _frozen_importlib_external as _bex  # noqa: E402
import os  # noqa: E402  (frozen, already imported by site)

STATS = {
    "find": [0, 0.0],        # calls, seconds
    "read": [0, 0.0, 0],     # calls, seconds, bytes
    "unmarshal": [0, 0.0, 0],
    "compile": [0, 0.0, 0],  # calls, seconds, source bytes
    "ext": [0, 0.0],
}
COMPILED = []
EXT_FILES = []
OPENED = {}
ORDER = []          # every file touched, first access first (sources, pycs, .so incl. DT_NEEDED libs)
_SEEN = set()
_MAPS_SEEN = set()


def _note(path):
    if path not in _SEEN:
        _SEEN.add(path)
        ORDER.append(path)


def _scan_maps():
    try:
        with open("/proc/self/maps") as f:
            for line in f:
                parts = line.split()
                if len(parts) >= 6 and ".so" in parts[5] and parts[5] not in _MAPS_SEEN:
                    _MAPS_SEEN.add(parts[5])
                    _note(parts[5])
    except OSError:
        pass


def _wrap_method(cls, name, key, count_bytes=None):
    orig = getattr(cls, name)

    def wrapper(self, *a, **k):
        t = time.perf_counter()
        try:
            r = orig(self, *a, **k)
        finally:
            s = STATS[key]
            s[0] += 1
            s[1] += time.perf_counter() - t
        if count_bytes is not None:
            STATS[key][2] += count_bytes(r, a)
        return r

    wrapper.__wrapped__ = orig
    setattr(cls, name, wrapper)


def _install():
    _wrap_method(_bex.FileFinder, "find_spec", "find")
    _wrap_method(_bex.FileLoader, "get_data", "read",
                 count_bytes=lambda r, a: len(r) if isinstance(r, (bytes, bytearray)) else 0)

    orig_s2c = _bex.SourceLoader.source_to_code

    def source_to_code(self, data, path, *, _optimize=-1):
        t = time.perf_counter()
        try:
            return orig_s2c(self, data, path, _optimize=_optimize)
        finally:
            s = STATS["compile"]
            s[0] += 1
            s[1] += time.perf_counter() - t
            s[2] += len(data)
            COMPILED.append(path)

    _bex.SourceLoader.source_to_code = source_to_code

    orig_cb = _bex._compile_bytecode

    def _compile_bytecode(data, name=None, bytecode_path=None, source_path=None):
        t = time.perf_counter()
        try:
            return orig_cb(data, name, bytecode_path, source_path)
        finally:
            s = STATS["unmarshal"]
            s[0] += 1
            s[1] += time.perf_counter() - t
            s[2] += len(data)

    _bex._compile_bytecode = _compile_bytecode

    orig_cm = _bex.ExtensionFileLoader.create_module

    def create_module(self, spec):
        t = time.perf_counter()
        try:
            return orig_cm(self, spec)
        finally:
            s = STATS["ext"]
            s[0] += 1
            s[1] += time.perf_counter() - t
            EXT_FILES.append(getattr(spec, "origin", None))
            if DUMP_FILES:
                _scan_maps()

    _bex.ExtensionFileLoader.create_module = create_module

    def audit(event, args):
        if event == "open":
            path = args[0]
            if isinstance(path, str):
                mode = args[1]
                if mode is None or "r" in mode or (isinstance(args[2], int) and (args[2] & 3) == 0):
                    OPENED[path] = OPENED.get(path, 0) + 1
                    if DUMP_FILES:
                        _note(path)

    sys.addaudithook(audit)


def _proc_io():
    out = {}
    try:
        with open("/proc/self/io") as f:
            for line in f:
                k, v = line.split(":")
                out[k.strip()] = int(v)
    except OSError:
        pass
    return out


def _maps():
    sos = {}
    try:
        with open("/proc/self/maps") as f:
            for line in f:
                parts = line.split()
                if len(parts) >= 6 and ".so" in parts[5]:
                    sos[parts[5]] = True
    except OSError:
        return None
    total = 0
    for p in sos:
        try:
            total += os.path.getsize(p)
        except OSError:
            pass
    return {"so_files": len(sos), "so_bytes": total}


def _rss_kb():
    try:
        with open("/proc/self/status") as f:
            for line in f:
                if line.startswith("VmRSS:"):
                    return int(line.split()[1])
    except OSError:
        pass
    try:
        import resource

        r = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        return r // 1024 if sys.platform == "darwin" else r
    except Exception:
        return None


DUMP_FILES = "--dump-files" in sys.argv


def main():
    args = sys.argv[1:]
    out = args[args.index("--out") + 1]
    targets = [t for t in args[args.index("--targets") + 1].split(",") if t]
    fake_cuda = "--fake-cuda" in args
    audit_pyc = "--audit-pyc" in args
    io0 = _proc_io()
    n0 = len(sys.modules)
    before = set(sys.modules)
    _install()
    guard = None
    res = {"targets": targets, "ok": {}, "errors": {}, "python": sys.version, "argv": sys.argv,
           "pycache_prefix": sys.pycache_prefix, "dont_write_bytecode": sys.dont_write_bytecode,
           "t0_wall": T0_WALL}
    t_import = time.perf_counter()
    if fake_cuda:
        # Emulate a GPU node for import purposes (CPU containers): serve facts from
        # PYSTARTUP_FAKE_CUDA, trip on anything else. Trips are recorded, not fatal.
        from pystartup.cudafacts import CudaGuard, CudaTouch, probe_facts

        import torch  # noqa: F401

        guard = CudaGuard(probe_facts()).arm()
    for t in targets:
        if guard is not None:
            guard.current_module = t
        t1 = time.perf_counter()
        try:
            __import__(t)
            res["ok"][t] = round(time.perf_counter() - t1, 4)
        except BaseException as e:  # noqa: BLE001 - CudaTouch is a BaseException
            res["ok"][t] = None
            res["errors"][t] = repr(e)[:500]
    t_done = time.perf_counter()
    if guard is not None:
        guard.disarm()
        res["cuda_trips"] = guard.trips
    res["import_s"] = round(t_done - t_import, 4)
    res["since_probe_start_s"] = round(t_done - T0, 4)
    res["breakdown_s"] = {k: round(v[1], 4) for k, v in STATS.items()}
    res["counts"] = {k: v[0] for k, v in STATS.items()}
    res["bytes"] = {"read": STATS["read"][2], "unmarshal": STATS["unmarshal"][2],
                    "compiled_source": STATS["compile"][2]}
    accounted = sum(v[1] for k, v in STATS.items() if k != "find") + STATS["find"][1]
    res["breakdown_s"]["exec_and_other"] = round(res["import_s"] - accounted, 4)
    new = [m for m in sys.modules if m not in before]
    res["modules_before"] = n0
    res["modules_new"] = len(new)
    files = []
    for m in new:
        f = getattr(sys.modules.get(m), "__file__", None)
        if isinstance(f, str):
            files.append(f)
    res["module_files_new"] = len(files)
    top = {}
    for m in new:
        top[m.split(".")[0]] = top.get(m.split(".")[0], 0) + 1
    res["modules_by_top_package"] = dict(sorted(top.items(), key=lambda kv: -kv[1])[:40])
    res["compiled_files"] = len(COMPILED)
    res["compiled_examples"] = COMPILED[:20]
    res["ext_modules"] = len(EXT_FILES)
    uniq_bytes = 0
    for p in OPENED:
        try:
            uniq_bytes += os.path.getsize(p)
        except OSError:
            pass
    res["opened_files"] = {"unique": len(OPENED), "bytes": uniq_bytes}
    io1 = _proc_io()
    if io0 and io1:
        res["proc_io_delta"] = {k: io1[k] - io0.get(k, 0) for k in io1}
    res["maps"] = _maps()
    res["rss_kb"] = _rss_kb()
    if audit_pyc:
        from pystartup.pycheck import audit_modules

        a = audit_modules(files, prefix=sys.pycache_prefix)
        res["pyc_audit"] = {"counts": a.counts, "source_bytes": a.source_bytes,
                            "needs_compile": a.needs_compile, "needs_compile_bytes": a.needs_compile_bytes,
                            "py_files": a.py_files, "py_bytes": a.py_bytes}
    if "--dump-modules" in args:
        res["module_order"] = new
    if DUMP_FILES:
        _scan_maps()
        res["files_in_order"] = [p for p in ORDER if os.path.isfile(p)]
    import json

    with open(out, "w") as f:
        json.dump(res, f, indent=1)


if __name__ == "__main__":
    main()
