"""Device facts without touching the CUDA driver, and a guard that serves them during preload.

Why: SGLang modules call `torch.cuda.is_available()` at import (95 files in v0.5.20 do
`_is_cuda = is_cuda()`), and `sgl_kernel` picks its sm90/sm100 build at import through
`torch.cuda.current_device()` + `get_device_properties()`. In a process that will later
fork TP workers, any of these calls runs `cuInit`, and a forked child can then no longer use
CUDA ("poisoned fork"; PyTorch docs, `multiprocessing-poison-fork-note`).

What this module does:

* `probe_facts()` runs NVML in a *separate short-lived process* (ctypes, no torch) and
  returns the visible devices' count, compute capability and name, honouring
  CUDA_VISIBLE_DEVICES. The calling process never opens a driver handle.
* `CudaGuard` replaces, while armed, the handful of `torch.cuda` entry points that only need
  those facts (is_available, device_count, current_device, get_device_capability,
  get_device_name, get_device_properties().major/.minor/.name). Every other way into the
  driver (`torch.cuda._lazy_init`, `torch._C._cuda_getDeviceCount`, `torch._C._cuda_init`,
  any other attribute of the properties object) raises `CudaTouch`.
* `CudaTouch` derives from BaseException so `except Exception:` in library code cannot
  swallow it. A swallowed trip is still detected (the guard counts trips).
* Every replacement is self-disarming: once `disarm()` runs, a reference captured during
  preload delegates to the real function, so children see real behaviour.
* `verify_facts()` runs in each child right after its first real CUDA init
  (`torch.cuda._lazy_call`) and aborts the child if the facts served during preload differ
  from the device.

Standard library only (torch is imported lazily, by the guard).
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import traceback
from dataclasses import asdict, dataclass, field
from typing import Any, Callable, Dict, List, Optional


class CudaTouch(BaseException):
    """Code tried to reach the CUDA driver while the zygote guard was armed."""


ACTIVE = None  # the armed CudaGuard of this process, if any (an early hook may arm it first)


@dataclass
class CudaFacts:
    count: int
    major: int
    minor: int
    name: str
    source: str = "nvml-probe"
    uuids: List[str] = field(default_factory=list)

    def to_json(self) -> str:
        return json.dumps(asdict(self))

    @classmethod
    def from_json(cls, s: str) -> "CudaFacts":
        d = json.loads(s)
        return cls(**{k: d[k] for k in ("count", "major", "minor", "name", "source", "uuids") if k in d})


# Runs in a separate interpreter (-I -S): ctypes + NVML only, prints one JSON line.
_PROBE_CODE = r"""
import ctypes, json, sys
out = {"devices": []}
try:
    nvml = ctypes.CDLL("libnvidia-ml.so.1")
except OSError as e:
    print(json.dumps({"error": "libnvidia-ml.so.1: %s" % e})); sys.exit(0)
rc = nvml.nvmlInit_v2()
if rc != 0:
    print(json.dumps({"error": "nvmlInit_v2 rc=%d" % rc})); sys.exit(0)
try:
    n = ctypes.c_uint(0)
    rc = nvml.nvmlDeviceGetCount_v2(ctypes.byref(n))
    if rc != 0:
        raise RuntimeError("nvmlDeviceGetCount_v2 rc=%d" % rc)
    for i in range(n.value):
        h = ctypes.c_void_p()
        if nvml.nvmlDeviceGetHandleByIndex_v2(i, ctypes.byref(h)) != 0:
            raise RuntimeError("handle %d" % i)
        ma, mi = ctypes.c_int(), ctypes.c_int()
        if nvml.nvmlDeviceGetCudaComputeCapability(h, ctypes.byref(ma), ctypes.byref(mi)) != 0:
            raise RuntimeError("capability %d" % i)
        name = ctypes.create_string_buffer(96)
        nvml.nvmlDeviceGetName(h, name, 96)
        uuid = ctypes.create_string_buffer(96)
        nvml.nvmlDeviceGetUUID(h, uuid, 96)
        out["devices"].append({"index": i, "major": ma.value, "minor": mi.value,
                               "name": name.value.decode(errors="replace"),
                               "uuid": uuid.value.decode(errors="replace")})
except Exception as e:
    out = {"error": str(e)}
finally:
    nvml.nvmlShutdown()
print(json.dumps(out))
"""


def visible_devices(devices: List[dict], cvd: Optional[str]) -> Optional[List[dict]]:
    """Apply CUDA_VISIBLE_DEVICES the way the CUDA runtime does, or None if we cannot be sure.

    Integers and GPU-<uuid> prefixes are supported; parsing stops at the first invalid entry
    (CUDA semantics). MIG entries are not modelled -> None (the zygote then preloads no
    module that needs device facts).
    """
    if cvd is None:
        return list(devices)
    cvd = cvd.strip()
    if cvd == "":
        return []
    out: List[dict] = []
    for tok in cvd.split(","):
        tok = tok.strip()
        if tok.startswith("MIG-"):
            return None
        if tok.lstrip("-").isdigit():
            idx = int(tok)
            if idx < 0 or idx >= len(devices):
                break
            out.append(devices[idx])
        elif tok.startswith("GPU-"):
            matches = [d for d in devices if d.get("uuid", "").startswith(tok)]
            if len(matches) != 1:
                break
            out.append(matches[0])
        else:
            break
    return out


def facts_from_devices(devs: List[dict], source: str) -> Optional[CudaFacts]:
    if not devs:
        return None
    caps = {(d["major"], d["minor"]) for d in devs}
    names = {d["name"] for d in devs}
    if len(caps) != 1 or len(names) != 1:
        return None  # heterogeneous node: a single fake would be wrong for some device
    (ma, mi), = caps
    return CudaFacts(count=len(devs), major=ma, minor=mi, name=names.pop(), source=source,
                     uuids=[d.get("uuid", "") for d in devs])


def probe_facts(timeout: float = 20.0, python: Optional[str] = None,
                env: Optional[Dict[str, str]] = None) -> Optional[CudaFacts]:
    """Device facts from NVML, obtained in a child process. None when unavailable or unsure."""
    fake = os.environ.get("PYSTARTUP_FAKE_CUDA")
    if fake:
        # CPU-only emulation (import studies and tests). Never set this on a GPU node.
        d = json.loads(fake)
        d.setdefault("source", "env-fake")
        return CudaFacts(**d)
    try:
        r = subprocess.run([python or sys.executable, "-I", "-S", "-c", _PROBE_CODE],
                           capture_output=True, text=True, timeout=timeout, env=env)
        data = json.loads(r.stdout.strip().splitlines()[-1])
    except Exception:
        return None
    if "error" in data:
        return None
    devs = visible_devices(data["devices"], (env or os.environ).get("CUDA_VISIBLE_DEVICES"))
    if devs is None:
        return None
    return facts_from_devices(devs, "nvml-probe")


class _FakeProps:
    """Stand-in for torch.cuda._CudaDeviceProperties while armed.

    Only major / minor / name are served. Any other attribute trips while armed, and
    resolves to the real properties once disarmed (so captured objects heal in children).
    """

    __slots__ = ("_guard", "_index")

    def __init__(self, guard: "CudaGuard", index: int):
        object.__setattr__(self, "_guard", guard)
        object.__setattr__(self, "_index", index)

    def __getattr__(self, item):
        g = object.__getattribute__(self, "_guard")
        idx = object.__getattribute__(self, "_index")
        if g.armed:
            if item == "major":
                return g.facts.major
            if item == "minor":
                return g.facts.minor
            if item == "name":
                return g.facts.name
            g._trip(f"get_device_properties().{item}")
        return getattr(g._orig["get_device_properties"](idx), item)

    def __repr__(self):
        return f"_FakeProps(index={object.__getattribute__(self, '_index')})"


class CudaGuard:
    """Serve device facts to importing code and trip on anything else (see module doc)."""

    SERVED = ("is_available", "device_count", "current_device", "get_device_capability",
              "get_device_name", "get_device_properties")
    TRIPPED = ("_lazy_init",)
    TRIPPED_C = ("_cuda_getDeviceCount", "_cuda_init", "_cuda_getDevice", "_cuda_setDevice")

    def __init__(self, facts: Optional[CudaFacts]):
        self.facts = facts
        self.armed = False
        self.trips: List[dict] = []
        self.current_module: Optional[str] = None  # set by the preloader for attribution
        self._orig: Dict[str, Callable] = {}
        self._orig_c: Dict[str, Callable] = {}
        self._patched_refs: List[tuple] = []  # (module, attr, original)
        self._torch = None

    # -- tripping -------------------------------------------------------------------------
    def _trip(self, what: str):
        tb = traceback.extract_stack(limit=12)[:-2]
        frames = [f"{os.path.basename(f.filename)}:{f.lineno}:{f.name}" for f in tb[-6:]]
        self.trips.append({"what": what, "module": self.current_module, "stack": frames})
        raise CudaTouch(f"zygote guard: {what} while preloading {self.current_module}")

    def _device_index(self, device=None) -> int:
        if device is None:
            return 0
        if isinstance(device, int):
            return device
        idx = getattr(device, "index", None)
        if isinstance(device, str):
            idx = int(device.split(":")[1]) if ":" in device else 0
        if idx is None:
            idx = 0
        return int(idx)

    def _check_index(self, idx: int) -> int:
        if self.facts is None or not (0 <= idx < self.facts.count):
            self._trip(f"device index {idx} (facts: {self.facts.count if self.facts else 'none'})")
        return idx

    # -- served entry points --------------------------------------------------------------
    def _make_served(self):
        g = self

        def is_available():
            if not g.armed:
                return g._orig["is_available"]()
            if g.facts is None:
                g._trip("is_available()")
            return g.facts.count > 0

        def device_count():
            if not g.armed:
                return g._orig["device_count"]()
            if g.facts is None:
                g._trip("device_count()")
            return g.facts.count

        def current_device():
            if not g.armed:
                return g._orig["current_device"]()
            if g.facts is None or g.facts.count == 0:
                g._trip("current_device()")
            return 0

        def get_device_capability(device=None):
            if not g.armed:
                return g._orig["get_device_capability"](device)
            if g.facts is None:
                g._trip("get_device_capability()")
            g._check_index(g._device_index(device))
            return (g.facts.major, g.facts.minor)

        def get_device_name(device=None):
            if not g.armed:
                return g._orig["get_device_name"](device)
            if g.facts is None:
                g._trip("get_device_name()")
            g._check_index(g._device_index(device))
            return g.facts.name

        def get_device_properties(device=None):
            if not g.armed:
                return g._orig["get_device_properties"](device)
            if g.facts is None:
                g._trip("get_device_properties()")
            return _FakeProps(g, g._check_index(g._device_index(device)))

        return {"is_available": is_available, "device_count": device_count,
                "current_device": current_device, "get_device_capability": get_device_capability,
                "get_device_name": get_device_name, "get_device_properties": get_device_properties}

    def _make_tripped(self, name: str, orig: Callable):
        g = self

        def tripped(*a, **k):
            if g.armed:
                g._trip(f"{name}()")
            return orig(*a, **k)

        tripped.__name__ = getattr(orig, "__name__", name)
        tripped.__wrapped__ = orig
        return tripped

    # -- arm / disarm ---------------------------------------------------------------------
    def arm(self):
        import torch  # noqa: F401  (torch import itself never initialises CUDA)
        import torch.cuda as tc

        self._torch = torch
        served = self._make_served()
        for name in self.SERVED:
            if hasattr(tc, name):
                self._orig[name] = getattr(tc, name)
                setattr(tc, name, served[name])
        orig_lazy = tc._lazy_init
        self._orig["_lazy_init"] = orig_lazy
        tripped_lazy = self._make_tripped("torch.cuda._lazy_init", orig_lazy)
        tc._lazy_init = tripped_lazy
        # torch submodules that bound the original function at import (e.g. `from . import _lazy_init`)
        for modname, mod in list(sys.modules.items()):
            if not modname.startswith("torch.cuda") or mod is None or mod is tc:
                continue
            for attr in ("_lazy_init",):
                if getattr(mod, attr, None) is orig_lazy:
                    self._patched_refs.append((mod, attr, orig_lazy))
                    setattr(mod, attr, tripped_lazy)
        C = torch._C
        for name in self.TRIPPED_C:
            fn = getattr(C, name, None)
            if fn is not None:
                self._orig_c[name] = fn
                setattr(C, name, self._make_tripped(f"torch._C.{name}", fn))
        self.armed = True
        global ACTIVE
        ACTIVE = self
        return self

    def disarm(self):
        """Stop serving facts. Module attributes go back to the originals; wrappers that
        were captured by reference keep delegating to them."""
        global ACTIVE
        if ACTIVE is self:
            ACTIVE = None
        if self._torch is None:
            self.armed = False
            return
        import torch.cuda as tc

        self.armed = False
        for name, fn in self._orig.items():
            setattr(tc, name, fn)
        for mod, attr, fn in self._patched_refs:
            setattr(mod, attr, fn)
        for name, fn in self._orig_c.items():
            setattr(self._torch._C, name, fn)


def verify_facts(facts: CudaFacts) -> List[str]:
    """Compare the facts served during preload with the real device (call after CUDA init)."""
    import torch

    problems = []
    n = torch.cuda.device_count()
    if n != facts.count:
        problems.append(f"device_count {n} != {facts.count}")
    for i in range(min(n, facts.count)):
        cap = torch.cuda.get_device_capability(i)
        if cap != (facts.major, facts.minor):
            problems.append(f"capability[{i}] {cap} != {(facts.major, facts.minor)}")
        name = torch.cuda.get_device_name(i)
        if name != facts.name:
            problems.append(f"name[{i}] {name!r} != {facts.name!r}")
    return problems


def nvidia_fds() -> List[str]:
    """Open file descriptors that point at NVIDIA device nodes (driver handles)."""
    fd_dir = "/proc/self/fd" if os.path.isdir("/proc/self/fd") else "/dev/fd"
    out = []
    try:
        for fd in os.listdir(fd_dir):
            try:
                target = os.readlink(os.path.join(fd_dir, fd))
            except OSError:
                continue
            if target.startswith("/dev/nvidia"):
                out.append(target)
    except OSError:
        pass
    return sorted(set(out))


def mapped_libraries(substrings=("libcuda.so", "libnvidia-ml.so")) -> Dict[str, bool]:
    """Which of the given shared libraries are mapped into this process (Linux only)."""
    res = {s: False for s in substrings}
    try:
        with open("/proc/self/maps") as f:
            for line in f:
                for s in substrings:
                    if s in line:
                        res[s] = True
    except OSError:
        return {s: None for s in substrings}  # type: ignore[misc]
    return res


def native_thread_count() -> Optional[int]:
    try:
        return len(os.listdir("/proc/self/task"))
    except OSError:
        return None


def torch_cuda_initialized() -> Optional[bool]:
    tc = sys.modules.get("torch.cuda")
    if tc is None:
        return False
    try:
        return bool(tc.is_initialized())
    except Exception:
        return None
