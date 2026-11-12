"""Deterministic mock engine whose caches mimic the real libraries' behaviour.

- triton-like: honours TRITON_CACHE_DIR (else ~/.triton/cache); per-file tmp +
  os.replace; a group JSON with ABSOLUTE child paths, so a moved cache misses
  (as Triton 3.7.1 runtime/cache.py get_group does).
- ninja-like (tvm-ffi): honours TVM_FFI_CACHE_DIR (else ~/.cache/tvm-ffi);
  build.ninja embeds absolute paths and is regenerated on a path change,
  which forces a rebuild (as tvm-ffi 0.1.11 does).
- tilelang-like: honours TILELANG_CACHE_DIR (else ~/.tilelang/cache), staging
  dir + atomic rename; with MOCK_DRIFT=1 it ignores the variable (a renamed
  variable, like SGLang 0.5.16 -> 0.5.20).
- mystery: an unknown library writing ~/.cache/mystery-kernels (skip with
  MOCK_SKIP_MYSTERY=1).
Each "compile" writes an ELF-magic file and sleeps a little.
"""
import json
import os
import sys
import time
import uuid

ELF = b"\x7fELF" + b"\0" * 60
KERNELS = ["k_a", "k_b", "k_c"]


def compile_bin(path: str) -> None:
    time.sleep(0.02)
    tmp = f"{path}.tmp.{uuid.uuid4().hex}"
    with open(tmp, "wb") as f:
        f.write(ELF)
    os.replace(tmp, path)


def triton_like() -> int:
    root = os.environ.get("TRITON_CACHE_DIR") or os.path.expanduser("~/.triton/cache")
    built = 0
    for k in KERNELS:
        d = os.path.join(root, f"hash_{k}")
        grp = os.path.join(d, f"__grp__{k}.json")
        if os.path.exists(grp):
            children = json.load(open(grp))["child_paths"]
            if all(os.path.exists(p) for p in children.values()):
                continue  # hit
        os.makedirs(d, exist_ok=True)
        cubin = os.path.join(d, f"{k}.cubin")
        compile_bin(cubin)
        tmp = grp + f".tmp.{uuid.uuid4().hex}"
        with open(tmp, "w") as f:
            json.dump({"child_paths": {f"{k}.cubin": os.path.abspath(cubin)}}, f)
        os.replace(tmp, grp)
        built += 1
    return built


def ninja_like() -> int:
    root = os.environ.get("TVM_FFI_CACHE_DIR") or os.path.expanduser("~/.cache/tvm-ffi")
    built = 0
    for k in KERNELS[:2]:
        d = os.path.join(root, f"{k}_0123456789abcdef")
        os.makedirs(d, exist_ok=True)
        ninja = f"build {d}/{k}.so: link {d}/main.o\n"
        nf = os.path.join(d, "build.ninja")
        so = os.path.join(d, f"{k}.so")
        if os.path.exists(nf) and open(nf).read() == ninja and os.path.exists(so):
            continue
        with open(nf, "w") as f:  # in place, like tvm-ffi's _maybe_write
            f.write(ninja)
        compile_bin(so)
        built += 1
    return built


def tilelang_like() -> int:
    if os.environ.get("MOCK_DRIFT") == "1":
        root = os.path.expanduser("~/.tilelang/cache")
    else:
        root = os.environ.get("TILELANG_CACHE_DIR") or os.path.expanduser("~/.tilelang/cache")
    built = 0
    for k in KERNELS:
        final = os.path.join(root, "0.1.12", "kernels", k)
        if os.path.exists(os.path.join(final, "kernel_lib.so")):
            continue
        stage = os.path.join(root, "0.1.12", ".staging", uuid.uuid4().hex)
        os.makedirs(stage)
        compile_bin(os.path.join(stage, "kernel_lib.so"))
        os.makedirs(os.path.dirname(final), exist_ok=True)
        try:
            os.rename(stage, final)
        except OSError:
            pass
        built += 1
    return built


def mystery() -> int:
    if os.environ.get("MOCK_SKIP_MYSTERY") == "1":
        return 0
    d = os.path.expanduser("~/.cache/mystery-kernels")
    os.makedirs(d, exist_ok=True)
    p = os.path.join(d, "fused.so")
    if os.path.exists(p):
        return 0
    compile_bin(p)
    return 1


def leak_tmp() -> None:
    """Like TileLang's Cython wrapper build: a randomly named .cpp/.so left in TMPDIR."""
    if os.environ.get("MOCK_LEAK_TMP") == "1":
        import tempfile
        fd, p = tempfile.mkstemp(suffix=".so")
        os.write(fd, ELF)
        os.close(fd)


def main() -> int:
    leak_tmp()
    b = {"triton": triton_like(), "tvm_ffi": ninja_like(), "tilelang": tilelang_like(), "mystery": mystery()}
    print("built", json.dumps(b))
    return 0


if __name__ == "__main__":
    sys.exit(main())
