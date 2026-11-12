"""Audits of REAL libraries at the versions both images pin (CPU, no GPU):
torch 2.13.0 Inductor (FX graph + AOTAutograd + C++ codecache) and apache-tvm-ffi 0.1.11 JIT.
Slow (about 1-2 min each); they measure relocation and read-only behaviour for real."""

import pytest

pytest.importorskip("torch")
pytest.importorskip("tvm_ffi")

pytestmark = pytest.mark.slow


def test_torch_inductor_cpu(audit):
    code, rep, out = audit("torch_compile_cpu.py", "--relocate", "--readonly", "--timeout", "300")
    assert code == 0, out
    assert rep["proposal"] == {"TORCHINDUCTOR_CACHE_DIR": "{jit_cache}/inductor"}
    assert rep["verification"]["inductor"] == "verified"
    assert rep["verification"]["inductor.fxgraph"] == "verified"
    b = rep["builds"]["inductor"]
    assert b["binaries_cold"] > 0 and b["binaries_warm"] == 0      # every .so reused
    assert rep["relocation"] == {} or sum(rep["relocation"].values()) == 0   # relocatable
    assert rep["readonly"]["returncode"] != 0                       # lock files: no read-only use
    pch = [r for r in rep["runs"]["redirect"]["roots"] if r["entry"] == "inductor.pch"]
    assert pch, "precompiled headers ignore TORCHINDUCTOR_CACHE_DIR"


def test_tvm_ffi_jit(audit):
    code, rep, out = audit("tvm_ffi_jit.py", "--relocate", "--readonly", "--timeout", "30")
    assert code == 0, out
    assert rep["verification"] == {"tvm_ffi": "verified"}
    assert rep["builds"]["tvm_ffi"]["binaries_cold"] == 3
    assert rep["builds"]["tvm_ffi"]["binaries_warm"] == 0
    assert rep["relocation"].get("tvm_ffi", 0) > 0                  # absolute paths: rebuilds after a move
    assert rep["readonly"]["returncode"] == -9                      # hangs on the lock until killed
    assert rep["readonly"]["write_attempts_into_ro_cache"]["tvm_ffi"] > 10
