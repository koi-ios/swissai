"""End-to-end audits of the mock engine: mapping, verification, gate codes, warm hits,
relocation and drift detection. Deterministic and fast (no GPU, no real compiler)."""

import os


def roots_of(rep, run):
    return {r["entry"]: r for r in rep["runs"][run]["roots"]}


def test_unknown_cache_fails_gate(audit):
    code, rep, out = audit("mock_engine.py")
    assert code == 11, out
    cold = roots_of(rep, "cold")
    assert {"triton", "tvm_ffi", "tilelang", "UNKNOWN"} <= set(cold)
    assert cold["UNKNOWN"]["path"].endswith("/home/.cache/mystery-kernels")
    # every known root was mapped to the variable that moves it
    assert rep["proposal"] == {"TILELANG_CACHE_DIR": "{jit_cache}/tilelang",
                               "TRITON_CACHE_DIR": "{jit_cache}/triton",
                               "TVM_FFI_CACHE_DIR": "{jit_cache}/tvm-ffi"}


def test_clean_engine_passes_and_warm_run_rebuilds_nothing(audit):
    code, rep, out = audit("mock_engine.py", "--relocate", env={"MOCK_SKIP_MYSTERY": "1"})
    assert code == 0, out
    assert rep["verification"] == {"tilelang": "verified", "triton": "verified", "tvm_ffi": "verified"}
    for entry, b in rep["builds"].items():
        assert b["binaries_cold"] > 0 and b["binaries_warm"] == 0, (entry, b)
    # writers are attributed to source lines by the audit hook
    w = roots_of(rep, "redirect")
    assert any("mock_engine.py" in k for k in w["triton"]["writers"])
    assert "atomic publish" in w["triton"]["write_pattern"]
    # relocation: triton-like (absolute group paths) and ninja-like rebuild; tilelang-like does not
    assert rep["relocation"].get("triton", 0) > 0
    assert rep["relocation"].get("tvm_ffi", 0) > 0
    assert rep["relocation"].get("tilelang", 0) == 0
    assert rep["runs"]["redirect"]["roots"][0]["path_baked"] >= 0


def test_drift_is_detected(audit):
    code, rep, out = audit("mock_engine.py", "--no-warm", env={"MOCK_SKIP_MYSTERY": "1", "MOCK_DRIFT": "1"})
    assert code == 12, out
    assert rep["verification"]["tilelang"] == "NOT MOVED"


def test_gate_mode_with_incomplete_env_toml(audit, tmp_path):
    toml = tmp_path / "env.toml"
    toml.write_text('image = "x"\n[env]\nTRITON_CACHE_DIR = "{jit_cache}/triton"\n'
                    'TVM_FFI_CACHE_DIR = "{jit_cache}/tvm-ffi"\nNCCL_NET = "x"\n')
    code, rep, out = audit("mock_engine.py", "--no-warm", "--env-toml", str(toml),
                           env={"MOCK_SKIP_MYSTERY": "1"})
    assert code == 10, out
    assert "NCCL_NET" not in rep["proposal"]
    outside = {r["entry"] for r in rep["outside"]}
    assert outside == {"tilelang"}


def test_leaked_temporaries_are_reported_not_gated(audit):
    code, rep, out = audit("mock_engine.py", "--no-warm", env={"MOCK_SKIP_MYSTERY": "1", "MOCK_LEAK_TMP": "1"})
    assert code == 0, out
    assert rep["temporaries"] and rep["temporaries"][0]["path"].endswith(".so")
    assert not rep["outside"]
