"""Knowledge base: matching precedence, citations verified against vendored source, static drift scan."""

import textwrap

import pytest

from cache_audit import citations, kb as kbmod, static

KB = kbmod.load()
H = {"HOME": "/h", "TMPDIR": "/tt", "USER": "u"}

CASES = [
    # vLLM 0.32.0 default (AOT) layout
    ("/h/.cache/vllm/torch_compile_cache/torch_aot_compile/abc/inductor_cache/triton/0/X/k.cubin", "triton"),
    ("/h/.cache/vllm/torch_compile_cache/torch_aot_compile/abc/inductor_cache/fxgraph/ab/k/e", "inductor.fxgraph"),
    ("/h/.cache/vllm/torch_compile_cache/torch_aot_compile/abc/rank_0_0/model", "vllm.torch_compile"),
    ("/h/.cache/vllm/deep_gemm/cache/kernel.x/k.cubin", "deep_gemm"),
    ("/h/.cache/vllm/gpu_p2p_access_cache_for_0,1,2,3.json", "vllm.p2p"),
    ("/h/.cache/vllm/flashinfer_autotune_cache/0.7.0/90a/x/r.json", "vllm.flashinfer_autotune"),
    ("/h/.cache/vllm/modelinfos/m.json", "vllm.root"),
    # SGLang 0.5.20 layout (import-time redirect under ~/.cache/sglang)
    ("/h/.cache/sglang/jit/sm90a/allreduce/build-x/deps-y/m.so", "sglang.jit"),
    ("/h/.cache/sglang/triton/ABC/k.cubin", "triton"),
    ("/h/.cache/sglang/inductor/fxgraph/x", "inductor.fxgraph"),
    ("/h/.cache/sglang/.cache/flashinfer/0.6.18/90a/cached_ops/m/m.so", "flashinfer"),
    ("/h/.cache/sglang/.cache/flashinfer/cubins/abc/x.cubin", "flashinfer.cubins"),
    ("/h/.cache/sglang/cute_aot/fp/k.o", "sglang.cute_aot"),
    ("/h/.cache/sglang/flashinfer/autotune/0.6.18/sm90/k/rank_0.json", "sglang.flashinfer_autotune"),
    ("/h/.cache/sglang/gpu_p2p_access_cache_for_0,1.json", "sglang.p2p"),
    ("/h/.cache/sglang/deep_gemm/cache/x", "deep_gemm"),
    ("/h/.cache/sglang/nv/ComputeCache/x", "cuda_compute_cache"),
    ("/h/.cache/sglang/rust_extensions/x.so", "sglang.root"),
    # libraries at their own defaults
    ("/h/.triton/cache/ABC/k.cubin", "triton"),
    ("/h/.tilelang/cache/0.1.12/linux-aarch64/kernels/sha/kernel_lib.so", "tilelang"),
    ("/h/.cache/tvm-ffi/m_0123/m.so", "tvm_ffi"),
    ("/h/.dj/cache/k.x/k.cubin", "deep_gemm"),
    ("/tt/torchinductor_u/fxgraph/x", "inductor.fxgraph"),
    ("/tt/torchinductor_u/precompiled_headers/x.pch", "inductor.pch"),
    ("/tt/u/cutlass_python_cache/x.mlir", "cute_dsl"),
    ("/tt/u/flash_attention_cute_dsl_cache/fp/k.o", "fa4.cute_dsl"),
    ("/h/.nv/ComputeCache/a/b", "cuda_compute_cache"),
    ("/h/.cache/huggingface/hub/models--x/blob", "hf_hub"),
    ("/anywhere/pkg/__pycache__/m.cpython-312.pyc", "python.bytecode"),
]


@pytest.mark.parametrize("path,expected", CASES)
def test_match(path, expected):
    m = KB.match(path, H)
    assert m is not None and m[0].id == expected, (path, m and m[0].id)


def test_variables_win_over_defaults():
    env = dict(H, FLASHINFER_WORKSPACE_BASE="/w", TILELANG_CACHE_DIR="/t", TRITON_CACHE_DIR="/tr")
    assert KB.match("/w/.cache/flashinfer/0.7.0/90a/cached_ops/x.so", env)[0].id == "flashinfer"
    assert KB.match("/t/0.1.12/kernels/x", env)[0].id == "tilelang"
    m = KB.match("/tr/ABC/k.cubin", env)
    assert m[0].id == "triton" and m[1] == "/tr"
    assert KB.match("/somewhere/else", env) is None


def test_every_citation_checks_out_against_vendored_source():
    cites = citations.load()
    assert len(cites) > 900
    assert citations.check(cites) == []


def test_every_kb_evidence_id_resolves_and_persistable_entries_have_evidence():
    ids = {c.id for c in citations.load()}
    for e in KB.entries:
        assert all(ev in ids for ev in e.evidence), (e.id, [x for x in e.evidence if x not in ids])
        if e.persist and e.vars and not e.id.startswith(("inductor.fxgraph", "inductor.aotautograd")):
            assert e.evidence, e.id


def _fake_site(tmp_path, name, version, files):
    di = tmp_path / f"{name}-{version}.dist-info"
    di.mkdir()
    (di / "METADATA").write_text(f"Metadata-Version: 2.1\nName: {name}\nVersion: {version}\n")
    rec = []
    for rel, text in files.items():
        p = tmp_path / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(text)
        rec.append(f"{rel},,")
    (di / "RECORD").write_text("\n".join(rec) + "\n")


def test_static_scan_flags_renamed_variable(tmp_path):
    # A future TileLang that renamed its cache variable: the KB variable vanishes and a new one appears.
    _fake_site(tmp_path, "tilelang", "9.9.9", {"tilelang/env.py": textwrap.dedent('''
        import os
        KERNEL_CACHE = os.environ.get("TILELANG_KERNEL_CACHE_DIR", "~/.tilelang/kcache")
    ''')})
    rep = static.scan(KB, [str(tmp_path)], scope=["tilelang"])
    assert ("tilelang", "TILELANG_CACHE_DIR", "tilelang") in rep.missing_vars
    assert "TILELANG_KERNEL_CACHE_DIR" in rep.new_vars
    assert rep.exit_code == 12


def test_static_scan_on_real_installed_packages():
    rep = static.scan(KB, None, scope=["apache-tvm-ffi", "tilelang", "torch"])
    assert {"apache-tvm-ffi", "tilelang", "torch"} <= set(rep.installed)
    present = {v for _, v, _ in rep.present_vars}
    assert {"TVM_FFI_CACHE_DIR", "TILELANG_CACHE_DIR", "TORCHINDUCTOR_CACHE_DIR"} <= present
    assert not [m for m in rep.missing_vars if m[2] in ("apache-tvm-ffi", "tilelang", "torch")]
    assert "TILELANG_TMP_DIR" in rep.new_vars  # a secondary knob the KB documents in notes only
