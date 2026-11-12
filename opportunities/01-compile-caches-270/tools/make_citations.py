#!/usr/bin/env python3
"""Turn (id, claim, path, quote[, after]) specs into exact citations.

The line number is found by searching the vendored file for ``quote`` (the
first match after line ``after``), so a citation can never point at the wrong
line. Writes kb/engines.citations.json; tests/test_kb_citations.py re-checks it.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SG = "vendor/sglang-v0.5.20/python/sglang"
VL = "vendor/vllm-v0.32.0/vllm"
SK = "vendor/servekit"
PT = "vendor/src/pytorch-v2.13.0"

SPECS = [
    # --- SGLang 0.5.20 engine wiring --------------------------------------------------------
    ("sglang.engine.redirect_call", "import sglang calls redirect_third_party_caches() before torch/FlashInfer import",
     f"{SG}/__init__.py", "_redirect_third_party_caches()", 12),
    ("sglang.engine.redirect_triton", "SGLang setdefaults TRITON_CACHE_DIR to $SGLANG_CACHE_DIR/triton",
     f"{SG}/srt/environ.py", '"TRITON_CACHE_DIR": os.path.join(base, "triton"),', 0),
    ("sglang.engine.redirect_inductor", "SGLang setdefaults TORCHINDUCTOR_CACHE_DIR to $SGLANG_CACHE_DIR/inductor",
     f"{SG}/srt/environ.py", '"TORCHINDUCTOR_CACHE_DIR": os.path.join(base, "inductor"),', 0),
    ("sglang.engine.redirect_nv", "SGLang setdefaults CUDA_CACHE_PATH to $SGLANG_CACHE_DIR/nv",
     f"{SG}/srt/environ.py", '"CUDA_CACHE_PATH": os.path.join(base, "nv"),', 0),
    ("sglang.engine.redirect_flashinfer", "SGLang setdefaults FLASHINFER_WORKSPACE_BASE to $SGLANG_CACHE_DIR",
     f"{SG}/srt/environ.py", '"FLASHINFER_WORKSPACE_BASE": base,', 0),
    ("sglang.engine.redirect_setdefault", "the redirect uses os.environ.setdefault (explicit values win)",
     f"{SG}/srt/environ.py", "os.environ.setdefault(key, value)", 1900),
    ("sglang.default.cache_root_env", "SGLANG_CACHE_DIR defaults to ~/.cache/sglang (not XDG)",
     f"{SG}/srt/environ.py", 'SGLANG_CACHE_DIR = EnvStr(os.path.expanduser("~/.cache/sglang"))', 0),
    ("sglang.env.jit_cache_dir", "SGLANG_JIT_CACHE_DIR is its own variable, unset by default",
     f"{SG}/srt/environ.py", "SGLANG_JIT_CACHE_DIR = EnvStr(None)", 0),
    ("sglang.default.jit_cache_dir", "unset SGLANG_JIT_CACHE_DIR resolves to ~/.cache/sglang/jit, not under SGLANG_CACHE_DIR",
     f"{SG}/kernels/jit/utils/compile/cache.py", 'configured = envs.SGLANG_JIT_CACHE_DIR.get() or "~/.cache/sglang/jit"', 0),
    ("sglang.env.cute_aot", "SGLANG_CUTE_AOT_CACHE_DIR defaults to $SGLANG_CACHE_DIR/cute_aot",
     f"{SG}/srt/environ.py", 'SGLANG_CUTE_AOT_CACHE_DIR = EnvStr(lambda: _default_cache_subdir("cute_aot"))', 0),
    ("sglang.env.dg_cache", "SGLANG_DG_CACHE_DIR defaults to $SGLANG_CACHE_DIR/deep_gemm",
     f"{SG}/srt/environ.py", 'SGLANG_DG_CACHE_DIR = EnvStr(lambda: _default_cache_subdir("deep_gemm"))', 0),
    ("sglang.engine.dg_env_set", "SGLang sets DG_JIT_CACHE_DIR from SGLANG_DG_CACHE_DIR at import",
     f"{SG}/srt/layers/deep_gemm_wrapper/compile_utils.py", 'os.environ["DG_JIT_CACHE_DIR"] = envs.SGLANG_DG_CACHE_DIR.get()', 0),
    ("sglang.engine.legacy_dirs_log", "SGLang 0.5.20 logs that ~/.triton, ~/.cache/flashinfer, ~/.cache/deep_gemm are no longer used",
     f"{SG}/srt/entrypoints/engine.py", '"Compiled-kernel caches now live under SGLANG_CACHE_DIR (%s). These "', 0),
    ("sglang.key.anchor_normalize", "SGLang JIT build keys replace known roots with anchor tokens (relocatable keys)",
     f"{SG}/kernels/jit/utils/compile/cache.py", 'text = text.replace(str(root), f"<{token}>")', 0),
    # --- servekit (what it persists) ---------------------------------------------------------
    # --- vLLM 0.32.0 engine wiring ----------------------------------------------------------
    ("vllm.default.cache_root", "VLLM_CACHE_ROOT defaults to $XDG_CACHE_HOME/vllm or ~/.cache/vllm",
     f"{VL}/envs.py", 'os.path.join(get_default_cache_root(), "vllm"),', 700),
    ("vllm.default.xdg", "vLLM's default cache root follows XDG_CACHE_HOME",
     f"{VL}/envs.py", '"XDG_CACHE_HOME",', 330),
    ("vllm.engine.inductor_forced", "vLLM sets TORCHINDUCTOR_CACHE_DIR under its per-hash compile dir (overrides the caller)",
     f"{VL}/compilation/compiler_interface.py", 'os.environ["TORCHINDUCTOR_CACHE_DIR"] = inductor_cache', 0),
    ("vllm.engine.triton_unless_set", "vLLM puts Triton under the per-hash dir only when the caller did not set TRITON_CACHE_DIR",
     f"{VL}/compilation/compiler_interface.py", "if triton_cache is None or triton_cache == _last_inductor_triton_cache_dir:", 0),
    ("vllm.engine.triton_set", "vLLM exports TRITON_CACHE_DIR process-wide after compile init",
     f"{VL}/compilation/compiler_interface.py", 'os.environ["TRITON_CACHE_DIR"] = triton_cache', 0),
    ("vllm.key.compile_factors", "vLLM compile dir hash = env, config, traced-code and compiler hashes",
     f"{VL}/compilation/backends.py", "factors = [env_hash, config_hash, code_hash, compiler_hash]", 0),
    ("vllm.layout.compile_dir", "compile cache dir is VLLM_CACHE_ROOT/torch_compile_cache/<hash>",
     f"{VL}/compilation/backends.py", 'envs.VLLM_CACHE_ROOT, "torch_compile_cache", hash_key', 0),
    ("vllm.layout.rank_dir", "per-rank subdir rank_<rank>_<dp_rank>/<prefix>",
     f"{VL}/compilation/backends.py", 'local_cache_dir = os.path.join(cache_dir, f"rank_{rank}_{dp_rank}", self.prefix)', 0),
    ("vllm.layout.aot_dir", "AOT compile artifacts under torch_compile_cache/torch_aot_compile/<sha256>",
     f"{VL}/compilation/decorators.py", '"torch_aot_compile",', 0),
    ("vllm.default.aot_on", "AOT compile is on by default for torch >= 2.10",
     f"{VL}/envs.py", 'if is_torch_equal_or_newer("2.10.0") and not disable_compile_cache()', 0),
    ("vllm.key.model_path_in_hash", "ModelConfig.compute_hash ignores served_model_name but not the model path",
     f"{VL}/config/model.py", '"served_model_name",', 450),
    ("vllm.key.all_fields", "get_hash_factors hashes every dataclass field not explicitly ignored",
     f"{VL}/config/utils.py", "if factor in ignored_factors:", 370),
    ("vllm.key.model_field", "ModelConfig has a `model` (name or path) field",
     f"{VL}/config/model.py", 'model: str = "Qwen/Qwen3-0.6B"', 0),
    ("vllm.key.cache_root_ignored", "VLLM_CACHE_ROOT itself is excluded from the compile hash",
     f"{VL}/envs.py", '"VLLM_CACHE_ROOT",', 2350),
    ("vllm.engine.dg_env_name", "vLLM uses DG_JIT_CACHE_DIR for DeepGEMM",
     f"{VL}/utils/deep_gemm.py", 'DEEP_GEMM_JIT_CACHE_ENV_NAME = "DG_JIT_CACHE_DIR"', 0),
    ("vllm.engine.dg_default", "vLLM defaults DG_JIT_CACHE_DIR to VLLM_CACHE_ROOT/deep_gemm when unset",
     f"{VL}/utils/deep_gemm.py", 'envs.VLLM_CACHE_ROOT, "deep_gemm"', 0),
    ("vllm.engine.dg_vendored", "vLLM falls back to its vendored vllm.third_party.deep_gemm",
     f"{VL}/utils/deep_gemm.py", 'module = importlib.import_module("vllm.third_party.deep_gemm")', 0),
    ("vllm.write.binary_safe", "binary compile-cache format is documented as multi-process safe",
     f"{VL}/envs.py", "#     Safe for multiple vllm serve processes accessing the same torch compile cache.", 0),
    ("vllm.write.unpacked_unsafe", "unpacked format is documented as not multi-process safe",
     f"{VL}/envs.py", "#     NOT multiprocess safe - race conditions may occur with multiple processes.", 0),
    ("vllm.env.startup_plan", "VLLM_ENABLE_STARTUP_PLAN persists memory profiling under VLLM_CACHE_ROOT/startup_plan",
     f"{VL}/envs.py", "# and the free-memory baseline) under VLLM_CACHE_ROOT/startup_plan/,", 0),
    ("vllm.env.flashinfer_autotune", "VLLM_FLASHINFER_AUTOTUNE_CACHE_DIR overrides the FlashInfer autotune cache dir",
     f"{VL}/envs.py", '"VLLM_FLASHINFER_AUTOTUNE_CACHE_DIR": lambda: os.getenv(', 0),
    ("vllm.engine.inductor_threads", "import vllm forces TORCHINDUCTOR_COMPILE_THREADS=1 (serial Inductor compile)",
     f"{VL}/env_override.py", 'os.environ["TORCHINDUCTOR_COMPILE_THREADS"] = "1"', 0),
    ("vllm.engine.triton_autotune_cache", "import vllm setdefaults TRITON_CACHE_AUTOTUNING=1 (autotune results go to TRITON_CACHE_DIR)",
     f"{VL}/env_override.py", 'os.environ.setdefault("TRITON_CACHE_AUTOTUNING", "1")', 0),
    ("vllm.engine.tilelang_tmp", "import vllm setdefaults TILELANG_CLEANUP_TEMP_FILES=1 (avoids a world-shared /tmp dir)",
     f"{VL}/env_override.py", 'os.environ.setdefault("TILELANG_CLEANUP_TEMP_FILES", "1")', 0),
    # --- Triton relocation (checked by hand; the torch_triton research file has the rest) ------
    ("triton.reloc.group_abs", "Triton group files keep a child only if its stored (absolute) path exists",
     "vendor/src/triton-3.7.1-5d6048a/python/triton/runtime/cache.py", "if os.path.exists(p):", 80),
    ("triton.write.tmpdir", "Triton put() writes into a tmp.pid_<pid>_<uuid4> dir inside the cache dir",
     "vendor/src/triton-3.7.1-5d6048a/python/triton/runtime/cache.py", 'temp_dir = os.path.join(self.cache_dir, f"tmp.pid_{pid}_{rnd_id}")', 0),
    ("triton.write.replace", "Triton publishes each file with os.replace (atomic), no lock",
     "vendor/src/triton-3.7.1-5d6048a/python/triton/runtime/cache.py", "os.replace(temp_path, filepath)", 100),
    # --- Inductor (torch 2.13.0), checked by hand after the local audit flagged them -----------
    ("inductor.env.cache_dir_var", "TORCHINDUCTOR_CACHE_DIR relocates the Inductor cache",
     f"{PT}/torch/_inductor/runtime/cache_dir_utils.py", 'cache_dir = os.environ.get("TORCHINDUCTOR_CACHE_DIR")', 0),
    ("inductor.engine.setdefault", "the first cache_dir() call pins TORCHINDUCTOR_CACHE_DIR in os.environ",
     f"{PT}/torch/_inductor/runtime/cache_dir_utils.py", 'os.environ["TORCHINDUCTOR_CACHE_DIR"] = cache_dir', 0),
    ("inductor.default.tmp", "default is <tempfile.gettempdir()>/torchinductor_<user>",
     f"{PT}/torch/_inductor/runtime/cache_dir_utils.py", '"torchinductor_" + sanitized_username,', 0),
    ("inductor.pch.default_dir", "precompiled headers deliberately use default_cache_dir(), ignoring TORCHINDUCTOR_CACHE_DIR",
     f"{PT}/torch/_inductor/codecache.py", '_HEADER_DIR = os.path.join(default_cache_dir(), "precompiled_headers")', 0),
    ("inductor.lock.dir", "Inductor takes lock files under <cache>/locks (a read-only cache fails with EACCES)",
     f"{PT}/torch/_inductor/codecache.py", 'lock_dir = os.path.join(cache_dir(), "locks")', 0),
    ("inductor.write.tmpname", "write_atomic temp name is .<pid>.<thread id>.tmp in the target dir (can collide across nodes)",
     f"{PT}/torch/_inductor/codecache.py", 'tmp_path = path.parent / f".{os.getpid()}.{threading.get_ident()}.tmp"', 0),
    ("inductor.write.rename", "write_atomic publishes with rename",
     f"{PT}/torch/_inductor/codecache.py", "tmp_path.rename(target=path)", 0),
    ("torch.macos.brew", "on macOS Inductor's OpenMP probe runs `brew --prefix libomp` (local tests only)",
     f"{PT}/torch/_inductor/cpp_builder.py", 'subprocess.check_output(["brew", "--prefix", "libomp"])', 0),
    # --- model-launch today --------------------------------------------------------------
    ("ml.env.no_cache_vars", "sglang_base sets SGL_ENABLE_JIT_DEEPGEMM=0 and no cache variable",
     "vendor/model-launch/src/swiss_ai_model_launch/assets/envs/sglang_base.toml", 'SGL_ENABLE_JIT_DEEPGEMM = "0"', 0),
    ("ml.launch.arch_placeholder", "sml resolves {arch} in the env TOML on the batch host with sed",
     "vendor/model-launch/src/swiss_ai_model_launch/launchers/framework.py", "sed \"s|{arch}|${SML_ARCH}|g\" \"$SML_ENV_FILE\" > \"$SML_RESOLVED_ENV\"", 0),
    ("ml.launch.writable", "rank sruns run with --container-writable",
     "vendor/model-launch/src/swiss_ai_model_launch/launchers/framework.py", '"    --container-writable \\\\\\n"', 0),
]


def find(path: Path, quote: str, after: int) -> int:
    lines = path.read_text(errors="replace").splitlines()
    for i in range(max(after, 0), len(lines)):
        if quote in lines[i]:
            return i + 1
    raise SystemExit(f"quote not found in {path}: {quote!r}")


def main() -> int:
    out = []
    for cid, claim, rel, quote, after in SPECS:
        line = find(ROOT / rel, quote, after)
        out.append({"id": cid, "claim": claim, "path": rel, "line": line, "quote": quote})
    dst = ROOT / "kb" / "engines.citations.json"
    dst.write_text(json.dumps(out, indent=1) + "\n")
    print(f"wrote {len(out)} citations to {dst.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
