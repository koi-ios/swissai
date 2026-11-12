#!/usr/bin/env bash
# Fetch the exact library sources that the two platform images ship, for the
# knowledge-base citations (kb/*.toml) and tests/test_kb_citations.py.
# Pins come from sglang v0.5.20 python/pyproject.toml and vLLM v0.32.0
# requirements/cuda.txt + cmake/external_projects/*.cmake.
set -euo pipefail
HERE="$(cd "$(dirname "$0")/.." && pwd)"
V="$HERE/vendor"; W="$V/wheels"; S="$V/src"
mkdir -p "$W" "$S"
PY=/Users/koiios/miniconda3/bin/python3

wheel() { # name version filename-substring
  local name="$1" ver="$2" pat="$3" dest="$S/${1}-${2}"
  [ -d "$dest" ] && { echo "have $dest"; return; }
  local url
  url=$(curl -s "https://pypi.org/pypi/${name}/${ver}/json" | $PY -c "
import json,sys
d=json.load(sys.stdin)
for u in d['urls']:
    if '$pat' in u['filename']:
        print(u['url']); break")
  [ -z "$url" ] && { echo "NO WHEEL $name $ver $pat"; return; }
  curl -sL -o "$W/$(basename "$url")" "$url"
  mkdir -p "$dest" && (cd "$dest" && unzip -q -o "$W/$(basename "$url")" 2>/dev/null || tar xzf "$W/$(basename "$url")")
  echo "got $name $ver -> $dest"
}

sparse() { # url ref dest paths...
  local url="$1" ref="$2" dest="$3"; shift 3
  [ -d "$dest" ] && { echo "have $dest"; return; }
  git init -q "$dest"
  git -C "$dest" remote add origin "$url"
  git -C "$dest" sparse-checkout set --no-cone "$@"
  git -C "$dest" fetch -q --depth 1 --filter=blob:none origin "$ref"
  git -C "$dest" checkout -q FETCH_HEAD
  echo "got $url@$ref -> $dest"
}

# PyTorch 2.13.0 (both images): Inductor, AOTAutograd, cpp_extension, triton pin
sparse https://github.com/pytorch/pytorch.git refs/tags/v2.13.0 "$S/pytorch-v2.13.0" \
  '/torch/_inductor/codecache.py' '/torch/_inductor/runtime/cache_dir_utils.py' \
  '/torch/_inductor/config.py' '/torch/_inductor/remote_cache.py' \
  '/torch/_inductor/runtime/autotune_cache.py' '/torch/_inductor/runtime/triton_heuristics.py' \
  '/torch/_inductor/cpp_builder.py' '/torch/_inductor/output_code.py' \
  '/torch/_functorch/_aot_autograd/autograd_cache.py' '/torch/_functorch/config.py' \
  '/torch/compiler/__init__.py' '/torch/compiler/_cache.py' '/torch/_dynamo/config.py' \
  '/torch/_dynamo/package.py' '/torch/_dynamo/precompile_context.py' \
  '/torch/utils/cpp_extension.py' '/torch/utils/_appdirs.py' '/torch/utils/file_baton.py' \
  '/torch/_inductor/package/*' \
  '/.ci/docker/triton_version.txt' '/.ci/docker/ci_commit_pins/triton.txt' '/.github/ci_commit_pins/triton.txt' \
  '/torch/_utils_internal.py'

# Pure-Python or source-carrying wheels at the pinned versions
wheel flashinfer-python 0.6.18 py3-none-any      # sglang 0.5.20
wheel flashinfer-python 0.7.0.post1 py3-none-any # vLLM 0.32.0
wheel apache-tvm-ffi 0.1.11 cp312-abi3-manylinux_2_24_aarch64
wheel tilelang 0.1.12 manylinux_2_34_aarch64
wheel quack-kernels 0.6.4 py3-none-any
wheel quack-kernels 0.6.5 py3-none-any
wheel flash-attn-4 4.0.0b35 py3-none-any        # newest on PyPI when sglang_base was built (2026-10-18)
wheel nvidia-cutlass-dsl-libs-base 4.6.2 aarch64
wheel nvidia-cutlass-dsl-libs-base 4.8.0 aarch64
wheel sgl-deep-gemm 0.2.0 aarch64
wheel humming-kernels 0.1.12 aarch64
wheel humming-kernels 0.1.17 aarch64
wheel tokenspeed-mla 0.1.8 py3-none-any
wheel tokenspeed-mla 0.2.15 py3-none-any
wheel cuda-tile 1.6.0rc5 tar.gz
wheel kernels 0.14.1 manylinux2014_aarch64
wheel numba 0.65.1 tar.gz
wheel xgrammar 0.2.1 tar.gz

# vLLM's vendored DeepGEMM (cmake/external_projects/deepgemm.cmake) + deep_jit submodule
sparse https://github.com/vllm-project/DeepGEMM.git 1e1842a833699298f7afc02eefb2ed168fff6938 "$S/vllm-DeepGEMM-1e1842a" \
  '/deep_gemm/*.py' '/deep_gemm/utils/*' '/csrc/jit/*' '/csrc/apis/*' '/csrc/*.hpp' '/csrc/*.cpp' '/.gitmodules'
# vLLM's vendored FA4 CuTe DSL (cmake/external_projects/vllm_flash_attn.cmake)
sparse https://github.com/vllm-project/flash-attention.git 9cd61de38763d712bb6ce56e2a02cc2bf718c89f "$S/vllm-flash-attention-9cd61de" \
  '/flash_attn/cute/*.py' '/.gitmodules'
echo done
