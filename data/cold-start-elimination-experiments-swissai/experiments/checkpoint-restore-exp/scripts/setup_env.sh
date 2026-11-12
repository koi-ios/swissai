#!/usr/bin/env bash
# One-time environment + model setup, phased + idempotent. See NOTES.md.
# Usage: scripts/setup_env.sh {venv|install|validate|download|cuda-checkpoint|all}
set -u
source "$(dirname "$0")/utils/common.sh"

PY="$VENV/bin/python"
PIP="$VENV/bin/pip"

hr() { printf '\n========== %s ==========\n' "$*"; }
die() { echo "ERROR: $*" >&2; exit 1; }

free_gb() { df -BG --output=avail "$HOME" | tail -1 | tr -dc '0-9'; }

do_venv() {
  hr "venv -> $VENV"
  if [ -x "$PY" ]; then echo "exists, skipping"; else
    python3.12 -m venv "$VENV" || die "venv creation failed (need python3.12-venv?)"
    "$PIP" install -q --upgrade pip setuptools wheel || die "pip upgrade failed"
  fi
  "$PY" --version
}

SERVEKIT_COMMIT="${SERVEKIT_COMMIT:-21575ac5e88e07d1c8422692a7e67cbbd8a30d38}"

do_install() {
  hr "install sglang[all]==$SGLANG_VERSION + servekit + huggingface_hub"
  echo "disk free before: $(free_gb) GiB"
  [ "$(free_gb)" -lt 25 ] && die "less than 25 GiB free; free space first"
  "$PIP" install -q "sglang[all]==$SGLANG_VERSION" || die "sglang install failed"
  "$PIP" install -q ninja || die "ninja install failed"
  # Public https + pinned commit -- no reliance on the servekit git submodule being
  # initialized (SSH remote) on whatever machine this runs on.
  "$PIP" install -q "servekit @ git+https://github.com/eth-easl/servekit@${SERVEKIT_COMMIT}" \
    || die "servekit install failed"
  "$PIP" install -q huggingface_hub hf_transfer || die "huggingface_hub install failed"
  echo "disk free after: $(free_gb) GiB"
}

do_validate() {
  hr "validate imports + CUDA"
  "$PY" - <<'PY' || die "validation failed"
import torch, sglang
print("sglang :", sglang.__version__)
print("torch  :", torch.__version__)
print("cuda   :", torch.version.cuda, "| available:", torch.cuda.is_available())
assert torch.cuda.is_available(), "CUDA not available to torch"
x = torch.ones(1024, 1024, device="cuda")
print("gpu    :", torch.cuda.get_device_name(0), "| matmul ok:", bool((x @ x).sum().item() > 0))
import servekit.cli  # noqa
print("servekit: import OK")
PY
}

do_download() {
  hr "download $MODEL_ID -> $MODEL_DIR"
  if [ -d "$MODEL_DIR" ] && ls "$MODEL_DIR"/*.safetensors >/dev/null 2>&1; then
    echo "weights present ($(du -sh "$MODEL_DIR" | cut -f1)), skipping"
    return 0
  fi
  echo "disk free before: $(free_gb) GiB"
  [ "$(free_gb)" -lt 20 ] && die "less than 20 GiB free; can't fit ~16 GB weights"
  mkdir -p "$MODEL_DIR"
  HF_HUB_ENABLE_HF_TRANSFER=1 "$VENV/bin/hf" download "$MODEL_ID" \
    --local-dir "$MODEL_DIR" || die "hf download failed"
  echo "downloaded: $(du -sh "$MODEL_DIR" | cut -f1)"
}

do_cuda_checkpoint() {
  hr "fetch cuda-checkpoint + build src/counter"
  mkdir -p "$HERE/bin"
  if [ -x "$HERE/bin/cuda-checkpoint" ]; then
    echo "bin/cuda-checkpoint already present, skipping fetch"
  else
    curl -fsSL -o "$HERE/bin/cuda-checkpoint" \
      https://raw.githubusercontent.com/NVIDIA/cuda-checkpoint/main/bin/x86_64_Linux/cuda-checkpoint \
      || die "cuda-checkpoint fetch failed"
    chmod +x "$HERE/bin/cuda-checkpoint"
  fi
  "$HERE/bin/cuda-checkpoint" 2>&1 | head -1

  if [ -x "$HERE/src/counter" ]; then
    echo "src/counter already built, skipping"
  else
    NVCC="$(command -v nvcc || echo /usr/local/cuda/bin/nvcc)"
    [ -x "$NVCC" ] || die "nvcc not found -- install nvidia-cuda-toolkit"
    "$NVCC" -o "$HERE/src/counter" "$HERE/src/counter.cu" || die "nvcc build failed"
  fi
  [ -x "$HERE/src/counter" ] && echo "src/counter: built ($(stat -c%s "$HERE/src/counter") bytes)"
}

CMD="${1:-all}"
case "$CMD" in
  venv)             do_venv ;;
  install)          do_venv; do_install ;;
  validate)         do_validate ;;
  download)         do_download ;;
  cuda-checkpoint)  do_cuda_checkpoint ;;
  all)              do_venv; do_install; do_validate; do_download; do_cuda_checkpoint ;;
  *) echo "usage: $0 {venv|install|validate|download|cuda-checkpoint|all}"; exit 2 ;;
esac
hr "done: $CMD"
