#!/usr/bin/env bash
# One-time setup on the rented GPU VM: machine checks, a venv with vLLM and
# servekit, the checkpoint on the network volume, a copy on local NVMe, and ENV.md.
#
# Before running:
#   - mount the provider's network volume and link it:  sudo ln -s /mnt/<volume> /net
#   - accept the model licence on its Hugging Face page, then `hf auth login` or export HF_TOKEN
#   - optional: export PROVIDER=... INSTANCE_TYPE=... (recorded in ENV.md)
set -euo pipefail
cd "$(dirname "$0")"
source ./config.env

say() { printf '\n== %s\n' "$*"; }
warn() { printf '!! %s\n' "$*" >&2; }

say "machine"
command -v nvidia-smi >/dev/null || { warn "no nvidia-smi: is this the GPU machine?"; exit 1; }
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader
ngpu=$(nvidia-smi --query-gpu=index --format=csv,noheader | wc -l)
(( ngpu >= 2 )) || warn "only $ngpu GPU: presharding and every TP 2 run need 2"
shm_gb=$(df -BG --output=size /dev/shm | tail -1 | tr -dc '0-9')
if (( shm_gb < 40 )); then
  warn "/dev/shm is ${shm_gb} GB; staging needs ~2x the model. A container's default is 64 MB: use a full VM,"
  warn "or: sudo mount -o remount,size=64G /dev/shm"
fi
for d in "$NET_ROOT" "$LOCAL_ROOT"; do
  mkdir -p "$d" 2>/dev/null || { sudo mkdir -p "$d" && sudo chown "$(id -u):$(id -g)" "$d"; }
  printf '%-16s %s, %s free\n' "$d" "$(findmnt -n -o FSTYPE,SOURCE -T "$d")" "$(df -h --output=avail "$d" | tail -1 | tr -d ' ')"
done
if [[ "$(findmnt -n -o SOURCE -T "$NET_ROOT")" == "$(findmnt -n -o SOURCE -T "$LOCAL_ROOT")" ]]; then
  warn "$NET_ROOT and $LOCAL_ROOT are on the same device: /net must be the NETWORK volume (see config.env)"
fi
free -g | head -2

say "venv and packages (vLLM $VLLM_VERSION, servekit $SERVEKIT_REF)"
if [[ ! -x "$VENV/bin/python" ]]; then
  if command -v uv >/dev/null; then uv venv --python 3.12 "$VENV"; else python3 -m venv "$VENV"; fi
fi
# shellcheck disable=SC1091
source "$VENV/bin/activate"
if command -v uv >/dev/null; then PIP=(uv pip install); else PIP=(pip install); python -m pip install -q -U pip; fi
"${PIP[@]}" "vllm==$VLLM_VERSION"
for extra in runai fastsafetensors instanttensor; do
  "${PIP[@]}" "vllm[$extra]==$VLLM_VERSION" || warn "vllm[$extra] did not install; its arm will be skipped"
done
"${PIP[@]}" "git+https://github.com/eth-easl/servekit.git@$SERVEKIT_REF" "huggingface_hub[cli]"
python -c "import vllm, torch; print('vllm', vllm.__version__, '| torch', torch.__version__, '| cuda', torch.version.cuda)"
servekit 2>&1 | head -2 || true

say "checkpoint $MODEL_ID"
if [[ ! -f "$MODEL_NET/config.json" ]]; then
  if command -v hf >/dev/null; then hf download "$MODEL_ID" --local-dir "$MODEL_NET"
  else huggingface-cli download "$MODEL_ID" --local-dir "$MODEL_NET"; fi
fi
if [[ ! -f "$MODEL_LOCAL/config.json" ]]; then
  echo "copying to local NVMe for the control arm"
  mkdir -p "$MODEL_LOCAL" && cp -a "$MODEL_NET/." "$MODEL_LOCAL/"
fi
du -sh "$MODEL_NET" "$MODEL_LOCAL"

say "what this vLLM offers, and the setup record"
python detect_options.py --refresh
python collect_env.py

say "done"
echo "Next, prove eviction works on this storage (reads a few GB):"
echo "  python evict.py --verify $MODEL_NET"
echo "Then: ./tests/selftest.sh   (one minute, no GPU), then ./run_baseline.sh"
echo "Stop the instance whenever nothing is running."
