#!/usr/bin/env bash
# Single config file for this machine. Edit this when moving to a different box --
# nothing else in scripts/ should need per-machine changes.
#
# Sourced by scripts/utils/common.sh (which everything else sources). Plain variable
# assignments only.

VENV="${VENV:-$HOME/ckpt-venv}"
# Model: swiss-ai/Apertus-8B-Instruct-2509 (16.1 GB bf16 weights) does NOT fit on this
# box's GPUs (RTX 3060, 12 GB VRAM each) -- the local-3090 experiment ran on a 24 GB
# card. Rather than go to TP=2 (out of scope for this pass) or fp8 quantization (would
# break the byte-identical-output comparison), this box uses Qwen2.5-3B-Instruct
# (3.09B params, ~6.2 GB bf16) instead -- fits comfortably in 12 GB with room for KV
# cache. This means results on this box measure checkpoint/restore MECHANISM and
# speedup ratios, not absolute-number parity with the local-3090's Apertus-8B numbers
# -- see results/results.md.
MODEL_ID="Qwen/Qwen2.5-3B-Instruct"
MODEL_DIR="${MODEL_DIR:-$HOME/models/Qwen2.5-3B-Instruct}"
SERVED_NAME="${SERVED_NAME:-Qwen/Qwen2.5-3B-Instruct}"
PORT="${PORT:-8080}"
CONTEXT_LEN="${CONTEXT_LEN:-8192}"
MEM_FRACTION="${MEM_FRACTION:-0.85}"
MAX_RUNNING="${MAX_RUNNING:-8}"
SGLANG_VERSION="${SGLANG_VERSION:-0.5.10}"
# CUDA graphs work fine here once the CUDA-runtime-version bug below is fixed (a stray
# CUDA 11 libcudart shadowing the venv's CUDA 12 one made ANY JIT kernel launch fail
# with "invalid resource handle", not just ones inside a captured graph -- see
# utils/common.sh's CUDA_HOME/LD_LIBRARY_PATH comments and results/results.md). No
# workaround needed; kept as a documented off-switch in case a future box hits a real
# graph-capture issue.
DISABLE_CUDA_GRAPH="${DISABLE_CUDA_GRAPH:-0}"

# GPU pinning: this box has 2x RTX 3060. Scope for this pass is GPU 0 only -- TP/2-GPU
# work is explicitly out of scope. TP_SIZE is derived from the GPU_IDS count so the
# multi-GPU path stays a one-line change later, not implemented/tested this pass.
GPU_IDS="${GPU_IDS:-0}"
TP_SIZE="$(( $(echo "$GPU_IDS" | tr ',' '\n' | wc -l) ))"

# Reference points, measured on THIS box by scripts/baseline.sh and scripts/restore.sh.
# Fill these in after the first run -- do not trust numbers measured on another machine
# (the old local-3090 experiment's scripts silently defaulted to its own numbers here,
# which is a footgun when moving boxes).
BASELINE_COLD_S="${BASELINE_COLD_S:-109.8}"      # measured: scripts/baseline.sh --cold, Qwen2.5-3B, cuda graphs on, 2026-08-20
NAIVE_RESTORE_S="${NAIVE_RESTORE_S:-23.252}"     # measured: scripts/checkpoint.sh && scripts/restore.sh --cold, cuda graphs on, 2026-08-20
