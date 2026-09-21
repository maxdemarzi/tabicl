#!/bin/bash
# Build FlashAttention-3 from source (Dao-AILab/flash-attention, hopper/). Hopper-class GPUs.
#
# Why this script exists: FA3 is imported as `flash_attn_interface`
# (src/tabicl/_model/attention.py). The PyPI package `flash-attn` is FA2 -- installing it
# leaves FA3 missing and the model falls back to SDPA without a word, which is exactly what
# the old INSTALL_FA3 option in runpod_bootstrap.sh did. And `flash-attn-3` on PyPI is a
# 0.0.0 pure-Python wheel: it cannot contain CUDA kernels. Do not install either for FA3.
#
# The build is slow and RAM-hungry (each nvcc job takes several GB), so parallelism is capped
# by memory as well as cores, and variants this project cannot use are compiled out: FP8
# (tabicl runs attention in fp16/bf16) and the SM80 fallback (this is for Hopper). Head dims
# are left alone -- the column and row blocks use head_dim 16, and disabling a size the model
# calls would turn a slow path into a runtime error.
set -euo pipefail
ROOT="${ROOT:-/workspace}"
SRC="$ROOT/flash-attention"

command -v nvcc >/dev/null || { echo "FAIL: nvcc not found -- need a CUDA *devel* image"; exit 1; }
python -m pip install -q ninja packaging 2>&1 | tail -1 || true

[ -d "$SRC" ] || git clone -q --depth 1 https://github.com/Dao-AILab/flash-attention.git "$SRC"
cd "$SRC/hopper"

cores=$(nproc)
ram_gb=$(awk '/MemAvailable/ {print int($2/1024/1024)}' /proc/meminfo)
jobs=$(( cores / 2 )); by_ram=$(( ram_gb / 6 ))
[ "$by_ram" -lt "$jobs" ] && jobs=$by_ram
[ "$jobs" -lt 1 ] && jobs=1
echo "building FA3: cores=$cores ram_avail=${ram_gb}G MAX_JOBS=$jobs"

export FLASH_ATTENTION_DISABLE_FP8=TRUE
export FLASH_ATTENTION_DISABLE_SM80=TRUE
MAX_JOBS=$jobs python setup.py install 2>&1 | tail -3

cd /
python -c "import flash_attn_interface as f; print('FA3 OK:', f.__file__)"
