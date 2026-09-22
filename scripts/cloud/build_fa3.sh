#!/bin/bash
# Build FlashAttention-3 from source (Dao-AILab/flash-attention, hopper/). Hopper-class GPUs.
#
# Why this script exists: FA3 is imported as `flash_attn_interface`
# (src/tabicl/_model/attention.py). The PyPI package `flash-attn` is FA2 -- installing it
# leaves FA3 missing and the model falls back to SDPA without a word, which is exactly what
# the old INSTALL_FA3 option in runpod_bootstrap.sh did. And `flash-attn-3` on PyPI is a
# 0.0.0 pure-Python wheel: it cannot contain CUDA kernels. Do not install either for FA3.
#
# FIRST ATTEMPT (2026-09-21): with only FP8 and SM80 compiled out, the build ran for the full
# 2.5 h pod deadline on 224 cores / 112 jobs and never finished -- the watchdog terminated the
# pod at $9.10 with nothing to show. FA3 instantiates a kernel for every combination of head
# dim, dtype and feature, so the fix is to compile ONLY what tabicl calls:
#   * backward      -- training needs it
#   * varlen        -- attention.py calls flash_attn_varlen_func
#   * fp16          -- attention.py casts to fp16 before the FA3 call
#   * head dim 64   -- the ICL blocks; the column/row blocks (head_dim 16) round up to it.
#                      fa3_check verifies that on the pod before any timing is trusted.
# and switch everything else off. Flags are applied only if this checkout's setup.py knows
# them, and any that are unknown are reported -- a silently ignored flag just means a slow
# build again.
#
# The build is paid for once: it produces a wheel. Set FA3_WHEEL to install from a wheel you
# already have (same CUDA / torch / Python / sm_90 image) and skip compilation entirely.
set -euo pipefail
ROOT="${ROOT:-/workspace}"
SRC="$ROOT/flash-attention"

# The runpod/pytorch images ship the full toolkit under /usr/local/cuda but do not put it on
# PATH, so `command -v nvcc` fails on an image that can build perfectly well.
if ! command -v nvcc >/dev/null && [ -x /usr/local/cuda/bin/nvcc ]; then
    export PATH="/usr/local/cuda/bin:$PATH"
fi
export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"
command -v nvcc >/dev/null || { echo "FAIL: nvcc not found -- need a CUDA *devel* image"; exit 1; }
python -m pip install -q ninja packaging 2>&1 | tail -1 || true

[ -d "$SRC" ] || git clone -q --depth 1 https://github.com/Dao-AILab/flash-attention.git "$SRC"
cd "$SRC/hopper"

if [ -n "${FA3_WHEEL:-}" ] && [ -f "$FA3_WHEEL" ]; then
    echo "installing FA3 from existing wheel $FA3_WHEEL"
    python -m pip install -q "$FA3_WHEEL"
    cd / && python -c "import flash_attn_interface as f; print('FA3 OK:', f.__file__)"
    exit 0
fi

cores=$(nproc)
ram_gb=$(awk '/MemAvailable/ {print int($2/1024/1024)}' /proc/meminfo)
jobs=$(( cores / 2 )); by_ram=$(( ram_gb / 6 ))
[ "$by_ram" -lt "$jobs" ] && jobs=$by_ram
[ "$jobs" -lt 1 ] && jobs=1
echo "building FA3: cores=$cores ram_avail=${ram_gb}G MAX_JOBS=$jobs"

WANT_OFF="FP8 SM80 SPLIT PAGEDKV APPENDKV LOCAL SOFTCAP PACKGQA CLUSTER HDIM96 HDIM128 HDIM192 HDIM256"
for f in $WANT_OFF; do
    if grep -q "FLASH_ATTENTION_DISABLE_$f" setup.py; then
        export "FLASH_ATTENTION_DISABLE_$f=TRUE"; echo "  off: $f"
    else
        echo "  WARNING: this setup.py has no FLASH_ATTENTION_DISABLE_$f -- that variant WILL be built"
    fi
done

# Build a wheel rather than installing in place, so it can be pulled back and reused.
start=$(date +%s)
MAX_JOBS=$jobs python setup.py bdist_wheel 2>&1 | tail -3
echo "build took $(( ($(date +%s) - start) / 60 )) min"
wheel=$(ls -t dist/*.whl | head -1)
python -m pip install -q "$wheel"
cp "$wheel" "$ROOT/" && echo "WHEEL: $ROOT/$(basename "$wheel")  <- pull this back and reuse via FA3_WHEEL"

cd /
python -c "import flash_attn_interface as f; print('FA3 OK:', f.__file__)"
