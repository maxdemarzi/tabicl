#!/bin/bash
# BM-06, option C: control vs AdamW, 20K steps, one arm per GPU on one pod.
#
# Why AdamW: Muon's published advantage (~100 Elo, ~64% win rate at 280K steps, TabICLv2
# section 7) is largely faster convergence, so it is the published effect most likely to be
# visible at short step counts. If it has not separated by 20K, the other two will not
# either, and the proxy is not viable at an affordable cost.
#
# The AdamW arm uses lr 1e-4 with regular weight decay, exactly as the paper's own AdamW
# ablation did. At Muon's 8e-4 it would lose for the wrong reason.
#
#   bash scripts/bm06_run.sh          # returns immediately; everything runs under setsid
set -euo pipefail
cd "$(dirname "$0")/.."
STEPS="${STEPS:-20000}"
ROOT="${ROOT:-/workspace}"
mkdir -p "$ROOT/logs"

# Data-parallel within each arm: the trainer splits the GLOBAL batch of 64 across ranks
# (64 -> 16 per GPU at 4), so this is the same optimization as a single-GPU run, and the
# paper's own recipe used 4 GPUs. Both arms get identical parallelism, so they stay
# comparable to each other. Measured single-GPU step: 3.0 s -> ~17 h for 20K steps.
GPUS_A="${GPUS_A:-0,1,2,3}"
GPUS_B="${GPUS_B:-4,5,6,7}"
NPER=$(echo "$GPUS_A" | tr ',' '\n' | wc -l | tr -d ' ')

launch() {  # name gpus extra
    local name=$1 gpus=$2 extra=$3
    setsid nohup env CUDA_VISIBLE_DEVICES="$gpus" STEPS="$STEPS" ABLATION="$name" SEED=42 \
        NUM_GPUS="$NPER" NJOBS=16 CKPT_DIR="$ROOT/ckpt/bm06-$name" EXTRA="$extra" \
        bash scripts/train_proxy.sh > "$ROOT/logs/train-$name.log" 2>&1 < /dev/null &
    echo "launched $name on gpus $gpus ($NPER per arm)"
}

launch control "$GPUS_A" ""
launch adamw   "$GPUS_B" "--muon False --lr 1e-4"

# The evaluator shares the first GPU of each arm. Slowing a rank slows that arm's wall
# clock, never its results -- every step is the same computation however long it takes.
setsid nohup python scripts/bm06_evaluator.py \
    --arm "control=$ROOT/ckpt/bm06-control=${GPUS_A%%,*}" \
    --arm "adamw=$ROOT/ckpt/bm06-adamw=${GPUS_B%%,*}" \
    --every 2500 --final-step "$STEPS" --anchor \
    > "$ROOT/logs/evaluator.log" 2>&1 < /dev/null &
echo "launched evaluator"
