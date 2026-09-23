#!/bin/bash
# TP-07: does one joint checkpoint match a single-task control on EACH task?
#
#   bash scripts/tp07_run.sh                        # 3 arms on gpus 0,1,2, seed 42
#   SEED=43 ROOT=/workspace/s43 bash scripts/tp07_run.sh
#
# Three proxy arms, one GPU each, same host, card, driver and code revision:
#
#   clf_control  the classifier recipe                           scored on cc18_narrow
#   reg_control  + --regression_method quantile                  scored on ctr23
#   joint        + --multitask True                              scored on BOTH
#
# All three use the classifier's LayerNorm with biases, including the regression control,
# although the released regressor was trained bias-free. That is deliberate: the question
# is joint-vs-single-task, and a regression control with a different norm would differ
# from the joint arm in two ways at once. norm_type is not what is being tested.
#
# Equal steps, and the joint arm splits each 64-dataset step 32/32 -- so it sees HALF as
# many datasets of each task as that task's control, and the pair of controls costs twice
# the joint arm. That is the trade TP-07 claims is free, so it is the one tested. Gate
# (IMPLEMENTATION_PLAN.md, Phase 4): joint >= each single-task control on its own suite.
#
# One GPU per arm, launched without torchrun: two consecutive rented 6-GPU hosts failed
# NCCL all-reduce, and the joint step's DDP path is untested on multiple GPUs anyway.
#
# Cost: stage 1 measured 2.65 s/step on one RTX PRO 6000 (benchmarks/RESULTS.md), so
# 20,000 steps is ~15 h per arm, ~44 GPU-hours for the three, plus evaluation on the
# same cards. Run at SEED=42,43,44 before reading anything into it.
set -euo pipefail
cd "$(dirname "$0")/.."
STEPS="${STEPS:-20000}"
RUN_ID="${RUN_ID:-TP-07}"
SEED="${SEED:-42}"
ROOT="${ROOT:-/workspace}"
GPU_CLF="${GPU_CLF:-0}"
GPU_REG="${GPU_REG:-1}"
GPU_JOINT="${GPU_JOINT:-2}"
REG_WEIGHT="${REG_WEIGHT:-1.0}"     # --multitask_reg_weight; tune only if 1.0 fails the gate
SHARED_EXTRA="${SHARED_EXTRA:-}"    # flags for ALL arms, e.g. "--qk_norm True --input_norm True"
# Prior workers per arm. 16 suits one GPU per arm on a big host. With all three arms on ONE
# GPU (GPU_CLF=GPU_REG=GPU_JOINT=0) the pod usually has fewer cores: keep 3 x this <= vCPUs.
NJOBS_PER_ARM="${NJOBS_PER_ARM:-16}"
mkdir -p "$ROOT/logs"

launch() {  # name gpu extra
    setsid nohup env CUDA_VISIBLE_DEVICES="$2" STEPS="$STEPS" ABLATION="$1" SEED="$SEED" \
        NUM_GPUS=1 NJOBS="$NJOBS_PER_ARM" CKPT_DIR="$ROOT/ckpt/$RUN_ID-$1-seed$SEED" EXTRA="$3" \
        bash scripts/train_proxy.sh > "$ROOT/logs/train-$1-seed$SEED.log" 2>&1 < /dev/null &
    echo "launched $1 on gpu $2  extra='${3:-<none>}'"
}

launch clf_control "$GPU_CLF"   "$SHARED_EXTRA"
launch reg_control "$GPU_REG"   "$SHARED_EXTRA --regression_method quantile --num_quantiles 999"
launch joint       "$GPU_JOINT" "$SHARED_EXTRA --multitask True --num_quantiles 999 --multitask_reg_weight $REG_WEIGHT"

# The evaluator scores each checkpoint on the suites its own config can do: the classifier
# on cc18_narrow, the regressor on ctr23, the joint checkpoint on both, under one config_id.
LEDGER="benchmarks/_results/${RUN_ID}-seed${SEED}.jsonl"
setsid nohup python scripts/bm06_evaluator.py \
    --arm "clf_control=$ROOT/ckpt/$RUN_ID-clf_control-seed$SEED=$GPU_CLF" \
    --arm "reg_control=$ROOT/ckpt/$RUN_ID-reg_control-seed$SEED=$GPU_REG" \
    --arm "joint=$ROOT/ckpt/$RUN_ID-joint-seed$SEED=$GPU_JOINT" \
    --every 2500 --final-step "$STEPS" --anchor --suite cc18_narrow --suite ctr23 \
    --run-id "$RUN_ID" --ledger "$LEDGER" \
    > "$ROOT/logs/evaluator-seed$SEED.log" 2>&1 < /dev/null &
echo "launched evaluator -> $LEDGER"

cat <<READ

Read it, per task, joint against that task's own control:

  python -m benchmarks.bm06_analyze $LEDGER --eval-suite cc18_narrow \\
      --control clf_control --treatment joint --expect better
  python -m benchmarks.bm06_analyze $LEDGER --eval-suite ctr23 \\
      --metric crps --secondary r2 --control reg_control --treatment joint --expect better

PASS  neither verdict says OPPOSITE SIGN at the final checkpoints (joint is not worse on
      either task). "not separated" passes: parity at half the compute is the claim.
FAIL  either says OPPOSITE SIGN. Retry with SHARED_EXTRA="--qk_norm True --input_norm True"
      (the report's stability fix, in all three arms) before tuning REG_WEIGHT.
READ
