#!/bin/bash
# Screen one ablation against a fresh control at proxy scale.
#
# Generalised from scripts/bm06_run.sh, which validated this setup: BM-06 showed the proxy
# reproduces a published ~100-Elo sign at every checkpoint from 2,500 steps
# (benchmarks/RESULTS.md).
#
#   RUN_ID=TP-01 ARM=tp01 ARM_EXTRA="--col_fourier_value True" bash scripts/ablate_run.sh
#
# The control is re-run every time rather than reused, because BM-06's control checkpoints
# were destroyed with its pod (BM-09). Re-running it is also the stronger design: control and
# treatment then share a host, a card, a driver and a code revision, so the only difference
# between them is the flag under test.
#
# Read the LATEST checkpoint. BM-06 found proxy effect sizes are inflated early -- the
# Muon/AdamW gap ran at 86-93% where the paper reports 64% at 280K steps -- so the sign and
# the ordering are trustworthy, the magnitude is not.
set -euo pipefail
cd "$(dirname "$0")/.."
STEPS="${STEPS:-20000}"
RUN_ID="${RUN_ID:?set RUN_ID}"
ARM="${ARM:?set ARM}"
ARM_EXTRA="${ARM_EXTRA:?set ARM_EXTRA}"
CONTROL_EXTRA="${CONTROL_EXTRA:-}"
ROOT="${ROOT:-/workspace}"
GPUS_A="${GPUS_A:-0,1,2,3}"
GPUS_B="${GPUS_B:-4,5,6,7}"
NPER=$(echo "$GPUS_A" | tr ',' '\n' | wc -l | tr -d ' ')
mkdir -p "$ROOT/logs"

launch() {  # name gpus extra
    setsid nohup env CUDA_VISIBLE_DEVICES="$2" STEPS="$STEPS" ABLATION="$1" SEED=42 \
        NUM_GPUS="$NPER" NJOBS=16 CKPT_DIR="$ROOT/ckpt/$RUN_ID-$1" EXTRA="$3" \
        bash scripts/train_proxy.sh > "$ROOT/logs/train-$1.log" 2>&1 < /dev/null &
    echo "launched $1 on gpus $2 ($NPER per arm)  extra='${3:-<none>}'"
}

# CONTROL_EXTRA holds whatever BOTH arms share. Putting the prior change in both arms and
# varying only the encoder isolates the encoder: a control on the old prior would differ from
# the treatment in two ways at once, and neither could be attributed.
launch control "$GPUS_A" "${CONTROL_EXTRA:-}"
launch "$ARM"  "$GPUS_B" "$ARM_EXTRA"

setsid nohup python scripts/bm06_evaluator.py \
    --arm "control=$ROOT/ckpt/$RUN_ID-control=${GPUS_A%%,*}" \
    --arm "$ARM=$ROOT/ckpt/$RUN_ID-$ARM=${GPUS_B%%,*}" \
    --every 2500 --final-step "$STEPS" --anchor ${EVAL_SUITES:---suite cc18_narrow} \
    --run-id "$RUN_ID" --ledger "benchmarks/_results/${RUN_ID}.jsonl" \
    > "$ROOT/logs/evaluator.log" 2>&1 < /dev/null &
echo "launched evaluator -> benchmarks/_results/${RUN_ID}.jsonl"
