#!/bin/bash
# BM-06 -- proxy-scale training for ablations.
#
# The constraint this script exists for: the real recipe is
# 500K + 40K + 10K steps x two checkpoints = six runs (scripts/train_v2_{clf,reg}_stage*.sh).
# Nothing in the TabPFN-3.5 backlog can be decided at that price. This is Stage 1's
# architecture and hyperparameters at a fraction of the steps, on a fixed prior stream, so
# that two runs differ ONLY in the thing being ablated.
#
#   bash scripts/train_proxy.sh                                   # control
#   ABLATION=tp01 EXTRA="--col_fourier_value True" bash scripts/train_proxy.sh
#
# Every knob below that is not the ablation is pinned to the Stage 1 values. Do not "improve"
# one of them for a single run -- the comparison is the deliverable, not the run.
#
# BEFORE TRUSTING ANY RESULT FROM THIS: run the sign-agreement check in
# IMPLEMENTATION_PLAN.md (Phase 0, BM-06). A proxy that does not reproduce the SIGN of a
# known full-scale effect is measuring noise, and a shorter run is cheaper than a wrong
# conclusion.
set -euo pipefail

STEPS="${STEPS:-25000}"            # ~5% of Stage 1. Raise if sign agreement fails.
ABLATION="${ABLATION:-control}"
EXTRA="${EXTRA:-}"                 # the ablation's flags, and nothing else
SEED="${SEED:-42}"
NUM_GPUS="${NUM_GPUS:-1}"
CKPT_DIR="${CKPT_DIR:-/workspace/ckpt/proxy-${ABLATION}-seed${SEED}}"
# Cap the prior workers and pin BLAS to one thread each.
#
# `os.cpu_count() - 2` looks reasonable and is catastrophic on a big pod: 128 vCPUs gave 126
# prior-generation workers, each starting its own 64-thread OpenBLAS pool, and the run died
# with "pthread_create failed ... Resource temporarily unavailable" before step 1. The
# traceback surfaces as a DataLoader worker being killed, which points nowhere near the real
# cause. Prior generation is many small independent jobs -- it wants processes, not threads,
# and nested BLAS parallelism inside each one is pure contention.
NJOBS="${NJOBS:-$(python -c 'import os; print(min(max((os.cpu_count() or 8) - 2, 4), 32))')}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"

echo "=== proxy run: ${ABLATION} (seed ${SEED}, ${STEPS} steps, ${NUM_GPUS} gpu) ==="
echo "extra flags : ${EXTRA:-<none>}"
echo "checkpoints : ${CKPT_DIR}"
echo "prior jobs  : ${NJOBS}"
[[ -n "$EXTRA" ]] || echo "(control run -- no ablation flags)"
mkdir -p "$CKPT_DIR"

# Resume if this run already has checkpoints, exactly as the stage scripts do: a cluster
# time limit mid-ablation must not silently restart from scratch and report a shorter run.
RESUME_ARGS=""
if ls "$CKPT_DIR"/step-*.ckpt >/dev/null 2>&1; then
    echo "resuming from existing checkpoints in $CKPT_DIR"
fi

torchrun --standalone --nproc_per_node="$NUM_GPUS" -m tabicl.train \
            --wandb_log False \
            --wandb_name "proxy_${ABLATION}_seed${SEED}" \
            --device cuda \
            --dtype float32 \
            --np_seed "$SEED" \
            --torch_seed "$SEED" \
            --max_steps "$STEPS" \
            --batch_size 64 \
            --micro_batch_size 4 \
            --lr 8e-4 \
            --muon True \
            --beta1 0.9 \
            --weight_decay 0.01 \
            --use_cautious_wd False \
            --scheduler cosine_with_restarts \
            --warmup_proportion 0.01 \
            --cosine_num_cycles 1 \
            --cosine_amplitude_decay 1 \
            --cosine_lr_end 1e-7 \
            --gradient_clipping 10.0 \
            --prior_type graph_scm \
            --prior_device cpu \
            --n_jobs "$NJOBS" \
            --batch_size_per_gp 4 \
            --min_features 1 \
            --max_features 100 \
            --max_classes 10 \
            --max_seq_len 1024 \
            --min_train_size 0.3 \
            --max_train_size 0.9 \
            --seq_len_per_gp True \
            --graph_noise False \
            --filter_unpredictable_graphs True \
            --filter_unpredictable_datasets True \
            --allow_act_warping False \
            --min_n_nodes 2 \
            --max_n_nodes 32 \
            --cauchy_dag_offset 0.0 \
            --embed_dim 128 \
            --col_num_blocks 3 \
            --col_nhead 8 \
            --col_num_inds 128 \
            --col_affine False \
            --col_feature_group same \
            --col_feature_group_size 3 \
            --col_target_aware True \
            --col_ssmax True \
            --row_num_blocks 3 \
            --row_nhead 8 \
            --row_num_cls 4 \
            --row_rope_base 100000 \
            --row_rope_interleaved False \
            --icl_num_blocks 12 \
            --icl_nhead 8 \
            --icl_ssmax True \
            --ssmax_type qassmax-mlp-elementwise \
            --ff_factor 2 \
            --norm_first True \
            --zero_init False \
            --use_flash_attn3 False \
            --checkpoint_dir "$CKPT_DIR" \
            --save_temp_every 500 \
            --save_perm_every 2500 \
            $RESUME_ARGS \
            $EXTRA

echo
echo "=== done: ${ABLATION} ==="
cat <<NEXT
Evaluate against the control and append to the ledger:

  python -m benchmarks.suites.real_small --run-id ${ABLATION} \\
      --config-id ${ABLATION} --device cuda --checkpoint ${CKPT_DIR}/step-${STEPS}.ckpt

  python -m benchmarks.report --ledger benchmarks/_results/real_small.jsonl \\
      --metric accuracy --against control

Record BOTH the ablation and the control, at >=3 seeds, before reading anything into it.
NEXT
