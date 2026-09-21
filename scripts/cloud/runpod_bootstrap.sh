#!/bin/bash
# Pod-side setup for TabICL benchmark and ablation work.
#
#   cd /workspace && tar xzf PAYLOAD.tgz && bash tabicl/scripts/cloud/runpod_bootstrap.sh
#
# Idempotent: safe to re-run after a pod restart. Installs the checkout in editable mode so
# that `import tabicl` resolves to the payload rather than to a PyPI wheel -- pod_doctor.py
# checks this and says so, because measuring released code while believing you are measuring
# your branch is the failure this whole lane exists to avoid.
#
# Does NOT start a long run. It leaves the pod ready and prints what to launch next, so that
# a bootstrap failure costs one minute of GPU time instead of being discovered an hour into
# a sweep.
set -euo pipefail

ROOT="${ROOT:-/workspace}"
REPO="$ROOT/tabicl"
export HF_HOME="${HF_HOME:-$ROOT/hf}"
export PIP_DISABLE_PIP_VERSION_CHECK=1

echo "=== tabicl pod bootstrap ==="
echo "repo    $REPO"
echo "HF_HOME $HF_HOME"
mkdir -p "$HF_HOME" "$ROOT/ckpt"
cd "$REPO"

# --- dependencies -----------------------------------------------------------
# torch is already in the image and is 3 GB; --no-deps on the editable install would drop
# scikit-learn etc., so install the extras explicitly and let pip skip the satisfied torch.
echo
echo "--- installing tabicl (editable) + pretrain/test extras ---"
python -m pip install -q -e ".[pretrain,test]" 2>&1 | tail -5

# FlashAttention-3 is Hopper+ only and is a separate, slow source build. Opt in explicitly:
#   INSTALL_FA3=1 bash runpod_bootstrap.sh
# This used to `pip install flash-attn`, which is FA2 -- the code imports FA3 as
# `flash_attn_interface`, so that installed the wrong library and fell back silently.
if [[ "${INSTALL_FA3:-0}" == "1" ]]; then
    echo
    echo "--- building FlashAttention-3 from source (slow; Hopper+ only) ---"
    bash scripts/cloud/build_fa3.sh || echo "FA3 build FAILED -- stage 2/3 will fall back to SDPA"
fi

# --- who am I ---------------------------------------------------------------
echo
echo "--- pod_doctor ---"
python scripts/cloud/pod_doctor.py || {
    echo
    echo "pod_doctor reported a fatal problem. Stopping before anything is measured."
    exit 1
}

# --- warm the checkpoint cache ----------------------------------------------
# The released v2 checkpoints download on first use. Doing it here means a later timing
# measurement is not polluted by a several-hundred-megabyte download.
echo
echo "--- warming the released checkpoint cache ---"
python - <<'PY'
import numpy as np
from tabicl import TabICLClassifier
X = np.random.default_rng(0).standard_normal((64, 5)).astype("float32")
y = (X[:, 0] > 0).astype(int)
clf = TabICLClassifier(n_estimators=1)
clf.fit(X, y)
clf.predict(X[:4])
print("classifier checkpoint ready")
PY

echo
echo "=== ready ==="
cat <<'NEXT'
Next, one of:

  # BM-02: the inference sweep this card was rented for
  python -m benchmarks.suites.speed --run-id BM-05-baseline \
      --preset full --device cuda --checkpoint tabicl-classifier-v2

  # BM-03: real-data accuracy baseline
  python -m benchmarks.suites.real_small --run-id BM-05-baseline \
      --config-id v2-default --device cuda

  # BM-06: proxy-scale ablation (see scripts/train_proxy.sh)
  bash scripts/train_proxy.sh

Results land in benchmarks/_results/*.jsonl. Pull them before terminating:
  runpod_launch.py ssh POD_ID     # prints the scp lines

The volume survives `stop` and keeps billing. `terminate` destroys it.
NEXT
