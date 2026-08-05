#!/usr/bin/env bash
set -euo pipefail
cd /workspace
rm -rf tabicl-src && mkdir -p tabicl-src
tar xzf payload.tar.gz -C tabicl-src
cd tabicl-src
echo "=== installing ==="
# --no-deps is load-bearing. `pip install -e .` resolves tabicl's torch>=2.2 and UPGRADES
# the image's torch (to 2.13.0), which then demands a newer NVIDIA driver than the host
# has: "driver is too old (found version 12080)". The pod is validated by the CUDA probe
# BEFORE setup runs, so we confirm a working torch and then break it ourselves -- every run
# on that host fails afterwards, and the failure looks like a bad host rather than our
# doing. Keep the image's torch; install only what is genuinely missing.
pip install -q -e . --no-deps 2>&1 | tail -3
pip install -q "relbench" "pyarrow" scikit-learn scipy einops psutil tqdm huggingface-hub 2>&1 | tail -3
python -c "import torch;print('torch kept at', torch.__version__, 'cuda', torch.cuda.is_available())"
echo "=== versions ==="
python - <<'PY'
import torch, pandas, sklearn, relbench, tabicl
print("torch", torch.__version__, "cuda", torch.cuda.is_available(), torch.cuda.get_device_name(0))
print("pandas", pandas.__version__, "sklearn", sklearn.__version__, "relbench", relbench.__version__)
PY
echo "SETUP_OK"
