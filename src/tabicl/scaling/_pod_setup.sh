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
# NOT piped through `tail`. A missing LICENSE (pyproject declares `license = {file=...}`)
# failed this install, and `| tail -3` delivered "note: This is an issue with the package
# mentioned above" with the package, and the reason, cut off. Output filters have eaten
# results on this project five times; a build log is the last place to add another.
pip install -q -e . --no-deps
pip install -q "relbench" "pyarrow" scikit-learn scipy einops psutil tqdm huggingface-hub

# Optional extras for a round that needs a package the benchmark does not: a backbone arm,
# a forecasting adapter. Opt-in through the environment so a one-off dependency never
# becomes a cost every other round pays -- and so a failure to install it fails THIS round
# loudly rather than silently changing what every future one measures.
if [ -n "${TABICL_EXTRA_PIP:-}" ]; then
  echo "=== extra: $TABICL_EXTRA_PIP ==="
  pip install -q ${TABICL_EXTRA_PIP}
  # AN EXTRA RESOLVES ITS OWN DEPENDENCIES AND CAN BREAK THE ONES THE BENCHMARK RUNS ON.
  # Drift-Resilient TabPFN installed cleanly and pulled numpy 2.x under a torch compiled
  # against 1.x. Torch then imported with "Failed to initialize NumPy: _ARRAY_API not found"
  # -- a *warning*, not an exception -- so `eval_track_record --help` still parsed, SETUP_OK
  # was printed, and the round started with every tensor<->array conversion dead. An import
  # that succeeds is not an environment that works.
  python - <<'PY'
import numpy, torch
torch.zeros(3).numpy()                 # both directions; either can be the dead one
torch.from_numpy(numpy.zeros(3))
print(f"numpy/torch bridge ok: numpy {numpy.__version__}, torch {torch.__version__}")

# THE TORCH FLOOR, CHECKED AFTER THE EXTRA AND NOT ONLY BEFORE IT. Pinning `numpy<2` to fix
# the bridge dragged torch 2.4.1 down to 2.1.2 -- and the bridge check then passed happily,
# because a downgraded torch is a perfectly working torch. It is just not one this model
# runs on: `Tensor.all(dim=tuple)` gained multi-dim support in 2.2 and raises TypeError
# below it, inside the forward pass, long after setup has declared success. Two different
# ways to break the environment, and a gate that only knew about one of them.
major, minor = (int(p) for p in torch.__version__.split(".")[:2])
if (major, minor) < (2, 2):
    raise SystemExit(
        f"the extra install downgraded torch to {torch.__version__}, below the 2.2 this "
        f"model requires. Install it with --no-deps, or pin torch alongside it.")
print("torch floor ok")
PY
fi
python -c "import torch;print('torch kept at', torch.__version__, 'cuda', torch.cuda.is_available())"
# Hugging Face credentials, if the driver uploaded them. Unauthenticated Hub requests are
# rate-limited and slower, and every round pays that on the TabICL checkpoint -- on a large
# tier the download happens at $3+/hr with the GPU at 0% utilisation, so it is worth removing.
#
# Read from a FILE that the driver scp'd with 0600, never from the ssh command line: an
# argument is visible in the remote process list and lands in any log that echoes the command.
# The token is exported for child processes and never printed.
if [ -f /workspace/.hf_token ]; then
  export HF_TOKEN="$(cat /workspace/.hf_token)"
  export HUGGING_FACE_HUB_TOKEN="$HF_TOKEN"
  mkdir -p /root/.cache/huggingface
  cp /workspace/.hf_token /root/.cache/huggingface/token
  chmod 600 /root/.cache/huggingface/token
  echo "hugging face token installed (${#HF_TOKEN} chars)"
else
  echo "no hugging face token supplied; downloads will be unauthenticated and rate-limited"
fi

# DIAGNOSTICS MAY NOT FAIL A BUILD. Under `set -e` this banner once aborted setup outright:
# `relbench.__version__` does not exist on every release, the heredoc exited non-zero, the
# script stopped before SETUP_OK, and four cycles in a row reported a failure that looked
# like a bad host. A banner that can end a run is not a banner.
echo "=== versions ==="
python - <<'PY' || echo "(version banner failed; not fatal)"
import torch, pandas, sklearn, tabicl
print("torch", torch.__version__, "cuda", torch.cuda.is_available(), torch.cuda.get_device_name(0))
print("pandas", pandas.__version__, "sklearn", sklearn.__version__)
try:
    import relbench
    print("relbench", getattr(relbench, "__version__", "(no __version__)"))
except Exception as exc:
    print("relbench import failed:", exc)
PY

# THE REAL GATE. Not a version string but the entry point the round actually calls: it
# imports the whole chain (torch, relbench, tabicl.scaling) and parses the arguments the
# work script is about to pass. If this exits non-zero the payload is broken and no work
# should start.
echo "=== gate ==="
python -m tabicl.scaling.eval_track_record --help > /dev/null
echo "gate ok: eval_track_record imports and parses"

# THE UNIT TESTS, RUN HERE BECAUSE THEY CANNOT RUN ANYWHERE ELSE. The development machine is
# a Windows checkout with no `tabicl` installed -- `import tabicl` fails, so the suite cannot
# even be collected there, and every runner change until now shipped to a pod unverified. The
# pod is the only environment that has the package, and setup is the only moment before a
# round commits hours to it. `--help` proves the module imports and parses; it proves nothing
# about whether the code does what it says.
#
# Kept fast on purpose: this runs on every round, and a gate that costs real time gets
# skipped. If the suite ever grows slow, mark the slow tests rather than dropping the gate.
echo "=== tests ==="
pip install -q pytest
# `set -e` is NOT in force for this (the script runs under `set -euo pipefail`, but a command
# in an `if` is exempt), and the exit status of a PIPELINE is its LAST stage -- so
# `pytest | tail` reports tail's success no matter what pytest did. That is the same class of
# bug as the `pip | tail -3` that once ate a build failure's only explanation. PIPESTATUS is
# read explicitly, and the tail is 25 lines because a pytest failure summary is worth having.
python -m pytest tests/ -q 2>&1 | tail -25
rc=${PIPESTATUS[0]}
if [ "$rc" -ne 0 ]; then
  echo "UNIT TESTS FAILED (rc=$rc) -- refusing to start the round. The payload is broken in"
  echo "a way that would otherwise be discovered hours in, or not at all."
  exit 1
fi
echo "tests ok"
echo "SETUP_OK"
