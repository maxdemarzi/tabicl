#!/usr/bin/env bash
# Setup, then a HARD GATE, then work in the background.
#
# The gate lives HERE rather than in the driver, because the previous arrangement wrote
# `setup.log` and never read it while `work.sh` ran on `;` regardless: four rounds in a row
# shipped nothing, and each printed output shaped like a failed experiment rather than a
# failed build. A setup that did not finish must not be followed by work that pretends it
# did.
set -uo pipefail
cd /workspace
rm -rf logs WORK_DONE work.log setup.log
mkdir -p logs

bash /workspace/_pod_setup.sh > /workspace/setup.log 2>&1
rc=$?
echo "--- setup tail (rc=$rc) ---"
tail -30 /workspace/setup.log
if [ $rc -ne 0 ] || ! grep -q SETUP_OK /workspace/setup.log; then
  echo "SETUP FAILED -- refusing to start work"
  exit 1
fi

# Detached, so the ssh session can drop without killing the round. The driver polls for
# /workspace/WORK_DONE and fetches logs BEFORE any teardown.
nohup bash /workspace/work.sh > /workspace/work.log 2>&1 &
echo "WORK_STARTED pid $!"
