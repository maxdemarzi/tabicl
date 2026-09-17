#!/usr/bin/env python3
"""Stop a RunPod pod at a wall-clock deadline, unconditionally.

The guard that holds when the other two are the thing that broke. Adapted from
``black_swan/scripts/cloud/deadline_stop.sh``, which exists because a studio was once found
Running on an L40S for 26 hours at 0% GPU.

It asks no questions. It sleeps, then it stops. Set the deadline above the measured run time
and it costs nothing; set it at all and the worst case is bounded.

    nohup python scripts/cloud/pod_deadline.py POD_ID --hours 6 >> deadline.log 2>&1 &

Runs **locally**, not on the pod, and deliberately so: the alternative is shipping an API
token into a pod's environment on an account that is shared with other people. A local
watchdog dies with the laptop, which is a real weakness -- but it is the lesser one, and
`--terminate` bounds the damage further by destroying the volume too.

This is the last line of defence, not the first. The first is remembering to run
``runpod_launch.py stop`` when the work finishes.
"""

from __future__ import annotations

import argparse
import datetime as _dt
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from runpod_launch import REST, _request  # noqa: E402


def _now() -> str:
    return _dt.datetime.now(_dt.timezone.utc).strftime("%H:%M:%SZ")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("pod_id")
    parser.add_argument("--hours", type=float, default=6.0, help="Deadline from now")
    parser.add_argument("--terminate", action="store_true",
                        help="Destroy the pod and its volume rather than stopping it. A stopped "
                             "pod keeps billing its volume; if nobody is watching, that is the "
                             "failure this flag exists for.")
    parser.add_argument("--poll", type=float, default=300.0,
                        help="Seconds between liveness checks; the pod going away early ends the watch")
    args = parser.parse_args()

    deadline = time.time() + args.hours * 3600.0
    verb = "TERMINATE" if args.terminate else "stop"
    print(f"[deadline] {_now()} armed for pod {args.pod_id}: {verb} in {args.hours:.2f} h", flush=True)

    while time.time() < deadline:
        time.sleep(min(args.poll, max(deadline - time.time(), 1.0)))
        try:
            pod = _request(f"{REST}/pods/{args.pod_id}")
        except SystemExit:
            # _request exits on HTTP error; a 404 means the pod is already gone.
            print(f"[deadline] {_now()} pod not reachable -- assuming already gone, standing down",
                  flush=True)
            return 0
        status = (pod or {}).get("desiredStatus")
        if status not in ("RUNNING", "STARTING", "PENDING"):
            print(f"[deadline] {_now()} pod is {status} -- nothing to do, standing down", flush=True)
            return 0
        remaining = (deadline - time.time()) / 3600.0
        print(f"[deadline] {_now()} pod {status}, {remaining:.2f} h left", flush=True)

    print(f"[deadline] {_now()} DEADLINE REACHED -- {verb} regardless of state", flush=True)
    if args.terminate:
        _request(f"{REST}/pods/{args.pod_id}", method="DELETE")
        print(f"[deadline] {_now()} terminated {args.pod_id}", flush=True)
    else:
        _request(f"{REST}/pods/{args.pod_id}/stop", method="POST")
        print(f"[deadline] {_now()} stopped {args.pod_id} (volume still bills)", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
