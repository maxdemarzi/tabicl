#!/usr/bin/env python3
"""Pull results and shut a pod down as soon as its run finishes -- or at a hard deadline.

A plain deadline (pod_deadline.py) bounds the worst case but bills for every hour between
the run finishing and the deadline firing: a run that ends at 3 a.m. on a $16/hr pod keeps
paying until morning. This watches for the run's DONE marker instead.

On DONE:     pull the ledger and logs, VERIFY the pull, then terminate.
On deadline: pull what exists, then STOP (not terminate) -- the volume, with every
             checkpoint on it, survives for a human to look at.
If a pull fails verification, the pod is stopped rather than terminated. Destroying the
volume is only ever done after the deliverable is confirmed to be on local disk.

Runs locally, because the alternative is putting an API token on a pod on a shared account.
Wrap it in `caffeinate -i` on macOS so the laptop does not sleep through it.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from runpod_launch import REST, _request  # noqa: E402

SSH_OPTS = ["-o", "StrictHostKeyChecking=accept-new", "-o", "UserKnownHostsFile=/dev/null",
            "-o", "LogLevel=ERROR", "-o", "ConnectTimeout=20"]


def now() -> str:
    return dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%d %H:%M:%SZ")


def log(msg: str) -> None:
    print(f"[finish] {now()} {msg}", flush=True)


def ssh(ip, port, cmd) -> subprocess.CompletedProcess:
    return subprocess.run(["ssh", "-p", str(port), "-n", *SSH_OPTS, f"root@{ip}", cmd],
                          capture_output=True, text=True, timeout=120)


def scp(ip, port, remote, local) -> bool:
    Path(local).parent.mkdir(parents=True, exist_ok=True)
    r = subprocess.run(["scp", "-r", "-P", str(port), *SSH_OPTS, f"root@{ip}:{remote}", str(local)],
                       capture_output=True, text=True, timeout=1800)
    return r.returncode == 0


def verified(ledger: Path, need: list[str]) -> bool:
    """The pull counts only if the ledger parses and holds a row for every required config."""
    if not ledger.is_file() or ledger.stat().st_size == 0:
        return False
    seen = set()
    for line in ledger.read_text().splitlines():
        try:
            r = json.loads(line)
        except json.JSONDecodeError:
            continue
        if not r.get("error"):
            seen.add(r.get("config_id"))
    missing = [c for c in need if c not in seen]
    if missing:
        log(f"ledger is missing {missing}")
    return not missing


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("pod_id")
    ap.add_argument("--done-file", required=True, help="Remote path whose existence means the run finished")
    ap.add_argument("--pull", action="append", default=[], help="remote=local (repeatable)")
    ap.add_argument("--ledger", required=True, help="Local ledger path to verify after pulling")
    ap.add_argument("--require", action="append", default=[], help="config_id that must be present")
    ap.add_argument("--hours", type=float, required=True, help="Hard deadline from now")
    ap.add_argument("--poll", type=float, default=300.0)
    args = ap.parse_args()

    deadline = time.time() + args.hours * 3600
    log(f"watching {args.pod_id}: finish on {args.done_file}, hard deadline in {args.hours:.1f} h")

    def endpoint():
        pod = _request(f"{REST}/pods/{args.pod_id}")
        return pod.get("publicIp"), (pod.get("portMappings") or {}).get("22"), pod.get("desiredStatus")

    def pull_all(ip, port) -> bool:
        ok = True
        for spec in args.pull:
            remote, local = spec.split("=", 1)
            if not scp(ip, port, remote, local):
                log(f"pull FAILED: {remote}")
                ok = False
        return ok and verified(Path(args.ledger), args.require)

    while True:
        try:
            ip, port, status = endpoint()
        except SystemExit:
            log("pod not reachable via API -- assuming it is gone; standing down")
            return 0
        if status not in ("RUNNING", "STARTING", "PENDING"):
            log(f"pod is {status}; standing down")
            return 0

        finished = False
        if ip and port:
            try:
                finished = ssh(ip, port, f"test -f {args.done_file} && echo yes").stdout.strip() == "yes"
            except subprocess.TimeoutExpired:
                log("ssh probe timed out; will retry")

        if finished:
            log("run reported DONE -- pulling results")
            if pull_all(ip, port):
                log("results verified on local disk -- terminating")
                _request(f"{REST}/pods/{args.pod_id}", method="DELETE")
                log(f"terminated {args.pod_id}")
            else:
                log("results NOT verified -- stopping instead, volume preserved")
                _request(f"{REST}/pods/{args.pod_id}/stop", method="POST")
            return 0

        if time.time() >= deadline:
            log("HARD DEADLINE -- pulling what exists, then stopping (volume preserved)")
            if ip and port:
                pull_all(ip, port)
            _request(f"{REST}/pods/{args.pod_id}/stop", method="POST")
            log(f"stopped {args.pod_id}")
            return 0

        left = (deadline - time.time()) / 3600
        log(f"running; {left:.2f} h to deadline")
        time.sleep(args.poll)


if __name__ == "__main__":
    raise SystemExit(main())
