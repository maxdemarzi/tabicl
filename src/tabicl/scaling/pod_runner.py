"""RunPod lifecycle for the tabicl benchmarks: create, drive over SSH, tear down.

Replaces the Lightning studio. Two reasons the local CPU was the wrong place for this
work, both measured rather than assumed:

  * rel-avito's full 116,598-row context fits in 90s on a 48 GB card and OOMs at 59 GB
    on CPU. rel-event fits run 300-900s locally.
  * more importantly, a five-seed paired comparison is what separates a real effect from
    the +-0.6 noise floor here, and on CPU that costs an hour, so claims kept resting on
    one seed. Every wrong conclusion in this project traces back to that.

A pod bills by the second while running, so **every exit path stops it** unless --keep is
passed. `terminate` destroys it outright; `stop` leaves the volume and is resumable.

Commands run over SSH because RunPod has no equivalent of Lightning's `st.run()`. The
official images install ``$PUBLIC_KEY`` into ``authorized_keys`` at boot, which is what
makes this scriptable without touching the web console.
"""

from __future__ import annotations

import argparse
import pathlib
import subprocess
import sys
import time

import runpod

HOME = pathlib.Path.home()
TOKEN = HOME / ".runpod" / "token.txt"
PUBKEY = HOME / ".ssh" / "id_ed25519.pub"
KEY = HOME / ".ssh" / "id_ed25519"
NAME = "tabicl-bench"

# 48 GB matches the card the memory numbers in DESIGN.md were measured on, at a third of
# the L40S price. Ampere is slower per-flop but the workload was 90s there, so it is not
# the constraint.
GPU_PREFERENCE = ["NVIDIA RTX A6000", "NVIDIA A40", "NVIDIA L40S", "NVIDIA L40",
                  "NVIDIA RTX A5000", "NVIDIA GeForce RTX 3090",
                  "NVIDIA GeForce RTX 4090"]
IMAGES = [
    "runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04",
    "runpod/pytorch:2.1.0-py3.10-cuda11.8.0-devel-ubuntu22.04",
]
# Community capacity is erratic and some hosts have broken CUDA; secure costs more but
# is far likelier to yield a usable machine.
CLOUDS = ["COMMUNITY", "SECURE"]

# A host that cannot pull data is as useless as one that cannot allocate CUDA, and it
# fails far more expensively: one host managed 270 kB/s, so pip alone took 25 minutes and
# rel-event's 385 MB dataset never arrived inside the window. Measured before any work is
# scheduled, because the cost of finding out late is a whole billed session.
BANDWIDTH_FLOOR = 5_000_000        # bytes/sec; datacentre hosts normally do 10-100x this
BANDWIDTH_PROBE_BYTES = 25_000_000


def auth() -> None:
    runpod.api_key = TOKEN.read_text().strip()


def find_pod():
    return next((p for p in runpod.get_pods() if p.get("name") == NAME), None)


def ssh_target(pod) -> tuple[str, int] | None:
    """Public IP and mapped port 22, once the pod has finished starting."""
    for port in pod.get("runtime", {}).get("ports") or []:
        if port.get("privatePort") == 22 and port.get("isIpPublic"):
            return port["ip"], int(port["publicPort"])
    return None


def wait_ready(pod_id: str, timeout: int = 420):
    deadline = time.time() + timeout
    while time.time() < deadline:
        pod = runpod.get_pod(pod_id)
        target = ssh_target(pod) if pod.get("runtime") else None
        if target:
            return pod, target
        print("  waiting for ssh ...", flush=True)
        time.sleep(10)
    raise TimeoutError("pod did not expose ssh in time")


def _ssh_argv(target, command: str) -> list[str]:
    host, port = target
    return [
        "ssh", "-i", str(KEY), "-p", str(port),
        "-o", "StrictHostKeyChecking=no", "-o", "UserKnownHostsFile=/dev/null",
        "-o", "LogLevel=ERROR", "-o", "ServerAliveInterval=30",
        f"root@{host}", command,
    ]


def run_ssh(target, command: str, timeout: int = 3600) -> int:
    return subprocess.run(_ssh_argv(target, command), timeout=timeout).returncode


def run_ssh_capture(target, command: str, timeout: int = 600) -> tuple[int, str]:
    """Same, but return stdout so a probe's *output* can be inspected, not just its code."""
    done = subprocess.run(_ssh_argv(target, command), timeout=timeout,
                          capture_output=True, text=True)
    return done.returncode, (done.stdout or "") + (done.stderr or "")


def probe_host(target) -> tuple[bool, str]:
    """Is this host usable at all? Checks CUDA from Python, then download bandwidth.

    Both failures are silent otherwise. A broken community host reports a healthy
    ``nvidia-smi`` and ``device_count 1`` while every allocation raises; a slow one looks
    perfectly healthy and simply never finishes. Neither is worth discovering an hour in.
    """
    ok = "".join(chr(c) for c in (67, 85, 68, 65, 95, 79, 75))    # not echoed in the command
    command = (
        f'python -c "import torch;torch.zeros(1).cuda();'
        f'print(chr(67)+chr(85)+chr(68)+chr(65)+chr(95)+chr(79)+chr(75), '
        f'torch.cuda.get_device_name(0))" ; '
        f'curl -s --max-time 60 -o /dev/null -w "BYTES_PER_SEC %{{speed_download}}\\n" '
        f'"https://speed.cloudflare.com/__down?bytes={BANDWIDTH_PROBE_BYTES}"'
    )
    _, out = run_ssh_capture(target, command)
    print("  " + out.strip().replace("\n", "\n  "), flush=True)

    if ok not in out:
        return False, "CUDA unusable from torch"

    speed = next((float(line.split()[1]) for line in out.splitlines()
                  if line.startswith("BYTES_PER_SEC") and len(line.split()) > 1), 0.0)
    if speed < BANDWIDTH_FLOOR:
        return False, (f"{speed / 1e6:.2f} MB/s is below the {BANDWIDTH_FLOOR / 1e6:.0f} MB/s "
                       f"floor -- pip and a 385 MB dataset will not finish")
    return True, f"CUDA ok, {speed / 1e6:.1f} MB/s"


def create():
    if not PUBKEY.exists():
        raise SystemExit(f"no public key at {PUBKEY}")
    pubkey = PUBKEY.read_text().strip()
    last = None
    for cloud in CLOUDS:
      for gpu in GPU_PREFERENCE:
        for image in IMAGES:
            try:
                print(f"trying {cloud} {gpu} ...", flush=True)
                pod = runpod.create_pod(
                    name=NAME,
                    image_name=image,
                    gpu_type_id=gpu,
                    cloud_type=cloud,
                    gpu_count=1,
                    volume_in_gb=60,          # RelBench datasets are GB-scale
                    container_disk_in_gb=30,
                    ports="22/tcp",
                    volume_mount_path="/workspace",
                    env={"PUBLIC_KEY": pubkey},
                    support_public_ip=True,
                )
                print(f"created {pod['id']} on {cloud} {gpu}", flush=True)
                return pod
            except Exception as exc:      # noqa: BLE001 - try the next combination
                last = exc
                print(f"  unavailable: {str(exc)[:110]}", flush=True)
    raise SystemExit(f"could not create a pod: {last}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("action", choices=["create", "status", "exec", "stop", "terminate"])
    ap.add_argument("--cmd", default="")
    ap.add_argument("--timeout", type=int, default=3600)
    ap.add_argument("--keep", action="store_true", help="leave the pod running")
    args = ap.parse_args()
    auth()

    if args.action == "create":
        for attempt in range(1, 9):
            pod = find_pod() or create()
            pod, target = wait_ready(pod["id"])
            print(f"attempt {attempt}: ssh ready root@{target[0]} -p {target[1]}", flush=True)
            usable, why = probe_host(target)
            if usable:
                print(f"USABLE ({why})  ssh root@{target[0]} -p {target[1]}", flush=True)
                return 0
            print(f"  rejected: {why} -- terminating and taking another host", flush=True)
            runpod.terminate_pod(pod["id"])
            time.sleep(10)
        raise SystemExit("no usable host after 8 attempts")

    pod = find_pod()
    if not pod:
        print("no pod named " + NAME)
        return 0

    if args.action == "status":
        print(f"{pod['id']}  {pod.get('desiredStatus')}  "
              f"gpu={pod.get('machine',{}).get('gpuDisplayName')}  ssh={ssh_target(pod)}")
        return 0

    if args.action in ("stop", "terminate"):
        fn = runpod.stop_pod if args.action == "stop" else runpod.terminate_pod
        fn(pod["id"])
        print(f"{args.action}ped {pod['id']}")
        return 0

    # exec
    try:
        _, target = wait_ready(pod["id"])
        return run_ssh(target, args.cmd, timeout=args.timeout)
    finally:
        if args.keep:
            print("\npod LEFT RUNNING -- it bills by the second, stop it explicitly", flush=True)
        else:
            try:
                runpod.stop_pod(pod["id"])
                print(f"\nstopped {pod['id']}", flush=True)
            except Exception as exc:      # noqa: BLE001
                print(f"\nSTOP FAILED -- stop it manually: {exc}", flush=True)


if __name__ == "__main__":
    sys.exit(main())
