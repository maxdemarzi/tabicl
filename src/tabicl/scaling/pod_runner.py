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


def run_ssh(target, command: str, timeout: int = 3600) -> int:
    host, port = target
    argv = [
        "ssh", "-i", str(KEY), "-p", str(port),
        "-o", "StrictHostKeyChecking=no", "-o", "UserKnownHostsFile=/dev/null",
        "-o", "LogLevel=ERROR", "-o", "ServerAliveInterval=30",
        f"root@{host}", command,
    ]
    return subprocess.run(argv, timeout=timeout).returncode


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
        # Verify CUDA works *from Python*, not just that nvidia-smi answers. A community
        # L40S reported a healthy nvidia-smi and device_count 1 while every allocation
        # failed with "CUDA unknown error" -- unusable, and only visible from inside torch.
        probe = ('python -c "import torch;torch.zeros(1).cuda();'
                 'print(chr(67)+chr(85)+chr(68)+chr(65)+chr(95)+chr(79)+chr(75), torch.cuda.get_device_name(0))"')
        for attempt in range(1, 9):
            pod = find_pod() or create()
            pod, target = wait_ready(pod["id"])
            print(f"attempt {attempt}: ssh ready root@{target[0]} -p {target[1]}", flush=True)
            if run_ssh(target, probe) == 0:
                print(f"USABLE  ssh root@{target[0]} -p {target[1]}", flush=True)
                return 0
            print("  CUDA unusable from torch -- terminating and taking another host", flush=True)
            runpod.terminate_pod(pod["id"])
            time.sleep(10)
        raise SystemExit("no usable host after 5 attempts")

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
