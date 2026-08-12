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

import os
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
# One name per concurrent run. `_find` matches on this and `terminate` kills what it finds,
# so two cycles sharing a name means either can tear down the other's pod mid-experiment --
# and a pod that outlives its job bills by the second. Set TABICL_POD_NAME to run more than
# one at a time; the default is unchanged, so an existing cycle with the variable unset
# behaves exactly as before and can still find and terminate its own pod.
NAME = os.environ.get("TABICL_POD_NAME", "tabicl-bench")

# 48 GB matches the card the memory numbers in DESIGN.md were measured on, at a third of
# the L40S price. Ampere is slower per-flop but the workload was 90s there, so it is not
# the constraint.
GPU_PREFERENCE = ["NVIDIA RTX A6000", "NVIDIA A40", "NVIDIA L40S", "NVIDIA L40",
                  "NVIDIA RTX A5000", "NVIDIA GeForce RTX 3090",
                  "NVIDIA GeForce RTX 4090"]

# A round that is bound by SYSTEM RAM rather than VRAM can ask for a larger tier.
#
# The five largest RelBench tasks (both rel-stack, both rel-amazon, and historically
# rel-hm) die with rc=137 -- SIGKILL from the kernel, not a CUDA error -- during the
# relational aggregation, which scans child tables of millions of rows whatever the fit
# pool is. `--offload cpu`, `--offload disk`, `--row-chunk` and `--train-pool` all address
# GPU memory or the fit pool and none of them touch that, which is why six attempts on
# rel-stack/user-badge failed the same way. On RunPod system RAM scales with the GPU tier,
# so the lever is the tier itself.
#
# TABICL_GPU takes a comma-separated preference list. Entries may be either the API's `id`
# ("NVIDIA H100 80GB HBM3") or the friendlier `displayName` ("H100 SXM"); `resolve_gpus`
# below maps the latter onto the former before any pod is created.
#
# BOTH NAMES EXIST AND ONLY ONE WORKS, WHICH HAS NOW COST TWO LAUNCHES. `create_pod` takes
# `gpu_type_id`, so a displayName raises "No GPU found with the specified ID" -- once per
# cloud, per image, per entry, so the failure arrives as a dozen identical lines that read
# like a capacity shortage rather than a typo. Accepting both and resolving up front is the
# fix; the alternative is remembering which of two plausible strings the API wanted.
_gpu = os.environ.get("TABICL_GPU", "").strip()
if _gpu:
    GPU_PREFERENCE = [g.strip() for g in _gpu.split(",") if g.strip()]
IMAGES = [
    "runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04",
    "runpod/pytorch:2.2.0-py3.10-cuda12.1.1-devel-ubuntu22.04",
]
# A round that needs a specific interpreter can demand one. `create` walks IMAGES in order
# per (cloud, GPU) and falls through to the 3.10 image whenever the 3.11 one is unavailable
# for that combination -- which is how a round needing Python >= 3.11 silently landed on
# 3.10.12 and failed its install. Set TABICL_IMAGE_MATCH to a substring to restrict the list
# rather than editing it, so the constraint lives with the round and not in the module.
_image_match = os.environ.get("TABICL_IMAGE_MATCH", "")
if _image_match:
    IMAGES = [i for i in IMAGES if _image_match in i]
    if not IMAGES:
        raise SystemExit(f"TABICL_IMAGE_MATCH={_image_match!r} matched no image")

# Community capacity is erratic and some hosts have broken CUDA; secure costs more but
# is far likelier to yield a usable machine.
#
# It is also likelier to STAY one. A community L40S was reclaimed nine minutes into a
# five-hour round: ssh went to "Connection refused", the logs went with it, and the pod was
# gone from the account before teardown could run. For a round measured in hours the price
# difference is smaller than the cost of losing it, so a long round can demand SECURE with
# TABICL_CLOUD rather than editing this list.
_cloud = os.environ.get("TABICL_CLOUD", "").upper()
if _cloud and _cloud not in ("COMMUNITY", "SECURE"):
    raise SystemExit(f"TABICL_CLOUD={_cloud!r} is not COMMUNITY or SECURE")
CLOUDS = [_cloud] if _cloud else ["COMMUNITY", "SECURE"]

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
    """Public IP and mapped port 22, once the pod has finished starting.

    ``or {}`` rather than a ``get`` default: RunPod returns the ``runtime`` key present and
    explicitly ``null`` while a pod is still booting, and a default only applies when the
    key is absent. So `status` raised AttributeError for the entire startup window --
    exactly when someone checking on a run is most likely to ask.
    """
    for port in (pod.get("runtime") or {}).get("ports") or []:
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
    """Same, but return stdout so a probe's *output* can be inspected, not just its code.

    **Decoded as UTF-8 with replacement, not with the locale codec.** `text=True` alone uses
    the Windows ANSI codepage, and a single non-ASCII byte in a remote log -- a progress bar,
    a unicode minus, an accented author name -- raises UnicodeDecodeError inside subprocess's
    reader thread. That is not a cosmetic failure: `cycle`'s poller runs through this
    function, so one stray byte would crash a healthy lane's driver, which then tears the pod
    down in its `finally`. A log that cannot be decoded must never be able to end a round.
    """
    done = subprocess.run(_ssh_argv(target, command), timeout=timeout,
                          capture_output=True, text=True,
                          encoding="utf-8", errors="replace")
    return done.returncode, (done.stdout or "") + (done.stderr or "")


# `pyproject.toml` requires torch>=2.2 and the model relies on it: `Tensor.all(dim=tuple)`
# in `layers.py` gained multi-dim support in 2.2 and raises TypeError below it. The package
# is installed with --no-deps (letting pip resolve torch once upgraded it past the pod's
# driver and broke hosts the CUDA probe had just cleared), so whatever torch the image
# happens to ship is what runs. A SECURE A6000 shipping 2.1.2 cost a whole pod cycle:
# every fit raised, after provisioning, upload, install and dataset download had all
# succeeded. The declared minimum is not a suggestion the host has read.
MIN_TORCH = (2, 2)


def probe_host(target) -> tuple[bool, str]:
    """Is this host usable at all? CUDA from Python, then torch version, then bandwidth.

    All three failures are silent otherwise. A broken community host reports a healthy
    ``nvidia-smi`` and ``device_count 1`` while every allocation raises; an old-torch host
    imports and allocates perfectly and then raises inside the model's forward pass; a slow
    one looks entirely healthy and simply never finishes. None is worth discovering an hour
    in, and the last two are worth discovering before the dataset download rather than
    after it.
    """
    ok = "".join(chr(c) for c in (67, 85, 68, 65, 95, 79, 75))    # not echoed in the command
    command = (
        f'python -c "import torch;torch.zeros(1).cuda();'
        f'print(chr(67)+chr(85)+chr(68)+chr(65)+chr(95)+chr(79)+chr(75), '
        f'torch.cuda.get_device_name(0));'
        f'print(\'TORCH_VERSION\', torch.__version__)" ; '
        f'curl -s --max-time 60 -o /dev/null -w "BYTES_PER_SEC %{{speed_download}}\\n" '
        f'"https://speed.cloudflare.com/__down?bytes={BANDWIDTH_PROBE_BYTES}"'
    )
    _, out = run_ssh_capture(target, command)
    print("  " + out.strip().replace("\n", "\n  "), flush=True)

    if ok not in out:
        return False, "CUDA unusable from torch"

    version = next((line.split()[1] for line in out.splitlines()
                    if line.startswith("TORCH_VERSION") and len(line.split()) > 1), "")
    try:
        parsed = tuple(int(p) for p in version.split("+")[0].split(".")[:2])
    except ValueError:
        parsed = ()
    if parsed and parsed < MIN_TORCH:
        return False, (f"torch {version} is below the required "
                       f"{'.'.join(map(str, MIN_TORCH))} -- the model's multi-dim "
                       f"Tensor.all raises on it, after everything else has succeeded")

    speed = next((float(line.split()[1]) for line in out.splitlines()
                  if line.startswith("BYTES_PER_SEC") and len(line.split()) > 1), 0.0)
    if speed < BANDWIDTH_FLOOR:
        return False, (f"{speed / 1e6:.2f} MB/s is below the {BANDWIDTH_FLOOR / 1e6:.0f} MB/s "
                       f"floor -- pip and a 385 MB dataset will not finish")
    return True, f"CUDA ok, torch {version}, {speed / 1e6:.1f} MB/s"


def _check_images() -> None:
    """Refuse to offer an image whose torch is below what the model needs.

    `runpod/pytorch:2.1.0-...` sat in this list as a fallback, so when the preferred image
    was unavailable the loop created a pod on it and the probe then rejected the pod for
    shipping torch 2.1.0 -- twice in one run, on hosts that were otherwise fine. The
    "bad hosts" were our own second choice. A minimum the code enforces on the host and
    violates in its own configuration is not a minimum.
    """
    for image in IMAGES:
        tag = image.split(":", 1)[-1]
        try:
            version = tuple(int(p) for p in tag.split("-", 1)[0].split(".")[:2])
        except ValueError:
            continue
        if version < MIN_TORCH:
            raise SystemExit(
                f"image {image!r} ships torch {'.'.join(map(str, version))}, below the "
                f"required {'.'.join(map(str, MIN_TORCH))}. Every pod created from it "
                f"would be provisioned and then rejected by the probe."
            )


def resolve_gpus(names: list[str]) -> list[str]:
    """Map each requested GPU onto an API `id`, accepting an id or a displayName.

    Raises before anything is provisioned if a name matches neither, because the
    alternative -- letting `create_pod` reject it -- is indistinguishable from the tier
    being sold out, and a sold-out tier is something you wait for rather than fix.
    """
    catalog = runpod.get_gpus()
    ids = {g["id"] for g in catalog}
    by_display = {g.get("displayName"): g["id"] for g in catalog if g.get("displayName")}
    out, unknown = [], []
    for n in names:
        if n in ids:
            out.append(n)
        elif n in by_display:
            print(f"  GPU {n!r} -> id {by_display[n]!r}", flush=True)
            out.append(by_display[n])
        else:
            unknown.append(n)
    if unknown:
        raise SystemExit(
            f"TABICL_GPU names not in the RunPod catalog: {unknown}. Use an id or a "
            f"displayName, e.g. 'NVIDIA H100 80GB HBM3' or 'H100 SXM'. Available: "
            + ", ".join(f"{g['id']!r} ({g.get('displayName')})" for g in catalog))
    return out


def create(offset: int = 0):
    if not PUBKEY.exists():
        raise SystemExit(f"no public key at {PUBKEY}")
    _check_images()
    pubkey = PUBKEY.read_text().strip()
    last = None
    # Rotate the GPU order by how many attempts have already been wasted. Without this the
    # loop retries the same preference in the same order and keeps landing on the machine
    # it just rejected: one run spent attempts 7 through 12 on a single host, terminating
    # each pod on arrival because the address was already blacklisted.
    resolved = resolve_gpus(GPU_PREFERENCE)
    gpus = resolved[offset % len(resolved):] + resolved[:offset % len(resolved)]
    for cloud in CLOUDS:
      for gpu in gpus:
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
                # The call can fail AFTER the machine exists -- a timeout on the response
                # still leaves a running pod, and we never saw its id. If one turned up
                # under our name, destroy it before trying the next combination; an
                # unnoticed pod bills by the second. Best-effort: a failure to look is not
                # a reason to abandon the retry loop.
                try:
                    stray = find_pod()
                    if stray is not None:
                        print(f"  found a stray {stray['id']} from the failed create -- "
                              f"terminating", flush=True)
                        runpod.terminate_pod(stray["id"])
                except Exception:         # noqa: BLE001 - never block the retry
                    pass
    raise SystemExit(f"could not create a pod: {last}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("action", choices=["create", "status", "exec", "stop",
                                       "terminate", "sweep"])
    ap.add_argument("--cmd", default="")
    ap.add_argument("--timeout", type=int, default=3600)
    ap.add_argument("--keep", action="store_true", help="leave the pod running")
    args = ap.parse_args()
    auth()

    if args.action == "create":
        # Rejected hosts are remembered. Without this the loop happily takes eight pods
        # from the same broken machine: one run hit 60.249.37.148 eight times in a row,
        # every one with CUDA unusable, because a fresh pod on a bad host looks exactly
        # like a fresh pod. Repeated failures also escalate to SECURE, since a community
        # datacentre with broken drivers stays broken for the whole retry loop.
        bad_hosts: set[str] = set()
        wasted = 0
        for attempt in range(1, 13):
            # Escalate on WASTED ATTEMPTS, not on distinct bad hosts. One persistently
            # broken machine can dominate a community pool: a run once burned all twelve
            # attempts against a single IP, correctly refusing to reuse it and never
            # escalating, because "two distinct bad hosts" was never reached.
            if wasted >= 3 and CLOUDS[0] != "SECURE":
                CLOUDS.reverse()
                print(f"  {wasted} wasted attempts ({len(bad_hosts)} bad host(s)) -- "
                      f"escalating to {CLOUDS[0]}", flush=True)
            pod = find_pod() or create(offset=wasted)
            try:
                pod, target = wait_ready(pod["id"])
            except TimeoutError as exc:
                # A host that reaches RUNNING and never exposes ssh is just another way of
                # being unusable, and it used to end the whole cycle: one such host aborted
                # a run that had twelve attempts left and a queue of work behind it. The
                # retry loop already exists for exactly this; a timeout belongs inside it,
                # not above it.
                print(f"attempt {attempt}: {exc} -- terminating and taking another host",
                      flush=True)
                wasted += 1
                runpod.terminate_pod(pod["id"])
                time.sleep(5)
                continue
            if target[0] in bad_hosts:
                print(f"attempt {attempt}: {target[0]} already rejected -- skipping",
                      flush=True)
                wasted += 1
                runpod.terminate_pod(pod["id"])
                time.sleep(5)
                continue
            print(f"attempt {attempt}: ssh ready root@{target[0]} -p {target[1]}", flush=True)
            usable, why = probe_host(target)
            if usable:
                print(f"USABLE ({why})  ssh root@{target[0]} -p {target[1]}", flush=True)
                return 0
            print(f"  rejected: {why} -- terminating and taking another host", flush=True)
            bad_hosts.add(target[0])
            wasted += 1
            runpod.terminate_pod(pod["id"])
            time.sleep(10)
        raise SystemExit(f"no usable host after 12 attempts; rejected {sorted(bad_hosts)}")

    if args.action == "sweep":
        # REPORT ONLY. This action used to TERMINATE every pod whose name lacked the
        # `tabicl-` prefix, and it was wired into the teardown of every cycle script.
        #
        # On 2026-08-10 it destroyed `bs-sft6`, a running pod belonging to someone else on
        # this account. That was not a near miss or a cost issue -- it was somebody's work,
        # and no amount of orphan-cleanup convenience justifies it.
        #
        # The reasoning that produced it was wrong in a specific way worth naming: I inferred
        # "not named tabicl-*" implies "an orphan of mine" from three consecutive orphans
        # that happened to be mine. The account is not mine, the inference never held, and a
        # destructive default should never rest on an inference of that kind. Untracked pods
        # cost money; other people's pods cost their work, and those are not comparable.
        #
        # It now lists and explains. Terminating anything is a human decision, made with the
        # ids below, by someone who knows what they belong to.
        pods = runpod.get_pods()
        mine = [x for x in pods if (x.get("name") or "").startswith("tabicl-")]
        other = [x for x in pods if not (x.get("name") or "").startswith("tabicl-")]
        print(f"pods on this account: {len(pods)}")
        for x in mine:
            print(f"  OURS      {x['id']}  {x.get('name')}  {x.get('desiredStatus')}")
        for x in other:
            print(f"  NOT OURS  {x['id']}  {x.get('name')}  {x.get('desiredStatus')}"
                  f"   <- do not terminate without checking who owns it")
        if other:
            print("")
            print(f"{len(other)} pod(s) this harness does not recognise. They may be "
                  f"orphans from a failed create, or they may be someone else's work. "
                  f"Terminate by id, deliberately, after checking.")
        return 0

    pod = find_pod()
    if not pod:
        print("no pod named " + NAME)
        return 0

    if args.action == "status":
        print(f"{pod['id']}  {pod.get('desiredStatus')}  "
              f"gpu={(pod.get('machine') or {}).get('gpuDisplayName')}  "
              f"ssh={ssh_target(pod) or 'not ready'}")
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
