"""RunPod: enumerate, price, plan and (only when asked twice) launch a GPU pod for TabICL.

Ported from black_swan's `scripts/cloud/runpod_launch.py` -- same account, same hard-won
lessons about this API -- and re-aimed at this project's workload, which is not an LLM lane:

  * **BM-02**, the inference sweep, is the VRAM-hungry job. The 1M-row end of the curve at
    100 columns builds a KV cache measured at 48 KiB/row/estimator (benchmarks/RESULTS.md),
    so it is memory-bound long before it is compute-bound.
  * **BM-06 and the Phase 2-4 ablations** are proxy-scale pre-training runs: one card,
    ~25K steps, not the 4-GPU full curriculum in scripts/train_v2_*.sh.

    python scripts/cloud/runpod_launch.py check    # credential + what is already billing
    python scripts/cloud/runpod_launch.py gpus     # prices and stock, live
    python scripts/cloud/runpod_launch.py plan     # the exact create body and $/hr, no call
    python scripts/cloud/runpod_launch.py payload  # build the transfer tarball -- NO network
    python scripts/cloud/runpod_launch.py ssh POD_ID

**`create` is the only subcommand that spends money, and it refuses without
`--yes-i-will-pay`.** Everything else is read-only. That split is the point of the file: the
expensive decision should be one reviewed command, and the twenty things you want to know
*before* making it should cost nothing.

Stdlib only, on purpose -- no `requests`, no `runpod` SDK -- so this stays runnable from a
bare Python anywhere the token is, including on a pod that is diagnosing itself.

The token is read from `~/runpod.token` (override with `RUNPOD_TOKEN_FILE`) or from
`RUNPOD_API_KEY`, at the point of use, and never echoed.

Cloudflare rejects the GraphQL endpoint with 403/1010 when no User-Agent is sent, which
reads exactly like a bad key. `_request` always sets one.
"""

from __future__ import annotations

import argparse
import io
import json
import os
import pathlib
import subprocess
import sys
import tarfile
import urllib.error
import urllib.request

REST = "https://rest.runpod.io/v1"
GRAPHQL = "https://api.runpod.io/graphql"
UA = "tabicl-runpod-launch/1"

HERE = pathlib.Path(__file__).resolve().parent
REPO = HERE.parent.parent

# --------------------------------------------------------------------------- #
# The launch recipe
# --------------------------------------------------------------------------- #

# CUDA 12.8 + torch 2.8 on Ubuntu 22.04. tabicl needs torch>=2.2 for Flash Attention 2;
# FlashAttention-3 is a separate build and is Hopper-or-newer only, which is one of the
# things pod_doctor.py reports -- the v2 recipe enables --use_flash_attn3 in stages 2 and 3,
# so a card that cannot provide it changes what the training scripts do.
IMAGE = "runpod/pytorch:1.1.0-cu1281-torch280-ubuntu2204"

# Fallback: the image behind this account's existing `rai-torch` template.
IMAGE_FALLBACK = "runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04"

# Two workloads, two different right answers. `--lane` picks one; `--gpu` overrides.
#
#   sweep  BM-02, the inference curve. Memory-bound: the KV cache is 48 KiB/row/estimator
#          (benchmarks/RESULTS.md), so the 1M-row end needs every GB. Short job, hours.
#          Defaults to the card the TabPFN-3.5 report used for its own Figure 7, so our
#          numbers land on the same hardware as the ones we are trying to beat.
#
#   train  BM-06 and the Phase 2-4 proxy ablations. The proxy model is small (embed_dim 128,
#          12 ICL blocks, d_model 512) and does NOT need 96 GB. Long job, many hours times
#          many ablations, so $/hr dominates. FlashAttention-3 wants sm_90+, which is why an
#          H100 leads here even though an A100 is cheaper -- stages 2 and 3 of the recipe
#          pass --use_flash_attn3 and fall back silently without it.
#
# THE IDS BELOW ARE THE REST-CREATE SPELLING, WHICH IS NOT THE ONE `gpus` PRINTS.
# The GraphQL catalogue exposes a short `displayName` ("RTX PRO 6000") alongside the real id
# ("NVIDIA RTX PRO 6000 Blackwell Server Edition"), and POST /v1/pods validates gpuTypeIds
# against its own enum that accepts only the latter. Worse, `stock()` on a wrong id returns
# the same empty result as a genuinely unavailable card -- so a typo reads as "out of stock"
# and sends you hunting for capacity that was there all along. That happened here: a plan for
# "NVIDIA RTX PRO 6000" reported NONE AVAILABLE while the real id was High stock.
# The authority is GET /v1/openapi.json; `gpus --ids` prints the reconciliation.
#
# Prices and stock read 2026-09-17; they drift, re-read with `gpus`.
LANES = {
    "sweep": {
        # 96 GB, sm_120. Secure was High stock at $2.09 when this was written.
        "gpu": "NVIDIA RTX PRO 6000 Blackwell Server Edition",
        "alternates": (
            "NVIDIA RTX PRO 6000 Blackwell Workstation Edition",   # 96 GB, ~$1.69 community
            "NVIDIA RTX PRO 6000 Blackwell Max-Q Workstation Edition",  # 96 GB, ~$1.64, ~300W
            "NVIDIA H100 NVL",                                     # 94 GB, sm_90, ~$2.59
            "NVIDIA A100 80GB PCIe",                               # 80 GB, sm_80, ~$1.59
        ),
    },
    "train": {
        # H100 PCIe had no stock in either cloud; NVL is the same sm_90 generation and does
        # have it, which is what matters for FlashAttention-3.
        "gpu": "NVIDIA H100 NVL",                                  # 94 GB, sm_90, FA3
        "alternates": (
            "NVIDIA H100 PCIe",                                    # 80 GB, sm_90, FA3
            "NVIDIA A100 80GB PCIe",                               # 80 GB, sm_80, no FA3
            "NVIDIA L40S",                                         # 48 GB, sm_89, no FA3
            "NVIDIA RTX A6000",                                    # 48 GB, sm_86, no FA3
        ),
    },
}

#: Never mix cards inside one measured comparison. A treatment run on an H100 against a
#: control run on an A100 is not an ablation, it is a hardware difference wearing one.
#: pod_doctor.json records gpu/sm/vram so a result stays attributable -- read it.
DEFAULT_LANE = "sweep"
GPU = LANES[DEFAULT_LANE]["gpu"]
GPU_ALTERNATES = LANES[DEFAULT_LANE]["alternates"]

POD_NAME = "tabicl-bench"

# Image (~20 GB) plus pip. tabicl's own deps are small and torch is already in the image.
CONTAINER_DISK_GB = 50

# /workspace, survives `stop`. Holds the HF cache (released v2 clf + reg checkpoints), the
# checkout, benchmark dataset caches, and ablation checkpoints -- the proxy model is small
# (embed_dim 128, 12 ICL blocks) but --save_perm_every 1000 over 25K steps is ~25 of them
# per run, and there will be several runs before anything is deleted.
VOLUME_GB = 100

MOUNT = "/workspace"


def _alternates(args: argparse.Namespace) -> tuple:
    """Alternates for the selected lane, unless --gpu was given explicitly.

    An explicit --gpu with this lane's alternates behind it would silently substitute a card
    the caller did not ask for, which is the one thing a measured comparison cannot survive.
    """
    if getattr(args, "gpu_explicit", False):
        return ()
    return LANES[getattr(args, "lane", DEFAULT_LANE)]["alternates"]


def create_body(args: argparse.Namespace) -> dict:
    """The exact JSON POSTed to /v1/pods. Built in one place so `plan` cannot drift from
    `create` -- a dry run that prints something other than what would be sent is worse
    than no dry run."""
    body = {
        "name": args.name,
        "imageName": args.image,
        "gpuTypeIds": [args.gpu, *([] if args.no_alternates else _alternates(args))],
        "gpuTypePriority": "availability",
        "gpuCount": getattr(args, "gpu_count", 1),
        "cloudType": args.cloud,
        "computeType": "GPU",
        "interruptible": args.spot,
        "containerDiskInGb": args.container_disk,
        "volumeInGb": args.volume,
        "volumeMountPath": MOUNT,
        "ports": ["22/tcp"],
        "supportPublicIp": True,
        # WITHOUT THIS THE POD EXITS ABOUT FIVE SECONDS AFTER IT STARTS.
        # Created with no templateId, RunPod runs the image's own CMD, which for
        # runpod/pytorch returns immediately -- the pod goes to EXITED ("Exited by Runpod")
        # with an empty publicIp and no error field anywhere in the record, so it reads as
        # "still starting" until you notice `desiredStatus`. It keeps billing disk meanwhile.
        #
        # `sshd -D` is the whole start command on purpose: it blocks, so it doubles as the
        # process that keeps the container alive, and there is no `sleep infinity` that can
        # hold a pod up while ssh is quietly dead. A first attempt did exactly that -- pod
        # RUNNING, port mapped, every connection refused -- because overriding the CMD also
        # skipped the image's own setup, and sshd will not start without host keys.
        # `ssh-keygen -A` is that missing step; `/run/sshd` is the privilege-separation dir
        # sshd refuses to start without.
        "dockerStartCmd": [
            "/bin/bash", "-c",
            "set -x; "
            "mkdir -p /root/.ssh /run/sshd && "
            "echo \"$PUBLIC_KEY\" >> /root/.ssh/authorized_keys && "
            "chmod 700 /root/.ssh && chmod 600 /root/.ssh/authorized_keys && "
            "(command -v sshd >/dev/null 2>&1 || "
            "  (apt-get update -qq && apt-get install -y -qq openssh-server)) && "
            "ssh-keygen -A && "
            "sed -i 's/^#*PermitRootLogin.*/PermitRootLogin prohibit-password/' "
            "  /etc/ssh/sshd_config && "
            "/usr/sbin/sshd -D -e",
        ],
        "env": {},
    }
    key = _ssh_pubkey()
    if key:
        # This account's SSH keys belong to six different RelationalAI people and none of
        # them is this machine's. Passing the key as env is how the one pod that currently
        # runs on this account was made reachable; relying on the account key list would
        # produce a pod nobody here can log into.
        body["env"]["PUBLIC_KEY"] = key
    return body


def _ssh_pubkey() -> str | None:
    path = pathlib.Path(os.environ.get("RUNPOD_SSH_PUBKEY",
                                       pathlib.Path.home() / ".ssh" / "id_ed25519.pub"))
    return path.read_text(encoding="utf-8").strip() if path.is_file() else None


# --------------------------------------------------------------------------- #
# transport
# --------------------------------------------------------------------------- #

def token() -> str:
    """From the environment or `~/runpod.token`, read at the point of use.

    Never returned to a printer, never interpolated into a command line. The one place it
    is allowed to appear is the Authorization header built in `_request`.
    """
    env = os.environ.get("RUNPOD_API_KEY")
    if env:
        return env.strip()
    path = pathlib.Path(os.environ.get("RUNPOD_TOKEN_FILE",
                                       pathlib.Path.home() / "runpod.token"))
    if not path.is_file():
        sys.exit(f"no credential: set RUNPOD_API_KEY or put the token in {path}")
    value = path.read_text(encoding="utf-8").strip()
    if not value:
        sys.exit(f"{path} is empty")
    return value


def _request(url: str, *, method: str = "GET", body: dict | None = None,
             verbose: bool = False) -> object:
    payload = json.dumps(body).encode() if body is not None else None
    if verbose:
        print(f"  -> {method} {url}" + (f"\n     {json.dumps(body)}" if body else ""),
              file=sys.stderr)
    request = urllib.request.Request(
        url, data=payload, method=method,
        headers={"Authorization": "Bearer " + token(),
                 "Content-Type": "application/json",
                 "User-Agent": UA})
    try:
        with urllib.request.urlopen(request, timeout=120) as response:
            raw = response.read()
    except urllib.error.HTTPError as exc:
        detail = exc.read()[:600].decode(errors="replace")
        if exc.code in (401, 403) and "1010" in detail:
            detail += ("\n(Cloudflare 1010 is a missing User-Agent, not a bad key -- "
                       "if you see this from this script the UA constant was dropped)")
        sys.exit(f"HTTP {exc.code} from {url}\n{detail}")
    return json.loads(raw) if raw else None


def gql(query: str, *, verbose: bool = False) -> dict:
    result = _request(GRAPHQL, method="POST", body={"query": query}, verbose=verbose)
    if isinstance(result, dict) and result.get("errors"):
        sys.exit("GraphQL error: " + json.dumps(result["errors"])[:600])
    return (result or {}).get("data", {})


# --------------------------------------------------------------------------- #
# read-only
# --------------------------------------------------------------------------- #

def gpu_table(verbose: bool = False) -> list[dict]:
    data = gql("query { gpuTypes { id displayName memoryInGb maxGpuCount "
               "securePrice communityPrice secureSpotPrice communitySpotPrice } }",
               verbose=verbose)
    return [g for g in data.get("gpuTypes", []) if (g.get("memoryInGb") or 0) > 0]


def stock(gpu_id: str, secure: bool, verbose: bool = False) -> dict:
    """`lowestPrice` is the only place stockStatus is exposed, and it is per (type, cloud).

    High/Medium/Low, or null when nothing matching the filters exists at all. Note that
    null here is 'none available right now', not 'no such GPU' -- the type still appears in
    `gpuTypes` with a price.
    """
    data = gql(f'query {{ gpuTypes(input:{{id:"{gpu_id}"}}) {{ lowestPrice('
               f'input:{{gpuCount:1, secureCloud:{"true" if secure else "false"}}}) '
               f"{{ uninterruptablePrice minimumBidPrice stockStatus }} }} }}", verbose=verbose)
    types = data.get("gpuTypes") or []
    return (types[0].get("lowestPrice") or {}) if types else {}


def datacenters_with(gpu_id: str, verbose: bool = False) -> list[str]:
    data = gql("query { dataCenters { id storageSupport gpuAvailability "
               "{ available stockStatus gpuTypeId } } }", verbose=verbose)
    out = []
    for dc in data.get("dataCenters", []):
        for entry in dc.get("gpuAvailability") or []:
            if entry.get("gpuTypeId") == gpu_id and entry.get("available"):
                out.append(f"{dc['id']}({entry.get('stockStatus')}"
                           f"{',netvol' if dc.get('storageSupport') else ''})")
    return out


def cmd_check(args: argparse.Namespace) -> int:
    """Prove the credential works and say what it is already paying for.

    The balance and the live-pod list are deliberately in the same output. A pod someone
    forgot about is the single most expensive thing this account can be doing, and it is
    invisible from a terminal that only ever runs `create`.
    """
    data = gql("query { myself { id clientBalance currentSpendPerHr spendLimit pods "
               "{ id name desiredStatus costPerHr gpuCount machine "
               "{ gpuDisplayName location } runtime { uptimeInSeconds } } } }",
               verbose=args.verbose)
    me = data.get("myself") or {}
    print(f"credential OK -- account {me.get('id')}")
    print(f"  balance ${me.get('clientBalance'):.2f}    "
          f"spend limit ${me.get('spendLimit')}/hr    "
          f"current spend ${me.get('currentSpendPerHr')}/hr")
    pods = me.get("pods") or []
    if not pods:
        print("  no pods -- nothing is billing GPU time")
        return 0
    print(f"  {len(pods)} pod(s):")
    for pod in pods:
        machine = pod.get("machine") or {}
        up = (pod.get("runtime") or {}).get("uptimeInSeconds")
        print(f"    {pod['id']}  {pod['name']:<24} {pod['desiredStatus']:<9} "
              f"{machine.get('gpuDisplayName')} x{pod.get('gpuCount')} "
              f"@ {machine.get('location')}  ${pod.get('costPerHr')}/hr"
              + (f"  up {up // 60} min" if up else ""))
    running = [p for p in pods if p.get("desiredStatus") == "RUNNING"]
    if running:
        print(f"\n  {len(running)} RUNNING and billing. These are not mine to stop -- "
              f"confirm they are yours before terminating anything.")
    return 0


def cmd_gpus(args: argparse.Namespace) -> int:
    rows = gpu_table(args.verbose)
    rows.sort(key=lambda g: (-(g["memoryInGb"]), g["id"]))
    print(f"{'gpu':<34}{'VRAM':>6}  {'sec OD':>7}{'sec spot':>9}"
          f"{'com OD':>8}{'com spot':>9}  max")
    for g in rows:
        if args.min_vram and g["memoryInGb"] < args.min_vram:
            continue
        def money(value):
            return f"${value:.2f}" if value else "-"
        print(f"{g['displayName']:<34}{g['memoryInGb']:>4}GB  "
              f"{money(g['securePrice']):>7}{money(g['secureSpotPrice']):>9}"
              f"{money(g['communityPrice']):>8}{money(g['communitySpotPrice']):>9}"
              f"  {g.get('maxGpuCount')}")
    print("\nstock and datacenters for the target and its alternates:")
    for gpu_id in (GPU, *GPU_ALTERNATES):
        secure, community = stock(gpu_id, True), stock(gpu_id, False)
        print(f"  {gpu_id:<34} secure {secure.get('stockStatus') or 'none':<7} "
              f"community {community.get('stockStatus') or 'none':<7} "
              f"{' '.join(datacenters_with(gpu_id, args.verbose)) or '(no listed DC)'}")
    print("\nA GPU can be rentable with '(no listed DC)': community hosts outside RunPod's\n"
          "own datacenters do not appear in the dataCenters map.")
    return 0


def cmd_plan(args: argparse.Namespace) -> int:
    """Everything `create` would do, priced, without doing it."""
    body = create_body(args)
    live = stock(args.gpu, args.cloud == "SECURE", args.verbose)
    if args.spot:
        rate = live.get("minimumBidPrice")
        basis = "spot / interruptible -- can be reclaimed mid-run"
    else:
        rate = live.get("uninterruptablePrice")
        basis = "on-demand"
    if rate is None:
        # `lowestPrice` returns nothing when stock is zero, which makes the plan print
        # "$None/hr" exactly when you most want to know what it would cost if it came back.
        # Fall back to the catalogue price and say that is what happened.
        field = {("SECURE", True): "secureSpotPrice", ("SECURE", False): "securePrice",
                 ("COMMUNITY", True): "communitySpotPrice",
                 ("COMMUNITY", False): "communityPrice"}[(args.cloud, bool(args.spot))]
        for entry in gpu_table(args.verbose):
            if entry.get("id") == args.gpu:
                rate = entry.get(field)
                basis += " -- catalogue price, nothing in stock right now"
                break
    # Measured from this account's own billing history: 1200 GB billed for 24 h cost
    # $3.333, i.e. $0.00278/GB/day. Container disk and volume are both billed.
    disk = (args.container_disk + args.volume) * 0.00278 / 24

    print("POST " + REST + "/pods")
    print(json.dumps(body, indent=2))
    print()
    print(f"  GPU     {args.gpu}  ({args.cloud.lower()}, {basis})")
    print(f"          alternates in priority order: "
          f"{', '.join(_alternates(args)) if not args.no_alternates else '(none)'}")
    print(f"          stock right now: {live.get('stockStatus') or 'NONE AVAILABLE'}")
    if rate:
        rate = round(rate * getattr(args, "gpu_count", 1), 4)
    print(f"  cost    ${rate}/hr GPU (x{getattr(args, 'gpu_count', 1)}) + ~${disk:.3f}/hr disk "
          f"({args.container_disk} GB container + {args.volume} GB volume)")
    if rate:
        print(f"          = ~${rate + disk:.2f}/hr, ~${(rate + disk) * 24:.2f}/day if left running")
    print(f"  ssh key {'from ' + str(pathlib.Path.home() / '.ssh' / 'id_ed25519.pub') if _ssh_pubkey() else 'NONE -- the pod will be unreachable over ssh'}")
    print()
    print("  `stop` keeps the volume and keeps billing it (~$"
          f"{args.volume * 0.00278 / 24:.3f}/hr). `terminate` destroys it.")
    print("\nto actually create it, re-run with:  create --yes-i-will-pay")
    return 0


# --------------------------------------------------------------------------- #
# payload
# --------------------------------------------------------------------------- #

#: Files that must be in the tarball or the pod cannot do its job. Checked after the
#: tar is built, because a payload that silently ships without the script it exists to
#: deliver is the failure mode this check was added for.
REQUIRED = (
    "scripts/cloud/runpod_bootstrap.sh",
    "scripts/cloud/pod_doctor.py",
    "benchmarks/suites/speed.py",
    "src/tabicl/__init__.py",
)


def cmd_payload(args: argparse.Namespace) -> int:
    """The working tree as one tar.gz -- no network, no credential.

    `--cached --others --exclude-standard`, i.e. tracked files AND untracked ones that
    .gitignore does not exclude, read from the WORKING TREE rather than from HEAD. Not
    `git archive HEAD`: that omits everything not yet committed, and the thing you edit
    immediately before a launch is the launch script.

    .gitignore already excludes `benchmarks/_cache/`, `benchmarks/_results/` and
    `__pycache__`, so it is also the correct exclusion list -- the pod refetches its own
    dataset cache and writes its own ledger.

    `.sh` files are rewritten to LF on the way in. A shell script carrying `#!/bin/bash\\r`
    dies with "bad interpreter" AFTER a transfer that reported success, which is the most
    confusing possible time to find out.
    """
    out = pathlib.Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    listing = subprocess.run(
        ["git", "ls-files", "-z", "--cached", "--others", "--exclude-standard"],
        cwd=REPO, check=True, capture_output=True).stdout
    names = sorted({n.decode() for n in listing.split(b"\0") if n})

    def _normalize(info: tarfile.TarInfo) -> tarfile.TarInfo:
        """Own every entry as root:root.

        A tarball written on macOS carries this machine's uid/gid (501/50). Extracting it as
        root on the pod makes tar try to honour that, fail with 'Cannot change ownership',
        and exit non-zero AFTER writing the files -- so `tar xzf ... && next_step` silently
        stops at a step that actually succeeded. Normalizing here is better than remembering
        --no-same-owner at the far end.
        """
        info.uid = info.gid = 0
        info.uname = info.gname = "root"
        return info

    with tarfile.open(out, "w:gz") as tar:
        for name in names:
            source = REPO / name
            if not source.is_file():
                continue
            if name.endswith(".sh"):
                blob = source.read_bytes().replace(b"\r\n", b"\n")
                info = tarfile.TarInfo(f"tabicl/{name}")
                info.size, info.mode = len(blob), 0o755
                tar.addfile(_normalize(info), io.BytesIO(blob))
            else:
                tar.add(source, arcname=f"tabicl/{name}", filter=_normalize)

        # Stamp the commit into the payload. The pod gets a tarball, not a clone, so
        # `git rev-parse` there returns nothing and every ledger row comes back with
        # commit_sha=None -- provenance the harness declares mandatory, silently absent
        # exactly where the expensive measurements are taken. Caught on the first real sweep.
        head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO,
                              capture_output=True, text=True, check=False).stdout.strip()
        dirty = subprocess.run(["git", "status", "--porcelain"], cwd=REPO,
                               capture_output=True, text=True, check=False).stdout.strip()
        blob = json.dumps({"commit_sha": head or None, "dirty": bool(dirty)}, indent=2).encode()
        info = tarfile.TarInfo("tabicl/.payload_provenance.json")
        info.size, info.mode = len(blob), 0o644
        tar.addfile(_normalize(info), io.BytesIO(blob))

    missing = [r for r in REQUIRED if r not in names]
    if missing:
        sys.exit(f"payload is missing {', '.join(missing)} -- it would be useless on the pod")

    print(f"{out}  {out.stat().st_size / 1e6:.2f} MB  ({len(names)} files)")
    print("\npush it with:")
    print(f"  python {pathlib.Path(__file__).name} ssh POD_ID    # prints the scp line")
    return 0


# --------------------------------------------------------------------------- #
# pod lifecycle
# --------------------------------------------------------------------------- #

def cmd_create(args: argparse.Namespace) -> int:
    if not args.yes_i_will_pay:
        cmd_plan(args)
        print("\nREFUSING: create bills continuously from the moment the pod starts.\n"
              "Pass --yes-i-will-pay if that is what you want.")
        return 1
    existing = gql("query { myself { pods { id name desiredStatus } } }",
                   verbose=args.verbose).get("myself", {}).get("pods") or []
    clash = [p for p in existing if p["name"] == args.name and p["desiredStatus"] == "RUNNING"]
    if clash and not args.allow_duplicate:
        print(f"REFUSING: {clash[0]['id']} is already RUNNING as '{args.name}'. "
              f"Use it, or pass --allow-duplicate.")
        return 1
    pod = _request(REST + "/pods", method="POST", body=create_body(args), verbose=args.verbose)
    print(json.dumps(pod, indent=2))
    print(f"\ncreated {pod.get('id')} -- it is billing now. "
          f"Next: {pathlib.Path(__file__).name} ssh {pod.get('id')}")
    return 0


def cmd_ssh(args: argparse.Namespace) -> int:
    pod = _request(f"{REST}/pods/{args.pod_id}", verbose=args.verbose)
    ip, ports = pod.get("publicIp"), pod.get("portMappings") or {}
    port = ports.get("22")
    if not (ip and port):
        print(f"pod {args.pod_id} has no public 22/tcp mapping yet "
              f"(status {pod.get('desiredStatus')}); it may still be starting.")
        return 1
    print(f"# {pod.get('name')} -- {pod.get('imageName')}  ${pod.get('costPerHr')}/hr")
    print(f"ssh -p {port} root@{ip}")
    print(f"scp -P {port} PAYLOAD.tgz root@{ip}:{MOUNT}/")
    print(f"ssh -p {port} root@{ip} 'cd {MOUNT} && tar xzf PAYLOAD.tgz && "
          f"bash tabicl/scripts/cloud/runpod_bootstrap.sh'")
    print("# results back -- the ledger is the deliverable, everything else is reproducible:")
    print(f"scp -P {port} -r root@{ip}:{MOUNT}/tabicl/benchmarks/_results ./benchmarks/_results.pod")
    print(f"scp -P {port} root@{ip}:{MOUNT}/tabicl/pod_doctor.json ./pod_doctor.json")
    print(f"scp -P {port} -r root@{ip}:{MOUNT}/ckpt ./ckpt.pod   # ablation checkpoints, if any")
    return 0


def cmd_stop(args: argparse.Namespace) -> int:
    print(json.dumps(_request(f"{REST}/pods/{args.pod_id}/stop", method="POST",
                              verbose=args.verbose), indent=2))
    print("stopped. The volume persists AND still bills. `terminate` to stop paying entirely.")
    return 0


def cmd_terminate(args: argparse.Namespace) -> int:
    if not args.yes_destroy_the_volume:
        print("REFUSING: terminate destroys the pod volume -- the results ledger, ablation "
              "checkpoints, the HF cache. Copy anything you want first (`ssh POD_ID` prints the scp lines), "
              "then pass --yes-destroy-the-volume.")
        return 1
    _request(f"{REST}/pods/{args.pod_id}", method="DELETE", verbose=args.verbose)
    print(f"terminated {args.pod_id}")
    return 0


# --------------------------------------------------------------------------- #

def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--verbose", action="store_true", help="print request URLs and bodies")
    subparsers = parser.add_subparsers(dest="command", required=True)

    def add_recipe_flags(sub):
        sub.add_argument("--lane", default=DEFAULT_LANE, choices=sorted(LANES),
                         help=f"workload preset: {', '.join(sorted(LANES))} (default {DEFAULT_LANE})")
        sub.add_argument("--gpu", default=None,
                         help="override the lane's card; disables alternates")
        sub.add_argument("--name", default=POD_NAME)
        sub.add_argument("--image", default=IMAGE,
                         help=f"default {IMAGE}; fallback {IMAGE_FALLBACK}")
        sub.add_argument("--cloud", default="COMMUNITY", choices=("COMMUNITY", "SECURE"))
        sub.add_argument("--spot", action="store_true",
                         help="interruptible; cheaper, and can be reclaimed mid-run")
        sub.add_argument("--container-disk", type=int, default=CONTAINER_DISK_GB)
        sub.add_argument("--volume", type=int, default=VOLUME_GB)
        sub.add_argument("--gpu-count", type=int, default=1,
                         help="GPUs on the pod. Independent arms of one comparison belong on ONE "
                              "multi-GPU pod: same host, same card, same driver, so the only "
                              "difference between them is the thing being ablated.")
        sub.add_argument("--no-alternates", action="store_true",
                         help="fail rather than substitute a different card")

    subparsers.add_parser("check").set_defaults(func=cmd_check)

    gpus = subparsers.add_parser("gpus")
    gpus.add_argument("--min-vram", type=int, default=0)
    gpus.add_argument("--ids", action="store_true",
                      help="print the real gpuTypeId next to the displayName; the id is what "
                           "`create` validates against and they are NOT the same string")
    gpus.set_defaults(func=cmd_gpus)

    plan = subparsers.add_parser("plan")
    add_recipe_flags(plan)
    plan.set_defaults(func=cmd_plan)

    payload = subparsers.add_parser("payload")
    payload.add_argument("--out", default=str(REPO.parent / "tabicl_payload.tgz"))
    payload.set_defaults(func=cmd_payload)

    create = subparsers.add_parser("create")
    add_recipe_flags(create)
    create.add_argument("--yes-i-will-pay", action="store_true")
    create.add_argument("--allow-duplicate", action="store_true")
    create.set_defaults(func=cmd_create)

    for name, func in (("ssh", cmd_ssh), ("stop", cmd_stop), ("terminate", cmd_terminate)):
        sub = subparsers.add_parser(name)
        sub.add_argument("pod_id")
        if name == "terminate":
            sub.add_argument("--yes-destroy-the-volume", action="store_true")
        sub.set_defaults(func=func)

    args = parser.parse_args()
    # Resolve the lane's card only when --gpu was not given, and remember which happened:
    # an explicit card must not be quietly swapped for a lane alternate.
    if hasattr(args, "lane"):
        args.gpu_explicit = args.gpu is not None
        if args.gpu is None:
            args.gpu = LANES[args.lane]["gpu"]
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
