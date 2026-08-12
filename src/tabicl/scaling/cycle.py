"""Drive one experimental round on a RunPod host, end to end: build, upload, gate, poll, fetch, tear down.

    python -m tabicl.scaling.cycle --work round.sh --workdir runs/2026-08-11
    python -m tabicl.scaling.cycle --work round.sh --workdir runs/2026-08-11 --attach
    python -m tabicl.scaling.cycle --work round.sh --workdir runs/2026-08-11 --keep

`pod_runner` owns the pod lifecycle; this owns everything between a working host and a
fetched log. **It lives in the repository on purpose.** Every previous version was written
into a session scratchpad, fixed there, and lost when the session ended -- the notes record
four consecutive rounds that shipped nothing while each failure printed output shaped like a
failed experiment, and three more distinct failures were rediscovered afterwards. The fixes
below are cheap; rediscovering them costs a pod cycle each.

The failures this encodes, each of which cost a round:

* **Line endings.** `_pod_setup.sh` lives in a Windows checkout and was CRLF, so bash read
  line 2 as ``set -euo pipefail\\r`` and died with "pipefail: invalid option name" -- which
  reads like the wrong shell, not the wrong file. scp is a byte copy and will not save you,
  and ``grep -q $'\\r'`` reported the file clean where ``od -c`` did not. Scripts are
  normalised to LF on upload, so no Windows edit can reintroduce it.
* **A missing LICENSE.** `pyproject` declares ``license = { file = "LICENSE" }``, so
  hatchling fails the build without it and the payload had never included it.
* **A stale payload.** An archive that exists is not an archive that carries your change.
  The runner is compared BYTE FOR BYTE against the working tree before anything launches.
* **Fetching on the happy path.** Logs were retrieved at the end of the `try`, so the setup
  failure that actually happened raised straight past it and terminated the host with
  `setup.log` still on it. Fetching now happens in the `finally`, before teardown.
* **Shell strings.** An inline multi-line script crossed PowerShell quoting *and* ssh
  argv-joining and arrived as a syntax error; ``$(...)`` in an ssh command string was
  expanded locally and tried to run `pgrep` on Windows. Remote scripts ship as FILES.
"""

from __future__ import annotations

import argparse
import hashlib
import pathlib
import subprocess
import sys
import tarfile
import time

# Relative to this file, so the driver works from any checkout and any working directory.
REPO = pathlib.Path(__file__).resolve().parents[3]
RUNNER = "src/tabicl/scaling/eval_track_record.py"
PAYLOAD_PARTS = ("pyproject.toml", "README.md", "LICENSE", "src", "tests")


# ----------------------------------------------------------------------------- payload

def lf_bytes(data: bytes) -> bytes:
    """CRLF -> LF. A shell script with CRLF fails on the host in ways that look like a bad
    host rather than a bad file."""
    return data.replace(b"\r\n", b"\n")


def build_payload(repo: pathlib.Path, out: pathlib.Path,
                  parts: tuple[str, ...] = PAYLOAD_PARTS) -> None:
    """Tar the source with Python's `tarfile`, not the shell's.

    Removes two documented failures at the source rather than working around them: Git Bash
    `tar` reads ``C:\\...`` as a remote ``host:path`` and dies, and a Windows-built archive
    stamps uid 197610, which GNU tar as root exits 2 trying to honour -- after writing every
    file correctly, so it looks like a broken payload and is not.
    """
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.exists():
        out.unlink()        # a stale archive that fails to rebuild ships the last round
    def scrub(ti: tarfile.TarInfo) -> tarfile.TarInfo:
        ti.uid = ti.gid = 0
        ti.uname = ti.gname = "root"
        return ti
    with tarfile.open(out, "w:gz") as tar:
        for rel in parts:
            p = repo / rel
            if p.exists():
                tar.add(p, arcname=rel, filter=scrub)


def verify_payload(repo: pathlib.Path, out: pathlib.Path, rel: str = RUNNER) -> None:
    """Refuse to launch unless the archive carries the code under test, byte for byte.

    Not "does the file exist": the failure that cost four rounds was an archive that existed
    and was stale. Content is the only check that could have caught it.
    """
    with tarfile.open(out) as tar:
        try:
            shipped = tar.extractfile(rel).read()
        except KeyError:
            raise SystemExit(f"payload does not contain {rel} -- refusing to launch")
    local = (repo / rel).read_bytes()
    if hashlib.sha256(shipped).hexdigest() != hashlib.sha256(local).hexdigest():
        raise SystemExit(f"payload's {rel} differs from the working tree -- refusing to "
                         f"launch a round on code you are not looking at")


# ----------------------------------------------------------------------------- transfer

# `[e]val_track_rec` is the bracket trick, and it is load-bearing rather than a flourish.
# Written as `pgrep -f eval_track_record`, the pattern matches THE PROBE'S OWN command line --
# ssh runs it through a shell whose argv contains the pattern -- so the count was never below
# one on a healthy host. That silently disabled the "nothing running and nothing new
# finishing" break: a lane whose work had been stopped polled on to its deadline instead of
# fetching and tearing down, billing by the second the whole way. A guard that cannot fire is
# indistinguishable from one that never needed to.
PROBE = ("cat /workspace/WORK_DONE 2>/dev/null; "
         "echo ---; (cat /workspace/logs/RC 2>/dev/null; echo) | head -20; "
         "echo ---; pgrep -fc '[e]val_track_rec' || echo 0; "
         "echo ---; tail -3 /workspace/work.log 2>/dev/null")


def printable(s: str) -> str:
    """Make remote text safe for THIS console's encoding.

    `run_ssh_capture` decodes as UTF-8, which fixed reading. Printing is a second boundary and
    it broke separately: a Windows console is cp1252, and one non-ASCII character in a fetched
    log tail raised UnicodeEncodeError *inside the poll loop*, killing a healthy lane's driver.
    Remote output is data, and data must never be able to end a round -- the same rule the
    decode side already follows.
    """
    enc = getattr(sys.stdout, "encoding", None) or "utf-8"
    return s.encode(enc, errors="replace").decode(enc, errors="replace")


def parse_probe(rc: int, out: str):
    """Split one poll response into (done, finished, running, tail), or None if it is not one.

    **An ssh failure is not a finished round.** The first version ran ``out.partition("---")``
    unconditionally, so when the host was reclaimed mid-round and ssh printed "Connection
    refused", that text -- which contains no separators -- landed whole in the `done` slot.
    The poller announced WORK_DONE, stopped waiting, failed to fetch anything from the dead
    host, and tore the round down. A vanished host produced output shaped like a completed
    experiment, which is this project's signature failure rather than a new one.

    The probe emits exactly three separators, so anything without them is not a reply.
    """
    if rc != 0 or out.count("---") < 3:
        return None
    done, _, rest = out.partition("---")
    finished, _, rest2 = rest.partition("---")
    running, _, tail = rest2.partition("---")
    return done.strip(), finished.split(), running.strip(), tail.strip()


def _scp_argv(key, port, extra: list[str]) -> list[str]:
    return ["scp", *extra, "-i", str(key), "-P", str(port),
            "-o", "StrictHostKeyChecking=no", "-o", "UserKnownHostsFile=/dev/null",
            "-o", "LogLevel=ERROR"]


def scp_up(target, local: pathlib.Path, remote: str) -> None:
    from tabicl.scaling import pod_runner as pr
    host, port = target
    subprocess.run(_scp_argv(pr.KEY, port, []) + [str(local), f"root@{host}:{remote}"],
                   check=True)
    print(f"  uploaded {local.name} -> {remote}", flush=True)


def upload_script(target, local: pathlib.Path, remote: str,
                  tmpdir: pathlib.Path) -> None:
    """Upload a shell script with line endings normalised to LF."""
    tmp = tmpdir / f".lf_{local.name}"
    tmp.write_bytes(lf_bytes(local.read_bytes()))
    scp_up(target, tmp, remote)
    tmp.unlink()


def scp_down(target, remote: str, local: pathlib.Path) -> int:
    from tabicl.scaling import pod_runner as pr
    host, port = target
    return subprocess.run(_scp_argv(pr.KEY, port, ["-r"]) +
                          [f"root@{host}:{remote}", str(local)]).returncode


# --------------------------------------------------------------------------------- main

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--work", required=True, help="the round's work.sh")
    ap.add_argument("--workdir", default="cycle-run", help="payload and fetched logs")
    ap.add_argument("--keep", action="store_true", help="leave the pod running")
    ap.add_argument("--attach", action="store_true",
                    help="poll a round already running on the current pod")
    ap.add_argument("--extra-pip", default="",
                    help="packages this round needs beyond the benchmark's own (a backbone "
                         "arm, a forecasting adapter). Passed to _pod_setup.sh as "
                         "TABICL_EXTRA_PIP; a failure to install fails the gate.")
    ap.add_argument("--poll", type=int, default=180)
    ap.add_argument("--hours", type=float, default=5.0)
    args = ap.parse_args()
    if "'" in args.extra_pip:
        raise SystemExit("--extra-pip must not contain a single quote; it is passed through "
                         "a remote shell as a single-quoted word")

    from tabicl.scaling import pod_runner as pr
    import runpod

    workdir = pathlib.Path(args.workdir).resolve()
    workdir.mkdir(parents=True, exist_ok=True)
    payload = workdir / "payload.tar.gz"
    work_sh = pathlib.Path(args.work).resolve()
    if not work_sh.exists():
        raise SystemExit(f"no work script at {work_sh}")
    remote_sh = pathlib.Path(__file__).with_name("_pod_remote.sh")

    pr.auth()
    adopted = False
    if not args.attach:
        build_payload(REPO, payload)
        verify_payload(REPO, payload)
        print(f"payload {payload.stat().st_size / 1e6:.1f} MB, runner verified "
              f"byte-identical", flush=True)
        if pr.find_pod() is None:
            rc = subprocess.run([sys.executable, "-m", "tabicl.scaling.pod_runner",
                                 "create"]).returncode
            if rc != 0:
                raise SystemExit("could not obtain a usable host")
        else:
            adopted = True

    pod = pr.find_pod()
    if pod is None:
        raise SystemExit("no pod")
    _, target = pr.wait_ready(pod["id"])
    print(f"pod {pod['id']} at root@{target[0]}:{target[1]}", flush=True)

    if adopted:
        # ADOPTING A POD SKIPS `create`, AND `create` IS WHERE THE HOST IS PROBED. Four lanes
        # died at install on 2026-08-11 because a previous aborted launch had left pods under
        # their names: each lane found one, skipped creation, and ran on a host nothing had
        # validated for CUDA, torch version or bandwidth. The tell was in the startup -- no
        # `trying SECURE`, no `created`, no `USABLE` -- which is easy to miss precisely because
        # nothing failed there. A reused host has to clear the same bar as a fresh one.
        usable, why = pr.probe_host(target)
        if not usable:
            raise SystemExit(
                f"adopted the existing pod {pod['id']} and it FAILED the probe: {why}. "
                f"Terminate it and rerun so a fresh host is created and validated.")
        print(f"adopted an existing pod and probed it: {why}", flush=True)

    fetched = workdir / "results"
    fetched.mkdir(exist_ok=True)
    try:
        if not args.attach:
            scp_up(target, payload, "/workspace/payload.tar.gz")
            # Hugging Face token, if this machine has one. Uploaded as a FILE so it never
            # appears in an ssh argument, a remote process list, or a log line -- and the
            # driver prints only whether one was found, never its contents. Unauthenticated
            # Hub downloads are rate-limited, which on a large tier is paid at the large
            # tier's price with the GPU idle.
            hf = pathlib.Path.home() / ".cache" / "huggingface" / "token"
            if hf.exists():
                scp_up(target, hf, "/workspace/.hf_token")
                pr.run_ssh(target, "chmod 600 /workspace/.hf_token", timeout=120)
            else:
                print("  no HF token at ~/.cache/huggingface/token -- downloads will be "
                      "unauthenticated", flush=True)
            for f, dest in ((pathlib.Path(__file__).with_name("_pod_setup.sh"),
                             "/workspace/_pod_setup.sh"),
                            (work_sh, "/workspace/work.sh"),
                            (remote_sh, "/workspace/remote.sh")):
                upload_script(target, f, dest, workdir)
            print("\n--- setup + gate ---", flush=True)
            # Built as one argv element for ssh; `subprocess` runs no local shell, so nothing
            # here is expanded on this side. That distinction cost a cycle once, when `$(...)`
            # in an ssh command string was expanded locally and tried to run `pgrep` on
            # Windows.
            cmd = "bash /workspace/remote.sh"
            if args.extra_pip:
                cmd = f"TABICL_EXTRA_PIP='{args.extra_pip}' " + cmd
            rc = pr.run_ssh(target, cmd, timeout=3600)
            if rc != 0:
                raise SystemExit(f"remote.sh failed rc={rc}; work was NOT started")

        deadline = time.time() + args.hours * 3600
        ssh_fails = 0
        while time.time() < deadline:
            time.sleep(args.poll)
            probe = parse_probe(*pr.run_ssh_capture(target, PROBE, timeout=300))
            if probe is None:
                ssh_fails += 1
                print(f"[{time.strftime('%H:%M:%S')}] poll unanswered ({ssh_fails}/3) -- "
                      f"the host may have been reclaimed", flush=True)
                if ssh_fails >= 3:
                    print("HOST UNREACHABLE three polls running: the round is LOST, not "
                          "done. Nothing here may be reported as a result.", flush=True)
                    break
                continue
            ssh_fails = 0
            done, finished, running, tail = probe
            print(f"[{time.strftime('%H:%M:%S')}] finished={finished} running={running}",
                  flush=True)
            print(f"    {printable(tail[-300:])}", flush=True)
            if "WORK_DONE" in done:
                print("WORK_DONE", flush=True)
                break
            # A dead round never writes WORK_DONE, and a poller that waits only for it sits
            # on a billed idle pod for hours.
            if running in ("0", "") and finished:
                print("nothing running and nothing new finishing -- stopping the poll",
                      flush=True)
                break
        else:
            print("deadline reached", flush=True)
    finally:
        # FETCH BEFORE TEARDOWN, from the `finally`. Logs on a terminated pod are gone.
        print("\n--- fetching logs ---", flush=True)
        for remote in ("/workspace/logs", "/workspace/work.log", "/workspace/setup.log"):
            scp_down(target, remote, fetched)
        print(f"logs in {fetched}", flush=True)
        if args.keep:
            print("\npod LEFT RUNNING -- it bills by the second", flush=True)
        else:
            try:
                runpod.terminate_pod(pod["id"])
                print(f"\nterminated {pod['id']}", flush=True)
            except Exception as exc:                        # noqa: BLE001
                print(f"\nTEARDOWN FAILED: {exc} -- terminate {pod['id']} BY HAND",
                      flush=True)
        subprocess.run([sys.executable, "-m", "tabicl.scaling.pod_runner", "sweep"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
