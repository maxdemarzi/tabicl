"""Result row schema and provenance capture for the benchmark harness.

One measurement is one :class:`ResultRow`. Rows are appended to a JSONL file and
never edited in place -- a superseded result is superseded by a new row, not by
rewriting the old one (see ``benchmarks/RESULTS.md``).

Every row carries enough provenance to be reproduced: the commit it was measured
at, whether the working tree was dirty, which ``tabicl`` was actually imported,
the checkpoint, the seed, and the hardware. A number without provenance is not a
result, so :func:`capture_provenance` is not optional and rows cannot be built
without it.
"""

from __future__ import annotations

import json
import os
import platform
import socket
import subprocess
import sys
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, Optional

SCHEMA_VERSION = 1

REPO_ROOT = Path(__file__).resolve().parents[2]


def _git(*args: str) -> Optional[str]:
    """Run a git command in the repo, returning None if it fails for any reason."""

    try:
        out = subprocess.run(
            ["git", *args],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if out.returncode != 0:
        return None
    return out.stdout.strip()


@dataclass(frozen=True)
class Provenance:
    """Everything needed to know where a number came from.

    Attributes
    ----------
    commit_sha : str or None
        Full SHA of HEAD, or None outside a git checkout.

    dirty : bool or None
        Whether the working tree had uncommitted changes. A dirty tree does not
        block a run, but it makes the result unreproducible and is recorded so
        that it can be filtered out later.

    branch : str or None
        Current branch name.

    tabicl_version : str or None
        Version of the imported ``tabicl``.

    tabicl_path : str or None
        Filesystem path of the imported ``tabicl`` package. This is checked
        rather than assumed: a PyPI install shadowing the working tree silently
        benchmarks released code instead of local changes, which is the single
        easiest way to produce a confidently wrong ablation.

    tabicl_is_local : bool or None
        Whether the imported package resolves inside this repository's ``src/``.

    python_version, platform_name, hostname, torch_version, device_name : str or None
        Environment and hardware identification.
    """

    commit_sha: Optional[str] = None
    dirty: Optional[bool] = None
    branch: Optional[str] = None
    tabicl_version: Optional[str] = None
    tabicl_path: Optional[str] = None
    tabicl_is_local: Optional[bool] = None
    python_version: Optional[str] = None
    platform_name: Optional[str] = None
    hostname: Optional[str] = None
    torch_version: Optional[str] = None
    device_name: Optional[str] = None


def capture_provenance(device: Optional[str] = None) -> Provenance:
    """Collect provenance for the current process.

    Parameters
    ----------
    device : str, optional
        Device the measurement will run on (``"cuda"``, ``"mps"``, ``"cpu"``).
        Used to resolve a human-readable accelerator name.

    Returns
    -------
    Provenance
        Best-effort provenance. Fields that cannot be determined are None rather
        than guessed.
    """

    commit = _git("rev-parse", "HEAD")
    status = _git("status", "--porcelain")
    branch = _git("rev-parse", "--abbrev-ref", "HEAD")

    # On a pod the checkout is an unpacked tarball with no .git, so every git call above
    # returns None and the rows carry no commit -- which is precisely where the expensive
    # measurements happen. `runpod_launch.py payload` writes the stamp below for that case.
    dirty = None if status is None else bool(status)
    if commit is None:
        stamp = REPO_ROOT / ".payload_provenance.json"
        try:
            if stamp.is_file():
                payload = json.loads(stamp.read_text())
                commit = payload.get("commit_sha")
                dirty = payload.get("dirty", dirty)
        except Exception:
            pass

    tabicl_version = tabicl_path = None
    tabicl_is_local = None
    try:
        import tabicl  # noqa: PLC0415

        tabicl_path = os.path.realpath(tabicl.__file__)
        tabicl_is_local = tabicl_path.startswith(str(REPO_ROOT / "src"))
        tabicl_version = getattr(tabicl, "__version__", None)
        if tabicl_version is None:
            try:
                from importlib.metadata import version  # noqa: PLC0415

                tabicl_version = version("tabicl")
            except Exception:
                tabicl_version = None
    except Exception:
        pass

    torch_version = device_name = None
    try:
        import torch  # noqa: PLC0415

        torch_version = torch.__version__
        if device is not None and device.startswith("cuda") and torch.cuda.is_available():
            device_name = torch.cuda.get_device_name(0)
        elif device == "mps":
            device_name = f"mps ({platform.machine()})"
        elif device == "cpu":
            device_name = f"cpu ({platform.processor() or platform.machine()})"
    except Exception:
        pass

    return Provenance(
        commit_sha=commit,
        dirty=dirty,
        branch=branch,
        tabicl_version=tabicl_version,
        tabicl_path=tabicl_path,
        tabicl_is_local=tabicl_is_local,
        python_version=sys.version.split()[0],
        platform_name=platform.platform(),
        hostname=socket.gethostname(),
        torch_version=torch_version,
        device_name=device_name,
    )


@dataclass
class ResultRow:
    """A single measurement.

    The key ``(run_id, suite, dataset, fold, config_id, metric)`` identifies a
    row for resume purposes: :class:`~benchmarks._core.runner.Runner` skips work
    whose key is already present in the ledger.

    Attributes
    ----------
    run_id : str
        Groups rows produced by one invocation of a suite. Conventionally names
        the backlog item, e.g. ``"BM-05-baseline"`` or ``"TP-01-proxy-seed0"``.

    suite, dataset, fold, config_id, metric : str / int
        What was measured, on what, under which configuration.

    value : float or None
        The metric value. None when ``error`` is set.

    wall_clock_s, peak_mem_bytes : float / int or None
        Cost of the measurement. Recorded on every row because the target is a
        Pareto frontier, not a single accuracy number.

    error : str or None
        Failure reason. A failed measurement is recorded as a row rather than
        dropped, so that gaps in a sweep are visible instead of silent.
    """

    run_id: str
    suite: str
    dataset: str
    fold: int
    config_id: str
    metric: str
    value: Optional[float] = None

    seed: Optional[int] = None
    checkpoint: Optional[str] = None
    device: Optional[str] = None
    config: Dict[str, Any] = field(default_factory=dict)

    wall_clock_s: Optional[float] = None
    peak_mem_bytes: Optional[int] = None

    error: Optional[str] = None
    notes: Optional[str] = None

    schema_version: int = SCHEMA_VERSION
    row_id: str = field(default_factory=lambda: uuid.uuid4().hex)
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    provenance: Optional[Dict[str, Any]] = None

    def key(self) -> tuple:
        """Identity used for resume. See class docstring."""

        return (self.run_id, self.suite, self.dataset, self.fold, self.config_id, self.metric)

    def to_json(self) -> str:
        return json.dumps(asdict(self), sort_keys=True, default=str)


class Ledger:
    """Append-only JSONL store of :class:`ResultRow`.

    Opened in append mode and flushed after every write, so an interrupted sweep
    loses at most the row in flight. The file is the durable record; nothing in
    this class mutates rows already on disk.

    Parameters
    ----------
    path : str or Path
        JSONL file. Created along with parent directories if absent.
    """

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def append(self, row: ResultRow) -> None:
        """Append one row and flush it to disk."""

        with self.path.open("a", encoding="utf-8") as fh:
            fh.write(row.to_json() + "\n")
            fh.flush()
            os.fsync(fh.fileno())

    def extend(self, rows: Iterable[ResultRow]) -> None:
        for row in rows:
            self.append(row)

    def __iter__(self) -> Iterator[Dict[str, Any]]:
        """Yield rows as dicts, skipping any line that fails to parse.

        A truncated final line is expected after a hard kill and must not make
        the whole ledger unreadable.
        """

        if not self.path.exists():
            return
        with self.path.open("r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    yield json.loads(line)
                except json.JSONDecodeError:
                    continue

    def completed_keys(self, run_id: Optional[str] = None) -> set:
        """Keys of rows already recorded, for resume.

        Rows carrying an ``error`` are excluded, so a failed measurement is
        retried on the next invocation rather than treated as done.
        """

        keys = set()
        for row in self:
            if run_id is not None and row.get("run_id") != run_id:
                continue
            if row.get("error"):
                continue
            keys.add(
                (
                    row.get("run_id"),
                    row.get("suite"),
                    row.get("dataset"),
                    row.get("fold"),
                    row.get("config_id"),
                    row.get("metric"),
                )
            )
        return keys
