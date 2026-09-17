"""Resumable execution of benchmark work items.

A sweep is a list of :class:`Task` plus a function that executes one. The runner
owns everything a suite should not have to reimplement: resume, timing, peak
memory, error capture, and provenance stamping.

Two properties matter more than they look:

- **Resume is keyed per (task, metric).** Long sweeps get interrupted -- by a
  cluster time limit, an OOM, or a laptop lid. Re-invoking picks up where it
  stopped instead of burning hours re-measuring.
- **Failures become rows.** A task that raises is recorded with its traceback
  rather than dropped, so a gap in a sweep is visible in the ledger instead of
  silently shrinking the dataset grid -- which would quietly flatter whichever
  method failed on the hard datasets.
"""

from __future__ import annotations

import gc
import time
import traceback
from dataclasses import asdict, dataclass, field
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence

from .schema import Ledger, ResultRow, capture_provenance


@dataclass(frozen=True)
class Task:
    """One unit of measurable work.

    Attributes
    ----------
    suite : str
        Suite name, e.g. ``"speed"`` or ``"real_small"``.

    dataset : str
        Dataset identifier. For non-dataset suites (speed sweeps) this is the
        varying condition, e.g. ``"rows=16k,cols=100"``.

    fold : int
        Split index, or 0 where the suite has no folds.

    config_id : str
        Identifies the method being compared. This is what aggregation treats as
        the "method", so it must distinguish exactly the thing an ablation varies
        -- e.g. ``"v2-clf-baseline"`` vs ``"v2-clf-fourier"``.

    metrics : tuple of str
        Metric names this task is expected to produce. Used for resume: a task
        whose metrics are all present is skipped.

    config : dict
        Full configuration, recorded verbatim for provenance.

    seed : int, optional
        Recorded on every row. Phase-gating decisions need >= 3 seeds.

    checkpoint : str, optional
        Checkpoint identity.

    payload : dict
        Arbitrary data the executor needs (paths, splits, model handles). Not
        serialized into the ledger.
    """

    suite: str
    dataset: str
    fold: int
    config_id: str
    metrics: Sequence[str]
    config: Dict[str, Any] = field(default_factory=dict)
    seed: Optional[int] = None
    checkpoint: Optional[str] = None
    payload: Dict[str, Any] = field(default_factory=dict, compare=False, hash=False)


def _peak_memory_reset(device: Optional[str]) -> None:
    if device is None:
        return
    try:
        import torch  # noqa: PLC0415

        if device.startswith("cuda") and torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()
    except Exception:
        pass


def _peak_memory_read(device: Optional[str]) -> Optional[int]:
    """Peak device memory in bytes, or None where the device cannot report it."""

    if device is None:
        return None
    try:
        import torch  # noqa: PLC0415

        if device.startswith("cuda") and torch.cuda.is_available():
            torch.cuda.synchronize()
            return int(torch.cuda.max_memory_allocated())
        if device == "mps" and torch.backends.mps.is_available():
            # MPS exposes current allocation only -- not a true peak. Recorded
            # because it is better than nothing, but not comparable to CUDA.
            return int(torch.mps.current_allocated_memory())
    except Exception:
        return None
    return None


def _synchronize(device: Optional[str]) -> None:
    """Make timings honest on asynchronous backends."""

    if device is None:
        return
    try:
        import torch  # noqa: PLC0415

        if device.startswith("cuda") and torch.cuda.is_available():
            torch.cuda.synchronize()
        elif device == "mps" and torch.backends.mps.is_available():
            torch.mps.synchronize()
    except Exception:
        pass


class Runner:
    """Executes tasks against a ledger, with resume.

    Parameters
    ----------
    ledger : Ledger
        Destination. Opened append-only.

    run_id : str
        Groups this sweep's rows. Conventionally the backlog item, e.g.
        ``"BM-05-baseline"``. Resume matches on this, so reusing a ``run_id``
        continues that sweep and changing it starts a fresh one.

    device : str, optional
        Passed through for memory/timing instrumentation and recorded on rows.

    resume : bool, default=True
        Skip tasks already completed under this ``run_id``.

    verbose : bool, default=True
    """

    def __init__(
        self,
        ledger: Ledger,
        run_id: str,
        device: Optional[str] = None,
        resume: bool = True,
        verbose: bool = True,
    ):
        self.ledger = ledger
        self.run_id = run_id
        self.device = device
        self.resume = resume
        self.verbose = verbose
        self._provenance = asdict(capture_provenance(device))

    def _log(self, msg: str) -> None:
        if self.verbose:
            print(msg, flush=True)

    def warn_if_not_local_tabicl(self) -> bool:
        """Warn when the imported ``tabicl`` is not this working tree.

        Benchmarking a released wheel while editing ``src/`` produces results that
        look fine and mean nothing. Returns True if the import is local.
        """

        is_local = self._provenance.get("tabicl_is_local")
        if is_local is False:
            self._log(
                "WARNING: imported tabicl is NOT this working tree\n"
                f"         imported: {self._provenance.get('tabicl_path')}\n"
                f"         version:  {self._provenance.get('tabicl_version')}\n"
                "         Local source changes will NOT be measured. "
                "Run `pip install -e .` to fix."
            )
        return bool(is_local)

    def _pending(self, tasks: Sequence[Task]) -> List[Task]:
        if not self.resume:
            return list(tasks)
        done = self.ledger.completed_keys(run_id=self.run_id)
        pending = []
        for task in tasks:
            keys = {
                (self.run_id, task.suite, task.dataset, task.fold, task.config_id, metric)
                for metric in task.metrics
            }
            if not keys.issubset(done):
                pending.append(task)
        return pending

    def run(
        self,
        tasks: Iterable[Task],
        execute: Callable[[Task], Dict[str, float]],
    ) -> int:
        """Execute tasks, appending one row per (task, metric).

        Parameters
        ----------
        tasks : iterable of Task

        execute : callable
            ``execute(task) -> {metric_name: value}``. Raising is allowed and is
            recorded as an error row; the sweep continues.

        Returns
        -------
        int
            Number of tasks executed (excluding those skipped by resume).
        """

        tasks = list(tasks)
        pending = self._pending(tasks)
        skipped = len(tasks) - len(pending)
        if skipped:
            self._log(f"[{self.run_id}] resuming: {skipped} of {len(tasks)} tasks already done")

        executed = 0
        for i, task in enumerate(pending, start=1):
            label = f"{task.suite}/{task.dataset}/fold{task.fold}/{task.config_id}"
            self._log(f"[{self.run_id}] ({i}/{len(pending)}) {label}")

            gc.collect()
            _peak_memory_reset(self.device)
            _synchronize(self.device)
            start = time.perf_counter()

            values: Dict[str, float] = {}
            error: Optional[str] = None
            try:
                values = execute(task)
            except Exception:
                error = traceback.format_exc(limit=8)
                self._log(f"[{self.run_id}]   FAILED: {error.strip().splitlines()[-1]}")

            _synchronize(self.device)
            elapsed = time.perf_counter() - start
            peak = _peak_memory_read(self.device)

            if error is not None:
                # One error row per expected metric, so resume retries all of them.
                for metric in task.metrics:
                    self._append(task, metric, None, elapsed, peak, error)
            else:
                for metric, value in values.items():
                    self._append(task, metric, value, elapsed, peak, None)

            executed += 1

        return executed

    def _append(
        self,
        task: Task,
        metric: str,
        value: Optional[float],
        elapsed: float,
        peak: Optional[int],
        error: Optional[str],
    ) -> None:
        self.ledger.append(
            ResultRow(
                run_id=self.run_id,
                suite=task.suite,
                dataset=task.dataset,
                fold=task.fold,
                config_id=task.config_id,
                metric=metric,
                value=None if value is None else float(value),
                seed=task.seed,
                checkpoint=task.checkpoint,
                device=self.device,
                config=task.config,
                wall_clock_s=elapsed,
                peak_mem_bytes=peak,
                error=error,
                provenance=self._provenance,
            )
        )
