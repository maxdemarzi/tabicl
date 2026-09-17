"""BM-02 -- inference speed, memory, and KV-cache size.

Mirrors Figure 7 of the TabPFN-3.5 technical report so our numbers are directly
comparable to theirs:

- forward (fit + predict) time as training rows grow, at fixed column count
- cached-predict time for 1 and for 100 test rows, against a cache built over
  the training rows
- **KV-cache size in bytes**, which their figure omits

That last metric is the point of the exercise. TP-05 (grouped-query attention)
exists to hold cache size flat while TP-06 quadruples the parameter count, and
TP-02 (in-context ECDF features) risks reintroducing an O(n_train) term into the
cache if TP-03 does not land with it. Neither is visible in an accuracy number.
The single-test-row cached time is the online-serving case: bound by the memory
bandwidth of moving the cache and weights, not by compute.

Usage
-----
    python -m benchmarks.suites.speed --run-id BM-05-baseline --rows 1000,4000
    python -m benchmarks.suites.speed --run-id BM-05-baseline --preset full

The ``full`` preset reproduces the report's 1k->1M sweep and needs a CUDA box
with substantial memory; the default preset is sized to run anywhere.
"""

from __future__ import annotations

import argparse
import statistics
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from .._core.runner import Runner, Task
from .._core.schema import Ledger

SUITE = "speed"

PRESETS = {
    # Runs on a laptop. Enough to exercise the path and catch gross regressions.
    "small": [1000, 2000, 4000],
    # Intermediate; needs a GPU but not a large one.
    "medium": [1000, 2000, 4000, 8000, 16000, 32000],
    # The report's sweep. CUDA, lots of memory.
    "full": [1000, 2000, 4000, 8000, 16000, 32000, 64000, 128000, 256000, 512000, 1000000],
}


def _cache_bytes(estimator) -> Optional[int]:
    """Exact byte size of a fitted estimator's KV cache.

    ``TabICLCache.cache_size_mb()`` truncates to whole megabytes, which reports 0
    for the small caches used in the default preset. Cache size is a gate metric
    for TP-05, so it is summed in bytes here rather than read off that helper.
    """

    caches = getattr(estimator, "model_kv_cache_", None)
    if not caches:
        return None

    total = 0
    for cache in caches.values():
        for part in ("col_cache", "icl_cache"):
            sub = getattr(cache, part, None)
            if sub is None or not getattr(sub, "kv", None):
                continue
            for entry in sub.kv.values():
                for tensor in (entry.key, entry.value):
                    if tensor is not None:
                        total += tensor.numel() * tensor.element_size()
        row_repr = getattr(cache, "row_repr", None)
        if row_repr is not None:
            total += row_repr.numel() * row_repr.element_size()
    return total


def _synth(n_rows: int, n_cols: int, seed: int) -> tuple:
    """Synthetic binary-classification data.

    Deliberately trivial to learn: this suite measures time and memory, and a
    hard problem would only add variance to the timings.
    """

    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n_rows, n_cols)).astype(np.float32)
    w = rng.standard_normal(n_cols).astype(np.float32)
    y = (X @ w > 0).astype(np.int64)
    return X, y


def _median_time(fn, repeats: int, warmup: int = 1) -> float:
    """Median wall-clock over repeats, after discarding warmup runs."""

    for _ in range(warmup):
        fn()
    samples = []
    for _ in range(repeats):
        start = time.perf_counter()
        fn()
        samples.append(time.perf_counter() - start)
    return float(statistics.median(samples))


def make_tasks(
    rows: List[int],
    n_cols: int,
    n_test: int,
    config_id: str,
    seed: int,
    checkpoint: Optional[str],
    repeats: int,
    device: Optional[str],
    n_estimators: int,
) -> List[Task]:
    """One task per (mode, training-row count)."""

    tasks: List[Task] = []
    common: Dict[str, Any] = {
        "n_cols": n_cols,
        "n_test": n_test,
        "repeats": repeats,
        "device": device,
        "n_estimators": n_estimators,
    }

    for n_rows in rows:
        tasks.append(
            Task(
                suite=SUITE,
                dataset=f"uncached/rows={n_rows}/cols={n_cols}",
                fold=0,
                config_id=config_id,
                metrics=("fit_predict_s",),
                config={**common, "mode": "uncached", "n_rows": n_rows},
                seed=seed,
                checkpoint=checkpoint,
                payload={"n_rows": n_rows, "mode": "uncached"},
            )
        )
        tasks.append(
            Task(
                suite=SUITE,
                dataset=f"cached/rows={n_rows}/cols={n_cols}",
                fold=0,
                config_id=config_id,
                metrics=("cache_build_s", "cached_predict_1_s", "cached_predict_100_s", "cache_bytes"),
                config={**common, "mode": "cached", "n_rows": n_rows},
                seed=seed,
                checkpoint=checkpoint,
                payload={"n_rows": n_rows, "mode": "cached"},
            )
        )
    return tasks


def make_executor(n_cols: int, n_test: int, repeats: int, device: Optional[str], n_estimators: int):
    """Build the ``execute(task)`` callable the runner drives."""

    def execute(task: Task) -> Dict[str, float]:
        from tabicl import TabICLClassifier  # imported late so provenance is captured first

        n_rows = task.payload["n_rows"]
        mode = task.payload["mode"]
        X, y = _synth(n_rows, n_cols, seed=task.seed or 0)

        kwargs: Dict[str, Any] = {"n_estimators": n_estimators, "random_state": task.seed or 0}
        if device is not None:
            kwargs["device"] = device

        if mode == "uncached":
            X_test, _ = _synth(n_test, n_cols, seed=(task.seed or 0) + 1)

            def once():
                clf = TabICLClassifier(**kwargs)
                clf.fit(X, y)
                clf.predict(X_test)

            return {"fit_predict_s": _median_time(once, repeats=repeats)}

        # cached
        X1, _ = _synth(1, n_cols, seed=(task.seed or 0) + 2)
        X100, _ = _synth(100, n_cols, seed=(task.seed or 0) + 3)

        clf = TabICLClassifier(kv_cache=True, **kwargs)
        build_s = _median_time(lambda: clf.fit(X, y), repeats=1, warmup=0)

        out = {
            "cache_build_s": build_s,
            "cached_predict_1_s": _median_time(lambda: clf.predict(X1), repeats=repeats),
            "cached_predict_100_s": _median_time(lambda: clf.predict(X100), repeats=repeats),
        }
        size = _cache_bytes(clf)
        if size is not None:
            out["cache_bytes"] = float(size)
        return out

    return execute


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run-id", required=True, help="Groups these rows, e.g. BM-05-baseline")
    parser.add_argument("--config-id", default="v2-default", help="Method identity for aggregation")
    parser.add_argument("--preset", choices=sorted(PRESETS), default="small")
    parser.add_argument("--rows", default=None, help="Comma-separated row counts, overrides --preset")
    parser.add_argument("--cols", type=int, default=100, help="Feature count (report uses 100)")
    parser.add_argument("--n-test", type=int, default=1024, help="Test rows for the uncached forward")
    parser.add_argument("--n-estimators", type=int, default=8)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default=None, help="cuda / mps / cpu; default lets tabicl choose")
    parser.add_argument("--checkpoint", default=None, help="Checkpoint identity, recorded on rows")
    parser.add_argument("--ledger", default="benchmarks/_results/speed.jsonl")
    parser.add_argument("--no-resume", action="store_true")
    args = parser.parse_args(argv)

    rows = [int(r) for r in args.rows.split(",")] if args.rows else PRESETS[args.preset]

    ledger = Ledger(Path(args.ledger))
    runner = Runner(ledger, run_id=args.run_id, device=args.device, resume=not args.no_resume)
    runner.warn_if_not_local_tabicl()

    tasks = make_tasks(
        rows=rows,
        n_cols=args.cols,
        n_test=args.n_test,
        config_id=args.config_id,
        seed=args.seed,
        checkpoint=args.checkpoint,
        repeats=args.repeats,
        device=args.device,
        n_estimators=args.n_estimators,
    )
    executor = make_executor(args.cols, args.n_test, args.repeats, args.device, args.n_estimators)

    executed = runner.run(tasks, executor)
    print(f"\n{executed} tasks executed -> {args.ledger}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
