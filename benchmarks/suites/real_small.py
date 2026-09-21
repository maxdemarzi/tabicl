"""BM-03 -- fixed real-data accuracy suite (and, since TP-07, regression suites too).

The per-ablation accuracy signal. Sized so a full sweep runs in hours rather than
days: ten frozen OpenML datasets, capped rows, a small fold count. Bigger boards
(BeyondArena, STRABLE, MulTaBench) are reserved for phase gates.

The dataset list and the fold construction are frozen -- see
:mod:`benchmarks._core.datasets`. Comparisons across time are only meaningful if
the evaluation set does not move underneath them.

Usage
-----
    PYTHONPATH=src python -m benchmarks.suites.real_small \\
        --run-id BM-05-baseline --config-id v2-default

    # compare two configurations in one sweep
    PYTHONPATH=src python -m benchmarks.suites.real_small \\
        --run-id TP-11-ablation --config-id no-power --norm-methods none

    # a regression suite is scored with TabICLRegressor, on CRPS / RMSE / MAE / R^2
    PYTHONPATH=src python -m benchmarks.suites.real_small --suite ctr23 \\
        --run-id TP-07 --config-id joint-step20000 --model-path joint.pt
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from .._core import datasets as ds
from .._core.runner import Runner, Task
from .._core.schema import Ledger

SUITE = "real_small"
METRICS = ("accuracy", "balanced_accuracy", "roc_auc", "log_loss")
REG_METRICS = ("crps", "rmse", "mae", "r2")
METRICS_BY_TASK = {"classification": METRICS, "regression": REG_METRICS}

#: Quantile levels CRPS is integrated over. Fixed, because a coarser or narrower grid
#: underestimates CRPS (see benchmarks/_core/scoring.py) and two arms must share one grid.
CRPS_LEVELS = tuple(round(0.01 * k, 2) for k in range(1, 100))


def _score(y_true, y_pred, y_proba, classes) -> Dict[str, float]:
    """Four complementary views of classifier quality.

    Accuracy alone hides calibration changes, and several backlog items
    (TP-02's ECDF features, TP-11's ensemble removal) are expected to move
    calibration more than they move accuracy. ``log_loss`` is the one to watch
    there; it is also the closest local proxy for what ScoringBench measures.
    """

    from sklearn.metrics import (  # noqa: PLC0415
        accuracy_score,
        balanced_accuracy_score,
        log_loss,
        roc_auc_score,
    )

    out = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
    }
    try:
        out["log_loss"] = float(log_loss(y_true, y_proba, labels=list(classes)))
    except Exception:
        pass
    try:
        if len(classes) == 2:
            out["roc_auc"] = float(roc_auc_score(y_true, y_proba[:, 1]))
        else:
            out["roc_auc"] = float(roc_auc_score(y_true, y_proba, multi_class="ovr", labels=list(classes)))
    except Exception:
        pass
    return out


def _score_regression(y_true, mean, quantiles, levels, y_train) -> Dict[str, float]:
    """CRPS, RMSE, MAE and R^2, on targets standardized by the training fold.

    Standardizing puts every dataset on one scale, so a paired test across datasets is not
    dominated by whichever has the largest target units. R^2 is scale-free anyway. CRPS is
    the headline: it scores the whole predictive distribution, which is what the regression
    head is trained for (pinball loss over quantiles), where RMSE sees only the mean.
    """

    from sklearn.metrics import r2_score  # noqa: PLC0415

    from .._core.scoring import crps_from_quantiles  # noqa: PLC0415

    mu = float(np.mean(y_train))
    sd = float(np.std(y_train)) or 1.0
    z = lambda a: (np.asarray(a, dtype=np.float64) - mu) / sd  # noqa: E731
    y, m = z(y_true), z(mean)
    return {
        "crps": crps_from_quantiles(y, z(quantiles), np.asarray(levels)),
        "rmse": float(np.sqrt(np.mean((y - m) ** 2))),
        "mae": float(np.mean(np.abs(y - m))),
        "r2": float(r2_score(y, m)),
    }


def make_tasks(
    specs,
    n_folds: int,
    max_rows: int,
    config_id: str,
    seed: int,
    clf_kwargs: Dict[str, Any],
    checkpoint: Optional[str],
    suite: str = SUITE,
) -> List[Task]:
    """One task per (dataset, fold).

    ``suite`` is recorded on every row and is part of the resume key, and it must name the
    DATASET SUITE, not this runner. Two suites can contain datasets with the same name that
    are not the same dataset -- ``steel-plates-fault`` is OpenML 1504 (binary) in
    ``real_small`` and 40982 (7-class) in ``cc18_narrow`` -- and rows from both written under
    one suite label would be averaged together by the aggregator without any error.
    """

    tasks: List[Task] = []
    for spec in specs:
        for fold in range(n_folds):
            tasks.append(
                Task(
                    suite=suite,
                    dataset=spec.name,
                    fold=fold,
                    config_id=config_id,
                    metrics=METRICS_BY_TASK[spec.task],
                    config={"max_rows": max_rows, "n_folds": n_folds, "openml_id": spec.openml_id,
                            **clf_kwargs},
                    seed=seed,
                    checkpoint=checkpoint,
                    payload={"spec": spec, "fold": fold, "max_rows": max_rows, "n_folds": n_folds},
                )
            )
    return tasks


def make_executor(clf_kwargs: Dict[str, Any]):
    def execute(task: Task) -> Dict[str, float]:
        from tabicl import TabICLClassifier, TabICLRegressor  # noqa: PLC0415

        spec = task.payload["spec"]
        seed = task.seed or 0

        X, y, _digest = ds.load(spec)
        X, y = ds.subsample(X, y, task.payload["max_rows"], seed=seed)

        folds = ds.make_folds(len(y), y, task.payload["n_folds"], seed=seed, task=spec.task)
        train_idx, test_idx = folds[task.payload["fold"]]

        X_train = X.iloc[train_idx] if hasattr(X, "iloc") else X[train_idx]
        X_test = X.iloc[test_idx] if hasattr(X, "iloc") else X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        if spec.task == "regression":
            y_train, y_test = y_train.astype(np.float64), y_test.astype(np.float64)
            reg = TabICLRegressor(random_state=seed, **clf_kwargs)
            reg.fit(X_train, y_train)
            pred = reg.predict(X_test, output_type=["mean", "quantiles"], alphas=list(CRPS_LEVELS))
            return _score_regression(y_test, pred["mean"], pred["quantiles"], CRPS_LEVELS, y_train)

        clf = TabICLClassifier(random_state=seed, **clf_kwargs)
        clf.fit(X_train, y_train)
        y_proba = clf.predict_proba(X_test)
        y_pred = clf.classes_[np.argmax(y_proba, axis=1)]

        return _score(y_test, y_pred, y_proba, clf.classes_)

    return execute


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--config-id", default="v2-default")
    parser.add_argument("--suite", default="real_small", choices=sorted(ds.SUITES))
    parser.add_argument("--n-folds", type=int, default=3)
    parser.add_argument("--max-rows", type=int, default=3000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default=None)
    parser.add_argument("--n-estimators", type=int, default=8)
    parser.add_argument("--norm-methods", default=None, help="Comma-separated, e.g. 'none' or 'none,power' (TP-11)")
    parser.add_argument("--checkpoint", default=None, help="Checkpoint identity, recorded on rows")
    parser.add_argument("--model-path", default=None,
                        help="Load weights from this file instead of the released checkpoint. This is "
                             "how a proxy-training checkpoint is evaluated; without it every run "
                             "silently measures the released model.")
    parser.add_argument("--datasets", default=None, help="Comma-separated subset of the frozen suite")
    parser.add_argument("--ledger", default="benchmarks/_results/real_small.jsonl")
    parser.add_argument("--no-resume", action="store_true")
    args = parser.parse_args(argv)

    specs = ds.SUITES[args.suite]
    if args.datasets:
        wanted = {n.strip() for n in args.datasets.split(",")}
        specs = tuple(s for s in specs if s.name in wanted)
        if not specs:
            parser.error(f"no datasets in suite '{args.suite}' matched {sorted(wanted)}")

    clf_kwargs: Dict[str, Any] = {"n_estimators": args.n_estimators}
    if args.device is not None:
        clf_kwargs["device"] = args.device
    if args.model_path is not None:
        clf_kwargs["model_path"] = args.model_path
    if args.norm_methods is not None:
        clf_kwargs["norm_methods"] = [m.strip() for m in args.norm_methods.split(",")]

    ledger = Ledger(Path(args.ledger))
    runner = Runner(ledger, run_id=args.run_id, device=args.device, resume=not args.no_resume)
    runner.warn_if_not_local_tabicl()

    tasks = make_tasks(specs, args.n_folds, args.max_rows, args.config_id, args.seed, clf_kwargs,
                       args.checkpoint, suite=args.suite)
    executed = runner.run(tasks, make_executor(clf_kwargs))
    print(f"\n{executed} tasks executed -> {args.ledger}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
