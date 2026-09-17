"""Dataset fetch, on-disk cache, and frozen fold definitions.

Two rules govern this module, both of them about not fooling ourselves later:

- **The suite list is frozen.** :data:`REAL_SMALL` is fixed on the day it is
  validated and does not change. A moving evaluation set makes every historical
  row in the ledger incomparable, which quietly destroys the whole point of
  keeping one.
- **Folds are deterministic.** Splits come from a seed and the dataset's own row
  count, never from wall-clock or set iteration order, so a result recorded in
  March can be reproduced in July.

Datasets are fetched from OpenML through scikit-learn's ``fetch_openml``, which
avoids a dependency on the ``openml`` package, and cached as ``.npz`` under
``benchmarks/_cache`` (gitignored).
"""

from __future__ import annotations

import hashlib
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Literal, Optional, Sequence, Tuple

import numpy as np

CACHE_DIR = Path(__file__).resolve().parents[1] / "_cache"

Task = Literal["classification", "regression"]


@dataclass(frozen=True)
class DatasetSpec:
    """A frozen reference to one evaluation dataset.

    Attributes
    ----------
    name : str
        Stable identifier used in the ledger. Never renamed -- doing so orphans
        every historical row for that dataset.

    openml_id : int
        OpenML data id. Pinned by id rather than name+version because ids are
        immutable.

    task : {"classification", "regression"}

    slices : tuple of str
        Which BeyondArena-style slices this dataset belongs to
        (``"high_card"``, ``"text"``, ``"low_dim"``, ``"high_dim"``). Lets a
        sweep be aggregated per slice, which is how TP-08 and TP-12 are judged.
    """

    name: str
    openml_id: int
    task: Task
    slices: Tuple[str, ...] = ()


#: Frozen suite for BM-03. Validated with ``python -m benchmarks._core.datasets --validate``;
#: see ``benchmarks/RESULTS.md`` for the validation record. **Do not edit this list** --
#: add a new named suite instead.
REAL_SMALL: Tuple[DatasetSpec, ...] = (
    DatasetSpec("credit-g", 31, "classification", ("low_dim",)),
    DatasetSpec("diabetes", 37, "classification", ("low_dim",)),
    DatasetSpec("vehicle", 54, "classification", ("low_dim",)),
    DatasetSpec("kc1", 1067, "classification", ("low_dim",)),
    DatasetSpec("phoneme", 1489, "classification", ("low_dim",)),
    DatasetSpec("qsar-biodeg", 1494, "classification", ("high_dim",)),
    DatasetSpec("wdbc", 1510, "classification", ("high_dim",)),
    DatasetSpec("blood-transfusion", 1464, "classification", ("low_dim",)),
    DatasetSpec("banknote-authentication", 1462, "classification", ("low_dim",)),
    DatasetSpec("steel-plates-fault", 1504, "classification", ("high_dim",)),
)

SUITES: Dict[str, Tuple[DatasetSpec, ...]] = {"real_small": REAL_SMALL}


def _cache_path(spec: DatasetSpec) -> Path:
    return CACHE_DIR / f"openml_{spec.openml_id}_{spec.name}.pkl"


def content_hash(X: np.ndarray, y: np.ndarray) -> str:
    """Stable digest of a fetched dataset.

    Recorded alongside results so that a silently re-uploaded OpenML dataset
    shows up as a hash mismatch instead of as an unexplained score change.
    """

    h = hashlib.sha256()
    h.update(np.ascontiguousarray(np.asarray(X, dtype=object).astype(str)).tobytes())
    h.update(np.ascontiguousarray(np.asarray(y).astype(str)).tobytes())
    return h.hexdigest()[:16]


def load(spec: DatasetSpec, cache_dir: Optional[Path] = None, force: bool = False):
    """Fetch a dataset, using the on-disk cache when present.

    Returns
    -------
    X : pandas.DataFrame
        Features, dtypes preserved -- categorical and string columns are left as
        they are, since TP-12 is specifically about how those are handled.

    y : ndarray

    digest : str
        :func:`content_hash` of the fetched data.
    """

    cache_dir = Path(cache_dir) if cache_dir is not None else CACHE_DIR
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = cache_dir / f"openml_{spec.openml_id}_{spec.name}.pkl"

    if path.exists() and not force:
        with path.open("rb") as fh:
            payload = pickle.load(fh)
        return payload["X"], payload["y"], payload["digest"]

    from sklearn.datasets import fetch_openml  # noqa: PLC0415

    bunch = fetch_openml(data_id=spec.openml_id, as_frame=True, parser="auto")
    X = bunch.data
    y = np.asarray(bunch.target)
    digest = content_hash(X.to_numpy(), y)

    with path.open("wb") as fh:
        pickle.dump({"X": X, "y": y, "digest": digest}, fh)
    return X, y, digest


def make_folds(
    n_samples: int,
    y: np.ndarray,
    n_folds: int,
    seed: int,
    task: Task,
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """Deterministic cross-validation splits.

    Stratified for classification so rare classes survive into every fold;
    plain K-fold for regression. Seeded, so the same call always yields the same
    splits.
    """

    from sklearn.model_selection import KFold, StratifiedKFold  # noqa: PLC0415

    if task == "classification":
        splitter = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
        return list(splitter.split(np.zeros(n_samples), y))
    splitter = KFold(n_splits=n_folds, shuffle=True, random_state=seed)
    return list(splitter.split(np.zeros(n_samples)))


def subsample(X, y, max_rows: int, seed: int):
    """Cap dataset size, preserving order.

    ScoringBench subsamples every dataset to 3,000 rows; BM-03 caps rows so a
    full sweep stays in the hours-not-days budget the plan asks for.
    """

    n = len(y)
    if n <= max_rows:
        return X, y
    rng = np.random.default_rng(seed)
    idx = np.sort(rng.choice(n, size=max_rows, replace=False))
    X_sub = X.iloc[idx] if hasattr(X, "iloc") else X[idx]
    return X_sub, y[idx]


def describe(spec: DatasetSpec) -> Dict[str, object]:
    """Fetch a dataset and summarise it, for suite validation."""

    X, y, digest = load(spec)
    n_rows, n_cols = X.shape
    dtypes = X.dtypes.astype(str)
    n_cat = int(sum(1 for d in dtypes if d in ("object", "category", "string")))
    max_card = 0
    for col in X.columns:
        if str(X[col].dtype) in ("object", "category", "string"):
            max_card = max(max_card, int(X[col].nunique()))
    return {
        "name": spec.name,
        "openml_id": spec.openml_id,
        "task": spec.task,
        "rows": n_rows,
        "cols": n_cols,
        "cat_cols": n_cat,
        "max_cardinality": max_card,
        "n_classes": int(len(np.unique(y))) if spec.task == "classification" else 0,
        "digest": digest,
    }


def main(argv: Optional[Sequence[str]] = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description="Validate and cache a frozen dataset suite.")
    parser.add_argument("--suite", default="real_small", choices=sorted(SUITES))
    parser.add_argument("--validate", action="store_true", help="Fetch each dataset and print a summary")
    args = parser.parse_args(argv)

    specs = SUITES[args.suite]
    if not args.validate:
        for spec in specs:
            print(f"{spec.name:28s} id={spec.openml_id:<6d} {spec.task}")
        return 0

    header = f"{'name':28s} {'id':>6s} {'rows':>7s} {'cols':>5s} {'cat':>4s} {'maxcard':>8s} {'cls':>4s}  digest"
    print(header)
    print("-" * len(header))
    failures = []
    for spec in specs:
        try:
            info = describe(spec)
        except Exception as exc:  # noqa: BLE001
            failures.append((spec.name, repr(exc)[:120]))
            print(f"{spec.name:28s} {spec.openml_id:>6d}  FAILED: {repr(exc)[:70]}")
            continue
        print(
            f"{info['name']:28s} {info['openml_id']:>6d} {info['rows']:>7d} {info['cols']:>5d} "
            f"{info['cat_cols']:>4d} {info['max_cardinality']:>8d} {info['n_classes']:>4d}  {info['digest']}"
        )

    if failures:
        print(f"\n{len(failures)} dataset(s) failed to fetch:")
        for name, err in failures:
            print(f"  {name}: {err}")
        return 1
    print(f"\nAll {len(specs)} datasets fetched and cached.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
