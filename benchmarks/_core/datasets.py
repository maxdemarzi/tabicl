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

#: BM-08. OpenML-CC18 (study 99) restricted to datasets with at most 500 features:
#: 62 of its 72 datasets. Sized to the ablation question rather than to convenience --
#: see benchmarks/RESULTS.md: the 10-dataset REAL_SMALL suite detects a true ~64% win rate
#: (the size of TabICLv2's own published ~100-Elo effects) only ~15% of the time at
#: p < 0.05, where ~60 datasets reach ~70%.
#:
#: The inclusion rule was fixed BEFORE any result on these datasets was seen and applied
#: mechanically: features <= 500. The 10 excluded are image-like or very wide
#: (mnist_784, Fashion-MNIST, CIFAR_10, Devnagari-Script, isolet, har, madelon, cnae-9,
#: Internet-Advertisements, Bioresponse) -- they cost more than the rest of the suite
#: combined per screen, and they belong to TP-10 (wide tables), not to ablation screening.
#: Saturated datasets are NOT removed here; `drop_uninformative` handles them at read time
#: so the suite itself stays fixed.
#:
#: Do not edit. Add a new named suite instead.
CC18_NARROW: Tuple[DatasetSpec, ...] = (
    DatasetSpec("kr-vs-kp", 3, "classification", ('low_dim', 'categorical')),
    DatasetSpec("letter", 6, "classification", ('low_dim', 'many_class')),
    DatasetSpec("balance-scale", 11, "classification", ('low_dim',)),
    DatasetSpec("mfeat-factors", 12, "classification", ('high_dim',)),
    DatasetSpec("mfeat-fourier", 14, "classification", ('high_dim',)),
    DatasetSpec("breast-w", 15, "classification", ('low_dim',)),
    DatasetSpec("mfeat-karhunen", 16, "classification", ('high_dim',)),
    DatasetSpec("mfeat-morphological", 18, "classification", ('low_dim',)),
    DatasetSpec("mfeat-zernike", 22, "classification", ('low_dim',)),
    DatasetSpec("cmc", 23, "classification", ('low_dim', 'categorical')),
    DatasetSpec("optdigits", 28, "classification", ('high_dim',)),
    DatasetSpec("credit-approval", 29, "classification", ('low_dim', 'categorical')),
    DatasetSpec("credit-g", 31, "classification", ('low_dim', 'categorical')),
    DatasetSpec("pendigits", 32, "classification", ('low_dim',)),
    DatasetSpec("diabetes", 37, "classification", ('low_dim',)),
    DatasetSpec("sick", 38, "classification", ('low_dim', 'categorical')),
    DatasetSpec("spambase", 44, "classification", ('high_dim',)),
    DatasetSpec("splice", 46, "classification", ('high_dim', 'categorical')),
    DatasetSpec("tic-tac-toe", 50, "classification", ('low_dim', 'categorical')),
    DatasetSpec("vehicle", 54, "classification", ('low_dim',)),
    DatasetSpec("electricity", 151, "classification", ('low_dim', 'categorical')),
    DatasetSpec("satimage", 182, "classification", ('low_dim',)),
    DatasetSpec("eucalyptus", 188, "classification", ('low_dim', 'categorical')),
    DatasetSpec("vowel", 307, "classification", ('low_dim', 'many_class', 'categorical')),
    DatasetSpec("analcatdata_authorship", 458, "classification", ('high_dim',)),
    DatasetSpec("analcatdata_dmft", 469, "classification", ('low_dim', 'categorical')),
    DatasetSpec("pc4", 1049, "classification", ('low_dim',)),
    DatasetSpec("pc3", 1050, "classification", ('low_dim',)),
    DatasetSpec("jm1", 1053, "classification", ('low_dim',)),
    DatasetSpec("kc2", 1063, "classification", ('low_dim',)),
    DatasetSpec("kc1", 1067, "classification", ('low_dim',)),
    DatasetSpec("pc1", 1068, "classification", ('low_dim',)),
    DatasetSpec("bank-marketing", 1461, "classification", ('low_dim', 'categorical')),
    DatasetSpec("banknote-authentication", 1462, "classification", ('low_dim',)),
    DatasetSpec("blood-transfusion-service-center", 1464, "classification", ('low_dim',)),
    DatasetSpec("first-order-theorem-proving", 1475, "classification", ('high_dim',)),
    DatasetSpec("ilpd", 1480, "classification", ('low_dim', 'categorical')),
    DatasetSpec("nomao", 1486, "classification", ('high_dim', 'categorical')),
    DatasetSpec("ozone-level-8hr", 1487, "classification", ('high_dim',)),
    DatasetSpec("phoneme", 1489, "classification", ('low_dim',)),
    DatasetSpec("qsar-biodeg", 1494, "classification", ('low_dim',)),
    DatasetSpec("wall-robot-navigation", 1497, "classification", ('low_dim',)),
    DatasetSpec("semeion", 1501, "classification", ('high_dim',)),
    DatasetSpec("wdbc", 1510, "classification", ('low_dim',)),
    DatasetSpec("adult", 1590, "classification", ('low_dim', 'categorical')),
    DatasetSpec("PhishingWebsites", 4534, "classification", ('low_dim', 'categorical')),
    DatasetSpec("GesturePhaseSegmentationProcessed", 4538, "classification", ('low_dim',)),
    DatasetSpec("cylinder-bands", 6332, "classification", ('low_dim', 'categorical')),
    DatasetSpec("dresses-sales", 23381, "classification", ('low_dim', 'categorical')),
    DatasetSpec("numerai28.6", 23517, "classification", ('low_dim',)),
    DatasetSpec("texture", 40499, "classification", ('low_dim', 'many_class')),
    DatasetSpec("connect-4", 40668, "classification", ('low_dim', 'categorical')),
    DatasetSpec("dna", 40670, "classification", ('high_dim', 'categorical')),
    DatasetSpec("churn", 40701, "classification", ('low_dim', 'categorical')),
    DatasetSpec("MiceProtein", 40966, "classification", ('high_dim', 'categorical')),
    DatasetSpec("car", 40975, "classification", ('low_dim', 'categorical')),
    DatasetSpec("mfeat-pixel", 40979, "classification", ('high_dim',)),
    DatasetSpec("steel-plates-fault", 40982, "classification", ('low_dim',)),
    DatasetSpec("wilt", 40983, "classification", ('low_dim',)),
    DatasetSpec("segment", 40984, "classification", ('low_dim',)),
    DatasetSpec("climate-model-simulation-crashes", 40994, "classification", ('low_dim',)),
    DatasetSpec("jungle_chess_2pcs_raw_endgame_complete", 41027, "classification", ('low_dim',)),
)

#: The ten CC18 datasets excluded from CC18_NARROW. Recorded, not yet a validated suite --
#: the natural evaluation set for TP-10 once someone pays to download them.
CC18_WIDE_IDS: Tuple[int, ...] = (300, 554, 1468, 1478, 1485, 4134, 40923, 40927, 40978, 40996,)

SUITES["cc18_narrow"] = CC18_NARROW



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
