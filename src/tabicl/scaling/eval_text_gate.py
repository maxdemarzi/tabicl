"""Do the text columns we throw away carry signal? Gate before building.

`RESEARCH.md` 6f. RelBench's own LightGBM baseline runs entity columns through torch_frame
with a `TextEmbedderConfig`, so free text becomes embeddings; our pipeline factorises
non-numeric columns into arbitrary integers and drops near-unique ones as identifiers. That
is the whole 31-point rel-f1 gap against their published baseline, and it means the content
of every text column in these databases has never entered our features.

**Cheap encoder first, deliberately.** This uses TF-IDF over character and word n-grams
followed by truncated SVD — no model download, no GPU, seconds per column. If text carries
signal, a sentence encoder is worth the dependency; if TF-IDF finds nothing, a transformer
almost certainly will not either, and the gate has cost minutes rather than a day. The same
sequencing that made the shared-key track record worth building and the graph-context idea
worth abandoning.

Reports per text column: coverage, cardinality, and standalone validation/test AUC of a
logistic model on its embedding alone. Compare against the task's calibrated number before
building anything.

Usage
-----
    python -m tabicl.scaling.eval_text_gate rel-trial study-outcome
"""

from __future__ import annotations

import argparse
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline

from relbench.datasets import get_dataset
from relbench.tasks import get_task

REFERENCE = {"rel-trial": 69.36, "rel-event": 81.76, "rel-avito": 65.61, "rel-f1": 82.48}


def _is_texty(series: pd.Series, n_rows: int) -> bool:
    """Free text, as opposed to a category or an id.

    Two conditions, and the second matters more than it looks: a column of long strings
    that are all distinct is an identifier (a URL, a study id), and embedding it just
    memorises the training set -- the same trap that made the entity-only baseline score
    below chance.
    """
    if not pd.api.types.is_object_dtype(series):
        return False
    non_null = series.dropna().astype(str)
    if len(non_null) < 20:
        return False
    mean_len = non_null.str.len().mean()
    n_unique = non_null.nunique()
    repeats = n_unique < 0.95 * len(non_null)
    return mean_len >= 15 and (repeats or mean_len >= 40)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("dataset", nargs="?", default="rel-trial")
    ap.add_argument("task", nargs="?", default="study-outcome")
    ap.add_argument("--components", type=int, default=64)
    args = ap.parse_args()

    db = get_dataset(args.dataset, download=True).get_db()
    task = get_task(args.dataset, args.task, download=True)
    key, target, entity = task.entity_col, task.target_col, task.entity_table
    train = task.get_table("train", mask_input_cols=False).df
    val = task.get_table("val", mask_input_cols=False).df
    test = task.get_table("test", mask_input_cols=False).df

    ent_df = db.table_dict[entity].df
    pk = db.table_dict[entity].pkey_col
    merged = {name: frame.merge(ent_df, left_on=key, right_on=pk, how="left")
              for name, frame in (("train", train), ("val", val), ("test", test))}

    candidates = [c for c in ent_df.columns
                  if c not in (pk, target) and _is_texty(ent_df[c], len(ent_df))]
    print(f"{args.dataset}/{args.task}: entity table {entity}, "
          f"{len(ent_df.columns)} columns, {len(candidates)} look like free text", flush=True)
    if not candidates:
        print("no text columns -- this task cannot benefit from 6f", flush=True)
        return

    y = {k: v[target].to_numpy() for k, v in merged.items()}
    print(f"\n{'column':<32} {'coverage':>9} {'unique':>8} {'val AUC':>9} {'test AUC':>9}",
          flush=True)
    for col in candidates:
        texts = {k: v[col].fillna("").astype(str) for k, v in merged.items()}
        coverage = float((merged["train"][col].notna()).mean())
        model = make_pipeline(
            TfidfVectorizer(sublinear_tf=True, min_df=3, max_features=50000,
                            ngram_range=(1, 2), strip_accents="unicode"),
            TruncatedSVD(n_components=args.components, random_state=0),
            LogisticRegression(max_iter=2000, random_state=0),
        )
        try:
            model.fit(texts["train"], y["train"])
            aucs = []
            for split in ("val", "test"):
                if len(np.unique(y[split])) < 2:
                    aucs.append(float("nan"))
                    continue
                p = model.predict_proba(texts[split])[:, 1]
                aucs.append(roc_auc_score(y[split], p) * 100)
        except Exception as exc:              # noqa: BLE001 - report, do not abort the sweep
            print(f"{col:<32} failed: {str(exc)[:50]}", flush=True)
            continue
        print(f"{col:<32} {coverage:>9.3f} "
              f"{merged['train'][col].nunique():>8} {aucs[0]:>9.2f} {aucs[1]:>9.2f}",
              flush=True)

    print(f"\nOur calibrated number on this task: "
          f"{REFERENCE.get(args.dataset, float('nan'))}. A column scoring near or above it "
          f"standalone is worth embedding properly; one near 50 is not.", flush=True)
    print("NOTE: TF-IDF is the cheap probe. A negative here is close to decisive; a "
          "positive is a floor, since a sentence encoder should beat bag-of-ngrams.",
          flush=True)


if __name__ == "__main__":
    main()
