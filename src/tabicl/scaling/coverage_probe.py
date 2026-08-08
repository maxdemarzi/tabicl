"""Coverage ratio across the wider RelBench family — the search for the rel-avito pathology.

`--drop-stale-arms` fires when the shared-key block's coverage collapses between validation
and test (ratio < 0.8). Across all twelve RelBenchV1 classification tasks that happens on
exactly five, all on rel-avito and rel-f1. The other seven, and all five held-out tasks, sit
at 0.96–1.03 — so RelBench cannot validate the rule, because it never engages there.

This asks whether the pathology exists anywhere else. Label-free: links, timestamps and the
existence of a label column, never its values.

**THE SUMMARY MUST NOT COUNT A FAILURE AS A CLEAN RESULT.** The first version of this script
probed thirteen tasks, ten raised AttributeError because the dbinfer tasks use a different
API, two raised MergeError on a datetime dtype mismatch, one succeeded — and it printed
"NO PATHOLOGY ANYWHERE ELSE", a conclusion about thirteen tasks drawn from one. Absence of
hits is not evidence when almost nothing was measured. The tally below separates MEASURED
from FAILED and refuses to conclude anything unless most tasks were measured.
"""
import time, warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
from relbench.datasets import get_dataset
from relbench.tasks import get_task

# dbinfer-* is deliberately absent: it requires `dbinfer-relbench-adapter` plus a `dgl`
# force-reinstall that its own installation notes warn conflicts with torch. Breaking the
# torch that every measurement in this project depends on, to probe a dataset, is not a
# trade worth making. Recorded as UNAVAILABLE rather than silently skipped.
TASKS = [
    ("rel-ratebeer", "brewer-dormant"),
    ("rel-ratebeer", "user-churn"),
    ("rel-ratebeer", "beer-churn"),
    ("rel-mimic", "patient-iculengthofstay"),
    ("rel-arxiv", "author-category"),
    ("rel-arxiv", "author-publication"),
    ("rel-salt", "item-plant"),
    ("rel-salt", "sales-office"),
]
UNAVAILABLE = ["dbinfer-avs/repeater", "dbinfer-retailrocket/cvr", "dbinfer-seznam/charge",
               "dbinfer-seznam/prepay", "dbinfer-diginetica/ctr",
               "dbinfer-diginetica/purchase", "dbinfer-stackexchange/churn",
               "dbinfer-stackexchange/upvote", "dbinfer-outbrain-small/ctr",
               "dbinfer-amazon/churn"]


def _ns(s):
    """Datetimes to a single resolution. rel-ratebeer stores us and the task table ns, and
    `merge_asof` refuses to join across them -- a dtype mismatch that surfaced as a
    MergeError and cost two tasks on the first run."""
    return pd.to_datetime(pd.Series(np.asarray(s)).astype("datetime64[ns]")).to_numpy()


def probe(ds, tk):
    from tabicl.scaling import key_target_history
    db = get_dataset(ds, download=True).get_db()
    t = get_task(ds, tk, download=True)
    key, tgt, ent = t.entity_col, t.target_col, t.entity_table
    f = {s: t.get_table(s, mask_input_cols=False).df for s in ("train", "val", "test")}
    tc = next(c for c in f["train"].columns
              if pd.api.types.is_datetime64_any_dtype(f["train"][c]))
    cand = None
    for n, tab in db.table_dict.items():
        fks = tab.fkey_col_to_pkey_table or {}
        if ent not in fks.values() or tab.time_col is None:
            continue
        efk = next(k for k, v in fks.items() if v == ent)
        for k2, v2 in fks.items():
            if v2 != ent:
                cand = (n, efk, k2, tab.time_col); break
        if cand:
            break
    if cand is None:
        return "no shared-key link table -- rule cannot fire"
    n, efk, kcol, ltc = cand
    sub = db.table_dict[n].df[[efk, kcol, ltc]].dropna()
    cov = {}
    for split in ("val", "test"):
        q = f[split]
        h = key_target_history(sub[[efk, kcol]], f["train"][key].to_numpy(),
                               f["train"][tgt].to_numpy(), _ns(f["train"][tc]),
                               q[key].to_numpy(), _ns(q[tc]),
                               label_horizon=t.timedelta, link_times=_ns(sub[ltc]))
        cov[split] = float(h["hist__positive_rate"].notna().mean())
    return cov["val"], cov["test"], (cov["test"] / cov["val"] if cov["val"] > 0 else 1.0)


print(f"{'task':<36}{'val cov':>9}{'test cov':>10}{'ratio':>8}  fires?", flush=True)
measured, failed, inapplicable, hits = [], [], [], []
for ds, tk in TASKS:
    label = f"{ds}/{tk}"
    try:
        t0 = time.perf_counter()
        got = probe(ds, tk)
        if isinstance(got, str):
            inapplicable.append(label)
            print(f"{label:<36}  {got}", flush=True)
            continue
        v, te, r = got
        measured.append(label)
        if r < 0.8:
            hits.append((label, r))
        print(f"{label:<36}{v:>9.1%}{te:>10.1%}{r:>8.2f}  "
              f"{'YES  <-- PATHOLOGY' if r < 0.8 else 'no':<20} "
              f"({time.perf_counter()-t0:.0f}s)", flush=True)
    except Exception as exc:                                   # noqa: BLE001
        failed.append(label)
        print(f"{label:<36}  FAILED {type(exc).__name__}: {str(exc)[:55]}", flush=True)

print(f"\nMEASURED {len(measured)}  |  inapplicable {len(inapplicable)}  |  "
      f"FAILED {len(failed)}  |  unavailable (dbinfer, needs dgl) {len(UNAVAILABLE)}",
      flush=True)
if hits:
    print(f"PATHOLOGY FOUND: {', '.join(f'{n} ({r:.2f})' for n, r in hits)}", flush=True)
    print("Testable ground: same pathology, no role in the rule's design.", flush=True)
elif len(measured) >= max(3, len(TASKS) // 2):
    print(f"No pathology in the {len(measured)} tasks actually MEASURED. With "
          f"{len(failed)} failed and {len(UNAVAILABLE)} unavailable, this bounds the search "
          f"rather than closing it.", flush=True)
else:
    print(f"INCONCLUSIVE -- only {len(measured)} task(s) measured. Absence of hits here is "
          f"absence of measurement, not evidence. Do not read this as a null.", flush=True)
