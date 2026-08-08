"""Coverage ratio on the five held-out RelBenchV1 classification tasks.

Label-free -- links and timestamps only -- so this is a PREDICTION, computed and printed
before any model runs. `--drop-stale-arms` fires below 0.8.

On the seven tasks the rule was designed from, the ratio separated cleanly: 0.57/0.69/0.72/
0.72 for the four where it fires, 0.96/0.99/1.03 for the three where it does not, and tuning
is harmful on exactly the ones that fire and have eligible history arms. Whether that
separation exists on databases the rule has never seen is the question.
"""
import sys, time, warnings
warnings.filterwarnings("ignore")
import pandas as pd
from relbench.datasets import get_dataset
from relbench.tasks import get_task
from tabicl.scaling import key_target_history

HELD_OUT = [("rel-stack", "user-engagement"), ("rel-stack", "user-badge"),
            ("rel-hm", "user-churn"), ("rel-amazon", "user-churn"),
            ("rel-amazon", "item-churn")]

print(f"{'task':<28}{'val cov':>9}{'test cov':>10}{'ratio':>8}  fires?", flush=True)
for ds, tk in HELD_OUT:
    try:
        t0 = time.perf_counter()
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
        if not cand:
            print(f"{ds+'/'+tk:<28}  no shared-key link table -- rule cannot fire", flush=True)
            continue
        n, efk, kcol, ltc = cand
        sub = db.table_dict[n].df[[efk, kcol, ltc]].dropna()
        cov = {}
        for split in ("val", "test"):
            q = f[split]
            h = key_target_history(sub[[efk, kcol]], f["train"][key].to_numpy(),
                                   f["train"][tgt].to_numpy(), f["train"][tc].to_numpy(),
                                   q[key].to_numpy(), q[tc].to_numpy(),
                                   label_horizon=t.timedelta,
                                   link_times=sub[ltc].to_numpy())
            cov[split] = float(h["hist__positive_rate"].notna().mean())
        r = cov["test"] / cov["val"] if cov["val"] > 0 else 1.0
        print(f"{ds+'/'+tk:<28}{cov['val']:>9.1%}{cov['test']:>10.1%}{r:>8.2f}  "
              f"{'YES' if r < 0.8 else 'no':<4} ({time.perf_counter()-t0:.0f}s)", flush=True)
    except Exception as exc:                                   # noqa: BLE001
        print(f"{ds+'/'+tk:<28}  {type(exc).__name__}: {str(exc)[:70]}", flush=True)
