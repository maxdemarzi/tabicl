"""Calendar features from the prediction timestamp, which is currently thrown away.

Both runners drop every datetime column when assembling the entity block -- the cutoff
included -- so nothing downstream knows *when* a prediction is being made. The child
aggregates know how long the history is and (now) how recent it is, but the model has no
way to tell a Monday from a Saturday.

That is worth something on tasks whose horizon is a few days. rel-avito asks whether a
user visits within four days and rel-event whether an invitation is ignored within seven;
both plainly have a weekly rhythm, and neither can express one.

Nothing here is leakage. The prediction time is an input to the prediction -- RelBench
hands it to every method, and the test rows carry it. What must never be built from it is
anything requiring data *after* it.

The **trend** column is separated deliberately. Days-since-epoch is monotone, so the test
split takes values no training row ever had, and a model that keys on it is extrapolating
off the end of its own support. Cyclical features have no such problem: December recurs.
So the two are separate arguments and separate flags, rather than one bundle that can win
for the wrong reason.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

__all__ = ["calendar_features"]


def calendar_features(timestamps, trend: bool = False, origin=None) -> pd.DataFrame:
    """Cyclical calendar features for each prediction timestamp.

    Parameters
    ----------
    timestamps : array-like of datetime64
        One prediction time per row.

    trend : bool, default=False
        Also emit ``trend``, days since ``origin``. Monotone, so it is off by default:
        every test row lies beyond the training range on it.

    origin : optional
        Reference point for ``trend``. Pass the training minimum so train and test are
        measured on the same scale; defaults to the minimum of ``timestamps``, which is
        only correct when they are the training ones.

    Returns
    -------
    pd.DataFrame
        Raw parts plus sine/cosine pairs. Both are emitted because they are not
        substitutes: the raw integer is what a tree splits on, the pair is what makes
        Sunday and Monday adjacent rather than 6 apart.
    """
    index = pd.DatetimeIndex(pd.to_datetime(np.asarray(timestamps)))
    out = pd.DataFrame(index=pd.RangeIndex(len(index)))

    # `np.asarray`, not `.to_numpy()`: these accessors return a bare ndarray on some pandas
    # versions and an Index on others, and only one of the two has that method.
    dayofweek = np.asarray(index.dayofweek, dtype=np.float64)
    month = np.asarray(index.month, dtype=np.float64)

    out["cal__dayofweek"] = dayofweek
    out["cal__day"] = np.asarray(index.day, dtype=np.float64)
    out["cal__month"] = month
    out["cal__is_weekend"] = (dayofweek >= 5).astype(np.float64)

    for name, values, period in (("dow", dayofweek, 7.0), ("month", month - 1.0, 12.0)):
        angle = 2.0 * np.pi * values / period
        out[f"cal__{name}_sin"] = np.sin(angle)
        out[f"cal__{name}_cos"] = np.cos(angle)

    if trend:
        base = pd.Timestamp(origin) if origin is not None else index.min()
        out["cal__trend"] = ((index - base) / pd.Timedelta(days=1)).to_numpy(dtype=np.float64)

    return out
