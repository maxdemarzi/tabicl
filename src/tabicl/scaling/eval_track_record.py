"""Does a trial's shared-key track record add to the pipeline on rel-trial?

rel-trial is the largest remaining gap: 66.50 against TabPFN-REL's 76.43. It is also the
one task where a foundation model beats the GNN, which points at featurisation rather than
graph structure, and it has proved insensitive to every lever tried -- column budget 0.0,
relation breadth +0.35, both inside the noise floor.

The candidate is the outcome history of trials sharing a sponsor, condition, facility or
intervention. Gated before being built (`key_target_history` docstring): each key scores
60-61 standalone test AUC at 76-88% coverage, against a whole pipeline at 66.50.

Three arms, and the middle one is not optional. On rel-event a neighbour-label feature
looked like a 74 AUC discovery and turned out to be mostly *degree* -- a count carrying no
outcome information at all. So the count columns get their own arm here, and the claim
"outcome history helps" only survives if the full block beats counts-only.

Nothing in `key_target_history` is rel-trial-specific -- it needs a link table, labels,
timestamps and a horizon -- so the runner takes the dataset and task as arguments. Use
`--gate` first on a new task: it prints each key's standalone AUC and coverage without
touching a GPU, and that ratio against the existing pipeline is what predicted the
difference between this working on rel-trial and the graph version failing on rel-event.

Usage
-----
    python -m tabicl.scaling.eval_track_record [dataset] [task] [--gate]
    python -m tabicl.scaling.eval_track_record rel-avito user-visits --gate
    python -m tabicl.scaling.eval_track_record --calibrated --seeds 5
"""

from __future__ import annotations

import argparse
import sys
import math
import time
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

from sklearn.metrics import roc_auc_score

from relbench.datasets import get_dataset
from relbench.tasks import get_task

from tabicl import TabICLClassifier
from tabicl.scaling import (Table, asof_statistics, entity_label_history,
                            key_target_history, two_hop_table)
from tabicl.scaling._guards import assert_no_perfect_feature
from tabicl.scaling._calendar import calendar_features
from tabicl.scaling._leakage import permutation_test, temporal_control

NOAMP = {k: {"use_amp": False} for k in ("COL_CONFIG", "ROW_CONFIG", "ICL_CONFIG")}


def inference_config(row_chunk: str, offload: str = "off",
                     disk_dir: str = "/workspace/offload") -> dict:
    """AMP off always; row chunking and output offloading per their flags.

    **This runner has never used the package's own scaling features.** Row-chunked column
    embedding is item 1 of four in `STATUS.md` -- "Working. Exact, not approximate", verified
    at ``max|dp| = 1.1e-05`` with identical AUC -- and `eval_track_record` contained zero
    references to ``row_chunk`` or ``offload``. The benchmark that exists to demonstrate
    scaling was not scaling.

    The two knobs address *different* tensors and compose, which is the whole reason both
    are here: ``row_chunk`` shrinks the column-embedding **activations**, ``offload`` moves
    the **output** tensors off the GPU. A shape can be too large for one and not the other.

    **Sizing, because the failure this addresses was silent.** TabICL is in-context, so the
    context and the queries pass through the stages *together*: sizes scale with
    ``n_context + n_query``, not ``n_context``. On rel-stack/user-badge that is ~10k + ~250k
    rows at 342 columns, the order of an L40S's 46 GB, and the process died in its first
    forward pass having printed **no traceback**. Ruled out by measurement rather than
    assumption: host RAM (503 GB, 471 free, no OOM-kill record), the 2h timeout (it died at
    12m27s), and a partial checkpoint (106 MB, no ``.incomplete``).

    **And row chunking alone did not save it** -- that run already passed ``--row-chunk
    auto``. Which points at where the defaults are asymmetric: out of the box
    ``COL_CONFIG.offload`` is already ``"auto"``, while ``ICL_CONFIG.offload`` is ``False``.
    The column stage offloads its outputs and the in-context stage never does, so at ~250k
    queries the ICL outputs stay resident. That is what ``--offload`` reaches and
    ``--row-chunk`` cannot: chunking shrinks activations, offloading moves outputs.

    This is a diagnosis, not a confirmed fix. It is consistent with every observation above
    and it has not yet been demonstrated to make the task run.

    Defaults are ``"off"`` for both, because every standing number was measured without them
    and "exact" is a claim about this package's own tests rather than about every shape it
    will now meet.
    """
    cfg = {k: dict(v) for k, v in NOAMP.items()}
    if row_chunk != "off":
        cfg["COL_CONFIG"]["row_chunk"] = True if row_chunk == "always" else "auto"
    if offload != "off":
        # ``offload`` is a field of MgrConfig, and MgrConfig is the *type* of each of the
        # three stage configs -- there is no separate manager key. Set it on all three, the
        # way ``use_amp`` already is. (`InferenceConfig.update_from_dict` raises KeyError on
        # anything outside COL/ROW/ICL, so a wrong key fails loudly rather than reading as
        # "offloading did not help" -- but it is still wrong, so it is spelled out.)
        for k in cfg:
            cfg[k]["offload"] = offload
            if offload == "disk":
                # Without a directory, disk offloading DISABLES ITSELF and raises only once
                # CPU memory is already insufficient -- which on a memory-capped container
                # is a race against the OOM killer that the OOM killer wins. Set it here so
                # the mode cannot be silently inert.
                cfg[k]["disk_offload_dir"] = disk_dir
    return cfg

# Aggregation windows are task-scale, not universal: clinical trials run for years, ad
# impressions for days. A single default would quietly handicap one task or the other.
DEFAULT_WINDOWS = {
    "rel-trial": "365,1095",
    "rel-avito": "7,30",
    "rel-event": "30,365",
    "rel-f1": "365,1095",
}

# Printed beside a result so it is never read against the wrong task's numbers.
# Keyed by dataset AND task. It was keyed by dataset alone, so running rel-event/user-repeat
# printed user-ignore's comparison figures beside it -- a wrong reference is worse than none,
# because it invites reading a 77.89 against an 85.38 that belongs to a different label.
REFERENCE = {
    ("rel-trial", "study-outcome"): "ours 72.26, TabPFN-REL 76.43, RelGNN 71.24, RDBLearn 72.89",
    ("rel-event", "user-ignore"): "ours 80.98, TabPFN-REL 85.38, RelGNN 86.18, RDBLearn 73.70",
    ("rel-event", "user-repeat"): "ours 77.89, TabPFN-REL 77.11, RelGNN 79.61, RDBLearn 76.81",
    ("rel-avito", "user-visits"): "ours 65.54, TabPFN-REL 66.68, RelGNN 66.18, RDBLearn 66.76",
    ("rel-f1", "driver-top3"): "ours 81.98, TabPFN-REL 79.98, RelGNN 85.69, RDBLearn 82.72",
}

# The flags each standing number was measured with. This is not documentation, it is the
# thing three runs got wrong in one afternoon: a promotion compared against a headline it
# did not share a configuration with. rel-f1 without `--max-columns none` scores 73.21
# against a standing 81.98 and looks like a catastrophic regression; it is a different
# experiment. `--timed-links-only` matters on rel-event alone -- it has two untimed
# `user_friends` link tables while the other three schemas have none, so the flag is a
# no-op everywhere else and omitting it there makes the structural arms an upper bound.
#
# `--max-columns 2` is named explicitly for three tasks even though 2 *was* the default when
# they were measured. It is not the default any more -- the sweep moved it to 4 on
# worst-case regret -- so an empty list here would quietly mean "4" and a run that believed
# it matched would not reproduce the number it was compared against.
STANDING_FLAGS = {
    # `--children 3` is listed only for rel-trial ON PURPOSE. The other three datasets have
    # exactly three timestamped child tables, so the old default took all of them and the
    # flag changed nothing; naming it there would imply a difference that does not exist and
    # send someone hunting for it.
    # `--categories 0` is listed for rel-trial and rel-event only: the default moved 0 -> 8
    # and those are the two datasets where categories emits anything at all. On rel-avito and
    # rel-f1 the block is empty either way, so naming it there would imply a difference that
    # does not exist.
    "rel-trial": ["--max-columns 2", "--children 3", "--categories 0"],
    "rel-event": ["--timed-links-only", "--max-columns 2", "--categories 0"],
    "rel-avito": ["--max-columns 2"],
    "rel-f1": ["--max-columns none"],
}


_CATEGORY_MAPS: dict = {}
_TIME_DELTA_COLS: dict = {}


def _numeric(df: pd.DataFrame, fit: bool = False, tag: str = "") -> np.ndarray:
    """Encode a feature frame, with categorical codes CONSISTENT ACROSS SPLITS.

    The previous version called ``pd.factorize`` on each frame independently. Factorize
    assigns codes by order of first appearance, so the same category received *different
    integers in train and in test* -- a value coded 3 for fitting could be 7 at prediction
    time. Every categorical column was therefore not merely arbitrarily ordered but
    inconsistently ordered, which is worse: the model learns a mapping that does not hold
    where it is applied.

    Categories are now learned on the fitting frame and reused. Unseen values encode to -1,
    which is a distinguishable "not in training" rather than a collision with a real code.
    """
    out = df.copy()
    # Key by the frame's own column set, so two arms with different columns cannot share a
    # map and no caller has to remember to pass a distinct tag.
    tag = tag or str(hash(tuple(df.columns)))
    # Drop columns whose VALUES are arrays or lists before anything tries to hash them.
    # rel-amazon carries such a column and `pd.factorize` raises `TypeError: unhashable
    # type: 'numpy.ndarray'` on it, which killed rel-amazon/item-churn after its features
    # were built and its controls had run.
    #
    # Dropping rather than encoding is the honest option: a per-row vector has no
    # categorical identity to factorize, and stringifying it would mint a unique category
    # per row -- a column of distinct codes carrying no signal, indistinguishable from a
    # row identifier, which is precisely the shape this project's leak controls exist to
    # catch. Anything genuinely useful in such a column needs deliberate featurisation, not
    # a silent cast.
    unhashable = [c for c in out.columns
                  if not pd.api.types.is_numeric_dtype(out[c])
                  and out[c].map(lambda v: isinstance(v, (np.ndarray, list, dict, set))).any()]
    if unhashable:
        print(f"  dropping {len(unhashable)} array-valued column(s) from {tag}: "
              f"{unhashable[:4]}{' ...' if len(unhashable) > 4 else ''}", flush=True)
        out = out.drop(columns=unhashable)
    for col in out.columns:
        if pd.api.types.is_numeric_dtype(out[col]):
            continue
        keyed = f"{tag}:{col}"
        if fit or keyed not in _CATEGORY_MAPS:
            codes, uniques = pd.factorize(out[col])
            _CATEGORY_MAPS[keyed] = {v: i for i, v in enumerate(uniques)}
            out[col] = codes
        else:
            out[col] = out[col].map(_CATEGORY_MAPS[keyed]).fillna(-1)
    return np.nan_to_num(out.to_numpy(dtype=np.float64), nan=0.0, posinf=0.0, neginf=0.0)


_COMPRESSORS: dict = {}


def compress_fit_apply(name: str, X: np.ndarray, k: int, fit: bool) -> np.ndarray:
    """Narrow a feature matrix by COMPRESSION instead of by SELECTION.

    ``--max-columns`` reaches a workable width by deleting source columns, ranked by
    coverage. That is target-free and cheap, and it throws information away: on three of
    four databases the standing configuration runs at ``max_columns=2``, so most of each
    child table never reaches the model at all.

    The alternative, from GOTabPFN (arXiv 2606.05441), is to keep every column and project
    to the same width. Their setting is high-dimensional low-sample biomedical data and
    their headline numbers do not transfer, but the mechanism is ours: a PCA-style
    compression in front of a **frozen** backbone, no retraining. And the problem is one
    this project measured directly -- 150 extra columns cost -3.36 on rel-trial while the
    same block at 18 columns cost -0.10, so width is expensive and the budget is paying for
    it by deletion.

    **Fitted on TRAIN ONLY and reused**, keyed by arm. Fitting per split would let test rows
    choose their own projection, which is the same leak the text block avoids by fitting its
    vectoriser on train alone -- and it would make the val arm mean something different from
    the test arm, the failure this project traced its val/test inversion to.
    """
    from sklearn.decomposition import PCA
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    if X.shape[1] <= k:
        return X
    if fit:
        pipe = make_pipeline(StandardScaler(), PCA(n_components=k, random_state=0))
        pipe.fit(X)
        _COMPRESSORS[name] = pipe
    pipe = _COMPRESSORS.get(name)
    if pipe is None:
        return X
    return pipe.transform(X)


def knn_context_indices(X: np.ndarray, Xe: np.ndarray, n_context: int, seed: int,
                        per_query: int = 8, max_queries: int = 5000) -> np.ndarray:
    """Rows to put in the context, chosen by proximity to the queries rather than at random.

    Every context experiment in this project has varied context SIZE (`--context-grid`) or
    ORDER (`random` vs `recent`). Content has never been tested, and two independent
    measurements say content is where the signal should be: fixing the size axis cost no
    accuracy and halved the variance here, and KernelICL (arXiv 2602.02162) measures a
    tabular ICL prediction as relying on only 11-29% of its context.

    **Expect this to underperform, and the reason is worth stating in advance.** That same
    paper measures retrieval in a *learned embedding* space at ~5 points above retrieval in
    raw input space, and these are raw features. This is the cheap version of the idea, and
    the paper predicts the cheap version is the weak one.

    Label-free: only ``Xe``'s feature geometry is consulted, never its labels. Queries are
    subsampled to ``max_queries`` because the neighbour search is |queries| x |pool| and
    rel-avito would otherwise pair 100k queries against 86k pool rows -- and the queries are
    only being used to locate a region, which a sample locates as well as the whole set.
    """
    from sklearn.neighbors import NearestNeighbors

    n = len(X)
    if n_context >= n:
        return np.arange(n)
    rng = np.random.default_rng(seed)
    q = Xe if len(Xe) <= max_queries else Xe[rng.choice(len(Xe), max_queries, replace=False)]

    mu, sigma = X.mean(axis=0), X.std(axis=0)
    sigma[sigma == 0] = 1.0
    nn = NearestNeighbors(n_neighbors=min(per_query, n)).fit((X - mu) / sigma)
    hits = nn.kneighbors((q - mu) / sigma, return_distance=False)

    # A row retrieved by many queries serves more of the query distribution than one
    # retrieved by a single query, so rank the union by retrieval frequency.
    counts = np.bincount(hits.ravel(), minlength=n)
    order = np.argsort(-counts, kind="stable")
    retrieved = int((counts > 0).sum())
    if retrieved >= n_context:
        return order[:n_context]
    # Top up at random rather than returning a shorter context, so the comparison against a
    # random draw varies selection and holds context LENGTH fixed.
    extra = rng.permutation(order[retrieved:])[: n_context - retrieved]
    return np.concatenate([order[:retrieved], extra])


def mmd_context_indices(X: np.ndarray, Xe: np.ndarray, n_context: int, seed: int,
                        n_features: int = 256, max_queries: int = 5000) -> np.ndarray:
    """Context chosen to MATCH THE QUERY DISTRIBUTION, not to sit nearest the queries.

    CRUMB, Heredge et al., *Efficient Prior Fitted Network Inference via Distributionally
    Matched Context Batching* (arXiv 2606.11473), at ``K = 1`` -- one shared context for all
    queries, which the paper names explicitly and which drops into the same slot as a random
    draw. Greedily minimises

        MMD^2 = (1/n^2) SS k(xi,xi')  -  (2/n|C|) SS k(xi,xj*)

    where the first term **penalises redundancy among the chosen rows** and the second
    rewards proximity to the queries. That first term is the entire difference from kNN
    selection, which is pure proximity, and it is why the paper reports MMD beating
    kNN-style strategies. It matters here for a specific reason: this project measured its
    val/test gap as a *population* problem -- validation entities appear in train 81.2% of
    the time against test's 58.6% -- and MMD minimisation aligns the context distribution to
    the queries rather than merely hugging them.

    Directly relevant rather than adjacent: CRUMB is architecture-agnostic, needs **no
    retraining**, and its reported evaluation includes **TabICLv2**, which is the checkpoint
    this runner loads.

    Random Fourier features make the greedy step a matrix-vector product: with
    ``k(x,y) = phi(x).phi(y)``, the redundancy term is ``||sum phi(xi)||^2`` and the
    proximity term is ``sum phi(xi) . mean phi(xj*)``, so each candidate is scored in one
    pass instead of recomputing a kernel matrix.

    Exactly ``n_context`` rows are returned, with no early stopping. The paper's adaptive
    variant halts when MMD stops improving, which is right for saving inference cost and
    wrong here: a shorter context would make this the context-SIZE experiment again, and the
    comparison must vary selection alone.
    """
    n = len(X)
    if n_context >= n:
        return np.arange(n)
    rng = np.random.default_rng(seed)
    q = Xe if len(Xe) <= max_queries else Xe[rng.choice(len(Xe), max_queries, replace=False)]

    mu, sigma = X.mean(axis=0), X.std(axis=0)
    sigma[sigma == 0] = 1.0
    Xs, qs = (X - mu) / sigma, (q - mu) / sigma

    # RBF via random Fourier features. Bandwidth by the median heuristic on a subsample.
    sub = Xs[rng.choice(n, size=min(1000, n), replace=False)]
    d2 = np.sum((sub[:, None, :] - sub[None, :, :]) ** 2, axis=-1)
    gamma = 1.0 / max(np.median(d2[d2 > 0]), 1e-9)
    W = rng.normal(scale=np.sqrt(2 * gamma), size=(Xs.shape[1], n_features))
    b = rng.uniform(0, 2 * np.pi, size=n_features)
    scale = np.sqrt(2.0 / n_features)
    Phi = scale * np.cos(Xs @ W + b)
    mu_q = (scale * np.cos(qs @ W + b)).mean(axis=0)

    norms = np.einsum("ij,ij->i", Phi, Phi)
    prox = Phi @ mu_q
    s = np.zeros(n_features)
    a = np.zeros(n)
    chosen = np.empty(n_context, dtype=np.int64)
    taken = np.zeros(n, dtype=bool)
    for m in range(n_context):
        # argmin over candidates of (2 a_c + ||phi_c||^2)/(m+1) - 2 prox_c; the dropped
        # terms are constant across candidates at this step.
        score = (2.0 * a + norms) / (m + 1) - 2.0 * prox
        score[taken] = np.inf
        c = int(np.argmin(score))
        chosen[m], taken[c] = c, True
        s += Phi[c]
        a += Phi @ Phi[c]
    return chosen


_TABFM = None


def _tabfm(device: str):
    """Load TabFM once and keep it. Refuses CPU when CUDA is present.

    `load()` takes `device` as a KEYWORD defaulting to None, which lands on CPU, and
    `TabFMClassifier` has no device argument at all -- so load time is the only place to set
    it and omitting it fails silently, producing correct AUCs about thirty times slower.
    That happened once already; the check is structural rather than remembered.
    """
    global _TABFM
    if _TABFM is None:
        from tabfm import tabfm_v1_0_0_pytorch as tabfm_v1_0_0
        _TABFM = tabfm_v1_0_0.load(device=device)
        if torch.cuda.is_available() and hasattr(_TABFM, "parameters"):
            where = {q.device.type for q in _TABFM.parameters()}
            if where and where != {"cuda"}:
                raise SystemExit(f"TabFM loaded onto {where} while CUDA is available.")
        print(f"backbone: TabFM 1.0.0 on {device}", flush=True)
    return _TABFM


def abstention_choice(split_val: dict) -> tuple[tuple | None, tuple | None, bool]:
    """Keep the tuned pick only if its validation ranking survives a time gap.

    `split_val` maps a configuration key ``(arm, size, order)`` to its ``(early, late)``
    validation AUCs, both measured from the same fit on two time-ordered halves of the
    validation set.

    The winner is chosen on the **early** half and then judged on the **late** half against
    the best `base` configuration. If it cannot beat the untuned arm out of sample, the
    ranking that produced it did not generalise across a gap, and `base` is returned.

    **This is deliberately not the question every failed instrument here asked.** Those
    tried to extract a better *ranking* from a validation split that is biased by sitting
    close to train; anything derived from a biased signal inherits the bias. This asks only
    whether the ranking *holds up* across a gap — a property of the signal, not a value read
    off it — and it acts by declining to tune rather than by tuning differently.

    Motivated by rel-avito, where tuning loses on both tasks: `user-clicks` untuned scores
    67.32 against 66.03 tuned, a **1.29** loss worth four places in the published field,
    because the arms there are near-identical and selection is fitting noise.

    Returns ``(chosen_key, winner_key, abstained)``. ``chosen_key`` is None when the input
    has no usable entry, in which case the caller keeps its own argmax.
    """
    usable = [(k, v) for k, v in split_val.items()
              if not (math.isnan(v[0]) or math.isnan(v[1]))]
    bases = [(k, v) for k, v in usable if k[0] == "base"]
    if not usable or not bases:
        return None, None, False
    win_k, win_v = max(usable, key=lambda kv: kv[1][0])
    base_k, base_v = max(bases, key=lambda kv: kv[1][0])
    if win_k[0] != "base" and win_v[1] <= base_v[1]:
        return base_k, win_k, True
    return win_k, win_k, False


def temporal_shift_grid(span_days: float, horizon_days: float) -> tuple[int, ...]:
    """How far back to move cutoffs when testing whether a feature reads the future.

    A temporal control withholds outcomes by rewinding every cutoff and checking the score
    does not improve. The shift has to land between two failure modes, and this project has
    now hit both:

    **Too large** removes every usable label, the feature goes constant, and the control
    passes at 0.5 having tested nothing — which a hardcoded 180/365 days did on rel-event,
    whose horizon is 7 days. Hence scaling to the task's own span.

    **Too small withholds nothing**, because an outcome only becomes readable one horizon
    after it is recorded. On rel-avito, span 8 days gave ``round(0.05*8) = 0`` and
    ``round(0.15*8) = 1``: the first "control" was the unshifted setting itself, and the
    only real shift was 1 day against a 4-day horizon. Coverage moved 0.572 → 0.571 — nothing
    was withheld — and the resulting 0.0053 difference was reported as a leak, which
    **excluded the entire `+rate` family from selection on that dataset.** rel-trial had the
    same defect more quietly: a 150-day first shift under a 365-day horizon.

    So: at least one horizon, at least one day, never zero, and deduplicated — two shifts
    that round to the same number are one control, not two.

    Returns the positive shifts only. The caller prepends 0.0, which `temporal_control`
    requires as its unshifted reference. An empty return means no valid shift exists and the
    control cannot say anything.
    """
    floor_days = max(float(horizon_days), 1.0)
    return tuple(sorted({round(max(f * float(span_days), floor_days))
                         for f in (0.05, 0.15)} - {0}))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("dataset", nargs="?", default="rel-trial")
    ap.add_argument("task", nargs="?", default="study-outcome")
    ap.add_argument("--gate", action="store_true",
                    help="standalone AUC and coverage per key, no GPU. Run this on a new "
                         "task before building anything: a key worth less than the "
                         "existing pipeline has no room to help.")
    ap.add_argument("--window-days", default=None,
                    help="comma-separated aggregation windows; defaults per dataset")
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--context", type=int, default=10000)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--n-estimators", type=int, default=4)
    ap.add_argument("--no-horizon", action="store_true",
                    help="ignore the 365-day resolution window. Wrong, and kept only to "
                         "measure what it is worth.")
    ap.add_argument("--stratify-context", action="store_true",
                    help="draw each resampled context in proportion to the training class "
                         "balance. Only meaningful with --resample > 1: it removes the "
                         "class-balance wobble between draws, which is noise added to the "
                         "quantity resampling exists to average down.")
    ap.add_argument("--cv-time-ordered", action="store_true",
                    help="forward-chaining folds: each is scored using only earlier rows. "
                         "Required for this benchmark -- random k-folds train on the future "
                         "to predict the past, and did exactly that, rewarding resampling "
                         "with +5.3 of CV score while test fell 0.76.")
    ap.add_argument("--cv-folds", type=int, default=0,
                    help="select on k-fold CV over train instead of the single validation "
                         "split. 0 keeps the current behaviour. The val splits here are "
                         "588-2013 rows and have twice blocked a real gain: child count was "
                         "undecidable (0.61 spread across every setting) and per-key "
                         "selection ranked options opposite to test. Selection noise, not "
                         "feature quality, is the binding constraint.")
    ap.add_argument("--resample", type=int, default=1,
                    help="average predictions over N independent context draws. 1 is the "
                         "existing behaviour exactly. Attacks the variance that dominates "
                         "these tasks: rel-event's replicates span 10.8 points on the "
                         "context draw alone. Distinct from --n-estimators, which varies "
                         "the model seed at a fixed context and measured at nothing.")
    ap.add_argument("--children", type=int, default=0,
                    help="how many timestamped child tables to aggregate; 0 means all, which "
                         "is now the default. The old default of 3 was never chosen -- it "
                         "was a hardcoded slice, and it only ever BIT on rel-trial: the "
                         "other three datasets have exactly three child tables, so it was a "
                         "no-op there. On rel-trial it discarded seven of ten tables and "
                         "picked which three NON-DETERMINISTICALLY -- two hosts selected "
                         "different sets, so the same command produced different features. "
                         "Using all of them removes that reproducibility defect outright and "
                         "is worth +0.53 on rel-trial. The cost is compute on wide schemas, "
                         "not accuracy. Old behaviour: --children 3 --children-order name. "
                         "The tables it keeps are whichever come first in dictionary order, "
                         "out of ten on "
                         "rel-trial. The base sweep says breadth matters: one child costs 30 "
                         "points on rel-event and 8 on rel-avito. 0 means all.")
    ap.add_argument("--text", action="store_true",
                    help="add a TF-IDF + SVD embedding of the entity table's free-text "
                         "columns as its own arm. RESEARCH 6f: our pipeline has never used "
                         "text content, and on rel-trial four columns score 60.2-64.1 "
                         "standalone against a 69.36 pipeline.")
    ap.add_argument("--text-components", type=int, default=32)
    ap.add_argument("--text-rows", type=int, default=20,
                    help="child rows per entity contributing text, most recent first. Was a "
                         "hardcoded head(20) in dataframe order, so the feature depended on "
                         "storage order rather than on time.")
    ap.add_argument("--top-keys", type=int, default=0,
                    help="keep only the N keys with the highest standalone validation AUC, "
                         "ranked by the same gate used before building. 0 keeps all. Every "
                         "key's block is otherwise concatenated indiscriminately, and the "
                         "gate already measures them as far apart as 53 to 82 on rel-event.")
    ap.add_argument("--max-columns", default="4",
                    help="per-child column budget, or 'none'. Chosen on validation across "
                         "all four tasks by worst-case regret, which is the right statistic "
                         "for a default -- it bounds the damage when the default is wrong. "
                         "4 is never more than 2.13 off the best value for a task; the 2 "
                         "that was hardcoded here costs 19.13 on rel-f1, and the None the "
                         "library ships costs 6.12 on rel-event and builds 2,000 columns in "
                         "349 s there. Still worth setting per task where you can: the "
                         "spread across values reaches 19 points.")
    ap.add_argument("--children-order", choices=["dict", "name"], default="dict",
                    help="how to order child tables before --children takes the first N. "
                         "'dict' is insertion order and is NOT stable across environments: "
                         "rel-trial's first three are ['designs','eligibilities',"
                         "'drop_withdrawals'] on one host and ['conditions_studies',"
                         "'designs','drop_withdrawals'] on another, from the same ten "
                         "candidates. 'name' sorts alphabetically and is reproducible. "
                         "Kept as the default only because the standing numbers were "
                         "measured under it.")
    ap.add_argument("--top-children", type=int, default=0,
                    help="keep the N child tables with the strongest single column on "
                         "validation, instead of the first N in dictionary order. WHICH "
                         "children, not how many -- the count has been swept, the identity "
                         "never has. --top-keys does exactly this for link tables and "
                         "measured them 53 to 82 apart on rel-event.")
    ap.add_argument("--categories", type=int, default=8,
                    help="emit a per-category proportion block for each categorical child "
                         "column: the N most frequent values plus an 'other' bucket. 0 is "
                         "off. DEFAULT CHANGED 0 -> 8 on 2026-08-07. Chosen on VALIDATION "
                         "across the three tasks where it applies, as max_columns was: "
                         "rel-trial +1.12 (SE 0.22), user-repeat +0.25 (SE 0.07), "
                         "user-ignore +0.26 (SE 0.18) -- positive on all three, negative on "
                         "none, worst-case regret zero. Calibrated TEST at 12 replicates "
                         "agrees: +0.93, +0.09, +0.15, none negative, though only "
                         "user-repeat clears the +-0.6 floor. On the other FOUR tasks (both "
                         "rel-avito, both rel-f1) it emits no columns at all and is skipped "
                         "with a note, so it cannot hurt them. Every standing number "
                         "predates this and was measured at 0.")
    ap.add_argument("--category-share", type=float, default=0.5,
                    help="minimum share of non-null rows the codebook must capture before a "
                         "column gets a histogram. Below it the column is free text in "
                         "disguise (rel-trial's eligibilities.criteria has 247k values and "
                         "its top 4 cover 0.1%%), and the block is K+1 constant columns.")
    ap.add_argument("--ensemble-configs", type=int, default=0,
                    help="average test predictions over the top N configurations by "
                         "validation score, instead of taking the argmax. 0 keeps the "
                         "argmax. This routes around the constraint that blocked everything "
                         "else measured this week: nine effects are real on test and the "
                         "validation split cannot reliably rank them, so hedging across the "
                         "top of a noisy ranking is strictly less exposed to that noise than "
                         "committing to its first element -- and it needs no instrument the "
                         "argmax does not already need. Ensembling over context draws "
                         "already cut variance 4.6x on rel-event; this ensembles over "
                         "configurations.")
    ap.add_argument("--decide-fit-pool", action="store_true",
                    help="decide whether to fit on train+val using validation alone, "
                         "instead of assuming it. Splits val by time into a selection part "
                         "and a decision part: settings are chosen on the first, then "
                         "fit-on-train versus fit-on-train+selection-part is compared on "
                         "the second, and whichever wins is used for the final fit on all "
                         "of train(+val). No test data is involved. Needed because "
                         "--fit-on-train-val is +2.48 on rel-f1 and -7.29 on rel-event, "
                         "and the ordinary validation column is IDENTICAL in both arms -- "
                         "it cannot see the difference, so enabling it per task by test "
                         "score would be test-selection.")
    ap.add_argument("--fit-on-train-val", action="store_true",
                    help="after the setting is chosen on validation, draw the final "
                         "context from train AND val instead of train alone. The same "
                         "pattern as refitting on all data after cross-validation, and "
                         "RelBench permits training on both -- only test is held out, and "
                         "no test information is involved. The extra data is not marginal: "
                         "+43%% of rows on rel-f1 and +35%% on rel-avito. This is the one "
                         "lever that does not depend on the selection instrument, which "
                         "rejected every large effect measured on 2026-08-05.")
    ap.add_argument("--budget-categoricals", action="store_true",
                    help="apply --max-columns to the nunique block as well, which it never "
                         "has. Measured +0.99 (SE 0.10, 5/5) on rel-trial with all ten "
                         "child tables -- the budget bounds numeric columns only, so every "
                         "categorical column emits a distinct-count however narrow the "
                         "budget is, and with ten children that is 38 columns of nothing.")
    ap.add_argument("--numeric-booleans", action="store_true",
                    help="aggregate boolean child columns as numbers, so a boolean history "
                         "yields its rate. They are categorical by default, which gives "
                         "each one a single nunique of 1 or 2 -- and with mode and the "
                         "histogram both off, that has been their entire contribution to "
                         "every number in the table.")
    ap.add_argument("--gap-validation", action="store_true",
                    help="select on a pseudo-validation split carved out of train that "
                         "reproduces the train->test gap, instead of on the provided "
                         "validation split. RelBench's validation sits NEARER to train "
                         "than test does -- 7 days against 15 on rel-event, 365 against "
                         "731 on rel-trial, 150 against 1,976 on rel-f1 -- so a setting "
                         "chosen on it is tuned for a shorter horizon than the one it is "
                         "scored at, and anything whose value grows with distance from "
                         "the training period is systematically undervalued. Context "
                         "recency is exactly such a setting: +7.50 on rel-event test, "
                         "rejected 3/3 by validation.")
    ap.add_argument("--context-grid", default="",
                    help="comma-separated context sizes for the calibrated sweep, "
                         "overriding the default cap//4, cap//2, cap. The default grid "
                         "cannot see a recency effect: on rel-event `recent` is +0.31 at "
                         "context 10,000 and +7.50 at 1,000, because half the training "
                         "period is not recent and the last few days are.")
    ap.add_argument("--context-orders", default="random",
                    help="comma-separated context selection rules to offer validation: "
                         "random (uniform over train, the only one ever used), recent (the "
                         "most recent rows by timestamp), recent-half (uniform within the "
                         "recent half). Offered as a selectable axis rather than a flag, "
                         "so a win is chosen on validation and eligible for the table. "
                         "This is selectable where resampling was not: RelBench splits are "
                         "temporal, so the validation split is a LATER period than train "
                         "and can see a recency effect -- the instrument that failed on "
                         "resampling was CV over held-out train rows, which cannot.")
    ap.add_argument("--calendar", action="store_true",
                    help="day of week, day, month, weekend flag and their sine/cosine "
                         "pairs, from the prediction timestamp. Every datetime column is "
                         "dropped when the entity block is assembled -- the cutoff "
                         "included -- so nothing downstream can tell a Monday from a "
                         "Saturday, on tasks with four- and seven-day horizons.")
    ap.add_argument("--label-history", action="store_true",
                    help="this entity's OWN earlier outcomes: count, positive rate, last "
                         "label, days since it resolved. The task table is a timestamped "
                         "table keyed by entity and the pipeline has only ever read its "
                         "key, cutoff and label -- never the labels of that entity's "
                         "earlier rows. Gated horizon-correct before being built: on "
                         "rel-f1 this ONE scalar beats the whole pipeline, 84.66 vs 81.98 "
                         "on driver-top3 and 74.27 vs 69.66 on driver-dnf. Worth most "
                         "where entities recur; useless on rel-trial, where every study "
                         "has exactly one outcome and coverage is 0.0%%. The pool is the "
                         "fitting split only, and a row cannot read its own label because "
                         "that label resolves one horizon after its own cutoff.")
    ap.add_argument("--explicit-blocks", action="store_true", default=True,
                    help="treat an empty feature block as a fatal error (the default). A "
                         "block that was asked for and emitted nothing reports +0.00 with "
                         "sd 0.00 and reads exactly like a clean null -- that is how depth-2 "
                         "was 'measured at no effect' for a week. Pass --no-explicit-blocks "
                         "when a block is enabled as a DEFAULT rather than requested, since "
                         "categories emits nothing on four of the seven tasks here and a "
                         "default must not crash on a schema it does not fit.")
    ap.add_argument("--no-explicit-blocks", dest="explicit_blocks", action="store_false",
                    help="see --explicit-blocks.")
    ap.add_argument("--entity-time-deltas", action="store_true",
                    help="keep the entity table's own datetime columns as days before the "
                         "cutoff instead of dropping them. We delete them outright today; "
                         "RDBLearn converts them by default. Turns `dob` into age at "
                         "prediction time and `joinedAt` into account tenure -- quantities "
                         "nothing else here expresses. Gated standalone on test: joinedAt "
                         "57.90 on rel-event, dob 54.23 on rel-f1, start_date 49.11 on "
                         "rel-trial, and rel-avito has no such columns. A column whose value "
                         "lies after the cutoff is dropped with a note, decided once on the "
                         "fitting frame so the feature space cannot differ across splits.")
    ap.add_argument("--match-novelty", action="store_true",
                    help="subsample validation so its share of entities ALREADY SEEN in "
                         "train matches test's. Uses no labels -- which test entities are "
                         "new is known at inference time -- so this corrects validation's "
                         "POPULATION rather than deriving a signal from its ranking, which "
                         "is where every previous instrument failed. Motivated by an "
                         "inversion measured on rel-avito/user-visits: validation ranks "
                         "+struct best (77.16) and base worst (69.22) while test ranks them "
                         "the other way, and the val-test gap grows monotonically with how "
                         "much an arm leans on entity history (base 2.92, +rate 8.03, "
                         "+counts 8.36, +history 10.76, +struct 11.73). Validation entities "
                         "are 81.2%% seen against test's 58.6%%, so track-record features "
                         "look far better there than they will perform.")
    ap.add_argument("--row-chunk", choices=["off", "auto", "always"], default="off",
                    help="row-chunked column embedding, the package's own scaling feature, "
                         "which this runner has never used. 'auto' decides per call from the "
                         "real tensor shape and free VRAM, so it is a no-op where memory "
                         "already suffices. Needed on the large RelBench databases: "
                         "rel-stack/user-engagement is 1.36M train rows and 88,137 test rows "
                         "at 342 columns, and died without it. Default 'off' because every "
                         "standing number was measured that way and chunking should be "
                         "opted into rather than silently changing what a rerun reproduces.")
    ap.add_argument("--offload", choices=["off", "auto", "cpu", "disk"], default="off",
                    help="where to keep inference OUTPUT tensors. Independent of "
                         "--row-chunk and composes with it: chunking shrinks the "
                         "column-embedding ACTIVATIONS, offloading moves the OUTPUTS off "
                         "the GPU, and a shape can be too large for one and not the other. "
                         "TabICL is in-context, so context and queries pass through the "
                         "embedder together and the activation scales with "
                         "(n_context + n_query), not n_context -- which is how "
                         "rel-stack/user-badge died in its first forward pass at ~260k "
                         "combined rows x 342 columns on a 46 GB L40S, printing no "
                         "traceback. Default 'off': every standing number was measured "
                         "that way.")
    ap.add_argument("--disk-offload-dir", default="/workspace/offload",
                    help="directory for memory-mapped offload files, used only by "
                         "--offload disk. Without one the library disables disk "
                         "offloading and raises only once CPU memory is already short, "
                         "which on a memory-capped container is a race the OOM killer wins.")
    ap.add_argument("--train-pool", type=int, default=0,
                    help="subsample the fit pool to N rows after features are built, "
                         "before inference. The fit uses --context rows, so the remaining "
                         "train rows are materialised and never used: rel-stack/user-badge "
                         "builds ~48 GB of arm matrices to draw 10,000 rows, then dies when "
                         "inference buffers land on top. 0 disables it, which is every "
                         "standing number. Refuses non-random --context-orders, where a "
                         "uniform subsample would change what recency means.")
    ap.add_argument("--self-join", action="store_true",
                    help="add NFA neighbourhood features: aggregate over OTHER task rows "
                         "sharing a categorical value, as of the cutoff. The one traversal "
                         "shape this package lacked -- every other leaves the entity table "
                         "through a foreign key. Causal by construction, label-free.")
    ap.add_argument("--self-join-columns", type=int, default=3,
                    help="how many grouping columns --self-join may use, chosen by non-null "
                         "coverage among columns that are neither near-unique nor "
                         "near-constant.")
    ap.add_argument("--self-join-narrow", action="store_true",
                    help="emit only neighbourhood COUNT and MEAN, all-history, no windows. "
                         "The full menu is 150 columns on rel-trial against a base measured "
                         "at --max-columns 2, and width is not free here: the same budget "
                         "is worth +3.0 on one task and -19.5 on another. Use this to test "
                         "the neighbourhood rather than the column count.")
    ap.add_argument("--compress-to", type=int, default=0,
                    help="project each arm to N components (StandardScaler + PCA, fitted on "
                         "TRAIN only, reused for val and test) instead of reaching a "
                         "workable width by deleting columns. 0 disables it. Pair with "
                         "--max-columns none: the comparison worth making is selection vs "
                         "compression AT EQUAL WIDTH, not narrow vs wide.")
    ap.add_argument("--context-select", choices=["random", "knn", "mmd"], default="random",
                    help="how the context rows are CHOSEN, as opposed to how many "
                         "(--context-grid) or in what order (--context-orders). 'knn' takes "
                         "the rows nearest the queries, ranked by how many queries retrieve "
                         "them; label-free. 'mmd' is CRUMB (arXiv 2606.11473) at K=1: greedily "
                         "minimise maximum mean discrepancy to the queries, which adds a "
                         "REDUNDANCY penalty kNN lacks. Content has never been tested here.")
    ap.add_argument("--backbone", choices=["tabicl", "tabfm"], default="tabicl",
                    help="which tabular foundation model runs underneath. Every standing "
                         "number is 'tabicl'. 'tabfm' is Google Research's TabFM 1.0.0 -- "
                         "zero-shot, frozen, in-context, same paradigm -- measured test-side "
                         "at +3.28/+1.58/+0.45/+0.27/-1.70 across five tasks (mean +0.78). "
                         "This flag exists so that can be put through the CALIBRATED "
                         "protocol, which is the only thing that moves a published cell.")
    ap.add_argument("--drop-stale-arms", action="store_true",
                    help="exclude history-dependent arms when the shared-key block's "
                         "COVERAGE collapses between validation and test. Label-free: reads "
                         "links and timestamps only, so it is computable at inference time, "
                         "and it says whether a feature EXISTS on test rows rather than "
                         "judging its score -- unlike gap-validation, decide-fit-pool, "
                         "abstain and ensemble-configs, which all tried to read a better "
                         "answer out of the same validation scores and all failed. Coverage "
                         "falls 37.0%%->17.8%% on user-clicks, 43.1%%->30.8%% on user-visits "
                         "and 96.4%%->71.8%% on driver-top3, the three tasks where tuning "
                         "loses, and holds within a few points on the four where it gains.")
    ap.add_argument("--stale-threshold", type=float, default=0.8,
                    help="coverage ratio below which the block counts as stale. 0.8 sits in "
                         "the gap between the four tasks that hold (0.96-1.03) and the three "
                         "that collapse (0.48-0.74). Deliberately NOT tuned: there is no room "
                         "to fit a threshold on seven points without fitting the seven.")
    ap.add_argument("--dimensions", action="store_true",
                    help="carry a dimension table's attributes on the timestamped FACT rows "
                         "that reference it -- a star join. The fact row's clock governs, so "
                         "the dimension needs none of its own; \"nothing bounds an untimed "
                         "table to a cutoff\" was the wrong reason to skip these. Gated on "
                         "test at 100%% coverage: VisitStream x AdsInfo scores 68.5 on "
                         "rel-avito/user-visits against a 65.54 pipeline and 64.6 on "
                         "user-clicks against 65.89. The risk is an attribute that is "
                         "UPDATED rather than fixed -- a final-state snapshot would hand a "
                         "pre-cutoff row a post-cutoff value -- which is what the temporal "
                         "control is for.")
    ap.add_argument("--siblings", action="store_true",
                    help="aggregate tables that hang off a PARENT referenced by one of the "
                         "entity's children -- sibling tables through a shared parent. The "
                         "traversal here only ever went entity->child->grandchild, so these "
                         "were unreachable at any depth. On rel-f1 that is the constructor's "
                         "entire record: we predict whether a driver fails to finish with "
                         "nothing about the car. Gated on test before building: one "
                         "constructor column scores 68.0 on driver-dnf against a 69.66 "
                         "pipeline, and 74.0 on driver-top3 against 81.98, both at 100%% "
                         "coverage. Reachability is not novelty, though -- a driver's own "
                         "results already encode car quality, so the margin may be small.")
    ap.add_argument("--depth2", action="store_true",
                    help="aggregate grandchild tables reached through each child. Recorded "
                         "here as unavailable on the whole benchmark; that was true of "
                         "rel-f1 (no timestamped depth-2 in the schema) and rel-trial (0 of "
                         "158,246 grandchild rows precede the cutoff -- the subtree IS the "
                         "label), and FALSE of the other two, where the blocker was this "
                         "package refusing depth-2 whenever entity keys repeat. Measured "
                         "before building: 25.8%% of (grandchild, cutoff) pairs are usable "
                         "on rel-event via users->events->event_attendees and 23.8%% on "
                         "rel-avito via UserInfo->SearchInfo->SearchStream. RDBLearn, which "
                         "beats our average, defaults to depth 2.")
    ap.add_argument("--abstain", action="store_true",
                    help="don't tune when the validation ranking cannot be trusted. Splits "
                         "validation by time, picks the winner on the EARLY half, and keeps "
                         "it only if it still beats the untuned `base` arm on the LATE half; "
                         "otherwise falls back to base. This does NOT ask which "
                         "configuration is best -- that question is where every instrument "
                         "built from this split has failed -- it asks whether the ranking "
                         "holds up out of sample at all. Motivated by rel-avito, where "
                         "tuning loses on both tasks and user-clicks drops 1.29 (four places "
                         "in the field) because the arms are near-identical and selection is "
                         "fitting noise. Costs nothing: the same fit is scored on both "
                         "halves.")
    ap.add_argument("--label-history-control", action="store_true",
                    help="negative control for --label-history: keep the label-FREE columns "
                         "(n_prior, days_since) and drop the label-derived ones "
                         "(positive_rate, last). A gain that survives this is recurrence and "
                         "recency of activity, not track record. This distinction has "
                         "already caught one result here -- most of an apparent "
                         "label-history gain on rel-event turned out to be "
                         "key_target_history's structural n_linked column. Requires "
                         "--label-history.")
    ap.add_argument("--calendar-trend", action="store_true",
                    help="also emit days since the training minimum. Separate from "
                         "--calendar on purpose: this one is monotone, so every test row "
                         "lies beyond the training range on it and a model keying on it "
                         "extrapolates off the end of its own support.")
    ap.add_argument("--time-deltas", action="store_true",
                    help="emit days-since-last-child-row, days-since-first, and span, for "
                         "all-history and each window. The timestamp is the one column "
                         "never aggregated, so nothing in the feature set currently says "
                         "WHEN -- and a count over a 7-day window cannot tell a user who "
                         "searched once yesterday from one who searched once six days ago.")
    ap.add_argument("--mode", action="store_true",
                    help="emit the modal value of each categorical child column over its "
                         "all-history prefix. Entity-relative rather than corpus-relative, "
                         "so unlike the histogram it stays meaningful at high cardinality. "
                         "Also never set by this runner before now.")
    ap.add_argument("--timed-links-only", action="store_true",
                    help="use only link tables that carry a timestamp. Required for a "
                         "reportable structural result: an untimed table's degree is "
                         "constant under a cutoff shift, so the temporal control cannot "
                         "see it and passing proves nothing about it.")
    ap.add_argument("--static-links", action="store_true",
                    help="ignore link-table timestamps, counting memberships that formed "
                         "after the cutoff. Wrong; kept to measure what the causal "
                         "filtering is worth, which on rel-event is the whole result.")
    ap.add_argument("--calibrated", action="store_true",
                    help="choose the arm and the context size on validation, then score "
                         "test once. A paired A/B is not eligible for the headline table; "
                         "this is.")
    args = ap.parse_args()
    if args.label_history_control and not args.label_history:
        raise SystemExit(
            "--label-history-control without --label-history builds no block at all, so it "
            "would report a clean null for a control that never ran. Pass both."
        )

    db = get_dataset(args.dataset, download=True).get_db()
    task = get_task(args.dataset, args.task, download=True)
    key, target, entity = task.entity_col, task.target_col, task.entity_table
    train = task.get_table("train", mask_input_cols=False).df
    test = task.get_table("test", mask_input_cols=False).df
    tcol = next(c for c in train.columns if pd.api.types.is_datetime64_any_dtype(train[c]))
    horizon = None if args.no_horizon else getattr(task, "timedelta", None)
    y, y_te = train[target].to_numpy(), test[target].to_numpy()
    max_cols = None if str(args.max_columns).lower() in ("none", "null", "") else int(args.max_columns)
    spec = args.window_days or DEFAULT_WINDOWS.get(args.dataset, "30,365")
    WINDOWS = [pd.Timedelta(days=int(d)) for d in spec.split(",")]
    print(f"{args.dataset}/{args.task}  horizon={horizon}  windows={spec}  "
          f"train={len(train)} test={len(test)}", flush=True)

    # --- release the train rows this run will never use ----------------------------------
    # The fit draws `--context` rows, 10,000 by default. Everything below builds features
    # for EVERY train row, for every arm, and then samples from the result:
    #
    #   rel-amazon/user-churn      1 arm  x 4.71M x  52 =  2.0 GB   lived
    #   rel-stack/user-engagement  5 arms x 1.36M x 350 = 19.0 GB   lived
    #   rel-stack/user-badge       5 arms x 3.39M x 350 = 47.5 GB   died, post-construction
    #   rel-amazon/item-churn                                       died, DURING construction
    #
    # Scale alone is not the constraint -- rel-amazon/user-churn survived more train rows
    # AND more test rows than user-badge, on one 52-column arm. It is arms x rows x columns.
    #
    # This sits before feature construction rather than after it, which is the correction
    # that matters. Placed after, it rescued user-badge (whose construction completed and
    # which died when inference buffers landed on the resident matrices) and was useless for
    # item-churn, which never reached that point. Placed here it bounds the aggregation, the
    # matrices and the inference buffers alike, and every downstream index -- `y`,
    # `time_order`, the gap-validation pools -- is derived from `train` after the cut, so
    # they cannot fall out of alignment.
    if args.train_pool and len(train) > args.train_pool:
        if [o.strip() for o in args.context_orders.split(",") if o.strip()] != ["random"]:
            raise SystemExit(
                "--train-pool subsamples the fit pool uniformly, which is statistically "
                "equivalent for RANDOM context draws and not for recency-ordered ones: the "
                "most recent rows of a uniform subsample span a far wider window than the "
                "most recent rows overall. Refusing rather than quietly changing what "
                f"--context-orders {args.context_orders!r} means.")
        if args.train_pool < 5 * args.context:
            raise SystemExit(f"--train-pool {args.train_pool} is under 5x --context "
                             f"{args.context}; the pool would barely exceed the draw.")
        keep = np.sort(np.random.default_rng(0).choice(
            len(train), size=args.train_pool, replace=False))
        print(f"train pool: {len(train):,} -> {args.train_pool:,} rows (uniform, seed 0), "
              f"applied BEFORE feature construction; the fit draws {args.context:,}",
              flush=True)
        train = train.iloc[keep].reset_index(drop=True)
        y = y[keep]

    # --- base features -------------------------------------------------------------------
    ent_df = db.table_dict[entity].df
    pk = db.table_dict[entity].pkey_col
    all_kids = [(n, fk, t.time_col) for n, t in db.table_dict.items()
                for fk, pt in (t.fkey_col_to_pkey_table or {}).items()
                if pt == entity and t.time_col]
    if args.children_order == "name":
        all_kids = sorted(all_kids, key=lambda k: k[0])
    kids = all_kids if args.children <= 0 else all_kids[: args.children]
    if args.children <= 0 or args.children >= len(all_kids) or args.top_children:
        pass
    elif args.children_order == "dict":
        # Dictionary order is not a property of the schema, it is a property of how this
        # machine happened to build the DB. Measured: this host's first three on rel-trial
        # are ['designs', 'eligibilities', 'drop_withdrawals'] while another's are
        # ['conditions_studies', 'designs', 'drop_withdrawals'] -- same ten candidates,
        # different three used, so "--children 3" names a different feature set on
        # different machines and two runs of the "same" configuration are not comparable.
        print(f"WARNING: taking the first {args.children} of {len(all_kids)} child tables "
              f"in dict order, which differs between environments. Use --children-order "
              f"name for reproducibility, or --top-children to choose on validation.",
              flush=True)
    if args.top_children and len(all_kids) > args.top_children:
        # WHICH child tables, not how many. `--children N` sweeps the count and has been
        # swept; the identity has always been `all_kids[:N]`, which is dictionary order --
        # the same storage-order selection that put `kids[:3]` on rel-trial's three least
        # useful children and picked child text rows by row position instead of by time.
        #
        # `--top-keys` already does this for link tables and measured them 53 to 82 apart
        # on rel-event. There is no reason child tables are more equal than link tables,
        # and no reason dictionary order should find the good ones.
        #
        # Ranked on VALIDATION and univariately, so it costs no fit: a column's own AUC
        # against the validation target needs no model. Distance from chance, because a
        # strongly anti-correlated column is as informative as a correlated one.
        val_rank = task.get_table("val", mask_input_cols=False).df
        truth = val_rank[target].to_numpy()
        # Several permutations, averaged. The null being estimated is "the largest AUC a
        # block this wide reaches by chance", and that maximum is itself a random variable
        # -- one draw of it is as noisy as the thing it is correcting for. Three is enough
        # to stop the correction from being the loudest term, and costs only AUCs.
        rng_null = np.random.default_rng(0)
        shuffles = [rng_null.permutation(truth) for _ in range(3)]

        def best_column(block, labels):
            """Largest distance from chance any single column of this block reaches."""
            if len(np.unique(labels)) < 2:
                return 0.0
            best = 0.0
            for col in block.columns:
                values = block[col].to_numpy(dtype=np.float64)
                if not np.isfinite(values).any():
                    continue
                filled = np.nan_to_num(values, nan=float(np.nanmedian(values)),
                                       posinf=0.0, neginf=0.0)
                if len(np.unique(filled)) < 2:
                    continue
                best = max(best, abs(roc_auc_score(labels, filled) * 100 - 50.0))
            return best

        scored = []
        for spec in all_kids:
            n, fk, tc = spec
            table = Table(db.table_dict[n].df, fk, n, time_column=tc, windows=WINDOWS,
                          max_columns=max_cols)
            block = asof_statistics(table, val_rank[key].to_numpy(),
                                    val_rank[tcol].to_numpy())
            # A maximum over k columns grows with k even under pure noise, so ranking
            # children by their best column would rank them by how WIDE they are. The same
            # maximum against a permuted target measures exactly that width, on this
            # block's own column count and missingness -- so the difference is the part
            # that is about signal. Same idea as the permutation control, applied to a
            # selection step rather than to a result.
            signal = best_column(block, truth)
            null = float(np.mean([best_column(block, s) for s in shuffles]))
            scored.append((signal - null, signal, null, len(block.columns), spec))
        scored.sort(key=lambda r: -r[0])
        print("child ranking on validation (best column above its own permuted null):",
              flush=True)
        for margin, signal, null, width, spec in scored:
            print(f"  {spec[0]:<22} {margin:>+6.1f}  (best {signal:.1f}, "
                  f"null {null:.1f}, {width} cols)", flush=True)
        kids = [s[4] for s in scored[: args.top_children]]
    print(f"child tables: using {len(kids)} of {len(all_kids)} available "
          f"{[k[0] for k in kids]}", flush=True)

    def build_base(frame, split="train"):
        base = frame.merge(ent_df, left_on=key, right_on=pk, how="left")
        drop = {target, key, pk, tcol}
        cols = [c for c in base.columns
                if c not in drop and not pd.api.types.is_datetime64_any_dtype(base[c])]
        blocks = [base[cols].reset_index(drop=True)]
        if args.entity_time_deltas:
            # The entity table's own datetime columns, kept as days BEFORE the cutoff rather
            # than dropped. RDBLearn does this by default; we deleted the information
            # outright. `dob` becomes age at prediction time, `joinedAt` becomes tenure --
            # quantities nothing else in the pipeline expresses.
            #
            # A datetime that lies AFTER the cutoff is a fact about the future, and no amount
            # of differencing fixes that. The admissible set is decided ONCE, on the fitting
            # frame, and reused -- deciding per split would let a column live in train and
            # vanish in test, which is a different feature space, not a safer one.
            dts = [c for c in base.columns
                   if c not in drop and pd.api.types.is_datetime64_any_dtype(base[c])]
            cut = frame[tcol].to_numpy()
            if split == "train" and "entity_time" not in _TIME_DELTA_COLS:
                keep = []
                for c in dts:
                    d = (cut - base[c].to_numpy()) / np.timedelta64(1, "D")
                    ahead = float(np.nanmean(d < 0)) if np.isfinite(d).any() else 1.0
                    if ahead > 0.005:
                        print(f"entity time delta: DROPPING {c!r} -- {ahead:.1%} of training "
                              f"rows have it after the cutoff, so it describes the future",
                              flush=True)
                    else:
                        keep.append(c)
                _TIME_DELTA_COLS["entity_time"] = keep
                print(f"entity time deltas: keeping {keep or 'nothing'} of {dts or 'none'}",
                      flush=True)
            kept = _TIME_DELTA_COLS.get("entity_time", [])
            if kept:
                blocks.append(pd.DataFrame(
                    {f"dt__{c}": (cut - base[c].to_numpy()) / np.timedelta64(1, "D")
                     for c in kept}).reset_index(drop=True))
        if args.calendar or args.calendar_trend:
            # Origin is the training minimum, so train and test share a scale. Taking each
            # frame's own minimum would silently reset it at test time.
            blocks.append(calendar_features(frame[tcol].to_numpy(),
                                            trend=args.calendar_trend,
                                            origin=pd.Timestamp(train[tcol].min())))
        for n, fk, tc in kids:
            t = Table(db.table_dict[n].df, fk, n, time_column=tc, windows=WINDOWS,
                      max_columns=max_cols,
                      top_k_categories=args.categories or None,
                      min_category_share=args.category_share,
                      numeric_booleans=args.numeric_booleans,
                      budget_categoricals=args.budget_categoricals,
                      time_deltas=args.time_deltas,
                      include_mode=args.mode)
            blocks.append(asof_statistics(t, frame[key].to_numpy(),
                                          frame[tcol].to_numpy()).add_prefix(f"{n}__"))
            if args.dimensions:
                # STAR JOIN: a dimension table's attributes carried on the FACT row that
                # references it. The fact row is timestamped, so the as-of filter is
                # unchanged and the dimension needs no clock of its own -- which is why
                # "nothing bounds an untimed table to a cutoff" was the wrong reason to
                # skip these. If a pre-cutoff fact row references an ad, the entity saw
                # that ad, and its category and location were what they were.
                #
                # The risk is an attribute that is UPDATED rather than fixed: a final-state
                # snapshot would hand a pre-cutoff row a post-cutoff value. Structural keys
                # (category, location) cannot change without being a different thing; a
                # price or title could. That is what the temporal control is for.
                ctab = db.table_dict[n]
                for pcol, pname in (ctab.fkey_col_to_pkey_table or {}).items():
                    if pname == entity or pname not in db.table_dict:
                        continue
                    ptab = db.table_dict[pname]
                    if not ptab.pkey_col:
                        continue
                    dim = ptab.df.drop(columns=[c for c in ptab.df.columns
                                                if pd.api.types.is_datetime64_any_dtype(ptab.df[c])],
                                       errors="ignore")
                    fact = ctab.df[[fk, tc, pcol]].dropna(subset=[fk, tc, pcol])
                    star = fact.merge(dim, left_on=pcol, right_on=ptab.pkey_col, how="inner")
                    star = star.drop(columns=[c for c in (pcol, ptab.pkey_col)
                                              if c in star.columns and c != fk])
                    if star.empty:
                        continue
                    t_dim = Table(star, fk, f"{n}_{pname}", time_column=tc, windows=WINDOWS,
                                  max_columns=max_cols,
                                  top_k_categories=args.categories or None,
                                  min_category_share=args.category_share,
                                  numeric_booleans=args.numeric_booleans,
                                  budget_categoricals=args.budget_categoricals,
                                  time_deltas=args.time_deltas, include_mode=args.mode)
                    dim_built.setdefault(split, []).append(
                        f"{n}x{pname} ({len(star):,} rows)")
                    blocks.append(asof_statistics(
                        t_dim, frame[key].to_numpy(),
                        frame[tcol].to_numpy()).add_prefix(f"{n}x{pname}__"))
            if args.siblings:
                # SIBLING tables, reached through a parent this child references. The
                # traversal so far only ever went entity -> child -> grandchild, so a table
                # hanging off a child's PARENT was unreachable at any depth. On rel-f1 that
                # is the whole of the constructor's record: we predict whether a driver
                # fails to finish with nothing at all about the car. Gated first -- one
                # constructor column scores 68.0 on driver-dnf against a 69.66 pipeline.
                ctab = db.table_dict[n]
                for pcol, ptab_name in (ctab.fkey_col_to_pkey_table or {}).items():
                    if ptab_name == entity:
                        continue                      # that is the link to the entity itself
                    for sname, stab in db.table_dict.items():
                        sfks = stab.fkey_col_to_pkey_table or {}
                        if sname == n or ptab_name not in sfks.values():
                            continue
                        if stab.time_col is None:
                            continue                  # untimed: nothing bounds the cutoff
                        sfk = next(k for k, v in sfks.items() if v == ptab_name)
                        sib = two_hop_table(
                            stab.df, sfk, ctab.df, pcol, fk, f"{n}_{ptab_name}_{sname}",
                            time_column=stab.time_col, child_time_column=tc,
                            windows=WINDOWS, max_columns=max_cols,
                            top_k_categories=args.categories or None,
                            min_category_share=args.category_share,
                            numeric_booleans=args.numeric_booleans,
                            budget_categoricals=args.budget_categoricals,
                            time_deltas=args.time_deltas, include_mode=args.mode)
                        sibling_built.setdefault(split, []).append(
                            f"{n}->{ptab_name}->{sname} ({len(sib.df):,} rows)")
                        blocks.append(asof_statistics(
                            sib, frame[key].to_numpy(),
                            frame[tcol].to_numpy()).add_prefix(f"{sname}_via_{ptab_name}__"))
            if args.depth2:
                # Grandchildren reached THROUGH this child. `two_hop_table` relabels them by
                # the entity and keeps their own clock, so this is an ordinary depth-1 as-of
                # aggregation and needs no per-child cutoff arithmetic -- which is what made
                # the old depth-2 path refuse whenever entity keys repeated.
                ctab = db.table_dict[n]
                for gname, gtab in db.table_dict.items():
                    gfks = gtab.fkey_col_to_pkey_table or {}
                    if n not in gfks.values() or gtab.time_col is None or not ctab.pkey_col:
                        continue
                    gfk = next(k for k, v in gfks.items() if v == n)
                    two = two_hop_table(
                        gtab.df, gfk, ctab.df, ctab.pkey_col, fk, f"{n}_{gname}",
                        time_column=gtab.time_col,
                        # The link's own timestamp, so a membership formed after the cutoff
                        # cannot admit its grandchildren. Omitting it is the permissive
                        # reading and this benchmark has already paid for that once.
                        child_time_column=tc,
                        windows=WINDOWS, max_columns=max_cols,
                        top_k_categories=args.categories or None,
                        min_category_share=args.category_share,
                        numeric_booleans=args.numeric_booleans,
                        budget_categoricals=args.budget_categoricals,
                        time_deltas=args.time_deltas, include_mode=args.mode)
                    # Per-split, not cumulative. Appending to one shared list made the
                    # count grow with every frame built -- the log read "2 path(s)" on train,
                    # "4" on test and "6" on val for the same two paths. The names were
                    # deduped for display so it looked almost right, which is the worst way
                    # for a count to be wrong.
                    built = depth2_built.setdefault(split, [])
                    built.append(f"{n}->{gname} ({len(two.df):,} rows)")
                    blocks.append(asof_statistics(
                        two, frame[key].to_numpy(),
                        frame[tcol].to_numpy()).add_prefix(f"{n}_{gname}__"))
        if args.label_history:
            # The label pool is the FITTING pool and nothing else. Every other choice here
            # is a protocol question wearing a feature costume: letting test rows read
            # validation outcomes would be defensible at inference time and indefensible
            # in a table that compares against methods which did not. Self-exclusion is
            # structural inside `entity_label_history` -- a row's own outcome resolves one
            # horizon after its own cutoff -- so nothing about `frame` needs checking here.
            hist = entity_label_history(
                train[key].to_numpy(), train[target].to_numpy(),
                train[tcol].to_numpy(), frame[key].to_numpy(),
                frame[tcol].to_numpy(), label_horizon=task.timedelta)
            if args.label_history_control:
                # NEGATIVE CONTROL. `n_prior` and `days_since` say how often and how
                # recently this entity appeared; they consult no outcome. `positive_rate`
                # and `last` are the only label-derived columns, and dropping them leaves a
                # block that is pure activity. If the control reproduces the gain, the gain
                # is recurrence, not track record -- which is exactly what happened to
                # `key_target_history` on rel-event, where most of an apparent label-history
                # effect turned out to be its structural `n_linked` column.
                hist = hist.drop(columns=["self__positive_rate", "self__last"])
            blocks.append(hist)
        out = pd.concat(blocks, axis=1)
        _audit_requested_blocks(out, split)
        return out

    depth2_built: dict = {}
    sibling_built: dict = {}
    dim_built: dict = {}
    reported = set()

    def _audit_requested_blocks(frame, split="train"):
        """Refuse to let a requested feature block be silently empty.

        Depth-2 spent a week "measured at no effect" because `max_columns` was deleting
        every grandchild column before the model saw one: an empty block does not raise,
        it reports +0.00 with sd 0.00 and reads exactly like a clean null. The categorical
        histogram has the same failure mode and its own gate -- `min_category_share` drops
        columns whose codebook captures too little, and on a schema of free-text columns
        that is *all* of them. So count what actually came out, once, and say it.
        """
        # Label history fails differently: the columns are always PRESENT, and on an entity
        # that never recurs they are simply all-NaN. Coverage is the quantity, not column
        # count -- rel-trial has `entities == rows` and would report a clean +0.00 null
        # forever. Say the coverage out loud and refuse the degenerate case.
        # Reported PER SPLIT, and deliberately not deduped across splits. Train coverage is
        # not the quantity that matters and actively misleads: on rel-f1 it reads 93% while
        # test reads 35%, because drivers race in seasons and the pool freezes at the end
        # of train. Ranking this feature on its training-row behaviour got the tasks
        # backwards once already.
        # Depth-2 reports what it BUILT, per split, and refuses to be silently empty. The
        # old depth-2 work spent a week "measured at no effect" because three separate
        # defects each emitted +0.00 with sd 0.00 -- output indistinguishable from a careful
        # null. A list of the paths actually materialised is the cheapest defence against
        # repeating that.
        if args.dimensions and f"dimensions {split}" not in reported:
            reported.add(f"dimensions {split}")
            built = dim_built.get(split, [])
            if not built:
                raise SystemExit(
                    "--dimensions was requested but no star join was built: no child of the "
                    "entity references a parent table with a primary key. An empty block "
                    "scores +0.00 with sd 0.00 and reads like a careful null."
                )
            print(f"dimensions [{split}]: {len(built)} join(s) -- {'; '.join(built)}",
                  flush=True)
        if args.siblings and f"siblings {split}" not in reported:
            reported.add(f"siblings {split}")
            built = sibling_built.get(split, [])
            if not built:
                raise SystemExit(
                    "--siblings was requested but no sibling table was built: no child of "
                    "the entity references a parent that has another TIMESTAMPED child. An "
                    "empty block scores +0.00 with sd 0.00 and reads like a careful null."
                )
            print(f"siblings [{split}]: {len(built)} path(s) -- {'; '.join(built)}",
                  flush=True)
        if args.depth2 and f"depth2 {split}" not in reported:
            reported.add(f"depth2 {split}")
            built = depth2_built.get(split, [])
            if not built:
                raise SystemExit(
                    "--depth2 was requested but no grandchild table was built. Either no "
                    "child has a timestamped child of its own, or the child tables lack "
                    "primary keys. On rel-f1 this is correct and structural; anywhere else "
                    "it means the traversal found nothing and any gap measured would be an "
                    "artefact of an empty block."
                )
            print(f"depth-2 [{split}]: {len(built)} path(s) -- "
                  f"{'; '.join(built)}", flush=True)
        if args.label_history and f"label history {split}" not in reported:
            reported.add(f"label history {split}")
            n_prior = frame["self__n_prior"]
            coverage = float((n_prior > 0).mean())
            print(f"label history [{split}]: {coverage:.1%} of rows have a resolved earlier "
                  f"outcome for the same entity (horizon {task.timedelta})", flush=True)
            if coverage == 0.0 and split == "train":
                raise SystemExit(
                    "--label-history was requested but NO row has a resolved earlier "
                    "outcome, so every column is NaN and any gap measured would be an "
                    "artefact of an empty block. This is structural where an entity "
                    "appears once (rel-trial: entities == rows), not a bug to work around."
                )
        for flag, suffix, label in ((args.categories, "__cat0", "category histogram"),
                                    (args.mode, "__mode", "prefix mode")):
            if not flag or label in reported:
                continue
            reported.add(label)
            got = [c for c in frame.columns if c.endswith(suffix)]
            if not got:
                # ABORT when a human asked for the block, SKIP when it is only a default.
                # An explicitly requested empty block reporting +0.00 is how depth-2 was
                # "measured at no effect" for a week, so that case must stop the run. But a
                # default cannot abort: categories emits nothing on FOUR of the seven tasks
                # here -- both rel-avito and both rel-f1, where --category-share rejects
                # every categorical column as free text -- and a user on such a schema would
                # get a crash instead of a model.
                message = (
                    f"the {label} emitted no columns. Likely the --category-share gate "
                    f"({args.category_share}) rejected every categorical column as free "
                    f"text, which is a fact about this schema rather than a failure."
                )
                # Was this block ASKED FOR, or is it just on by default? argparse cannot
                # tell -- `--categories 8` and the default 8 are the same namespace -- and
                # the distinction decides whether an empty block aborts or is skipped.
                # sys.argv can tell, so it does.
                flag = "--" + label.split()[0]
                asked = any(a.split("=")[0] == flag for a in sys.argv[1:])
                if args.explicit_blocks and asked:
                    raise SystemExit(
                        f"--{label.split()[0]} was requested but {message} Any gap measured "
                        f"here would be an artefact of an empty block."
                    )
                print(f"NOTE: {message} Continuing without it -- this arm is therefore "
                      f"identical to the one without the flag, and any difference between "
                      f"them is noise.", flush=True)
                continue
            print(f"{label}: {len(got)} columns over "
                  f"{len({c.split('__')[0] for c in got})} child tables", flush=True)

    # --- which keys does this schema even offer? ----------------------------------------
    link_specs = []
    for name, tbl in db.table_dict.items():
        for fk, pt in (tbl.fkey_col_to_pkey_table or {}).items():
            if pt != entity:
                continue
            for other in (tbl.fkey_col_to_pkey_table or {}):
                if other != fk:
                    # Qualify by table: the same key name appears in several link tables
                    # (rel-event has `event` twice, rel-f1 `raceId` three times) and an
                    # unqualified prefix silently collides the blocks on concat.
                    short = f"{name}_{other}".replace("_id", "").replace("_ID", "")
                    tc = tbl.time_col if not args.static_links else None
                    cols = [fk, other] + ([tc] if tc else [])
                    link_specs.append((short, tbl.df[cols], fk, other, tc))
    if args.timed_links_only:
        # Control 4 is BLIND to untimed link tables: with no time filter their n_linked
        # does not change when the cutoff is shifted, so it contributes nothing for the
        # control to detect and a pass says nothing about them. Dropping them is what makes
        # a structure-only result defensible rather than merely untested.
        dropped = [s[0] for s in link_specs if not s[4]]
        link_specs = [s for s in link_specs if s[4]]
        if dropped:
            print(f"dropping untimed link tables {dropped}: their structural counts cannot "
                  f"be temporally controlled", flush=True)
    if args.top_keys and len(link_specs) > args.top_keys:
        # Rank on VALIDATION, never on test: this is a selection step like any other, and
        # ranking it on test is the error the calibrated protocol exists to prevent.
        val_rank = task.get_table("val", mask_input_cols=False).df
        scored = []
        for spec in link_specs:
            short, frame, fk, other, ltc = spec
            block = key_target_history(
                frame[[fk, other]], label_entities=train[key].to_numpy(),
                label_values=y, label_times=train[tcol].to_numpy(),
                query_entities=val_rank[key].to_numpy(),
                query_times=val_rank[tcol].to_numpy(), label_horizon=horizon,
                link_times=frame[ltc].to_numpy() if ltc else None)
            rate = block["hist__positive_rate"].to_numpy()
            truth = val_rank[target].to_numpy()
            filled = np.where(np.isnan(rate), y.mean(), rate)
            auc = (roc_auc_score(truth, filled) * 100
                   if len(np.unique(truth)) > 1 else 50.0)
            # Rank by distance from chance: a strongly *anti*-correlated key is as
            # informative as a correlated one, and a rate near 50 is the useless case.
            scored.append((abs(auc - 50.0), auc, spec))
        scored.sort(key=lambda r: -r[0])
        print("key ranking on validation: "
              + ", ".join(f"{s[2][0]} {s[1]:.1f}" for s in scored), flush=True)
        link_specs = [s[2] for s in scored[: args.top_keys]]
        print(f"keeping top {args.top_keys}: {[s[0] for s in link_specs]}", flush=True)

    timed = [s[0] for s in link_specs if s[4]]
    untimed = [s[0] for s in link_specs if not s[4]]
    print(f"candidate keys: {[s[0] for s in link_specs] or 'NONE'}", flush=True)
    print(f"  link tables WITH timestamps (causal): {timed or 'none'}", flush=True)
    if untimed:
        print(f"  link tables WITHOUT timestamps: {untimed} -- memberships formed after a "
              f"cutoff are visible to it, so any lift from those keys is an UPPER BOUND",
              flush=True)
    if not link_specs:
        print("no table links two entities of this type -- this feature cannot be built "
              "on this task", flush=True)
        return

    if args.gate:
        # Standalone AUC per key, before any base features or GPU work. A key worth less
        # than the pipeline it must improve has no room; that ratio is what separated
        # rel-trial (61 against 66.5, worked) from rel-event's graph version (68 against
        # 83, did not).
        val = task.get_table("val", mask_input_cols=False).df
        print(f"\n{'key':<20} {'coverage':>9} {'val AUC':>9} {'test AUC':>9}", flush=True)
        for short, frame, fk, other, ltc in link_specs:
            row = []
            for split in (val, test):
                block = key_target_history(
                    frame[[fk, other]], label_entities=train[key].to_numpy(),
                    label_values=y, label_times=train[tcol].to_numpy(),
                    query_entities=split[key].to_numpy(),
                    query_times=split[tcol].to_numpy(), label_horizon=horizon,
                    link_times=frame[ltc].to_numpy() if ltc else None,
                )
                rate = block["hist__positive_rate"].to_numpy()
                truth = split[target].to_numpy()
                cov = float(np.mean(~np.isnan(rate)))
                filled = np.where(np.isnan(rate), y.mean(), rate)
                auc = (roc_auc_score(truth, filled) * 100
                       if len(np.unique(truth)) > 1 else float("nan"))
                row.append((cov, auc))
            print(f"{short:<20} {row[0][0]:>9.3f} {row[0][1]:>9.2f} {row[1][1]:>9.2f}",
                  flush=True)
        print("\nCompare against the task's existing calibrated number before building.",
              flush=True)
        return

    b_tr = build_base(train)
    b_te = build_base(test, "test").reindex(columns=b_tr.columns, fill_value=np.nan)

    # --- NFA: inter-row structure WITHIN the table (--self-join) --------------------------
    # Every other traversal here leaves the entity table through a foreign key. This one
    # links task rows to each other by SHARED ATTRIBUTE VALUES -- the incidence graph of
    # Cucumides & Geerts (arXiv 2602.03945), whose measurement on relbench-trial is the
    # reason to try it: fixed within-table aggregations (0.7254) beat row-local LightGBM
    # (0.7009), both GNN baselines (0.6860, 0.6861), a relational foundation model (0.7116
    # finetuned) AND learned message passing over the cross-table structure we already build
    # (0.7180). No retraining, which is our constraint.
    #
    # Neighbours are drawn from train + test rows together, and that is label-free by
    # construction: only ATTRIBUTES are aggregated, never a label or a target. Restricting
    # neighbours to train would make a test row's neighbourhood systematically older than a
    # train row's, which is the population mismatch this project has already measured as the
    # cause of its val/test inversion.
    #
    # Causality needs no control: a row's cutoff IS its timestamp and the as-of scan counts
    # strictly earlier rows, so no row is ever its own neighbour or sees a contemporaneous
    # one.
    nfa_tr = nfa_te = nfa_va = None
    if args.self_join:
        from tabicl.scaling._relational import nfa_columns, neighbour_aggregates
        # All three splits in one pool. The calibrated path selects on validation, so a
        # `+nfa` train/test pair without a matching validation arm would KeyError during
        # selection -- and building validation's neighbourhoods from a different pool than
        # test's would change what the feature MEANS between the split that chooses it and
        # the split it is judged on, which is the exact failure this project traced its
        # val/test inversion to.
        _val_nfa = task.get_table("val", mask_input_cols=False).df
        pool = pd.concat([train.assign(__split=0), _val_nfa.assign(__split=2),
                          test.assign(__split=1)], ignore_index=True)
        pool = pool.merge(ent_df, left_on=key, right_on=pk, how="left")
        gcols = nfa_columns(pool, exclude={target, key, pk, tcol, "__split"},
                            max_columns=args.self_join_columns)
        if not gcols:
            print("  --self-join: no column groups rows usefully (every candidate is "
                  "near-unique or near-constant); skipping", flush=True)
        else:
            vcols = [c for c in pool.columns
                     if c not in {target, key, pk, tcol, "__split", *gcols}
                     and pd.api.types.is_numeric_dtype(pool[c])
                     and not pd.api.types.is_bool_dtype(pool[c])]
            print(f"  --self-join: grouping on {gcols}, aggregating {len(vcols)} numeric "
                  f"columns over {len(pool):,} rows", flush=True)
            blk = neighbour_aggregates(
                pool, tcol, gcols, value_columns=vcols,
                windows=None if args.self_join_narrow else WINDOWS,
                keep_stats=("__count", "__mean") if args.self_join_narrow else None)
            nfa_tr = blk[pool["__split"] == 0].reset_index(drop=True)
            nfa_te = blk[pool["__split"] == 1].reset_index(drop=True)
            nfa_va = blk[pool["__split"] == 2].reset_index(drop=True)
            print(f"  --self-join: {blk.shape[1]} neighbourhood columns", flush=True)

    # --- text block (RESEARCH 6f) --------------------------------------------------------
    # Fitted on TRAIN ONLY and applied to val/test. Fitting the vectoriser on all splits
    # would let test vocabulary and IDF weights inform the representation -- a leak that
    # produces a large confident number rather than an error, which is this family's
    # signature failure.
    text_cols, text_models, child_text = [], {}, []
    child_series = None
    if args.text:
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.decomposition import TruncatedSVD
        from sklearn.pipeline import make_pipeline

        ent_text = db.table_dict[entity].df
        merged_tr = train.merge(ent_text, left_on=key, right_on=pk, how="left")

        # Child-table text, aggregated AS OF each row's cutoff. The gate found
        # `eligibilities.criteria` at 63.95 with full coverage -- as strong as the best
        # entity column and not otherwise used. It concatenated without regard to time,
        # so its number was an upper bound; here only rows dated at or before the cutoff
        # contribute. Ignoring that is what inflated rel-event's structural result by 8.
        child_text = []
        for cname, ctbl in db.table_dict.items():
            cfk = next((fk for fk, pt in (ctbl.fkey_col_to_pkey_table or {}).items()
                        if pt == entity), None)
            if cfk is None or not ctbl.time_col:
                continue
            for col in ctbl.df.columns:
                if col in (cfk, ctbl.time_col):
                    continue
                s = ctbl.df[col]
                if not pd.api.types.is_object_dtype(s):
                    continue
                nn = s.dropna().astype(str)
                if len(nn) < 20 or nn.str.len().mean() < 15:
                    continue
                child_text.append((f"{cname}.{col}", cname, cfk, col, ctbl.time_col))
        if child_text:
            print(f"child text columns: {[c[0] for c in child_text]}", flush=True)

        def child_series(frame, spec):
            _, cname, cfk, col, ctc = spec
            src = db.table_dict[cname].df[[cfk, ctc, col]].dropna(subset=[cfk, col])
            q = pd.DataFrame({"_row": np.arange(len(frame)),
                              cfk: frame[key].to_numpy(),
                              "_cut": frame[tcol].to_numpy()})
            j = q.merge(src, on=cfk, how="left")
            j = j[j[ctc].isna() | (j[ctc] <= j["_cut"])]
            # Most RECENT rows, not the first twenty in dataframe order. The cap is a cost
            # control, but taking "whichever pandas happened to store first" made the
            # feature depend on storage order -- and where an entity has more rows than the
            # cap, recency is the only defensible tie-break for an as-of feature.
            j = j.sort_values(ctc, kind="stable", na_position="first")
            agg = (j.dropna(subset=[col]).astype({col: str})
                   .groupby("_row")[col].apply(lambda v: " ".join(v.tail(args.text_rows))))
            out = pd.Series("", index=range(len(frame)), dtype=object)
            out.loc[agg.index] = agg.to_numpy()
            return out

        for spec in child_text:
            merged_tr[spec[0]] = child_series(train, spec).to_numpy()
        entity_texty = [c for c in ent_text.columns
                        if c not in (pk, target)
                        and pd.api.types.is_object_dtype(ent_text[c])
                        and len(ent_text[c].dropna()) >= 20
                        and ent_text[c].dropna().astype(str).str.len().mean() >= 15]
        for col in entity_texty + [c[0] for c in child_text]:
            texts = merged_tr[col].fillna("").astype(str)
            if texts.str.len().sum() == 0:
                continue
            # Components must fit the column's own vocabulary; a fixed 64 killed
            # biospec_retention outright at 11 features.
            try:
                vec = TfidfVectorizer(sublinear_tf=True, min_df=3, max_features=50000,
                                      ngram_range=(1, 2), strip_accents="unicode")
                n_feat = vec.fit(texts).transform(texts[:1]).shape[1]
                k = int(min(args.text_components, max(2, n_feat - 1)))
                model = make_pipeline(
                    TfidfVectorizer(sublinear_tf=True, min_df=3, max_features=50000,
                                    ngram_range=(1, 2), strip_accents="unicode"),
                    TruncatedSVD(n_components=k, random_state=0))
                model.fit(texts)
            except Exception as exc:          # noqa: BLE001
                print(f"  text column {col} skipped: {str(exc)[:60]}", flush=True)
                continue
            text_cols.append(col)
            text_models[col] = model
        print(f"text columns embedded: {text_cols or 'none'}", flush=True)

    def text_block(frame):
        if not text_cols:
            return pd.DataFrame(index=range(len(frame)))
        merged = frame.merge(db.table_dict[entity].df, left_on=key, right_on=pk, how="left")
        for spec in child_text:
            if spec[0] in text_cols:
                merged[spec[0]] = child_series(frame, spec).to_numpy()
        blocks = []
        for col in text_cols:
            emb = text_models[col].transform(merged[col].fillna("").astype(str))
            blocks.append(pd.DataFrame(
                emb, columns=[f"txt_{col}_{i}" for i in range(emb.shape[1])]))
        return pd.concat(blocks, axis=1)

    x_tr, x_te = text_block(train), text_block(test)

    def track(entities, times, labels, shift=None):
        stamps = np.asarray(times)
        if shift:
            stamps = stamps - pd.Timedelta(days=shift)
        blocks = []
        for short, frame, fk, other, ltc in link_specs:
            blocks.append(key_target_history(
                frame[[fk, other]], label_entities=train[key].to_numpy(),
                label_values=labels, label_times=train[tcol].to_numpy(),
                query_entities=entities, query_times=stamps,
                label_horizon=horizon, prefix=f"{short}__",
                link_times=frame[ltc].to_numpy() if ltc else None,
            ))
        return pd.concat(blocks, axis=1) if blocks else pd.DataFrame(index=range(len(entities)))

    t_tr = track(train[key].to_numpy(), train[tcol].to_numpy(), y)
    t_te = track(test[key].to_numpy(), test[tcol].to_numpy(), y)
    count_cols = [c for c in t_tr.columns if c.endswith("n_prior")]
    struct_cols = [c for c in t_tr.columns if c.endswith("n_linked")]
    print(f"track-record blocks: {list(t_tr.columns)}", flush=True)

    # --- controls, before any comparison -------------------------------------------------
    def rate_block(labels, shift=None):
        block = track(test[key].to_numpy(), test[tcol].to_numpy(), labels, shift=shift)
        return block[[c for c in block.columns if c.endswith("positive_rate")]].mean(axis=1)

    def rate_only_score(labels, shift=None):
        rate = rate_block(labels, shift=shift)
        return roc_auc_score(y_te, rate.fillna(np.nanmean(labels)).to_numpy())

    print("\ncontrol 1: permutation (null = permuted distribution)", flush=True)
    perm = permutation_test(rate_only_score, y, n_permutations=5, n_sigma=3.0)
    print(f"  {perm!r}", flush=True)

    # Shifts must be scaled to the task, not fixed in days. A shift far larger than the
    # task's own time span removes *every* usable label, the feature goes constant, and
    # the control passes at exactly 0.5 having tested nothing -- which is what a hardcoded
    # 180/365 days did on rel-event, whose horizon is 7 days. Coverage is printed so a
    # vacuous pass is visible rather than reassuring.
    # ...and a shift far SMALLER than the label horizon withholds nothing, because an
    # outcome only becomes readable one horizon after it is recorded. On rel-avito this grid
    # degenerated completely: train span is 8 days, so round(0.05*8)=0 and round(0.15*8)=1.
    # The first "control" was the unshifted setting itself, and the only real shift was 1 day
    # against a 4-day horizon -- coverage moved 0.572 to 0.571, i.e. nothing was withheld --
    # yet the 0.0053 difference that produced tripped the leak verdict and EXCLUDED the whole
    # `+rate` family from selection on that dataset. Both ends now have a guard.
    span_days = float((train[tcol].max() - train[tcol].min()) / pd.Timedelta(days=1))
    horizon_days = (float(horizon / pd.Timedelta(days=1)) if horizon is not None else 0.0)
    candidates = temporal_shift_grid(span_days, horizon_days)
    shifts = (0.0, *candidates)
    print(f"control 2: temporal (span {span_days:.0f}d, horizon {horizon_days:.0f}d, "
          f"shifts {tuple(candidates)}d)", flush=True)
    if not candidates:
        print("  *** control CANNOT RUN: no shift is both positive and at least one "
              "horizon. Any verdict here would be about nothing.", flush=True)
    base_vals = rate_block(y)
    base_cov = float(base_vals.notna().mean())
    # Coverage is a poor proxy for "did this shift withhold anything". On rel-avito a full
    # 4-day rewind moves coverage only 0.572 -> 0.568, because users have dense histories and
    # dropping four days rarely removes a user's LAST event -- but it can still change what
    # every rate is computed over. What matters is whether the feature VALUES moved.
    moved = 0.0
    for s in candidates:
        sh = rate_block(y, shift=s)
        cov = float(sh.notna().mean())
        a, b = base_vals.to_numpy(dtype=float), sh.to_numpy(dtype=float)
        both_nan = np.isnan(a) & np.isnan(b)
        changed = float(np.mean(~(both_nan | np.isclose(np.nan_to_num(a), np.nan_to_num(b)))))
        moved = max(moved, changed)
        print(f"  coverage at -{s:.0f}d: {cov:.3f} (unshifted {base_cov:.3f}); "
              f"{changed:.1%} of rows changed value", flush=True)
        if base_cov > 0 and cov < 0.1 * base_cov:
            print("  *** control is VACUOUS at this shift -- nearly all labels removed, "
                  "so a pass proves nothing", flush=True)
    # A control that changed almost nothing tested almost nothing, and must not be allowed to
    # veto an arm. Excluding a feature on an untested basis is treating absence of evidence
    # as evidence -- which is exactly what barred rel-avito's `+rate` family, on a 0.0053
    # difference produced by a 1-day rewind against a 4-day horizon.
    temporal_informative = moved >= 0.05
    temporal = temporal_control(lambda days: rate_only_score(y, shift=days), shifts=shifts)
    print(f"  {temporal!r}", flush=True)

    # The counts arm needs its own temporal control, and it is the one a permutation test
    # cannot cover: counts do not depend on label *values*, so shuffling leaves them
    # unchanged. Reporting a counts-only result on the strength of a rate-only control --
    # which is what happened on rel-event first time round -- controls nothing.
    def count_only_score(shift=None):
        block = track(test[key].to_numpy(), test[tcol].to_numpy(), y, shift=shift)
        counts = block[[c for c in block.columns if c.endswith("n_prior")]].sum(axis=1)
        return roc_auc_score(y_te, counts.to_numpy())

    print("control 3: temporal, on the COUNT columns", flush=True)
    temporal_counts = temporal_control(lambda days: count_only_score(shift=days),
                                       shifts=shifts)
    print(f"  {temporal_counts!r}", flush=True)

    # And on the structural column, which is the arm validation actually keeps choosing.
    # Controls 1-3 all test columns the `+struct` arm does not contain, so passing them
    # says nothing about it -- reporting a struct-only result on their strength would be
    # the same mistake as reporting a counts-only result on a rate-only control.
    def struct_only_score(shift=None):
        block = track(test[key].to_numpy(), test[tcol].to_numpy(), y, shift=shift)
        deg = block[[c for c in block.columns if c.endswith("n_linked")]].sum(axis=1)
        return roc_auc_score(y_te, deg.to_numpy())

    print("control 4: temporal, on the STRUCTURAL column (n_linked)", flush=True)
    temporal_struct = temporal_control(lambda days: struct_only_score(shift=days),
                                       shifts=shifts)
    print(f"  {temporal_struct!r}", flush=True)
    if not temporal_struct.passed:
        print("  *** n_linked reaches past the cutoff. Where a link table carries no "
              "timestamp this is expected and unfixable -- exclude those keys rather than "
              "reporting the arm.", flush=True)

    # And the question that decides whether the counts are even a label feature: pure
    # structural degree consults no labels at all, so if it scores alike there is no
    # leakage question to answer.
    struct = t_te[struct_cols].sum(axis=1).to_numpy()
    prior = t_te[count_cols].sum(axis=1).to_numpy()
    print(f"  structural degree alone (no labels): "
          f"{roc_auc_score(y_te, struct) * 100:.2f}", flush=True)
    print(f"  resolved-label counts alone:         "
          f"{roc_auc_score(y_te, prior) * 100:.2f}", flush=True)
    # Each arm is gated by the controls that test *its own* columns, so one failing family
    # does not block a clean one -- and, more importantly, a passing family cannot vouch
    # for an arm it never touched.
    # Text arms are gated by the label-derived controls only where they include label
    # features. A pure text embedding derives from entity columns, not from any label, so
    # the permutation and temporal controls have nothing to say about it -- and a control
    # that cannot make a feature's value move cannot clear it either.
    # An indeterminate temporal control does not veto. See `temporal_informative` above:
    # a control whose shift changed under 5% of feature values tested almost nothing, and
    # excluding an arm on that basis treats absence of evidence as evidence.
    temporal_ok = temporal.passed or not temporal_informative
    if not temporal.passed and not temporal_informative:
        print(f"  NOTE: the temporal control returned LEAK but changed only {moved:.1%} of "
              f"feature values, so it is INDETERMINATE and is not excluding anything. Any "
              f"arm below carrying `+rate` is untested on this axis, not cleared by it.",
              flush=True)
    # COVERAGE COLLAPSE. An arm scored on validation rows that have neighbours and then
    # applied to test rows that do not is being judged on a population that will not exist.
    # This reads links and timestamps ONLY -- never a label, never a test outcome -- so it is
    # computable at inference time, and it is a statement about whether the feature EXISTS on
    # test rows rather than about its score. That is what makes it unlike gap-validation,
    # decide-fit-pool, abstain and ensemble-configs, all of which tried to read a better
    # answer out of the same validation scores.
    #
    # Measured across the seven tasks: coverage falls 37.0% -> 17.8% on user-clicks,
    # 43.1% -> 30.8% on user-visits and 96.4% -> 71.8% on driver-top3 -- exactly the three
    # where tuning LOSES -- and holds within a few points on the four where it gains.
    # driver-dnf drops as far as driver-top3 and still gains, because every one of its
    # history arms already fails its controls and there is nothing left to mis-select. Both
    # conditions are needed and both are label-free.
    stale = False
    if args.drop_stale_arms:
        _val = task.get_table("val", mask_input_cols=False).df
        t_va = track(_val[key].to_numpy(), _val[tcol].to_numpy(), y)
        # Coverage means "this row HAS a neighbour", and only the rate columns say that.
        # `n_prior` and `n_linked` are integer counts filled with 0, never NaN, so
        # `.notna().any(axis=1)` over the whole block is True for every row and reported
        # 100% -> 100% on a task measured at 37.0% -> 17.8%. The rule could never fire.
        rate_cols = [c for c in t_te.columns if c.endswith("positive_rate")]
        if not rate_cols:
            raise SystemExit(
                "--drop-stale-arms needs the shared-key rate columns to measure coverage, "
                "and this task's track-record block has none. Without them every row looks "
                "covered and the rule silently never fires."
            )
        v_cov = float(t_va[rate_cols].notna().any(axis=1).mean()) if len(t_va) else 0.0
        t_cov = float(t_te[rate_cols].notna().any(axis=1).mean()) if len(t_te) else 0.0
        ratio = t_cov / v_cov if v_cov > 0 else 1.0
        stale = ratio < args.stale_threshold
        print(f"shared-key coverage: val {v_cov:.1%} -> test {t_cov:.1%} ({ratio:.2f}x)"
              f"{'   ** STALE -- history arms excluded **' if stale else ''}", flush=True)
    ok = {"base": True, "+text": True,
          # NFA aggregates ATTRIBUTES of earlier rows, never a label, so the label controls
          # that gate the history arms do not apply to it -- exactly as for `+text`. Its
          # causality is structural: a row's cutoff is its own timestamp.
          "+nfa": True,
          "+text+rate": perm.passed and temporal_ok and temporal_counts.passed and not stale,
          "+struct": temporal_struct.passed and not stale,
          "+counts": temporal_counts.passed and not stale,
          "+rate": perm.passed and temporal_ok and temporal_counts.passed and not stale,
          "+history": perm.passed and temporal_ok and temporal_counts.passed
                      and temporal_struct.passed and not stale}
    # An INCONCLUSIVE control is not a pass, and admitting an arm on one must be visible.
    # Controls 3 and 4 score a sub-block; where that block lands at or below chance the
    # temporal test cannot speak, and until 2026-08-09 it returned *** LEAK *** instead --
    # excluding five of seven arms on both rel-amazon tasks, and `+struct` on
    # rel-event/user-repeat and rel-trial/study-outcome, our two best results.
    for _lbl, _rep in (("counts", temporal_counts), ("n_linked", temporal_struct)):
        if getattr(_rep, "inconclusive", False):
            print(f"  NOTE: the {_lbl} control is INCONCLUSIVE, not a pass -- its block "
                  f"scores {_rep.observed:.4f}, at or below chance, so the temporal test "
                  f"has no verdict to give. The arms it gates are admitted and validation "
                  f"decides. A large gain from an arm whose own block cannot beat chance "
                  f"would itself be suspicious and should be investigated, not banked.",
                  flush=True)
    print(f"\narm eligibility: {ok}", flush=True)
    if not any(v for k, v in ok.items() if k != "base"):
        print("CONTROLS FAILED for every feature arm -- nothing to measure", flush=True)
        return
    for name, passed in ok.items():
        if not passed:
            print(f"  {name} is EXCLUDED from selection: its own controls failed", flush=True)

    # --- three arms ----------------------------------------------------------------------
    def stack(base, block, cols=None, fit=False, tag=""):
        chosen = block if cols is None else block[cols]
        return _numeric(pd.concat([base.reset_index(drop=True),
                                   chosen.reset_index(drop=True)], axis=1),
                        fit=fit, tag=tag or "stack")

    # `+rate` is the label-history block *without* the structural column: n_prior,
    # n_positive, positive_rate. It is the composition that produced rel-trial's 69.36,
    # before n_linked existed. Keeping it separate matters -- folding n_linked into
    # `+history` meant a control failing on the structural column excluded an arm that
    # never contained it, which is a two-variable comparison wearing a verdict's clothing.
    rate_cols = [c for c in t_tr.columns if not c.endswith("n_linked")]
    # Each arm fits its category map on TRAIN and reuses it for val/test, keyed by arm so
    # two arms with different column sets cannot share a stale map.
    arms = {
        "base": (_numeric(b_tr, fit=True, tag="base"), _numeric(b_te, tag="base")),
        "+struct": (stack(b_tr, t_tr, struct_cols, fit=True, tag="struct"),
                    stack(b_te, t_te, struct_cols, tag="struct")),
        **({"+text": (stack(b_tr, x_tr), stack(b_te, x_te)),
            "+text+rate": (stack(pd.concat([b_tr.reset_index(drop=True),
                                            x_tr.reset_index(drop=True)], axis=1),
                                 t_tr, rate_cols),
                           stack(pd.concat([b_te.reset_index(drop=True),
                                            x_te.reset_index(drop=True)], axis=1),
                                 t_te, rate_cols))} if text_cols else {}),
        **({"+nfa": (stack(b_tr, nfa_tr, fit=True, tag="nfa"),
                     stack(b_te, nfa_te, tag="nfa"))} if nfa_tr is not None else {}),
        "+counts": (stack(b_tr, t_tr, count_cols), stack(b_te, t_te, count_cols)),
        "+rate": (stack(b_tr, t_tr, rate_cols), stack(b_te, t_te, rate_cols)),
        "+history": (stack(b_tr, t_tr), stack(b_te, t_te)),
    }
    # An arm whose own controls failed is not offered to validation at all. Selection
    # cannot be allowed to pick a leaking arm and have the protocol launder it.
    arms = {k: v for k, v in arms.items() if ok.get(k, True)}
    if args.compress_to:
        # Fit on TRAIN, apply to eval. Each arm gets its own projection because the arms
        # carry different columns; sharing one would mean the components describe a feature
        # set the arm does not have.
        before = {k: v[0].shape[1] for k, v in arms.items()}
        arms = {k: (compress_fit_apply(k, v[0], args.compress_to, fit=True),
                    compress_fit_apply(k, v[1], args.compress_to, fit=False))
                for k, v in arms.items()}
        print("compressed: " + ", ".join(
            f"{k} {before[k]}->{v[0].shape[1]}" for k, v in arms.items()), flush=True)
    # Cheap insurance on the runner that produces the standing table. It drops the target
    # correctly today; `eval_depth2` did not, and scored AUC 100.00 in both arms of a
    # paired comparison whose difference read as a clean +0.00. A regression here would be
    # far more expensive, and a control that only runs when someone suspects something is
    # not a control.
    for name, (X_arm, _) in arms.items():
        assert_no_perfect_feature(X_arm, y, context=f"arm {name!r}")
    widths = ", ".join(f"{k} {v[0].shape[1]}" for k, v in arms.items())
    print(f"feature widths: {widths}; {len(arms['base'][0])} train rows, "
          f"context={args.context}", flush=True)
    def score(X, Xe, rows, seed, truth=None, fit_y=None, return_probs=False):
        """Fit and score, optionally averaging predictions over several context draws.

        `RESEARCH` item 10. The largest measured weakness on this branch is not bias but
        variance from *which rows land in the context*: rel-event's calibrated replicates
        span 10.8 points and rel-avito's random arm spanned 9.1. Averaging probabilities
        over independent draws attacks that directly.

        It is not `n_estimators`, which varies the model seed at a **fixed** context and was
        measured at nothing on every task -- the ensembling that matters here is over the
        context, which is the thing that actually moves.
        """
        target_y = y_te if truth is None else truth
        # Labels for the FIT set. Defaults to train's, but --fit-on-train-val
        # passes the concatenated train+val labels alongside a taller X.
        source_y = y if fit_y is None else fit_y
        draws = max(1, args.resample)
        n = len(X)
        probs = None
        for d in range(draws):
            # Draw d=0 is exactly the unresampled behaviour, so --resample 1 reproduces
            # every earlier number and the comparison stays single-variable.
            if d == 0:
                if args.context_select == "knn":
                    take = knn_context_indices(X, Xe, len(rows), seed)
                elif args.context_select == "mmd":
                    take = mmd_context_indices(X, Xe, len(rows), seed)
                else:
                    take = rows
            elif args.stratify_context:
                # Draw each class in proportion to its share of the training set. A uniform
                # draw lets class balance wander between draws, which is noise added to the
                # very quantity resampling exists to average down -- and an accidentally
                # skewed context is not hypothetical here: the graph-context arm once built
                # one at a 0.02 positive rate against a 0.163 base rate.
                r = np.random.default_rng(seed * 1000 + d)
                parts = []
                for cls in np.unique(source_y):
                    pool = np.flatnonzero(source_y == cls)
                    want = int(round(len(rows) * len(pool) / n))
                    parts.append(r.choice(pool, size=min(want, len(pool)), replace=False))
                take = np.concatenate(parts)
            else:
                take = np.random.default_rng(seed * 1000 + d).choice(
                    n, size=len(rows), replace=False)
            if args.backbone == "tabfm":
                from tabfm import TabFMClassifier
                # No random_state and no device: seeds vary the CONTEXT DRAW only, which is
                # the variance this benchmark actually cares about. max_num_rows defaults to
                # 100 and must be raised to the run's own context or the arms are not
                # comparable on data seen.
                clf = TabFMClassifier(model=_tabfm(args.device),
                                      n_estimators=args.n_estimators,
                                      max_num_rows=len(take),
                                      max_num_features=X.shape[1]).fit(X[take], source_y[take])
            else:
                clf = TabICLClassifier(
                    n_estimators=args.n_estimators, device=args.device, random_state=seed,
                    inference_config=inference_config(args.row_chunk, args.offload,
                                                     args.disk_offload_dir)).fit(X[take], source_y[take])
            p = clf.predict_proba(Xe)[:, 1]
            probs = p if probs is None else probs + p
        if return_probs:
            return probs / draws
        return roc_auc_score(target_y, probs / draws) * 100

    if args.calibrated:
        val = task.get_table("val", mask_input_cols=False).df
        y_va = val[target].to_numpy()
        b_va = build_base(val, "val").reindex(columns=b_tr.columns, fill_value=np.nan)
        t_va = track(val[key].to_numpy(), val[tcol].to_numpy(), y)
        x_va = text_block(val)
        val_arms = {k: v for k, v in {
            "base": _numeric(b_va),
            "+text": stack(b_va, x_va),
            "+text+rate": stack(pd.concat([b_va.reset_index(drop=True),
                                           x_va.reset_index(drop=True)], axis=1),
                                t_va, rate_cols),
            "+struct": stack(b_va, t_va, struct_cols),
            **({"+nfa": stack(b_va, nfa_va, tag="nfa")} if nfa_va is not None else {}),
            "+counts": stack(b_va, t_va, count_cols),
            "+rate": stack(b_va, t_va, rate_cols),
            "+history": stack(b_va, t_va),
        }.items() if k in arms}
        if args.compress_to:
            # fit=False: reuse the TRAIN projection. Fitting here would let validation pick
            # its own components, so the arm validation SELECTS on would not be the arm test
            # is judged on -- the exact shape of this project's val/test inversion.
            val_arms = {k: compress_fit_apply(k, v, args.compress_to, fit=False)
                        for k, v in val_arms.items()}
        def cv_score(name, size, seed):
            """Selection criterion from k-fold CV over train, instead of one small split.

            Each fold fits on a subsample of the other folds and scores the held-out one, so
            every training row contributes to the criterion. A 960-row validation split
            cannot resolve a 0.6-point difference; k folds over 12,000 rows can. Test is
            still touched exactly once, after selection -- this changes only *what the
            selection listens to*, not how many times the answer is consulted.
            """
            X, _ = arms[name]
            n = len(X)
            rng = np.random.default_rng(seed)
            if args.cv_time_ordered:
                # Forward chaining: fold k is scored using only rows BEFORE it. Random
                # k-folds break the temporal ordering the benchmark rests on -- a random
                # fold trains on the future to predict the past, which rewarded context
                # resampling with +5.3 of CV while test fell 0.76. A criterion that grows
                # more confident as test degrades is worse than a noisy one.
                order = np.argsort(train[tcol].to_numpy(), kind="stable")
                blocks = np.array_split(order, args.cv_folds + 1)
                folds = blocks[1:]                       # first block is history only
                past = {i: np.concatenate(blocks[:i + 1]) for i in range(len(folds))}
            else:
                order = rng.permutation(n)
                folds = np.array_split(order, args.cv_folds)
                past = None
            scores = []
            for i, f in enumerate(folds):
                if len(np.unique(y[f])) < 2:
                    continue
                rest = past[i] if past is not None else np.setdiff1d(order, f,
                                                                     assume_unique=False)
                if len(rest) < 50:
                    continue
                # Reuse `score`, which already averages over --resample draws. The first
                # version built its own classifier call: it ignored --resample (returning an
                # identical 90.37 for 1 and 3 draws, so the criterion was blind to the very
                # setting it judged) and then threw a torch TypeError the main path never
                # hits. A selection criterion should exercise the scoring path it selects
                # for, not a parallel reimplementation of it.
                take = rng.choice(rest, size=min(size, len(rest)), replace=False)
                scores.append(score(X, X[f], take, seed, truth=y[f]))
            return float(np.mean(scores)) if scores else 0.0

        orders = [o.strip() for o in args.context_orders.split(",") if o.strip()]
        unknown = [o for o in orders if o not in ("random", "recent", "recent-half")]
        if unknown:
            raise SystemExit(f"unknown context order(s) {unknown}")
        # Train row indices in time order. Stable, so ties keep their original order and
        # the choice is reproducible.
        time_order = np.argsort(train[tcol].to_numpy(), kind="stable")

        # A validation split carved from train so that the gap between the pool a setting
        # is fitted on and the rows it is judged on matches the gap between train and test.
        # Both are subsets of the same feature matrix, so this costs no extra build.
        pool_order, pseudo_val = time_order, None
        if args.gap_validation:
            t_train = train[tcol].to_numpy()
            gap = test[tcol].to_numpy().min() - t_train.max()
            cut = t_train[time_order[-len(val):]].min() if len(val) < len(train) else t_train.max()
            pseudo_val = time_order[-len(val):]
            pool_order = np.array([i for i in time_order if t_train[i] <= cut - gap])
            day = np.timedelta64(1, "D")
            print(f"gap-validation: pseudo-val = last {len(pseudo_val)} train rows "
                  f"(from {pd.Timestamp(t_train[pseudo_val].min()).date()}), pool = "
                  f"{len(pool_order)} rows ending {pd.Timestamp(t_train[pool_order].max()).date()} "
                  f"if any, a {gap / day:.0f}-day gap matching train->test", flush=True)
            # The check that matters: the pool must actually end a full gap before the
            # pseudo-validation rows start, or the instrument is not doing the one thing it
            # exists to do.
            if len(pool_order):
                actual = (t_train[pseudo_val].min() - t_train[pool_order].max()) / day
                print(f"  realised gap {actual:.0f} days vs target {gap / day:.0f}",
                      flush=True)
                if actual < gap / day - 1e-6:
                    raise SystemExit(
                        f"pool ends only {actual:.0f} days before the pseudo-validation "
                        f"rows, short of the {gap / day:.0f}-day target -- the split does "
                        f"not reproduce the train->test geometry it exists to reproduce."
                    )
            if len(pool_order) < 200:
                raise SystemExit(
                    f"gap-validation leaves only {len(pool_order)} pool rows on this task; "
                    f"the train->test gap is too large a fraction of the training span for "
                    f"this instrument to exist here."
                )

        def draw(order, seed, n, size, pool=None):
            pool = time_order if pool is None else pool
            size = min(size, len(pool))
            rng = np.random.default_rng(seed)
            if order == "recent":
                # Deterministic given a size: if this helps, it helps without a draw.
                return pool[-size:]
            if order == "recent-half":
                half = pool[len(pool) // 2:]
                return rng.choice(half, size=min(size, len(half)), replace=False)
            return rng.choice(pool, size=size, replace=False)

        # Validation split by TIME, not at random: the whole point is to test whether a
        # ranking survives a gap, and a random half sits at the same temporal distance from
        # train as the half it is judging. The 60/40 cut matches --decide-fit-pool's.
        if args.match_novelty:
            # Label-free: membership in the training entity set, nothing about outcomes.
            seen_tr = set(train[key].unique())
            v_seen = val[key].isin(seen_tr).to_numpy()
            t_rate = float(test[key].isin(seen_tr).mean())
            n_unseen = int((~v_seen).sum())
            # Hold the unseen rows and drop seen ones until the ratio matches test's. Keeping
            # every unseen row loses the least data; the alternative direction would discard
            # the scarce half.
            keep_seen = int(round(n_unseen * t_rate / max(1e-9, 1 - t_rate)))
            idx_seen = np.flatnonzero(v_seen)
            rng_nov = np.random.default_rng(0)
            if keep_seen < len(idx_seen):
                idx_seen = rng_nov.choice(idx_seen, size=keep_seen, replace=False)
            keep = np.sort(np.concatenate([idx_seen, np.flatnonzero(~v_seen)]))
            print(f"match-novelty: validation {v_seen.mean():.1%} seen -> "
                  f"{v_seen[keep].mean():.1%}, matching test's {t_rate:.1%}; "
                  f"{len(keep):,} of {len(val):,} rows kept", flush=True)
            val = val.iloc[keep].reset_index(drop=True)
            y_va = y_va[keep]
            val_arms = {k: v[keep] for k, v in val_arms.items()}
        v_order = np.argsort(val[tcol].to_numpy(), kind="stable")
        va_cut = max(1, int(0.6 * len(v_order)))
        va_early, va_late = v_order[:va_cut], v_order[va_cut:]
        # Abstention needs an ordinary validation column to split. `--cv-folds` scores on
        # train folds and `--gap-validation` on a pseudo-validation set carved out of train,
        # so neither produces the val predictions this reads -- and gap-validation is
        # refuted anyway.
        split_halves = bool(args.abstain) and not args.cv_folds and not args.gap_validation
        if args.abstain and not split_halves:
            raise SystemExit(
                "--abstain needs the ordinary validation column and cannot be combined with "
                "--cv-folds or --gap-validation. Silently ignoring it would report an "
                "abstention run that never abstained."
            )
        if split_halves:
            print(f"\nabstention: validation split by time into {len(va_early)} early / "
                  f"{len(va_late)} late rows", flush=True)

        results = []
        for seed in range(args.seeds):
            best = None
            candidates: list = []
            split_val: dict = {}
            criterion = f"{args.cv_folds}-fold CV over train" if args.cv_folds \
                else "validation only"
            print(f"\n-- seed {seed}: selection ({criterion}) --", flush=True)
            # Capped at --context, not at the training-set size: rel-avito has 86,619 rows
            # and a grid scaled to that asks for contexts an L40S will not fit, so the run
            # dies rather than reporting a smaller honest number.
            cap = min(len(arms["base"][0]), args.context)
            if args.context_grid:
                grid = sorted({min(int(s), cap) for s in args.context_grid.split(",")})
            else:
                grid = sorted({max(1000, cap // 4), max(2000, cap // 2), cap})
                if len(orders) > 1:
                    # Context size was swept for *random* draws and settled there. Its
                    # interaction with a recency rule is a different question, and the
                    # default grid cannot see it: on rel-event, `recent` is +0.31 at
                    # context 10,000 and +7.50 at 1,000, because 10,000 of 19,239 rows is
                    # half the training period and 1,000 is the last few days of it. A
                    # grid that starts at cap//4 would have measured the diluted version
                    # and concluded recency does nothing.
                    grid = sorted(set(grid) | {min(1000, cap), min(2000, cap)})
            for name in arms:
                for size in grid:
                    for order in orders:
                        n = len(arms[name][0])
                        # `recent` with a context covering every training row selects the
                        # same set as `random`, so offering both would put a coin flip in
                        # the selection rather than a choice.
                        if order != "random" and min(size, n) >= n:
                            continue
                        rows = draw(order, seed, n, size, pool_order)
                        if args.cv_folds:
                            v = cv_score(name, size, seed)
                        elif pseudo_val is not None:
                            # Judged on the latest train rows, having been fitted only on
                            # rows a full train->test gap earlier. Same matrix, so the
                            # features are identical to the ones test will see.
                            v = score(arms[name][0], arms[name][0][pseudo_val], rows, seed,
                                      truth=y[pseudo_val])
                        elif split_halves:
                            # ONE fit, then three AUCs off the same predictions: the whole
                            # validation set and each time-ordered half. Taking the probs
                            # and deriving `v` from them is what makes --abstain free; a
                            # second `score()` call here would refit the model and double
                            # the grid's cost, which is what it did when first written.
                            p_va = score(arms[name][0], val_arms[name], rows, seed,
                                         truth=y_va, return_probs=True)
                            v = roc_auc_score(y_va, p_va) * 100
                            split_val[(name, size, order)] = tuple(
                                roc_auc_score(y_va[idx], p_va[idx]) * 100
                                if len(np.unique(y_va[idx])) > 1 else float("nan")
                                for idx in (va_early, va_late))
                        else:
                            v = score(arms[name][0], val_arms[name], rows, seed, truth=y_va)
                        label = f"{name}/{order}" if len(orders) > 1 else name
                        print(f"  {label:<20} context={size:<6} "
                              f"{'cv' if args.cv_folds else 'val'}={v:.2f}", flush=True)
                        candidates.append((v, name, size, order))
                        if best is None or v > best[0]:
                            best = (v, name, size, order)
            val_auc, name, size, order = best

            if split_halves and split_val:
                # Pick on the EARLY half, then ask whether that pick survives on the LATE
                # half against the untuned arm. A winner that cannot beat `base` out of
                # sample is a winner the ranking invented, and taking it is how user-clicks
                # loses 1.29. Note what is NOT being asked: not "which arm is best" -- every
                # instrument built on that question has failed here -- but "does this
                # ranking generalise across a time gap at all".
                chosen_k, win_k, abstained = abstention_choice(split_val)
                if chosen_k is not None:
                    we, wl = split_val[win_k]
                    ce, cl = split_val[chosen_k]
                    if abstained:
                        print(f"  ABSTAIN: {win_k[0]} won the early half ({we:.2f} vs base "
                              f"{ce:.2f}) but lost the late half ({wl:.2f} vs {cl:.2f}) -- "
                              f"falling back to base", flush=True)
                    else:
                        print(f"  ranking holds across the val gap ({win_k[0]}: {we:.2f} "
                              f"early, {wl:.2f} late) -- tuning kept", flush=True)
                    # ONLY the abstention changes the pick. When the ranking holds, the
                    # ordinary full-validation argmax stands untouched -- so this flag is a
                    # pure veto, which is what its help text promises.
                    #
                    # It did not start that way, and the difference was visible: with the
                    # veto firing ZERO times, the arms still differed, because this branch
                    # was silently replacing the pick with the winner on the early 60% of
                    # validation rather than the argmax over all of it. That made the A/B
                    # two variables instead of one.
                    #
                    # Incidentally it priced the second, and the price is NOT negligible --
                    # selecting on 60% of validation instead of 100%, with the veto never
                    # firing, cost +0.15 / 0.00 / -0.11 / **-0.70** on user-clicks,
                    # user-visits, rel-trial and user-ignore: **mean -0.17, worst -0.70**.
                    # An earlier note here called it "about nothing, mean +0.01" from the
                    # first three; user-ignore changed that. Any future instrument that
                    # spends part of validation on a meta-decision is paying this, and on
                    # rel-event it exceeds the +-0.6 floor.
                    if abstained:
                        name, size, order = chosen_k
                        val_auc = next(c[0] for c in candidates
                                       if (c[1], c[2], c[3]) == chosen_k)

            if args.ensemble_configs > 1:
                # Hedge instead of committing. The validation ranking is real but noisy --
                # on rel-event its top choice was 3.9 below the best configuration
                # available -- so averaging over the top of it is strictly less exposed to
                # that noise than taking its first element, and it needs no instrument the
                # argmax does not already need.
                top = sorted(candidates, key=lambda c: -c[0])[: args.ensemble_configs]
                probs = None
                for _, nm, sz, od in top:
                    rows_i = draw(od, seed, len(arms[nm][0]), sz)
                    p_i = score(arms[nm][0], arms[nm][1], rows_i, seed, return_probs=True)
                    probs = p_i if probs is None else probs + p_i
                auc = roc_auc_score(y_te, probs / len(top)) * 100
                picks = ", ".join(f"{nm}@{sz}/{od}" for _, nm, sz, od in top)
                print(f"  ensembled {len(top)} of {len(candidates)}: {picks} "
                      f"-> VAL(best) {val_auc:.2f}  TEST {auc:.2f}", flush=True)
                results.append((auc, name, size, val_auc, order))
                continue

            use_pool = args.fit_on_train_val
            if args.decide_fit_pool:
                # Decide the pool question on VALIDATION, never on test. Split val by time:
                # the earlier part joins the fitting pool, the later part judges whether
                # that helped. Both halves are validation rows, so this is a
                # validation-only decision procedure.
                #
                # It exists because the ordinary validation column is identical whether or
                # not val joins the final fit -- the selection is unchanged, so the usual
                # instrument is blind to a difference worth +2.48 on rel-f1 and -7.29 on
                # rel-event.
                v_order = np.argsort(val[tcol].to_numpy(), kind="stable")
                cut = max(1, int(0.6 * len(v_order)))
                add_idx, judge_idx = v_order[:cut], v_order[cut:]
                judge_X, judge_y = val_arms[name][judge_idx], y_va[judge_idx]
                n_tr = len(arms[name][0])

                plain_rows = draw(order, seed, n_tr, size)
                plain = score(arms[name][0], judge_X, plain_rows, seed, truth=judge_y)

                aug_X = np.vstack([arms[name][0], val_arms[name][add_idx]])
                aug_y = np.concatenate([y, y_va[add_idx]])
                aug_take = len(aug_X) if size >= n_tr else min(size, len(aug_X))
                aug_rows = np.random.default_rng(seed).choice(
                    len(aug_X), size=aug_take, replace=False)
                augmented = score(aug_X, judge_X, aug_rows, seed,
                                  truth=judge_y, fit_y=aug_y)

                use_pool = augmented > plain
                print(f"  pool decision on held-out val: train {plain:.2f} vs "
                      f"train+val {augmented:.2f} -> "
                      f"{'train+val' if use_pool else 'train only'}", flush=True)

            if use_pool:
                # Refit on everything available once the setting is chosen -- the same
                # pattern as refitting on all data after cross-validation. The validation
                # split was used to *select*, and RelBench permits training on train and
                # val; only test is held out. No test information is involved.
                #
                # This is the one lever today that does not depend on the selection
                # instrument, and the extra data is not marginal: +43% of rows on rel-f1
                # and +35% on rel-avito.
                pool_X = np.vstack([arms[name][0], val_arms[name]])
                pool_y = np.concatenate([y, y_va])
                order_pool = np.argsort(
                    np.concatenate([train[tcol].to_numpy(), val[tcol].to_numpy()]),
                    kind="stable")
                m = len(pool_X)
                # If validation asked for the whole training set, it was saying "use every
                # row available" -- and at fit time that is train+val. Capping at
                # len(train) would use 1,353 of rel-f1's 1,941 rows and throw away the
                # extra data this flag exists to add.
                #
                # Only when the grid's top really was the whole set. On rel-trial the grid
                # tops out at 10,000 of 11,994, so choosing 10,000 is a genuine preference
                # for a bounded context and is left alone.
                take = m if size >= len(arms[name][0]) else min(size, m)
                rng = np.random.default_rng(seed)
                if order == "recent":
                    rows = order_pool[-take:]
                elif order == "recent-half":
                    half = order_pool[len(order_pool) // 2:]
                    rows = rng.choice(half, size=min(take, len(half)), replace=False)
                else:
                    rows = rng.choice(m, size=take, replace=False)
                auc = score(pool_X, arms[name][1], rows, seed, fit_y=pool_y)
            else:
                n = len(arms[name][0])
                rows = draw(order, seed, n, size)
                auc = score(arms[name][0], arms[name][1], rows, seed)
            results.append((auc, name, size, val_auc, order))
            print(f"  chosen {name} context={size} order={order} -> "
                  f"VAL {val_auc:.2f}  TEST {auc:.2f}", flush=True)

        aucs = np.array([r[0] for r in results])
        vals = np.array([r[3] for r in results])
        chose = [r[1] for r in results]
        # (The old warning here said a counts-only choice was uncontrolled. That was true
        # before controls 3 and 4 existed; now every arm is gated by a control on its own
        # columns and an ineligible arm is never offered to validation, so the warning
        # fired on results that *were* controlled. A warning that cries wolf gets ignored.)
        print(f"\n{args.dataset}/{args.task}  CALIBRATED TEST ROC-AUC x100 = {aucs.mean():.2f} "
              f"+- {aucs.std(ddof=1) if len(aucs) > 1 else 0:.2f} over {len(aucs)} "
              f"replicates (range {aucs.min():.2f}-{aucs.max():.2f})", flush=True)
        # PER-SEED, on one machine-readable line, because every A/B here is a PAIRED
        # comparison and the summary sd above is the wrong denominator for it -- that
        # mistake once turned a 4.7-SE result into an undecided 1.6. Until now the per-seed
        # values had to be scraped back out of the log by hand over ssh, which is how a
        # 10-of-12 pairing nearly got reported as the 12-seed answer. One line, and the step
        # where the arithmetic goes wrong disappears.
        print("PERSEED	" + "	".join(f"{v:.4f}" for v in aucs), flush=True)
        # Machine-readable line so a table can be assembled across tasks without re-running.
        print(f"TABLEROW\t{args.dataset}/{args.task}\t{len(train)}\t{len(val)}\t{len(test)}"
              f"\t{vals.mean():.2f}\t{vals.std(ddof=1) if len(vals) > 1 else 0:.2f}"
              f"\t{aucs.mean():.2f}\t{aucs.std(ddof=1) if len(aucs) > 1 else 0:.2f}"
              f"\t{max(set(chose), key=chose.count)}", flush=True)
        print(f"validation chose: {chose}; sizes {[r[2] for r in results]}", flush=True)
        if len(orders) > 1:
            picked = [r[4] for r in results]
            print(f"context order chosen: {picked} "
                  f"({picked.count('random')}/{len(picked)} random)", flush=True)
        print(f"reference: {REFERENCE.get((args.dataset, args.task), 'see PERFORMANCE.md')}", flush=True)
        # Say out loud whether this run is even comparable to that reference. A promotion
        # measured at a different configuration from the number it is beating is half a
        # measurement, and three runs in one afternoon were exactly that.
        expected = STANDING_FLAGS.get(args.dataset, [])

        def satisfied(flag: str) -> bool:
            """Is this run actually configured the way the standing number was?

            Compares the *value* for --max-columns rather than merely noticing the flag is
            present: the earlier version treated any `none` as satisfying the requirement
            and could not tell 2 from 4 at all, which is precisely the distinction that
            matters now that the default moved.
            """
            name, _, want = flag.partition(" ")
            if name == "--timed-links-only":
                return args.timed_links_only
            if name == "--max-columns":
                norm = lambda v: "none" if str(v).lower() in ("none", "null", "") else str(v)
                return norm(args.max_columns) == norm(want)
            return True

        missing = [f for f in expected if not satisfied(f)]
        if missing:
            print(f"WARNING: the standing {args.dataset} number was measured with "
                  f"{' '.join(expected)} and this run was not. The comparison above is "
                  f"between different configurations, not different methods.", flush=True)
        elif expected:
            print(f"configuration matches the standing number ({' '.join(expected)})",
                  flush=True)
        return

    header = "".join(f"{name:>10}" for name in arms)
    print(f"\n{'seed':>5}{header}", flush=True)
    results = {k: [] for k in arms}
    for seed in range(args.seeds):
        rng = np.random.default_rng(seed)
        n = len(arms["base"][0])
        rows = rng.choice(n, size=min(args.context, n), replace=False)
        t0 = time.perf_counter()
        for name, (X, Xe) in arms.items():
            results[name].append(score(X, Xe, rows, seed))
        cells = "".join(f"{results[name][-1]:>10.2f}" for name in arms)
        print(f"{seed:>5}{cells}   ({time.perf_counter() - t0:.0f}s)", flush=True)

    # Every contrast that exists, paired by seed. Naming both sides keeps a reader from
    # attributing a gap to the wrong difference, which is this project's recurring error.
    pairs = [("+history", "base"), ("+history", "+counts"), ("+counts", "+struct"),
             ("+struct", "base"), ("+history", "+struct")]
    for hi, lo in pairs:
        if hi not in results or lo not in results:
            continue
        g = np.array(results[hi]) - np.array(results[lo])
        print(f"{hi} over {lo}: mean {g.mean():+.2f} sd "
              f"{g.std(ddof=1) if len(g) > 1 else 0:.2f} over {len(g)} seeds, "
              f"{(g > 0).sum()}/{len(g)} positive", flush=True)
    # PER ARM, machine-readable, one line each. These are the numbers a cross-run A/B should
    # be computed from -- NOT the calibrated block's. Measured on this benchmark: two runs of
    # a FIXED configuration correlate at r = 0.88-0.94 across seeds, so pairing cuts the
    # standard error by 2.4-3.0x; two CALIBRATED runs correlate at r = 0.34 and -0.03,
    # because the selection step picks a different configuration per seed in each arm and
    # "seed i" is no longer the same experiment. Pairing through the selection step buys
    # nothing, which is why every effect in this project has been so hard to resolve.
    for name, vals in results.items():
        print(f"PERSEED_ARM	{name}	" + "	".join(f"{v:.4f}" for v in vals), flush=True)
    means = ", ".join(f"{k} {np.mean(v):.2f}" for k, v in results.items())
    print(f"means: {means}  ({REFERENCE.get((args.dataset, args.task), 'see PERFORMANCE.md')})",
          flush=True)
    print("NOTE: the +-0.6 floor applies.", flush=True)


if __name__ == "__main__":
    main()
