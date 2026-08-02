"""Small benchmark for the scaling upgrades, sized for a 12 GB GPU.

Run::

    python -m tabicl.scaling.bench --all
    python -m tabicl.scaling.bench --rowchunk --max-rows 60000

Every figure printed here is measured, not projected. Peak memory comes from
``torch.cuda.max_memory_allocated`` around the forward pass.
"""

from __future__ import annotations

import argparse
import time
from contextlib import contextmanager

import numpy as np
import torch

from .._sklearn.classifier import TabICLClassifier
from ._mqa import collapse_kv_heads, kv_cache_bytes
from ._relational import Table, flatten_relational
from ._rowchunk import chunked_set_transformer
from ._ttc import think_predict_proba

MB = 1024**2


@contextmanager
def measure(device: str):
    """Yield a dict that receives peak MiB and elapsed seconds."""
    stats = {}
    cuda = device.startswith("cuda") and torch.cuda.is_available()
    if cuda:
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        base = torch.cuda.max_memory_allocated()
    t0 = time.perf_counter()
    try:
        yield stats
    finally:
        if cuda:
            torch.cuda.synchronize()
            stats["peak_mib"] = (torch.cuda.max_memory_allocated() - base) / MB
        else:
            stats["peak_mib"] = float("nan")
        stats["seconds"] = time.perf_counter() - t0


def _backbone(device: str):
    """Fit a tiny model just to obtain a loaded backbone."""
    rng = np.random.RandomState(0)
    clf = TabICLClassifier(n_estimators=1, device=device, random_state=0)
    clf.fit(rng.rand(64, 5).astype(np.float32), rng.randint(0, 2, 64))
    return clf


def bench_rowchunk(
    device: str, max_rows: int, n_features: int, chunk_size: int, min_rows: int = 4000
) -> None:
    """Peak memory of the column-embedding stage, unchunked vs row-chunked.

    This isolates the ``(N, C, d)`` activation that TabPFN-3 Section 2.4.1 targets.
    """
    print("\n=== 1. Row-chunked column embedding ===")
    print(f"device={device}  n_features={n_features}  chunk_size={chunk_size}\n")
    clf = _backbone(device)
    tf_col = clf.model_.col_embedder.tf_col
    d_model = clf.model_.embed_dim

    header = f"{'rows':>9} {'unchunked MiB':>15} {'chunked MiB':>13} {'saving':>8} {'unchunk s':>10} {'chunk s':>9} {'max|diff|':>11}"
    print(header)
    print("-" * len(header))

    rows = min_rows
    while rows <= max_rows:
        src = torch.randn(1, n_features, rows, d_model, device=device)
        train_size = rows // 2
        ref = None
        with torch.no_grad():
            try:
                with measure(device) as m_ref:
                    ref = tf_col(src, train_size=train_size)
                unchunked = f"{m_ref['peak_mib']:15.1f}"
                t_unchunked = f"{m_ref['seconds']:10.2f}"
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                unchunked, t_unchunked = f"{'OOM':>15}", f"{'-':>10}"

            # inplace consumes its input, so give it a private copy. The clone is
            # allocated outside `measure`, so it raises the baseline rather than the
            # reported delta.
            src_c = src.clone()
            with measure(device) as m_chunk:
                got = chunked_set_transformer(
                    tf_col, src_c, train_size, chunk_size=chunk_size, inplace=True
                )

        if ref is not None:
            diff = f"{(got - ref).abs().max().item():11.2e}"
            saving = f"{m_ref['peak_mib'] / max(m_chunk['peak_mib'], 1e-9):7.2f}x"
        else:
            diff, saving = f"{'n/a':>11}", f"{'n/a':>8}"

        print(
            f"{rows:9d} {unchunked} {m_chunk['peak_mib']:13.1f} {saving:>8} "
            f"{t_unchunked} {m_chunk['seconds']:9.2f} {diff}"
        )
        del src, src_c, ref, got
        if device.startswith("cuda"):
            torch.cuda.empty_cache()
        rows *= 2


def bench_offload(device: str, n_train: int, n_test: int, n_features: int, chunk_size: int) -> None:
    """Row chunking vs TabICLv2's existing offload path, through the real predict path.

    This is the comparison that matters. Beating the naive unchunked path proves little,
    because nobody runs that at scale -- TabICLv2 already offloads. TabPFN-3's actual
    criticism (Section 2.4.1) is that offloading costs ~250 GB host RAM at 1M x 500, or a
    ~4x slowdown. Chunking is only worth having if it beats *that*.

    The two are independent mechanisms -- offload moves outputs off the GPU, chunking
    shrinks activations -- so the combination is measured too.
    """
    print("\n=== 1b. Row chunking vs offload ===")
    print(f"n_train={n_train}  n_test={n_test}  n_features={n_features}  chunk_size={chunk_size}\n")

    rng = np.random.RandomState(0)
    X = rng.rand(n_train + n_test, n_features).astype(np.float32)
    w = rng.randn(n_features)
    y = ((X @ w + 0.3 * rng.randn(len(X))) > np.median(X @ w)).astype(int)
    X_tr, y_tr, X_te = X[:n_train], y[:n_train], X[n_train:]

    variants = [
        ("naive (gpu, no chunk)", "gpu", False),
        ("offload=cpu", "cpu", False),
        ("row_chunk", "gpu", True),
        ("offload=cpu + row_chunk", "cpu", True),
    ]

    header = f"{'variant':<26}{'peak MiB':>10}{'seconds':>9}   {'agrees with first':>17}"
    print(header)
    print("-" * len(header))

    reference = None
    for label, offload, chunked in variants:
        cfg = {
            "COL_CONFIG": {
                "offload": offload,
                "row_chunk": chunked,
                "row_chunk_size": chunk_size,
                "col_chunk_size": 32,
            }
        }
        clf = TabICLClassifier(n_estimators=1, device=device, random_state=0, inference_config=cfg)
        clf.fit(X_tr, y_tr)
        try:
            with measure(device) as m:
                proba = clf.predict_proba(X_te)
            if reference is None:
                reference, agree = proba, "(reference)"
            else:
                agree = f"{np.abs(proba - reference).max():.2e}"
            print(f"{label:<26}{m['peak_mib']:>10.1f}{m['seconds']:>9.2f}   {agree:>17}")
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            print(f"{label:<26}{'OOM':>10}{'-':>9}   {'-':>17}")
        del clf
        if device.startswith("cuda"):
            torch.cuda.empty_cache()


def bench_mqa(device: str, n_train: int, n_features: int) -> None:
    """KV-cache size and accuracy cost of collapsing to a single KV head."""
    print("\n=== 2. Multi-query KV cache ===")
    rng = np.random.RandomState(0)
    X = rng.rand(n_train + 2000, n_features).astype(np.float32)
    w = rng.randn(n_features)
    y = ((X @ w + 0.3 * rng.randn(len(X))) > np.median(X @ w)).astype(int)
    X_tr, y_tr, X_te, y_te = X[:n_train], y[:n_train], X[n_train:], y[n_train:]

    from sklearn.metrics import roc_auc_score

    clf = TabICLClassifier(n_estimators=1, device=device, random_state=0, kv_cache=True)
    clf.fit(X_tr, y_tr)
    auc_full = roc_auc_score(y_te, clf.predict_proba(X_te)[:, 1])

    caches = getattr(clf, "model_kv_cache_", None)
    if not caches:
        print("  kv_cache unavailable on this estimator; skipping")
        return

    total_full = total_mqa = 0
    for cache in (caches.values() if isinstance(caches, dict) else caches):
        for sub in ("col_cache", "icl_cache"):
            c = getattr(cache, sub, None)
            if c is None:
                continue
            total_full += kv_cache_bytes(c)
            total_mqa += kv_cache_bytes(collapse_kv_heads(c))

    print(f"  n_train={n_train}  n_features={n_features}  nhead={clf.model_.icl_nhead}")
    print(f"  cache full multi-head : {total_full / MB:9.1f} MiB")
    print(f"  cache single KV head  : {total_mqa / MB:9.1f} MiB  ({total_full / max(total_mqa,1):.1f}x smaller)")
    print(f"  per-row full          : {total_full / max(n_train,1):9.1f} B/row")
    print(f"  per-row MQA           : {total_mqa / max(n_train,1):9.1f} B/row")
    # Actually run inference off the collapsed cache, so the accuracy cost of the
    # post-hoc approximation is measured rather than asserted. `expand_kv_heads`
    # broadcasts the stored single head back for the stock attention kernel; the
    # saving is in what is *stored*.
    from ._mqa import expand_kv_heads

    nhead = clf.model_.icl_nhead
    cache_iter = caches.values() if isinstance(caches, dict) else caches
    for cache in cache_iter:
        for sub in ("col_cache", "icl_cache"):
            c = getattr(cache, sub, None)
            if c is None:
                continue
            collapsed = expand_kv_heads(collapse_kv_heads(c, mode="mean"), nhead)
            c.kv.update(collapsed.kv)

    try:
        auc_mqa = roc_auc_score(y_te, clf.predict_proba(X_te)[:, 1])
        delta = f"{auc_mqa - auc_full:+.4f}"
    except Exception as exc:  # shape/kernel mismatch is itself a finding
        auc_mqa, delta = float("nan"), f"failed: {type(exc).__name__}"

    print(f"  ROC-AUC full cache    : {auc_full:.4f}")
    print(f"  ROC-AUC collapsed     : {auc_mqa:.4f}  (delta {delta})")
    print("  NOTE: a faithful port needs pretraining with a single KV head;")
    print("        the delta above is the cost of collapsing a checkpoint that was")
    print("        trained with 8 (see DESIGN.md).")


def bench_relational(device: str) -> None:
    """Flattened relational features vs. the entity table alone."""
    print("\n=== 3. Relational flattening ===")
    import pandas as pd
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import train_test_split

    rng = np.random.default_rng(0)
    n_users = 1200
    users = pd.DataFrame(
        {
            "user_id": np.arange(n_users),
            "age": rng.integers(18, 80, n_users),
            "region": rng.integers(0, 5, n_users),
            "predict_at": pd.Timestamp("2026-01-01") + pd.to_timedelta(rng.integers(200, 400, n_users), unit="D"),
        }
    )
    # Signal lives only in the child table, so the baseline should be near chance.
    propensity = rng.random(n_users)
    rows = []
    for uid, p in enumerate(propensity):
        for _ in range(rng.integers(1, 25)):
            rows.append(
                {
                    "user_id": uid,
                    "ts": pd.Timestamp("2026-01-01") + pd.to_timedelta(int(rng.integers(0, 500)), unit="D"),
                    "amount": float(rng.gamma(2.0, 10.0 * (0.5 + p))),
                    "category": int(rng.integers(0, 4)),
                }
            )
    txns = pd.DataFrame(rows)
    y = (propensity > 0.5).astype(int)

    flat = flatten_relational(
        users, "user_id", [Table(txns, "user_id", "txn", time_column="ts")], cutoff_column="predict_at"
    )
    baseline = users.drop(columns=["user_id", "predict_at"])

    for label, frame in (("entity table only", baseline), ("flattened relational", flat)):
        Xtr, Xte, ytr, yte = train_test_split(
            frame.to_numpy(dtype=np.float64), y, test_size=0.3, random_state=0, stratify=y
        )
        clf = TabICLClassifier(n_estimators=2, device=device, random_state=0)
        clf.fit(np.nan_to_num(Xtr), ytr)
        auc = roc_auc_score(yte, clf.predict_proba(np.nan_to_num(Xte))[:, 1])
        print(f"  {label:<22} n_features={frame.shape[1]:>3}  ROC-AUC={auc:.4f}")


def bench_ttc(device: str) -> None:
    """Accuracy gain from extra test-time compute, against the added seconds."""
    print("\n=== 4. Test-time compute ===")
    from sklearn.datasets import make_classification
    from sklearn.metrics import log_loss, roc_auc_score
    from sklearn.model_selection import train_test_split

    X, y = make_classification(
        n_samples=1400, n_features=20, n_informative=8, class_sep=0.7, flip_y=0.08, random_state=0
    )
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.35, random_state=0, stratify=y)
    est = TabICLClassifier(n_estimators=2, device=device, random_state=0)

    with measure(device) as m0:
        base = TabICLClassifier(n_estimators=2, device=device, random_state=0)
        base.fit(X_tr, y_tr)
        p_base = base.predict_proba(X_te)
    print(
        f"  base            : logloss={log_loss(y_te, p_base):.4f}  "
        f"AUC={roc_auc_score(y_te, p_base[:,1]):.4f}  {m0['seconds']:.1f}s"
    )

    for n_perm in (2, 4):
        with measure(device) as m1:
            res = think_predict_proba(est, X_tr, y_tr, X_te, n_permutations=n_perm, random_state=0)
        print(
            f"  thinking (n={n_perm}) : logloss={log_loss(y_te, res.proba):.4f}  "
            f"AUC={roc_auc_score(y_te, res.proba[:,1]):.4f}  {m1['seconds']:.1f}s  "
            f"blend_w={res.blend_weight:.2f}"
        )


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--all", action="store_true")
    p.add_argument("--rowchunk", action="store_true")
    p.add_argument("--offload", action="store_true")
    p.add_argument("--mqa", action="store_true")
    p.add_argument("--relational", action="store_true")
    p.add_argument("--ttc", action="store_true")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--max-rows", type=int, default=64000)
    p.add_argument("--min-rows", type=int, default=4000)
    p.add_argument("--n-features", type=int, default=100)
    p.add_argument("--chunk-size", type=int, default=8192)
    p.add_argument("--mqa-train", type=int, default=8000)
    p.add_argument("--offload-train", type=int, default=40000)
    p.add_argument("--offload-test", type=int, default=4000)
    args = p.parse_args()

    if args.device.startswith("cuda") and torch.cuda.is_available():
        name = torch.cuda.get_device_name(0)
        total = torch.cuda.get_device_properties(0).total_memory / MB
        print(f"GPU: {name}  ({total:.0f} MiB)")

    run_all = args.all or not any((args.rowchunk, args.offload, args.mqa, args.relational, args.ttc))
    if run_all or args.rowchunk:
        bench_rowchunk(args.device, args.max_rows, args.n_features, args.chunk_size, args.min_rows)
    if run_all or args.offload:
        bench_offload(
            args.device, args.offload_train, args.offload_test, args.n_features, args.chunk_size
        )
    if run_all or args.mqa:
        bench_mqa(args.device, args.mqa_train, 20)
    if run_all or args.relational:
        bench_relational(args.device)
    if run_all or args.ttc:
        bench_ttc(args.device)


if __name__ == "__main__":
    main()
