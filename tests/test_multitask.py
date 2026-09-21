"""Tests for TP-07: one checkpoint serving both classification and regression.

Whether joint training *helps* needs a GPU run against a single-task control. What these
tests pin down is the part that can be wrong for free:

- a multitask model loaded with a single-task checkpoint reproduces that model exactly, in
  every forward path (train, inference, KV cache, repr cache), for both tasks -- so the
  shared trunk and per-task heads are wired the way the report describes and nothing else
  changed;
- the task embedding reaches test rows identically through the cached and uncached paths,
  the same boundary TP-04 and TP-05 had to respect;
- each task's loss trains the shared trunk and its own head, and never the other head;
- a single-task model and checkpoint are untouched.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from tabicl._model.tabicl import TabICL

TINY = dict(
    embed_dim=16,
    col_num_blocks=1,
    col_nhead=2,
    col_num_inds=4,
    row_num_blocks=1,
    row_nhead=2,
    row_num_cls=2,
    icl_num_blocks=2,
    icl_nhead=2,
    num_quantiles=15,
    # With the zero_init default every residual branch starts as identity and the features
    # never reach the CLS tokens, so the column-level task embedding could not affect the
    # output and several tests here would pass vacuously. The v2 recipe trains this way.
    zero_init=False,
)

N_TRAIN, N_TEST, N_FEAT = 24, 6, 5


def _clf_model(**kw):
    return TabICL(max_classes=10, **TINY, **kw)


def _reg_model(**kw):
    return TabICL(max_classes=0, **TINY, **kw)


def _joint_model(**kw):
    return TabICL(max_classes=10, multitask=True, **TINY, **kw)


def _data(task, seed=0):
    g = torch.Generator().manual_seed(seed)
    X = torch.randn(2, N_TRAIN + N_TEST, N_FEAT, generator=g)
    if task == "classification":
        # Every table must see the same classes for batched inference
        y = torch.arange(N_TRAIN).remainder(3).repeat(2, 1).float()
    else:
        y = torch.randn(2, N_TRAIN, generator=g)
    return X, y


def _randomize_task_embeddings(model):
    with torch.no_grad():
        model.col_embedder.task_embed.normal_(0, 0.5)
        model.icl_predictor.task_embed.normal_(0, 0.5)


def _all_outputs(model, X, y):
    """Every forward path the model has, keyed by name."""

    out = {}
    model.train()
    torch.manual_seed(0)
    out["train"] = model(X, y)
    model.eval()
    with torch.no_grad():
        out["inference"] = model(X, y)
        X_train, X_test = X[:, :N_TRAIN], X[:, N_TRAIN:]
        for mode in ("kv", "repr"):
            out[f"{mode}_store"] = model.forward_with_cache(
                X_train=X_train, y_train=y, X_test=X_test, store_cache=True, cache_mode=mode
            )
            out[f"{mode}_use"] = model.forward_with_cache(X_test=X_test, use_cache=True, store_cache=False)
            model.clear_cache()
    return out


def _assert_same_outputs(a, b):
    assert a.keys() == b.keys()
    for name in a:
        torch.testing.assert_close(a[name], b[name], rtol=1e-5, atol=1e-5, msg=f"path {name!r} differs")


# --------------------------------------------------------------------------- #
# single-task models are untouched
# --------------------------------------------------------------------------- #


def test_off_by_default():
    model = _clf_model()
    assert not model.multitask
    assert model.task == "classification"
    assert _reg_model().task == "regression"


@pytest.mark.parametrize("make", [_clf_model, _reg_model])
def test_single_task_state_dict_keys_are_unchanged(make):
    """Released checkpoints must keep loading with strict=True."""

    keys = make().state_dict().keys()
    assert not any("task_embed" in k or ".reg_" in k for k in keys)


def test_single_task_model_rejects_the_other_task():
    model = _clf_model()
    model.set_task("classification")  # its own task is fine
    with pytest.raises(ValueError, match="single-task classification"):
        model.set_task("regression")


def test_unknown_task_is_rejected():
    with pytest.raises(ValueError, match="task must be one of"):
        _joint_model().set_task("ranking")


# --------------------------------------------------------------------------- #
# construction
# --------------------------------------------------------------------------- #


def test_multitask_needs_both_heads_sized():
    with pytest.raises(ValueError, match="max_classes > 0"):
        TabICL(max_classes=0, multitask=True, **TINY)
    with pytest.raises(ValueError, match="num_quantiles > 0"):
        TabICL(max_classes=10, multitask=True, **{**TINY, "num_quantiles": 0})


def test_only_label_encoders_heads_and_task_embeddings_are_added():
    """The report's split: everything between label encoder and head is shared."""

    single = set(_clf_model().state_dict())
    added = set(_joint_model().state_dict()) - single
    assert added, "multitask added nothing"
    allowed = ("col_embedder.reg_y_encoder.", "icl_predictor.reg_y_encoder.", "icl_predictor.reg_decoder.",
               "col_embedder.task_embed", "icl_predictor.task_embed")
    assert all(k.startswith(allowed) for k in added), sorted(added)
    assert single <= set(_joint_model().state_dict())


def test_task_embeddings_start_at_zero():
    model = _joint_model()
    assert torch.count_nonzero(model.col_embedder.task_embed) == 0
    assert torch.count_nonzero(model.icl_predictor.task_embed) == 0


def test_refuses_to_run_without_a_task():
    """A forgotten set_task must fail loudly, not quietly use the classification head."""

    model = _joint_model()
    X, y = _data("classification")
    with pytest.raises(RuntimeError, match="set_task"):
        model(X, y)
    model.eval()
    with pytest.raises(RuntimeError, match="set_task"):
        model.forward_with_cache(X_train=X[:, :N_TRAIN], y_train=y, store_cache=True)


@pytest.mark.parametrize("task,width", [("classification", 3), ("regression", 15)])
def test_output_width_follows_the_task(task, width):
    model = _joint_model().set_task(task)
    X, y = _data(task)
    model.train()
    assert model(X, y).shape[-1] == (10 if task == "classification" else 15)
    model.eval()
    with torch.no_grad():
        assert model(X, y).shape == (2, N_TEST, width)


# --------------------------------------------------------------------------- #
# the core claim: a joint model can be exactly a single-task model
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("task", ["classification", "regression"])
def test_reproduces_the_single_task_model_on_every_path(task):
    torch.manual_seed(0)
    single = _clf_model() if task == "classification" else _reg_model()
    joint = _joint_model()
    joint.load_single_task_state_dict(single.state_dict(), task)
    joint.set_task(task)

    X, y = _data(task)
    _assert_same_outputs(_all_outputs(single, X, y), _all_outputs(joint, X, y))


def test_regression_statistics_match_the_single_task_model():
    torch.manual_seed(0)
    single = _reg_model().eval()
    joint = _joint_model()
    joint.load_single_task_state_dict(single.state_dict(), "regression")
    joint.set_task("regression").eval()

    X, y = _data("regression")
    with torch.no_grad():
        a = single.predict_stats(X, y, output_type=["mean", "median", "quantiles"])
        b = joint.predict_stats(X, y, output_type=["mean", "median", "quantiles"])
    for k in a:
        torch.testing.assert_close(a[k], b[k], rtol=1e-5, atol=1e-5)


def test_predict_stats_refuses_the_classification_task():
    model = _joint_model().set_task("classification").eval()
    X, y = _data("classification")
    with pytest.raises(AssertionError, match="regression"):
        model.predict_stats(X, y)


@pytest.mark.parametrize("task", ["classification", "regression"])
def test_loading_reports_exactly_what_a_single_task_checkpoint_cannot_supply(task):
    single = _clf_model() if task == "classification" else _reg_model()
    missing = _joint_model().load_single_task_state_dict(single.state_dict(), task)
    assert {"col_embedder.task_embed", "icl_predictor.task_embed"} <= set(missing)
    other = "reg_" if task == "classification" else "y_encoder"
    assert all("task_embed" in k or other in k or "decoder" in k for k in missing)


def test_warm_start_from_both_checkpoints():
    """Trunk and classification head from one, regression head from the other."""

    torch.manual_seed(0)
    clf, reg = _clf_model(), _reg_model()
    joint = _joint_model()
    joint.load_single_task_state_dict(clf.state_dict(), "classification")
    missing = joint.load_single_task_state_dict(reg.state_dict(), "regression", heads_only=True)

    assert not any(k.startswith(("icl_predictor.reg_", "col_embedder.reg_")) for k in missing)
    torch.testing.assert_close(joint.icl_predictor.reg_decoder[2].weight, reg.icl_predictor.decoder[2].weight)
    torch.testing.assert_close(joint.col_embedder.reg_y_encoder.weight, reg.col_embedder.y_encoder.weight)
    # ...and loading the second head did not disturb the first model's trunk
    torch.testing.assert_close(joint.icl_predictor.tf_icl.blocks[0].linear1.weight,
                               clf.icl_predictor.tf_icl.blocks[0].linear1.weight)

    X, y = _data("classification")
    _assert_same_outputs(_all_outputs(clf, X, y), _all_outputs(joint.set_task("classification"), X, y))


def test_an_architecture_mismatch_is_refused():
    """The v2 recipes differ in bias_free_ln; loading across it must not half-succeed."""

    biased = _reg_model(bias_free_ln=False).state_dict()
    unbiased = _reg_model(bias_free_ln=True).state_dict()
    with pytest.raises(ValueError, match="not in this model"):
        _joint_model(bias_free_ln=True).load_single_task_state_dict(biased, "regression")
    with pytest.raises(ValueError, match="lacks keys"):
        _joint_model(bias_free_ln=False).load_single_task_state_dict(unbiased, "regression")


def test_loader_is_only_for_multitask_models():
    with pytest.raises(ValueError, match="multitask"):
        _clf_model().load_single_task_state_dict(_clf_model().state_dict(), "classification")


# --------------------------------------------------------------------------- #
# the task input
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("module", ["col_embedder", "icl_predictor"])
def test_each_task_embedding_changes_the_output(module):
    """Both injection points must actually be wired to the prediction."""

    torch.manual_seed(0)
    model = _joint_model().set_task("regression").eval()
    X, y = _data("regression")
    with torch.no_grad():
        before = model(X, y)
        getattr(model, module).task_embed[1].normal_(0, 0.5)
        after = model(X, y)
    assert not torch.allclose(before, after)


@pytest.mark.parametrize("task", ["classification", "regression"])
@pytest.mark.parametrize("mode", ["kv", "repr"])
def test_cached_and_uncached_agree_with_a_trained_task_embedding(task, mode):
    """With zero embeddings this would pass even if test rows never got one.

    The task embedding is on every row, unlike the label embedding which is on train rows
    only, so the cached pass has to add it to fresh test rows itself. Randomizing it makes a
    missed or doubled application visible.
    """

    torch.manual_seed(0)
    model = _joint_model()
    _randomize_task_embeddings(model)
    model.set_task(task).eval()

    X, y = _data(task)
    X_train, X_test = X[:, :N_TRAIN], X[:, N_TRAIN:]
    with torch.no_grad():
        plain = model(X, y)
        stored = model.forward_with_cache(X_train=X_train, y_train=y, X_test=X_test,
                                          store_cache=True, cache_mode=mode)
        reused = model.forward_with_cache(X_test=X_test, use_cache=True, store_cache=False)
    torch.testing.assert_close(stored, plain, rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(reused, plain, rtol=1e-5, atol=1e-5)


def test_switching_task_clears_the_cache():
    model = _joint_model().set_task("classification").eval()
    X, y = _data("classification")
    with torch.no_grad():
        model.forward_with_cache(X_train=X[:, :N_TRAIN], y_train=y, store_cache=True)
    assert model.has_cache
    model.set_task("classification")
    assert model.has_cache, "re-selecting the same task should keep the cache"
    model.set_task("regression")
    assert not model.has_cache


def test_a_cache_from_the_other_task_is_refused():
    """An external cache bypasses set_task's clearing, so it is checked at use."""

    model = _joint_model().set_task("classification").eval()
    X, y = _data("classification")
    with torch.no_grad():
        model.forward_with_cache(X_train=X[:, :N_TRAIN], y_train=y, store_cache=True)
    cache = model._cache
    model.set_task("regression")
    with pytest.raises(ValueError, match="built for classification"):
        model.forward_with_cache(X_test=X[:, N_TRAIN:], cache=cache)


def test_the_cache_task_survives_copying():
    from tabicl._model.kv_cache import TabICLCache

    model = _joint_model().set_task("regression").eval()
    X, y = _data("regression")
    with torch.no_grad():
        model.forward_with_cache(X_train=X[:, :N_TRAIN], y_train=y, store_cache=True)
    cache = model._cache
    assert cache.task == "regression"
    assert cache.slice_batch(0, 1).task == "regression"
    assert cache.to("cpu").task == "regression"
    assert TabICLCache.concat([cache, cache]).task == "regression"


# --------------------------------------------------------------------------- #
# joint training
# --------------------------------------------------------------------------- #


def _loss(model, task, seed=0):
    X, y = _data(task, seed)
    y_test = (torch.arange(N_TEST).remainder(3).repeat(2, 1).float() if task == "classification"
              else torch.randn(2, N_TEST))
    pred = model(X, y)
    if task == "classification":
        return F.cross_entropy(pred.flatten(end_dim=-2), y_test.long().flatten())
    alphas = torch.linspace(0, 1, TINY["num_quantiles"] + 2)[1:-1]
    err = y_test.unsqueeze(-1) - pred
    return torch.maximum(alphas * err, (alphas - 1) * err).mean()


@pytest.mark.parametrize("task", ["classification", "regression"])
def test_a_task_loss_trains_the_trunk_and_its_own_head_only(task):
    """Otherwise the heads are not really task-specific, or the trunk is not shared."""

    torch.manual_seed(0)
    model = _joint_model().train()
    _loss(model.set_task(task), task).backward()

    grads = {n: p.grad for n, p in model.named_parameters()}
    own = ("reg_y_encoder", "reg_decoder") if task == "regression" else ("y_encoder", "decoder")
    other = ("y_encoder", "decoder") if task == "regression" else ("reg_y_encoder", "reg_decoder")

    def owned_by(name, parts):
        return any(f"{module}.{part}." in name for module in ("col_embedder", "icl_predictor") for part in parts)

    for name, grad in grads.items():
        if owned_by(name, other):
            assert grad is None, f"{task} loss reached {name}"
    assert any(g is not None and g.abs().sum() > 0 for n, g in grads.items() if owned_by(n, own))
    assert grads["icl_predictor.tf_icl.blocks.0.linear1.weight"].abs().sum() > 0
    assert grads["col_embedder.in_linear.weight"].abs().sum() > 0

    # Zero-initialized task embeddings still learn, and only the active task's row does
    idx = 0 if task == "classification" else 1
    for module in ("col_embedder", "icl_predictor"):
        g = grads[f"{module}.task_embed"]
        assert g[idx].abs().sum() > 0
        assert g[1 - idx].abs().sum() == 0


def test_one_optimizer_step_over_both_tasks():
    """The step the trainer will take: accumulate one loss per task, then update once."""

    torch.manual_seed(0)
    model = _joint_model().train()
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    before = {n: p.detach().clone() for n, p in model.named_parameters()}

    for task in ("classification", "regression"):
        (_loss(model.set_task(task), task) / 2).backward()
    opt.step()

    changed = {n for n, p in model.named_parameters() if not torch.equal(p, before[n])}
    for name in ("icl_predictor.decoder.2.weight", "icl_predictor.reg_decoder.2.weight",
                 "icl_predictor.task_embed", "col_embedder.task_embed",
                 "icl_predictor.tf_icl.blocks.0.linear1.weight"):
        assert name in changed, name
    for task in ("classification", "regression"):
        assert torch.isfinite(_loss(model.set_task(task), task))


# --------------------------------------------------------------------------- #
# sklearn wrappers pick their own head
# --------------------------------------------------------------------------- #


@pytest.fixture
def joint_checkpoint(tmp_path):
    torch.manual_seed(0)
    config = dict(max_classes=10, multitask=True, **TINY)
    path = tmp_path / "joint.ckpt"
    torch.save({"config": config, "state_dict": TabICL(**config).state_dict()}, path)
    return path


def test_wrappers_select_their_task(joint_checkpoint):
    from tabicl import TabICLClassifier, TabICLRegressor

    rng = np.random.default_rng(0)
    X = rng.normal(size=(40, 4))

    clf = TabICLClassifier(model_path=joint_checkpoint, n_estimators=1, device="cpu")
    clf.fit(X, (X[:, 0] > 0).astype(int))
    assert clf.model_.task == "classification"
    assert clf.predict_proba(X[:5]).shape == (5, 2)

    reg = TabICLRegressor(model_path=joint_checkpoint, n_estimators=1, device="cpu")
    reg.fit(X, X[:, 0] * 2.0)
    assert reg.model_.task == "regression"
    assert reg.predict(X[:5]).shape == (5,)
