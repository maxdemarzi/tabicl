"""Tests for TP-07 trainer wiring: one optimizer step over both tasks.

Single-task runs were checked bit-identical to the pre-TP-07 trainer (weights and logged
metrics after 3 steps, both tasks, AdamW and Muon) by running the old ``_run.py`` from git
side by side; that comparison needs the old file, so it is not repeated here. What is
pinned down here is the joint step itself: that its gradient is exactly the weighted sum of
the two tasks' gradients -- so the zero-weighted DDP term and the micro-batch scaling add
nothing -- and the configuration around it.
"""

from __future__ import annotations

import importlib.util
import sys
import types

import pytest
import torch

# Checked before stubbing: spawned dataloader workers do not inherit the stub
HAS_XGBOOST = importlib.util.find_spec("xgboost") is not None

# The prior package imports xgboost at load time for the v1 TreeSCM.
sys.modules.setdefault("xgboost", types.SimpleNamespace(XGBRegressor=object, XGBClassifier=object))

from tabicl._model.tabicl import TabICL  # noqa: E402
from tabicl.train._run import Trainer  # noqa: E402
from tabicl.train._train_config import build_parser  # noqa: E402

TINY = [
    "--device", "cpu", "--wandb_log", "False", "--n_jobs", "1", "--prior_type", "mlp_scm",
    "--batch_size", "8", "--micro_batch_size", "2", "--gradient_clipping", "0",
    "--embed_dim", "16", "--col_num_blocks", "1", "--col_nhead", "2", "--col_num_inds", "4",
    "--row_num_blocks", "1", "--row_nhead", "2", "--row_num_cls", "2",
    "--icl_num_blocks", "1", "--icl_nhead", "2", "--num_quantiles", "15",
    # With zero_init most gradients are exactly zero at step 0, which would let the gradient
    # comparisons below pass vacuously; the v2 recipe trains with it off. Warmup would start
    # the learning rate at 0, so the first step would move nothing.
    "--zero_init", "False", "--scheduler", "constant",
]


def _trainer(*extra):
    torch.manual_seed(0)
    return Trainer(build_parser().parse_args(TINY + list(extra)))


def _batch(task, B, seed=0):
    g = torch.Generator().manual_seed(seed)
    T, H, train = 40, 6, 30
    X = torch.randn(B, T, H, generator=g)
    if task == "regression":
        y = torch.randn(B, T, generator=g)
    else:
        y = torch.randint(0, 3, (B, T), generator=g).float()
    return X, y, torch.full((B,), H), torch.full((B,), T), torch.full((B,), train)


def _step_grads(trainer, batches):
    """Run one training step and return the gradients it would apply, without applying them."""

    captured = {}

    def capture():
        for name, p in trainer.raw_model.named_parameters():
            captured[name] = None if p.grad is None else p.grad.clone()

    trainer.optimizer.step = capture
    trainer.run_batch(batches)
    return captured


def _both(seed=0):
    return {"classification": _batch("classification", 4, seed), "regression": _batch("regression", 4, seed)}


# --------------------------------------------------------------------------- #
# configuration
# --------------------------------------------------------------------------- #


def test_single_task_is_the_default():
    trainer = _trainer()
    assert trainer.tasks == ("classification",)
    assert not trainer.raw_model.multitask
    assert "multitask" not in trainer.model_config, "single-task configs must stay loadable by older code"


def test_multitask_builds_a_joint_model_and_two_priors():
    trainer = _trainer("--multitask", "True")
    assert trainer.tasks == ("classification", "regression")
    assert trainer.raw_model.multitask
    assert trainer.model_config["multitask"] is True
    assert set(trainer.dataloaders) == {"classification", "regression"}
    assert trainer.dataloaders["regression"].dataset.regression
    assert not trainer.dataloaders["classification"].dataset.regression


def test_the_batch_is_split_between_the_tasks():
    """A joint step sees as many datasets as a single-task step."""

    trainer = _trainer("--multitask", "True", "--batch_size", "7")
    assert trainer.task_batch_sizes == {"classification": 4, "regression": 3}
    assert trainer.dataloaders["classification"].dataset.batch_size == 4
    assert trainer.dataloaders["regression"].dataset.batch_size == 3


def test_multitask_needs_a_dataset_per_task():
    with pytest.raises(ValueError, match="at least 2"):
        _trainer("--multitask", "True", "--batch_size", "1")


def test_pregenerated_multitask_needs_both_priors(tmp_path):
    with pytest.raises(ValueError, match="--reg_prior_dir"):
        _trainer("--multitask", "True", "--prior_dir", str(tmp_path))


def test_unsupported_regression_method_is_refused():
    with pytest.raises(NotImplementedError):
        _trainer("--multitask", "True", "--regression_method", "gaussian")


def test_muon_keeps_one_group_for_single_task():
    trainer = _trainer("--muon", "True")
    assert len(trainer.optimizer.param_groups) == 1


def test_muon_sends_task_embeddings_to_adamw():
    trainer = _trainer("--multitask", "True", "--muon", "True")
    muon, adamw = trainer.optimizer.param_groups
    assert muon["use_muon"] and not adamw["use_muon"]
    model = trainer.raw_model
    assert {id(p) for p in adamw["params"]} == {id(model.col_embedder.task_embed), id(model.icl_predictor.task_embed)}
    n_params = len(list(model.parameters()))
    assert len(muon["params"]) + len(adamw["params"]) == n_params


# --------------------------------------------------------------------------- #
# the joint step
# --------------------------------------------------------------------------- #


def test_joint_gradient_is_the_weighted_sum_of_the_task_gradients():
    """The zero-weighted DDP term and the per-task scaling must add nothing else."""

    weight = 0.3
    trainer = _trainer("--multitask", "True", "--multitask_reg_weight", str(weight))
    batches = _both()

    clf = _step_grads(trainer, {"classification": batches["classification"]})
    reg = _step_grads(trainer, {"regression": batches["regression"]})
    joint = _step_grads(trainer, batches)

    checked = 0
    for name, g in joint.items():
        if g is None:  # buffers-as-parameters that never train, e.g. RoPE frequencies
            assert clf[name] is None and reg[name] is None, name
            continue
        torch.testing.assert_close(g, clf[name] + reg[name], rtol=1e-5, atol=1e-7, msg=name)
        checked += g.abs().sum() > 0
    assert checked > 0.9 * len(joint), "most gradients should be nonzero, or this test proves nothing"


def test_the_regression_weight_scales_only_regression_gradients():
    batch = {"regression": _batch("regression", 4)}
    full = _step_grads(_trainer("--multitask", "True"), batch)
    half = _step_grads(_trainer("--multitask", "True", "--multitask_reg_weight", "0.5"), batch)
    for name in full:
        if full[name] is not None:
            torch.testing.assert_close(half[name], full[name] * 0.5, rtol=1e-5, atol=1e-8, msg=name)


def test_each_task_leaves_the_other_head_an_exact_zero_gradient():
    """Defined (so DDP sees every parameter) but exactly zero (so nothing leaks)."""

    trainer = _trainer("--multitask", "True")
    grads = _step_grads(trainer, {"classification": _batch("classification", 4)})
    reg_only = {n for n, _ in trainer.raw_model.named_parameters()
                if n.startswith(("col_embedder.reg_", "icl_predictor.reg_"))}
    assert reg_only
    for name in reg_only:
        assert grads[name] is not None, name
        assert torch.count_nonzero(grads[name]) == 0, name


def test_joint_step_updates_trunk_both_heads_and_task_embeddings():
    trainer = _trainer("--multitask", "True", "--muon", "True")
    model = trainer.raw_model
    before = {n: p.detach().clone() for n, p in model.named_parameters()}
    results = trainer.run_batch(_both())

    assert set(results) == {"ce", "accuracy", "pinball"}
    assert all(torch.isfinite(torch.tensor(v)) for v in results.values())
    changed = {n for n, p in model.named_parameters() if not torch.equal(p, before[n])}
    for name in ("icl_predictor.decoder.2.weight", "icl_predictor.reg_decoder.2.weight",
                 "col_embedder.task_embed", "icl_predictor.task_embed",
                 "icl_predictor.tf_icl.blocks.0.linear1.weight"):
        assert name in changed, name


def test_logged_losses_are_unweighted():
    """So a joint run's curves read the same as the single-task control's."""

    batch = {"regression": _batch("regression", 4)}
    a = _trainer("--multitask", "True").run_batch(batch)
    b = _trainer("--multitask", "True", "--multitask_reg_weight", "0.1").run_batch(batch)
    assert a["pinball"] == pytest.approx(b["pinball"])


def test_a_saved_joint_checkpoint_serves_both_wrappers(tmp_path):
    trainer = _trainer("--multitask", "True", "--checkpoint_dir", str(tmp_path))
    trainer.run_batch(_both())
    trainer.save_checkpoint("step-1.ckpt")

    ckpt = torch.load(tmp_path / "step-1.ckpt", weights_only=True)
    model = TabICL(**ckpt["config"])
    model.load_state_dict(ckpt["state_dict"])  # strict
    assert model.multitask

    import numpy as np
    from tabicl import TabICLClassifier, TabICLRegressor

    X = np.random.default_rng(0).normal(size=(30, 4))
    path = tmp_path / "step-1.ckpt"
    clf = TabICLClassifier(model_path=path, n_estimators=1, device="cpu").fit(X, (X[:, 0] > 0).astype(int))
    reg = TabICLRegressor(model_path=path, n_estimators=1, device="cpu").fit(X, X[:, 0])
    assert clf.predict(X[:3]).shape == (3,)
    assert reg.predict(X[:3]).shape == (3,)


@pytest.mark.skipif(not HAS_XGBOOST,
                    reason="dataloader workers import the real prior package, which needs xgboost")
def test_end_to_end_joint_training_on_the_real_priors(tmp_path):
    trainer = _trainer("--multitask", "True", "--max_steps", "2", "--batch_size_per_gp", "2",
                       "--max_seq_len", "64", "--min_seq_len", "32", "--max_features", "6",
                       "--checkpoint_dir", str(tmp_path), "--save_temp_every", "1000", "--save_perm_every", "1000")
    trainer.train()
    assert trainer.curr_step == 2
