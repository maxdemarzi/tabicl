"""Tests for TP-01: Fourier value encoding in the cell encoder.

The motivating claim from the TabPFN-3.5 report is that a bank of learned frequencies
"captures small differences better than a linear projection of the raw number", and that
this matters most for ordinal-encoded categoricals with high cardinality.

The tests below pin that as a structural property rather than as a benchmark number: under a
linear projection the embeddings of a sequence of values are collinear, so adjacent codes are
*necessarily* near-identical no matter what the weights are. Under Fourier features they are
not. That is the whole difference, and it is checkable without training anything.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from tabicl._model.layers import FourierValueEncoder, SkippableLinear
from tabicl._model.tabicl import TabICL


@pytest.fixture
def encoder():
    torch.manual_seed(0)
    return FourierValueEncoder(3, 64, n_freqs=16)


# --------------------------------------------------------------------------- #
# drop-in compatibility with SkippableLinear
# --------------------------------------------------------------------------- #


def test_output_shape_matches_skippable_linear(encoder):
    x = torch.randn(2, 5, 40, 3)
    assert encoder(x).shape == SkippableLinear(3, 64)(x).shape


def test_skip_value_contract_is_preserved(encoder):
    """Padded cells must pass through as the sentinel, as SkippableLinear does."""

    x = torch.randn(2, 10, 3)
    x[0, 3] = -100.0
    x[1, 7] = -100.0
    out = encoder(x)

    assert torch.all(out[0, 3] == -100.0)
    assert torch.all(out[1, 7] == -100.0)
    assert not torch.any(out[0, 4] == -100.0)


def test_partially_skipped_cell_is_not_skipped(encoder):
    """Only cells where EVERY group position is the sentinel are skipped."""

    x = torch.randn(1, 4, 3)
    x[0, 2, 0] = -100.0  # one position, not all three
    out = encoder(x)
    assert not torch.all(out[0, 2] == -100.0)


def test_handles_arbitrary_batch_dimensions(encoder):
    for shape in [(7, 3), (2, 7, 3), (2, 4, 7, 3), (2, 3, 4, 7, 3)]:
        assert encoder(torch.randn(*shape)).shape == (*shape[:-1], 64)


def test_gradients_flow_to_the_frequency_bank(encoder):
    out = encoder(torch.randn(2, 6, 3))
    out.sum().backward()
    assert encoder.freqs.grad is not None
    assert torch.any(encoder.freqs.grad != 0)


def test_frequencies_can_be_frozen():
    enc = FourierValueEncoder(3, 32, n_freqs=8, learnable_freqs=False)
    assert not enc.freqs.requires_grad
    enc(torch.randn(2, 6, 3)).sum().backward()
    assert enc.freqs.grad is None


def test_frequency_bank_is_log_spaced_at_init():
    enc = FourierValueEncoder(2, 32, n_freqs=8, min_freq=1.0, max_freq=32.0)
    row = enc.freqs[0].detach().numpy()
    assert row[0] == pytest.approx(1.0)
    assert row[-1] == pytest.approx(32.0)
    ratios = row[1:] / row[:-1]
    np.testing.assert_allclose(ratios, ratios[0], rtol=1e-5)


# --------------------------------------------------------------------------- #
# the property the change exists for
# --------------------------------------------------------------------------- #


def _embed_sequence(module, values, n_groups=1):
    """Embed a 1-D sequence of scalar values, replicated across group positions."""

    x = torch.tensor(values, dtype=torch.float32).reshape(-1, 1).repeat(1, n_groups)
    with torch.no_grad():
        return module(x.unsqueeze(0)).squeeze(0)


def test_linear_projection_makes_value_embeddings_collinear():
    """The baseline limitation, stated as a fact about the module rather than a benchmark.

    A linear map sends v -> v*w + b, so the embeddings of any set of values lie on a single
    line. Centred, that matrix has rank 1 whatever the weights are.
    """

    torch.manual_seed(0)
    linear = SkippableLinear(1, 64)
    emb = _embed_sequence(linear, np.arange(50, dtype=float) / 10.0)
    centred = emb - emb.mean(0, keepdim=True)
    rank = torch.linalg.matrix_rank(centred, rtol=1e-4).item()
    assert rank == 1


def test_fourier_encoding_breaks_that_collinearity():
    torch.manual_seed(0)
    enc = FourierValueEncoder(1, 64, n_freqs=16)
    emb = _embed_sequence(enc, np.arange(50, dtype=float) / 10.0)
    centred = emb - emb.mean(0, keepdim=True)
    rank = torch.linalg.matrix_rank(centred, rtol=1e-4).item()
    assert rank > 1


def test_adjacent_ordinal_codes_are_better_separated_than_under_a_linear_map():
    """The high-cardinality case the report calls out.

    Adjacent integer codes are unrelated categories. Measured as separation between
    neighbours relative to the spread of the whole column, Fourier features must do better
    than a linear projection -- under which the ratio is fixed by the geometry at 1/(N-1)
    regardless of weights.
    """

    torch.manual_seed(0)
    # Standard-scaled ordinal codes for a 200-category column.
    codes = np.arange(200, dtype=float)
    values = (codes - codes.mean()) / codes.std()

    def separation(module):
        emb = _embed_sequence(module, values)
        neighbour = torch.linalg.norm(emb[1:] - emb[:-1], dim=-1).mean()
        spread = torch.linalg.norm(emb - emb.mean(0, keepdim=True), dim=-1).mean()
        return (neighbour / spread).item()

    linear_sep = separation(SkippableLinear(1, 64))
    fourier_sep = separation(FourierValueEncoder(1, 64, n_freqs=16))

    assert fourier_sep > linear_sep


# --------------------------------------------------------------------------- #
# model wiring
# --------------------------------------------------------------------------- #


def test_flag_is_off_by_default_so_existing_checkpoints_load():
    model = TabICL(max_classes=10)
    assert isinstance(model.col_embedder.in_linear, SkippableLinear)


def test_flag_swaps_the_encoder():
    model = TabICL(max_classes=10, col_fourier_value=True, col_fourier_freqs=16)
    assert isinstance(model.col_embedder.in_linear, FourierValueEncoder)
    assert model.col_embedder.in_linear.freqs.shape == (3, 16)  # (G, F)


def test_parameter_cost_is_negligible():
    """~8K parameters against 27M: this must not be mistaken for a capacity change."""

    off = sum(p.numel() for p in TabICL(max_classes=10).parameters())
    on = sum(p.numel() for p in TabICL(max_classes=10, col_fourier_value=True).parameters())
    assert 0 < (on - off) < 0.001 * off


def test_forward_pass_runs_with_the_flag_on():
    torch.manual_seed(0)
    model = TabICL(max_classes=10, col_fourier_value=True, col_fourier_freqs=8).eval()
    # X is (B, T, H); the train/test split is implied by len(y_train).
    X = torch.randn(1, 40, 6)
    y_train = torch.randint(0, 3, (1, 30)).float()
    with torch.no_grad():
        out = model(X, y_train, d=torch.tensor([6]))
    assert out.shape[0] == 1 and torch.isfinite(out).all()
