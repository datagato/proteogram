"""Unit tests for the similarity-attribution tools (no trained model needed).

These pin down the properties the docs rely on:
- masking replaces regions with the neutral sentinel 128, never 0
- exact Shapley satisfies efficiency, symmetry and the null-player axiom
- permutation Shapley converges to the exact values
- channel_shapley's efficiency_residual is ~0 on a real (tiny) network
- baseline-corrected Grad x Input gives ~zero attribution to neutral pixels
"""

import numpy as np
import pytest
import torch
import torch.nn as nn

from proteogram.v2.gradcam import NEUTRAL_PIXEL_VALUE, GradCAM
from proteogram.v2.masking import (
    _TO_TENSOR,
    active_residues,
    mask_channels,
    mask_pixels,
    mask_residues,
    residue_scores,
)
from proteogram.v2.shapley import (
    channel_shapley,
    exact_shapley,
    permutation_shapley,
    residue_shapley,
)
from PIL import Image


N = 16  # small square proteogram for fast tests


@pytest.fixture
def arr():
    """A padded proteogram: 10 real residues, the rest neutral padding."""
    rng = np.random.default_rng(0)
    a = np.full((N, N, 3), NEUTRAL_PIXEL_VALUE, dtype=np.uint8)
    real = rng.integers(0, 256, size=(10, 10, 3), dtype=np.uint8)
    real[real == NEUTRAL_PIXEL_VALUE] = 0  # keep "real" pixels non-neutral
    a[:10, :10] = real
    return a


@pytest.fixture
def net():
    torch.manual_seed(0)
    model = nn.Sequential(
        nn.Conv2d(3, 4, 3, padding=1),
        nn.ReLU(),
        nn.AdaptiveAvgPool2d(4),
        nn.Flatten(),
        nn.Linear(64, 8),
    )
    return model.eval()


def _embed_tensor(a):
    return _TO_TENSOR(Image.fromarray(a)).unsqueeze(0).float()


# ---------------------------------------------------------------------------
# masking
# ---------------------------------------------------------------------------

def test_active_residues_ignores_padding(arr):
    assert active_residues(arr).tolist() == list(range(10))


def test_mask_residues_uses_neutral_and_copies(arr):
    orig = arr.copy()
    out = mask_residues(arr, [2])
    assert (out[2, :, :] == NEUTRAL_PIXEL_VALUE).all()
    assert (out[:, 2, :] == NEUTRAL_PIXEL_VALUE).all()
    assert (out[3, 3] == arr[3, 3]).all()
    np.testing.assert_array_equal(arr, orig)  # input untouched


def test_mask_residues_keep_is_complement(arr):
    out = mask_residues(arr, [2], keep=True)
    np.testing.assert_array_equal(out[2, :, :], arr[2, :, :])
    np.testing.assert_array_equal(out[:, 2, :], arr[:, 2, :])
    assert (out[3, 4] == NEUTRAL_PIXEL_VALUE).all()


def test_mask_pixels(arr):
    out = mask_pixels(arr, [1 * N + 3])
    assert (out[1, 3] == NEUTRAL_PIXEL_VALUE).all()
    assert (out[3, 1] == arr[3, 1]).all()  # (j, i) not touched
    kept = mask_pixels(arr, [1 * N + 3], keep=True)
    np.testing.assert_array_equal(kept[1, 3], arr[1, 3])
    assert (kept[0, 0] == NEUTRAL_PIXEL_VALUE).all()


def test_mask_channels(arr):
    out = mask_channels(arr, [1])
    assert (out[:, :, 1] == NEUTRAL_PIXEL_VALUE).all()
    np.testing.assert_array_equal(out[:, :, 0], arr[:, :, 0])
    kept = mask_channels(arr, [1], keep=True)
    np.testing.assert_array_equal(kept[:, :, 1], arr[:, :, 1])
    assert (kept[:, :, [0, 2]] == NEUTRAL_PIXEL_VALUE).all()


def test_residue_scores_row_plus_column():
    m = np.zeros((4, 4))
    m[1, :] = 1.0
    s = residue_scores(m)
    assert s[1] == pytest.approx((4 + 1) / 8)
    assert s[0] == pytest.approx(1 / 8)


# ---------------------------------------------------------------------------
# Shapley estimators on games with known answers
# ---------------------------------------------------------------------------

def _additive_game(weights):
    return lambda S: float(sum(weights[i] for i in S))


def test_exact_shapley_additive_game():
    w = [0.3, -0.2, 0.9]
    np.testing.assert_allclose(exact_shapley(_additive_game(w), 3), w)


def test_exact_shapley_axioms_on_interaction_game():
    # v(S) = 1 if {0,1} subset of S; player 2 is a null player
    v = lambda S: 1.0 if {0, 1} <= S else 0.0
    phi = exact_shapley(v, 3)
    np.testing.assert_allclose(phi, [0.5, 0.5, 0.0])        # symmetry + null
    assert phi.sum() == pytest.approx(v(frozenset({0, 1, 2})) - v(frozenset()))


def test_permutation_shapley_converges_to_exact():
    v = lambda S: float(len(S) ** 2 + (2 in S))
    exact = exact_shapley(v, 4)
    approx = permutation_shapley(v, 4, n_permutations=4000,
                                 rng=np.random.default_rng(0))
    np.testing.assert_allclose(approx, exact, atol=0.1)


def test_permutation_shapley_efficiency_holds_per_sample():
    v = lambda S: float(np.sin(len(S)) + 0.1 * sum(S))
    phi = permutation_shapley(v, 5, n_permutations=3,
                              rng=np.random.default_rng(1))
    assert phi.sum() == pytest.approx(v(frozenset(range(5))) - v(frozenset()))


# ---------------------------------------------------------------------------
# Shapley on a real (tiny) network
# ---------------------------------------------------------------------------

def test_channel_shapley_efficiency(net, arr):
    target = np.roll(arr, 1, axis=0)
    with torch.no_grad():
        t_emb = net(_embed_tensor(target))
    res = channel_shapley(net, torch.device('cpu'), arr, t_emb)
    assert abs(res['efficiency_residual']) < 1e-6
    assert res['shap_sum'] == pytest.approx(res['explained_delta'], abs=1e-6)


def test_residue_shapley_padding_gets_zero(net, arr):
    with torch.no_grad():
        t_emb = net(_embed_tensor(arr))
    phi = residue_shapley(net, torch.device('cpu'), arr, t_emb,
                          n_permutations=2, rng=np.random.default_rng(0))
    assert phi.shape == (N,)
    assert (phi[10:] == 0).all()
    assert np.abs(phi[:10]).sum() > 0


# ---------------------------------------------------------------------------
# Grad x Input baseline correction
# ---------------------------------------------------------------------------

def test_decomposed_attribution_zero_on_neutral_pixels(net, arr):
    gc = GradCAM(net, device='cpu')
    q = _embed_tensor(arr)
    t = _embed_tensor(np.roll(arr, 1, axis=1))
    attr_display, attr_raw, cos = gc.compute_decomposed(q, t)

    assert attr_raw.shape == (3, N, N)
    assert -1.0 <= cos <= 1.0
    neutral = (arr == NEUTRAL_PIXEL_VALUE).transpose(2, 0, 1)
    assert attr_raw[neutral].max() < 1e-6
    assert attr_raw[~neutral].max() > 1e-4
    # display copy is per-channel min-max scaled to [0, 1]
    for k in range(3):
        assert attr_display[k].min() == pytest.approx(0.0)
        assert attr_display[k].max() == pytest.approx(1.0)


def test_gradcam_compute_shape_and_range(net, arr):
    gc = GradCAM(net, device='cpu')
    cam = gc.compute(_embed_tensor(arr), _embed_tensor(arr[::-1].copy()))
    assert cam.shape == (N, N)
    assert cam.min() >= 0.0 and cam.max() <= 1.0 + 1e-6
