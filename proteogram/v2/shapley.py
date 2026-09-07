"""Shapley-value attribution for proteogram similarity.

Why Shapley rather than a saliency ratio
----------------------------------------
Gradient-based attribution (``gradcam.compute_decomposed``) produces numbers
whose *scale* has no principled meaning — you can compare two pixels within a
channel, but a statistic that aggregates across channels, such as the
Energy/Distance ratio, is a heuristic with no guarantee behind it.

Shapley values satisfy **efficiency**: they sum exactly to
``f(everything) - f(nothing)``.  For a query→target pair that means

    phi_ch0 + phi_ch1 + phi_ch2  ==  cos(query, target) - cos(neutral, target)

so each channel's contribution is a share of an actually measurable quantity.
"VdW accounts for 42% of this pair's similarity" is a falsifiable statement;
"the E/D ratio is 1.77" is not.

Two applications live here
--------------------------
- :func:`channel_shapley` — the three colour channels as three players.  With
  only 3 players the Shapley values are computed **exactly** from all 8
  coalitions: 8 forward passes per pair, no sampling, no approximation.
- :func:`residue_shapley` — one player per residue.  Exact is impossible
  (2^N coalitions), so this uses permutation sampling, which is unbiased and
  converges as ``n_permutations`` grows.  Cost is
  ``n_permutations * (n_residues + 1)`` forward passes.

Both mask "removed" players to the neutral sentinel (128) — see
:mod:`proteogram.v2.masking` for why that value and not 0.

Caveat shared by all occlusion-based attribution: masked images are off the
training manifold.  Shapley inherits this; it buys axiomatic aggregation, not
freedom from that assumption.
"""

from __future__ import annotations

from itertools import combinations
from math import factorial
from typing import Callable, Optional, Sequence

import numpy as np

from .masking import (
    active_residues,
    cosine,
    embed_array,
    mask_channels,
    mask_residues,
    neutral_like,
)


# ---------------------------------------------------------------------------
# Generic estimators (no proteogram knowledge)
# ---------------------------------------------------------------------------

def exact_shapley(value_fn: Callable[[frozenset], float],
                  n_players: int) -> np.ndarray:
    """Exact Shapley values by enumerating every coalition.

    Only tractable for small ``n_players`` (2**n evaluations); intended for the
    3-channel case.

    Args:
        value_fn:  Callable mapping a coalition (frozenset of player indices)
                   to a scalar payoff.  Called once per distinct coalition.
        n_players: Number of players.

    Returns:
        Float array of shape ``(n_players,)`` of Shapley values.  These sum to
        ``value_fn(all) - value_fn(empty)`` up to floating-point error.
    """
    players = list(range(n_players))

    # Evaluate every coalition once and cache
    cache: dict = {}
    for size in range(n_players + 1):
        for combo in combinations(players, size):
            key = frozenset(combo)
            cache[key] = value_fn(key)

    phi = np.zeros(n_players, dtype=np.float64)
    n_fact = factorial(n_players)
    for i in players:
        others = [p for p in players if p != i]
        for size in range(len(others) + 1):
            weight = factorial(size) * factorial(n_players - size - 1) / n_fact
            for combo in combinations(others, size):
                s = frozenset(combo)
                phi[i] += weight * (cache[s | {i}] - cache[s])
    return phi


def permutation_shapley(value_fn: Callable[[frozenset], float],
                        n_players: int,
                        n_permutations: int,
                        rng: Optional[np.random.Generator] = None,
                        progress: Optional[Callable] = None) -> np.ndarray:
    """Unbiased Shapley estimates via permutation sampling.

    For each sampled permutation, players are added one at a time and each
    player is credited with the marginal change it causes.  Averaging over
    permutations converges to the exact Shapley values.

    Args:
        value_fn:       Coalition → payoff callable.
        n_players:      Number of players.
        n_permutations: Permutations to sample.  Cost is
                        ``n_permutations * (n_players + 1)`` calls to
                        ``value_fn``.
        rng:            Optional numpy Generator for reproducibility.
        progress:       Optional callable invoked once per permutation.

    Returns:
        Float array of shape ``(n_players,)``.  Sums to
        ``value_fn(all) - value_fn(empty)`` in expectation.
    """
    rng = rng or np.random.default_rng()
    phi = np.zeros(n_players, dtype=np.float64)

    for _ in range(n_permutations):
        order = rng.permutation(n_players)
        coalition: set = set()
        prev = value_fn(frozenset())
        for p in order:
            coalition.add(int(p))
            cur = value_fn(frozenset(coalition))
            phi[int(p)] += cur - prev
            prev = cur
        if progress is not None:
            progress()

    return phi / n_permutations


# ---------------------------------------------------------------------------
# Application 1 — channel-level (exact)
# ---------------------------------------------------------------------------

#: Human-readable channel names.  Note each channel carries a different
#: quantity in the upper vs. lower triangle of the proteogram, so these are
#: shorthand rather than a single physical term.
IMAGE_CHANNEL_NAMES = (
    'ch0_vdw_att/es_att',
    'ch1_vdw_rep/es_rep',
    'ch2_distance/hydrophobicity',
)


def channel_shapley(embed_net, device, query_arr: np.ndarray,
                    target_emb) -> dict:
    """Exact Shapley values for the three proteogram channels.

    Answers "how much of this pair's similarity does each channel account
    for?" with 8 forward passes and an exactness guarantee, replacing the
    heuristic Energy/Distance ratio.

    Args:
        embed_net:  Embedding network in eval mode.
        device:     Torch device.
        query_arr:  ``uint8`` query proteogram ``(H, W, 3)``.
        target_emb: Precomputed target embedding ``(1, d)``.

    Returns:
        Dict with per-channel Shapley values (``shap_ch0`` …), the empty and
        full coalition scores, the efficiency residual (should be ~0), the
        share of explained similarity attributable to the energy channels, and
        a Shapley-based energy/distance ratio directly comparable in spirit to
        the old E/D ratio.

    Interpreting the aggregates — mind the null baseline:
        The "energy" group is two channels (0, 1) and "distance" is one (2),
        so a model with **no channel preference at all** does not score 0.5 or
        1.0.  It scores:

            energy_share            -> 2/3  (0.667)
            energy_distance_ratio   -> 2.0

        Those are the values to compare against, not 0.5 and 1.0.  An E/D
        ratio of 1.77 is therefore *below* no-preference, i.e. relatively
        distance-leaning — the opposite of what a naive "1.0 means equal"
        reading suggests.  ``energy_share_vs_null`` is reported for
        convenience: it is positive only when the energy channels genuinely
        carry more than their share.
    """
    def value_fn(coalition: frozenset) -> float:
        masked = mask_channels(query_arr, coalition, keep=True)
        return cosine(embed_array(embed_net, masked, device), target_emb)

    phi = exact_shapley(value_fn, 3)

    f_empty = value_fn(frozenset())
    f_full  = value_fn(frozenset({0, 1, 2}))
    total   = f_full - f_empty

    energy = float(phi[0] + phi[1])
    dist   = float(phi[2])

    return {
        'shap_ch0': float(phi[0]),
        'shap_ch1': float(phi[1]),
        'shap_ch2': float(phi[2]),
        'shap_sum': float(phi.sum()),
        'cos_full': f_full,
        'cos_neutral': f_empty,
        'explained_delta': float(total),
        # Efficiency axiom self-check: must be ~0 for an exact computation.
        'efficiency_residual': float(phi.sum() - total),
        # Share of the explained similarity carried by the energy channels.
        # Undefined when the pair explains ~nothing, hence the guard.
        'energy_share': float(energy / total) if abs(total) > 1e-9 else float('nan'),
        # Same, expressed relative to the 2/3 no-preference baseline: positive
        # means the energy channels carry more than their proportional share.
        'energy_share_vs_null': (
            float(energy / total - 2.0 / 3.0) if abs(total) > 1e-9 else float('nan')
        ),
        # Shapley analogue of the old E/D ratio, for side-by-side comparison.
        # No-preference value is 2.0 (two energy channels vs. one distance).
        'shap_energy_distance_ratio': (
            float(energy / dist) if abs(dist) > 1e-9 else float('nan')
        ),
    }


# ---------------------------------------------------------------------------
# Application 2 — residue-level (sampled)
# ---------------------------------------------------------------------------

def residue_shapley(embed_net, device, query_arr: np.ndarray, target_emb,
                    n_permutations: int = 3,
                    rng: Optional[np.random.Generator] = None,
                    residues: Optional[Sequence[int]] = None) -> np.ndarray:
    """Per-residue Shapley values via permutation sampling.

    Unlike Grad-CAM — whose 7x7 feature map upsamples to ~29x29-pixel blocks —
    this resolves individual residues, because a residue genuinely is one
    player here.

    Padding residues are excluded automatically (they cannot change the score,
    so spending forward passes on them is waste); their returned value is 0.

    Args:
        embed_net:      Embedding network in eval mode.
        device:         Torch device.
        query_arr:      ``uint8`` query proteogram ``(H, W, 3)``.
        target_emb:     Precomputed target embedding ``(1, d)``.
        n_permutations: Permutations to sample.  Cost is
                        ``n_permutations * (n_active_residues + 1)`` forward
                        passes.  Low values are noisy — raise until the values
                        stabilise.
        rng:            Optional numpy Generator.
        residues:       Explicit player set; defaults to the non-padding
                        residues detected in ``query_arr``.

    Returns:
        Float array of shape ``(H,)`` — one Shapley value per residue index,
        zero for residues excluded from the player set.
    """
    n = query_arr.shape[0]
    active = np.asarray(residues if residues is not None
                        else active_residues(query_arr), dtype=int)
    if active.size == 0:
        return np.zeros(n, dtype=np.float64)

    def value_fn(coalition: frozenset) -> float:
        kept = active[list(coalition)] if coalition else []
        masked = mask_residues(query_arr, kept, keep=True)
        return cosine(embed_array(embed_net, masked, device), target_emb)

    phi_active = permutation_shapley(value_fn, len(active), n_permutations, rng)

    phi = np.zeros(n, dtype=np.float64)
    phi[active] = phi_active
    return phi


def residue_shapley_map(phi: np.ndarray) -> np.ndarray:
    """Broadcast per-residue Shapley values to an ``(H, W)`` map for plotting.

    Pixel (i, j) receives the mean of the two residues' values, matching the
    fact that the pixel encodes their interaction.
    """
    return (phi[:, None] + phi[None, :]) / 2.0
