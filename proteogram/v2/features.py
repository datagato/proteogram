"""Per-channel energy / property histograms for v2 proteograms.

These histograms feed the EMD term of the composite similarity (§4.4 of
``docs/v2_similarity_design.md``). They are persisted next to each
embedding and are ~KB per protein — cheap to keep and fast to compare.

The functions here are numpy-only by design so they can be computed in
data-prep pipelines that do not pull torch.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import numpy as np


# Channel order in v2 proteograms:
#   upper triangle: [VdW_attractive, VdW_repulsive, Cα_distance]
#   lower triangle: [ES_attractive,  ES_repulsive,  Hydrophobicity_Δ]
UPPER_CHANNELS = ('vdw_att', 'vdw_rep', 'distogram')
LOWER_CHANNELS = ('es_att', 'es_rep', 'hyd_delta')


@dataclass(frozen=True)
class HistogramSpec:
    """Bin spec for one physical channel.

    Attributes
    ----------
    n_bins : int
        Number of bins in the resulting histogram.
    range_lo, range_hi : float
        Fixed bin range. Pinned across the whole dataset so EMD on the
        bin index makes sense. (Two histograms of different ranges are
        not comparable.)
    log_scale : bool
        If True, values are log1p-transformed before binning. Used for
        the heavy-tailed VdW and ES energy channels.
    """

    n_bins: int
    range_lo: float
    range_hi: float
    log_scale: bool = False


# Default specs. These were chosen for synthetic data and the design-doc
# defaults; users with real MD data should re-fit them on a small sample.
DEFAULT_SPECS: Dict[str, HistogramSpec] = {
    'vdw_att': HistogramSpec(32, 0.0, 12.0, log_scale=True),
    'vdw_rep': HistogramSpec(32, 0.0, 12.0, log_scale=True),
    'distogram': HistogramSpec(32, 0.0, 50.0, log_scale=False),  # Å
    'es_att': HistogramSpec(32, 0.0, 12.0, log_scale=True),
    'es_rep': HistogramSpec(32, 0.0, 12.0, log_scale=True),
    'hyd_delta': HistogramSpec(32, 0.0, 5.0, log_scale=False),
}


def _channel_values(image: np.ndarray, channel: str) -> np.ndarray:
    """Pull the upper- or lower-triangular pixels for one named channel.

    Image shape: ``(3, N, N)`` (channels-first; matches torchvision).
    """
    if image.ndim != 3 or image.shape[0] != 3:
        raise ValueError(f'expected (3, N, N) image; got {image.shape}')
    n = image.shape[-1]
    iu = np.triu_indices(n, k=1)  # strictly upper, no diagonal
    il = np.tril_indices(n, k=-1)
    if channel in UPPER_CHANNELS:
        ch = UPPER_CHANNELS.index(channel)
        return image[ch][iu]
    elif channel in LOWER_CHANNELS:
        ch = LOWER_CHANNELS.index(channel)
        return image[ch][il]
    else:
        raise KeyError(f'unknown channel {channel!r}')


def channel_histogram(
    image: np.ndarray,
    channel: str,
    spec: HistogramSpec | None = None,
) -> np.ndarray:
    """Compute a normalised histogram for one named channel.

    Returns a ``(n_bins,)`` array summing to 1 (unless the channel is
    empty, in which case it sums to 0).
    """
    spec = spec or DEFAULT_SPECS[channel]
    vals = _channel_values(image, channel).astype(np.float64, copy=False)
    if spec.log_scale:
        vals = np.log1p(np.clip(vals, 0.0, None))
        # Bin edges must live in the same log1p-transformed space.
        # Before this fix, edges were linspace(0, range_hi) while transformed
        # values only reach log1p(range_hi) ≈ 2.56, so all mass piled into
        # the first ~21% of bins.
        edges = np.linspace(0.0, np.log1p(spec.range_hi), spec.n_bins + 1)
    else:
        edges = np.linspace(spec.range_lo, spec.range_hi, spec.n_bins + 1)
    h, _ = np.histogram(vals, bins=edges)
    s = h.sum()
    return (h / s) if s > 0 else h.astype(np.float64)


def all_histograms(image: np.ndarray) -> Dict[str, np.ndarray]:
    """Compute the full set of per-channel histograms for one image."""
    return {
        ch: channel_histogram(image, ch, DEFAULT_SPECS[ch])
        for ch in UPPER_CHANNELS + LOWER_CHANNELS
    }
