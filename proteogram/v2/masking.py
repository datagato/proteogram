"""Neutral-value masking helpers shared by the attribution tools.

Several explainability methods in this package work by removing information
from a proteogram and measuring what happens to the similarity score:

- ``shapley.py``   forms coalitions by masking everything outside the coalition
- ``scripts/v2/validate_attribution_faithfulness.py`` builds deletion and
  insertion curves by masking progressively larger region sets

They all need the same notion of "remove this region", so it lives here once.

What "removed" means
--------------------
Regions are replaced with :data:`~proteogram.v2.gradcam.NEUTRAL_PIXEL_VALUE`
(128), the same sentinel used as the pad fill in ``gradcam._pad_to_size`` and
written for structural zeros by ``normalisation.ZERO_FILL_VALUE``.  Using the
sentinel rather than 0 matters: 0 is a meaningful (very low) energy, whereas
128 is what the pipeline already writes where there is genuinely nothing.

Because a proteogram pixel (i, j) encodes the interaction between residues i
and j, "masking residue i" means masking row i *and* column i together.
"""

from __future__ import annotations

from typing import Iterable, Optional, Sequence

import numpy as np
import torch
from PIL import Image
from torchvision import transforms as T

from .gradcam import (
    IMAGENET_MEAN,
    IMAGENET_STD,
    NEUTRAL_PIXEL_VALUE,
    _pad_to_size,
)


_TO_TENSOR = T.Compose([
    T.ToTensor(),
    T.Normalize(mean=list(IMAGENET_MEAN), std=list(IMAGENET_STD)),
])


# ---------------------------------------------------------------------------
# Loading and embedding
# ---------------------------------------------------------------------------

def load_padded_array(path: str, size: int = 200) -> np.ndarray:
    """Load a proteogram JPG as the padded ``uint8`` array the model sees.

    Args:
        path: Path to the proteogram JPG.
        size: Target square size (default 200, matching the training pipeline).

    Returns:
        ``uint8`` array of shape ``(size, size, 3)``.
    """
    return _pad_to_size(Image.open(path), target=size)


def embed_array(embed_net, arr: np.ndarray, device) -> torch.Tensor:
    """Embed a ``uint8`` ``(H, W, 3)`` proteogram array.

    Args:
        embed_net: Embedding network (classification head removed), in eval mode.
        arr:       ``uint8`` proteogram array.
        device:    Torch device.

    Returns:
        Embedding tensor of shape ``(1, d)``.
    """
    tensor = _TO_TENSOR(Image.fromarray(arr)).unsqueeze(0).to(device).float()
    with torch.no_grad():
        return embed_net(tensor)


def cosine(a: torch.Tensor, b: torch.Tensor) -> float:
    """Cosine similarity between two ``(1, d)`` embedding tensors."""
    return float(torch.nn.functional.cosine_similarity(a, b, dim=1).item())


# ---------------------------------------------------------------------------
# Masking
# ---------------------------------------------------------------------------

def neutral_like(arr: np.ndarray) -> np.ndarray:
    """Return an all-neutral array with the same shape/dtype as ``arr``."""
    return np.full_like(arr, NEUTRAL_PIXEL_VALUE)


def active_residues(arr: np.ndarray) -> np.ndarray:
    """Return indices of residues that actually carry signal.

    A proteogram for an N-residue protein is padded up to 200x200 with the
    neutral value, so rows beyond N are empty.  Treating those as players in a
    Shapley computation wastes forward passes on regions that cannot change the
    score, so callers use this to restrict the player set to real residues.

    Args:
        arr: ``uint8`` proteogram array ``(H, W, 3)``.

    Returns:
        Integer array of residue indices whose row or column is not entirely
        neutral.
    """
    non_neutral = (arr != NEUTRAL_PIXEL_VALUE).any(axis=2)   # (H, W)
    return np.flatnonzero(non_neutral.any(axis=1) | non_neutral.any(axis=0))


def mask_residues(arr: np.ndarray, residues: Iterable[int],
                  keep: bool = False) -> np.ndarray:
    """Mask (or keep only) whole residues — row i plus column i.

    Args:
        arr:      ``uint8`` proteogram ``(H, W, 3)``.
        residues: Residue indices to act on.
        keep:     If ``False`` (deletion), the listed residues are set to the
                  neutral value.  If ``True`` (insertion), everything *except*
                  the listed residues is neutral.

    Returns:
        A new ``uint8`` array; ``arr`` is not modified.
    """
    if keep:
        out = neutral_like(arr)
        for i in residues:
            out[i, :, :] = arr[i, :, :]
            out[:, i, :] = arr[:, i, :]
        return out

    out = arr.copy()
    for i in residues:
        out[i, :, :] = NEUTRAL_PIXEL_VALUE
        out[:, i, :] = NEUTRAL_PIXEL_VALUE
    return out


def mask_pixels(arr: np.ndarray, flat_indices: Sequence[int],
                keep: bool = False) -> np.ndarray:
    """Mask (or keep only) individual pixels, given flat ``i * W + j`` indices."""
    n = arr.shape[0]
    idx = np.asarray(list(flat_indices), dtype=np.int64)
    if keep:
        out = neutral_like(arr)
        if idx.size:
            out[idx // n, idx % n, :] = arr[idx // n, idx % n, :]
        return out

    out = arr.copy()
    if idx.size:
        out[idx // n, idx % n, :] = NEUTRAL_PIXEL_VALUE
    return out


def mask_channels(arr: np.ndarray, channels: Iterable[int],
                  keep: bool = False) -> np.ndarray:
    """Mask (or keep only) whole colour channels.

    Args:
        arr:      ``uint8`` proteogram ``(H, W, 3)``.
        channels: Channel indices to act on.
        keep:     If ``False``, the listed channels become neutral.  If
                  ``True``, every channel *except* those listed becomes neutral.

    Returns:
        A new ``uint8`` array.
    """
    channels = set(int(c) for c in channels)
    out = arr.copy()
    for k in range(arr.shape[2]):
        drop = (k not in channels) if keep else (k in channels)
        if drop:
            out[:, :, k] = NEUTRAL_PIXEL_VALUE
    return out


def apply_mask(arr: np.ndarray, regions, granularity: str,
               keep: bool = False) -> np.ndarray:
    """Dispatch to :func:`mask_residues` or :func:`mask_pixels`.

    Args:
        arr:         ``uint8`` proteogram ``(H, W, 3)``.
        regions:     Residue indices or flat pixel indices.
        granularity: ``"residue"`` or ``"pixel"``.
        keep:        Deletion (``False``) or insertion (``True``) semantics.
    """
    if granularity == 'residue':
        return mask_residues(arr, regions, keep=keep)
    return mask_pixels(arr, regions, keep=keep)


def residue_scores(attr_map: np.ndarray) -> np.ndarray:
    """Collapse an ``(H, W)`` pixel attribution map to a per-residue score.

    Residue i owns row i and column i, so its score is the mean attribution
    over both.
    """
    n = attr_map.shape[0]
    return np.array([
        (attr_map[i, :].sum() + attr_map[:, i].sum()) / (2 * n)
        for i in range(n)
    ])
