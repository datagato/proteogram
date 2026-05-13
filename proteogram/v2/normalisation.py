"""Global percentile normalisation for proteogram energy maps.

This module is fully self-contained.  It provides:

- ``CHANNEL_NAMES``       — canonical ordered list of the 6 energy channels.
- ``normalize_map_global`` — clip-and-scale a map to [0, 255] using
                             corpus-level percentile bounds.
- ``normalize_map_perprotein`` — original per-protein min-max (kept here so
                             both strategies live in one place).
- ``load_norm_stats``     — read a ``norm_stats.json`` produced by
                             ``scripts/v2/compute_norm_stats.py``.
- ``save_norm_stats``     — write a stats dict to JSON (used by the compute
                             script and in tests).

Typical usage
-------------
>>> from proteogram.v2.normalisation import normalize_map_global, load_norm_stats
>>> stats = load_norm_stats("/path/to/norm_stats.json")
>>> normed, err = normalize_map_global(vdw_matrix, stats["vdw_attractive"])
"""

from __future__ import annotations

import json
import os
from typing import Dict, Optional, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

#: Canonical channel names — must match the keys used in ``norm_stats.json``.
CHANNEL_NAMES: Tuple[str, ...] = (
    "vdw_attractive",
    "vdw_repulsive",
    "es_attractive",
    "es_repulsive",
    "distance",
    "hydrophobicity",
)

#: Sentinel pixel value written into structurally-zero cells (lower-triangle
#: fill) so they don't contaminate the energy-scale endpoints.
ZERO_FILL_VALUE: int = 128


# ---------------------------------------------------------------------------
# Stats I/O
# ---------------------------------------------------------------------------

def load_norm_stats(json_path: str) -> Dict[str, Dict[str, float]]:
    """Load normalisation bounds from a JSON file.

    The JSON must contain one key per channel (see ``CHANNEL_NAMES``), each
    mapping to a dict with ``"p_low"`` and ``"p_high"`` float values.  An
    optional ``"_meta"`` key is ignored.

    Args:
        json_path: Path to ``norm_stats.json``.

    Returns:
        Dict mapping channel name → ``{"p_low": float, "p_high": float}``.

    Raises:
        FileNotFoundError: If ``json_path`` does not exist.
        KeyError:          If a required channel is absent from the file.
    """
    if not os.path.exists(json_path):
        raise FileNotFoundError(f"norm_stats.json not found: {json_path}")
    with open(json_path) as fh:
        stats = json.load(fh)

    missing = [ch for ch in CHANNEL_NAMES if ch not in stats]
    if missing:
        raise KeyError(
            f"norm_stats.json is missing channels: {missing}.  "
            f"Re-run compute_norm_stats.py to regenerate the file."
        )
    return stats


def save_norm_stats(stats: dict, json_path: str) -> None:
    """Write normalisation bounds to a JSON file.

    Args:
        stats:     Dict mapping channel → ``{"p_low": …, "p_high": …}``.
        json_path: Destination path.
    """
    os.makedirs(os.path.dirname(os.path.abspath(json_path)), exist_ok=True)
    with open(json_path, "w") as fh:
        json.dump(stats, fh, indent=2)
    print(f"Saved norm stats → {json_path}")


# ---------------------------------------------------------------------------
# Normalisation functions
# ---------------------------------------------------------------------------

def normalize_map_global(
    arr: np.ndarray,
    channel_stats: Dict[str, float],
) -> Tuple[np.ndarray, str]:
    """Normalise an energy/property map to [0, 255] using corpus-level bounds.

    Unlike per-protein min-max normalisation, this preserves the inter-protein
    energy scale: a weakly-packed protein genuinely maps to lower pixel values
    than a tightly-packed one.

    Structural zero cells (the unfilled lower triangle, stored as exact 0 in
    the raw matrices) are mapped to ``ZERO_FILL_VALUE`` (128, mid-grey) so
    they are visually neutral and do not contaminate the low-energy range.

    Args:
        arr:            Input energy/property matrix (upper triangle populated;
                        lower triangle zeros).
        channel_stats:  Dict with ``"p_low"`` and ``"p_high"`` keys (physical
                        units: kJ/mol, Å, or dimensionless hydrophobicity).

    Returns:
        Tuple of:
        - ``uint8`` array normalised to [0, 255].
        - Error string, or ``""`` if successful.
    """
    err = ""
    try:
        p_low  = float(channel_stats["p_low"])
        p_high = float(channel_stats["p_high"])
        scale  = p_high - p_low

        if scale == 0:
            return np.full_like(arr, ZERO_FILL_VALUE, dtype="uint8"), "zero scale range"

        # Record which cells are structural zeros before any arithmetic
        zero_mask = arr == 0

        # Clip to [p_low, p_high] then scale linearly to [0, 255]
        clipped    = np.clip(arr, p_low, p_high)
        normalised = ((clipped - p_low) / scale * 255).astype("uint8")

        # Remap structural zeros to mid-grey sentinel
        normalised[zero_mask] = ZERO_FILL_VALUE

    except Exception as exc:
        err = f"normalize_map_global failed: {exc}"
        normalised = np.full_like(arr, ZERO_FILL_VALUE, dtype="uint8")

    return normalised, err


def normalize_map_perprotein(arr: np.ndarray) -> Tuple[np.ndarray, str]:
    """Normalise a map to [0, 255] using its own min/max (original v2 strategy).

    Kept here alongside ``normalize_map_global`` so both strategies are in one
    module.  The ``ProteogramV2.normalize_map`` static method delegates here.

    Args:
        arr: Input array.

    Returns:
        Tuple of (normalised ``uint8`` array, error string).
    """
    err = ""
    try:
        arr_min, arr_max = arr.min(), arr.max()
        scale = arr_max - arr_min
        if scale == 0:
            return np.full_like(arr, 0, dtype="uint8"), "zero scale range (per-protein)"
        arr = ((arr - arr_min) / scale * 255).astype("uint8")
    except Exception as exc:
        err = f"normalize_map_perprotein failed: {exc}"
        arr = np.zeros_like(arr, dtype="uint8")
    return arr, err


# ---------------------------------------------------------------------------
# Convenience dispatcher used by ProteogramV2.calculate_proteogram()
# ---------------------------------------------------------------------------

def normalise_channel(
    arr: np.ndarray,
    channel_name: str,
    norm_stats: Optional[Dict[str, Dict[str, float]]],
) -> Tuple[np.ndarray, str]:
    """Normalise one channel using global stats when available, per-protein otherwise.

    Args:
        arr:          Raw energy/property matrix.
        channel_name: Key into ``norm_stats`` (one of ``CHANNEL_NAMES``).
        norm_stats:   Loaded norm_stats dict, or ``None`` for per-protein mode.

    Returns:
        Tuple of (normalised ``uint8`` array, error string).
    """
    if norm_stats is not None and channel_name in norm_stats:
        return normalize_map_global(arr, norm_stats[channel_name])
    return normalize_map_perprotein(arr)
