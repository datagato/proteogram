"""Deterministic synthetic v2 proteograms.

Used for end-to-end smoke tests of the new metric-learning stack when no
SCOPe / MD data is available. We emit numpy arrays in the same channel
order as a real v2 proteogram (3 channels, asymmetric upper / lower
triangles) and a fold label per image. Same fold ⇒ same prototype with
small per-image jitter, so retrieval should be easy when the model has
learned anything at all.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Tuple

import numpy as np


@dataclass
class SyntheticConfig:
    n_folds: int = 12
    samples_per_fold: int = 14
    image_size: int = 32
    noise: float = 0.25            # heavy jitter; folds overlap somewhat
    confusable_pairs: int = 3      # number of fold pairs that share half a prototype
    seed: int = 0


def _make_prototype(rng: np.random.Generator, n: int) -> np.ndarray:
    """Build one (3, N, N) prototype with structured upper / lower halves."""
    img = np.zeros((3, n, n), dtype=np.float32)
    iu = np.triu_indices(n, k=1)
    il = np.tril_indices(n, k=-1)
    diag = np.arange(n)

    # ---- upper triangle: VdW(att/rep) + Cα distogram --------------------
    # Pick a few "contact clusters" and place gaussian bumps in the
    # attractive channel; correlate the repulsive channel inversely.
    for _ in range(rng.integers(2, 5)):
        i = int(rng.integers(0, n))
        j = int(rng.integers(0, n))
        if i == j:
            continue
        sigma = float(rng.uniform(1.5, 4.0))
        amp = float(rng.uniform(0.4, 1.0))
        ii, jj = np.meshgrid(np.arange(n), np.arange(n), indexing='ij')
        bump = amp * np.exp(-((ii - i) ** 2 + (jj - j) ** 2) / (2 * sigma ** 2))
        img[0] += bump
        img[1] -= 0.5 * bump  # opposite-sign Pauli-repulsion analogue
    # distogram: |i - j| with a fold-specific scale baked in via prototype caller
    ii, jj = np.meshgrid(np.arange(n), np.arange(n), indexing='ij')
    img[2] = np.abs(ii - jj).astype(np.float32) / n

    # zero the lower triangle of these channels (asymmetric layout!)
    for c in (0, 1, 2):
        img[c][il] = 0.0

    # ---- lower triangle: ES(att/rep) + Hyd Δ ---------------------------
    es_att = np.zeros((n, n), dtype=np.float32)
    es_rep = np.zeros((n, n), dtype=np.float32)
    hyd = np.zeros((n, n), dtype=np.float32)
    for _ in range(rng.integers(2, 5)):
        i = int(rng.integers(0, n))
        j = int(rng.integers(0, n))
        if i == j:
            continue
        sigma = float(rng.uniform(1.5, 4.0))
        amp = float(rng.uniform(0.4, 1.0))
        bump = amp * np.exp(-((ii - i) ** 2 + (jj - j) ** 2) / (2 * sigma ** 2))
        if rng.random() < 0.5:
            es_att += bump
        else:
            es_rep += bump
        hyd += 0.3 * bump
    # store in lower triangle
    img[0][il] = es_att[il]
    img[1][il] = es_rep[il]
    img[2][il] = hyd[il]

    img = np.clip(img, 0.0, None)
    return img


def make_dataset(cfg: SyntheticConfig) -> Tuple[np.ndarray, np.ndarray]:
    """Return ``(images, labels)``.

    ``images`` shape: ``(N_total, 3, H, W)``; ``labels`` shape ``(N_total,)``.
    """
    rng = np.random.default_rng(cfg.seed)
    n = cfg.image_size

    prototypes = [_make_prototype(rng, n) for _ in range(cfg.n_folds)]
    # Make a few fold pairs share their upper triangle (geometry) but
    # differ in the lower triangle (chemistry). Forces the chemistry
    # stream and the EMD term to actually pull weight.
    iu = np.triu_indices(n, k=1)
    diag = (np.arange(n), np.arange(n))
    for k in range(min(cfg.confusable_pairs, cfg.n_folds // 2)):
        a, b = 2 * k, 2 * k + 1
        for c in (0, 1, 2):
            prototypes[b][c][iu] = prototypes[a][c][iu]
            prototypes[b][c][diag] = prototypes[a][c][diag]
    images: List[np.ndarray] = []
    labels: List[int] = []
    for fold, proto in enumerate(prototypes):
        for _ in range(cfg.samples_per_fold):
            jitter = rng.normal(0.0, cfg.noise, size=proto.shape).astype(np.float32)
            sample = np.clip(proto + jitter, 0.0, None)
            images.append(sample)
            labels.append(fold)
    X = np.stack(images, axis=0)
    y = np.asarray(labels, dtype=np.int64)
    perm = rng.permutation(len(X))
    return X[perm], y[perm]


def to_torch_tensor(images: np.ndarray):
    """Lazy-import torch to avoid a hard dep at module import time."""
    import torch
    return torch.from_numpy(images.astype('float32'))
