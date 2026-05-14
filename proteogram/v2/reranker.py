"""Pairwise cross-encoder re-ranker for v2 retrieval.

After the composite scorer in :mod:`proteogram.v2.metrics` returns the
top-K candidates, this module scores each pair ``(query, candidate)``
with a small CNN that sees both proteograms together as a 6-channel
input. The teacher signal during training is a structural-alignment
score (TM-score from GTalign / US-align in production; a synthetic
fold-overlap score in the smoke tests).
"""

from __future__ import annotations

from typing import Dict, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class PairwiseCrossEncoder(nn.Module):
    """Tiny pair-CNN: 6 input channels (q.RGB || t.RGB) → scalar score.

    Input shape: ``(B, 6, N, N)``.
    """

    def __init__(self, base_channels: int = 32):
        super().__init__()
        c = base_channels
        self.net = nn.Sequential(
            nn.Conv2d(6, c, kernel_size=3, padding=1),
            nn.BatchNorm2d(c),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(c, 2 * c, kernel_size=3, padding=1),
            nn.BatchNorm2d(2 * c),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(2 * c, 4 * c, kernel_size=3, padding=1),
            nn.BatchNorm2d(4 * c),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
        )
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(4 * c, c),
            nn.ReLU(inplace=True),
            nn.Linear(c, 1),
            nn.Sigmoid(),  # teacher TM-score is in [0, 1]
        )

    def forward(self, q: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        if q.shape != t.shape:
            raise ValueError(f'pair shape mismatch: {q.shape} vs {t.shape}')
        x = torch.cat([q, t], dim=1)
        z = self.net(x)
        return self.head(z).squeeze(-1)

    @torch.no_grad()
    def rerank(
        self,
        query_image: torch.Tensor,
        candidates: List[Tuple[str, torch.Tensor, float]],
    ) -> List[Tuple[str, float]]:
        """Score and re-sort the top-K candidates.

        Each candidate tuple is ``(key, image_tensor, prior_score)``.
        Returns ``[(key, refined_score)]`` sorted descending.
        """
        was_training = self.training
        self.eval()
        out: List[Tuple[str, float]] = []
        device = query_image.device
        for key, img, _prior in candidates:
            s = self.forward(
                query_image.unsqueeze(0).to(device),
                img.unsqueeze(0).to(device),
            ).item()
            out.append((key, float(s)))
        if was_training:
            self.train()
        out.sort(key=lambda kv: kv[1], reverse=True)
        return out


def distillation_loss(pred: torch.Tensor, teacher: torch.Tensor) -> torch.Tensor:
    """MSE distillation against a teacher score in [0, 1]."""
    return F.mse_loss(pred, teacher.clamp(0.0, 1.0))
