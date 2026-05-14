"""Losses for v2 metric learning.

Two losses are provided:

* :class:`SupConLoss` — Khosla et al. 2020 supervised contrastive loss.
  Pulls samples sharing a SCOPe label together in embedding space and
  pushes the rest apart. We use it at fold level by default.
* :func:`triplet_with_mahalanobis` — triplet margin loss using a learned
  Mahalanobis distance ``d²(e, e') = (e − e')ᵀ Lᵀ L (e − e')``. Trains
  the metric head described in §4.3 of the design doc.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class SupConLoss(nn.Module):
    """Supervised contrastive loss (NT-Xent variant).

    Parameters
    ----------
    temperature : float
        Softmax temperature. Lower values sharpen the contrast.
    """

    def __init__(self, temperature: float = 0.1):
        super().__init__()
        self.t = temperature

    def forward(self, features: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """Compute SupCon loss.

        Parameters
        ----------
        features : (B, D) — already L2-normalised embeddings.
        labels   : (B,)   — integer class labels.
        """
        if features.dim() != 2:
            raise ValueError(f'expected (B, D) features; got {tuple(features.shape)}')
        b = features.size(0)
        device = features.device

        # cosine similarity matrix scaled by temperature
        sim = features @ features.T / self.t
        # mask: 1 where labels match, excluding the diagonal
        labels = labels.contiguous().view(-1, 1)
        positives = (labels == labels.T).float().to(device)
        diag = torch.eye(b, device=device)
        positives = positives - diag                # remove self-pairs
        positives = positives.clamp(min=0.0)

        # numerical stability: subtract row-max before exp
        sim = sim - sim.max(dim=1, keepdim=True).values.detach()
        exp_sim = torch.exp(sim) * (1.0 - diag)     # exclude self-pairs
        log_prob = sim - torch.log(exp_sim.sum(dim=1, keepdim=True) + 1e-12)

        pos_count = positives.sum(dim=1).clamp(min=1.0)
        loss = -(positives * log_prob).sum(dim=1) / pos_count

        # Rows with zero positives contribute 0 (singleton classes in batch)
        valid = (positives.sum(dim=1) > 0).float()
        if valid.sum() == 0:
            return torch.zeros((), device=device, requires_grad=True)
        return (loss * valid).sum() / valid.sum()


def triplet_with_mahalanobis(
    anchor: torch.Tensor,
    positive: torch.Tensor,
    negative: torch.Tensor,
    L: nn.Parameter,
    margin: float = 0.5,
) -> torch.Tensor:
    """Triplet margin loss under a learned Mahalanobis metric.

    ``L`` is a ``(k, D)`` parameter; the implied PSD metric matrix is
    ``M = Lᵀ L`` and ``d²(a, b) = || L (a - b) ||²``.
    """
    da = ((anchor - positive) @ L.T).pow(2).sum(dim=1)
    dn = ((anchor - negative) @ L.T).pow(2).sum(dim=1)
    return F.relu(da - dn + margin).mean()
