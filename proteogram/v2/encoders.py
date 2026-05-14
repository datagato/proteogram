"""Two-stream encoders for asymmetric Proteogram v2 images.

Proteogram v2 stacks physically distinct information in the two image
triangles (upper = MD-derived VdW + Cα distogram; lower = electrostatic +
hydrophobicity Δ). A standard image encoder treats both triangles as a
single tensor, so any pooling step blurs the asymmetry.

This module provides:

* :func:`split_triangles` — separate an N×N×3 tensor into upper / lower
  per-pixel masked tensors (shared diagonal kept on the upper side).
* :class:`GeM` — Generalized-Mean pooling with a learnable exponent ``p``.
  At ``p → 1`` this reduces to GAP; at ``p → ∞`` it reduces to max-pool.
* :class:`TriangleStreamEncoder` — two ResNet-18 backbones (separate
  weights, optional shared init), GeM pooling, and a projection head per
  stream that produces L2-normalised embeddings ``e_U`` and ``e_L``.

The encoder also exposes :meth:`embed` returning a dict that matches the
on-disk schema documented in ``docs/v2_similarity_design.md`` §3.1.
"""

from __future__ import annotations

from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as tv_models


# ---------------------------------------------------------------------------
# Triangle decomposition
# ---------------------------------------------------------------------------

def split_triangles(x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Split an asymmetric proteogram into per-triangle tensors.

    Parameters
    ----------
    x : torch.Tensor
        Shape ``(B, 3, N, N)``. The upper and lower triangle of each
        channel encode different physical quantities.

    Returns
    -------
    (U, L) : Tuple[torch.Tensor, torch.Tensor]
        Both shape ``(B, 3, N, N)``. ``U`` keeps the upper triangle and the
        diagonal; ``L`` keeps the strictly lower triangle. The "other"
        side is set to zero in each tensor so a convolution receives a
        physically-valid input rather than a mix of channels.
    """
    if x.dim() != 4:
        raise ValueError(f'split_triangles expects (B, C, N, N); got {tuple(x.shape)}')
    n = x.size(-1)
    if x.size(-2) != n:
        raise ValueError(f'split_triangles expects square spatial dims; got {tuple(x.shape)}')
    # `triu` keeps diagonal; `tril(diagonal=-1)` strips it. We keep the
    # diagonal on the upper side so it isn't double-counted.
    upper_mask = torch.triu(torch.ones(n, n, device=x.device, dtype=x.dtype))
    lower_mask = torch.tril(torch.ones(n, n, device=x.device, dtype=x.dtype), diagonal=-1)
    return x * upper_mask, x * lower_mask


# ---------------------------------------------------------------------------
# Generalized-Mean pooling
# ---------------------------------------------------------------------------

class GeM(nn.Module):
    """Generalized-Mean pooling with a learnable exponent.

    Reduces a feature map ``(B, C, H, W)`` to ``(B, C)`` via
    ``(mean(x^p))^(1/p)``. Falls back to GAP at p=1, max-pool at p=∞.
    """

    def __init__(self, p: float = 3.0, eps: float = 1e-6, learnable: bool = True):
        super().__init__()
        init = torch.full((1,), float(p))
        self.p = nn.Parameter(init) if learnable else init
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.clamp(min=self.eps).pow(self.p)
        x = F.adaptive_avg_pool2d(x, 1).pow(1.0 / self.p)
        return x.flatten(1)

    def extra_repr(self) -> str:
        p_val = float(self.p.detach().cpu().item()) if isinstance(self.p, nn.Parameter) else float(self.p.item())
        return f'p={p_val:.3f}, learnable={isinstance(self.p, nn.Parameter)}'


# ---------------------------------------------------------------------------
# Two-stream encoder
# ---------------------------------------------------------------------------

def _resnet18_trunk() -> Tuple[nn.Module, int]:
    """Return a ResNet-18 backbone with the avgpool + fc removed.

    Output is a feature map ``(B, 512, H', W')`` ready for GeM pooling.
    """
    net = tv_models.resnet18(weights=None)
    feat_dim = net.fc.in_features  # 512 for ResNet-18
    trunk = nn.Sequential(*list(net.children())[:-2])  # strip avgpool + fc
    return trunk, feat_dim


class TriangleStreamEncoder(nn.Module):
    """Two-backbone encoder operating on upper and lower triangles.

    Parameters
    ----------
    emb_dim : int
        Output embedding dimensionality per stream. Joint embedding has
        size ``2 * emb_dim``.
    num_classes : int, optional
        If set, an auxiliary classification head is added on top of the
        joint embedding for the cross-entropy term in the loss.
    share_init : bool
        If True, both streams start from identical weights (a copy of one
        ResNet-18 init). Useful early in training for stability; the two
        streams diverge as they specialise.
    """

    def __init__(
        self,
        emb_dim: int = 256,
        num_classes: int | None = None,
        share_init: bool = True,
    ):
        super().__init__()
        trunk_u, feat_dim = _resnet18_trunk()
        trunk_l, _ = _resnet18_trunk()
        if share_init:
            trunk_l.load_state_dict(trunk_u.state_dict())

        self.backbone_U = trunk_u
        self.backbone_L = trunk_l
        self.gem_U = GeM(p=3.0)
        self.gem_L = GeM(p=3.0)
        self.proj_U = nn.Sequential(
            nn.Linear(feat_dim, emb_dim),
            nn.BatchNorm1d(emb_dim),
        )
        self.proj_L = nn.Sequential(
            nn.Linear(feat_dim, emb_dim),
            nn.BatchNorm1d(emb_dim),
        )

        self.classifier = (
            nn.Linear(2 * emb_dim, num_classes) if num_classes is not None else None
        )

        self.emb_dim = emb_dim

    # -- forward paths ------------------------------------------------------

    def _embed_stream(
        self, trunk: nn.Module, gem: nn.Module, proj: nn.Sequential, x: torch.Tensor
    ) -> torch.Tensor:
        feat = trunk(x)
        pooled = gem(feat)
        e = proj(pooled)
        return F.normalize(e, dim=1)

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Run both streams and return a dict of tensors.

        Returns keys: ``e_U`` (B, emb), ``e_L`` (B, emb), ``e`` (B, 2*emb),
        and ``logits`` (B, num_classes) if a classifier was configured.
        """
        U, L = split_triangles(x)
        e_U = self._embed_stream(self.backbone_U, self.gem_U, self.proj_U, U)
        e_L = self._embed_stream(self.backbone_L, self.gem_L, self.proj_L, L)
        e = torch.cat([e_U, e_L], dim=1)
        out = {'e_U': e_U, 'e_L': e_L, 'e': e}
        if self.classifier is not None:
            out['logits'] = self.classifier(e)
        return out

    @torch.no_grad()
    def embed(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Inference helper. Same as :meth:`forward` but without grads."""
        was_training = self.training
        self.eval()
        out = self.forward(x)
        if was_training:
            self.train()
        return out
