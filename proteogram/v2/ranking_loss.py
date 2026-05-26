"""Physics-informed ranking loss using USalign TM-scores as ground truth.

Replaces/supplements CrossEntropyLoss with a ListNet-style loss that directly
optimises the rank order of cosine similarities to match structural TM-score
rankings — bypassing the intermediate classification objective that causes
performance to collapse below the class level.

Motivation
----------
CrossEntropyLoss with SCOP class labels gives the model gradient signal only
to separate 7 broad classes.  The embeddings learn "is this an all-alpha or
all-beta protein?" rather than "how similar are these two structures?"  The
ranking loss closes this gap: for every pair in a batch, the model is told
the exact TM-score from USalign, and is trained so that its cosine similarities
respect the same order.

Loss formulation (ListNet)
--------------------------
For each query protein i in a batch of N:
    gt_dist[j]   = softmax(TM_score(i, j) / temperature)   for j ≠ i
    pred_logits  = cosine_sim(embed_i, embed_j)             for j ≠ i
    loss_i       = -sum(gt_dist * log_softmax(pred_logits))

Total loss = mean(loss_i) over i where at least one TM-score is known.

Combined training loss = α * CrossEntropy + β * ListNet

Why ListNet?
  - Differentiable: no sort operations.
  - Directly minimises KL(TM-score distribution || cosine-sim distribution).
  - Gracefully handles missing pairs: any (i, j) without a known TM-score is
    assigned a neutral score (0.5) and down-weighted by a confidence mask.

Usage
-----
    from proteogram.v2.ranking_loss import TmScoreRankingLoss
    ranking_loss = TmScoreRankingLoss("usalign_out_544.tsv")
    loss = ranking_loss.listnet_loss(embeddings, protein_ids)
"""

from __future__ import annotations

import os
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# TM-score store
# ---------------------------------------------------------------------------

class TmScoreStore:
    """In-memory lookup for pairwise USalign TM-scores.

    Loads a USalign all-vs-all TSV once; subsequent lookups are O(1) dict
    accesses.  Missing pairs return ``None`` so callers can build confidence
    masks.

    Args:
        usalign_tsv: Path to USalign output TSV.  Expected columns:
            ``#PDBchain1``, ``PDBchain2``, ``TM1``, ``TM2``.
            IDs are extracted as the filename stem before the first ``:``
            and ``.ent`` extension, matching the convention in
            ``evaluate_methods_v2.py``.
        symmetrise: Average TM1 and TM2 for each pair so that
            score(A, B) == score(B, A).  Defaults to ``True``.
    """

    def __init__(self, usalign_tsv: str, symmetrise: bool = True) -> None:
        self._scores: Dict[Tuple[str, str], float] = {}
        self._load(usalign_tsv, symmetrise)

    # ------------------------------------------------------------------
    def _extract_id(self, raw: str) -> str:
        stem = os.path.splitext(os.path.basename(raw).split(':')[0])[0]
        return stem

    def _load(self, path: str, symmetrise: bool) -> None:
        df = pd.read_csv(path, sep='\t')
        for _, row in df.iterrows():
            id1 = self._extract_id(str(row['#PDBchain1']))
            id2 = self._extract_id(str(row['PDBchain2']))
            tm1 = float(row['TM1'])
            tm2 = float(row['TM2'])
            if symmetrise:
                avg = (tm1 + tm2) / 2.0
                self._scores[(id1, id2)] = avg
                self._scores[(id2, id1)] = avg
            else:
                self._scores[(id1, id2)] = tm1
                self._scores[(id2, id1)] = tm2
        print(f"[TmScoreStore] Loaded {len(self._scores) // 2} unique pairs "
              f"from {os.path.basename(path)}")

    def get(self, id1: str, id2: str) -> Optional[float]:
        return self._scores.get((id1, id2))

    def coverage(self, ids) -> float:
        """Fraction of all pairs in ``ids`` that have a known TM-score."""
        n = len(ids)
        if n < 2:
            return 0.0
        known = sum(1 for i in range(n) for j in range(n)
                    if i != j and (ids[i], ids[j]) in self._scores)
        return known / (n * (n - 1))


# ---------------------------------------------------------------------------
# Ranking loss
# ---------------------------------------------------------------------------

class TmScoreRankingLoss:
    """ListNet ranking loss driven by USalign TM-scores.

    Args:
        usalign_tsv:      Path to USalign TSV (or pre-built ``TmScoreStore``).
        temperature:      Softmax temperature for GT TM-score distribution.
                          Lower values create sharper targets (default 0.1).
        missing_score:    TM-score assigned to pairs not in the USalign file
                          (default 0.5 = structurally ambiguous).
        missing_weight:   Gradient weight for missing-pair positions relative
                          to known pairs (default 0.0 = ignore missing pairs).
    """

    def __init__(
        self,
        usalign_tsv,
        temperature: float = 0.1,
        missing_score: float = 0.5,
        missing_weight: float = 0.0,
    ) -> None:
        if isinstance(usalign_tsv, TmScoreStore):
            self.store = usalign_tsv
        else:
            self.store = TmScoreStore(usalign_tsv)
        self.temperature = temperature
        self.missing_score = missing_score
        self.missing_weight = missing_weight

    # ------------------------------------------------------------------
    # Batch matrix helpers
    # ------------------------------------------------------------------

    def _build_tm_matrix(
        self, protein_ids: list
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return (tm_matrix, confidence_mask) for a batch.

        Returns:
            tm_matrix:       (N, N) float tensor.  Diagonal = 1.0 (self-match).
            confidence_mask: (N, N) float tensor.  1.0 for known pairs,
                             ``missing_weight`` for unknown pairs, 0.0 on diagonal.
        """
        N = len(protein_ids)
        tm = torch.full((N, N), self.missing_score)
        conf = torch.full((N, N), self.missing_weight)

        for i in range(N):
            tm[i, i] = 1.0
            conf[i, i] = 0.0
            for j in range(N):
                if i == j:
                    continue
                score = self.store.get(protein_ids[i], protein_ids[j])
                if score is not None:
                    tm[i, j] = score
                    conf[i, j] = 1.0

        return tm, conf

    # ------------------------------------------------------------------
    # ListNet loss (primary)
    # ------------------------------------------------------------------

    def listnet_loss(
        self,
        embeddings: torch.Tensor,
        protein_ids: list,
    ) -> torch.Tensor:
        """Compute ListNet ranking loss for a batch.

        Args:
            embeddings:  (N, d) embedding tensor with gradient.
            protein_ids: List of N SCOPe domain IDs (e.g. ``'d1a3wa3'``).

        Returns:
            Scalar loss tensor (0.0 if no known pairs exist in the batch).
        """
        device = embeddings.device
        N = embeddings.size(0)

        tm_matrix, conf_mask = self._build_tm_matrix(protein_ids)
        tm_matrix = tm_matrix.to(device)
        conf_mask = conf_mask.to(device)

        # Predicted cosine similarity matrix (N x N)
        emb_norm = F.normalize(embeddings, dim=1)
        cos_sim = emb_norm @ emb_norm.T          # (N, N), grad flows through

        losses = []
        for i in range(N):
            # Off-diagonal indices
            off_diag = [j for j in range(N) if j != i]
            if not off_diag:
                continue

            gt_scores  = tm_matrix[i, off_diag]   # (N-1,)
            weights    = conf_mask[i, off_diag]    # (N-1,)
            pred_logits = cos_sim[i, off_diag]     # (N-1,) with grad

            if weights.sum() == 0:
                continue

            # GT probability distribution over candidates
            gt_dist = F.softmax(gt_scores / self.temperature, dim=0)

            # Predicted log-probability distribution
            log_pred = F.log_softmax(pred_logits, dim=0)

            # Weighted KL-divergence (cross-entropy term)
            ce = -(gt_dist * log_pred * weights).sum() / weights.sum()
            losses.append(ce)

        if not losses:
            return torch.tensor(0.0, device=device, requires_grad=True)
        return torch.stack(losses).mean()

    # ------------------------------------------------------------------
    # Pairwise MSE (simpler baseline, useful for ablation)
    # ------------------------------------------------------------------

    def pairwise_mse_loss(
        self,
        embeddings: torch.Tensor,
        protein_ids: list,
    ) -> torch.Tensor:
        """Pairwise regression loss: minimize (cosine_sim - TM_score)².

        Args:
            embeddings:  (N, d) embedding tensor with gradient.
            protein_ids: List of N SCOPe domain IDs.

        Returns:
            Scalar MSE loss tensor (0.0 if no known pairs exist).
        """
        device = embeddings.device

        tm_matrix, conf_mask = self._build_tm_matrix(protein_ids)
        tm_matrix = tm_matrix.to(device)
        conf_mask = conf_mask.to(device)

        emb_norm = F.normalize(embeddings, dim=1)
        cos_sim = emb_norm @ emb_norm.T

        # Only known off-diagonal pairs
        mask = conf_mask.bool()
        diag = torch.eye(len(protein_ids), dtype=torch.bool, device=device)
        mask = mask & ~diag

        if mask.sum() == 0:
            return torch.tensor(0.0, device=device, requires_grad=True)

        pred = cos_sim[mask]
        gt   = tm_matrix[mask]
        return F.mse_loss(pred, gt)

    # ------------------------------------------------------------------
    # Batch coverage reporting (for logging)
    # ------------------------------------------------------------------

    def batch_coverage(self, protein_ids: list) -> float:
        """Fraction of off-diagonal pairs with known TM-scores."""
        return self.store.coverage(protein_ids)
