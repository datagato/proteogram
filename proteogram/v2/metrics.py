"""Composite similarity metrics for v2 retrieval.

Implements the four similarity terms from §4 of the design doc plus a
linear combiner. Everything is pure numpy so the metric stack can be
exercised independently of the encoder (useful for tests, ablations,
and offline ranking studies).

Inputs are in this normalised form:

    record = {
        'e_U': (D,) float32,        # upper-triangle embedding (L2-norm)
        'e_L': (D,) float32,        # lower-triangle embedding (L2-norm)
        'e':   (2D,) float32,       # joint embedding [e_U ⊕ e_L]
        'H':   { channel: (B,) }    # per-channel histograms
    }
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# Term 1 / 2 — triangle-wise cosine
# ---------------------------------------------------------------------------

def cos_split(q: Dict, t: Dict) -> Tuple[float, float]:
    """Return ``(s_geom, s_chem) = (cos(e_U), cos(e_L))``."""
    return float(np.dot(q['e_U'], t['e_U'])), float(np.dot(q['e_L'], t['e_L']))


# ---------------------------------------------------------------------------
# Term 3 — RBF over learned Mahalanobis distance
# ---------------------------------------------------------------------------

@dataclass
class MahalanobisRBF:
    """RBF over a learned Mahalanobis distance.

    ``L`` is a ``(k, D)`` matrix; the implied PSD metric is ``M = Lᵀ L``
    and the kernel is ``exp(-γ · ||L (a − b)||²)``.

    Use :meth:`fit_diagonal_from_pairs` for a quick, gradient-free fit
    that scales each dimension by inverse within-class variance and
    re-normalises so the median squared distance = 1. This is
    intentionally lightweight; production code should fit by triplet
    loss against the encoder (see ``scripts/v2/train_metric_head.py``).
    """

    L: np.ndarray
    gamma: float = 1.0

    def distance_sq(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        diff = (a - b) @ self.L.T
        return np.einsum('...d,...d->...', diff, diff)

    def kernel(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        return np.exp(-self.gamma * self.distance_sq(a, b))

    @classmethod
    def fit_diagonal_from_pairs(
        cls,
        embeddings: np.ndarray,
        labels: np.ndarray,
    ) -> 'MahalanobisRBF':
        """Cheap closed-form fit. Diagonal L scaled by inverse class-variance."""
        d = embeddings.shape[1]
        within = np.zeros(d, dtype=np.float64)
        counts = 0
        for c in np.unique(labels):
            mask = labels == c
            if mask.sum() < 2:
                continue
            within += embeddings[mask].var(axis=0)
            counts += 1
        if counts == 0:
            within[:] = 1.0
        else:
            within = within / counts
        within = np.clip(within, 1e-6, None)
        L = np.diag(1.0 / np.sqrt(within)).astype(np.float32)
        # gamma tuned so median pairwise distance² = 1
        idx = np.random.RandomState(0).randint(0, len(embeddings), size=(min(2000, len(embeddings)), 2))
        a = embeddings[idx[:, 0]]
        b = embeddings[idx[:, 1]]
        diffs = (a - b) @ L.T
        med = np.median(np.sum(diffs * diffs, axis=1))
        gamma = 1.0 / max(med, 1e-6)
        return cls(L=L, gamma=float(gamma))


# ---------------------------------------------------------------------------
# Term 4 — Earth Mover's Distance over per-channel histograms
# ---------------------------------------------------------------------------

def emd_1d(p: np.ndarray, q: np.ndarray) -> float:
    """1-D Wasserstein distance between two histograms on the same grid.

    Closed form via the integrated CDF difference. Histograms must be
    on identical bin edges and ideally normalised.
    """
    if p.shape != q.shape:
        raise ValueError(f'shape mismatch {p.shape} vs {q.shape}')
    cp = np.cumsum(p)
    cq = np.cumsum(q)
    return float(np.sum(np.abs(cp - cq)))


def emd_score(q: Dict, t: Dict, channels: Sequence[str], kappa: float = 1.0) -> float:
    """``exp(-κ · Σ_c EMD(H_c(q), H_c(t)))`` over the named channels."""
    total = 0.0
    for ch in channels:
        total += emd_1d(q['H'][ch], t['H'][ch])
    return float(np.exp(-kappa * total))


# ---------------------------------------------------------------------------
# Composite scoring
# ---------------------------------------------------------------------------

@dataclass
class CompositeScorer:
    """Linear combiner over the four similarity terms.

    Defaults are equal weights for explainability; see
    :meth:`fit_grid_search` for the validation-set tuned variant.
    """

    weights: np.ndarray = field(default_factory=lambda: np.array([0.25, 0.25, 0.25, 0.25], dtype=np.float64))
    mahal: MahalanobisRBF | None = None
    emd_channels: Tuple[str, ...] = ('vdw_att', 'vdw_rep', 'es_att', 'es_rep')
    kappa: float = 1.0

    def components(self, q: Dict, t: Dict) -> Tuple[float, float, float, float]:
        s_geom, s_chem = cos_split(q, t)
        s_mahal = float(self.mahal.kernel(q['e'], t['e'])) if self.mahal is not None else 0.0
        s_emd = emd_score(q, t, self.emd_channels, kappa=self.kappa) if 'H' in q and 'H' in t else 0.0
        return s_geom, s_chem, s_mahal, s_emd

    def score(self, q: Dict, t: Dict) -> float:
        comps = np.array(self.components(q, t), dtype=np.float64)
        return float(np.dot(self.weights, comps))

    # -- search helpers ----------------------------------------------------

    def topk(
        self,
        query: Dict,
        candidates: Dict[str, Dict],
        k: int = 10,
    ) -> List[Tuple[str, float]]:
        scored = [(key, self.score(query, rec)) for key, rec in candidates.items()]
        scored.sort(key=lambda kv: kv[1], reverse=True)
        return scored[:k]

    # -- weight tuning -----------------------------------------------------

    @classmethod
    def fit_grid_search(
        cls,
        scorers_per_query: List[List[Tuple[float, float, float, float, int]]],
        step: float = 0.1,
        objective: str = 'recall@10',
        k: int = 10,
        mahal: MahalanobisRBF | None = None,
        emd_channels: Tuple[str, ...] = ('vdw_att', 'vdw_rep', 'es_att', 'es_rep'),
        kappa: float = 1.0,
    ) -> 'CompositeScorer':
        """Pick weights on the validation set by grid search.

        Each query yields a list of ``(s1, s2, s3, s4, relevant)`` tuples
        — one per candidate, with ``relevant ∈ {0, 1}``. Weights are
        searched on the simplex {w : w_i ≥ 0, Σ w_i = 1} at the given
        step, optimising the chosen ranking metric.
        """
        from itertools import product

        def _grid(step):
            n = int(round(1.0 / step))
            for w in product(range(n + 1), repeat=4):
                if sum(w) == n:
                    yield np.array(w, dtype=np.float64) / n

        best_w = None
        best_score = -np.inf
        for w in _grid(step):
            scores = []
            for cands in scorers_per_query:
                arr = np.array([row[:4] for row in cands], dtype=np.float64)
                rel = np.array([row[4] for row in cands], dtype=np.int32)
                s = arr @ w
                order = np.argsort(-s)
                top = rel[order][:k]
                if objective == 'recall@10':
                    denom = max(int(rel.sum()), 1)
                    scores.append(top.sum() / denom)
                elif objective == 'precision@1':
                    scores.append(float(top[0])) if len(top) else scores.append(0.0)
                else:
                    raise ValueError(f'unknown objective {objective!r}')
            mean = float(np.mean(scores))
            if mean > best_score:
                best_score = mean
                best_w = w
        return cls(
            weights=best_w if best_w is not None else np.array([0.25] * 4),
            mahal=mahal,
            emd_channels=emd_channels,
            kappa=kappa,
        )


# ---------------------------------------------------------------------------
# Ranking metrics
# ---------------------------------------------------------------------------

def precision_at_k(relevance: Sequence[int], k: int) -> float:
    rel = list(relevance)[:k]
    if not rel:
        return 0.0
    return sum(rel) / k


def recall_at_k(relevance: Sequence[int], total_relevant: int, k: int) -> float:
    if total_relevant <= 0:
        return 0.0
    return sum(list(relevance)[:k]) / total_relevant


def average_precision(relevance: Sequence[int]) -> float:
    rel = list(relevance)
    hits = 0
    s = 0.0
    for i, r in enumerate(rel, start=1):
        if r:
            hits += 1
            s += hits / i
    return s / hits if hits else 0.0


def ndcg_at_k(relevance: Sequence[int], k: int) -> float:
    rel = np.asarray(list(relevance)[:k], dtype=np.float64)
    if rel.size == 0 or rel.sum() == 0:
        return 0.0
    discounts = 1.0 / np.log2(np.arange(2, rel.size + 2))
    dcg = float((rel * discounts).sum())
    ideal = float(np.sort(rel)[::-1].dot(discounts))
    return dcg / ideal if ideal else 0.0


def reciprocal_rank(relevance: Sequence[int]) -> float:
    for i, r in enumerate(relevance, start=1):
        if r:
            return 1.0 / i
    return 0.0
