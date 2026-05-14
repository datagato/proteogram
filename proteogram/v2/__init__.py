# Heavy domain modules (Biopython, OpenMM, PIL etc.) are optional. The
# metric-learning extensions added per docs/v2_similarity_design.md work
# fine without them; this guard keeps the package importable in slimmer
# environments (CI, smoke tests).
try:
    from .proteogram import ProteogramV2  # noqa: F401
except Exception:  # pragma: no cover
    ProteogramV2 = None  # type: ignore[assignment]

try:
    from .image_similarity import Img2Vec  # noqa: F401
except Exception:  # pragma: no cover
    Img2Vec = None  # type: ignore[assignment]

try:
    from .nonbonded_forces import NonBondedForceModel  # noqa: F401
except Exception:  # pragma: no cover
    NonBondedForceModel = None  # type: ignore[assignment]

# v2 metric-learning extensions (added per docs/v2_similarity_design.md).
# These are intentionally optional — the heavy modules import torch and
# torchvision lazily so the legacy import path keeps working in
# environments where only the proteogram-creation deps are installed.
try:
    from .encoders import TriangleStreamEncoder, GeM, split_triangles  # noqa: F401
    from .losses import SupConLoss, triplet_with_mahalanobis  # noqa: F401
except Exception:  # pragma: no cover - torch optional at import time
    pass

from .features import all_histograms, channel_histogram, DEFAULT_SPECS  # noqa: F401
from .metrics import (  # noqa: F401
    CompositeScorer,
    MahalanobisRBF,
    cos_split,
    emd_1d,
    emd_score,
    precision_at_k,
    recall_at_k,
    average_precision,
    ndcg_at_k,
    reciprocal_rank,
)


__all__ = [
    'ProteogramV2',
    'Img2Vec',
    'NonBondedForceModel',
    # v2 metric-learning extensions
    'TriangleStreamEncoder', 'GeM', 'split_triangles',
    'SupConLoss', 'triplet_with_mahalanobis',
    'all_histograms', 'channel_histogram', 'DEFAULT_SPECS',
    'CompositeScorer', 'MahalanobisRBF',
    'cos_split', 'emd_1d', 'emd_score',
    'precision_at_k', 'recall_at_k', 'average_precision',
    'ndcg_at_k', 'reciprocal_rank',
]