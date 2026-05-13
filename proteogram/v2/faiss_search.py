"""FAISS-based Approximate Nearest Neighbour index for proteogram embeddings.

This module is fully self-contained and has no dependency on Img2Vec or any other
proteogram class.  It operates on plain numpy float32 arrays and stores a key list
(filename strings) so integer FAISS indices can be mapped back to protein IDs.

Typical usage
-------------
>>> from proteogram.v2.faiss_search import FaissIndex
>>> import numpy as np

>>> # Build from a dict of {filename: embedding_tensor} (same format as Img2Vec.dataset)
>>> index = FaissIndex.from_dataset(img2vec.dataset)

>>> # All-vs-all search (returns same format as Img2Vec.sim_dict)
>>> sim_dict = index.search_all(top_k=5)

>>> # Single-query search
>>> hits = index.search_one(query_vec, top_k=5)

>>> # Persistence
>>> index.save("/path/to/corpus.faiss")
>>> index2 = FaissIndex.load("/path/to/corpus.faiss")
"""

from __future__ import annotations

import os
import pickle
from typing import Dict, List, Tuple

import numpy as np
import torch


# ---------------------------------------------------------------------------
# Public constants
# ---------------------------------------------------------------------------

#: Default lower percentile for nlist auto-selection.
_NLIST_SQRT_FACTOR: float = 1.0


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _l2_normalise(mat: np.ndarray) -> np.ndarray:
    """Return an L2-normalised copy of *mat* (shape N×d, float32).

    Does NOT modify the input array in-place so the caller's embeddings stay intact.
    """
    mat = mat.copy().astype(np.float32)
    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1.0, norms)   # avoid div-by-zero for zero vectors
    mat /= norms
    return mat


def _stack_dataset(dataset: Dict[str, torch.Tensor]) -> Tuple[List[str], np.ndarray]:
    """Convert an Img2Vec-style dataset dict to an ordered (keys, matrix) pair.

    Args:
        dataset: Mapping of filename → 1-D or 1×d embedding tensor.

    Returns:
        keys:   List of filenames in the same row order as the matrix.
        matrix: float32 numpy array of shape (N, d).
    """
    keys = list(dataset.keys())
    vecs = torch.cat([dataset[k].cpu().reshape(1, -1) for k in keys]).float().numpy()
    return keys, vecs


# ---------------------------------------------------------------------------
# FaissIndex
# ---------------------------------------------------------------------------

class FaissIndex:
    """Wraps a FAISS IVFFlat or IVF-PQ index with a key mapping.

    Parameters
    ----------
    keys:
        Ordered list of protein filenames.  ``keys[i]`` is the protein
        corresponding to FAISS integer index ``i``.
    vecs_norm:
        L2-normalised embedding matrix, shape (N, d), float32.  Stored so
        single-query searches can normalise the query the same way.
    index:
        A trained and populated FAISS index (inner-product metric).
    """

    def __init__(self,
                 keys: List[str],
                 vecs_norm: np.ndarray,
                 index) -> None:
        self.keys = keys
        self.vecs_norm = vecs_norm
        self._index = index

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    @classmethod
    def from_dataset(cls,
                     dataset: Dict[str, torch.Tensor],
                     use_pq: bool = False,
                     nlist: int = None,
                     nprobe: int = None,
                     pq_m: int = 8,
                     pq_nbits: int = 8) -> "FaissIndex":
        """Build a FAISS index from an Img2Vec-style embedding dataset.

        Embeddings are L2-normalised before indexing so inner-product search
        is equivalent to cosine similarity.

        Args:
            dataset:  ``{filename: embedding_tensor}`` dict (same as
                      ``Img2Vec.dataset``).
            use_pq:   Use IVF-PQ compressed index.  Recommended for corpora
                      larger than 100 K proteins.  Slightly lower recall but
                      4–32× lower memory.  Defaults to ``False`` (IVFFlat,
                      exact).
            nlist:    Number of Voronoi cells.  Defaults to
                      ``max(1, int(sqrt(N)))``.
            nprobe:   Cells searched per query.  Higher → better recall,
                      slower.  Defaults to ``max(1, nlist // 10)``.
            pq_m:     Sub-quantiser count for IVF-PQ.  Must divide ``d``
                      evenly.  Auto-reduced if necessary.
            pq_nbits: Bits per sub-quantiser (IVF-PQ only).  8 is standard.

        Returns:
            A trained and populated ``FaissIndex`` ready for search.

        Raises:
            ImportError: If ``faiss`` is not installed.
            ValueError:  If ``dataset`` is empty.
        """
        try:
            import faiss
        except ImportError as exc:
            raise ImportError(
                "faiss is required.  Install with:\n"
                "  uv add faiss-cpu          # CPU\n"
                "  uv add faiss-gpu          # GPU / CUDA build"
            ) from exc

        if not dataset:
            raise ValueError("dataset is empty — embed_dataset() must be called first.")

        keys, vecs = _stack_dataset(dataset)
        vecs_norm = _l2_normalise(vecs)

        N, d = vecs_norm.shape
        _nlist = nlist if nlist is not None else max(1, int(N ** _NLIST_SQRT_FACTOR ** 0.5))

        # Cannot have more cells than vectors during training
        _nlist = min(_nlist, N)
        _nprobe = nprobe if nprobe is not None else max(1, _nlist // 10)

        quantiser = faiss.IndexFlatIP(d)

        if use_pq and N >= 256:
            # pq_m must divide d evenly
            while d % pq_m != 0 and pq_m > 1:
                pq_m -= 1
            index = faiss.IndexIVFPQ(
                quantiser, d, _nlist, pq_m, pq_nbits,
                faiss.METRIC_INNER_PRODUCT,
            )
            index_type = "IVF-PQ"
        else:
            if use_pq and N < 256:
                print("WARNING: corpus too small for IVF-PQ (N<256) — falling back to IVFFlat.")
            index = faiss.IndexIVFFlat(quantiser, d, _nlist, faiss.METRIC_INNER_PRODUCT)
            index_type = "IVFFlat"

        index.train(vecs_norm)
        index.add(vecs_norm)
        index.nprobe = _nprobe

        print(
            f"FAISS index built: {index.ntotal} vectors | d={d} | "
            f"nlist={_nlist} | nprobe={_nprobe} | type={index_type}"
        )
        return cls(keys=keys, vecs_norm=vecs_norm, index=index)

    # ------------------------------------------------------------------
    # Search
    # ------------------------------------------------------------------

    def search_all(self, top_k: int = 10) -> Dict[str, List[Tuple[str, float]]]:
        """Batch cosine-similarity search for every vector in the index.

        Returns a dict with the same structure as ``Img2Vec.sim_dict``:
        ``{filename: [(target_filename, score), ...]}``.

        Self-hits (rank 0, score ≈ 1.0) are included so callers can
        decide whether to strip them.

        Args:
            top_k: Number of results to return per query (including self-hit).

        Returns:
            Similarity dict keyed by query filename.
        """
        scores_mat, idx_mat = self._index.search(self.vecs_norm, top_k + 1)

        sim_dict: Dict[str, List[Tuple[str, float]]] = {}
        for i, key in enumerate(self.keys):
            hits: List[Tuple[str, float]] = []
            for rank in range(top_k + 1):
                j = int(idx_mat[i, rank])
                if j < 0:               # FAISS pads with -1 when fewer results exist
                    continue
                hits.append((self.keys[j], float(scores_mat[i, rank])))
                if len(hits) > top_k:
                    break
            sim_dict[key] = hits
        return sim_dict

    def search_one(self,
                   query_vec: np.ndarray,
                   top_k: int = 10,
                   exclude_self_key: str = None) -> List[Tuple[str, float]]:
        """Search for the ``top_k`` most similar proteins to a single query.

        Args:
            query_vec:        1-D float embedding (not necessarily normalised).
            top_k:            Number of results to return.
            exclude_self_key: If provided, any hit matching this key is skipped
                              (useful when the query is already in the corpus).

        Returns:
            List of ``(filename, cosine_score)`` tuples, descending by score.
        """
        qvec = query_vec.copy().reshape(1, -1).astype(np.float32)
        norm = np.linalg.norm(qvec)
        if norm > 0:
            qvec /= norm

        scores, indices = self._index.search(qvec, top_k + 1)
        results: List[Tuple[str, float]] = []
        for rank in range(top_k + 1):
            j = int(indices[0, rank])
            if j < 0:
                continue
            key = self.keys[j]
            if exclude_self_key and key == exclude_self_key:
                continue
            results.append((key, float(scores[0, rank])))
            if len(results) >= top_k:
                break
        return results

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, index_path: str) -> None:
        """Save the FAISS index and key mapping to disk.

        Two files are written:
        - ``index_path``             — the FAISS binary index
        - ``index_path + '.keys.pkl'`` — the ordered key list

        Args:
            index_path: Destination path, e.g. ``/data/corpus.faiss``.
        """
        try:
            import faiss
        except ImportError as exc:
            raise ImportError("faiss required for save().") from exc

        os.makedirs(os.path.dirname(os.path.abspath(index_path)), exist_ok=True)
        faiss.write_index(self._index, index_path)
        keys_path = index_path + ".keys.pkl"
        with open(keys_path, "wb") as fh:
            pickle.dump({"keys": self.keys, "vecs_norm": self.vecs_norm}, fh)
        print(f"Saved FAISS index   → {index_path}")
        print(f"Saved key mapping   → {keys_path}")

    @classmethod
    def load(cls, index_path: str) -> "FaissIndex":
        """Load a previously saved FAISS index and key mapping.

        Args:
            index_path: Path to the ``.faiss`` file written by ``save()``.

        Returns:
            A ready-to-use ``FaissIndex`` instance.
        """
        try:
            import faiss
        except ImportError as exc:
            raise ImportError("faiss required for load().") from exc

        index = faiss.read_index(index_path)
        keys_path = index_path + ".keys.pkl"
        with open(keys_path, "rb") as fh:
            data = pickle.load(fh)
        keys = data["keys"]
        vecs_norm = data["vecs_norm"]
        print(f"Loaded FAISS index  ← {index_path}  ({index.ntotal} vectors)")
        return cls(keys=keys, vecs_norm=vecs_norm, index=index)

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def n_vectors(self) -> int:
        """Number of vectors stored in the index."""
        return self._index.ntotal

    @property
    def dim(self) -> int:
        """Embedding dimension."""
        return self._index.d

    def __repr__(self) -> str:
        return (
            f"FaissIndex(n={self.n_vectors}, d={self.dim}, "
            f"nprobe={getattr(self._index, 'nprobe', 'N/A')})"
        )
