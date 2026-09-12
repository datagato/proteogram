"""FAISS approximate nearest neighbour search over Proteogram embeddings.

Img2Vec.similarities() scores every query against every corpus vector, which
is O(N^2) in both time and memory and becomes the bottleneck well before the
embedding step does. This module wraps a FAISS IVF index over the same
embeddings so search cost scales with nprobe rather than corpus size.

Nothing here imports Img2Vec. The index operates on plain float32 numpy
arrays plus an ordered key list, so the integer ids FAISS returns can be
mapped back to proteogram filenames. faiss itself is imported lazily inside
the methods that need it, so importing proteogram.v2 stays cheap.

Embeddings are L2-normalised before indexing, so the inner-product metric
FAISS searches with is equivalent to the cosine similarity Img2Vec reports.

Example:
-----------

    from proteogram.v2.faiss_search import FaissIndex

    index = FaissIndex.from_dataset(img_sim.dataset)
    sim_dict = index.search_all(top_k=5)      # same shape as Img2Vec.sim_dict
    hits = index.search_one(query_vec, top_k=5)

    index.save('corpus.faiss')
    index = FaissIndex.load('corpus.faiss')
"""
from __future__ import annotations

import os
import pickle
from contextlib import contextmanager
from typing import Dict, List, Tuple

import numpy as np
import torch


_FAISS_INSTALL_HINT = (
    "faiss is required for ANN search. Install with 'uv add faiss-cpu', or "
    "'uv add faiss-gpu' for a CUDA build."
)


def _import_faiss():
    """Import faiss, re-raising with an install hint if it is missing."""
    try:
        import faiss
    except ImportError as exc:
        raise ImportError(_FAISS_INSTALL_HINT) from exc
    return faiss


def _l2_normalise(mat: np.ndarray) -> np.ndarray:
    """Return an L2-normalised float32 copy of mat (shape N x d).

    Normalising a copy rather than in place matters because the caller's
    embeddings are the same arrays Img2Vec keeps in self.dataset.
    """
    mat = mat.astype(np.float32, copy=True)
    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    # A zero vector has no direction; leave it as-is instead of dividing by 0.
    norms[norms == 0] = 1.0
    mat /= norms
    return mat


def _stack_dataset(dataset: Dict[str, torch.Tensor]) -> Tuple[List[str], np.ndarray]:
    """Flatten an Img2Vec-style dataset dict into an ordered (keys, matrix) pair.

    Args:
        dataset: mapping of filename to a 1-D or 1 x d embedding tensor.

    Returns:
        keys: filenames in the same row order as the matrix.
        matrix: float32 array of shape (N, d).
    """
    keys = list(dataset.keys())
    vecs = torch.cat([dataset[k].cpu().reshape(1, -1) for k in keys]).float().numpy()
    return keys, vecs


class FaissIndex:
    """A FAISS IVFFlat or IVF-PQ index paired with its proteogram key mapping.

    Parameters:
    -----------
    keys: ordered proteogram filenames. keys[i] is the proteogram stored at
        FAISS integer id i.
    vecs_norm: the L2-normalised embedding matrix, shape (N, d), float32. Kept
        so search_all() can re-query the corpus and so queries can be
        normalised the same way the corpus was.
    index: a trained, populated FAISS index using METRIC_INNER_PRODUCT.

    See also:
    -----------
    FaissIndex.from_dataset(): build an index from Img2Vec.dataset
    FaissIndex.search_all(): all-vs-all search, returns an Img2Vec.sim_dict
    FaissIndex.search_one(): search a single query vector
    FaissIndex.save()/load(): persist and restore an index
    """

    def __init__(self, keys: List[str], vecs_norm: np.ndarray, index) -> None:
        self.keys = keys
        self.vecs_norm = vecs_norm
        self._index = index

    @classmethod
    def from_dataset(cls,
                     dataset: Dict[str, torch.Tensor],
                     use_pq: bool = False,
                     nlist: int = None,
                     nprobe: int = None,
                     pq_m: int = 8,
                     pq_nbits: int = 8) -> "FaissIndex":
        """Build an index from an Img2Vec-style embedding dataset.

        Args:
            dataset: {filename: embedding_tensor}, i.e. Img2Vec.dataset.
            use_pq: use a product-quantised (IVF-PQ) index. Worth it above
                roughly 100k proteograms, where the uncompressed vectors stop
                fitting comfortably in memory; costs some recall. Ignored for
                corpora under 256 vectors, which are too small to train PQ.
            nlist: number of Voronoi cells. Defaults to sqrt(N), the usual
                FAISS starting point, capped so cells never outnumber vectors.
            nprobe: cells visited per query. Higher is more accurate and
                slower. Defaults to nlist // 10.
            pq_m: number of PQ sub-quantisers. Must divide d evenly, and is
                reduced automatically until it does.
            pq_nbits: bits per sub-quantiser. 8 is the standard choice.

        Returns:
            A trained, populated FaissIndex ready to search.

        Raises:
            ImportError: faiss is not installed.
            ValueError: dataset is empty.
        """
        faiss = _import_faiss()

        if not dataset:
            raise ValueError('dataset is empty, call embed_dataset() first.')

        keys, vecs = _stack_dataset(dataset)
        vecs_norm = _l2_normalise(vecs)
        n_vecs, dim = vecs_norm.shape

        _nlist = nlist if nlist is not None else int(n_vecs ** 0.5)
        # Training clusters the corpus itself, so it cannot ask for more
        # centroids than there are vectors to cluster.
        _nlist = max(1, min(_nlist, n_vecs))
        _nprobe = nprobe if nprobe is not None else _nlist // 10
        _nprobe = max(1, min(_nprobe, _nlist))

        quantiser = faiss.IndexFlatIP(dim)
        if use_pq and n_vecs >= 256:
            while dim % pq_m != 0 and pq_m > 1:
                pq_m -= 1
            index = faiss.IndexIVFPQ(quantiser, dim, _nlist, pq_m, pq_nbits,
                                     faiss.METRIC_INNER_PRODUCT)
            index_type = 'IVF-PQ'
        else:
            if use_pq:
                print('Corpus too small to train IVF-PQ (N<256), using IVFFlat instead.')
            index = faiss.IndexIVFFlat(quantiser, dim, _nlist,
                                       faiss.METRIC_INNER_PRODUCT)
            index_type = 'IVFFlat'

        index.train(vecs_norm)
        index.add(vecs_norm)
        index.nprobe = _nprobe

        print(f'Built {index_type} index: {index.ntotal} vectors, d={dim}, '
              f'nlist={_nlist}, nprobe={_nprobe}')
        return cls(keys=keys, vecs_norm=vecs_norm, index=index)

    @property
    def nprobe(self) -> int:
        """Number of cells visited per query."""
        return self._index.nprobe

    @nprobe.setter
    def nprobe(self, value: int) -> None:
        self._index.nprobe = max(1, min(int(value), self._index.nlist))

    def _reachable(self) -> int:
        """Estimate how many hits a query can return at the current nprobe.

        An IVF search only ever sees the vectors inside the cells it probes,
        so asking for more than that yields -1 padding no matter how large
        top_k is. Cells are not evenly filled, so treat this as a guide.
        """
        nlist = self._index.nlist
        return max(1, int(round(self.n_vectors * self._index.nprobe / nlist)))

    @contextmanager
    def _nprobe_for(self, top_k: int):
        """Temporarily raise nprobe so top_k results are actually reachable.

        Without this a deep request (measure_similarity_v2.py asks for the
        whole corpus ordering) silently comes back short, padded with -1,
        because the cells probed simply do not hold that many vectors. The
        original nprobe is restored afterwards so one deep search does not
        leave the index scanning exhaustively for every later query.
        """
        nlist = self._index.nlist
        original = self._index.nprobe
        while self._index.nprobe < nlist and self._reachable() < top_k:
            self._index.nprobe = min(nlist, self._index.nprobe * 2)
        if self._index.nprobe != original:
            print(f'Raised nprobe {original} -> {self._index.nprobe} of {nlist} cells '
                  f'to reach {top_k} results'
                  + (' (exhaustive, no ANN speedup at this depth)'
                     if self._index.nprobe >= nlist else ''))
        try:
            yield
        finally:
            self._index.nprobe = original

    def search_all(self, top_k: int = 10) -> Dict[str, List[Tuple[str, float]]]:
        """Search the whole corpus against itself.

        Self-hits are kept at rank 0 (score ~1.0) so the result matches what
        Img2Vec.similarities() produces and callers can strip them or not.

        nprobe is widened automatically if top_k is deeper than the current
        setting can reach, so the returned ranking is never silently truncated.

        Args:
            top_k: results per query, self-hit included.

        Returns:
            {filename: [(target_filename, score), ...]}, the same structure as
            Img2Vec.sim_dict.
        """
        top_k = min(top_k, self.n_vectors)
        with self._nprobe_for(top_k):
            scores_mat, idx_mat = self._index.search(self.vecs_norm, top_k)

        sim_dict: Dict[str, List[Tuple[str, float]]] = {}
        for i, key in enumerate(self.keys):
            hits = [(self.keys[j], float(scores_mat[i, rank]))
                    for rank, j in enumerate(idx_mat[i])
                    if j >= 0]   # FAISS pads short result rows with -1
            sim_dict[key] = hits
        return sim_dict

    def search_one(self,
                   query_vec: np.ndarray,
                   top_k: int = 10,
                   exclude_self_key: str = None) -> List[Tuple[str, float]]:
        """Search the corpus for the top_k proteograms closest to one query.

        Args:
            query_vec: 1-D embedding, normalised here so it need not be.
            top_k: number of results to return.
            exclude_self_key: drop any hit with this key, for when the query
                is itself part of the indexed corpus.

        Returns:
            [(filename, cosine_score), ...], highest score first.
        """
        # Search one deeper than asked so dropping the self-hit still leaves
        # top_k results.
        n_request = min(top_k + 1, self.n_vectors)

        qvec = np.asarray(query_vec, dtype=np.float32).reshape(1, -1).copy()
        norm = np.linalg.norm(qvec)
        if norm > 0:
            qvec /= norm

        with self._nprobe_for(n_request):
            scores, indices = self._index.search(qvec, n_request)
        results: List[Tuple[str, float]] = []
        for rank, j in enumerate(indices[0]):
            if j < 0:
                continue
            key = self.keys[j]
            if exclude_self_key and key == exclude_self_key:
                continue
            results.append((key, float(scores[0, rank])))
            if len(results) >= top_k:
                break
        return results

    def save(self, index_path: str) -> None:
        """Write the index and its key mapping to disk.

        Two files are produced: index_path holds the FAISS binary index, and
        index_path + '.keys.pkl' holds the key list and normalised vectors
        needed to reconstruct the wrapper.

        Args:
            index_path: destination path, e.g. 'corpus.faiss'.
        """
        faiss = _import_faiss()

        os.makedirs(os.path.dirname(os.path.abspath(index_path)), exist_ok=True)
        faiss.write_index(self._index, index_path)
        keys_path = index_path + '.keys.pkl'
        with open(keys_path, 'wb') as pklout:
            pickle.dump({'keys': self.keys, 'vecs_norm': self.vecs_norm}, pklout)
        print(f'Saved FAISS index to {index_path}')
        print(f'Saved key mapping to {keys_path}')

    @classmethod
    def load(cls, index_path: str) -> "FaissIndex":
        """Restore an index written by save().

        Args:
            index_path: path to the .faiss file, with its .keys.pkl companion
                alongside it.

        Returns:
            A ready-to-search FaissIndex.
        """
        faiss = _import_faiss()

        index = faiss.read_index(index_path)
        keys_path = index_path + '.keys.pkl'
        with open(keys_path, 'rb') as pklin:
            data = pickle.load(pklin)
        print(f'Loaded FAISS index from {index_path} ({index.ntotal} vectors)')
        return cls(keys=data['keys'], vecs_norm=data['vecs_norm'], index=index)

    @property
    def n_vectors(self) -> int:
        """Number of vectors held in the index."""
        return self._index.ntotal

    @property
    def dim(self) -> int:
        """Embedding dimension."""
        return self._index.d

    def __repr__(self) -> str:
        return (f'FaissIndex(n={self.n_vectors}, d={self.dim}, '
                f'nprobe={self.nprobe})')
