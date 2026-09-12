"""Tests for the FAISS ANN search backend.

Run with: uv run --extra search --extra test pytest tests/test_faiss_search.py

The corpora here are random gaussian vectors, which is close to the worst case
for an IVF index: there is no cluster structure for the coarse quantiser to
exploit, so approximate recall is low. That is deliberate. These tests check
the index's mechanics (defaults, truncation, state, persistence), not the
recall the real proteogram embeddings achieve, which only a run against the
actual corpus can tell you.
"""
import numpy as np
import pytest
import torch

faiss = pytest.importorskip('faiss', reason='install the "search" extra to run these')

from proteogram.v2.faiss_search import FaissIndex


N_VECS = 600
DIM = 128


@pytest.fixture(scope='module')
def vectors():
    rng = np.random.default_rng(0)
    return rng.normal(size=(N_VECS, DIM)).astype(np.float32)


@pytest.fixture
def dataset(vectors):
    return {f'p{i}.jpg': torch.from_numpy(vectors[i].copy()) for i in range(N_VECS)}


@pytest.fixture
def index(dataset):
    return FaissIndex.from_dataset(dataset)


def brute_force_top_k(vectors, k):
    """Exact cosine ranking, as Img2Vec.similarities() would produce it."""
    unit = vectors / np.linalg.norm(vectors, axis=1, keepdims=True)
    return np.argsort(-(unit @ unit.T), axis=1)[:, :k]


def test_defaults_follow_sqrt_n(index):
    # nlist defaulting to N rather than sqrt(N) gives every vector its own cell,
    # which trains slowly and is not an approximate index in any useful sense.
    assert index._index.nlist == int(N_VECS ** 0.5)
    assert index.nprobe == index._index.nlist // 10
    assert index.n_vectors == N_VECS
    assert index.dim == DIM


def test_empty_dataset_rejected():
    with pytest.raises(ValueError, match='empty'):
        FaissIndex.from_dataset({})


def test_source_embeddings_are_not_normalised_in_place(dataset, vectors):
    FaissIndex.from_dataset(dataset)
    after = np.stack([dataset[f'p{i}.jpg'].numpy() for i in range(N_VECS)])
    np.testing.assert_allclose(after, vectors)


def test_deep_ranking_is_not_truncated(index):
    # measure_similarity_v2.py asks for the whole corpus ordering so Recall@K
    # can be computed at any K afterwards. An IVF index only returns what sits
    # in the cells it probes, so this has to widen nprobe rather than pad -1.
    sim_dict = index.search_all(top_k=N_VECS)
    assert {len(hits) for hits in sim_dict.values()} == {N_VECS}


def test_deep_search_does_not_leave_nprobe_raised(index):
    before = index.nprobe
    index.search_all(top_k=N_VECS)
    assert index.nprobe == before
    index.search_one(np.zeros(DIM, dtype=np.float32), top_k=N_VECS)
    assert index.nprobe == before


def test_exhaustive_ranking_matches_brute_force(index, vectors):
    sim_dict = index.search_all(top_k=N_VECS)
    expected = brute_force_top_k(vectors, 20)
    for i in range(N_VECS):
        got = [key for key, _ in sim_dict[f'p{i}.jpg'][:20]]
        assert got == [f'p{j}.jpg' for j in expected[i]]


def test_self_hit_leads_each_ranking(index):
    sim_dict = index.search_all(top_k=5)
    for i in range(N_VECS):
        key, score = sim_dict[f'p{i}.jpg'][0]
        assert key == f'p{i}.jpg'
        assert score == pytest.approx(1.0, abs=1e-4)


def test_raising_nprobe_improves_recall(index, vectors):
    expected = [{f'p{j}.jpg' for j in row} for row in brute_force_top_k(vectors, 20)]

    def recall_at_20():
        sim_dict = index.search_all(top_k=20)
        found = sum(len({k for k, _ in sim_dict[f'p{i}.jpg']} & expected[i])
                    for i in range(N_VECS))
        return found / (N_VECS * 20)

    default = recall_at_20()
    index.nprobe = index._index.nlist
    assert recall_at_20() > default


def test_nprobe_setter_clamps_to_valid_range(index):
    index.nprobe = 10 ** 6
    assert index.nprobe == index._index.nlist
    index.nprobe = -5
    assert index.nprobe == 1


def test_search_one_can_drop_the_query_itself(index, vectors):
    hits = index.search_one(vectors[7], top_k=5, exclude_self_key='p7.jpg')
    assert len(hits) == 5
    assert all(key != 'p7.jpg' for key, _ in hits)
    # Scores must still come back descending after the self-hit is removed.
    assert [s for _, s in hits] == sorted((s for _, s in hits), reverse=True)


def test_search_one_normalises_the_query(index, vectors):
    scaled = index.search_one(vectors[3] * 17.0, top_k=5)
    plain = index.search_one(vectors[3], top_k=5)
    assert [key for key, _ in scaled] == [key for key, _ in plain]
    # Scaling then normalising in float32 costs a few ulps, so scores match
    # approximately rather than bit for bit.
    for (_, a), (_, b) in zip(scaled, plain):
        assert a == pytest.approx(b, abs=1e-5)


def test_zero_vector_does_not_divide_by_zero(dataset):
    dataset['zero.jpg'] = torch.zeros(DIM)
    idx = FaissIndex.from_dataset(dataset)
    hits = idx.search_one(np.zeros(DIM, dtype=np.float32), top_k=3)
    assert len(hits) == 3


def test_save_load_round_trip(index, tmp_path):
    path = str(tmp_path / 'nested' / 'corpus.faiss')   # parent dirs are created
    index.save(path)
    restored = FaissIndex.load(path)
    assert restored.keys == index.keys
    assert restored.search_all(top_k=10) == index.search_all(top_k=10)


def test_pq_falls_back_on_a_corpus_too_small_to_train(vectors):
    small = {f'p{i}.jpg': torch.from_numpy(vectors[i].copy()) for i in range(100)}
    idx = FaissIndex.from_dataset(small, use_pq=True)
    assert isinstance(idx._index, faiss.IndexIVFFlat)


def test_pq_index_is_searchable(dataset):
    idx = FaissIndex.from_dataset(dataset, use_pq=True)
    assert isinstance(idx._index, faiss.IndexIVFPQ)
    sim_dict = idx.search_all(top_k=5)
    assert len(sim_dict) == N_VECS


def test_pq_sub_quantiser_count_is_reduced_to_divide_the_dimension(dataset):
    # 7 does not divide 128, so from_dataset() has to walk pq_m down to 4.
    idx = FaissIndex.from_dataset(dataset, use_pq=True, pq_m=7)
    assert DIM % idx._index.pq.M == 0


def test_nlist_cannot_exceed_the_corpus_size(vectors):
    tiny = {f'p{i}.jpg': torch.from_numpy(vectors[i].copy()) for i in range(4)}
    idx = FaissIndex.from_dataset(tiny, nlist=500)
    assert idx._index.nlist == 4
