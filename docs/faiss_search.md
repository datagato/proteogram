# Approximate Nearest Neighbour Search with FAISS

*Applies to `scripts/v2/measure_similarity_v2.py` and `proteogram.v2.FaissIndex`.*

> **Read this before turning on `--faiss`.** Used with its default settings it
> is *slower* than the brute-force path it replaces. The speedup only exists
> if you also cap the ranking depth with `--faiss_top_k`.

## Why

`Img2Vec.similarities()` scores every query against every corpus vector. That
is O(N^2) in time and memory, and it becomes the bottleneck long before the
embedding step does. A FAISS IVF index instead partitions the corpus into
Voronoi cells and visits only a few of them per query, so search cost scales
with `nprobe` rather than with corpus size.

Embeddings are L2-normalised before indexing, so the inner-product metric
FAISS searches with is exactly the cosine similarity the brute-force path
reports. Both paths fill `Img2Vec.sim_dict` with the same
`{filename: [(target, score), ...]}` structure, self-hit at rank 0, so
`evaluate_methods_v2.py` does not care which one produced the results.

## The catch: ranking depth

`measure_similarity_v2.py` ranks the **whole corpus** for every query by
default, so Recall@K can be computed afterwards at any K. This is a sensible
benchmarking default, but it is the worst possible case for an ANN index. An
IVF search only ever returns vectors that live in the cells it probes, so
asking for all N results forces it to probe every cell. At that point it is
doing the same work as brute force plus the indexing overhead.

`FaissIndex.search_all()` detects this and widens `nprobe` until the requested
depth is actually reachable, printing what it did. It never silently returns a
short ranking. (An earlier revision did, returning 200 of 2008 requested
results per query and writing blank CSV cells that crashed
`evaluate_methods_v2.py` when it tried to parse them as `target,score`.)

So: **cap the depth.** Set `--faiss_top_k` to the largest K you actually
evaluate at. The script prints a warning that metrics beyond that K are not
computable from the run.

## Measured cost

Random 512-dimensional vectors, single CPU. `N=2008` is the size of the SCOPe
eval set used in Step 4; `N=13503` is the released demo corpus.

| corpus | brute force, full ranking | `--faiss`, full ranking | `--faiss --faiss_top_k 100` |
|---|---|---|---|
| N = 2,008 | 0.11 s | 1.46 s | 0.07 s |
| N = 13,503 | 4.28 s | 77.37 s | 1.60 s |

Index build time is small and is not the problem: 0.10 s at N=2,008 and 0.16 s
at N=13,503.

Two things to take from this. Plain `--faiss` costs you roughly **18x** at the
demo corpus size and gets worse as the corpus grows, because the full ranking
is exhaustive either way. With the depth capped at 100, FAISS is **~2.7x
faster** than brute force at N=13,503, and the gap widens with N since the
brute-force path stays O(N^2) while the capped ANN path does not.

Note that these are random vectors, which have no cluster structure for the
coarse quantiser to exploit. Real proteogram embeddings are clustered by fold
and superfamily, so recall at a given `nprobe` should be better than what
random data suggests -- but measure it on your own corpus rather than assuming.

## Recall

ANN trades recall for speed, and the default `nprobe = nlist // 10` is
aggressive. On random 512-d vectors, Recall@20 against the exact ranking was
**0.26** at the default and **0.71** at `nprobe = nlist // 2`. Again, random
vectors are the worst case -- but the shape of the tradeoff is real.

Before trusting `--faiss` numbers in a comparison against GTalign, USalign or
Foldseek, run both paths on the same corpus and confirm the retrieval metrics
agree. If they do not, raise `nprobe`:

```python
from proteogram.v2 import FaissIndex

index = FaissIndex.from_dataset(img_sim.dataset)
index.nprobe = index._index.nlist // 2   # clamped to [1, nlist]
```

## Usage

Install the extra (FAISS is optional; the brute-force path needs nothing):

```bash
uv sync --extra search
```

Then from `scripts/v2/`:

```bash
# Rank the top 100 per query with an ANN index
python measure_similarity_v2.py --no-embed --faiss --faiss_top_k 100

# Compressed index, for corpora large enough that the raw vectors are a
# memory problem (roughly >100k proteograms)
python measure_similarity_v2.py --no-embed --faiss --faiss_top_k 100 --faiss_pq
```

The index is saved next to `embed_file` with a `.faiss` extension and reused
on later runs; pass `--overwrite` to rebuild it, or `--faiss_index_file` to put
it somewhere else. A companion `.keys.pkl` holds the filename mapping.

### Flags

| Flag | Meaning |
|---|---|
| `--faiss` | Use the ANN index instead of brute-force cosine search |
| `--faiss_top_k N` | Rank only the top N per query. **This is what makes it fast.** Defaults to the whole corpus |
| `--faiss_pq` | Use a product-quantised (IVF-PQ) index: much lower memory, some recall lost. Ignored below 256 vectors, which is too few to train |
| `--faiss_index_file` | Where to save/load the index. Defaults to `embed_file` with a `.faiss` extension |

## Library API

`FaissIndex` is independent of `Img2Vec` -- it operates on plain float32 numpy
arrays plus an ordered key list, and imports `faiss` lazily, so importing
`proteogram.v2` works without the extra installed.

```python
from proteogram.v2 import FaissIndex

index = FaissIndex.from_dataset(img_sim.dataset)   # {filename: tensor}
sim_dict = index.search_all(top_k=100)             # same shape as Img2Vec.sim_dict
hits = index.search_one(query_vec, top_k=10, exclude_self_key='d1abca_.jpg')

index.save('corpus.faiss')
index = FaissIndex.load('corpus.faiss')
```

`Img2Vec` also wraps this: `build_faiss_index()`, `similarities_faiss()`,
`save_faiss_index()`, `load_faiss_index()`.

### Tuning

| Parameter | Default | Effect |
|---|---|---|
| `nlist` | `sqrt(N)` | Voronoi cells. More cells means finer partitioning and slower training; capped at N |
| `nprobe` | `nlist // 10` | Cells visited per query. **The main recall/speed dial** |
| `pq_m` | 8 | IVF-PQ sub-quantisers. Must divide the embedding dimension; reduced automatically until it does |
| `pq_nbits` | 8 | Bits per sub-quantiser |

`search_all()` and `search_one()` raise `nprobe` temporarily when the requested
depth needs it, then put it back, so one deep query does not leave the index
scanning exhaustively for everything afterwards.

## Tests

```bash
uv run --extra search --extra test pytest tests/test_faiss_search.py
```

Covers the `nlist`/`nprobe` defaults, that deep rankings are not truncated,
that `nprobe` is restored after a deep search, self-hit ordering, exactness
against brute force when the search is exhaustive, the recall/`nprobe`
relationship, save/load round trips, and the IVF-PQ fallbacks.
