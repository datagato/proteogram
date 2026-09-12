# Approximate Nearest Neighbour Search with FAISS

*Applies to `scripts/v2/measure_similarity_v2.py` and `proteogram.v2.FaissIndex`.*

> **Read this before turning on `--faiss`.** Plain `--faiss` ranks the whole
> corpus, which makes it *5x slower* than brute force. Always pair it with
> `--faiss_top_k`. On the released 13,503-proteogram corpus a tuned index is
> ~5x faster than brute force at 96% Recall@10; the shipped default is
> deliberately conservative at 1.9x and 99.1%. See
> [Measured cost and recall](#measured-cost-and-recall).

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

## Measured cost and recall

Measured on the **released 13,503-proteogram corpus embeddings** (the ResNet18
superfamily CE checkpoint, 512-d), retrieving top-10, single machine, 4 FAISS
threads against a 4-thread OpenBLAS brute-force reference. Brute-force top-10
over this corpus takes 1.82 s.

`nlist` defaults to `sqrt(N) = 116` here, so `nprobe` is the fraction of the
corpus scanned. Recall@10 is measured against the exact cosine ranking:

| `nprobe` | % of corpus scanned | time | vs brute force | Recall@10 |
|---|---|---|---|---|
| 1 | 0.9% | 0.14 s | **12.6x** | 0.841 |
| 2 | 1.7% | 0.25 s | **7.3x** | 0.916 |
| 3 | 2.6% | 0.34 s | 5.4x | 0.946 |
| 4 | 3.4% | 0.38 s | 4.8x | 0.962 |
| 6 | 5.2% | 0.52 s | 3.5x | 0.977 |
| 8 | 6.9% | 0.69 s | 2.6x | 0.984 |
| 11 *(default)* | 9.5% | 0.94 s | 1.9x | 0.991 |
| 16 | 13.8% | 1.65 s | 1.1x | 0.997 |
| 24 | 20.7% | 2.97 s | 0.6x | 0.999 |
| 116 | 100% | 9.76 s | 0.2x | 1.000 |

There is a broad useful range here. Around `nprobe = 4` (3.4% of the corpus)
the index is roughly **5x faster than brute force while keeping 96% of the
exact top-10**. Pushing to `nprobe = 1` buys 12.6x at 84% recall.

The shipped default of `nprobe = nlist // 10` sits at the conservative end:
1.9x faster, 99.1% recall. That is a defensible default -- it barely perturbs
the ranking -- but if search time matters, lowering it is where the gains are.

Two settings to avoid:

- **A full-corpus ranking.** At `nprobe = nlist` the index scans everything and
  is **5x slower** than brute force (9.76 s vs 1.82 s), since it does the same
  work plus indexing overhead. This is what plain `--faiss` does by default,
  which is why `--faiss_top_k` matters.
- **`nprobe` above ~15% of the corpus.** Past that the index is slower than
  brute force for recall gains in the third decimal place.

Index build time is negligible and is not part of this tradeoff: 0.16 s.

### Why random test vectors are not a proxy

Earlier revisions of this document quoted figures measured on random gaussian
vectors. Those understated real performance by a wide margin and have been
removed. At matched N and dimension, Recall@10 on real proteogram embeddings
versus random vectors:

| % of corpus scanned | real embeddings | random gaussian |
|---|---|---|
| 4.3% | 0.702 | 0.225 |
| 8.7% | 0.869 | 0.307 |
| 17.4% | 0.968 | 0.437 |
| 34.8% | 0.998 | 0.633 |

Roughly a 3x difference at the same scanned fraction. This is expected: the
model is trained to cluster structures by fold and superfamily, so the coarse
quantiser has genuine structure to exploit, whereas isotropic gaussian vectors
have none. Benchmark against real embeddings.

### Scaling

The numbers above are for one corpus size. Brute-force cost grows as O(N^2)
while the IVF path at fixed `nprobe/nlist` does not, so the advantage should
widen with N -- but that has not been measured beyond 13,503 on real data, and
the useful `nprobe` may shift as `nlist = sqrt(N)` grows. Re-run the sweep if
you move to a substantially larger corpus.

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
| `nprobe` | `nlist // 10` | Cells visited per query. **The main recall/speed dial.** The default is conservative; `nlist // 30` gave ~5x at 96% Recall@10 on the released corpus |
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
