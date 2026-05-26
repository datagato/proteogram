# Proteogram v2 — Three Targeted Improvements

## Overview

This document covers the design, implementation, and validation plan for three independent but complementary improvements to the Proteogram v2 pipeline:

| # | Name | File(s) Affected | Impact |
|---|------|-----------------|--------|
| [1](#1-faiss-approximate-nearest-neighbour-search) | FAISS Approximate Nearest Neighbour Search | `proteogram/v2/image_similarity.py`, `scripts/v2/measure_similarity_v2.py` | Scales corpus search from minutes to milliseconds |
| [2](#2-global-percentile-normalisation) | Global Percentile Normalisation | `proteogram/v2/proteogram.py`, `scripts/v2/create_v2_proteograms.py` | Preserves physically meaningful inter-protein scale |
| [3](#3-grad-cam-explainability) | Grad-CAM Explainability | `proteogram/v2/image_similarity.py` (new method), new `scripts/v2/explain_similarity.py` | Residue-pair attribution for any similar pair |

Each section follows the same structure: motivation → design decisions → full implementation → validation steps.

---

## Operational Notes (May 2026): Environment Setup + Long v2 Runs

This section documents practical lessons from running the current v2 pipeline on Linux with mixed toolchains (`uv`, local Conda bootstrap, OpenMM).

### A. Why `create_v2_proteograms.py` may be slow

`scripts/v2/create_v2_proteograms.py` runs a full MD pipeline per protein (minimization + NPT + NVT + production) before image export. If OpenMM CUDA is unavailable, this falls back to CPU and runtime increases significantly.

At default MD lengths, CPU runtime can be many minutes per protein; with 2,008 proteins this can become multi-day if not accelerated.

### B. Distinguish PyTorch CUDA vs OpenMM CUDA

It is common to have:

- `torch.cuda.is_available() == True`
- OpenMM platforms = `['Reference', 'CPU', 'OpenCL']`

In this case, similarity scripts can use GPU (PyTorch), but MD in v2 proteogram generation still runs without CUDA.

Check OpenMM platforms directly:

```bash
python - <<'PY'
from openmm import Platform
names = [Platform.getPlatform(i).getName() for i in range(Platform.getNumPlatforms())]
print('OpenMM platforms:', names)
print('CUDA available:', 'CUDA' in names)
PY
```

### C. Conda bootstrap pitfall encountered

In this run, `conda` was pointing to a local bootstrap install under:

`scripts/v2/exit/bin/conda`

This caused solver and lock issues (e.g., sqlite lock, libmamba plugin mismatch), and prevented reliable environment creation.

Recommended safeguards:

1. Confirm which conda is active (`which conda`, `conda info --base`).
2. Prefer a stable system conda/mamba/micromamba install for OpenMM-CUDA env creation.
3. If needed, force classic solver when libmamba plugin is unavailable.

### D. Minimal reliable runbook (CUDA-capable OpenMM env)

```bash
# 1) create env with python 3.11 (recommended for this project stack)
conda create -n proteogram-openmm-cuda -c conda-forge python=3.11 openmm pdbfixer -y

# 2) activate env
conda activate proteogram-openmm-cuda

# 3) verify OpenMM CUDA platform visibility
python - <<'PY'
from openmm import Platform
print([Platform.getPlatform(i).getName() for i in range(Platform.getNumPlatforms())])
PY

# 4) install project in editable mode
cd /path/to/proteogram
pip install -e .

# 5) run v2 proteogram creation
cd scripts/v2
python create_v2_proteograms.py --overwrite
```

### E. Monitoring a long run

During `create_v2_proteograms.py`, JPG outputs are written incrementally (not only at end). To watch output growth:

```bash
# from repo root
watch -n 5 'find data/scope2.08_all_proteograms_v2 -maxdepth 1 -name "*.jpg" | wc -l'

# or from scripts/v2
watch -n 5 'find ../data/scope2.08_all_proteograms_v2 -maxdepth 1 -name "*.jpg" | wc -l'
```

### F. Current observed status

- Run is progressing through structures and skipping out-of-range chains (`sequence length outside [20, 200]`) as designed.
- Output directory image count should rise continuously as proteins complete.
- If OpenMM CUDA remains unavailable, OpenCL/CPU execution is expected and slower than CUDA.

---

## 1. FAISS Approximate Nearest Neighbour Search

### 1.1 Motivation

The current `Img2Vec.similarities()` method computes cosine similarity between every pair of embeddings in the corpus:

```python
# proteogram/v2/image_similarity.py — existing inner loop (O(N²))
for image_path_i, embedding_i in tqdm(self.dataset.items()):
    for image_path_j, embedding_j in self.dataset.items():
        sim = cosine(embedding_i, embedding_j)[0].item()
```

For a corpus of N proteins this is O(N²) in both time and sequential memory access. At the current SCOPe 2.08 scale (~100 K domains) this already takes tens of minutes. The AlphaFold Database (AFDB) contains ~200 million predicted structures — brute-force search there would take weeks.

FAISS (Facebook AI Similarity Search) replaces this with an **Inverted File index with Product Quantisation (IVF-PQ)** that gives sub-linear query time with controllable recall-accuracy trade-offs.

### 1.2 Design Decisions

#### Index type: `IndexIVFFlat` for small corpora, `IndexIVFPQ` for large

| Corpus size | Recommended index | Reason |
|-------------|------------------|--------|
| ≤ 100 K | `IndexIVFFlat` | Exact L2/IP; no quantisation error; fast enough |
| 100 K – 10 M | `IndexIVFPQ` | 8–32× memory reduction; ~1% recall loss |
| > 10 M | `IndexIVFPQ` + `OPQ` pre-rotation | Best recall at extreme scale |

The `Img2Vec` class will default to `IndexIVFFlat` and let callers opt into `IndexIVFPQ`.

#### Inner-product (IP) vs. L2

FAISS supports both. Because the rest of the codebase uses **cosine similarity**, we L2-normalise embeddings before indexing and use **inner product** — on unit vectors, inner product equals cosine similarity exactly. This avoids any change to how scores are interpreted.

#### `nlist` (number of Voronoi cells)

A rule of thumb is `nlist = sqrt(N)`. For 100 K vectors: `nlist = 316`. For 10 M vectors: `nlist = 3162`. These values will be set automatically if not specified.

#### `nprobe` (cells searched at query time)

Higher `nprobe` → better recall, slower query. Default: `nprobe = max(1, nlist // 10)`. Can be tuned by the caller.

#### Backward compatibility

The existing `similarities()` method signature must not change. FAISS is added as an optional code path activated by passing `use_faiss=True`. The `.sim_dict` output format stays identical so `measure_similarity_v2.py` and `evaluate_methods_v2.py` require no changes.

### 1.3 New Dependency

```toml
# pyproject.toml — add to [project] dependencies
"faiss-cpu>=1.8; extra != 'cuda12'",
"faiss-gpu>=1.8; extra == 'cuda12'",
```

Or install manually:
```bash
# CPU
uv add faiss-cpu

# GPU (CUDA 12)
uv add faiss-gpu
```

### 1.4 Implementation

#### 1.4.1 New method: `Img2Vec.build_faiss_index()`

Add to `proteogram/v2/image_similarity.py` inside the `Img2Vec` class:

```python
def build_faiss_index(self,
                      use_pq: bool = False,
                      nlist: int = None,
                      nprobe: int = None,
                      pq_m: int = 8,
                      pq_nbits: int = 8) -> None:
    """Build a FAISS index from the currently loaded embedding dataset.

    Embeddings are L2-normalised before indexing so that inner-product
    search is equivalent to cosine similarity.

    Args:
        use_pq:   If True, use IVF-PQ (compressed) index. Recommended for
                  corpora > 100 K. Defaults to False (IVFFlat, exact).
        nlist:    Number of Voronoi cells. Defaults to sqrt(N).
        nprobe:   Number of cells to search at query time. Higher = better
                  recall, slower query. Defaults to nlist // 10.
        pq_m:     Number of PQ sub-quantisers (IVF-PQ only). Must divide
                  the embedding dimension evenly.
        pq_nbits: Bits per sub-quantiser (IVF-PQ only). 8 is standard.
    """
    try:
        import faiss
    except ImportError:
        raise ImportError(
            "faiss is required for build_faiss_index(). "
            "Install with: uv add faiss-cpu  (or faiss-gpu for GPU builds)."
        )

    if not self.dataset:
        raise RuntimeError("embed_dataset() must be called before build_faiss_index().")

    # Stack embeddings and keys in a consistent order
    keys = list(self.dataset.keys())
    vecs = torch.cat([self.dataset[k].cpu() for k in keys]).float()  # (N, d)

    # L2-normalise so inner product == cosine similarity
    faiss.normalize_L2(vecs.numpy())

    N, d = vecs.shape
    _nlist = nlist if nlist is not None else max(1, int(N ** 0.5))
    _nprobe = nprobe if nprobe is not None else max(1, _nlist // 10)

    # Build quantiser (flat inner-product)
    quantiser = faiss.IndexFlatIP(d)

    if use_pq:
        # Ensure pq_m divides d evenly
        while d % pq_m != 0 and pq_m > 1:
            pq_m -= 1
        index = faiss.IndexIVFPQ(quantiser, d, _nlist, pq_m, pq_nbits,
                                  faiss.METRIC_INNER_PRODUCT)
    else:
        index = faiss.IndexIVFFlat(quantiser, d, _nlist,
                                    faiss.METRIC_INNER_PRODUCT)

    index.train(vecs.numpy())
    index.add(vecs.numpy())
    index.nprobe = _nprobe

    # Store on instance for re-use across queries
    self._faiss_index = index
    self._faiss_keys = keys          # maps integer index → filename key
    self._faiss_vecs_norm = vecs     # keep L2-normalised vecs for query normalisation

    print(f"FAISS index built: {index.ntotal} vectors | d={d} | "
          f"nlist={_nlist} | nprobe={_nprobe} | "
          f"type={'IVF-PQ' if use_pq else 'IVFFlat'}")
```

#### 1.4.2 New method: `Img2Vec.similarities_faiss()`

Add directly below `build_faiss_index()`:

```python
def similarities_faiss(self,
                        n: int = 10,
                        save_result_images_dir: str = None,
                        pad_fn=None) -> float:
    """Compute top-N similar images for every entry in the corpus using FAISS.

    Populates self.sim_dict with the same format as similarities(), so all
    downstream scripts (evaluate_methods_v2.py, measure_similarity_v2.py)
    work without modification.

    Call build_faiss_index() first.

    Args:
        n:                      Top-N results per query (self-hit included
                                at rank 0; callers should request n+1 and
                                strip the self-hit themselves if needed).
        save_result_images_dir: Optional directory to write result images.
        pad_fn:                 Optional padding callable passed to save_images().

    Returns:
        float: Wall-clock seconds spent in FAISS search (excludes image saving).
    """
    try:
        import faiss
    except ImportError:
        raise ImportError("faiss not installed. Run: uv add faiss-cpu")

    if not hasattr(self, '_faiss_index'):
        raise RuntimeError("Call build_faiss_index() before similarities_faiss().")

    keys = self._faiss_keys
    vecs = self._faiss_vecs_norm.numpy()   # already L2-normalised

    start = time()
    # Batch query: search all N vectors at once — single FAISS call
    scores_matrix, indices_matrix = self._faiss_index.search(vecs, n + 1)
    elapsed = time() - start

    # Build sim_dict in the same format as similarities()
    self.sim_dict = {}
    for i, key in enumerate(keys):
        hits = []
        for rank in range(n + 1):
            j = indices_matrix[i, rank]
            if j < 0:               # FAISS pads with -1 when fewer results exist
                continue
            target_key = keys[j]
            score = float(scores_matrix[i, rank])
            hits.append((target_key, score))
        self.sim_dict[key] = hits   # includes self-hit at rank 0

    if save_result_images_dir:
        for image_path in self.sim_dict:
            self.save_images(os.path.join(self.files[0].rsplit('/', 1)[0], image_path),
                             save_result_images_dir, pad_fn=pad_fn)

    return elapsed
```

#### 1.4.3 FAISS index persistence

Add two methods for saving and loading the built index:

```python
def save_faiss_index(self, index_path: str) -> None:
    """Persist the FAISS index and key mapping to disk.

    Args:
        index_path: File path for the index (e.g. 'corpus.faiss').
                    A companion '<index_path>.keys.pkl' file is written
                    alongside for the key mapping.
    """
    import faiss, pickle
    faiss.write_index(self._faiss_index, index_path)
    keys_path = index_path + '.keys.pkl'
    with open(keys_path, 'wb') as f:
        pickle.dump(self._faiss_keys, f)
    print(f"Saved FAISS index → {index_path}")
    print(f"Saved key mapping → {keys_path}")


def load_faiss_index(self, index_path: str) -> None:
    """Load a previously saved FAISS index and key mapping.

    Args:
        index_path: Path to the '.faiss' index file.
    """
    import faiss, pickle
    self._faiss_index = faiss.read_index(index_path)
    keys_path = index_path + '.keys.pkl'
    with open(keys_path, 'rb') as f:
        self._faiss_keys = pickle.load(f)
    # Reconstruct normalised vecs for future queries (needed for single-query search)
    keys = self._faiss_keys
    vecs = torch.cat([self.dataset[k].cpu() for k in keys]).float()
    faiss.normalize_L2(vecs.numpy())
    self._faiss_vecs_norm = vecs
    print(f"Loaded FAISS index from {index_path} "
          f"({self._faiss_index.ntotal} vectors)")
```

#### 1.4.4 Update `measure_similarity_v2.py`

Add `--faiss` flag and wire it up:

```python
# Add to the argparse block
parser.add_argument('--faiss', action='store_true',
                    help='Use FAISS ANN index for similarity search instead of '
                         'brute-force cosine similarity. Much faster for large corpora.')
parser.add_argument('--faiss_index_file', type=str, default=None,
                    help='Path to save/load the FAISS index. Defaults to '
                         'embed_file with .faiss extension.')
parser.add_argument('--faiss_pq', action='store_true',
                    help='Use IVF-PQ compressed index (recommended for > 100K proteins). '
                         'Slightly lower recall but much lower memory.')

# Replace the similarities() call block with:
if args.faiss:
    faiss_index_file = args.faiss_index_file or embed_file.replace('.pkl', '.faiss')
    if os.path.exists(faiss_index_file) and not args.overwrite:
        print(f'Loading existing FAISS index from {faiss_index_file}')
        img_sim.load_faiss_index(faiss_index_file)
    else:
        print('Building FAISS index ...')
        img_sim.build_faiss_index(use_pq=args.faiss_pq)
        img_sim.save_faiss_index(faiss_index_file)
    sim_time = img_sim.similarities_faiss(
        n=n_results,
        save_result_images_dir=None,
        pad_fn=pad_to_size)
else:
    sim_time = img_sim.similarities(n=n_results,
                                     save_result_images_dir=None,
                                     pad_fn=pad_to_size)
```

#### 1.4.5 Single-protein query update in `query_similar_proteins.py`

```python
# Replace the inner loop in query_similar_proteins.py with:
def query_with_faiss(img_sim, query_embedding, top_k, corpus_dir):
    """Query a built FAISS index with a single new embedding."""
    import faiss
    import numpy as np

    query_vec = query_embedding.cpu().float().numpy()       # (1, d)
    faiss.normalize_L2(query_vec)
    scores, indices = img_sim._faiss_index.search(query_vec, top_k + 1)

    results = []
    for rank in range(top_k + 1):
        j = indices[0, rank]
        if j < 0:
            continue
        key = img_sim._faiss_keys[j]
        score = float(scores[0, rank])
        if key != os.path.basename(query_path):  # skip self-hit if present
            results.append((key, score))
        if len(results) >= top_k:
            break
    return results
```

### 1.5 Validation Steps

#### Step 1 — Recall parity test (automated)

Run both methods on the eval set and assert that FAISS Recall@K ≥ 0.99 × brute-force Recall@K at every SCOPe level:

```python
# scripts/v2/tests/test_faiss_recall.py
import pytest
from proteogram.v2 import Img2Vec
import torch, pickle, os

EMBED_FILE = os.environ.get('EMBED_FILE', 'corpus_embeddings.pkl')

@pytest.fixture(scope='module')
def img_sim():
    sim = Img2Vec('resnet_ft', dataset_dir=[], device='cpu')
    with open(EMBED_FILE, 'rb') as f:
        sim.dataset = pickle.load(f)
    return sim

def test_faiss_topk_recall_at_5(img_sim):
    """FAISS top-5 results should overlap ≥99% with brute-force top-5."""
    TOP_K = 5
    # Brute-force
    img_sim.similarities(n=TOP_K)
    bf_dict = {k: set(t for t, _ in v[:TOP_K]) for k, v in img_sim.sim_dict.items()}

    # FAISS IVFFlat
    img_sim.build_faiss_index(use_pq=False)
    img_sim.similarities_faiss(n=TOP_K)
    faiss_dict = {k: set(t for t, _ in v[1:TOP_K+1]) for k, v in img_sim.sim_dict.items()}

    overlaps = []
    for key in bf_dict:
        if key in faiss_dict:
            overlap = len(bf_dict[key] & faiss_dict[key]) / TOP_K
            overlaps.append(overlap)

    mean_recall = sum(overlaps) / len(overlaps)
    print(f'Mean FAISS/BF overlap at top-{TOP_K}: {mean_recall:.4f}')
    assert mean_recall >= 0.99, f'FAISS recall too low: {mean_recall:.4f}'
```

Run:
```bash
EMBED_FILE=/path/to/corpus_embeddings.pkl pytest scripts/v2/tests/test_faiss_recall.py -v
```

#### Step 2 — Timing benchmark

```bash
# Brute-force
time python measure_similarity_v2.py --no-embed

# FAISS IVFFlat
time python measure_similarity_v2.py --no-embed --faiss

# FAISS IVF-PQ (large corpus)
time python measure_similarity_v2.py --no-embed --faiss --faiss_pq
```

Expected results on ~10 K eval set:

| Method | Expected time |
|--------|--------------|
| Brute-force | ~2–5 min |
| FAISS IVFFlat | < 5 sec |
| FAISS IVF-PQ | < 2 sec |

#### Step 3 — MAP@K parity

Run `evaluate_methods_v2.py` on outputs from both methods. FAISS MAP@K should be within ±0.005 of brute-force MAP@K at all SCOPe levels. Any larger gap indicates the `nprobe` needs increasing.

#### Step 4 — Index round-trip test

```python
# Verify save/load produces identical results
img_sim.build_faiss_index()
img_sim.save_faiss_index('/tmp/test.faiss')

img_sim2 = Img2Vec(model_file, dataset_dir=[], device='cpu')
img_sim2.dataset = img_sim.dataset
img_sim2.load_faiss_index('/tmp/test.faiss')

img_sim.similarities_faiss(n=5)
img_sim2.similarities_faiss(n=5)

for key in img_sim.sim_dict:
    assert img_sim.sim_dict[key] == img_sim2.sim_dict[key], f"Mismatch at {key}"
print("Round-trip test passed.")
```

---

## 2. Global Percentile Normalisation

### 2.1 Motivation

The current `ProteogramV2.normalize_map()` applies **per-protein min-max normalisation** independently to each energy channel:

```python
# proteogram/v2/proteogram.py — current implementation
arr = ((arr - arr.min()) * (1 / (arr.max() - arr.min()) * 255)).astype('uint8')
```

This has a critical flaw: every protein's energy map is stretched to fill the full [0, 255] dynamic range, regardless of the actual energy magnitudes. A small, weakly-interacting loop region and a tightly-packed hydrophobic core will produce identical grey levels after normalisation. The model never sees the absolute energy scale — only the relative rank order within each protein.

**Concrete example**: Protein A has VdW attractive energies in [-50, -5] kJ/mol and Protein B has VdW attractive energies in [-200, -20] kJ/mol. After per-protein normalisation, both are mapped to [0, 255]. A CNN comparing the two images cannot tell that Protein B has 4× stronger packing.

Global percentile normalisation computes bounds from the entire training corpus once, then applies those fixed bounds to every protein — preserving inter-protein energy scale in the pixel values.

### 2.2 Design Decisions

#### Percentile instead of global min/max

Extreme outlier structures (e.g., very short peptides, structures with unusual post-translational modifications) would dominate a global min/max and compress most proteins into a narrow band. Using the **1st and 99th percentiles** clips ~2% of values but gives a robust, representative range.

#### Separate bounds per channel

Each of the 6 channels (VdW attractive, VdW repulsive, ES attractive, ES repulsive, distance, hydrophobicity) has a different physical unit and magnitude range. Bounds must be computed and stored independently per channel.

#### Where bounds are stored

A single JSON file `norm_stats.json` is written alongside the proteogram corpus. It is read at proteogram creation time when global normalisation is enabled. This keeps the bounds portable and version-controlled.

#### Backward compatibility flag

The new behaviour is opt-in via `--global_norm` flag in `create_v2_proteograms.py`. Per-protein normalisation remains the default so existing proteogram datasets are unaffected.

### 2.3 New File: `scripts/v2/compute_norm_stats.py`

This one-time script samples up to `--max_samples` existing `.npy` energy matrices (or re-runs the MD pipeline for a random subset) to compute global percentile bounds.

```python
#!/usr/bin/env python
"""Compute global percentile normalisation statistics from a corpus of energy matrices.

Run this ONCE after generating a representative sample of proteograms with
--save_npy_matrices (a new flag added to create_v2_proteograms.py). Outputs
norm_stats.json which is read by create_v2_proteograms.py --global_norm.

Usage:
    python compute_norm_stats.py \\
        --npy_dir /path/to/energy_matrices \\
        --out_file /path/to/norm_stats.json \\
        --low_pct 1.0 \\
        --high_pct 99.0 \\
        --max_samples 5000
"""
import argparse
import json
import glob
import os
import numpy as np
from tqdm import tqdm

CHANNEL_NAMES = [
    'vdw_attractive',
    'vdw_repulsive',
    'es_attractive',
    'es_repulsive',
    'distance',
    'hydrophobicity',
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--npy_dir', required=True,
                        help='Directory of .npy files, one per channel per protein '
                             '(naming: <protein_id>_<channel>.npy).')
    parser.add_argument('--out_file', required=True,
                        help='Output JSON path for normalisation bounds.')
    parser.add_argument('--low_pct', type=float, default=1.0,
                        help='Lower percentile bound (default: 1.0).')
    parser.add_argument('--high_pct', type=float, default=99.0,
                        help='Upper percentile bound (default: 99.0).')
    parser.add_argument('--max_samples', type=int, default=5000,
                        help='Maximum number of energy matrices to sample per channel '
                             '(default: 5000). More samples = more accurate statistics.')
    args = parser.parse_args()

    # Collect all values per channel across the sampled corpus
    channel_values = {ch: [] for ch in CHANNEL_NAMES}

    for channel in CHANNEL_NAMES:
        files = sorted(glob.glob(os.path.join(args.npy_dir, f'*_{channel}.npy')))
        if not files:
            print(f'WARNING: No .npy files found for channel "{channel}" in {args.npy_dir}')
            continue

        # Random subsample if corpus is large
        if len(files) > args.max_samples:
            rng = np.random.default_rng(seed=42)
            files = list(rng.choice(files, size=args.max_samples, replace=False))

        print(f'Channel {channel}: sampling {len(files)} matrices ...')
        for fpath in tqdm(files, desc=channel):
            arr = np.load(fpath)
            # Only include non-zero upper-triangle values (lower triangle is 0)
            vals = arr[arr != 0].ravel()
            channel_values[channel].append(vals)

    # Compute and store bounds
    stats = {}
    for channel in CHANNEL_NAMES:
        if not channel_values[channel]:
            stats[channel] = {'p_low': 0.0, 'p_high': 255.0}
            continue
        all_vals = np.concatenate(channel_values[channel])
        p_low  = float(np.percentile(all_vals, args.low_pct))
        p_high = float(np.percentile(all_vals, args.high_pct))
        stats[channel] = {'p_low': p_low, 'p_high': p_high}
        print(f'  {channel}: p{args.low_pct}={p_low:.4f}  p{args.high_pct}={p_high:.4f}  '
              f'N={len(all_vals):,}')

    stats['_meta'] = {
        'low_pct': args.low_pct,
        'high_pct': args.high_pct,
        'n_files_per_channel': args.max_samples,
        'npy_dir': args.npy_dir,
    }

    os.makedirs(os.path.dirname(os.path.abspath(args.out_file)), exist_ok=True)
    with open(args.out_file, 'w') as f:
        json.dump(stats, f, indent=2)
    print(f'\nSaved normalisation stats → {args.out_file}')


if __name__ == '__main__':
    main()
```

### 2.4 Implementation: Changes to `proteogram/v2/proteogram.py`

#### 2.4.1 New static method: `normalize_map_global()`

Add alongside the existing `normalize_map()`:

```python
@staticmethod
def normalize_map_global(arr: np.ndarray,
                          p_low: float,
                          p_high: float) -> tuple[np.ndarray, str]:
    """Normalise an energy/property map to [0, 255] using corpus-level percentile bounds.

    Unlike normalize_map(), which uses per-protein min/max, this method
    applies fixed bounds derived from the full training corpus so that
    inter-protein energy scale is preserved in pixel values.

    Zero values (unfilled lower-triangle entries) are mapped to 128 (mid-grey)
    to distinguish them visually from true low-energy interactions (which map
    near 0) — matching the existing gray padding convention in the training code.

    Args:
        arr:    Input energy matrix (upper triangle populated; lower = 0).
        p_low:  Lower percentile bound in physical units (kJ/mol or Å).
        p_high: Upper percentile bound in physical units.

    Returns:
        Tuple of (normalised uint8 array, error string or '').
    """
    err = ''
    try:
        scale = p_high - p_low
        if scale == 0:
            return np.full_like(arr, 128, dtype='uint8'), 'zero scale range'

        # Clip to [p_low, p_high] then scale to [0, 255]
        clipped = np.clip(arr, p_low, p_high)
        normalised = ((clipped - p_low) / scale * 255).astype('uint8')

        # Remap structural zeros (unfilled lower triangle) to mid-grey (128)
        # so they don't contaminate the 0-end of the energy scale
        normalised[arr == 0] = 128

    except Exception as e:
        err = f'Problem in normalize_map_global: {e}'
        normalised = np.full_like(arr, 128, dtype='uint8')
    return normalised, err
```

#### 2.4.2 Update `calculate_proteogram()` to accept `norm_stats`

Modify the method signature and normalisation block:

```python
def calculate_proteogram(self,
                          return_simulated_pdb: bool = False,
                          debug: bool = False,
                          subtract_solvent_energies: bool = True,
                          memory_efficient: bool = False,
                          norm_stats: dict = None):   # <-- NEW parameter
    """
    ... (existing docstring) ...

    Args:
        ...
        norm_stats: Optional dict loaded from norm_stats.json. When supplied,
                    normalize_map_global() is used for all 6 channels instead
                    of per-protein min-max. Keys: 'vdw_attractive', 'vdw_repulsive',
                    'es_attractive', 'es_repulsive', 'distance', 'hydrophobicity'.
                    Each value is a dict with 'p_low' and 'p_high'.
    """
    # ... existing MD pipeline code unchanged ...

    # ---- Replace the normalisation block ----
    def _norm(arr, channel_name):
        if norm_stats and channel_name in norm_stats:
            s = norm_stats[channel_name]
            return self.normalize_map_global(arr, s['p_low'], s['p_high'])
        return self.normalize_map(arr)

    norm_disto_map,   disto_err   = _norm(disto_map,    'distance')
    norm_hydro_map,   hydro_err   = _norm(hydro_map,    'hydrophobicity')
    norm_vdw_att_map, vdw_att_err = _norm(vdw_e_att,   'vdw_attractive')
    norm_vdw_rep_map, vdw_rep_err = _norm(vdw_e_rep,   'vdw_repulsive')
    norm_es_att_map,  es_att_err  = _norm(es_e_att,    'es_attractive')
    norm_es_rep_map,  es_rep_err  = _norm(es_e_rep,    'es_repulsive')
    # ... rest of stacking unchanged ...
```

#### 2.4.3 Update `create_v2_proteograms.py`

```python
# Add to argparse
parser.add_argument('--global_norm', action='store_true',
                    help='Use global percentile normalisation bounds from norm_stats.json '
                         'instead of per-protein min-max. Requires --norm_stats_file.')
parser.add_argument('--norm_stats_file', type=str, default=None,
                    help='Path to norm_stats.json produced by compute_norm_stats.py.')
parser.add_argument('--save_npy_matrices', action='store_true',
                    help='Save raw energy matrices as .npy files alongside proteogram JPGs. '
                         'Required input for compute_norm_stats.py.')

# Load norm_stats once before the proteogram creation loop
norm_stats = None
if args.global_norm:
    if not args.norm_stats_file or not os.path.exists(args.norm_stats_file):
        raise ValueError('--global_norm requires --norm_stats_file pointing to norm_stats.json')
    import json
    with open(args.norm_stats_file) as f:
        norm_stats = json.load(f)
    print(f'Loaded global norm stats from {args.norm_stats_file}')

# Pass norm_stats into the ProteogramV2 call inside the creation loop
proteogram_data, errors = prot.calculate_proteogram(
    subtract_solvent_energies=True,
    memory_efficient=args.memory_efficient,
    norm_stats=norm_stats,   # <-- new
)
```

### 2.5 End-to-End Workflow

```bash
# Step 1: Generate proteograms with raw .npy matrix saving (first pass or subset)
python create_v2_proteograms.py --save_npy_matrices

# Step 2: Compute global bounds from saved matrices
python compute_norm_stats.py \
    --npy_dir /path/to/proteograms/energy_matrices \
    --out_file /path/to/norm_stats.json \
    --max_samples 5000

# Step 3: Re-generate proteograms using global normalisation
python create_v2_proteograms.py \
    --global_norm \
    --norm_stats_file /path/to/norm_stats.json \
    --overwrite
```

### 2.6 Validation Steps

#### Step 1 — Sanity check: pixel distribution

For a random sample of 100 proteograms, compare the pixel value histograms between per-protein and global normalisation:

```python
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import glob

per_protein_files = glob.glob('/path/to/proteograms_per_protein/*.jpg')[:100]
global_files      = glob.glob('/path/to/proteograms_global/*.jpg')[:100]

for label, files in [('per-protein', per_protein_files), ('global', global_files)]:
    pixels = np.concatenate([np.array(Image.open(f)).ravel() for f in files])
    plt.hist(pixels, bins=50, alpha=0.6, label=label)

plt.legend()
plt.xlabel('Pixel value')
plt.title('Pixel distribution: per-protein vs. global normalisation')
plt.savefig('norm_comparison.png', dpi=150)
```

Expected result: global normalisation produces a wider, less clipped distribution with meaningful variation near 0 and 255. Per-protein should look nearly uniform (every image uses the full range).

#### Step 2 — Visual inspection

Side-by-side comparison of the same protein normalised both ways:

```python
from PIL import Image
import matplotlib.pyplot as plt

pdb_id = 'd3kfda_'
per_protein = Image.open(f'/path/per_protein/{pdb_id}.jpg')
global_norm  = Image.open(f'/path/global/{pdb_id}.jpg')

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
ax1.imshow(per_protein); ax1.set_title('Per-protein normalisation'); ax1.axis('off')
ax2.imshow(global_norm);  ax2.set_title('Global normalisation');     ax2.axis('off')
plt.savefig(f'{pdb_id}_norm_comparison.png', dpi=150)
```

Proteins with strong hydrophobic cores (e.g., globins, immunoglobulins) should appear noticeably brighter in the VdW channels under global normalisation compared to proteins with weak packing.

#### Step 3 — Downstream MAP@K comparison

Retrain the ResNet18 model on global-normalised proteograms and compare MAP@K on the eval set against the baseline model trained on per-protein normalised proteograms. Expected outcome: global normalisation improves MAP@K at the fold and superfamily levels (where energy magnitude differences are most discriminative), with neutral or marginal effect at the class level.

#### Step 4 — Robustness check: unseen proteins

Compute the fraction of pixel values clipped to 0 or 255 for 50 randomly selected held-out proteins (not used in `compute_norm_stats.py`). If > 5% of non-zero pixels are clipped, the percentile bounds are too tight and `--low_pct`/`--high_pct` should be widened (e.g., 0.5 and 99.5).

---

## 3. Grad-CAM Explainability

### 3.1 Motivation

When Proteogram reports that two proteins share 87% cosine similarity, a structural biologist naturally asks: *which residue-residue interactions drove that score?* Currently there is no answer — the model is a black box.

Because proteogram pixels directly encode pairwise residue interactions (pixel at row `i`, column `j` represents the interaction between residue `i` and residue `j`), a saliency heatmap over the input image is directly interpretable as a **residue-pair importance map**. This is a unique property of the proteogram representation that does not exist for most CV tasks.

Grad-CAM (Gradient-weighted Class Activation Mapping) computes a heatmap by backpropagating the gradient of a target score through the last convolutional layer of the CNN. High activation regions in the heatmap indicate which spatial features (and thus which residue pairs) most influenced the model's output.

### 3.2 Design Decisions

#### Target layer selection

For ResNet18, the natural target is the output of `layer4` (the last residual block), which has spatial resolution 7×7 for 224px input or 13×13 for 200px padded proteograms. This gives meaningful spatial resolution after upsampling back to the full NxN image.

For the custom ConvNet, the target is `block4` (after the 4th MaxPool, spatial resolution ≈ 12×12 for 200px input).

#### Score to differentiate

Standard Grad-CAM differentiates with respect to the **class logit** for a classification task. For a *retrieval* task we instead differentiate with respect to the **cosine similarity score** between a query and a target embedding. This gives a "similarity-attribution" heatmap: *which parts of the query proteogram, when activated, push the cosine similarity with the target higher?*

Formally, if `f_q` and `f_t` are the embedding vectors for query and target:

```
S = cos(f_q, f_t) = (f_q · f_t) / (||f_q|| · ||f_t||)
```

We compute `∂S / ∂A_k` for each activation map `A_k` in the target convolutional layer.

#### Output format

The Grad-CAM heatmap is:
- An NxN float32 array in [0, 1] — matching the proteogram dimensions
- Saved as both a matplotlib figure (with residue axis labels) and a raw `.npy` file
- Overlaid as a semi-transparent colour map on top of the original proteogram image

### 3.3 Implementation

#### 3.3.1 New method: `Img2Vec.gradcam_similarity()`

Add to `proteogram/v2/image_similarity.py`:

```python
def gradcam_similarity(self,
                        query_image_path: str,
                        target_image_path: str,
                        output_dir: str,
                        query_sequence: str = None,
                        target_sequence: str = None) -> np.ndarray:
    """Compute a Grad-CAM residue-pair importance map for a query→target similarity.

    The heatmap shows which residue-pair interactions in the QUERY proteogram
    most influence the cosine similarity with the TARGET proteogram.

    The model must be a ResNet18 fine-tuned with train_multiple_models.py
    (--model resnet18) or the from-scratch ConvNet (--model cnn).

    Args:
        query_image_path:  Path to the query proteogram JPG.
        target_image_path: Path to the target proteogram JPG.
        output_dir:        Directory to save the heatmap figure and .npy file.
        query_sequence:    Optional 1-letter amino acid sequence for axis labels.
        target_sequence:   Optional 1-letter amino acid sequence (unused currently,
                           reserved for cross-proteogram attribution in future).

    Returns:
        np.ndarray: Upsampled Grad-CAM heatmap, shape (H, W), values in [0, 1].
    """
    import torch.nn.functional as F

    os.makedirs(output_dir, exist_ok=True)

    # ------------------------------------------------------------------ #
    # 1. Identify the target convolutional layer                          #
    # ------------------------------------------------------------------ #
    target_layer = self._get_gradcam_target_layer()

    # ------------------------------------------------------------------ #
    # 2. Register forward/backward hooks                                  #
    # ------------------------------------------------------------------ #
    activations = {}
    gradients = {}

    def _save_activation(module, input, output):
        activations['value'] = output.detach()

    def _save_gradient(module, grad_input, grad_output):
        gradients['value'] = grad_output[0].detach()

    fwd_hook = target_layer.register_forward_hook(_save_activation)
    bwd_hook = target_layer.register_full_backward_hook(_save_gradient)

    try:
        # ------------------------------------------------------------------ #
        # 3. Forward pass for both query and target                           #
        # ------------------------------------------------------------------ #
        query_tensor  = self._load_and_preprocess(query_image_path)   # (1, 3, H, W)
        target_tensor = self._load_and_preprocess(target_image_path)  # (1, 3, H, W)

        # Embeddings from the penultimate layer
        # Switch to full model (not self.embed which stripped the head)
        self.model.eval()
        query_tensor  = query_tensor.to(self.device).requires_grad_(True)
        target_tensor = target_tensor.to(self.device)

        # Get embedding for query (triggers forward hook and saves activations)
        query_feat  = self.embed(query_tensor)                   # (1, d)
        target_feat = self.embed(target_tensor).detach()         # (1, d)

        # ------------------------------------------------------------------ #
        # 4. Compute cosine similarity and differentiate                      #
        # ------------------------------------------------------------------ #
        # Manually compute cosine similarity (not through nn.CosineSimilarity
        # so we can call backward on the scalar)
        q_norm = F.normalize(query_feat, dim=1)
        t_norm = F.normalize(target_feat, dim=1)
        cos_sim = (q_norm * t_norm).sum()                        # scalar

        self.model.zero_grad()
        cos_sim.backward()

        # ------------------------------------------------------------------ #
        # 5. Compute Grad-CAM weights                                         #
        # ------------------------------------------------------------------ #
        grads  = gradients['value']           # (1, C, h, w)
        acts   = activations['value']         # (1, C, h, w)

        # Global-average-pool the gradients over the spatial dims → (1, C, 1, 1)
        weights = grads.mean(dim=(2, 3), keepdim=True)

        # Weighted combination of activation maps → (1, 1, h, w)
        cam = (weights * acts).sum(dim=1, keepdim=True)
        cam = F.relu(cam)                      # keep only positive contributions

        # Normalise to [0, 1]
        cam_min, cam_max = cam.min(), cam.max()
        if cam_max > cam_min:
            cam = (cam - cam_min) / (cam_max - cam_min)

        # ------------------------------------------------------------------ #
        # 6. Upsample to input image size                                     #
        # ------------------------------------------------------------------ #
        input_h = query_tensor.shape[2]
        input_w = query_tensor.shape[3]
        cam_upsampled = F.interpolate(cam,
                                       size=(input_h, input_w),
                                       mode='bilinear',
                                       align_corners=False)
        cam_np = cam_upsampled.squeeze().cpu().numpy()   # (H, W)

    finally:
        fwd_hook.remove()
        bwd_hook.remove()

    # ------------------------------------------------------------------ #
    # 7. Save outputs                                                      #
    # ------------------------------------------------------------------ #
    query_name  = os.path.splitext(os.path.basename(query_image_path))[0]
    target_name = os.path.splitext(os.path.basename(target_image_path))[0]
    stem = f'{query_name}_vs_{target_name}'

    # Save raw heatmap
    npy_path = os.path.join(output_dir, f'{stem}_gradcam.npy')
    np.save(npy_path, cam_np)

    # Save overlay figure
    query_img = np.array(Image.open(query_image_path).convert('RGB'))
    self._save_gradcam_figure(
        query_img=query_img,
        cam=cam_np,
        cos_sim=cos_sim.item(),
        query_name=query_name,
        target_name=target_name,
        output_dir=output_dir,
        query_sequence=query_sequence,
    )

    print(f'Grad-CAM saved → {output_dir}/{stem}_gradcam.png')
    return cam_np


def _get_gradcam_target_layer(self):
    """Return the last convolutional layer for Grad-CAM based on architecture."""
    children = list(self.model.children())
    # ResNet18: children order is conv1, bn1, relu, maxpool, layer1, layer2, layer3, layer4, avgpool, fc
    # Find the last nn.Sequential that contains Conv2d layers
    target = None
    for child in children:
        if isinstance(child, nn.Sequential):
            for submodule in child.modules():
                if isinstance(submodule, nn.Conv2d):
                    target = child
    if target is None:
        # Fallback: use the last Conv2d found anywhere in the model
        for module in self.model.modules():
            if isinstance(module, nn.Conv2d):
                target = module
    return target


def _load_and_preprocess(self, image_path: str) -> torch.Tensor:
    """Load and preprocess a single proteogram image matching training transforms."""
    img = Image.open(image_path).convert('RGB')
    # Apply the same pad-to-200 + ImageNet normalisation used in training
    from torchvision import transforms as T
    import numpy as np
    arr = np.array(img)
    H, W = arr.shape[:2]
    target = 200

    def get_pad(curr, tgt):
        d = tgt - curr
        if d <= 0:
            return (0, 0)
        p1 = d // 2
        return (p1, d - p1)

    padding = (get_pad(H, target), get_pad(W, target), (0, 0))
    arr = np.pad(arr, padding, constant_values=128)
    arr = arr[:target, :target, :]
    img_padded = Image.fromarray(arr.astype('uint8'))

    transform = T.Compose([
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    return transform(img_padded).unsqueeze(0)   # (1, 3, H, W)


def _save_gradcam_figure(self,
                          query_img: np.ndarray,
                          cam: np.ndarray,
                          cos_sim: float,
                          query_name: str,
                          target_name: str,
                          output_dir: str,
                          query_sequence: str = None) -> None:
    """Save a 3-panel Grad-CAM figure: original, heatmap, overlay."""
    import matplotlib.pyplot as plt
    import matplotlib.cm as cm
    from matplotlib.colors import Normalize

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle(
        f'Grad-CAM: {query_name}  →  {target_name}  (cosine similarity = {cos_sim:.4f})',
        fontsize=13, y=1.01
    )

    # Panel 1: original query proteogram
    axes[0].imshow(query_img)
    axes[0].set_title('Query proteogram', fontsize=11)
    axes[0].axis('off')

    # Panel 2: Grad-CAM heatmap alone
    im = axes[1].imshow(cam, cmap='hot', vmin=0, vmax=1)
    axes[1].set_title('Grad-CAM heatmap\n(high = important residue pairs)', fontsize=11)
    axes[1].axis('off')
    plt.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04)

    # Panel 3: overlay (proteogram + semi-transparent heatmap)
    axes[2].imshow(query_img)
    overlay = axes[2].imshow(cam, cmap='hot', alpha=0.55, vmin=0, vmax=1)
    axes[2].set_title('Overlay', fontsize=11)
    axes[2].axis('off')
    plt.colorbar(overlay, ax=axes[2], fraction=0.046, pad=0.04)

    # Optional: add residue index tick labels if sequence is provided
    if query_sequence and len(query_sequence) <= 200:
        step = max(1, len(query_sequence) // 20)   # show ~20 tick labels
        ticks = list(range(0, len(query_sequence), step))
        tick_labels = [f'{i}\n{query_sequence[i]}' for i in ticks]
        for ax in axes:
            ax.set_xticks(ticks); ax.set_xticklabels(tick_labels, fontsize=6)
            ax.set_yticks(ticks); ax.set_yticklabels(tick_labels, fontsize=6)
            ax.tick_params(axis='both', length=2)

    plt.tight_layout()
    stem = f'{query_name}_vs_{target_name}'
    fig_path = os.path.join(output_dir, f'{stem}_gradcam.png')
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
```

#### 3.3.2 New script: `scripts/v2/explain_similarity.py`

```python
#!/usr/bin/env python
"""Generate Grad-CAM residue-pair importance maps for similar protein pairs.

For each query in the eval set, explains the similarity to its top-1 hit
(or a user-specified target) using Grad-CAM over the last convolutional layer.

Usage — explain top-1 hit for all eval proteograms:
    python explain_similarity.py

Usage — explain a specific query→target pair:
    python explain_similarity.py \\
        --query /path/to/d3kfda_.jpg \\
        --target /path/to/d1yl4r1.jpg \\
        --query_seq ACDEFGHIKLMNPQRSTVWY... \\
        --output_dir gradcam_results/

Usage — explain top-K hits for the 50 queries with lowest MAP@K (worst cases):
    python explain_similarity.py --explain_worst 50 --top_k 3
"""
import argparse
import os
import pickle
import torch
import pandas as pd

from proteogram.v2 import Img2Vec
from proteogram.common import read_yaml


def main():
    parser = argparse.ArgumentParser(description='Grad-CAM explainability for Proteogram.')
    parser.add_argument('--query', '-q', type=str, default=None,
                        help='Path to a single query proteogram JPG.')
    parser.add_argument('--target', '-t', type=str, default=None,
                        help='Path to a single target proteogram JPG.')
    parser.add_argument('--query_seq', type=str, default=None,
                        help='1-letter amino acid sequence of the query protein '
                             '(optional, for residue axis labels).')
    parser.add_argument('--output_dir', '-o', type=str, default='gradcam_output',
                        help='Directory to save Grad-CAM figures and .npy files.')
    parser.add_argument('--explain_worst', type=int, default=None,
                        help='Explain the top-1 hit for the N queries with the lowest '
                             'MAP@K score (most informative failures). '
                             'Requires proteogram_sim_results in config.yml.')
    parser.add_argument('--top_k', type=int, default=1,
                        help='Number of top hits to explain per query (default: 1).')
    args = parser.parse_args()

    config = read_yaml('config.yml')
    model_file = config['model_file']
    embed_file = config['embed_file']
    corpus_dir  = config['proteograms_for_sim_dir']

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    img_sim = Img2Vec(model_file, dataset_dir=[], device=device)

    # Load corpus embeddings
    with open(embed_file, 'rb') as f:
        img_sim.dataset = pickle.load(f)

    os.makedirs(args.output_dir, exist_ok=True)

    if args.query and args.target:
        # Single pair mode
        img_sim.gradcam_similarity(
            query_image_path=args.query,
            target_image_path=args.target,
            output_dir=args.output_dir,
            query_sequence=args.query_seq,
        )

    elif args.explain_worst:
        # Explain worst-performing queries from similarity results
        results_file = config.get('proteogram_sim_results')
        if not results_file or not os.path.exists(results_file):
            raise FileNotFoundError(
                f'proteogram_sim_results not found: {results_file}. '
                'Run measure_similarity_v2.py first.')

        results_df = pd.read_csv(results_file, sep='\t')
        # Sort by MAP@K score (ascending = worst first) if the column exists,
        # otherwise just take the last N rows as a proxy
        n = args.explain_worst
        queries_to_explain = results_df.head(n)

        for _, row in queries_to_explain.iterrows():
            query_path = row.iloc[0]
            top_hits = [row.iloc[k+1].split(',')[0]   # filename part of 'filename,score'
                        for k in range(min(args.top_k, len(row) - 1))]
            for target_stem in top_hits:
                target_path = os.path.join(corpus_dir, target_stem + '.jpg')
                if not os.path.exists(target_path):
                    print(f'Target not found, skipping: {target_path}')
                    continue
                img_sim.gradcam_similarity(
                    query_image_path=query_path,
                    target_image_path=target_path,
                    output_dir=args.output_dir,
                )
    else:
        parser.print_help()


if __name__ == '__main__':
    main()
```

### 3.4 Validation Steps

#### Step 1 — Sanity check: heatmap is not uniform

For 10 random query-target pairs, verify the heatmap has meaningful spatial variation (std > 0.05):

```python
import numpy as np
import glob

npy_files = glob.glob('gradcam_output/*_gradcam.npy')
for f in npy_files:
    cam = np.load(f)
    std = cam.std()
    max_val = cam.max()
    print(f'{f}: std={std:.4f}  max={max_val:.4f}')
    assert std > 0.05, f'Heatmap appears uniform for {f} — check hook registration'
```

#### Step 2 — Biological sanity check

Run Grad-CAM on a well-studied protein pair from the same SCOPe superfamily (e.g., two globins: haemoglobin α-chain vs. myoglobin). Expect high activation at:
- The haem-binding pocket region (residues ~60–90 and ~130–150 in the sequence)
- The conserved F-helix contacts
- The hydrophobic core residues

Cross-reference high-activation residue pairs against the known structural alignment from US-align or GTalign. If the top-10 residue pairs by Grad-CAM score overlap significantly with the US-align-identified structurally equivalent residue pairs, the explainer is working correctly.

#### Step 3 — Negative control

Run Grad-CAM on a query-target pair from *different* SCOPe classes (e.g., an all-alpha vs. an all-beta protein with low cosine similarity score ~0.3). The heatmap should be diffuse and low-magnitude — no clear hotspot — because no specific structural motif is driving the (low) similarity.

```python
# Confirm: mean activation for negative pairs should be < mean activation for positive pairs
import numpy as np

positive_cams = [np.load(f) for f in glob.glob('gradcam_output/same_class_*.npy')]
negative_cams = [np.load(f) for f in glob.glob('gradcam_output/diff_class_*.npy')]

pos_mean = np.mean([c.max() for c in positive_cams])
neg_mean = np.mean([c.max() for c in negative_cams])
print(f'Positive pairs max activation: {pos_mean:.4f}')
print(f'Negative pairs max activation: {neg_mean:.4f}')
assert pos_mean > neg_mean, 'Grad-CAM not discriminating positive/negative pairs'
```

#### Step 4 — Hook cleanup test

Verify hooks are always removed even when an exception occurs mid-computation (hooks left dangling slow down subsequent forward passes and may accumulate memory):

```python
# Deliberately pass a corrupted image path and confirm no hook leakage
from proteogram.v2 import Img2Vec
import torch

img_sim = Img2Vec(model_file, dataset_dir=[], device='cpu')
img_sim.dataset = {}  # minimal setup

hook_count_before = len(list(img_sim.model._forward_hooks.values()))
try:
    img_sim.gradcam_similarity('/nonexistent/query.jpg', '/nonexistent/target.jpg', '/tmp')
except Exception:
    pass
hook_count_after = len(list(img_sim.model._forward_hooks.values()))
assert hook_count_before == hook_count_after, 'Forward hooks leaked after exception!'
print('Hook cleanup test passed.')
```

---

## 4. Combined Integration Checklist

Before merging all three changes, run through this checklist end-to-end on the eval set:

```
[ ] pip install faiss-cpu  (or faiss-gpu) added to pyproject.toml
[ ] compute_norm_stats.py runs without error on 500 random energy matrices
[ ] norm_stats.json is committed to the repository alongside the dataset
[ ] create_v2_proteograms.py --global_norm produces visually distinct images
    vs. per-protein normalised equivalents (visual inspection on 5 proteins)
[ ] measure_similarity_v2.py --faiss produces sim_dict identical in format to
    existing brute-force output (evaluate_methods_v2.py accepts it unchanged)
[ ] FAISS Recall@5 ≥ 0.99 × brute-force Recall@5 (automated test passes)
[ ] MAP@K from FAISS results within ±0.005 of MAP@K from brute-force results
[ ] explain_similarity.py runs on a single pair and produces a 3-panel PNG
[ ] Grad-CAM heatmap std > 0.05 for same-class pairs (sanity check passes)
[ ] No memory leaks: all forward/backward hooks removed after gradcam_similarity()
[ ] All existing tests in scripts/v2/tests/ still pass
[ ] README.md updated with:
      - New --global_norm / --norm_stats_file flags in Step 1
      - New --faiss / --faiss_pq flags in Step 4
      - New explain_similarity.py in the scripts reference table
```

---

## 5. Config additions (`scripts/v2/config.example.yml`)

Add these keys to the example config for discoverability:

```yaml
# ── Global normalisation (Improvement 2) ──────────────────────────────
# Path to norm_stats.json produced by compute_norm_stats.py.
# Required when running create_v2_proteograms.py --global_norm.
norm_stats_file: /path/to/norm_stats.json

# ── FAISS index (Improvement 1) ───────────────────────────────────────
# Path to save/load the FAISS index (auto-derived from embed_file if omitted).
faiss_index_file: /path/to/corpus_embeddings.faiss

# ── Grad-CAM output (Improvement 3) ───────────────────────────────────
# Directory to write Grad-CAM figures and .npy heatmap files.
gradcam_output_dir: /path/to/gradcam_output
```
