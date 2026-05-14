# Proteogram v2 — Similarity Metric & Embedding Design

## 1. Motivation

The current v2 search stack (`proteogram/v2/image_similarity.py`) treats a proteogram as a generic RGB image, runs it through an ImageNet-pretrained backbone (or a fine-tuned ResNet18 / from-scratch ConvNet), pulls the penultimate feature vector, and ranks candidates with a single `nn.CosineSimilarity(dim=1)` call.

That works, but it leaves several v2-specific signals on the floor:

1. **Asymmetry is meaningful.** v2 packs two physically distinct sets of maps into the upper and lower triangles. Standard CNN augmentations (horizontal/vertical flip, random crop) and a single global-pool head can wash that asymmetry out.
2. **Channels carry semantics.** R/G/B in v2 are not arbitrary — they are VdW attractive, VdW repulsive, and Cα distances (upper) and electrostatic attractive, electrostatic repulsive, and hydrophobicity Δ (lower). A monolithic embedding cannot reweight "geometry vs chemistry" per query.
3. **Cosine is rotation-invariant but distribution-blind.** Two domains with very similar embedding directions but very different *energy distributions* will score the same. v2's MD energies have rich heavy-tailed distributions that a distribution-aware metric can exploit.
4. **No re-ranking stage.** The pipeline returns the top-K from a single nearest-neighbour pass. Re-ranking with a heavier model (or with structural signal from GTalign/USalign as a teacher) is a standard win in image and document retrieval.

This document proposes (a) a redesigned embedding for v2, (b) a composite similarity metric that sits *on top of* cosine, and (c) an evaluation protocol to prove the changes are improvements rather than just changes.

---

## 2. Design overview

```
Proteogram v2 (NxN, 3ch, asymmetric)
        │
        ▼
Triangle decomposition
   ├── Upper-triangle tensor U (VdW att, VdW rep, Cα dist)
   └── Lower-triangle tensor L (ES att, ES rep, Hyd Δ)
        │
        ▼
Two-stream encoder (shared init, separate weights)
   ├── CNN-U → GeM pool → L2-norm → e_U  (256-d)
   └── CNN-L → GeM pool → L2-norm → e_L  (256-d)
        │
        ▼
Composite embedding e = [e_U ⊕ e_L]   (512-d)
        │
        ▼
Composite similarity S(q, t) =
        w₁ · cos(e_U, e_U')           ← geometry channel
      + w₂ · cos(e_L, e_L')           ← chemistry channel
      + w₃ · RBF-Mahalanobis(e, e')   ← learned metric
      + w₄ · EMD(H(q), H(t))          ← energy-distribution match
        │
        ▼
Top-K candidates → cross-encoder re-ranker (optional)
        │
        ▼
Final ranked list
```

The key design moves are: split the asymmetric image into two physically meaningful streams, learn an embedding per stream, *and* combine multiple complementary similarity signals at scoring time rather than relying on a single cosine.

---

## 3. Embedding redesign

### 3.1 What changes vs the current `Img2Vec`

| Aspect | Current (v2) | Proposed |
|---|---|---|
| Input handling | Whole RGB image, ImageNet normalisation | Split into upper/lower triangle tensors; per-channel normalisation using v2 channel statistics (vdw, es, dist, hyd) |
| Augmentations | torchvision default (resize, normalise) | **No** horizontal/vertical flip (would swap triangles); allow random N×N crops along the diagonal, jitter only on Cα-distance channel |
| Backbone | Single ResNet18 / ConvNet / ViT | Two backbones (shared init, independent weights) — one per triangle |
| Pooling | Global Average Pool | **GeM (Generalized-Mean) pool**, p learnable per stream |
| Head | `nn.Linear` classifier head, embeddings taken from penultimate | Projection MLP → L2-normalised 256-d embedding per stream |
| Loss | Cross-entropy on SCOPe class | **Supervised contrastive (SupCon)** at SCOPe fold/superfamily/family level + auxiliary CE |
| Output | One vector per image | `e_U`, `e_L`, `e = [e_U ⊕ e_L]`, plus per-image energy histograms `H(q)` |
| Persistence | `torch.save({'model','embeddings':dict})` | Same on-disk schema, but `embeddings[path]` is a `dict(e_U, e_L, H)` instead of a flat tensor |

### 3.2 Two-stream encoder

```python
class TriangleStreamEncoder(nn.Module):
    def __init__(self, backbone='resnet18', emb_dim=256):
        super().__init__()
        self.backbone_U = _build_backbone(backbone, in_ch=3)  # VdW att, VdW rep, Cα dist
        self.backbone_L = _build_backbone(backbone, in_ch=3)  # ES att, ES rep, Hyd Δ
        self.gem_U = GeM()
        self.gem_L = GeM()
        self.proj_U = nn.Sequential(nn.Linear(512, emb_dim), nn.BatchNorm1d(emb_dim))
        self.proj_L = nn.Sequential(nn.Linear(512, emb_dim), nn.BatchNorm1d(emb_dim))

    def forward(self, x):
        U, L = split_triangles(x)         # masks the other triangle to 0
        f_U  = self.gem_U(self.backbone_U(U))
        f_L  = self.gem_L(self.backbone_L(L))
        e_U  = F.normalize(self.proj_U(f_U), dim=1)
        e_L  = F.normalize(self.proj_L(f_L), dim=1)
        return e_U, e_L
```

`split_triangles` zeroes out the opposite triangle so the convolution receives a physically valid input. The diagonal is shared; either half can keep it.

### 3.3 Loss

```
L = α · SupCon(e_U, y_fold)         # geometry-aware contrastive
  + β · SupCon(e_L, y_fold)         # chemistry-aware contrastive
  + γ · SupCon([e_U⊕e_L], y_fold)   # joint contrastive
  + δ · CE(linear_head([e_U⊕e_L]), y_class)   # auxiliary classification
```

Default: α=β=0.4, γ=1.0, δ=0.2. SupCon is computed at the **fold** level by default (closer to "useful retrieval target" than `class`); a curriculum that anneals from `class` → `fold` → `superfamily` is straightforward to add.

### 3.4 Energy histograms

Per image, also persist a small fixed-length histogram per energy channel (e.g. 32 bins, log-spaced for VdW/ES, linear for distance and Hyd Δ). These are O(KB) per protein and feed the EMD term in §4.

---

## 4. Composite similarity metric

The proposed score is a learned linear combination of four terms.

### 4.1 Term 1 — Triangle-wise cosine (geometry channel)

```
s₁(q, t) = cos(e_U(q), e_U(t))
```

Captures shape/MD-geometry alignment without being polluted by chemistry mismatches. Useful when the user wants "structurally similar regardless of charge profile" (e.g. cross-family fold detection).

### 4.2 Term 2 — Triangle-wise cosine (chemistry channel)

```
s₂(q, t) = cos(e_L(q), e_L(t))
```

Captures electrostatic/hydrophobic compatibility. Useful for binding-pocket-style queries where chemistry matters as much as backbone shape.

### 4.3 Term 3 — RBF-Mahalanobis kernel on the joint embedding

A learned positive-semidefinite metric on the 512-d concatenated embedding:

```
d²(e, e') = (e − e')ᵀ M (e − e'),   M = LᵀL    # L is a learned 64×512 matrix
s₃(q, t) = exp(−γ · d²(e, e'))
```

Trained with triplet loss over (anchor, positive, negative) triples sampled at SCOPe fold level. This gives the system a **non-cosine** notion of similarity that can model anisotropy in embedding space (some directions matter more than others). It also makes the score *bounded in (0, 1]*, which is convenient for combining.

### 4.4 Term 4 — Earth Mover's Distance over energy histograms

For each of the four MD energy channels `c ∈ {VdW_att, VdW_rep, ES_att, ES_rep}`:

```
EMD_c(q, t) = W₁(H_c(q), H_c(t))    # 1-D Wasserstein on the histogram
s₄(q, t)    = exp(−κ · Σ_c EMD_c)
```

Distribution-level matching catches cases where two domains have *similar embeddings but very different energetic regimes* (think: same fold, but one is highly polar and the other isn't). EMD on a 32-bin 1-D histogram is O(B), trivial at retrieval time.

### 4.5 Combining the terms

```
S(q, t) = Σᵢ wᵢ · sᵢ(q, t)         (Σ wᵢ = 1, wᵢ ≥ 0)
```

Two ways to set `w`:

- **Validation-set grid search** — fast, no extra training; pick the simplex point that maximises Recall@10 at fold level on a held-out SCOPe split.
- **Learn-to-rank** — train a tiny MLP that takes `(s₁, s₂, s₃, s₄)` as input and outputs the score, supervised by *pairwise SCOPe-fold labels* with a margin-ranking loss. This generalises to non-linear combinations (e.g. "if `s₁` is very high, downweight `s₄`").

### 4.6 Cross-encoder re-ranker (optional, applied to top-K only)

For the top-K candidates returned by `S`, run a small transformer that consumes the **pair** of proteograms together (channel-concatenated to 6-channel input, or processed as two patches with cross-attention). It outputs a single scalar refined score.

Train it with **distilled supervision from GTalign or TM-score**: for each (q, t) pair in the training corpus, set the target to the GTalign TM-score, optimise MSE or pairwise margin. This is essentially adding a supervised structural-alignment teacher into the retrieval stack without paying its inference cost on every pair (only K = 10–50 per query).

---

## 5. Implementation plan

### 5.1 Code layout (proposed additions only)

```
proteogram/v2/
  encoders.py                # TriangleStreamEncoder, GeM, split_triangles
  metrics.py                 # cosine_split, mahalanobis_rbf, emd_histograms,
                             #   composite_score, LearnedRanker
  reranker.py                # PairwiseCrossEncoder (top-K re-ranking)
  losses.py                  # SupCon, triplet_with_mahalanobis
  features.py                # energy histograms (channel statistics)

scripts/v2/
  train_two_stream.py        # train TriangleStreamEncoder
  train_metric_head.py       # train M (Mahalanobis) and w (combiner)
  train_reranker.py          # distil from GTalign/TM-score
  measure_similarity_v2.py   # extended: --metric {cosine,composite,reranked}
  evaluate_methods_v2.py     # extended with new metrics described in §6
```

### 5.2 Backwards compatibility

- The on-disk embedding file becomes a `dict[path] -> {'e_U','e_L','H'}`. Add a loader path that detects the old `Tensor` schema and falls back to legacy cosine — no breaking change for existing users.
- `--metric cosine` keeps the current behaviour bit-for-bit. New behaviour is opt-in via `--metric composite` or `--metric reranked`.

### 5.3 Training data

- Reuse `scripts/v2/create_balanced_scope_train_eval_lists.py` for the splits.
- For the cross-encoder, generate a `pair_score.tsv` table of `(query, target, tm_score)` from existing GTalign and USalign result dirs (`gtalign_results_dir`, `usalign_results` already in `config.yml`).

### 5.4 Compute budget (per stage)

| Stage | Compute | Notes |
|---|---|---|
| TriangleStreamEncoder train | 1× A100, ~12 h on full SCOPe training split | Dominates total cost |
| Mahalanobis + combiner | 1× A100, ~30 min | Tiny model, cached embeddings |
| Cross-encoder re-ranker | 1× A100, ~6 h | Trained on top-50 candidate pairs only |
| Inference (full DB, ~14k structures) | <1 ms per query for ANN over `e`, +5 ms for cross-encoder over top-50 | Embedding once, ANN over `e`, then re-rank |

---

## 6. How to measure improvements

The whole point is that "we got better numbers on cosine" is not enough. Below is the protocol.

### 6.1 Datasets & splits

- **SCOPe 2.08** with the existing balanced splits from `create_balanced_scope_train_eval_lists.py`.
- Hold out **two** independent eval sets:
  - `eval-fold` — measures retrieval at the SCOPe **fold** level (the level v2 is trained for).
  - `eval-superfamily` — measures generalisation one level up.
- Also evaluate on a **time-held-out** set (PDB entries deposited after training cut-off) to detect overfitting to SCOPe biases.

### 6.2 Primary retrieval metrics

For every query in the eval set, score every database entry, sort, and compute:

| Metric | What it tells us |
|---|---|
| **Top-K Precision** (K=1, 5, 10) | How often the top match is in the same SCOPe fold |
| **Recall@K** (K=10, 50) | Fraction of true positives surfaced inside the top-K |
| **MAP** (Mean Average Precision) | Whole-list ranking quality |
| **NDCG@10** | Quality with graded relevance (class > fold > superfamily > family) |
| **MRR** | Reciprocal rank of the first true positive |

Report each at every SCOPe level, not just one. Numbers should improve at the level the model was trained on **and not collapse** at the others.

### 6.3 Calibration metrics

Cosine values are not really probabilities. The composite score should be calibrated against an external structural ground truth:

- **Spearman ρ between `S(q, t)` and TM-score** on a held-out pair set.
- **AUROC for "same-fold" pair classification** using `S` as the score.
- **Reliability diagram** for the cross-encoder output if it's trained as a regressor.

A composite score that improves Top-K but *worsens* Spearman vs TM-score is a warning sign — the model is gaming SCOPe categorical labels rather than learning structural similarity.

### 6.4 Comparison baselines

Run head-to-head against:

1. **Current v2 cosine** (the model in `proteogram/v2/image_similarity.py`).
2. **GTalign** (already wired into `evaluate_methods_v2.py`).
3. **US-align** (already wired in).
4. **Each ablation** of the proposed system (see §6.5).

Report wall-clock per query for each method — the composite metric should still be orders of magnitude faster than full structural alignment to be worth the complexity.

### 6.5 Ablation matrix

A composite design is only credible if each part earns its place:

| Variant | s₁ (geom cos) | s₂ (chem cos) | s₃ (RBF-Mah) | s₄ (EMD) | Re-rank |
|---|:-:|:-:|:-:|:-:|:-:|
| Baseline v2 (current)            |  whole-image cos |  —  |  —  |  —  |  —  |
| A — two-stream cosine only       | ✓ | ✓ |   |   |   |
| B — A + Mahalanobis              | ✓ | ✓ | ✓ |   |   |
| C — A + EMD                      | ✓ | ✓ |   | ✓ |   |
| D — A + B + C                    | ✓ | ✓ | ✓ | ✓ |   |
| E — D + cross-encoder re-rank    | ✓ | ✓ | ✓ | ✓ | ✓ |

Each row is run with the **same trained backbone**, so deltas attribute cleanly to each metric component.

### 6.6 Statistical significance

- For every paired comparison (e.g. "E vs Baseline on Recall@10"), report **bootstrap 95% CIs** on the per-query metric (1000 resamples) and a **Wilcoxon signed-rank** p-value.
- Treat "improvement" as significant only when the lower CI bound > 0 *and* p < 0.01.

### 6.7 Failure analysis (qualitative but mandatory)

Reuse the existing `save_bad_searches_dir` and `save_good_searches_dir` plumbing. For each variant, dump the top-K visualisation for:

- 50 queries with the largest Recall@10 *gain* over baseline → confirms what the new metric helps with.
- 50 queries with the largest Recall@10 *loss* → checks that the new metric isn't trading one failure mode for another.

Without this step you can move every aggregate number in the right direction and still ship something that's worse on the cases the user cares about.

### 6.8 Reporting template

```
Method           | P@1 | P@5 | R@10 | MAP | NDCG@10 | ρ(TM) | t/query
-----------------|-----|-----|------|-----|---------|-------|--------
Baseline v2 cos  | ... | ... | ...  | ... |  ...    |  ...  | ...
A two-stream     | ... | ... | ...  | ... |  ...    |  ...  | ...
B + Mahalanobis  | ... | ... | ...  | ... |  ...    |  ...  | ...
C + EMD          | ... | ... | ...  | ... |  ...    |  ...  | ...
D = B + C        | ... | ... | ...  | ... |  ...    |  ...  | ...
E + re-rank      | ... | ... | ...  | ... |  ...    |  ...  | ...
GTalign (ref)    | ... | ... | ...  | ... |  ...    |   1.0 | ...
US-align (ref)   | ... | ... | ...  | ... |  ...    |  ...  | ...
```

---

## 7. Risks & open questions

- **Cost of MD already dominates.** Adding a second backbone barely matters at training time, but it doubles embedding size on disk. For the full SCOPe DB this is small; worth flagging for users with much larger corpora.
- **Triangle masking and convolutions.** Zeroing one triangle creates a sharp diagonal discontinuity. If this turns out to bias the encoder, an alternative is to copy each triangle into a full N×N tensor (mirror onto the diagonal) — slightly wasteful, but removes the discontinuity.
- **Re-ranker label noise.** TM-score is a useful teacher but not a perfect ground truth (it has its own biases). Distillation should be regularised (label smoothing or quantile bucketing) so the student can outperform the teacher on category-level retrieval.
- **Learnable weights vs interpretability.** Learned `w` is more accurate but harder to explain to a domain user. Keep the validation-set grid-search baseline as a backstop interpretable variant.

---

## 8. Milestones

1. **M1 (week 1–2):** Implement `TriangleStreamEncoder`, train on existing splits, log Recall@10 vs current baseline. Gate: ≥ +2 pp Recall@10 at fold level.
2. **M2 (week 3):** Add Mahalanobis + EMD terms and a learn-to-rank combiner. Gate: ≥ +1 pp on top of M1, no regression on superfamily.
3. **M3 (week 4–5):** Distil cross-encoder re-ranker from GTalign. Gate: closes ≥ 50% of the remaining gap to GTalign on Recall@10, while staying < 10× cosine query latency.
4. **M4 (week 6):** Full ablation table, CIs, failure analysis, write-up; PR into `scripts/v2/`.
