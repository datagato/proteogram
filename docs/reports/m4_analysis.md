# M4 Ablation Analysis — Detailed Review

**Date:** 2026-05-04  
**Scope:** Full evaluation of the v2 composite similarity stack (M1–M4) against
the baseline cosine. Covers experiment design, result interpretation, bugs, and
what the numbers actually mean vs. what is being claimed.

---

## 1. Experiment Structure

The ablation tests six variants of the similarity pipeline against 168 synthetic
queries (12 folds × 14 samples, 32×32 px proteograms, K=10):

| Variant | Description |
|---|---|
| `baseline_joint_cos` | Whole-image cosine on joint embedding (current v2) |
| `A_two_stream_cos` | Geometry cosine + chemistry cosine (two separate streams) |
| `B_plus_mahalanobis` | A + RBF-Mahalanobis kernel on joint embedding |
| `C_plus_emd` | A + Earth Mover's Distance over energy histograms |
| `D_full_composite` | A + B + C combined with grid-searched weights (0,0,0.1,0.9) |
| `E_plus_rerank` | D + cross-encoder re-ranker on top-20 candidates |

---

## 2. Raw Results Table

| Variant | P@1 | P@5 | Recall@10 | MAP | NDCG@10 | MRR |
|---|---|---|---|---|---|---|
| baseline | 1.000 | 0.9988 | **0.7683** | 0.9974 | 0.9995 | 1.000 |
| A | 1.000 | 0.9988 | **0.7683** | 0.9974 | 0.9995 | 1.000 |
| B | 1.000 | 0.9988 | **0.7679** | 0.9974 | 0.9995 | 1.000 |
| C | 1.000 | 0.9988 | **0.7683** | 0.9984 | 0.9998 | 1.000 |
| D | 1.000 | **1.0000** | **0.7669** | 0.9968 | 0.9999 | 1.000 |
| E | 1.000 | **1.0000** | **0.7692** | **1.0000** | **1.0000** | 1.000 |

### Wilcoxon signed-rank vs baseline (p < 0.01 = significant ✓, p < 0.05 = marginal ·)

| Variant | P@1 | P@5 | Recall@10 | MAP | NDCG@10 | MRR |
|---|---|---|---|---|---|---|
| A | p=1 | p=1 | p=1 | p=1 | p=1 | p=1 |
| B | p=1 | p=1 | p=1 | p=1 | p=1 | p=1 |
| C | p=1 | p=1 | p=1 | p=0.028 · | p=1 | p=1 |
| D | p=1 | p=1 | p=0.31 | **p=0.007 ✓** | p=0.92 | p=1 |
| E | p=1 | p=1 | p=1 | p=0.025 · | p=1 | p=1 |

---

## 3. What the Numbers Actually Mean

### 3.1 Recall@10 is at the theoretical ceiling

With 12 folds of 14 samples each, a query has exactly **13 relevant documents**
in the database (14 minus itself). Retrieving K=10 gives a hard ceiling of:

```
max Recall@10 = 10 / 13 = 0.7692...
```

The baseline already achieves **0.7683** — within 0.001 of the ceiling.
Variant E reaches exactly **0.7692**, the mathematical maximum. There is no
room for any variant to demonstrate improvement on Recall@10 on this dataset.

Every report headline that treats Recall@10 differences as meaningful is
misleading. Differences in the 4th decimal place are noise, not signal.

### 3.2 D is statistically significantly WORSE than baseline on MAP

The implementation summary states: *"D and E significantly improve MAP over
baseline (Wilcoxon p ≤ 0.025)."* This is incorrect for D.

Looking at raw MAP values:
- Baseline MAP: **0.9974**
- D MAP: **0.9968**

D's MAP is lower, not higher. The Wilcoxon test (W=84.5, z=−2.698,
p=0.007) is significant, but the negative z-statistic confirms D is
**significantly worse** than baseline on MAP. The only variant that
genuinely improves MAP is E (MAP=1.0000, p=0.025).

### 3.3 Variant A is completely identical to baseline

Two-stream cosine (A) is bit-for-bit identical to the baseline on every single
metric:
- Same Recall@10: 0.7683150183150185
- Same MAP: 0.9973868168788446
- Same NDCG: 0.999459768183303
- Wilcoxon W=0, z=0, p=1 on everything

The two-stream encoder is producing embeddings that yield exactly the same
ranking as the whole-image baseline cosine. The joint embedding `e = [e_U ⊕ e_L]`
is being compared with joint cosine in both cases. The two-stream decomposition
is providing zero lift over baseline on this dataset.

### 3.4 P@5 improvement in D and E is a single tie-break

Baseline P@5 = 0.9988095... = 167/168 perfect. D and E reach P@5 = 1.0000.
This is a difference of exactly one query (168 × 0.9988 ≈ 167 correct vs 168).
One query's top-5 shifted from "one miss" to "all correct". The bootstrap CI
for D's P@5 is [1.000, 1.000] with zero variance — there is no statistical
content here.

---

## 4. Core Issues

### Issue 1 — Histogram binning bug (features.py:93–95) — CRITICAL

For log-scale channels (`vdw_att`, `vdw_rep`, `es_att`, `es_rep`) the code
applies `log1p` to values and then bins on a linear grid:

```python
# features.py lines 93–95
if spec.log_scale:
    vals = np.log1p(np.clip(vals, 0.0, None))   # transforms to [0, log1p(max)]
edges = np.linspace(spec.range_lo, spec.range_hi, spec.n_bins + 1)  # [0, 12] linear
```

`DEFAULT_SPECS` sets `range_hi = 12.0` for these channels. After `log1p`,
values live in `[0, log1p(12)] ≈ [0, 2.56]`. The linear grid spans `[0, 12]`.
Every single value after transformation falls into the first ~21% of the bin
range. The remaining ~21 bins are always empty.

This means EMD on these channels is comparing histograms where all mass is
jammed into the leftmost few bins, regardless of the actual energy distribution.
The EMD values it produces are meaningful only in the sense that they are
consistently computed — but the bins do not correspond to interpretable energy
ranges.

**Fix:** After `log1p`, either (a) set `range_hi = log1p(original_range_hi)`,
or (b) apply the log1p inside the bin-edge computation:

```python
if spec.log_scale:
    vals = np.log1p(np.clip(vals, 0.0, None))
    edges = np.linspace(0.0, np.log1p(spec.range_hi), spec.n_bins + 1)
else:
    edges = np.linspace(spec.range_lo, spec.range_hi, spec.n_bins + 1)
```

### Issue 2 — Grid search overfits on 84 samples and weights don't transfer

The M2 grid search runs on 84 samples and finds weights `(0.0, 0.0, 0.1, 0.9)`.
On the M2 84-sample val split, D achieves Recall@10 = 1.000 (perfect). On the
full M4 set of 168 queries, D's Recall@10 = 0.7669, which is slightly worse
than baseline.

The weights learned on 84 synthetic samples with a specific confusable-pair
structure do not generalize. The grid at step=0.1 has only 66 points on the
3-simplex, which is far too coarse for reliable weight selection. The ~0.001
differences that separate variants on this dataset are smaller than the grid
resolution.

### Issue 3 — Synthetic dataset cannot validate the method

The design doc (§8 M1 gate) requires "≥ +2 pp Recall@10 at fold level." The
synthetic dataset makes this ungradeable: at ceiling (0.7692), no method can
achieve +2 pp. All variants are within ±0.0014 of baseline on Recall@10.

For a 12-fold dataset where each fold has 14 samples, retrieval is too easy.
Cosine similarity on random embeddings would likely perform near-ceiling too,
because within-fold samples are constructed to be similar and between-fold
samples different. The 3 "confusable" fold pairs give the EMD term something
to do, but they are a tiny fraction of the 168 × 167 / 2 pairs evaluated.

### Issue 4 — Mahalanobis metric is always diagonal (metrics.py:84)

The design doc (§4.3) specifies an `L` matrix of shape `64×512` trained by
triplet loss, allowing anisotropy. The implementation fits a closed-form
diagonal from within-class variance:

```python
L = np.diag(1.0 / np.sqrt(within)).astype(np.float32)   # metrics.py line 84
```

This is a `(D, D)` matrix where D=128 (joint embedding dim), but all off-diagonal
entries are zero. It cannot capture correlations between dimensions. The M2
report lists `L_shape: [128, 128]` which makes it look full-rank, but it is
not — it is a scaled identity.

This is the correct lightweight fallback described in the `fit_diagonal_from_pairs`
docstring, but the M2 report headline ("Mahalanobis L matrix shape: 128×128")
implies a full learned metric that doesn't exist yet.

### Issue 5 — Re-ranker teacher is step-function synthetic scores

The M3 re-ranker is trained against a synthetic teacher that produces scores
≈0.9 for same-fold pairs and ≈0.15 for different-fold pairs. This is
near-binary. On real data, TM-score is continuous and noisy across [0, 1].
The re-ranker AUROC saturating at 1.000 by epoch 3 confirms it is learning a
near-trivial binary classifier, not a structural similarity regressor. The
M3 report itself notes this; the concern is that the M4 ablation treats variant
E as a validated component when the re-ranker's real-data behavior is unknown.

### Issue 6 — Wall-clock comparison is brute-force, not ANN

The design doc (§5.4) claims "<1 ms per query for ANN over `e`." The M4 timings
are for **brute-force pairwise scoring** over all 168 candidates. D takes 0.91 s
for 168 queries, which is 5.4 ms per query even at this trivial scale. Adding
EMD (C) alone costs 0.68 s vs 0.029 s for baseline — a 23× slowdown. At 14k
real SCOPe structures, brute-force EMD would take minutes per query. The <1 ms
claim requires FAISS or similar ANN, which has not been implemented or measured.

---

## 5. What Actually Works vs. What Doesn't

### Works
- M1 training is clean: loss decreases monotonically, val Recall@10 improves 0.881→0.976
- Embeddings are correctly L2-normalised (norms ≈ 1.0, verified by M1 checks)
- SupCon loss implementation is sound
- EMD formula (`emd_1d`) is mathematically correct for comparing two histograms
  on the same bins — the issue is the bins themselves
- Re-ranker (M3) converges and produces scores in (0, 1) as expected
- The end-to-end pipeline runs without errors

### Doesn't Work / Not Demonstrated
- **Two-stream encoder (A) provides zero improvement over baseline** on this
  dataset — identical metrics
- **Full composite (D) is statistically significantly worse than baseline on MAP**
  (p=0.007), contradicting the design promise
- **Recall@10 cannot be measured meaningfully** — dataset ceiling effect
- **EMD histograms are binned incorrectly** for log-scale channels
- **M1 gate ("≥+2 pp Recall@10") cannot be assessed** because of the ceiling
- **M2 gate ("≥+1 pp on top of M1") is also ungradeable** for same reason

---

## 6. Interpretation

The implementation summary frames the results optimistically: D significantly
improves MAP (p=0.007), E achieves MAP=1.0. Looking at the direction of the
Wilcoxon z-statistic for D (z=−2.698), D is actually **significantly worse**
than baseline. E is better, but the improvement is marginal (p=0.025 misses
the p<0.01 threshold the design doc specifies).

The honest read of the M4 results is:

> On this synthetic dataset, the composite similarity stack shows no measurable
> benefit over the baseline joint cosine on any metric except the re-ranker (E)
> on MAP. The dataset is too easy to expose differences. The histogram
> binning bug means the EMD term is operating on malformed histograms. The
> two-stream encoder, after training, produces embeddings that rank identically
> to the whole-image baseline. Before running on real SCOPe data, the histogram
> bug must be fixed and the evaluation dataset must be replaced with one that
> has sufficient difficulty to produce spread across variants.

---

## 7. Required Actions Before Real-Data Runs

| Priority | Action | File / Location |
|---|---|---|
| P0 | Fix histogram bin edges for log-scale channels | `proteogram/v2/features.py:95` |
| P0 | Re-run M2 and M4 after histogram fix | `scripts/v2/train_metric_head.py`, `run_ablation.py` |
| P1 | Replace synthetic dataset with real SCOPe subset (≥500 queries, multiple difficulty tiers) | `scripts/v2/_synthetic_dataset.py` or new loader |
| P1 | Add Spearman ρ vs TM-score to M4 report (design doc §6.3 requirement) | `scripts/v2/run_ablation.py` |
| P2 | Implement full Mahalanobis via triplet loss (not diagonal closed-form) | `scripts/v2/train_metric_head.py` |
| P2 | Add ANN timing benchmark (FAISS / torch.cdist at SCOPe scale) | `scripts/v2/run_ablation.py` |
| P2 | Add failure analysis dump (50 best / 50 worst queries per variant) | `scripts/v2/run_ablation.py` |

---

## 8. Summary

| Claim | Status |
|---|---|
| Two-stream encoder improves over baseline | **Not demonstrated** — variant A is identical to baseline |
| Composite metric (D) improves MAP over baseline | **False** — D MAP is significantly lower (p=0.007) |
| Re-ranker (E) improves retrieval | **Marginally true on MAP** (p=0.025), not significant at p<0.01 threshold |
| M1 gate: ≥+2 pp Recall@10 | **Not assessable** — dataset at ceiling |
| M2 gate: ≥+1 pp on top of M1 | **Not assessable** — dataset at ceiling |
| M3 gate: closes ≥50% gap to GTalign | **Not assessable** — no GTalign comparison run |
| EMD histogram computation is correct | **No** — bin edges mismatched with log1p-transformed values |
| Mahalanobis L is a learned 64×512 matrix | **No** — it is a diagonal (D×D) matrix from closed-form variance |
