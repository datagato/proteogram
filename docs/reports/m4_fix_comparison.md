# Before/After Fix Comparison — Histogram Binning Bug

**Fix applied:** `proteogram/v2/features.py` — log-scale channel bin edges corrected  
**Stages re-run:** M1 (to regenerate histograms), M2 (re-fit weights), M4 (ablation)

---

## The Bug and The Fix

### Root cause (`features.py:93–95`, before)

```python
if spec.log_scale:
    vals = np.log1p(np.clip(vals, 0.0, None))   # values land in [0, 2.56]
edges = np.linspace(spec.range_lo, spec.range_hi, spec.n_bins + 1)  # bins span [0, 12]
```

For channels `vdw_att`, `vdw_rep`, `es_att`, `es_rep` — all with `range_hi=12.0` and
`log_scale=True` — the `log1p` transform maps values to `[0, log1p(12)] ≈ [0, 2.56]`,
but the bin edges still span `[0, 12]` linearly. Every transformed value falls into the
first ~21% of the 32 bins. The remaining ~25 bins are always empty. EMD on these
histograms has almost no dynamic range — all proteins look nearly identical.

### Fix

```python
if spec.log_scale:
    vals = np.log1p(np.clip(vals, 0.0, None))
    edges = np.linspace(0.0, np.log1p(spec.range_hi), spec.n_bins + 1)
else:
    edges = np.linspace(spec.range_lo, spec.range_hi, spec.n_bins + 1)
```

After the fix, bins span `[0, log1p(12)] ≈ [0, 2.56]`, matching the transformed value
range. The full 32-bin space is now usable.

---

## M2 (Metric Head) — Val Split Results (84 queries)

| Variant | Recall@10 (before) | Recall@10 (after) | MAP (before) | MAP (after) |
|---|---|---|---|---|
| baseline_joint_cos | 0.9976 | 0.9974 | 0.9947 | 0.9988 |
| A_two_stream_cos | 0.9976 | 0.9974 | 0.9947 | 0.9988 |
| B_plus_mahalanobis | 0.9976 | 0.9974 | 0.9946 | 0.9988 |
| C_plus_emd | 0.9976 | 0.9974 | 0.9952 | 0.9990 |
| **D_full_composite** | **1.0000** ← reported as best | **0.9868** ← now worst | **0.9978** | **0.9767** |

**Grid-searched weights:**
- Before fix: `(w_geom, w_chem, w_mahal, w_emd) = (0.0, 0.0, 0.1, 0.9)`
- After fix: `(0.0, 0.0, 0.2, 0.8)`

In both cases the grid search discards geometry and chemistry cosines entirely
and gives nearly all weight to EMD. The old M2 report claimed D was best
(`Recall@10 = 1.000`). After the fix D is the worst variant on the val set.

---

## M4 (Full Ablation) — All 168 Queries, K=10

### Recall@10

| Variant | Before fix | After fix | Delta |
|---|---|---|---|
| baseline_joint_cos | 0.7683 | 0.7683 | 0.0000 |
| A_two_stream_cos | 0.7683 | 0.7683 | 0.0000 |
| B_plus_mahalanobis | 0.7679 | 0.7683 | +0.0005 |
| C_plus_emd | 0.7683 | 0.7683 | 0.0000 |
| **D_full_composite** | **0.7669** | **0.7486** | **−0.0183** |
| E_plus_rerank | 0.7692 | 0.7688 | −0.0005 |

### MAP

| Variant | Before fix | After fix | Delta |
|---|---|---|---|
| baseline_joint_cos | 0.9974 | 0.9990 | +0.0016 |
| A_two_stream_cos | 0.9974 | 0.9990 | +0.0016 |
| B_plus_mahalanobis | 0.9974 | 0.9989 | +0.0015 |
| C_plus_emd | 0.9984 | 0.9993 | +0.0009 |
| **D_full_composite** | **0.9968** | **0.9787** | **−0.0181** |
| E_plus_rerank | 1.0000 | 0.9961 | −0.0038 |

### P@1

| Variant | Before fix | After fix | Delta |
|---|---|---|---|
| baseline_joint_cos | 1.0000 | 1.0000 | 0.0000 |
| A_two_stream_cos | 1.0000 | 1.0000 | 0.0000 |
| B_plus_mahalanobis | 1.0000 | 1.0000 | 0.0000 |
| C_plus_emd | 1.0000 | 1.0000 | 0.0000 |
| **D_full_composite** | 1.0000 | **0.9940** | **−0.0060** |
| E_plus_rerank | 1.0000 | 1.0000 | 0.0000 |

---

## Statistical Significance — After Fix (Wilcoxon vs Baseline)

| Variant | P@5 | Recall@10 | MAP | NDCG@10 |
|---|---|---|---|---|
| A_two_stream_cos | p=1 | p=1 | p=1 | p=1 |
| B_plus_mahalanobis | p=1 | p=1 | p=1 | p=1 |
| C_plus_emd | p=1 | p=1 | p=1 | p=1 |
| **D_full_composite** | **p=0.008 ✗** | **p<0.0001 ✗** | **p<0.0001 ✗** | **p=0.0001 ✗** |
| E_plus_rerank | p=1 | p=1 | p=0.019 ✗ | p=1 |

✗ = significantly WORSE than baseline (all z-scores are negative).

D is now highly significantly worse than baseline on all key metrics. The fix
exposed what the broken histograms were masking.

---

## What the Fix Reveals

### 1. EMD with correct histograms actively hurts retrieval

With broken histograms, all proteins had nearly identical EMD signatures (all mass
in the first ~3 bins). EMD variance was tiny. The grid search gave it 90% weight
because tiny differences in the first bin marginally helped on the 84-sample fit
set. The overall composite score was dominated by a near-constant term, making
D behave almost identically to baseline.

After the fix, EMD has real variance — energy distribution differences across 32 bins
are properly captured. But the high noise in the synthetic dataset (per-sample gaussian
jitter with σ=0.25) means that same-fold pairs can have very different energy histograms,
and cross-fold pairs can have similar ones. Giving 80% weight to this noisy EMD signal
degrades retrieval substantially (Recall@10 drops from 0.7683 to 0.7486, MAP from 0.9990
to 0.9787).

### 2. The weight grid search is overfitting to 84 samples

The grid search found weights (0, 0, 0.2, 0.8) on 84 samples that gave
Recall@10=0.987 on those samples. On the remaining 84 (M2 val set) those weights
gave Recall@10=0.987 too — consistent. But on the full 168-query M4 test, those
same weights produce Recall@10=0.749, far below baseline 0.768. The 84/84 split
is too small and too easy to expose this generalization failure.

### 3. Variants A, B, C remain indistinguishable from baseline

The two-stream cosine (A) continues to produce identical results to the whole-image
baseline cosine — no improvement from splitting geometry and chemistry. Adding the
Mahalanobis kernel (B) also produces no change. Only the EMD term (C) and re-ranker
(E) produce any difference at all — and after the histogram fix, both are negative.

### 4. The re-ranker (E) is less impressive post-fix

Before the fix, E achieved MAP=1.000 on 168 queries. After the fix, E's MAP is
0.9961 (Wilcoxon p=0.019 vs baseline, significantly worse). The re-ranker was
distilled against the pre-fix D scores as a teacher, so it learned to mimic a
signal corrupted by the histogram bug.

---

## Summary

| Claim | Before fix | After fix |
|---|---|---|
| D beats baseline on Recall@10 | False (barely changed) | False (−0.0183, p<0.0001 worse) |
| D beats baseline on MAP | False (D was worse) | False (D much worse, p<0.0001) |
| EMD adds value over cosine | Unclear (near-degenerate histograms) | No — actively harmful |
| Fix reveals a real improvement | N/A | No — composite method needs rethinking |

The histogram fix is correct and necessary. It exposed that the composite metric
design does not work on this dataset: the EMD term, now properly computed, adds noise
rather than signal. Before running on real SCOPe data, the weighting strategy needs to
change. Equal weights (0.25, 0.25, 0.25, 0.25) or a much larger held-out set for
grid search (≥500 queries) should be tried first.
