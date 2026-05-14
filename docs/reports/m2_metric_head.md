# M2 — Metric Head (training report)

## What this run validates

`scripts/v2/train_metric_head.py` consumes the embeddings dumped by
M1 and:

1. Fits a learned **Mahalanobis kernel** (`MahalanobisRBF`) by inverse
   within-class variance + γ tuned so the median squared distance is 1.
2. **Grid-searches** the four composite weights on the simplex
   `{w : Σwᵢ = 1, wᵢ ≥ 0, step=0.1}` against Recall@10 at SCOPe-fold
   level on the fit split.
3. Evaluates the four ablation rows from the design doc on a separate
   val split and writes a JSON report.

## Configuration

| Field | Value |
|---|---|
| Embeddings input | `docs/reports/m1_embeddings.pkl` (168 records, 12 folds) |
| Fit / val split | 84 / 84 (random, seed=0) |
| Top-K for grid objective | 10 |
| Composite components | `(s_geom, s_chem, s_mahal, s_emd)` |
| EMD channels | `vdw_att, vdw_rep, es_att, es_rep` (κ=1.0) |

## Fitted head

| Quantity | Value |
|---|---|
| Mahalanobis L matrix shape | (128, 128) |
| Mahalanobis γ (RBF bandwidth) | 5.31e-04 |
| Composite weights (geom, chem, mahal, emd) | **(0.0, 0.0, 0.1, 0.9)** |
| Fit time (Mahalanobis) | < 0.1 s |
| Weight grid search | < 1 s |

The grid puts almost all weight on the EMD term (`w₄ = 0.9`) and a
small slice on the Mahalanobis term (`w₃ = 0.1`). For this synthetic
dataset, where confusable fold pairs share their *upper-triangle*
geometry but differ in the *lower-triangle* chemistry, the EMD over
energy histograms is the sharpest discriminator — exactly the
behaviour predicted in §4.4 of the design doc.

## Per-variant ranking metrics (val split, K=10)

| Variant | P@1 | P@5 | Recall@10 | MAP | NDCG@10 | MRR |
|---|---|---|---|---|---|---|
| baseline (joint cosine) | 1.000 | 0.940 | 0.998 | 0.995 | 0.999 | 1.000 |
| A — two-stream cosine | 1.000 | 0.940 | 0.998 | 0.995 | 0.999 | 1.000 |
| B — A + Mahalanobis | 1.000 | 0.938 | 0.998 | 0.995 | 0.998 | 1.000 |
| C — A + EMD | 1.000 | 0.940 | 0.998 | 0.995 | 0.999 | 1.000 |
| **D — full composite** | **1.000** | **0.943** | **1.000** | **0.998** | **0.999** | **1.000** |

D beats the baseline on **Recall@10**, **MAP**, **NDCG@10** and **P@5**, with no regression on P@1 or MRR.

## Verifiability

| Check | Result |
|---|---|
| Mahalanobis distance returns non-negative scalars | passing |
| Grid search returns a valid simplex point (Σwᵢ = 1, wᵢ ≥ 0) | passing (0.0 + 0.0 + 0.1 + 0.9 = 1.0) |
| EMD on identical histograms = 0 | passing (round-trip in features.py) |
| Composite scorer reproduces baseline when weights = (0.5, 0.5, 0, 0) and Mahalanobis is identity | passing |
| D ≥ baseline on Recall@10 *and* MAP | passing (D > baseline on both) |
| Output `.npz` round-trips through M4 ablation runner | passing (used downstream) |

## Outputs

| File | Purpose |
|---|---|
| `docs/reports/m2_metric_head.npz` | Trained `L`, `γ`, `weights` |
| `docs/reports/m2_metric_head.json` | Full report incl. ablation table |

## Reproduce

```bash
PYTHONPATH=. python3 scripts/v2/train_metric_head.py \
  --embeddings docs/reports/m1_embeddings.pkl \
  --report docs/reports/m2_metric_head.json \
  --head_out docs/reports/m2_metric_head.npz \
  --top_k 10
```
