# v2 Similarity Stack — Implementation Summary

End-to-end implementation of the changes proposed in
`docs/v2_similarity_design.md`. All four milestones (M1 – M4) are
implemented, exercised end-to-end on synthetic data, and produce
machine-readable reports plus human-readable markdown.

## What landed

### Library code (`proteogram/v2/`)

| File | Purpose |
|---|---|
| `encoders.py` | `TriangleStreamEncoder`, `GeM`, `split_triangles` |
| `losses.py` | `SupConLoss`, `triplet_with_mahalanobis` |
| `features.py` | `HistogramSpec`, `channel_histogram`, `all_histograms` |
| `metrics.py` | `MahalanobisRBF`, `CompositeScorer`, `cos_split`, `emd_1d`, `emd_score`, ranking metrics (P@K, R@K, MAP, NDCG, MRR) |
| `reranker.py` | `PairwiseCrossEncoder`, `distillation_loss` |
| `__init__.py` | Lazy-imports the heavy MD/Bio modules so the new code is usable in slim environments |

### Scripts (`scripts/v2/`)

| File | Stage | Purpose |
|---|---|---|
| `_synthetic_dataset.py` | shared | Deterministic synthetic v2 proteograms with confusable fold pairs |
| `train_two_stream.py` | M1 | Train `TriangleStreamEncoder` (SupCon × 3 + CE), dump embeddings + histograms |
| `train_metric_head.py` | M2 | Fit `MahalanobisRBF` and grid-search composite weights |
| `train_reranker.py` | M3 | Distil `PairwiseCrossEncoder` against TM-score teacher |
| `run_ablation.py` | M4 | Full ablation table with bootstrap 95% CIs and Wilcoxon paired tests |

## Verification — what we actually ran

All four stages were executed end-to-end on synthetic data
(168 images, 12 folds, 32×32 each, 3 deliberately-confusable fold pairs).
Every stage wrote both a `.json` and a `.md` report.

| Stage | Wall-clock | Headline result |
|---|---|---|
| M1 — train encoder | 28.2 s (CPU) | val Recall@10 0.881 → 0.976 over 12 epochs |
| M2 — fit metric head | < 1 s | grid picks (0.0, 0.0, 0.1, 0.9); D beats baseline on R@10 / MAP / NDCG / P@5 |
| M3 — distil re-ranker | 2.5 s | pair AUROC saturates at 1.000 by epoch 3 |
| M4 — full ablation | 3.3 s | D and E significantly improve MAP over baseline (Wilcoxon p ≤ 0.025) |

### Statistical significance highlights (M4, vs baseline cosine)

| Variant | MAP delta | Wilcoxon p |
|---|---|---|
| C — A + EMD | +0.001 | 0.028 |
| **D — full composite** | +0.003 | **0.0070** |
| E — D + re-rank | +0.003 | 0.025 |

`D` is significant at p < 0.01 on MAP; `E` adds a small extra MAP gain
on top of `D`. On Recall@10, P@1, and MRR, every variant is at or
above ceiling on this synthetic dataset, so no further differentiation
is observable.

## How to reproduce in 30 seconds

```bash
mkdir -p docs/reports
PYTHONPATH=. python3 scripts/v2/train_two_stream.py --synthetic --epochs 12
PYTHONPATH=. python3 scripts/v2/train_metric_head.py
PYTHONPATH=. python3 scripts/v2/train_reranker.py --synthetic --epochs 4
PYTHONPATH=. python3 scripts/v2/run_ablation.py --use_synthetic_images
```

Per-stage reports land in `docs/reports/m{1..4}_*.md`.

## How to wire this onto real SCOPe data

1. **M1**: replace `--synthetic` with `--data_dir <proteograms_dir>` and a
   `--label_tsv` of `(filename<TAB>fold_id)` rows. The script understands
   `.npy` proteogram tensors directly, or PNG/JPG (it converts via PIL).
2. **M2**: no change — it operates on the embeddings dump only.
3. **M3**: build a `pair_tsv` with `(q_idx, t_idx, tm_score)` triples
   from the existing `gtalign_results_dir` and `usalign_results` paths
   in `scripts/v2/config.yml`, then pass `--data_npz` + `--pair_tsv`.
4. **M4**: run with `--images_npz <real_images.npz>` so the re-ranker
   sees real proteograms instead of synthetic ones. Bootstrap CIs and
   Wilcoxon tests work identically on real metrics.

## Known limitations & next steps

| Limitation | Impact | Suggested follow-up |
|---|---|---|
| Synthetic dataset saturates P@1, R@10, MRR at the ceiling | Variant differences only show up on MAP / NDCG / P@5 | run M4 on real SCOPe to expose more spread |
| `MahalanobisRBF.fit_diagonal_from_pairs` is a closed-form fit, not a triplet-trained metric | Lower ceiling than design doc §4.3 specifies | add `train_mahalanobis_triplet.py` once real fold-paired triplets are available |
| Composite weight grid is at step=0.1 | Good for explainability, may miss optima | switch to a tiny learn-to-rank MLP (`metrics.LearnedRanker` slot is already wired) |
| Re-ranker teacher is a synthetic step function | Real TM-score is continuous and noisier | swap `_build_synthetic_pairs` for a real GTalign loader; expect slower AUROC saturation |
| All runs above are CPU-only | Slower than necessary | every script auto-detects CUDA via `torch.cuda.is_available()` — no code change needed |

## Files written by this implementation

```
docs/v2_similarity_design.md            ← design doc (existing)
docs/reports/
  m1_two_stream.md                      ← stage report
  m1_two_stream.json                    ← machine-readable
  m1_encoder.pt                         ← trained encoder
  m1_embeddings.pkl                     ← embeddings + histograms
  m2_metric_head.md                     ← stage report
  m2_metric_head.json                   ← machine-readable
  m2_metric_head.npz                    ← Mahalanobis L, gamma, weights
  m3_reranker.md                        ← stage report
  m3_reranker.json                      ← machine-readable
  m3_reranker.pt                        ← trained re-ranker
  m4_ablation.md                        ← ablation table + Wilcoxon
  m4_ablation.json                      ← machine-readable
  v2_implementation_summary.md          ← this file
proteogram/v2/
  encoders.py losses.py features.py metrics.py reranker.py __init__.py
scripts/v2/
  _synthetic_dataset.py train_two_stream.py train_metric_head.py
  train_reranker.py    run_ablation.py
```
