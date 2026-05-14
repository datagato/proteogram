# M3 — Cross-Encoder Re-ranker (training report)

## What this run validates

`PairwiseCrossEncoder` (proteogram/v2/reranker.py) consumes a pair
of v2 proteograms as a 6-channel tensor (`q.RGB || t.RGB`) and emits a
single calibrated similarity in (0, 1). It is **distilled** against a
TM-score-style teacher: in production this is GTalign / US-align, in
this run a synthetic "fold overlap" teacher (≈1 for same-fold pairs,
≈0 for different-fold pairs).

## Configuration

| Field | Value |
|---|---|
| Pair source | synthetic; 1500 sampled pairs |
| Train / val pair split | 1196 / 299 (random, seed=0) |
| Backbone width | 16 base channels |
| Loss | MSE distillation against teacher score in [0, 1] |
| Optimizer | Adam, lr 1e-3 |
| Epochs | 4 |
| Batch size | 16 |
| Device | CPU |

## Training dynamics

| epoch | train MSE | val MSE | val pair-AUC |
|---|---|---|---|
| 1 | 0.0563 | 0.0349 | 0.987 |
| 2 | 0.0286 | 0.0451 | 0.995 |
| 3 | 0.0190 | 0.0221 | 1.000 |
| 4 | 0.0133 | 0.0362 | 1.000 |

Train MSE drops 0.056 → 0.013; pair AUROC saturates at 1.000 by epoch 3.

## Verifiability

| Check | Result |
|---|---|
| Re-ranker forward pass on two `(B, 3, N, N)` tensors returns `(B,)` in [0, 1] | passing (Sigmoid head) |
| Distillation loss strictly decreases on training set | passing (0.056 → 0.013) |
| Pair AUROC on held-out pairs > 0.95 | passing (saturates at 1.000) |
| `rerank()` returns a re-sorted top-K list | passing (consumed by M4 stage E) |
| Wall-clock | 2.5 s / 4 epochs on CPU |

## Notes for real-data runs

The synthetic teacher is binary-ish (same-fold ≈ 0.9 / different-fold ≈ 0.15). On real
data, pull TM-score from the existing GTalign / US-align outputs:

```bash
# Build a pair score table from existing config-dir results, then:
python3 scripts/v2/train_reranker.py \
  --data_npz <real_proteograms.npz> \
  --pair_tsv <pair_tm_scores.tsv> \
  --epochs 6 --batch 32
```

A near-binary teacher is a less-informative signal than continuous TM-score, so
on real data expect **slower, lower-saturating** AUROC curves than the synthetic
results above.

## Outputs

| File | Purpose |
|---|---|
| `docs/reports/m3_reranker.pt` | Trained `PairwiseCrossEncoder` state dict |
| `docs/reports/m3_reranker.json` | Full history + final metrics |

## Reproduce

```bash
PYTHONPATH=. python3 scripts/v2/train_reranker.py \
  --synthetic --epochs 4 --batch 16 --n_pairs 1500 \
  --report docs/reports/m3_reranker.json \
  --checkpoint docs/reports/m3_reranker.pt
```
