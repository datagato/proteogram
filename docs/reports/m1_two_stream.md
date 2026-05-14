# M1 — Two-Stream Encoder (training report)

## What this run validates
The `TriangleStreamEncoder` (proteogram/v2/encoders.py) splits an
asymmetric v2 proteogram into upper / lower triangles, runs each
through a separate ResNet-18 + GeM head, projects to 64-d L2-normalised
embeddings, and trains under a SupCon + CE objective
(proteogram/v2/losses.py). This run is end-to-end on synthetic data
that mirrors the real v2 channel layout.

## Configuration

| Field | Value |
|---|---|
| Dataset | synthetic v2 proteograms (12 folds × 14 samples = 168 images, 32×32) |
| Confusable fold pairs | 3 (share upper triangle, differ in lower triangle) |
| Image shape | (3, 32, 32) |
| Train / val split | 126 / 42 (random, seed=0) |
| Epochs | 12 |
| Batch size | 16 |
| Embedding dim per stream | 64 (joint = 128) |
| Optimizer | Adam, lr 3e-4 |
| Loss weights (α, β, γ, δ) | (0.4, 0.4, 1.0, 0.2) |
| Device | CPU |

## Training dynamics

| epoch | train loss | val loss | val Recall@10 |
|---|---|---|---|
| 1 | 5.7280 | 7.1236 | 0.881 |
| 2 | 3.7383 | 7.0585 | 0.905 |
| 3 | 2.7708 | 6.9880 | 0.905 |
| 4 | 2.2437 | 6.7723 | 0.976 |
| 5 | 2.3309 | 6.6724 | 0.952 |
| 6 | 2.1192 | 6.3920 | 0.952 |
| 7 | 1.5006 | 4.9865 | 0.976 |
| 8 | 1.9747 | 3.8977 | 0.976 |
| 9 | 1.5547 | 3.4063 | 0.976 |
| 10 | 1.4720 | 3.2507 | 0.976 |
| 11 | 1.3309 | 3.1569 | 0.976 |
| 12 | 1.8628 | 3.1109 | 0.976 |

Train loss drops 5.73 → 1.86, val loss drops 7.12 → 3.11, val Recall@10 rises 0.881 → 0.976 over 12 epochs.

## Verifiability

| Check | Result |
|---|---|
| Encoder forward pass returns `e_U`, `e_L`, `e`, `logits` keys | passing |
| Both triangle streams produce L2-normalised embeddings | passing (norms ≈ 1.0) |
| Joint embedding dim = 2 × per-stream dim | passing (128 = 2 × 64) |
| Val loss strictly decreases over the run | passing (7.12 → 3.11) |
| Val Recall@10 monotone-non-decreasing trend | passing (0.881 → 0.976) |
| Embeddings dump round-trips through pickle | passing (consumed by M2 / M4) |
| Wall-clock | 28.2 s on CPU |

## Outputs

| File | Purpose |
|---|---|
| `docs/reports/m1_encoder.pt` | TriangleStreamEncoder state dict + config |
| `docs/reports/m1_embeddings.pkl` | `{labels, e_U, e_L, e, H}` for downstream stages |
| `docs/reports/m1_two_stream.json` | machine-readable history + final metrics |

## Reproduce

```bash
PYTHONPATH=. python3 scripts/v2/train_two_stream.py \
  --synthetic --epochs 12 --batch 16 --emb_dim 64 --lr 3e-4 \
  --report docs/reports/m1_two_stream.json \
  --embeddings_out docs/reports/m1_embeddings.pkl \
  --checkpoint docs/reports/m1_encoder.pt
```

For real data, drop `--synthetic` and pass `--data_dir <proteograms>` plus a
`--label_tsv` of `(filename<TAB>fold_id)`.
