"""Distil a PairwiseCrossEncoder against a structural-alignment teacher.

In production the teacher is GTalign / US-align TM-score. For the
synthetic smoke test we use a simple "fold overlap" teacher: 1.0 for
same-fold pairs and a noisy 0.0..0.3 for different-fold pairs.

Usage::

    uv run python scripts/v2/train_reranker.py \\
        --synthetic --epochs 5 --report docs/reports/m3_reranker.json
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

import sys
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from proteogram.v2.reranker import PairwiseCrossEncoder, distillation_loss  # noqa: E402

from _synthetic_dataset import SyntheticConfig, make_dataset  # noqa: E402


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument('--synthetic', action='store_true')
    p.add_argument('--data_npz', type=str, default=None,
                   help='npz with keys X (B, 3, N, N) and y (B,) for real data.')
    p.add_argument('--pair_tsv', type=str, default=None,
                   help='TSV with columns: q_idx<TAB>t_idx<TAB>tm_score (real data).')
    p.add_argument('--epochs', type=int, default=5)
    p.add_argument('--batch', type=int, default=32)
    p.add_argument('--lr', type=float, default=1e-3)
    p.add_argument('--n_pairs', type=int, default=2000)
    p.add_argument('--report', type=str, default='docs/reports/m3_reranker.json')
    p.add_argument('--checkpoint', type=str, default='docs/reports/m3_reranker.pt')
    p.add_argument('--seed', type=int, default=0)
    return p.parse_args()


def _build_synthetic_pairs(X, y, n_pairs, seed):
    rng = np.random.default_rng(seed)
    pairs = []
    for _ in range(n_pairs):
        i, j = rng.integers(0, len(X), size=2)
        if i == j:
            continue
        same = y[i] == y[j]
        # Teacher TM-score analogue
        tm = float(rng.uniform(0.85, 1.0)) if same else float(rng.uniform(0.0, 0.3))
        pairs.append((int(i), int(j), tm))
    return pairs


def main() -> None:
    args = _parse_args()
    import torch

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    if args.synthetic or args.data_npz is None:
        cfg = SyntheticConfig(seed=args.seed)
        X, y = make_dataset(cfg)
    else:
        d = np.load(args.data_npz)
        X, y = d['X'].astype(np.float32), d['y'].astype(np.int64)

    pairs = _build_synthetic_pairs(X, y, args.n_pairs, args.seed) \
        if (args.synthetic or args.pair_tsv is None) else None
    if pairs is None:
        rows = []
        with open(args.pair_tsv) as fh:
            for ln in fh:
                a, b, tm = ln.strip().split('\t')
                rows.append((int(a), int(b), float(tm)))
        pairs = rows

    rng = np.random.default_rng(args.seed)
    rng.shuffle(pairs)
    cut = int(0.8 * len(pairs))
    tr_pairs, va_pairs = pairs[:cut], pairs[cut:]

    model = PairwiseCrossEncoder(base_channels=16).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)

    history = {'epoch': [], 'train_mse': [], 'val_mse': [], 'val_pair_auc': []}
    t0 = time.time()
    for epoch in range(1, args.epochs + 1):
        model.train()
        rng.shuffle(tr_pairs)
        running, n = 0.0, 0
        for i in range(0, len(tr_pairs), args.batch):
            batch = tr_pairs[i:i + args.batch]
            qs = torch.from_numpy(np.stack([X[a] for a, _, _ in batch])).to(device)
            ts = torch.from_numpy(np.stack([X[b] for _, b, _ in batch])).to(device)
            tm = torch.tensor([t for _, _, t in batch], dtype=torch.float32, device=device)
            pred = model(qs, ts)
            loss = distillation_loss(pred, tm)
            opt.zero_grad(); loss.backward(); opt.step()
            running += float(loss.item()) * len(batch)
            n += len(batch)
        train_mse = running / max(n, 1)

        # validation
        model.eval()
        with torch.no_grad():
            val_loss, val_n = 0.0, 0
            preds, golds = [], []
            for i in range(0, len(va_pairs), args.batch):
                batch = va_pairs[i:i + args.batch]
                qs = torch.from_numpy(np.stack([X[a] for a, _, _ in batch])).to(device)
                ts = torch.from_numpy(np.stack([X[b] for _, b, _ in batch])).to(device)
                tm = torch.tensor([t for _, _, t in batch], dtype=torch.float32, device=device)
                pred = model(qs, ts)
                val_loss += float(distillation_loss(pred, tm).item()) * len(batch)
                val_n += len(batch)
                preds.extend(pred.cpu().tolist())
                golds.extend([1 if t > 0.5 else 0 for t in tm.cpu().tolist()])
            val_mse = val_loss / max(val_n, 1)

            # pair-AUROC: rank predictions vs binary "same fold" labels
            pred_arr = np.asarray(preds)
            gold_arr = np.asarray(golds)
            order = np.argsort(-pred_arr)
            tp = (gold_arr[order] == 1).cumsum()
            fp = (gold_arr[order] == 0).cumsum()
            P = max(int(gold_arr.sum()), 1)
            N = max(int((1 - gold_arr).sum()), 1)
            tpr = tp / P
            fpr = fp / N
            auc = float(np.trapezoid(tpr, fpr))

        history['epoch'].append(epoch)
        history['train_mse'].append(train_mse)
        history['val_mse'].append(val_mse)
        history['val_pair_auc'].append(auc)
        print(f'epoch {epoch}: train_mse={train_mse:.4f}  val_mse={val_mse:.4f}  val_AUC={auc:.3f}')

    elapsed = time.time() - t0
    Path(args.checkpoint).parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), args.checkpoint)

    report = {
        'milestone': 'M3',
        'config': vars(args),
        'history': history,
        'final': {
            'train_mse': history['train_mse'][-1],
            'val_mse': history['val_mse'][-1],
            'val_pair_auc': history['val_pair_auc'][-1],
        },
        'n_train_pairs': len(tr_pairs),
        'n_val_pairs': len(va_pairs),
        'elapsed_sec': elapsed,
        'device': device,
    }
    Path(args.report).parent.mkdir(parents=True, exist_ok=True)
    with open(args.report, 'w') as fh:
        json.dump(report, fh, indent=2)
    print(f'wrote report → {args.report}')


if __name__ == '__main__':
    main()
