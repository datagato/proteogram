"""Train a TriangleStreamEncoder on a directory of v2 proteograms.

Usage (real data)::

    uv run python scripts/v2/train_two_stream.py \\
        --data_dir <proteograms_dir> \\
        --label_tsv <annot.tsv>      \\
        --epochs 30 --batch 32 --lr 3e-4

Usage (synthetic smoke test, no data needed)::

    uv run python scripts/v2/train_two_stream.py --synthetic --epochs 5

The script writes a JSON metrics report to ``--report`` (default
``docs/reports/m1_two_stream.json``) plus the trained checkpoint and a
pickled embeddings dump (``e_U``, ``e_L``, ``e``, plus per-channel
histograms for downstream EMD).
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

# Allow running this file directly without `uv run` having set sys.path.
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from proteogram.v2.encoders import TriangleStreamEncoder  # noqa: E402
from proteogram.v2.losses import SupConLoss  # noqa: E402
from proteogram.v2.features import all_histograms  # noqa: E402

from _synthetic_dataset import SyntheticConfig, make_dataset, to_torch_tensor  # noqa: E402


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument('--data_dir', type=str, default=None,
                   help='Directory of v2 proteogram .npy or image files.')
    p.add_argument('--label_tsv', type=str, default=None,
                   help='TSV with columns: filename<TAB>fold_id.')
    p.add_argument('--resize', type=int, default=None,
                   help='Optional square resize for real-image inputs (pixels).')
    p.add_argument('--synthetic', action='store_true',
                   help='Use the deterministic synthetic dataset.')
    p.add_argument('--epochs', type=int, default=10)
    p.add_argument('--batch', type=int, default=32)
    p.add_argument('--lr', type=float, default=3e-4)
    p.add_argument('--emb_dim', type=int, default=128)
    p.add_argument('--temperature', type=float, default=0.1)
    p.add_argument('--weights', type=float, nargs=4, default=[0.4, 0.4, 1.0, 0.2],
                   metavar=('a', 'b', 'g', 'd'),
                   help='Loss weights for SupCon(U), SupCon(L), SupCon(joint), CE.')
    p.add_argument('--checkpoint', type=str, default='docs/reports/m1_encoder.pt')
    p.add_argument('--embeddings_out', type=str, default='docs/reports/m1_embeddings.pkl')
    p.add_argument('--report', type=str, default='docs/reports/m1_two_stream.json')
    p.add_argument('--seed', type=int, default=0)
    return p.parse_args()


def _load_data(args) -> Tuple[np.ndarray, np.ndarray]:
    if args.synthetic or args.data_dir is None:
        cfg = SyntheticConfig(seed=args.seed)
        return make_dataset(cfg)
    # Real-data loader stub: load .npy files referenced by the TSV.
    rows: List[Tuple[str, int]] = []
    with open(args.label_tsv) as fh:
        for line in fh:
            parts = line.rstrip().split('\t')
            if len(parts) >= 2:
                rows.append((parts[0], int(parts[1])))
    images = []
    labels = []
    for fname, fold in rows:
        path = os.path.join(args.data_dir, fname)
        if path.endswith('.npy'):
            arr = np.load(path)
            if args.resize is not None:
                from PIL import Image
                # Assume CHW float-like input; convert to HWC uint8 for PIL resize.
                if arr.ndim == 3 and arr.shape[0] in (1, 3):
                    hwc = np.transpose(arr, (1, 2, 0))
                    hwc = np.clip(hwc * 255.0, 0, 255).astype(np.uint8)
                    resized = Image.fromarray(hwc).resize((args.resize, args.resize), Image.BILINEAR)
                    arr = np.asarray(resized).transpose(2, 0, 1).astype(np.float32) / 255.0
            images.append(arr)
        else:
            from PIL import Image
            img = Image.open(path).convert('RGB')
            if args.resize is not None:
                img = img.resize((args.resize, args.resize), Image.BILINEAR)
            arr = np.asarray(img).transpose(2, 0, 1).astype(np.float32) / 255.0
            images.append(arr)
        labels.append(fold)
    return np.stack(images, axis=0), np.asarray(labels, dtype=np.int64)


def _train_val_split(X: np.ndarray, y: np.ndarray, val_frac: float = 0.25,
                     seed: int = 0) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(X))
    cut = int(round(len(X) * (1 - val_frac)))
    tr, va = idx[:cut], idx[cut:]
    return X[tr], y[tr], X[va], y[va]


def _embed_all(model, X, batch=64, device='cpu') -> Dict[str, np.ndarray]:
    import torch
    es_U, es_L, es = [], [], []
    model.eval()
    with torch.no_grad():
        for i in range(0, len(X), batch):
            xb = torch.from_numpy(X[i:i + batch]).to(device)
            out = model(xb)
            es_U.append(out['e_U'].cpu().numpy())
            es_L.append(out['e_L'].cpu().numpy())
            es.append(out['e'].cpu().numpy())
    return {
        'e_U': np.concatenate(es_U),
        'e_L': np.concatenate(es_L),
        'e':   np.concatenate(es),
    }


def _recall_at_k(emb: np.ndarray, labels: np.ndarray, k: int = 10) -> float:
    # Brute-force cosine; embeddings are L2-normalised already.
    sims = emb @ emb.T
    np.fill_diagonal(sims, -np.inf)
    order = np.argsort(-sims, axis=1)[:, :k]
    hits = (labels[order] == labels[:, None]).any(axis=1)
    return float(hits.mean())


def main() -> None:
    args = _parse_args()
    import torch

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    t0 = time.time()
    X, y = _load_data(args)
    Xtr, ytr, Xva, yva = _train_val_split(X, y, val_frac=0.25, seed=args.seed)
    n_classes = int(y.max() + 1)

    model = TriangleStreamEncoder(emb_dim=args.emb_dim, num_classes=n_classes,
                                  share_init=True).to(device)
    sup = SupConLoss(temperature=args.temperature)
    ce = torch.nn.CrossEntropyLoss()
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)

    a, b, g, d = args.weights
    history = {'epochs': [], 'loss': [], 'val_recall@10': [], 'val_loss': []}

    n_train = len(Xtr)
    rng = np.random.default_rng(args.seed)
    for epoch in range(1, args.epochs + 1):
        model.train()
        perm = rng.permutation(n_train)
        running = 0.0
        n_batches = 0
        for i in range(0, n_train, args.batch):
            sel = perm[i:i + args.batch]
            xb = torch.from_numpy(Xtr[sel]).to(device)
            yb = torch.from_numpy(ytr[sel]).to(device)
            out = model(xb)
            loss = (
                a * sup(out['e_U'], yb)
                + b * sup(out['e_L'], yb)
                + g * sup(out['e'], yb)
                + d * ce(out['logits'], yb)
            )
            opt.zero_grad()
            loss.backward()
            opt.step()
            running += float(loss.item())
            n_batches += 1
        train_loss = running / max(n_batches, 1)

        # validation
        emb_v = _embed_all(model, Xva, batch=args.batch, device=device)
        r10 = _recall_at_k(emb_v['e'], yva, k=min(10, len(Xva) - 1))
        # quick val loss
        with torch.no_grad():
            xb = torch.from_numpy(Xva[: min(64, len(Xva))]).to(device)
            yb = torch.from_numpy(yva[: min(64, len(Xva))]).to(device)
            out_v = model(xb)
            val_loss = float((
                a * sup(out_v['e_U'], yb)
                + b * sup(out_v['e_L'], yb)
                + g * sup(out_v['e'], yb)
                + d * ce(out_v['logits'], yb)
            ).item())
        history['epochs'].append(epoch)
        history['loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['val_recall@10'].append(r10)
        print(f'epoch {epoch:>3d}  loss={train_loss:.4f}  val_loss={val_loss:.4f}  val_R@10={r10:.3f}')

    elapsed = time.time() - t0

    # save checkpoint, embeddings, and report
    Path(args.checkpoint).parent.mkdir(parents=True, exist_ok=True)
    Path(args.embeddings_out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.report).parent.mkdir(parents=True, exist_ok=True)
    torch.save({'model': model.state_dict(),
                'emb_dim': args.emb_dim,
                'num_classes': n_classes}, args.checkpoint)

    emb_full = _embed_all(model, X, batch=args.batch, device=device)
    histos = [all_histograms(img) for img in X]
    dump = {
        'labels': y.tolist(),
        'e_U': emb_full['e_U'],
        'e_L': emb_full['e_L'],
        'e':   emb_full['e'],
        'H':   histos,
    }
    with open(args.embeddings_out, 'wb') as fh:
        pickle.dump(dump, fh)

    report = {
        'milestone': 'M1',
        'config': vars(args),
        'data': {
            'n_total': int(len(X)),
            'n_train': int(len(Xtr)),
            'n_val': int(len(Xva)),
            'n_classes': n_classes,
            'image_shape': list(X.shape[1:]),
        },
        'history': history,
        'final': {
            'train_loss': history['loss'][-1],
            'val_loss': history['val_loss'][-1],
            'val_recall@10': history['val_recall@10'][-1],
        },
        'elapsed_sec': elapsed,
        'device': device,
    }
    with open(args.report, 'w') as fh:
        json.dump(report, fh, indent=2)
    print(f'wrote report → {args.report}')
    print(f'wrote embeddings → {args.embeddings_out}')
    print(f'wrote checkpoint → {args.checkpoint}')


if __name__ == '__main__':
    main()
