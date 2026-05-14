"""Fit Mahalanobis kernel + composite weights on dumped M1 embeddings.

Reads the pickle written by ``train_two_stream.py`` (``e_U``, ``e_L``,
``e``, ``H``, ``labels``), fits ``MahalanobisRBF`` and the linear
combiner via grid search at SCOPe-fold level, and writes both the
trained metric head and a JSON report.

Usage::

    uv run python scripts/v2/train_metric_head.py \\
        --embeddings docs/reports/m1_embeddings.pkl \\
        --report docs/reports/m2_metric_head.json
"""

from __future__ import annotations

import argparse
import json
import pickle
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

import sys
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from proteogram.v2.metrics import (  # noqa: E402
    CompositeScorer, MahalanobisRBF,
    cos_split, emd_score,
    precision_at_k, recall_at_k, average_precision, ndcg_at_k, reciprocal_rank,
)


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument('--embeddings', type=str, default='docs/reports/m1_embeddings.pkl')
    p.add_argument('--report', type=str, default='docs/reports/m2_metric_head.json')
    p.add_argument('--head_out', type=str, default='docs/reports/m2_metric_head.npz')
    p.add_argument('--val_frac', type=float, default=0.5)
    p.add_argument('--top_k', type=int, default=10)
    p.add_argument('--seed', type=int, default=0)
    return p.parse_args()


def _to_records(dump) -> List[Dict]:
    out = []
    n = len(dump['labels'])
    for i in range(n):
        out.append({
            'e_U': dump['e_U'][i],
            'e_L': dump['e_L'][i],
            'e':   dump['e'][i],
            'H':   dump['H'][i],
            'label': int(dump['labels'][i]),
        })
    return out


def _components(query, target, mahal: MahalanobisRBF, kappa: float = 1.0,
                emd_channels=('vdw_att', 'vdw_rep', 'es_att', 'es_rep')) -> Tuple[float, float, float, float]:
    s_geom, s_chem = cos_split(query, target)
    s_mahal = float(mahal.kernel(query['e'], target['e']))
    s_emd = emd_score(query, target, emd_channels, kappa=kappa)
    return s_geom, s_chem, s_mahal, s_emd


def _ranking_metrics(records: List[Dict], scorer, k: int) -> Dict[str, float]:
    p1, p5, r10, mapv, ndcg, mrr = [], [], [], [], [], []
    for i, q in enumerate(records):
        scored = []
        for j, t in enumerate(records):
            if i == j:
                continue
            scored.append((j, scorer(q, t)))
        scored.sort(key=lambda kv: kv[1], reverse=True)
        rel = [1 if records[j]['label'] == q['label'] else 0 for j, _ in scored]
        total_rel = sum(1 for r in records if r['label'] == q['label']) - 1
        p1.append(precision_at_k(rel, 1))
        p5.append(precision_at_k(rel, 5))
        r10.append(recall_at_k(rel, total_rel, k))
        mapv.append(average_precision(rel))
        ndcg.append(ndcg_at_k(rel, k))
        mrr.append(reciprocal_rank(rel))
    return {
        'P@1': float(np.mean(p1)),
        'P@5': float(np.mean(p5)),
        f'Recall@{k}': float(np.mean(r10)),
        'MAP': float(np.mean(mapv)),
        f'NDCG@{k}': float(np.mean(ndcg)),
        'MRR': float(np.mean(mrr)),
    }


def main() -> None:
    args = _parse_args()
    with open(args.embeddings, 'rb') as fh:
        dump = pickle.load(fh)

    records = _to_records(dump)
    rng = np.random.default_rng(args.seed)
    idx = rng.permutation(len(records))
    cut = int(round(len(records) * (1 - args.val_frac)))
    fit_idx, val_idx = idx[:cut], idx[cut:]
    fit_records = [records[i] for i in fit_idx]
    val_records = [records[i] for i in val_idx]

    # 1) Mahalanobis from joint embeddings
    e_fit = np.stack([r['e'] for r in fit_records])
    y_fit = np.asarray([r['label'] for r in fit_records])
    t0 = time.time()
    mahal = MahalanobisRBF.fit_diagonal_from_pairs(e_fit, y_fit)
    fit_time = time.time() - t0

    # 2) Build per-query candidate matrix (s1, s2, s3, s4, rel) on fit set
    scorers_per_query = []
    for i, q in enumerate(fit_records):
        rows = []
        for j, t in enumerate(fit_records):
            if i == j:
                continue
            comps = _components(q, t, mahal)
            rel = 1 if q['label'] == t['label'] else 0
            rows.append((*comps, rel))
        scorers_per_query.append(rows)

    # 3) Grid search weights
    t0 = time.time()
    composite = CompositeScorer.fit_grid_search(
        scorers_per_query, step=0.1, objective='recall@10', k=args.top_k,
        mahal=mahal,
    )
    weight_search_time = time.time() - t0

    # 4) Evaluate ablation A/B/C/D on the val records
    def scorer_A(q, t):  # two-stream cosine only
        s_geom, s_chem = cos_split(q, t)
        return 0.5 * s_geom + 0.5 * s_chem

    def scorer_B(q, t):  # A + Mahalanobis
        s_geom, s_chem = cos_split(q, t)
        s_m = float(mahal.kernel(q['e'], t['e']))
        return (s_geom + s_chem + s_m) / 3.0

    def scorer_C(q, t):  # A + EMD
        s_geom, s_chem = cos_split(q, t)
        s_e = emd_score(q, t, ('vdw_att', 'vdw_rep', 'es_att', 'es_rep'))
        return (s_geom + s_chem + s_e) / 3.0

    def scorer_D(q, t):  # A + B + C
        s_geom, s_chem = cos_split(q, t)
        s_m = float(mahal.kernel(q['e'], t['e']))
        s_e = emd_score(q, t, ('vdw_att', 'vdw_rep', 'es_att', 'es_rep'))
        return composite.weights @ np.array([s_geom, s_chem, s_m, s_e])

    def scorer_baseline(q, t):  # whole-image cosine on joint embedding
        return float(np.dot(q['e'], t['e']))

    metrics = {
        'baseline_joint_cos': _ranking_metrics(val_records, scorer_baseline, args.top_k),
        'A_two_stream_cos':   _ranking_metrics(val_records, scorer_A, args.top_k),
        'B_plus_mahalanobis': _ranking_metrics(val_records, scorer_B, args.top_k),
        'C_plus_emd':         _ranking_metrics(val_records, scorer_C, args.top_k),
        'D_full_composite':   _ranking_metrics(val_records, scorer_D, args.top_k),
    }

    np.savez(args.head_out, L=mahal.L, gamma=np.array([mahal.gamma]),
             weights=composite.weights)

    report = {
        'milestone': 'M2',
        'config': vars(args),
        'fit_time_sec': fit_time,
        'weight_search_time_sec': weight_search_time,
        'mahalanobis': {
            'L_shape': list(mahal.L.shape),
            'gamma': mahal.gamma,
        },
        'composite_weights': composite.weights.tolist(),
        'metrics_val': metrics,
        'n_fit': len(fit_records),
        'n_val': len(val_records),
    }
    Path(args.report).parent.mkdir(parents=True, exist_ok=True)
    with open(args.report, 'w') as fh:
        json.dump(report, fh, indent=2)
    print(f'wrote report → {args.report}')
    print(json.dumps(metrics, indent=2))


if __name__ == '__main__':
    main()
