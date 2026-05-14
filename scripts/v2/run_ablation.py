"""End-to-end ablation harness for the v2 metric stack.

Loads dumped M1 embeddings (and optionally an M3 re-ranker checkpoint)
and evaluates the full ablation table from
``docs/v2_similarity_design.md`` §6.5:

    Baseline  : whole-image cosine on the joint embedding
    A         : two-stream cosine
    B         : A + Mahalanobis
    C         : A + EMD
    D         : A + Mahalanobis + EMD with grid-tuned weights
    E         : D + cross-encoder re-rank on the top-K

For every variant we report P@1, P@5, Recall@K, MAP, NDCG@K, MRR plus
bootstrap 95% CIs and a Wilcoxon paired test against the baseline.

Outputs:

    docs/reports/m4_ablation.json   # machine readable
    docs/reports/m4_ablation.md     # human readable

This script is the verification entry point. Re-run it whenever any of
the upstream models change.
"""

from __future__ import annotations

import argparse
import json
import math
import pickle
import time
from pathlib import Path
from typing import Callable, Dict, List, Tuple

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
    p.add_argument('--metric_head', type=str, default='docs/reports/m2_metric_head.npz')
    p.add_argument('--reranker', type=str, default='docs/reports/m3_reranker.pt')
    p.add_argument('--images_npz', type=str, default=None,
                   help='Optional: real proteogram tensors for re-ranker. If '
                        'absent and --use_synthetic_images is set, synthesises '
                        'them from the same seed as the M1 dataset.')
    p.add_argument('--use_synthetic_images', action='store_true')
    p.add_argument('--top_k', type=int, default=10)
    p.add_argument('--rerank_k', type=int, default=20)
    p.add_argument('--bootstrap', type=int, default=500)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--report_json', type=str, default='docs/reports/m4_ablation.json')
    p.add_argument('--report_md', type=str, default='docs/reports/m4_ablation.md')
    return p.parse_args()


def _to_records(dump) -> List[Dict]:
    out = []
    for i in range(len(dump['labels'])):
        out.append({
            'e_U': dump['e_U'][i],
            'e_L': dump['e_L'][i],
            'e':   dump['e'][i],
            'H':   dump['H'][i],
            'label': int(dump['labels'][i]),
        })
    return out


def _per_query(records: List[Dict], scorer: Callable, k: int) -> Dict[str, np.ndarray]:
    p1, p5, rk, mapv, ndcg, mrr = [], [], [], [], [], []
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
        rk.append(recall_at_k(rel, total_rel, k))
        mapv.append(average_precision(rel))
        ndcg.append(ndcg_at_k(rel, k))
        mrr.append(reciprocal_rank(rel))
    return {
        'P@1': np.asarray(p1),
        'P@5': np.asarray(p5),
        f'Recall@{k}': np.asarray(rk),
        'MAP': np.asarray(mapv),
        f'NDCG@{k}': np.asarray(ndcg),
        'MRR': np.asarray(mrr),
    }


def _bootstrap_ci(values: np.ndarray, B: int, seed: int) -> Tuple[float, float, float]:
    rng = np.random.default_rng(seed)
    means = []
    n = len(values)
    if n == 0:
        return 0.0, 0.0, 0.0
    for _ in range(B):
        idx = rng.integers(0, n, size=n)
        means.append(float(values[idx].mean()))
    arr = np.asarray(means)
    return float(values.mean()), float(np.percentile(arr, 2.5)), float(np.percentile(arr, 97.5))


def _wilcoxon_signed_rank(a: np.ndarray, b: np.ndarray) -> Dict[str, float]:
    """Two-sided Wilcoxon paired test using normal approximation.

    Pure-numpy so we don't pull scipy. Acceptable for n ≥ 20.
    """
    diff = a - b
    diff = diff[diff != 0]
    n = len(diff)
    if n < 6:
        return {'W': 0.0, 'z': 0.0, 'p_two_sided': 1.0, 'n': int(n)}
    abs_diff = np.abs(diff)
    order = np.argsort(abs_diff)
    ranks = np.empty(n)
    # average-rank ties
    sorted_abs = abs_diff[order]
    i = 0
    while i < n:
        j = i + 1
        while j < n and sorted_abs[j] == sorted_abs[i]:
            j += 1
        avg = 0.5 * (i + j + 1)
        ranks[i:j] = avg
        i = j
    inv = np.empty(n, dtype=np.int64)
    inv[order] = np.arange(n)
    signed = np.sign(diff) * ranks[inv]
    W_plus = float(signed[signed > 0].sum())
    W_minus = float(-signed[signed < 0].sum())
    W = min(W_plus, W_minus)
    mu = n * (n + 1) / 4.0
    sigma = math.sqrt(n * (n + 1) * (2 * n + 1) / 24.0)
    z = (W - mu) / sigma if sigma > 0 else 0.0
    # two-sided normal p-value
    p = math.erfc(abs(z) / math.sqrt(2))
    return {'W': float(W), 'z': float(z), 'p_two_sided': float(p), 'n': int(n)}


def _run_rerank(records: List[Dict], scorer_fn: Callable, images: np.ndarray,
                rerank_k: int, k: int, reranker_ckpt: str) -> Dict[str, np.ndarray]:
    import torch
    from proteogram.v2.reranker import PairwiseCrossEncoder
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = PairwiseCrossEncoder(base_channels=16).to(device)
    state = torch.load(reranker_ckpt, map_location=device)
    model.load_state_dict(state)
    model.eval()

    p1, p5, rk, mapv, ndcg, mrr = [], [], [], [], [], []
    with torch.no_grad():
        for i, q in enumerate(records):
            scored = [(j, scorer_fn(q, t)) for j, t in enumerate(records) if j != i]
            scored.sort(key=lambda kv: kv[1], reverse=True)
            head = scored[:rerank_k]
            qx = torch.from_numpy(images[i]).unsqueeze(0).to(device)
            cs = torch.from_numpy(np.stack([images[j] for j, _ in head])).to(device)
            qs = qx.expand(len(head), -1, -1, -1)
            refined = model(qs, cs).cpu().numpy()
            head_with_new = [(jx, float(s)) for (jx, _), s in zip(head, refined)]
            head_with_new.sort(key=lambda kv: kv[1], reverse=True)
            tail = scored[rerank_k:]
            new_order = head_with_new + tail
            rel = [1 if records[j]['label'] == q['label'] else 0 for j, _ in new_order]
            total_rel = sum(1 for r in records if r['label'] == q['label']) - 1
            p1.append(precision_at_k(rel, 1))
            p5.append(precision_at_k(rel, 5))
            rk.append(recall_at_k(rel, total_rel, k))
            mapv.append(average_precision(rel))
            ndcg.append(ndcg_at_k(rel, k))
            mrr.append(reciprocal_rank(rel))
    return {
        'P@1': np.asarray(p1),
        'P@5': np.asarray(p5),
        f'Recall@{k}': np.asarray(rk),
        'MAP': np.asarray(mapv),
        f'NDCG@{k}': np.asarray(ndcg),
        'MRR': np.asarray(mrr),
    }


def main() -> None:
    args = _parse_args()
    with open(args.embeddings, 'rb') as fh:
        dump = pickle.load(fh)
    records = _to_records(dump)
    K = args.top_k

    # Mahalanobis + composite weights
    head = np.load(args.metric_head)
    mahal = MahalanobisRBF(L=head['L'], gamma=float(head['gamma'][0]))
    weights = head['weights']

    def s_baseline(q, t):
        return float(np.dot(q['e'], t['e']))

    def s_A(q, t):
        a, b = cos_split(q, t)
        return 0.5 * a + 0.5 * b

    def s_B(q, t):
        a, b = cos_split(q, t)
        m = float(mahal.kernel(q['e'], t['e']))
        return (a + b + m) / 3.0

    def s_C(q, t):
        a, b = cos_split(q, t)
        e = emd_score(q, t, ('vdw_att', 'vdw_rep', 'es_att', 'es_rep'))
        return (a + b + e) / 3.0

    def s_D(q, t):
        a, b = cos_split(q, t)
        m = float(mahal.kernel(q['e'], t['e']))
        e = emd_score(q, t, ('vdw_att', 'vdw_rep', 'es_att', 'es_rep'))
        return float(weights @ np.array([a, b, m, e]))

    variants: List[Tuple[str, Callable]] = [
        ('baseline_joint_cos', s_baseline),
        ('A_two_stream_cos',   s_A),
        ('B_plus_mahalanobis', s_B),
        ('C_plus_emd',         s_C),
        ('D_full_composite',   s_D),
    ]

    timings = {}
    per_query: Dict[str, Dict[str, np.ndarray]] = {}
    for name, fn in variants:
        t0 = time.time()
        per_query[name] = _per_query(records, fn, k=K)
        timings[name] = time.time() - t0
        print(f'{name}: {timings[name]:.2f}s')

    # M3 re-rank pass (if a re-ranker checkpoint exists and we have images)
    images = None
    if args.images_npz:
        d = np.load(args.images_npz)
        images = d['X'].astype(np.float32)
    elif args.use_synthetic_images:
        from _synthetic_dataset import SyntheticConfig, make_dataset
        cfg = SyntheticConfig(seed=args.seed)
        X, y = make_dataset(cfg)
        # important: align order with the embeddings dump
        if not np.array_equal(np.asarray([r['label'] for r in records]), y):
            print('warning: synthetic image order differs from embeddings order; skipping rerank')
            images = None
        else:
            images = X

    if images is not None and Path(args.reranker).exists():
        t0 = time.time()
        per_query['E_plus_rerank'] = _run_rerank(records, s_D, images,
                                                 rerank_k=args.rerank_k, k=K,
                                                 reranker_ckpt=args.reranker)
        timings['E_plus_rerank'] = time.time() - t0
        print(f'E_plus_rerank: {timings["E_plus_rerank"]:.2f}s')
    else:
        print('skipping E_plus_rerank (no reranker / no images)')

    # Aggregate metrics + CIs + Wilcoxon vs baseline
    summary = {}
    metric_names = list(per_query['baseline_joint_cos'].keys())
    base_arr = per_query['baseline_joint_cos']
    for name in per_query:
        summary[name] = {}
        for m in metric_names:
            arr = per_query[name][m]
            mean, lo, hi = _bootstrap_ci(arr, args.bootstrap, args.seed)
            summary[name][m] = {'mean': mean, 'ci95_lo': lo, 'ci95_hi': hi}
            if name != 'baseline_joint_cos':
                w = _wilcoxon_signed_rank(arr, base_arr[m])
                summary[name][m]['vs_baseline'] = w

    report = {
        'milestone': 'M4',
        'config': vars(args),
        'top_k': K,
        'n_queries': len(records),
        'timings_sec_per_variant': timings,
        'metrics': summary,
    }
    Path(args.report_json).parent.mkdir(parents=True, exist_ok=True)
    with open(args.report_json, 'w') as fh:
        json.dump(report, fh, indent=2)

    # Markdown rendering
    lines = ['# v2 Ablation Report (M4)', '',
             f'- Queries: **{len(records)}**',
             f'- Top-K: **{K}**',
             f'- Bootstrap resamples: **{args.bootstrap}**', '',
             '## Per-variant ranking metrics',
             '',
             '| Variant | P@1 | P@5 | Recall@K | MAP | NDCG@K | MRR |',
             '|---|---|---|---|---|---|---|']
    for name in per_query:
        row = [name]
        for m in metric_names:
            v = summary[name][m]
            row.append(f"{v['mean']:.3f} [{v['ci95_lo']:.3f}, {v['ci95_hi']:.3f}]")
        lines.append('| ' + ' | '.join(row) + ' |')
    lines += ['', '## Wilcoxon signed-rank vs baseline (p < 0.01 = significant)', '']
    lines.append('| Variant | ' + ' | '.join(metric_names) + ' |')
    lines.append('|' + '---|' * (1 + len(metric_names)))
    for name in per_query:
        if name == 'baseline_joint_cos':
            continue
        row = [name]
        for m in metric_names:
            w = summary[name][m].get('vs_baseline', {})
            p = w.get('p_two_sided', 1.0)
            marker = '✓' if p < 0.01 else ('·' if p < 0.05 else ' ')
            row.append(f'p={p:.3g} {marker}')
        lines.append('| ' + ' | '.join(row) + ' |')
    lines += ['', '## Wall-clock per variant (seconds)', '']
    lines.append('| Variant | seconds |')
    lines.append('|---|---|')
    for name, sec in timings.items():
        lines.append(f'| {name} | {sec:.3f} |')

    with open(args.report_md, 'w') as fh:
        fh.write('\n'.join(lines) + '\n')

    print(f'wrote → {args.report_json}')
    print(f'wrote → {args.report_md}')


if __name__ == '__main__':
    main()
