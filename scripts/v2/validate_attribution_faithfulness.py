"""Measure whether proteogram attribution maps are actually faithful.

An attribution map is only worth interpreting if the regions it highlights are
the regions the model actually relies on.  Saliency methods can produce
confident-looking heatmaps that carry little information about the model's
computation, so this script tests that property directly instead of assuming
it, using standard deletion / insertion curves (Petsiuk et al., RISE, 2018).

The test
--------
For a query→target pair with cosine similarity S:

- **Deletion**: progressively replace the highest-attributed regions of the
  query with the neutral sentinel (128 — the same value used for padding and
  structural zeros), re-embed, and re-measure cosine similarity.  A faithful
  attribution makes S collapse quickly, so a LOW deletion AUC is good.
- **Insertion**: start from an all-neutral image and progressively restore the
  highest-attributed regions.  A faithful attribution recovers S quickly, so a
  HIGH insertion AUC is good.

Both are compared against a random-ordering baseline on the same pair.  An
attribution method that cannot beat random ordering is not explaining the
model, whatever its heatmaps look like.

Methods compared
----------------
- ``gradcam``   — the combined Grad-CAM map (``GradCAM.compute``).  Note this
  is computed at the last conv layer (7x7 for a 200x200 proteogram) and
  bilinearly upsampled, so it cannot resolve individual residue pairs.
- ``gradxinput`` — the baseline-corrected Grad × Input attribution
  (``GradCAM.compute_decomposed``), summed over the three channels.  This one
  is genuinely per-pixel.
- ``shapley``   — per-residue Shapley values via permutation sampling
  (``shapley.residue_shapley``).  Axiomatically grounded but far more
  expensive: ``--shapley_permutations * (n_residues + 1)`` forward passes per
  pair, versus one backward pass for the gradient methods.  This arm exists to
  answer whether that cost buys real faithfulness — if it does not beat
  ``gradxinput`` here, the cheaper method is the right default.  Residue
  granularity only.
- ``random``    — control.

Granularity
-----------
``--granularity residue`` (default) masks whole residues (row i + column i
together), which matches how a proteogram encodes structure and is the fair
setting for comparing against the coarse Grad-CAM map.  ``pixel`` masks
individual residue *pairs*.

Usage
-----
    # From scripts/v2/
    uv run python validate_attribution_faithfulness.py \\
        --model_file /path/to/model.pt \\
        --proteograms_dir /path/to/proteograms/eval \\
        --auto \\
        --usalign_results /path/to/usalign_out.tsv \\
        --annotations_tsv /path/to/annotations.tsv \\
        --output_dir /path/to/faithfulness
"""

import argparse
import os
import sys

import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from proteogram.v2 import Img2Vec
from proteogram.v2.gradcam import GradCAM
from proteogram.v2.masking import (
    apply_mask,
    cosine,
    embed_array,
    load_padded_array,
    residue_scores,
)
from proteogram.v2.shapley import residue_shapley

# Pair selection helpers are shared with the energy-channel explainer.
from explain_energy_channels import (
    auto_select_pairs,
    find_proteogram,
    load_pairs_from_file,
)

#: Methods whose attribution is only defined per residue, not per pixel.
RESIDUE_ONLY_METHODS = ('shapley',)


# ---------------------------------------------------------------------------
# Region ordering
# ---------------------------------------------------------------------------

def ordered_regions(attr, granularity: str, rng) -> list:
    """Return regions ordered most- to least-important.

    Each region is either a residue index (``residue``) or a flat pixel index
    (``pixel``).

    Args:
        attr:        ``(H, W)`` pixel attribution map, a ``(H,)`` per-residue
                     score vector, or ``None`` for a random ordering.
        granularity: ``residue`` or ``pixel``.
        rng:         Numpy Generator used for the random ordering.
    """
    if granularity == 'residue':
        if attr is None:
            order = np.arange(200)
            rng.shuffle(order)
            return list(order)
        scores = attr if attr.ndim == 1 else residue_scores(attr)
        return list(np.argsort(-scores))

    # pixel granularity
    if attr is None:
        order = np.arange(200 * 200)
        rng.shuffle(order)
        return list(order)
    if attr.ndim == 1:
        raise ValueError(
            'per-residue attribution cannot order individual pixels; '
            'run this method with --granularity residue')
    return list(np.argsort(-attr.ravel()))


# ---------------------------------------------------------------------------
# Curves
# ---------------------------------------------------------------------------

def _auc(xs: np.ndarray, ys: np.ndarray) -> float:
    trapz = getattr(np, 'trapezoid', None) or np.trapz
    return float(trapz(ys, xs))


def curve(embed_net, device, query_arr, target_emb, order, granularity,
          n_steps: int, insertion: bool) -> tuple:
    """Compute a deletion or insertion curve.

    Returns (fractions, cosine values, AUC).
    """
    total = len(order)
    fractions = np.linspace(0.0, 1.0, n_steps + 1)
    sims = []
    for f in fractions:
        k = int(round(f * total))
        regions = order[:k]
        arr = apply_mask(query_arr, regions, granularity, keep=insertion)
        emb = embed_array(embed_net, arr, device)
        sims.append(cosine(emb, target_emb))
    sims = np.array(sims)
    return fractions, sims, _auc(fractions, sims)


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument('--model_file', '-m', required=True)
    parser.add_argument('--proteograms_dir', '-p', required=True)
    parser.add_argument('--output_dir', '-o', default='faithfulness')
    parser.add_argument('--pairs_file', default=None,
                        help='TSV (no header): query_id, target_id.')
    parser.add_argument('--auto', action='store_true',
                        help='Auto-select pairs from USalign results.')
    parser.add_argument('--usalign_results', default=None)
    parser.add_argument('--annotations_tsv', default=None)
    parser.add_argument('--top_k_auto', type=int, default=5,
                        help='Pairs per category in auto mode (default: 5).')
    parser.add_argument('--granularity', choices=['residue', 'pixel'],
                        default='residue',
                        help='Mask whole residues (default) or single pixels.')
    parser.add_argument('--n_steps', type=int, default=20,
                        help='Points per curve (default: 20).')
    parser.add_argument('--no_shapley', action='store_true',
                        help='Skip the Shapley arm (it dominates runtime).')
    parser.add_argument('--shapley_permutations', type=int, default=3,
                        help='Permutations sampled per pair for residue-level '
                             'Shapley (default: 3). Cost is '
                             'n_permutations * (n_residues + 1) forward passes '
                             'per pair. Low values are noisy — raise until the '
                             'ranking stabilises.')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--device', default=None)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    device = args.device or ('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}')

    if args.pairs_file:
        pairs_by_category = {'specified': load_pairs_from_file(args.pairs_file)}
    elif args.auto:
        if not args.usalign_results or not args.annotations_tsv:
            parser.error('--auto requires --usalign_results and --annotations_tsv')
        pairs_by_category = auto_select_pairs(
            args.usalign_results, args.annotations_tsv, top_k=args.top_k_auto)
    else:
        parser.error('Specify either --pairs_file or --auto')

    img2vec = Img2Vec(args.model_file, args.proteograms_dir, device=device)
    gcam = GradCAM(embed_net=img2vec.embed, device=str(img2vec.device))
    rng = np.random.default_rng(args.seed)

    rows = []
    for category, pairs in pairs_by_category.items():
        print(f'\n--- Category: {category} ({len(pairs)} pairs) ---')
        for query_id, target_id in tqdm(pairs, leave=False):
            try:
                q_path = find_proteogram(query_id, args.proteograms_dir)
                t_path = find_proteogram(target_id, args.proteograms_dir)
            except FileNotFoundError as exc:
                print(f'  SKIP: {exc}')
                continue

            q_arr = load_padded_array(q_path)
            t_arr = load_padded_array(t_path)
            t_emb = embed_array(img2vec.embed, t_arr, img2vec.device)
            base_sim = cosine(embed_array(img2vec.embed, q_arr, img2vec.device),
                              t_emb)

            # Attribution maps to compare
            combined, _ = gcam.compute_from_paths(q_path, t_path)
            _, attr_raw, _ = gcam.compute_decomposed_from_paths(q_path, t_path)
            maps = {
                'gradcam':    combined,
                'gradxinput': attr_raw.sum(axis=0),
                'random':     None,
            }

            # Shapley is residue-level only and costs
            # n_permutations * (n_residues + 1) forward passes per pair.
            if not args.no_shapley and args.granularity == 'residue':
                maps['shapley'] = residue_shapley(
                    img2vec.embed, img2vec.device, q_arr, t_emb,
                    n_permutations=args.shapley_permutations, rng=rng)

            for method, amap in maps.items():
                order = ordered_regions(amap, args.granularity, rng)
                _, del_sims, del_auc = curve(
                    img2vec.embed, img2vec.device, q_arr, t_emb, order,
                    args.granularity, args.n_steps, insertion=False)
                _, ins_sims, ins_auc = curve(
                    img2vec.embed, img2vec.device, q_arr, t_emb, order,
                    args.granularity, args.n_steps, insertion=True)
                rows.append({
                    'category': category,
                    'query_id': query_id,
                    'target_id': target_id,
                    'method': method,
                    'base_cosine': base_sim,
                    'deletion_auc': del_auc,
                    'insertion_auc': ins_auc,
                    'faithfulness_gap': ins_auc - del_auc,
                    'deletion_curve': ';'.join(f'{v:.4f}' for v in del_sims),
                    'insertion_curve': ';'.join(f'{v:.4f}' for v in ins_sims),
                })

    if not rows:
        print('No pairs evaluated — check inputs.')
        return

    df = pd.DataFrame(rows)
    csv_path = os.path.join(args.output_dir, 'faithfulness_curves.csv')
    df.to_csv(csv_path, index=False)
    print(f'\nPer-pair curves saved → {csv_path}')

    # Summary
    print(f'\nAttribution faithfulness  (granularity={args.granularity}, '
          f'n={df["query_id"].nunique()} queries)')
    print('=' * 78)
    print(f'{"method":<12} | {"deletion AUC":>13} | {"insertion AUC":>14} | {"gap":>8}')
    print('-' * 78)
    summary = df.groupby('method')[
        ['deletion_auc', 'insertion_auc', 'faithfulness_gap']].mean()
    for method in ['gradcam', 'gradxinput', 'shapley', 'random']:
        if method not in summary.index:
            continue
        r = summary.loc[method]
        print(f'{method:<12} | {r["deletion_auc"]:>13.4f} | '
              f'{r["insertion_auc"]:>14.4f} | {r["faithfulness_gap"]:>8.4f}')
    print('-' * 78)
    print('Lower deletion AUC = better.  Higher insertion AUC = better.')
    print('A method whose gap does not clearly beat `random` is not explaining')
    print('the model, regardless of how its heatmaps look.')

    summary.to_csv(os.path.join(args.output_dir, 'faithfulness_summary.csv'))

    # Mean curves plot
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    fractions = np.linspace(0.0, 1.0, args.n_steps + 1)
    for method in ['gradcam', 'gradxinput', 'shapley', 'random']:
        sub = df[df['method'] == method]
        if sub.empty:
            continue
        for ax, col, title in (
                (axes[0], 'deletion_curve', 'Deletion (lower is better)'),
                (axes[1], 'insertion_curve', 'Insertion (higher is better)')):
            curves = np.array([[float(v) for v in s.split(';')]
                               for s in sub[col]])
            ax.plot(fractions, curves.mean(axis=0), marker='o', ms=3,
                    label=method)
            ax.set_title(title)
            ax.set_xlabel(f'fraction of {args.granularity}s masked')
            ax.set_ylabel('cosine similarity to target')
            ax.legend()
            ax.grid(alpha=0.3)
    fig.tight_layout()
    plot_path = os.path.join(args.output_dir, 'faithfulness_curves.png')
    fig.savefig(plot_path, dpi=150)
    plt.close(fig)
    print(f'\nCurves plot saved → {plot_path}')
    print('\nDone.')


if __name__ == '__main__':
    main()
