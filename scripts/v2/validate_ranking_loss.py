"""Validate that a ranking-loss-trained model better correlates with TM-scores.

This script measures the alignment between the model's cosine similarity
predictions and USalign TM-scores — the ground truth structural similarity.
It is the primary validation tool for the physics-informed ranking loss (idea #2).

Key metrics
-----------
- Spearman ρ: rank correlation between cosine sims and TM-scores.
  A perfect retrieval model should have ρ → 1.0.
- Kendall τ: alternative rank correlation, more robust to ties.
- Pearson r: linear correlation (how well cosine sims map to TM-score magnitude).
- NDCG@K: normalised discounted cumulative gain, directly related to MAP@K.

All metrics are computed across all off-diagonal pairs in the eval set, and
also per-query (mean ± std) so you can see variance.

Outputs
-------
- Console table comparing baseline vs. ranking-loss model on all metrics.
- Scatter plot: cosine similarity vs. TM-score (one dot per protein pair).
- Per-query Spearman histogram.
- Saved to --output_dir.

Usage
-----
    # Compare two models on the same eval set:
    python validate_ranking_loss.py \\
        --baseline_model /data/baseline_resnet18.pt \\
        --ranking_model  /data/ranking_resnet18.pt \\
        --proteograms_dir /data/proteograms_v2/eval \\
        --usalign_results /data/usalign_out_544.tsv \\
        --output_dir /data/ranking_validation

    # Validate a single model:
    python validate_ranking_loss.py \\
        --ranking_model /data/ranking_resnet18.pt \\
        --proteograms_dir /data/proteograms_v2/eval \\
        --usalign_results /data/usalign_out_544.tsv \\
        --output_dir /data/ranking_validation
"""

import argparse
import os
import sys

import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr, kendalltau, pearsonr
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from proteogram.v2.image_similarity import Img2Vec
from proteogram.v2.ranking_loss import TmScoreStore


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def embed_proteograms(model_path: str, proteograms_dir: str, device: str) -> dict:
    """Embed all proteograms in a directory and return {protein_id: embedding}."""
    img2vec = Img2Vec(model_path, proteograms_dir, device=device)
    img2vec.embed_dataset()
    # Strip .jpg extension and directory prefix from keys
    embeddings = {}
    for path, emb in img2vec.dataset.items():
        pid = os.path.splitext(os.path.basename(path))[0]
        embeddings[pid] = emb
    return embeddings


def compute_cosine_matrix(embeddings: dict) -> tuple:
    """Return (protein_ids, cosine_sim_matrix) as (list, np.ndarray)."""
    ids  = sorted(embeddings.keys())
    vecs = torch.stack([embeddings[i].squeeze() for i in ids])   # (N, d)
    vecs = F.normalize(vecs, dim=1)
    cos_mat = (vecs @ vecs.T).cpu().numpy()                      # (N, N)
    return ids, cos_mat


def build_tm_matrix(store: TmScoreStore, ids: list) -> np.ndarray:
    """Build (N, N) TM-score matrix; NaN for unknown pairs."""
    N = len(ids)
    tm = np.full((N, N), np.nan)
    np.fill_diagonal(tm, 1.0)
    for i, id1 in enumerate(ids):
        for j, id2 in enumerate(ids):
            if i != j:
                score = store.get(id1, id2)
                if score is not None:
                    tm[i, j] = score
    return tm


def rank_correlation_metrics(cos_mat: np.ndarray, tm_mat: np.ndarray) -> dict:
    """Compute global rank correlation metrics over all off-diagonal known pairs."""
    N = cos_mat.shape[0]
    cos_vals, tm_vals = [], []
    for i in range(N):
        for j in range(N):
            if i != j and not np.isnan(tm_mat[i, j]):
                cos_vals.append(cos_mat[i, j])
                tm_vals.append(tm_mat[i, j])

    if len(cos_vals) < 10:
        return {'spearman': np.nan, 'kendall': np.nan, 'pearson': np.nan,
                'n_pairs': len(cos_vals)}

    cos_arr = np.array(cos_vals)
    tm_arr  = np.array(tm_vals)

    sp, _ = spearmanr(cos_arr, tm_arr)
    kt, _ = kendalltau(cos_arr, tm_arr)
    pr, _ = pearsonr(cos_arr, tm_arr)

    return {
        'spearman': sp,
        'kendall':  kt,
        'pearson':  pr,
        'n_pairs':  len(cos_vals),
        'cos_vals': cos_arr,
        'tm_vals':  tm_arr,
    }


def per_query_spearman(cos_mat: np.ndarray, tm_mat: np.ndarray) -> np.ndarray:
    """Compute per-query Spearman ρ and return array of shape (N,)."""
    N = cos_mat.shape[0]
    rhos = []
    for i in range(N):
        mask = np.array([j != i and not np.isnan(tm_mat[i, j]) for j in range(N)])
        if mask.sum() < 5:
            continue
        sp, _ = spearmanr(cos_mat[i, mask], tm_mat[i, mask])
        rhos.append(sp)
    return np.array(rhos)


def ndcg_at_k(cos_mat: np.ndarray, tm_mat: np.ndarray, k: int = 10) -> float:
    """Mean NDCG@K using TM-scores as relevance grades."""
    N = cos_mat.shape[0]
    ndcgs = []
    for i in range(N):
        off_diag = [j for j in range(N) if j != i and not np.isnan(tm_mat[i, j])]
        if not off_diag:
            continue
        sorted_by_cos = sorted(off_diag, key=lambda j: cos_mat[i, j], reverse=True)[:k]
        sorted_by_tm  = sorted(off_diag, key=lambda j: tm_mat[i, j],  reverse=True)[:k]

        # NDCG: relevance = TM-score
        dcg  = sum(tm_mat[i, j] / np.log2(r + 2) for r, j in enumerate(sorted_by_cos))
        idcg = sum(tm_mat[i, j] / np.log2(r + 2) for r, j in enumerate(sorted_by_tm))
        ndcgs.append(dcg / idcg if idcg > 0 else 0.0)
    return float(np.mean(ndcgs)) if ndcgs else 0.0


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_scatter(metrics_baseline, metrics_ranking, output_dir: str) -> None:
    """Scatter plot: cosine sim vs TM-score for baseline and ranking models."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharex=True, sharey=True)
    fig.suptitle("Cosine Similarity vs USalign TM-Score", fontsize=14)

    for ax, m, label in zip(
        axes,
        [metrics_baseline, metrics_ranking],
        ["Baseline (CrossEntropy)", "Ranking-loss (ListNet + TM-score)"],
    ):
        if m is None or np.isnan(m.get('spearman', np.nan)):
            ax.text(0.5, 0.5, "No data", ha='center', va='center',
                    transform=ax.transAxes)
            ax.set_title(label)
            continue
        # Downsample for readability if > 5000 pairs
        cos_v = m['cos_vals']
        tm_v  = m['tm_vals']
        if len(cos_v) > 5000:
            idx = np.random.choice(len(cos_v), 5000, replace=False)
            cos_v, tm_v = cos_v[idx], tm_v[idx]

        ax.scatter(tm_v, cos_v, alpha=0.3, s=4, color='steelblue')
        ax.set_xlabel("TM-score (USalign)", fontsize=11)
        ax.set_ylabel("Cosine similarity", fontsize=11)
        ax.set_title(
            f"{label}\nSpearman ρ = {m['spearman']:.4f}  "
            f"Pearson r = {m['pearson']:.4f}",
            fontsize=10,
        )
        # Diagonal reference line
        lims = [max(ax.get_xlim()[0], ax.get_ylim()[0]),
                min(ax.get_xlim()[1], ax.get_ylim()[1])]
        ax.plot(lims, lims, 'r--', lw=0.8, alpha=0.6, label='perfect')
        ax.legend(fontsize=8)

    plt.tight_layout()
    out = os.path.join(output_dir, "scatter_cosine_vs_tmscore.png")
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Scatter plot saved → {out}")


def plot_spearman_hist(rhos_baseline, rhos_ranking, output_dir: str) -> None:
    """Per-query Spearman ρ histogram comparison."""
    fig, ax = plt.subplots(figsize=(9, 5))
    bins = np.linspace(-1, 1, 40)
    if rhos_baseline is not None and len(rhos_baseline):
        ax.hist(rhos_baseline, bins=bins, alpha=0.6, label='Baseline',
                color='steelblue')
    if rhos_ranking is not None and len(rhos_ranking):
        ax.hist(rhos_ranking, bins=bins, alpha=0.6, label='Ranking-loss',
                color='coral')
    ax.axvline(0, color='black', lw=0.8, ls='--')
    ax.set_xlabel("Per-query Spearman ρ  (cosine sim vs TM-score)", fontsize=11)
    ax.set_ylabel("Number of queries", fontsize=11)
    ax.set_title("Distribution of per-query rank correlation with TM-score", fontsize=12)
    ax.legend(fontsize=10)
    plt.tight_layout()
    out = os.path.join(output_dir, "spearman_histogram.png")
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Spearman histogram saved → {out}")


def print_table(label: str, global_m: dict, rhos: np.ndarray,
                ndcg10: float, ndcg: dict) -> None:
    """Pretty-print a metric table for one model."""
    print(f"\n{'='*60}")
    print(f"  {label}")
    print(f"{'='*60}")
    if global_m is None:
        print("  (no data)")
        return
    print(f"  Pairs evaluated:      {global_m['n_pairs']:,}")
    print(f"  Spearman ρ (global):  {global_m['spearman']:.4f}")
    print(f"  Kendall  τ (global):  {global_m['kendall']:.4f}")
    print(f"  Pearson  r (global):  {global_m['pearson']:.4f}")
    if rhos is not None and len(rhos):
        print(f"  Per-query Spearman:   mean={rhos.mean():.4f}  "
              f"std={rhos.std():.4f}  "
              f"median={np.median(rhos):.4f}")
    print(f"  NDCG@10:              {ndcg10:.4f}")
    for k_val, val in ndcg.items():
        print(f"  NDCG@{k_val:<3d}:             {val:.4f}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--ranking_model', '-r', required=True,
                        help="Path to ranking-loss-trained model (.pt).")
    parser.add_argument('--baseline_model', '-b', default=None,
                        help="Path to baseline (CrossEntropy-trained) model (.pt). "
                             "If omitted, only the ranking model is evaluated.")
    parser.add_argument('--proteograms_dir', '-p', required=True,
                        help="Directory of eval proteogram JPGs.")
    parser.add_argument('--usalign_results', '-u', required=True,
                        help="USalign all-vs-all TSV (columns: #PDBchain1, PDBchain2, TM1, TM2).")
    parser.add_argument('--output_dir', '-o', default='ranking_validation',
                        help="Directory to save figures and summary CSV.")
    parser.add_argument('--top_k', type=int, default=10,
                        help="K for NDCG@K (default: 10).")
    parser.add_argument('--device', default=None,
                        help="Torch device (default: auto-detect).")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    device = args.device or ('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # Load TM-score ground truth
    print(f"\nLoading TM-scores from {args.usalign_results} …")
    store = TmScoreStore(args.usalign_results)

    results = {}
    for label, model_path in [
        ("Baseline", args.baseline_model),
        ("Ranking-loss", args.ranking_model),
    ]:
        if model_path is None:
            results[label] = None
            continue

        print(f"\nEmbedding proteins with {label} model: {model_path} …")
        embeddings = embed_proteograms(model_path, args.proteograms_dir, device)
        ids, cos_mat = compute_cosine_matrix(embeddings)
        tm_mat = build_tm_matrix(store, ids)

        known_frac = (~np.isnan(tm_mat) & ~np.eye(len(ids), dtype=bool)).mean()
        print(f"  {len(ids)} proteins | {known_frac*100:.1f}% pairs have TM-scores")

        global_m = rank_correlation_metrics(cos_mat, tm_mat)
        rhos = per_query_spearman(cos_mat, tm_mat)
        ndcg10 = ndcg_at_k(cos_mat, tm_mat, k=args.top_k)
        ndcg_all = {k: ndcg_at_k(cos_mat, tm_mat, k=k) for k in [5, 20, 50]}

        print_table(label, global_m, rhos, ndcg10, ndcg_all)

        results[label] = {
            'global': global_m,
            'rhos': rhos,
            'ndcg10': ndcg10,
            'ndcg_all': ndcg_all,
        }

    # Save summary CSV
    rows = []
    for label, res in results.items():
        if res is None:
            continue
        m = res['global']
        rows.append({
            'model': label,
            'n_pairs': m['n_pairs'],
            'spearman': round(m['spearman'], 4),
            'kendall':  round(m['kendall'], 4),
            'pearson':  round(m['pearson'], 4),
            'ndcg10':   round(res['ndcg10'], 4),
            'mean_per_query_spearman': round(res['rhos'].mean(), 4) if res['rhos'] is not None else None,
        })
    if rows:
        csv_path = os.path.join(args.output_dir, 'ranking_validation_summary.csv')
        pd.DataFrame(rows).to_csv(csv_path, index=False)
        print(f"\nSummary CSV saved → {csv_path}")

    # Plots
    bl = results.get('Baseline')
    rk = results.get('Ranking-loss')

    plot_scatter(
        bl['global'] if bl else None,
        rk['global'] if rk else None,
        args.output_dir,
    )
    plot_spearman_hist(
        bl['rhos'] if bl else None,
        rk['rhos'] if rk else None,
        args.output_dir,
    )

    print("\nValidation complete.")


if __name__ == '__main__':
    main()
