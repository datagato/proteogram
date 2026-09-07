"""Validate the impact of global percentile normalisation vs. per-protein min-max.

This is the before/after validation for idea #2 (global percentile
normalisation, see docs/improvements_v2.md).  It answers the question the
design doc asserts but never measures: does global normalisation actually
preserve physically meaningful inter-protein energy scale, compared to the
original per-protein min-max strategy?

It operates on the raw per-channel ``.npy`` matrices produced by
``create_v2_proteograms.py --save_npy_matrices`` (physical units: kJ/mol for
energy channels, Å for distance) — the same inputs ``compute_norm_stats.py``
consumes to build ``norm_stats.json``. It does NOT require re-running MD or
retraining a model, so it is cheap to run after regenerating a norm_stats.json.

Key metrics
-----------
- Inter-protein variance of mean pixel intensity, per channel, per strategy.
  Per-protein min-max always rescales each protein to use close to the full
  [0, 255] range by construction, which destroys scale differences between
  proteins. Global normalisation should retain markedly more of that spread.
  Reported as the ratio (global variance / per-protein variance) — the
  headline "did this do anything" number.

- Correlation between each protein's raw 99th-percentile value (a per-protein
  proxy for "how energetic is this channel for this protein") and its
  normalised mean pixel intensity, under each strategy. Per-protein
  normalisation should show ~0 correlation (it erases the raw scale by
  design); global normalisation should show strong correlation (it's a
  corpus-wide affine map, so this is close to a consistency check, but a low
  value flags a broken/overly-clipped norm_stats.json).

- Clipping/saturation rate for global normalisation: fraction of non-zero
  pixels pinned to exactly 0 or 255 after clipping to [p_low, p_high]. High
  values mean the percentile bounds are too tight for this corpus (or the
  corpus used to fit them doesn't match the corpus being normalised).

- Dynamic-range sanity check for per-protein normalisation (should always be
  ~[0, 255] barring constant matrices) as a baseline sanity check.

Outputs
-------
- Console table comparing both strategies on all metrics, per channel.
- Scatter plot: raw p99 vs. normalised mean, one panel per strategy.
- Saved per-protein/per-channel stats CSV.

Usage
-----
    python validate_normalisation.py \\
        --npy_dir /data/proteograms_v2/energy_matrices \\
        --norm_stats_file /data/norm_stats.json \\
        --output_dir /data/normalisation_validation
"""

import argparse
import glob
import os

import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
from tqdm import tqdm

from proteogram.v2.normalisation import (
    CHANNEL_NAMES,
    load_norm_stats,
    normalize_map_global,
    normalize_map_perprotein,
)


def _collect_protein_ids(npy_dir: str, channel: str) -> list:
    pattern = os.path.join(npy_dir, f'*_{channel}.npy')
    files = sorted(glob.glob(pattern))
    suffix = f'_{channel}.npy'
    return [os.path.basename(f)[: -len(suffix)] for f in files]


def compute_per_protein_stats(npy_dir: str, norm_stats: dict, max_samples: int) -> pd.DataFrame:
    """Compute raw and normalised stats for every protein x channel.

    Returns a long-form DataFrame with one row per (protein_id, channel):
    raw_p99, norm_mean_perprotein, norm_mean_global, global_saturated_frac,
    perprotein_min, perprotein_max.
    """
    rows = []
    for channel in CHANNEL_NAMES:
        ids = _collect_protein_ids(npy_dir, channel)
        if not ids:
            print(f'  WARNING: no .npy files found for channel "{channel}" in {npy_dir}')
            continue
        if len(ids) > max_samples:
            rng = np.random.default_rng(seed=42)
            ids = list(rng.choice(ids, size=max_samples, replace=False))

        for pid in tqdm(ids, desc=f'  {channel}', leave=False):
            arr = np.load(os.path.join(npy_dir, f'{pid}_{channel}.npy')).astype(np.float64)
            nonzero = arr[arr != 0]
            if nonzero.size == 0:
                continue

            raw_p99 = float(np.percentile(nonzero, 99))

            pp_arr, pp_err = normalize_map_perprotein(arr)
            pp_nonzero = pp_arr[arr != 0]

            row = {
                'protein_id': pid,
                'channel': channel,
                'raw_p99': raw_p99,
                'perprotein_mean': float(pp_nonzero.mean()) if pp_nonzero.size else np.nan,
                'perprotein_min': int(pp_arr.min()),
                'perprotein_max': int(pp_arr.max()),
                'perprotein_err': pp_err,
            }

            if norm_stats is not None and channel in norm_stats:
                g_arr, g_err = normalize_map_global(arr, norm_stats[channel])
                g_nonzero = g_arr[arr != 0]
                row['global_mean'] = float(g_nonzero.mean()) if g_nonzero.size else np.nan
                row['global_saturated_frac'] = (
                    float(np.mean((g_nonzero == 0) | (g_nonzero == 255)))
                    if g_nonzero.size else np.nan
                )
                row['global_err'] = g_err
            else:
                row['global_mean'] = np.nan
                row['global_saturated_frac'] = np.nan
                row['global_err'] = 'no norm_stats for channel'

            rows.append(row)

    return pd.DataFrame(rows)


def summarise(df: pd.DataFrame) -> pd.DataFrame:
    """Per-channel summary: variance ratio, correlations, saturation, range."""
    summary_rows = []
    for channel, grp in df.groupby('channel'):
        grp = grp.dropna(subset=['perprotein_mean'])
        if len(grp) < 3:
            continue

        pp_var = float(grp['perprotein_mean'].var())
        has_global = grp['global_mean'].notna().sum() >= 3

        row = {
            'channel': channel,
            'n_proteins': len(grp),
            'perprotein_mean_variance': pp_var,
            'perprotein_min_of_mins': int(grp['perprotein_min'].min()),
            'perprotein_max_of_maxs': int(grp['perprotein_max'].max()),
        }

        if has_global:
            g = grp.dropna(subset=['global_mean'])
            g_var = float(g['global_mean'].var())
            row['global_mean_variance'] = g_var
            row['variance_ratio_global_over_perprotein'] = (
                g_var / pp_var if pp_var > 0 else float('nan')
            )
            row['global_saturation_rate'] = float(g['global_saturated_frac'].mean())

            pp_r, _ = pearsonr(g['raw_p99'], g['perprotein_mean'])
            pp_rho, _ = spearmanr(g['raw_p99'], g['perprotein_mean'])
            gl_r, _ = pearsonr(g['raw_p99'], g['global_mean'])
            gl_rho, _ = spearmanr(g['raw_p99'], g['global_mean'])

            row['pearson_r_perprotein_vs_raw_p99'] = pp_r
            row['spearman_rho_perprotein_vs_raw_p99'] = pp_rho
            row['pearson_r_global_vs_raw_p99'] = gl_r
            row['spearman_rho_global_vs_raw_p99'] = gl_rho
        else:
            for k in ('global_mean_variance', 'variance_ratio_global_over_perprotein',
                      'global_saturation_rate', 'pearson_r_perprotein_vs_raw_p99',
                      'spearman_rho_perprotein_vs_raw_p99', 'pearson_r_global_vs_raw_p99',
                      'spearman_rho_global_vs_raw_p99'):
                row[k] = float('nan')

        summary_rows.append(row)

    return pd.DataFrame(summary_rows)


def plot_scatter(df: pd.DataFrame, output_dir: str) -> None:
    channels = [c for c in CHANNEL_NAMES if c in df['channel'].unique()]
    if not channels:
        return
    fig, axes = plt.subplots(len(channels), 2, figsize=(9, 3.2 * len(channels)), squeeze=False)
    for i, channel in enumerate(channels):
        grp = df[df['channel'] == channel]
        axes[i][0].scatter(grp['raw_p99'], grp['perprotein_mean'], s=8, alpha=0.5)
        axes[i][0].set_title(f'{channel}: per-protein norm')
        axes[i][0].set_xlabel('raw p99 (physical units)')
        axes[i][0].set_ylabel('normalised mean pixel')

        g = grp.dropna(subset=['global_mean'])
        axes[i][1].scatter(g['raw_p99'], g['global_mean'], s=8, alpha=0.5, color='darkorange')
        axes[i][1].set_title(f'{channel}: global percentile norm')
        axes[i][1].set_xlabel('raw p99 (physical units)')
        axes[i][1].set_ylabel('normalised mean pixel')
    fig.tight_layout()
    out_path = os.path.join(output_dir, 'raw_vs_normalised_scatter.png')
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f'Saved scatter plot -> {out_path}')


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument('--npy_dir', required=True,
                        help='Directory of raw per-channel .npy matrices '
                             '(from create_v2_proteograms.py --save_npy_matrices).')
    parser.add_argument('--norm_stats_file', required=True,
                        help='Path to norm_stats.json (from compute_norm_stats.py).')
    parser.add_argument('--output_dir', default='normalisation_validation',
                        help='Directory for the stats CSV and scatter plot.')
    parser.add_argument('--max_samples', type=int, default=2000,
                        help='Max proteins to sample per channel (default: 2000).')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    norm_stats = load_norm_stats(args.norm_stats_file)

    print(f'Computing per-protein stats from {args.npy_dir} ...')
    df = compute_per_protein_stats(args.npy_dir, norm_stats, args.max_samples)
    if df.empty:
        print('No data found — check --npy_dir.')
        return

    stats_path = os.path.join(args.output_dir, 'per_protein_normalisation_stats.csv')
    df.to_csv(stats_path, index=False)
    print(f'Saved per-protein stats -> {stats_path}')

    summary = summarise(df)
    summary_path = os.path.join(args.output_dir, 'normalisation_summary.csv')
    summary.to_csv(summary_path, index=False)

    print('\nGlobal percentile vs. per-protein min-max normalisation')
    print('=' * 78)
    for _, row in summary.iterrows():
        print(f"\nChannel: {row['channel']}  (n={int(row['n_proteins'])} proteins)")
        print(f"  Per-protein dynamic range across corpus: "
              f"[{row['perprotein_min_of_mins']}, {row['perprotein_max_of_maxs']}]  "
              f"(expect ~[0, 255] -- normalisation always stretches to full range)")
        print(f"  Inter-protein variance of mean pixel value:")
        print(f"    per-protein min-max: {row['perprotein_mean_variance']:.2f}")
        print(f"    global percentile:   {row['global_mean_variance']:.2f}")
        print(f"    ratio (global / per-protein): {row['variance_ratio_global_over_perprotein']:.2f}x"
              f"  <- headline number: >1 means global norm preserves more "
              f"inter-protein signal")
        print(f"  Correlation(raw p99, normalised mean):")
        print(f"    per-protein min-max: pearson r={row['pearson_r_perprotein_vs_raw_p99']:.3f}, "
              f"spearman rho={row['spearman_rho_perprotein_vs_raw_p99']:.3f}  "
              f"(expect ~0 -- scale info is destroyed by design)")
        print(f"    global percentile:   pearson r={row['pearson_r_global_vs_raw_p99']:.3f}, "
              f"spearman rho={row['spearman_rho_global_vs_raw_p99']:.3f}  "
              f"(expect close to 1 -- flags a broken/mismatched norm_stats.json if low)")
        print(f"  Global normalisation saturation rate (pixels clipped to 0 or 255): "
              f"{row['global_saturation_rate']:.1%}  "
              f"(>5-10% suggests percentile bounds are too tight for this corpus)")

    plot_scatter(df, args.output_dir)
    print(f'\nSummary saved -> {summary_path}')
    print('\nDone.')


if __name__ == '__main__':
    main()
