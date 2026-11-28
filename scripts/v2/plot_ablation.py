#!/usr/bin/env python
"""Channel-ablation figures and table from evaluate_methods_v2.py runs.

Reads ablation_runs/index.tsv (written by ablation_eval.sh: one row per eval,
with variant 'full' for the reference model and 'drop-<channel>' for each
leave-one-channel-out model) and plots the remote-homology metrics of each
variant, averaged per superfamily as in the paper:

  rcpatk_superfamily  remote capped P@10, superfamily (hits outside the query's family)
  rcpatk_fold         remote capped P@10, fold (hits outside the query's superfamily)
  rp1_superfamily     remote P@1, superfamily
  rp1_fold            remote P@1, fold

For each variant: mean over seeds with one dot per seed and SD whiskers
(when there are 2+ seeds). The aligners (GTalign, US-align, Foldseek) are
drawn as dashed reference lines; they do not depend on the model. Each
ablated model is compared with the full model of the same seed (or the
full-model seed mean when that seed is missing) by a paired Wilcoxon test
over superfamilies: * = p < 0.05 in every seed, dagger = in some.

Subsets: 'all' (every query, per_superfamily_metrics.tsv), 'novel_family'
and 'novel_superfamily' (query types, <subset>_per_superfamily_metrics.tsv).

Outputs in --out_dir, per subset: ablation_<subset>.png (absolute values),
ablation_<subset>_delta.png (change from the full model), one PNG per
metric, and ablation_<subset>.tsv.

Example (from scripts/v2):
  python plot_ablation.py --index ../data/.../evaluation/ablation_runs/index.tsv \\
      --out_dir ../data/.../evaluation/ablation_runs/figures
"""

import argparse
import os

import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

CHANNELS = ['vdw_attractive', 'vdw_repulsive', 'distance', 'elec_attractive', 'elec_repulsive',
            'hydrophobicity']
LABELS = {'full': 'Full model', 'vdw_attractive': 'VdW attractive', 'vdw_repulsive': 'VdW repulsive',
          'distance': 'Cα distance', 'elec_attractive': 'Elec. attractive',
          'elec_repulsive': 'Elec. repulsive', 'hydrophobicity': 'Hydrophobicity'}
# Same channel colours as explain_similarity_maps.py; the full model is neutral.
COLORS = {'full': '#52514e', 'vdw_attractive': '#2a78d6', 'vdw_repulsive': '#eb6834',
          'distance': '#1baf7a', 'elec_attractive': '#eda100', 'elec_repulsive': '#e87ba4',
          'hydrophobicity': '#008300'}
INK, INK2, GRID = '#0b0b0b', '#52514e', '#e4e3df'
METRICS = {'rcpatk_superfamily': 'Remote cP@10, superfamily', 'rcpatk_fold': 'Remote cP@10, fold',
           'rp1_superfamily': 'Remote P@1, superfamily', 'rp1_fold': 'Remote P@1, fold'}
ALIGNERS = {'GTalign': 'GTalign', 'USalign': 'US-align', 'Foldseek': 'Foldseek'}
ALIGNER_STYLES = {'GTalign': (0, (6, 2)), 'USalign': (0, (2, 2)), 'Foldseek': (0, (6, 2, 1, 2))}
SUBSET_FILE = {'all': 'per_superfamily_metrics.tsv',
               'novel_family': 'novel_family_per_superfamily_metrics.tsv',
               'novel_superfamily': 'novel_superfamily_per_superfamily_metrics.tsv'}

plt.rcParams.update({'font.size': 9, 'axes.edgecolor': INK2, 'axes.labelcolor': INK,
                     'xtick.color': INK2, 'ytick.color': INK2, 'text.color': INK,
                     'savefig.dpi': 300})


def per_sf(eval_dir, subset, method='Proteogram'):
    """{metric: Series indexed by superfamily} for one method, or None if the file is missing."""
    f = os.path.join(eval_dir, SUBSET_FILE[subset])
    if not os.path.exists(f):
        return None
    t = pd.read_csv(f, sep='\t')
    t = t[t.method == method].set_index('superfamily')
    return {m: t[m].astype(float) for m in METRICS if m in t}


def channel_of(variant):
    return 'full' if variant == 'full' else variant.replace('drop-', '')


def collect(index, subset):
    """Per variant: {seed: {metric: per-superfamily Series}}, and aligner macro values."""
    data, aligners = {}, {}
    for _, r in index.iterrows():
        v = per_sf(r.eval_dir, subset)
        if v is None:
            print(f'  {subset}: no {SUBSET_FILE[subset]} in {r.eval_dir}; skipped')
            continue
        data.setdefault(channel_of(r.variant), {})[int(r.seed)] = v
        if not aligners and r.variant == 'full':
            for a in ALIGNERS:
                al = per_sf(r.eval_dir, subset, a)
                if al:
                    aligners[a] = {m: s.mean() for m, s in al.items()}
    return data, aligners


def summarise(data, metrics):
    """Rows of mean / SD / delta / paired tests per variant and metric."""
    rows = []
    full = data.get('full', {})
    for ch in ['full'] + CHANNELS:
        if ch not in data:
            continue
        for m in metrics:
            seeds = sorted(data[ch])
            vals = [data[ch][s][m].mean() for s in seeds]
            row = dict(variant=ch, metric=m, n_seeds=len(vals), seeds=','.join(map(str, seeds)),
                       mean=np.mean(vals), sd=np.std(vals, ddof=1) if len(vals) > 1 else np.nan,
                       n_superfamilies=int(data[ch][seeds[0]][m].notna().sum()))
            if ch != 'full' and full:
                deltas, ps = [], []
                full_mean = pd.concat([full[s][m] for s in full], axis=1).mean(axis=1)
                for s in seeds:
                    ref = full[s][m] if s in full else full_mean
                    both = pd.concat([data[ch][s][m], ref], axis=1).dropna()
                    deltas.append(both.iloc[:, 0].mean() - both.iloc[:, 1].mean())
                    diff = both.iloc[:, 0] - both.iloc[:, 1]
                    ps.append(wilcoxon(diff).pvalue if (diff != 0).any() else 1.0)
                row.update(delta=np.mean(deltas), deltas=','.join(f'{x:.5f}' for x in deltas),
                           sd_delta=np.std(deltas, ddof=1) if len(deltas) > 1 else np.nan,
                           p_values=','.join(f'{p:.3g}' for p in ps),
                           n_sig=sum(p < 0.05 for p in ps))
            rows.append(row)
    return pd.DataFrame(rows)


def style(ax):
    ax.yaxis.grid(True, color=GRID, lw=0.6, zorder=0)
    ax.set_axisbelow(True)
    for s in ('top', 'right'):
        ax.spines[s].set_visible(False)


def panel(ax, data, table, aligners, metric, delta=False, legend=False):
    variants = [c for c in ['full'] + CHANNELS if c in data]
    if delta:
        variants = [v for v in variants if v != 'full']
    t = table[table.metric == metric].set_index('variant')
    x = np.arange(len(variants))
    for i, v in enumerate(variants):
        key, sdk = ('delta', 'sd_delta') if delta else ('mean', 'sd')
        y, sd = t.loc[v, key], t.loc[v, sdk]
        ax.bar(i, y, 0.7, color=COLORS[v], zorder=2, linewidth=0)
        if not np.isnan(sd):
            ax.errorbar(i, y, yerr=sd, fmt='none', ecolor=INK, elinewidth=1, capsize=2.5, zorder=4)
        if delta:
            pts = [float(x) for x in str(t.loc[v, 'deltas']).split(',')]
        else:
            pts = [data[v][s][metric].mean() for s in sorted(data[v])]
        if True:
            offs = np.linspace(-0.12, 0.12, len(pts)) if len(pts) > 1 else [0]
            ax.scatter(i + np.asarray(offs), pts, s=8, color='white', edgecolor=INK, lw=0.6, zorder=5)
        if v != 'full' and 'n_sig' in t:
            k, n = t.loc[v, 'n_sig'], t.loc[v, 'n_seeds']
            mark = '*' if k == n and k > 0 else ('†' if k > 0 else '')
            if mark:
                top = y + (0 if np.isnan(sd) else sd)
                if delta and y < 0:
                    ax.text(i, y - (0 if np.isnan(sd) else sd) - 0.004, mark, ha='center', va='top',
                            fontsize=10 if mark == '*' else 7)
                else:
                    ax.text(i, top + 0.006, mark, ha='center', va='bottom',
                            fontsize=10 if mark == '*' else 7)
    if delta:
        ax.axhline(0, color=INK2, lw=0.8)
        # Noise yardstick: the full model's spread across seeds.
        sd_full = t.loc['full', 'sd'] if 'full' in t.index else np.nan
        if not np.isnan(sd_full):
            ax.axhspan(-sd_full, sd_full, color=GRID, alpha=0.8, zorder=1, lw=0,
                       label='±1 SD of the full model across seeds')
        ax.set_ylabel('Change from full model')
        ax.set_xlabel('Channel left out', fontsize=8)
    else:
        for a, vals in aligners.items():
            if metric in vals:
                ax.axhline(vals[metric], color=INK2, lw=1, ls=ALIGNER_STYLES[a], zorder=3,
                           label=ALIGNERS[a])
        ax.set_ylim(0, 1.0)
        ax.set_ylabel('Per-superfamily mean')
    ax.set_xticks(x)
    ax.set_xticklabels([('' if v == 'full' or delta else '− ') + LABELS[v] if v != 'full'
                        else LABELS[v] for v in variants], rotation=35, ha='right', fontsize=8)
    ax.set_xlim(-0.6, len(variants) - 0.4)
    ax.set_title(METRICS[metric], loc='left', fontsize=10, fontweight='bold')
    style(ax)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--index', required=True, help='ablation_runs/index.tsv from ablation_eval.sh.')
    p.add_argument('--out_dir', required=True)
    p.add_argument('--subsets', default='all,novel_family,novel_superfamily')
    p.add_argument('--metrics', default=','.join(METRICS))
    args = p.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    index = pd.read_csv(args.index, sep='\t', dtype={'drop_channels': str}).fillna('')
    metrics = [m for m in args.metrics.split(',') if m in METRICS]
    note = ('Bars: mean over seeds (dots = seeds, whiskers = SD). Paired Wilcoxon over superfamilies '
            'vs the full model: * p<0.05 in every seed, † in some.')

    for subset in args.subsets.split(','):
        data, aligners = collect(index, subset)
        if 'full' not in data or len(data) < 2:
            print(f'{subset}: need the full model and at least one ablation; skipped')
            continue
        table = summarise(data, metrics)
        # delta panels need the full model's SD row even when only drawing drops
        table['deltas'] = table.get('deltas', pd.Series(dtype=object))
        table.to_csv(os.path.join(args.out_dir, f'ablation_{subset}.tsv'), sep='\t', index=False,
                     float_format='%.4f')
        ncol = 2 if len(metrics) > 1 else 1
        nrow = int(np.ceil(len(metrics) / ncol))
        for delta in (False, True):
            fig, axes = plt.subplots(nrow, ncol, figsize=(5.2 * ncol, 3.9 * nrow), squeeze=False)
            for ax, m in zip(axes.ravel(), metrics):
                panel(ax, data, table, aligners, m, delta=delta)
            for ax in axes.ravel()[len(metrics):]:
                ax.axis('off')
            handles, labels = axes.ravel()[0].get_legend_handles_labels()
            if handles:
                fig.legend(handles, labels, loc='lower right', ncol=len(handles), fontsize=7,
                           frameon=False, bbox_to_anchor=(0.995, 0.0))
            what = 'change from the full model' if delta else 'remote homology'
            fig.suptitle(f'Channel ablation (leave one channel out), {what}: '
                         f'{subset.replace("_", " ")} queries', x=0.01, ha='left', fontsize=10)
            fig.text(0.01, 0.005, note, fontsize=7, color=INK2)
            fig.tight_layout(rect=(0, 0.03, 1, 0.96))
            fig.savefig(os.path.join(args.out_dir, f'ablation_{subset}{"_delta" if delta else ""}.png'))
            plt.close(fig)
        for m in metrics:
            fig, ax = plt.subplots(figsize=(5.4, 4.0))
            panel(ax, data, table, aligners, m)
            ax.legend(fontsize=7, frameon=False, loc='upper right')
            fig.text(0.01, 0.005, note.replace('. ', '.\n'), fontsize=6.5, color=INK2)
            fig.tight_layout(rect=(0, 0.06, 1, 1))
            fig.savefig(os.path.join(args.out_dir, f'ablation_{subset}_{m}.png'))
            plt.close(fig)
        print(f'\n== {subset} queries')
        show = table.copy()
        show['variant'] = show.variant.map(lambda v: LABELS[v] if v == 'full' else f'- {LABELS[v]}')
        with pd.option_context('display.width', 200, 'display.max_columns', 20):
            print(show.drop(columns=['seeds']).to_string(index=False, float_format=lambda v: f'{v:.3f}'))
    print(f'\nWrote {args.out_dir}')


if __name__ == '__main__':
    main()
