#!/usr/bin/env python
"""Explain Proteogram similarity: which regions of two proteins made them similar.

For each query -> target pair this splits the search cosine similarity exactly
over the model's last-layer grid cells (proteogram.v2.similarity_maps), maps
the split back onto both proteograms and onto residues, and optionally checks
it against the structural alignment from USalign.

What it computes per pair
-------------------------
- Exact maps: a query map and a target map (each sums to the cosine) and the
  query-cell x target-cell matrix, whose top entries are drawn as links ("this
  query region matched that target region"). With --center_list, also centered
  maps that remove what every protein shares (corpus mean embedding).
- Residue scores: each map is upsampled to the native L x L proteogram, the
  lower triangle's 180-degree rotation is undone, and pixel scores are summed
  per residue (upper = VdW + C-alpha distance, lower = electrostatics +
  hydrophobicity).
- Structural check (needs USalign and the .ent structures):
    * residue AUROC: do high-scoring residues fall in the USalign-superposed
      core (aligned pairs within 5 A)?
    * conserved-contact enrichment: share of the pair-score mass on contacts
      conserved under the alignment, against a null that shifts scores along
      each diagonal (keeps the distance-from-diagonal profile, since local
      contacts are both salient and easy to align);
    * link agreement: for the top cell-pair links, the fraction of the query
      region's aligned residues whose partners fall in the linked target
      region, against the average over all target cells.
  Interpret the check on confidently aligned pairs (TM-score >= 0.5); for
  remote pairs the alignment itself is weak.
- Faithfulness (--faithfulness): deletion / insertion curves, removing the
  query's top cells (by the exact map) versus random cells.

Outputs (--out_dir, default <proteograms_dir>/../evaluation/explain_runs/explain_<time>)
---------------------------------------------------------------------------------------
summary.tsv              one row per pair: cosine, alignment, all check metrics
report.txt               console output, including means by SCOPe relation and TM-score
run_info.json            command, arguments, model meta, git commit
pairs/<q>__<t>/          maps.npz, residues_query.tsv, residues_target.tsv, links.tsv,
                         figure.png (first --max_figures pairs), query/target PDBs with
                         scores in the B-factor column (query superposed onto the target
                         when USalign ran), view_scores.pml / view_links.pml for PyMOL,
                         and pymol_scores.png / pymol_links.png when PyMOL is found.

Choosing pairs
--------------
  --pair Q T                     one pair (repeatable)
  --pairs_file FILE              two whitespace/tab-separated IDs per line
  --similarity_tsv FILE          search results; each query's top --top_k hits
      [--queries FILE] [--relations fold,class] [--max_pairs N]
  IDs are SCOPe sids (d1afwb1); image paths or names with .png also work.
  Note: similarity/proteogram_similarity_results-eval.tsv is overwritten by every
  search, so copy the one that matches --model_file into the run first.

Examples (from scripts/v2)
--------------------------
  python explain_similarity_maps.py --model_file ../data/.../proteogram_model_...acc67.6.pt \\
      --pair d1afwb1 d2aq7a1 --pair d1afwb1 d2d3tc1

  python explain_similarity_maps.py --model_file MODEL.pt --similarity_tsv SIM.tsv \\
      --queries ../data/.../heldout_family_test_list_...lst --top_k 1 \\
      --center_list ../data/.../heldout_family_eval_corpus_list_...lst --faithfulness

PyMOL is optional and only run as a program (pymol -cq), so it is not a Python
dependency. USalign defaults to ~/Documents/bin/USalign/USalign.
"""

import argparse
import datetime
import json
import os
import shutil
import subprocess
import sys

import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import ConnectionPatch, Patch, Rectangle

from proteogram.common import read_yaml
from proteogram.v2 import similarity_maps as sm
from proteogram.v2 import usalign as ua

trapz = getattr(np, 'trapezoid', None) or np.trapz

# Palette (categorical slots 1-4) and ink, as in the paper figures.
LINK_COLORS = ['#2a78d6', '#eb6834', '#1baf7a', '#eda100']
INK, INK2, GRID = '#0b0b0b', '#52514e', '#e4e3df'
HEAT = LinearSegmentedColormap.from_list(
    'heat', [(0.98, 0.55, 0.30, 0.0), (0.98, 0.55, 0.30, 0.35), (1.0, 0.72, 0.45, 0.85)])
SCORE_RAMP = ['#f2f1ee', '#f6dccf', '#f3c3ab', '#f0a985', '#ec8f62',
              '#e3743f', '#cf5d28', '#b04a1e', '#8c3a17', '#6b2c12']


class Tee:
    """Copy stdout to a file."""
    def __init__(self, path):
        self.f = open(path, 'w')
        self.out = sys.stdout

    def write(self, s):
        self.out.write(s)
        self.f.write(s)

    def flush(self):
        self.out.flush()
        self.f.flush()


def norm_id(x):
    b = os.path.basename(str(x).strip())
    return b.rsplit('.', 1)[0] if b.lower().endswith(('.png', '.jpg', '.jpeg')) else b


def git_commit():
    try:
        here = os.path.dirname(os.path.abspath(__file__))
        c = subprocess.run(['git', 'rev-parse', 'HEAD'], cwd=here, capture_output=True,
                           text=True).stdout.strip()
        dirty = subprocess.run(['git', 'status', '--porcelain'], cwd=here, capture_output=True,
                               text=True).stdout.strip()
        return c + ('-dirty' if dirty else '')
    except Exception:
        return ''


# ---------------------------------------------------------------------------
# Inputs
# ---------------------------------------------------------------------------

def read_ids(path):
    with open(path) as f:
        return [norm_id(ln.split()[0]) for ln in f if ln.strip() and not ln.startswith('#')]


def relation(a, b, annot):
    """Closest shared SCOPe level of two sids, from the annotation table."""
    if annot is None or a not in annot.index or b not in annot.index:
        return 'unknown'
    x, y = annot.loc[a], annot.loc[b]
    for lvl, col in (('family', 'SCOPeFamily'), ('superfamily', 'SCOPeSuperfamily'),
                     ('fold', 'SCOPeFold'), ('class', 'SCOPeClass')):
        if x[col] == y[col]:
            return lvl
    return 'none'


def build_pairs(args, annot):
    """List of (query, target, extra-columns dict). A --pairs_file with a header
    row naming 'query' and 'target' (e.g. select_explain_pairs.py's pairs.tsv)
    has its other columns carried into summary.tsv."""
    pairs = [(norm_id(q), norm_id(t), {}) for q, t in (args.pair or [])]
    if args.pairs_file:
        with open(args.pairs_file) as f:
            first = f.readline().split()
        if 'query' in first and 'target' in first:
            df = pd.read_csv(args.pairs_file, sep='\t', dtype=str)
            for _, r in df.iterrows():
                extra = {k: v for k, v in r.items() if k not in ('query', 'target')}
                pairs.append((norm_id(r['query']), norm_id(r['target']), extra))
        else:
            with open(args.pairs_file) as f:
                for ln in f:
                    p = ln.split()
                    if len(p) >= 2 and not ln.startswith('#'):
                        pairs.append((norm_id(p[0]), norm_id(p[1]), {}))
    if args.similarity_tsv:
        keep = set(read_ids(args.queries)) if args.queries else None
        rels = set(args.relations.split(',')) if args.relations else None
        sim = pd.read_csv(args.similarity_tsv, sep='\t', dtype=str)
        for _, row in sim.iterrows():
            q = norm_id(row['query_image'])
            if keep is not None and q not in keep:
                continue
            hits = [norm_id(v.rsplit(',', 1)[0]) for v in row.iloc[1:] if isinstance(v, str)]
            hits = [h for h in hits if h != q][:args.top_k]
            for h in hits:
                if rels is None or relation(q, h, annot) in rels:
                    pairs.append((q, h, {}))
    # The same pair can sit in two strata; keep each (query, target, stratum) once.
    seen, out = set(), []
    for q, t, extra in pairs:
        key = (q, t, extra.get('stratum'))
        if key not in seen and q != t:
            seen.add(key)
            out.append((q, t, extra))
    return out[:args.max_pairs] if args.max_pairs else out


class StructureFinder:
    """Find <sid>.ent under a SCOPe pdbstyle tree (<dir>/<sid[2:4]>/<sid>.ent)."""
    def __init__(self, root, production_dir):
        self.root, self.production_dir, self.index = root, production_dir, None

    def ent(self, sid):
        if not self.root:
            return None
        p = os.path.join(self.root, sid[2:4], sid + '.ent')
        if os.path.exists(p):
            return p
        if self.index is None:
            self.index = {}
            for d, _, files in os.walk(self.root):
                for f in files:
                    if f.endswith('.ent'):
                        self.index[f[:-4]] = os.path.join(d, f)
        return self.index.get(sid)

    def production(self, sid):
        p = os.path.join(self.production_dir or '', sid + '_production.pdb')
        return p if os.path.exists(p) else None

    def residues(self, sid, L):
        """Proteogram-ordered residues whose count matches the image (L), else None.
        Prefers the crystal .ent; falls back to the CG production structure."""
        for kind, path in (('ent', self.ent(sid)), ('production', self.production(sid))):
            if path:
                pres = ua.proteogram_residues(path)
                if len(pres) == L:
                    return pres, kind
                print(f'  {sid}: {kind} structure has {len(pres)} residues, image has {L}; skipped')
        return None, None


# ---------------------------------------------------------------------------
# Structural checks
# ---------------------------------------------------------------------------

def auroc(score, label):
    pos, neg = int(label.sum()), int((~label).sum())
    if pos == 0 or neg == 0:
        return float('nan')
    r = pd.Series(score).rank().to_numpy()
    return float((r[label].sum() - pos * (pos + 1) / 2) / (pos * neg))


def conserved_contacts(c1, c2, partner, ok):
    """Contacts of structure 1 whose aligned counterpart is a contact of structure 2."""
    i, j = np.nonzero(c1)
    keep = ok[i] & ok[j]
    i, j = i[keep], j[keep]
    k, l = partner[i], partner[j]
    hit = c2[np.minimum(k, l), np.maximum(k, l)]
    out = np.zeros_like(c1)
    out[i[hit], j[hit]] = True
    return out


def mass_test(P, mask, n_null, rng):
    """Share of pair-score mass on `mask` vs a diagonal-shift null.

    P, mask: (L, L), upper triangle only. Each null draw rolls every diagonal of
    P by an independent random shift, which keeps how much score sits at each
    sequence separation. Returns dict of observed share, area share,
    enrichment, null mean, z and one-sided p.
    """
    L = P.shape[0]
    total = P[np.triu_indices(L, 1)].sum()
    n_pairs = L * (L - 1) / 2
    if mask.sum() == 0 or total <= 0:
        return dict(share=np.nan, area=float(mask.sum() / n_pairs), enrichment=np.nan,
                    null_share=np.nan, z=np.nan, p=np.nan)
    obs = P[mask].sum() / total
    draws = np.zeros(n_null)
    for d in range(1, L):
        m = np.diagonal(mask, d).astype(float)
        if not m.any():
            continue
        v = np.diagonal(P, d)
        # corr[s] = sum_k m[k] * v[k - s]: the masked sum for every circular shift s
        corr = np.fft.irfft(np.fft.rfft(m) * np.conj(np.fft.rfft(v)), n=len(v))
        draws += corr[rng.integers(0, len(v), size=n_null)]
    draws /= total
    sd = draws.std(ddof=1)
    return dict(share=float(obs), area=float(mask.sum() / n_pairs),
                enrichment=float(obs / (mask.sum() / n_pairs)), null_share=float(draws.mean()),
                z=float((obs - draws.mean()) / sd) if sd > 0 else np.nan,
                p=float((1 + (draws >= obs).sum()) / (1 + n_null)))


def link_agreement(links, regions_q, regions_t, partner_ok):
    """Observed vs baseline agreement of cell links with the alignment (weighted by link value)."""
    Rt = np.stack([r.residues for r in regions_t])
    obs, base, w = [], [], []
    for a, b, val in links:
        sq = regions_q[a].residues & (partner_ok >= 0)
        n = sq.sum()
        if n == 0:
            continue
        hits = Rt[:, partner_ok[sq]].sum(1)
        obs.append(hits[b] / n)
        base.append(hits.mean() / n)
        w.append(max(val, 0.0))
    if not obs or sum(w) == 0:
        return np.nan, np.nan
    return float(np.average(obs, weights=w)), float(np.average(base, weights=w))


# ---------------------------------------------------------------------------
# Figures and PyMOL
# ---------------------------------------------------------------------------

def segments(mask):
    """1-based residue ranges of a boolean mask, as 'a-b+c-d' for PyMOL."""
    idx = np.nonzero(mask)[0] + 1
    if len(idx) == 0:
        return ''
    out, start, prev = [], idx[0], idx[0]
    for x in idx[1:]:
        if x != prev + 1:
            out.append(f'{start}-{prev}')
            start = x
        prev = x
    out.append(f'{start}-{prev}')
    return '+'.join(out)


def draw_proteogram(ax, native_img, native_map, title):
    gray = native_img.astype(float).mean(axis=2) / 255.0
    ax.imshow(gray, cmap='gray', vmin=0, vmax=1, interpolation='nearest')
    lo, hi = native_map.min(), native_map.max()
    m = (native_map - lo) / (hi - lo) if hi > lo else np.zeros_like(native_map)
    im = ax.imshow(m, cmap=HEAT, vmin=0, vmax=1, interpolation='bilinear')
    L = gray.shape[0]
    ax.plot([0, L - 1], [0, L - 1], color='white', lw=0.6, ls=(0, (2, 2)), alpha=0.7)
    ax.text(0.98, 0.04, 'lower: elec + hydro', transform=ax.transAxes, ha='right',
            fontsize=7, color='white')
    ax.text(0.02, 0.96, 'upper: VdW + distance', transform=ax.transAxes, ha='left', va='top',
            fontsize=7, color='white')
    ax.set_title(title, fontsize=9, loc='left')
    ax.set_xticks([])
    ax.set_yticks([])
    return im


def pair_figure(path, qid, tid, rel, d, nat_q, nat_t, map_q, map_t, links, reg_q, reg_t,
                score_q, score_t, aln, annot, ok_q, ok_t):
    fig = plt.figure(figsize=(10, 9.2))
    gs = fig.add_gridspec(3, 3, height_ratios=[4.2, 1.3, 1.3], width_ratios=[1, 1, 0.04],
                          hspace=0.45, wspace=0.12)
    ax_q, ax_t = fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1])
    sccs = lambda s: annot.loc[s, 'SCOPeFamily'] if annot is not None and s in annot.index else ''
    draw_proteogram(ax_q, nat_q, map_q, f'Query {qid}  {sccs(qid)}')
    im = draw_proteogram(ax_t, nat_t, map_t, f'Target {tid}  {sccs(tid)}')
    cb = fig.colorbar(im, cax=fig.add_subplot(gs[0, 2]))
    cb.set_label('Contribution to similarity (min to max)', fontsize=8)
    cb.ax.tick_params(labelsize=7)

    handles = []
    for n, (a, b, val) in enumerate(links):
        col = LINK_COLORS[n % len(LINK_COLORS)]
        centers = []
        for ax, reg in ((ax_q, reg_q[a]), (ax_t, reg_t[b])):
            (r0, r1), (c0, c1) = reg.rows, reg.cols
            ax.add_patch(Rectangle((c0 - 0.5, r0 - 0.5), c1 - c0, r1 - r0, fill=False,
                                   edgecolor=col, lw=1.8))
            ax.text(c1, r0, str(n + 1), color=col, fontsize=8, fontweight='bold',
                    ha='left', va='bottom')
            centers.append(((c0 + c1) / 2, (r0 + r1) / 2))
        fig.add_artist(ConnectionPatch(xyA=centers[0], coordsA='data', axesA=ax_q,
                                       xyB=centers[1], coordsB='data', axesB=ax_t,
                                       color=col, lw=1.0, alpha=0.8))
        handles.append(Patch(color=col, label=f'{n + 1}: {reg_q[a].label()}  ↔  '
                                              f'{reg_t[b].label()}  ({100 * val / d.cos:.2f}% '
                                              f'of the score, lift {d.lift[a, b]:.2f})'))

    for row, (sid, score, regs, cells, ok) in enumerate(
            ((qid, score_q, reg_q, [l[0] for l in links], ok_q),
             (tid, score_t, reg_t, [l[1] for l in links], ok_t)), start=1):
        ax = fig.add_subplot(gs[row, :2])
        x = np.arange(1, len(score) + 1)
        for n, c in enumerate(cells):
            m = regs[c].residues
            for seg in segments(m).split('+'):
                if seg:
                    s0, s1 = (int(v) for v in seg.split('-'))
                    ax.axvspan(s0 - 0.5, s1 + 0.5, color=LINK_COLORS[n % 4], alpha=0.15, lw=0)
        lo, hi = score.min(), score.max()
        ax.plot(x, (score - lo) / (hi - lo) if hi > lo else score * 0, color=INK, lw=1.0)
        if ok is not None:
            ax.plot(x[ok], np.full(ok.sum(), -0.08), '|', color=INK2, ms=4, mew=0.6)
        ax.set_xlim(0.5, len(score) + 0.5)
        ax.set_ylim(-0.15, 1.05)
        ax.set_ylabel('Score (min-max)', fontsize=8)
        ax.set_title(f'{"Query" if row == 1 else "Target"} {sid} per-residue score'
                     + ('  (ticks: aligned within 5 Å)' if ok is not None else ''),
                     fontsize=8, loc='left')
        ax.tick_params(labelsize=7)
        ax.yaxis.grid(True, color=GRID, lw=0.6)
        ax.set_axisbelow(True)
        for s in ('top', 'right'):
            ax.spines[s].set_visible(False)
    fig.axes[-1].set_xlabel('Residue (proteogram order)', fontsize=8)
    fig.legend(handles=handles, loc='lower center', ncol=1, fontsize=7, frameon=False,
               bbox_to_anchor=(0.5, -0.01))
    al = (f'  |  TM-score {aln.tm1:.2f} / {aln.tm2:.2f}, RMSD {aln.rmsd:.1f} Å'
          if aln is not None else '')
    fig.suptitle(f'{qid} → {tid}: cosine {d.cos:.3f}, shared {rel}{al}', fontsize=10,
                 x=0.02, ha='left')
    fig.subplots_adjust(left=0.07, right=0.94, top=0.93, bottom=0.08 + 0.022 * len(handles))
    fig.savefig(path, dpi=200)
    plt.close(fig)


CHANNEL_COLORS = ['#2a78d6', '#eb6834', '#1baf7a', '#eda100', '#e87ba4', '#008300']
CHANNEL_LABELS = ['VdW attractive', 'VdW repulsive', 'Cα distance',
                  'Elec. attractive', 'Elec. repulsive', 'Hydrophobicity']
DIVERGE = LinearSegmentedColormap.from_list('div', ['#2a78d6', '#f2f1ee', '#eb6834'])


def channel_figure(path, qid, tid, rel, d, ch, gq, gt):
    """Six channel maps per protein, channel totals, and each link's channel mix."""
    fig = plt.figure(figsize=(13, 8.6))
    gs = fig.add_gridspec(3, 7, height_ratios=[1, 1, 1.15], width_ratios=[1] * 6 + [0.06],
                          hspace=0.5, wspace=0.18)
    for row, (sid, cells, geom, tot, blank, side) in enumerate(
            ((qid, ch['q_cells'], gq, ch['q_tot'], ch['q_blank'], 'Query'),
             (tid, ch['t_cells'], gt, ch['t_tot'], ch['t_blank'], 'Target'))):
        natives = [sm.cells_to_native(c, geom) for c in cells]
        v = max(np.abs(n).max() for n in natives) or 1.0
        for k, nat in enumerate(natives):
            ax = fig.add_subplot(gs[row, k])
            im = ax.imshow(nat, cmap=DIVERGE, vmin=-v, vmax=v, interpolation='bilinear')
            L = nat.shape[0]
            ax.plot([0, L - 1], [0, L - 1], color=INK2, lw=0.5, ls=(0, (2, 2)))
            share = 100 * tot[k] / tot.sum() if tot.sum() else np.nan
            ax.set_title(f'{CHANNEL_LABELS[k]}\n{share:.0f}% ({"upper" if k < 3 else "lower"})',
                         fontsize=8)
            ax.set_xticks([])
            ax.set_yticks([])
            if k == 0:
                ax.set_ylabel(f'{side} {sid}', fontsize=8)
        cb = fig.colorbar(im, cax=fig.add_subplot(gs[row, 6]))
        cb.ax.tick_params(labelsize=6)
        cb.set_label('Shapley value\nper pixel', fontsize=7)

    # channel totals, query vs target side
    ax = fig.add_subplot(gs[2, :3])
    x = np.arange(6)
    for off, tot, alpha, lab in ((-0.2, ch['q_tot'], 1.0, 'query channels removed'),
                                 (0.2, ch['t_tot'], 0.45, 'target channels removed')):
        ax.bar(x + off, tot, 0.38, color=CHANNEL_COLORS, alpha=alpha, label=lab, linewidth=0)
    ax.axhline(0, color=INK2, lw=0.6)
    ax.set_xticks(x)
    ax.set_xticklabels(CHANNEL_LABELS, rotation=30, ha='right', fontsize=7)
    ax.set_ylabel('Shapley value (cosine)', fontsize=8)
    ax.set_title(f'Channel totals: sum = cosine {d.cos:.3f} − blank '
                 f'({ch["q_blank"]:.3f} query / {ch["t_blank"]:.3f} target side)', fontsize=8, loc='left')
    ax.legend(fontsize=7, frameon=False)
    ax.yaxis.grid(True, color=GRID, lw=0.6)
    ax.set_axisbelow(True)
    for s in ('top', 'right'):
        ax.spines[s].set_visible(False)
    ax.tick_params(labelsize=7)

    # each link's channel mix (query side)
    ax = fig.add_subplot(gs[2, 4:])
    for i, phis in enumerate(ch['links']):
        tot = phis.sum()
        left_pos = left_neg = 0.0
        for k in range(6):
            share = phis[k] / tot if tot else 0.0
            if share >= 0:
                ax.barh(i, share, left=left_pos, color=CHANNEL_COLORS[k], height=0.6, linewidth=0)
                left_pos += share
            else:
                ax.barh(i, share, left=left_neg, color=CHANNEL_COLORS[k], height=0.6, linewidth=0)
                left_neg += share
    ax.set_yticks(range(len(ch['links'])))
    ax.set_yticklabels([f'link {i + 1}' for i in range(len(ch['links']))], fontsize=7)
    ax.invert_yaxis()
    ax.axvline(0, color=INK2, lw=0.6)
    ax.set_xlabel('Share of the link\'s contribution', fontsize=8)
    ax.set_title('Channel mix of the top links (query side)', fontsize=8, loc='left')
    ax.tick_params(labelsize=7)
    for s in ('top', 'right'):
        ax.spines[s].set_visible(False)
    ax.legend(handles=[Patch(color=c, label=l) for c, l in zip(CHANNEL_COLORS, CHANNEL_LABELS)],
              fontsize=6.5, frameon=False, loc='upper left', bbox_to_anchor=(1.0, 1.0))
    fig.suptitle(f'{qid} → {tid} ({rel}): similarity split by channel', fontsize=10,
                 x=0.02, ha='left')
    fig.subplots_adjust(left=0.04, right=0.88, top=0.9, bottom=0.12)
    fig.savefig(path, dpi=200)
    plt.close(fig)


def write_pymol(pair_dir, q_pdb, t_pdb, links, reg_q, reg_t, pymol_exe):
    """Write view_scores.pml / view_links.pml and render PNGs if PyMOL is available."""
    cols = ''.join(f'set_color s{k}, [{int(h[1:3], 16)}, {int(h[3:5], 16)}, {int(h[5:7], 16)}]\n'
                   for k, h in enumerate(SCORE_RAMP))
    head = (f'load {os.path.basename(q_pdb)}, query\nload {os.path.basename(t_pdb)}, target\n'
            'hide everything\nshow cartoon\nbg_color white\nset grid_mode, 1\n'
            'set ray_opaque_background, 0\n')
    scores = head + cols + 'color s0, all\n' + ''.join(
        f'color s{k}, b > {10 * k - 0.001:.3f}\n' for k in range(1, 10)) + 'orient\n'
    lk = head + 'color grey80, all\n'
    # Links share residues; colour the weakest first so link 1 ends up on top.
    for n, (a, b, _) in reversed(list(enumerate(links))):
        h = LINK_COLORS[n % 4]
        lk += f'set_color l{n + 1}, [{int(h[1:3], 16)}, {int(h[3:5], 16)}, {int(h[5:7], 16)}]\n'
        for obj, reg in (('query', reg_q[a]), ('target', reg_t[b])):
            sel = segments(reg.residues)
            if sel:
                lk += f'color l{n + 1}, {obj} and resi {sel}\n'
    lk += 'orient\n'
    for name, text in (('view_scores.pml', scores), ('view_links.pml', lk)):
        with open(os.path.join(pair_dir, name), 'w') as f:
            f.write(text)
    if not pymol_exe:
        return
    for name, png in (('view_scores.pml', 'pymol_scores.png'), ('view_links.pml', 'pymol_links.png')):
        render = os.path.join(pair_dir, '_render.pml')
        with open(render, 'w') as f:
            f.write(f'@{name}\npng {png}, width=1800, height=900, dpi=300, ray=1\nquit\n')
        try:
            subprocess.run([pymol_exe, '-cq', '_render.pml'], cwd=pair_dir, capture_output=True,
                           timeout=300)
        except Exception as e:
            print(f'  PyMOL render failed: {e}')
        os.remove(render)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    here = os.path.dirname(os.path.abspath(__file__))
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--model_file', required=True, help='ResNet18 classifier checkpoint used for search.')
    p.add_argument('--config', default=os.path.join(here, 'config.yml'))
    p.add_argument('--proteograms_dir', help='Proteogram images (default: config all_proteograms_dir).')
    p.add_argument('--structures_dir', help='SCOPe pdbstyle tree (default: config scope_structures_dir).')
    p.add_argument('--production_dir', help='CG production structures, the fallback when an .ent does '
                   'not match the image (default: <proteograms_dir>/production_structures).')
    p.add_argument('--annot_file', help='Annotation TSV with SCOPe columns (default: config annot_file).')
    p.add_argument('--pair', nargs=2, action='append', metavar=('QUERY', 'TARGET'))
    p.add_argument('--pairs_file')
    p.add_argument('--similarity_tsv')
    p.add_argument('--queries', help='Restrict --similarity_tsv to these query IDs (one per line).')
    p.add_argument('--top_k', type=int, default=1, help='Hits per query from --similarity_tsv.')
    p.add_argument('--relations', help='Keep only pairs sharing this SCOPe level, e.g. "fold,class" '
                   '(family, superfamily, fold, class, none).')
    p.add_argument('--max_pairs', type=int)
    p.add_argument('--center_list', help='IDs whose mean embedding centers the maps (e.g. the eval corpus).')
    p.add_argument('--center_file', help='Precomputed mean embedding (.pt), e.g. corpus_mean_embedding.pt '
                   'from select_explain_pairs.py; used instead of --center_list.')
    p.add_argument('--usalign', default='~/Documents/bin/USalign/USalign')
    p.add_argument('--no_usalign', action='store_true')
    p.add_argument('--pymol', default=None, help='PyMOL executable (default: found on PATH).')
    p.add_argument('--no_pymol', action='store_true')
    p.add_argument('--n_links', type=int, default=4, help='Links drawn per pair (distinct cells).')
    p.add_argument('--stat_links', type=int, default=10, help='Links scored in the agreement check.')
    p.add_argument('--contact_cutoff', type=float, default=8.0, help='C-alpha contact distance (A).')
    p.add_argument('--min_seq_sep', type=int, default=3)
    p.add_argument('--n_null', type=int, default=1000)
    p.add_argument('--faithfulness', action='store_true')
    p.add_argument('--channels', action='store_true',
                   help='Split every map over the six proteogram channels with exact Shapley values '
                        '(64 extra forward passes per protein per pair).')
    p.add_argument('--channel_fill', choices=['black', 'gray'], default='black',
                   help='Value a removed channel is set to (black = no interaction).')
    p.add_argument('--n_random', type=int, default=5, help='Random orders in the faithfulness test.')
    p.add_argument('--mask_fill', choices=['black', 'gray'], default='black')
    p.add_argument('--max_figures', type=int, default=25)
    p.add_argument('--out_dir')
    p.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    p.add_argument('--seed', type=int, default=0)
    args = p.parse_args()

    cfg = read_yaml(args.config) if os.path.exists(args.config) else {}
    rel_to = lambda x: os.path.normpath(os.path.join(here, x)) if x and not os.path.isabs(x) else x
    prot_dir = args.proteograms_dir or rel_to(cfg.get('all_proteograms_dir'))
    struct_dir = args.structures_dir or rel_to(cfg.get('scope_structures_dir'))
    prod_dir = args.production_dir or (os.path.join(prot_dir, 'production_structures') if prot_dir else None)
    annot_path = args.annot_file or rel_to(cfg.get('annot_file'))
    if not prot_dir or not os.path.isdir(prot_dir):
        sys.exit(f'Proteogram directory not found: {prot_dir}')

    stamp = datetime.datetime.now().strftime('%Y%m%d-%H%M%S')
    out_dir = args.out_dir or os.path.join(os.path.dirname(os.path.normpath(prot_dir)), 'evaluation',
                                           'explain_runs', f'explain_{stamp}')
    os.makedirs(os.path.join(out_dir, 'pairs'), exist_ok=True)
    sys.stdout = Tee(os.path.join(out_dir, 'report.txt'))
    started = datetime.datetime.now()
    rng = np.random.default_rng(args.seed)

    annot = None
    if annot_path and os.path.exists(annot_path):
        annot = pd.read_csv(annot_path, sep='\t', usecols=['SCOPeID', 'SCOPeClass', 'SCOPeFold',
                                                           'SCOPeSuperfamily', 'SCOPeFamily'],
                            dtype=str).drop_duplicates('SCOPeID').set_index('SCOPeID')
    pairs = build_pairs(args, annot)
    if not pairs:
        sys.exit('No pairs: give --pair, --pairs_file or --similarity_tsv.')

    usalign = None
    if not args.no_usalign:
        # A path, a command on PATH, or (if the default path is missing) USalign on PATH.
        cand = os.path.expanduser(args.usalign)
        usalign = (cand if os.access(cand, os.X_OK) else shutil.which(args.usalign)
                   or shutil.which('USalign') or cand)
    if usalign and not os.access(usalign, os.X_OK):
        print(f'USalign not found at {usalign}; structural checks skipped.')
        usalign = None
    pymol = None if args.no_pymol else (args.pymol or shutil.which('pymol'))

    model, meta = sm.load_resnet18_checkpoint(args.model_file, args.device)
    grid, resize = sm.search_grid(meta)
    print(f'Model {os.path.basename(args.model_file)}: grid {grid}, '
          f'{"resize" if resize else "pad"} mode, device {args.device}')
    print(f'{len(pairs)} pairs | USalign: {usalign or "off"} | PyMOL: {pymol or "off"} | '
          f'out: {out_dir}\n')

    image_cache = {}

    def image_path(sid):
        for ext in ('.png', '.jpg', '.jpeg'):
            f = os.path.join(prot_dir, sid + ext)
            if os.path.exists(f):
                return f
        return None

    def features(sid):
        if sid not in image_cache:
            f = image_path(sid)
            if f is None:
                raise FileNotFoundError(f'No proteogram image for {sid} in {prot_dir}')
            x, native, geom = sm.load_proteogram(f, grid, resize, meta.get('drop_channels'))
            x = x.to(args.device)
            with torch.no_grad():
                fm = sm.layer4_features(model, x)[0]
            if len(image_cache) >= 128:          # ~3 MB per image; keep memory bounded
                image_cache.pop(next(iter(image_cache)))
            image_cache[sid] = (x, native, geom, fm)
        return image_cache[sid]

    center = None
    if args.center_file:
        center = torch.load(args.center_file, map_location=args.device).reshape(-1)
        print(f'Centering on {os.path.basename(args.center_file)}\n')
    elif args.center_list:
        ids = [s for s in read_ids(args.center_list) if image_path(s)]
        embs = []
        for k in range(0, len(ids), 32):
            xs = torch.cat([sm.load_proteogram(image_path(s), grid, resize, meta.get('drop_channels'))[0] for s in ids[k:k + 32]])
            embs.append(sm.embed(model, xs.to(args.device)))
        center = torch.cat(embs).mean(0)
        print(f'Centering on the mean embedding of {len(ids)} proteograms from '
              f'{os.path.basename(args.center_list)}\n')

    finder = StructureFinder(struct_dir, prod_dir)
    rows, done = [], {}
    for n_pair, (qid, tid, pmeta) in enumerate(pairs, 1):
        rel = relation(qid, tid, annot)
        tag = f', {pmeta["stratum"]}' if pmeta.get('stratum') else ''
        print(f'[{n_pair}/{len(pairs)}] {qid} -> {tid} ({rel}{tag})')
        if (qid, tid) in done:                     # same pair listed in another stratum
            rows.append({**pmeta, **done[(qid, tid)]})
            print('  already computed')
            continue
        try:
            xq, nat_q, gq, fq = features(qid)
            xt, nat_t, gt, ft = features(tid)
        except FileNotFoundError as e:
            print(f'  {e}; skipped')
            continue
        d = sm.decompose(fq, ft, center)
        check = float(torch.nn.functional.cosine_similarity(fq.double().mean((1, 2)),
                                                            ft.double().mean((1, 2)), dim=0))
        pair_dir = os.path.join(out_dir, 'pairs', f'{qid}__{tid}')
        os.makedirs(pair_dir, exist_ok=True)

        map_q, map_t = sm.cells_to_native(d.query_cells, gq), sm.cells_to_native(d.target_cells, gt)
        res_q, res_t = sm.native_to_residues(map_q), sm.native_to_residues(map_t)
        reg_q, reg_t = sm.cell_regions(d.query_cells.shape, gq), sm.cell_regions(d.target_cells.shape, gt)
        links = sm.top_links(d.pair, args.n_links)
        row = dict(query=qid, target=tid, relation=rel, cosine=d.cos,
                   decomposition_error=abs(d.pair.sum() - check), L_query=gq.L, L_target=gt.L,
                   top_link_share=links[0][2] / d.cos if links else np.nan,
                   top10_cell_share_query=np.sort(d.query_cells.ravel())[::-1][:10].sum() / d.cos)
        extra = {}
        if center is not None:
            cq = sm.native_to_residues(sm.cells_to_native(d.query_centered, gq))
            ct = sm.native_to_residues(sm.cells_to_native(d.target_centered, gt))
            extra = dict(query_centered=d.query_centered, target_centered=d.target_centered)
            row['centered_total'] = d.centered_total

        # ---- structure + alignment
        pres_q, kind_q = finder.residues(qid, gq.L)
        pres_t, kind_t = finder.residues(tid, gt.L)
        aln = None
        ok_q = ok_t = None
        q_pdb, t_pdb = os.path.join(pair_dir, 'query.pdb'), os.path.join(pair_dir, 'target.pdb')
        if pres_q is not None and pres_t is not None:
            row.update(structure_query=kind_q, structure_target=kind_t)
            ua.write_indexed_pdb(pres_q, q_pdb)
            ua.write_indexed_pdb(pres_t, t_pdb)
            if usalign:
                try:
                    aln = ua.run_usalign(usalign, q_pdb, t_pdb, pair_dir, gq.L, gt.L)
                except Exception as e:
                    print(f'  USalign: {e}')
        if aln is not None:
            ok_q = (aln.partner >= 0) & aln.close
            partner_q = np.where(ok_q, aln.partner, -1)
            partner_t = -np.ones(gt.L, dtype=int)
            partner_t[partner_q[ok_q]] = np.nonzero(ok_q)[0]
            ok_t = partner_t >= 0
            cq_map = ua.contact_map(pres_q.ca, args.contact_cutoff, args.min_seq_sep)
            ct_map = ua.contact_map(pres_t.ca, args.contact_cutoff, args.min_seq_sep)
            cons_q = conserved_contacts(cq_map, ct_map, partner_q, ok_q)
            cons_t = conserved_contacts(ct_map, cq_map, partner_t, ok_t)
            row.update(tm_query=aln.tm1, tm_target=aln.tm2, rmsd=aln.rmsd,
                       aligned_length=aln.aligned_length, aligned_close_query=ok_q.mean(),
                       auroc_query=auroc(res_q.score, ok_q), auroc_target=auroc(res_t.score, ok_t))
            for side, P, mask in (('query', res_q.pairs, cons_q), ('target', res_t.pairs, cons_t)):
                mt = mass_test(P, mask, args.n_null, rng)
                row.update({f'contact_{k}_{side}': v for k, v in mt.items()})
            if center is not None:
                row['auroc_query_centered'] = auroc(cq.score, ok_q)
                row['auroc_target_centered'] = auroc(ct.score, ok_t)
            for how, score in (('contribution', d.pair), ('lift', d.lift)):
                o, b = link_agreement(sm.top_links(score, args.stat_links), reg_q, reg_t, partner_q)
                row[f'link_agree_{how}'], row[f'link_base_{how}'] = o, b
        # Scores (0-100, per protein) in the B-factor column; query superposed when aligned.
        if pres_q is not None:
            ua.write_indexed_pdb(pres_q, q_pdb, 100 * res_q.score / res_q.score.max(),
                                 rotation=aln.rotation if aln is not None else None)
        if pres_t is not None:
            ua.write_indexed_pdb(pres_t, t_pdb, 100 * res_t.score / res_t.score.max())

        # ---- faithfulness
        if args.faithfulness:
            fr = np.linspace(0, 1, 11)
            t_emb = ft.mean((1, 2))[None]
            orders = {'exact': np.argsort(d.query_cells.ravel())[::-1]}
            if center is not None:
                orders['centered'] = np.argsort(d.query_centered.ravel())[::-1]
            for name, order in orders.items():
                de, ins = sm.deletion_insertion(model, xq, t_emb, order, d.query_cells.shape, fr,
                                                args.mask_fill)
                row[f'deletion_auc_{name}'] = float(trapz(de, fr))
                row[f'insertion_auc_{name}'] = float(trapz(ins, fr))
            rd, ri = [], []
            for _ in range(args.n_random):
                de, ins = sm.deletion_insertion(model, xq, t_emb, rng.permutation(d.query_cells.size),
                                                d.query_cells.shape, fr, args.mask_fill)
                rd.append(trapz(de, fr))
                ri.append(trapz(ins, fr))
            row['deletion_auc_random'], row['insertion_auc_random'] = float(np.mean(rd)), float(np.mean(ri))

        # ---- channels
        ch = None
        if args.channels:
            cs_q = sm.channel_split(model, nat_q, ft, 'query', grid, resize, args.channel_fill)
            cs_t = sm.channel_split(model, nat_t, fq, 'target', grid, resize, args.channel_fill)
            row['channel_split_error'] = float(max(abs(cs_q.full - d.pair).max(), abs(cs_t.full - d.pair).max()))
            row['cos_blank_query'], row['cos_blank_target'] = cs_q.values[0], cs_t.values[0]
            for side, cs in (('query', cs_q), ('target', cs_t)):
                tot = cs.totals
                for k, (name, _, _) in enumerate(sm.CHANNELS):
                    row[f'ch_{name}_{side}'] = float(tot[k])
                    row[f'ch_share_{name}_{side}'] = float(tot[k] / tot.sum()) if tot.sum() else np.nan
            shape_q, shape_t = d.query_cells.shape, d.target_cells.shape
            ch = dict(
                q_cells=np.stack([cs_q.cells(k).reshape(shape_q) for k in range(6)]),
                t_cells=np.stack([cs_t.cells(k).reshape(shape_t) for k in range(6)]),
                q_res=np.stack([sm.native_to_residues(sm.cells_to_native(cs_q.cells(k).reshape(shape_q), gq)).score
                                for k in range(6)]),
                t_res=np.stack([sm.native_to_residues(sm.cells_to_native(cs_t.cells(k).reshape(shape_t), gt)).score
                                for k in range(6)]),
                links=np.array([[cs_q.phi[k, a, b] for k in range(6)] for a, b, _ in links]),
                q_tot=cs_q.totals, t_tot=cs_t.totals, q_blank=cs_q.values[0], t_blank=cs_t.values[0])
            extra.update(channel_names=np.array([c[0] for c in sm.CHANNELS]),
                         channel_query_cells=ch['q_cells'], channel_target_cells=ch['t_cells'],
                         channel_query_totals=cs_q.totals, channel_target_totals=cs_t.totals)

        # ---- per-pair outputs
        np.savez_compressed(os.path.join(pair_dir, 'maps.npz'), pair=d.pair, query_cells=d.query_cells,
                            target_cells=d.target_cells, query_native=map_q, target_native=map_t,
                            cosine=d.cos, **extra)
        for side, sid, res, pres, ok in (('query', qid, res_q, pres_q, ok_q),
                                         ('target', tid, res_t, pres_t, ok_t)):
            df = pd.DataFrame({'row': np.arange(1, len(res.score) + 1), 'score': res.score,
                               'score_upper': res.score_upper, 'score_lower': res.score_lower,
                               'share_of_cosine': res.score / d.cos})
            if ch is not None:
                for k, (name, _, _) in enumerate(sm.CHANNELS):
                    df[f'ch_{name}'] = ch['q_res' if side == 'query' else 't_res'][k]
            if pres is not None:
                df.insert(1, 'resid', pres.resid_labels())
                df.insert(2, 'aa', list(pres.sequence))
            if ok is not None:
                df['aligned_within_5A'] = ok
                if side == 'query':
                    df['partner_row'] = np.where(ok_q, aln.partner + 1, 0)
            df.to_csv(os.path.join(pair_dir, f'residues_{side}.tsv'), sep='\t', index=False,
                      float_format='%.6g')
        link_rows = [dict(link=n + 1, query_cell=a, target_cell=b, contribution=v,
                          share_of_cosine=v / d.cos, lift=d.lift[a, b],
                          query_region=reg_q[a].label(), target_region=reg_t[b].label(),
                          query_resi=segments(reg_q[a].residues), target_resi=segments(reg_t[b].residues))
                     for n, (a, b, v) in enumerate(links)]
        if ch is not None:
            for r, phis in zip(link_rows, ch['links']):
                tot = phis.sum()
                r.update({f'ch_share_{name}': (phis[k] / tot if tot else np.nan)
                          for k, (name, _, _) in enumerate(sm.CHANNELS)})
        pd.DataFrame(link_rows).to_csv(
            os.path.join(pair_dir, 'links.tsv'), sep='\t', index=False, float_format='%.6g')
        if n_pair <= args.max_figures:
            pair_figure(os.path.join(pair_dir, 'figure.png'), qid, tid, rel, d, nat_q, nat_t,
                        map_q, map_t, links, reg_q, reg_t, res_q.score, res_t.score, aln, annot,
                        ok_q, ok_t)
            if pres_q is not None and pres_t is not None:
                write_pymol(pair_dir, q_pdb, t_pdb, links, reg_q, reg_t, pymol)
            if ch is not None:
                channel_figure(os.path.join(pair_dir, 'channels.png'), qid, tid, rel, d, ch, gq, gt)
        msg = f'  cos {d.cos:.4f} (split error {row["decomposition_error"]:.1e})'
        if aln is not None:
            msg += (f' | TM {aln.tm1:.2f}/{aln.tm2:.2f} | residue AUROC {row["auroc_query"]:.2f}/'
                    f'{row["auroc_target"]:.2f} | conserved-contact enrichment '
                    f'{row["contact_enrichment_query"]:.2f} (p {row["contact_p_query"]:.3f})')
        if args.faithfulness:
            msg += (f' | deletion AUC {row["deletion_auc_exact"]:.3f} vs random '
                    f'{row["deletion_auc_random"]:.3f}')
        print(msg)
        done[(qid, tid)] = row
        rows.append({**pmeta, **row})

    if not rows:
        sys.exit('No pairs could be processed.')
    summary = pd.DataFrame(rows)
    summary.to_csv(os.path.join(out_dir, 'summary.tsv'), sep='\t', index=False, float_format='%.6g')

    # ---- overview
    print(f'\n{len(summary)} pairs. Max split error {summary.decomposition_error.max():.1e} '
          '(cell-pair contributions vs the search cosine).')
    cols = [c for c in ['cosine'] + [f'ch_share_{n}_query' for n, _, _ in sm.CHANNELS] + ['auroc_query', 'auroc_target', 'contact_enrichment_query',
                        'contact_enrichment_target', 'link_agree_contribution', 'link_base_contribution',
                        'link_agree_lift', 'link_base_lift', 'deletion_auc_exact', 'deletion_auc_random',
                        'insertion_auc_exact', 'insertion_auc_random'] if c in summary]
    if 'tm_query' in summary:
        summary['tm_bin'] = np.where(summary[['tm_query', 'tm_target']].max(axis=1) >= 0.5,
                                     'TM>=0.5', 'TM<0.5')
        summary.loc[summary.tm_query.isna(), 'tm_bin'] = 'no alignment'
        p_q = summary.contact_p_query
        summary['contact_sig_query'] = (p_q < 0.05).where(p_q.notna()).astype(float)
        cols.append('contact_sig_query')
        groups = ['relation', 'tm_bin']
    else:
        groups = ['relation']
    if 'stratum' in summary:
        groups.insert(0, 'stratum')
    if 'aligners_miss_fold' in summary and 'stratum' in summary:
        fr = summary[summary.stratum == 'fold_remote'].copy()
        if len(fr):
            fr['aligners'] = np.where(fr.aligners_miss_fold.astype(str) == 'True',
                                      'all aligners miss', 'some aligner hits')
            col = pd.Series(None, index=summary.index, dtype=object)
            col.loc[fr.index] = fr.aligners.astype(object)
            summary['fold_remote_by_aligners'] = col
            groups.append('fold_remote_by_aligners')
    with pd.option_context('display.width', 200, 'display.max_columns', 30):
        for g in groups:
            t = summary.groupby(g)[cols].mean().round(3)
            t.insert(0, 'n', summary.groupby(g).size())
            print(f'\nMeans by {g}:\n{t.T.to_string()}')
    print('\nReading: AUROC > 0.5 and enrichment > 1 mean high-scoring residues / pairs sit in the '
          'aligned core more than chance; link_agree > link_base means links point at aligned '
          'regions; deletion AUC below random means the top cells matter to the score.')

    with open(os.path.join(out_dir, 'run_info.json'), 'w') as f:
        json.dump({'started': started.isoformat(timespec='seconds'),
                   'finished': datetime.datetime.now().isoformat(timespec='seconds'),
                   'command': sys.argv, 'args': vars(args), 'git_commit': git_commit(),
                   'model_meta': {k: v for k, v in meta.items() if k != 'val_summary'},
                   'grid': grid, 'resize': resize, 'proteograms_dir': prot_dir,
                   'structures_dir': struct_dir, 'usalign': usalign, 'pymol': pymol,
                   'n_pairs': len(summary)}, f, indent=2, default=str)
    print(f'\nWrote {out_dir}')


if __name__ == '__main__':
    main()
