"""Select query->target pairs to explain from proteogram similarity results.

Labels come from SCOPe 2.08 dir.cla (superfamily / fold), because USalign
TM-scores are not available for this test set.  Writes one 2-column TSV per
category (input for --pairs_file) plus selected_pairs.tsv with scores/labels,
and prints retrieval sanity metrics.

Categories (one pair per query, N per category):
  same_sf_high_sim    top-1 hit is in the query's superfamily, highest cosine  (true positives)
  diff_fold_high_sim  top-1 hit is in a different fold, highest cosine        (confident false positives)
  same_sf_low_sim     least-similar same-superfamily partner, lowest cosine   (missed homologs)
  diff_fold_low_sim   least-similar different-fold partner, lowest cosine     (true negatives / control)

usage: python make_pairs.py results.tsv dir.cla.txt out_dir [N]
"""
import os
import sys

import pandas as pd

res_tsv, cla_file, out_dir = sys.argv[1:4]
N = int(sys.argv[4]) if len(sys.argv) > 4 else 8

sccs = {}
with open(cla_file) as fh:
    for line in fh:
        if line.startswith('#') or not line.strip():
            continue
        f = line.split()
        sccs[f[0]] = f[3]


def sf(s):
    return '.'.join(sccs[s].split('.')[:3])


def fold(s):
    return '.'.join(sccs[s].split('.')[:2])


stem = lambda p: os.path.splitext(os.path.basename(str(p)))[0]

df = pd.read_csv(res_tsv, sep='\t')
hits = {}
for _, row in df.iterrows():
    q = stem(row.iloc[0])
    hs = []
    for v in row.iloc[1:]:
        if isinstance(v, str) and ',' in v:
            name, score = v.rsplit(',', 1)
            t = stem(name)
            if t != q and t in sccs:
                hs.append((t, float(score)))
    if q in sccs and hs:
        hits[q] = hs  # sorted by descending cosine

# Retrieval sanity check (does the model behave on these proteograms?)
n = len(hits)
top1_sf = sum(sf(h[0][0]) == sf(q) for q, h in hits.items()) / n
top1_fold = sum(fold(h[0][0]) == fold(q) for q, h in hits.items()) / n
has_sf = [q for q, h in hits.items() if any(sf(t) == sf(q) for t, _ in h)]
top5_sf = sum(any(sf(t) == sf(q) for t, _ in hits[q][:5]) for q in has_sf) / len(has_sf)
print(f'queries={n}  top1 superfamily={top1_sf:.3f}  top1 fold={top1_fold:.3f}  '
      f'top5 superfamily (queries with >=1 sf partner, n={len(has_sf)})={top5_sf:.3f}')

cands = {c: [] for c in ['same_sf_high_sim', 'diff_fold_high_sim',
                         'same_sf_low_sim', 'diff_fold_low_sim']}
for q, h in hits.items():
    t, s = h[0]
    if sf(t) == sf(q):
        cands['same_sf_high_sim'].append((q, t, s))
    elif fold(t) != fold(q):
        cands['diff_fold_high_sim'].append((q, t, s))
    same = [(t, s) for t, s in h if sf(t) == sf(q)]
    if same:
        cands['same_sf_low_sim'].append((q, *same[-1]))
    diff = [(t, s) for t, s in h if fold(t) != fold(q)]
    if diff:
        cands['diff_fold_low_sim'].append((q, *diff[-1]))

rows = []
for cat, lst in cands.items():
    lst.sort(key=lambda x: x[2], reverse=cat.endswith('high_sim'))
    chosen, used = [], set()
    for q, t, s in lst:  # each protein at most once per category
        if q in used or t in used:
            continue
        chosen.append((q, t, s))
        used |= {q, t}
        if len(chosen) == N:
            break
    pd.DataFrame([(q, t) for q, t, _ in chosen]).to_csv(
        os.path.join(out_dir, f'pairs_{cat}.tsv'), sep='\t', header=False, index=False)
    for q, t, s in chosen:
        rows.append(dict(category=cat, query=q, target=t, cosine=round(s, 4),
                         query_sccs=sccs[q], target_sccs=sccs[t]))
    print(f'{cat:20s} candidates={len(lst):5d} selected={len(chosen)}')

pd.DataFrame(rows).to_csv(os.path.join(out_dir, 'selected_pairs.tsv'), sep='\t', index=False)
