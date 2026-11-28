#!/usr/bin/env python
"""Choose query -> target pairs for explain_similarity_maps.py from an eval run.

Re-embeds the eval corpus with the model that run used, rebuilds each query's
full ranked list (the eval's per_query_metrics.tsv keeps only the top 10, and
a query's first remote hit can rank lower), checks the top 10 against the
eval, and samples pairs in strata that match the remote-homology metrics of
evaluate_methods_v2.py:

  family_top1         top-1 hit, same family (reference: near-duplicate matches)
  superfamily_remote  first hit outside the query's family, same superfamily
                      (rP@1 superfamily = 1)
  fold_remote         first hit outside the query's superfamily, same fold
                      (rP@1 fold = 1). Every query where all three aligners
                      miss (their rP@1 fold = 0) is included first, flagged
                      aligners_miss_fold, then random others up to --n.
  fold_remote_miss    first hit outside the query's superfamily, different fold
                      (rP@1 fold = 0): the confident errors, as a control
  novel_superfamily   novel-superfamily queries (superfamily absent from
                      training) -> their top-1 hit

Outputs in --out_dir: pairs.tsv (query, target, stratum, relation, ...; the
aligner-miss cases come first so they get figures), corpus_ids.lst and
corpus_mean_embedding.pt (for --center_file), and ranked_check.txt.

Example (from scripts/v2):
  python select_explain_pairs.py --model_file MODEL.pt \\
      --eval_dir ../data/.../evaluation/eval_runs/eval_20261125-225516_superfamily_k10 \\
      --out_dir ../data/.../evaluation/explain_runs/pairs_run3
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd
import torch

from proteogram.common import read_yaml
from proteogram.v2 import similarity_maps as sm

ALIGNERS = ('GTalign', 'USalign', 'Foldseek')


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--model_file', required=True)
    p.add_argument('--eval_dir', required=True, help='evaluate_methods_v2.py run folder for this model.')
    p.add_argument('--proteograms_dir', help='Default: config all_proteograms_dir.')
    p.add_argument('--config', default=os.path.join(here, 'config.yml'))
    p.add_argument('--query_types', help='Query-type TSV (SCOPeID, query_type); default: the one in '
                   "the eval's run_info.json.")
    p.add_argument('--n', type=int, default=100, help='Pairs per stratum (family_top1 gets n/2).')
    p.add_argument('--out_dir', required=True)
    p.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    p.add_argument('--batch_size', type=int, default=32)
    p.add_argument('--seed', type=int, default=0)
    args = p.parse_args()

    cfg = read_yaml(args.config) if os.path.exists(args.config) else {}
    rel_to = lambda x: os.path.normpath(os.path.join(here, x)) if x and not os.path.isabs(x) else x
    prot_dir = args.proteograms_dir or rel_to(cfg.get('all_proteograms_dir'))
    os.makedirs(args.out_dir, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    labels = pd.read_csv(os.path.join(args.eval_dir, 'scope_eval_set_labels.tsv'), sep='\t', dtype=str)
    labels = labels.rename(columns={'pdb_id_chain': 'sid'}).set_index('sid')
    pqm = pd.read_csv(os.path.join(args.eval_dir, 'per_query_metrics.tsv'), sep='\t',
                      dtype={'query_id': str, 'top_hits': str})
    qtypes = {}
    qt_path = args.query_types
    if not qt_path:
        import json
        info = json.load(open(os.path.join(args.eval_dir, 'run_info.json')))
        qt = (info.get('args') or {}).get('query_types')
        qt_path = rel_to(qt) if qt else None
    if qt_path and os.path.exists(qt_path):
        qtypes = pd.read_csv(qt_path, sep='\t', dtype=str).set_index('SCOPeID')['query_type'].to_dict()
    else:
        print('No query-type file found; novel_superfamily stratum skipped.')

    # ---- embed the corpus with the eval's model
    model, meta = sm.load_resnet18_checkpoint(args.model_file, args.device)
    grid, resize = sm.search_grid(meta)
    ids = [s for s in labels.index if os.path.exists(os.path.join(prot_dir, s + '.png'))]
    if len(ids) < len(labels):
        print(f'{len(labels) - len(ids)} corpus proteograms missing in {prot_dir}')
    embs = []
    for k in range(0, len(ids), args.batch_size):
        xs = torch.cat([sm.load_proteogram(os.path.join(prot_dir, s + '.png'), grid, resize,
                                           meta.get('drop_channels'))[0]
                        for s in ids[k:k + args.batch_size]])
        embs.append(sm.embed(model, xs.to(args.device)).cpu())
        print(f'\rEmbedded {min(k + args.batch_size, len(ids))}/{len(ids)}', end='', file=sys.stderr)
    print(file=sys.stderr)
    E = torch.cat(embs)
    torch.save(E.mean(0), os.path.join(args.out_dir, 'corpus_mean_embedding.pt'))
    with open(os.path.join(args.out_dir, 'corpus_ids.lst'), 'w') as f:
        f.write('\n'.join(ids) + '\n')
    En = torch.nn.functional.normalize(E.double(), dim=1)
    S = (En @ En.T).numpy()
    np.fill_diagonal(S, -np.inf)
    order = np.argsort(-S, axis=1, kind='stable')
    lab = labels.loc[ids]
    fam, sf, fold = lab.family.to_numpy(), lab.superfamily.to_numpy(), lab.fold.to_numpy()

    # ---- check the top 10 against the eval
    prot = pqm[pqm.method == 'Proteogram'].set_index('query_id')
    idx = {s: i for i, s in enumerate(ids)}
    same, overlap = 0, []
    for q, row in prot.iterrows():
        if q not in idx or not isinstance(row.top_hits, str):
            continue
        ev = row.top_hits.split(',')
        mine = [ids[j] for j in order[idx[q], :len(ev)]]
        same += mine == ev
        overlap.append(len(set(mine) & set(ev)) / len(ev))
    msg = (f'Top-10 check vs {os.path.basename(os.path.normpath(args.eval_dir))}: identical lists for '
           f'{same}/{len(overlap)} queries, mean overlap {np.mean(overlap):.3f}')
    print(msg)
    with open(os.path.join(args.out_dir, 'ranked_check.txt'), 'w') as f:
        f.write(msg + '\n')
    if np.mean(overlap) < 0.95:
        print('Warning: rankings differ from the eval; check the model file and proteograms.')

    # ---- per-query remote hits
    rp1 = {m: pqm[pqm.method == m].set_index('query_id')['rp1_fold'] for m in ALIGNERS}

    def relation(i, j):
        for lv, arr in (('family', fam), ('superfamily', sf), ('fold', fold)):
            if arr[i] == arr[j]:
                return lv
        return 'class' if lab['class'].iat[i] == lab['class'].iat[j] else 'none'

    rows = []
    for q in prot.index:
        if q not in idx:
            continue
        i = idx[q]
        o = order[i]
        base = dict(query=q, query_type=qtypes.get(q, ''),
                    **{f'{m}_rp1_fold': rp1[m].get(q, np.nan) for m in ALIGNERS})
        top = o[0]
        rows.append(dict(base, kind='top1', target=ids[top], search_rank=1, search_cosine=S[i, top],
                         relation=relation(i, top)))
        for kind, finer in (('remote_sf', fam), ('remote_fold', sf)):
            keep = np.nonzero(finer[o] != finer[i])[0]
            if len(keep):
                j = o[keep[0]]
                rows.append(dict(base, kind=kind, target=ids[j], search_rank=int(keep[0]) + 1,
                                 search_cosine=S[i, j], relation=relation(i, j)))
    cand = pd.DataFrame(rows)
    miss = cand[[f'{m}_rp1_fold' for m in ALIGNERS]].eq(0).all(axis=1)
    cand['aligners_miss_fold'] = miss

    def take(df, n):
        return df if len(df) <= n else df.iloc[rng.permutation(len(df))[:n]]

    picks = []
    fr = cand[(cand.kind == 'remote_fold') & (cand.relation == 'fold')]
    must = fr[fr.aligners_miss_fold]
    picks.append(must.assign(stratum='fold_remote'))
    picks.append(take(fr[~fr.aligners_miss_fold], max(0, args.n - len(must))).assign(stratum='fold_remote'))
    picks.append(take(cand[(cand.kind == 'remote_fold') & ~cand.relation.isin(['family', 'superfamily', 'fold'])],
                      args.n).assign(stratum='fold_remote_miss'))
    picks.append(take(cand[(cand.kind == 'remote_sf') & (cand.relation == 'superfamily')], args.n)
                 .assign(stratum='superfamily_remote'))
    picks.append(take(cand[(cand.kind == 'top1') & (cand.relation == 'family')], args.n // 2)
                 .assign(stratum='family_top1'))
    if qtypes:
        picks.append(take(cand[(cand.kind == 'top1') & (cand.query_type == 'novel_superfamily')], args.n)
                     .assign(stratum='novel_superfamily'))
    out = pd.concat(picks, ignore_index=True)
    out['priority'] = out.aligners_miss_fold & (out.stratum == 'fold_remote')
    out = out.sort_values(['priority', 'stratum'], ascending=[False, True], kind='stable')
    cols = ['query', 'target', 'stratum', 'relation', 'query_type', 'search_rank', 'search_cosine',
            'aligners_miss_fold'] + [f'{m}_rp1_fold' for m in ALIGNERS]
    out[cols].to_csv(os.path.join(args.out_dir, 'pairs.tsv'), sep='\t', index=False, float_format='%.6g')

    print(f'\nFirst hit outside the query\'s superfamily is in the same fold for {len(fr)} of '
          f'{int((cand.kind == "remote_fold").sum())} queries; all three aligners miss {len(must)} of those.')
    print('Pairs per stratum:')
    print(out.groupby('stratum').size().to_string())
    print(f'{int(out.priority.sum())} aligner-miss fold cases listed first. Wrote '
          f'{os.path.join(args.out_dir, "pairs.tsv")}')


if __name__ == '__main__':
    main()
