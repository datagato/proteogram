"""
Evaluate proteogram approach vs. gtalign.

PRECISION@K AND MAP@K METRICS FOR CLASS-LEVEL EVALUATION:

Precision@K for Classes:
    P@K = (# of top-K results in same class as query) / K
        
    Explanation: Measures what fraction of the top K search results belong to the 
    same structural class as the query protein. Not rank-aware; treats all top-K 
    results equally. A score of 1.0 means all top-K results are in the correct class.

Mean Average Precision@K for Classes:
    MAP@K = (1 / min(R, K)) × Σ(i=1 to K) [P(i) × rel(i)]
        
    where:
        - R = total number of proteins in corpus with same class as query
        - P(i) = precision at rank i = (# relevant results up to rank i) / i
        - rel(i) = 1 if result at rank i is in correct class, 0 otherwise
    
    Explanation: Rank-aware metric that gives higher scores when relevant results 
    appear earlier in the ranking. Normalizes by min(R, K) to account for queries 
    with fewer than K relevant items in the corpus. A score of 1.0 means all top-K 
    results are in the correct class AND they all appear first in the ranking.

Pertinent metric calculation info can be found at 
https://weaviate.io/blog/retrieval-evaluation-metrics
"""
import argparse
import pandas as pd
import numpy as np
import os
import glob
import re
import shutil
from collections import OrderedDict

from Bio.SCOP import Scop

from proteogram.common import read_yaml


def lookup(df, filter_col, filter_val, value_col):
    """Return the first matching value or None (avoids iloc[0] on empty results)."""
    match = df.loc[df[filter_col] == filter_val, value_col]
    return match.iloc[0] if not match.empty else None


def calc_patk_mapk(results_df, label_df, top_k, query_id_fn=None, limit_to_ids=None):
    """Calculate Precision@K and MAP@K for a structural alignment results DataFrame.

    Column 0 is the query identifier; columns 1+ are retrieved target identifiers.
    Both query and target IDs are looked up against label_df['pdb_id_chain'].

    Args:
        results_df: DataFrame where col 0 is query ID and cols 1+ are target IDs.
        label_df: DataFrame with columns pdb_id_chain, family, superfamily, fold, class.
        top_k: K for Precision@K and MAP@K.
        query_id_fn: Optional callable to transform the raw col-0 value into a
            pdb_id_chain key (e.g. stripping a filename suffix). Defaults to identity.
        limit_to_ids: Optional set of pdb_id_chain strings. If provided, only rows
            whose transformed query ID is in this set are evaluated.

    Returns:
        dict with keys patk_{fam,sfam,fold,class} and map_{fam,sfam,fold,class}.
    """
    if query_id_fn is None:
        query_id_fn = lambda x: x

    levels = ('family', 'superfamily', 'fold', 'class')
    patk   = {lv: [] for lv in levels}
    mapk   = {lv: [] for lv in levels}
    ratk   = {lv: [] for lv in levels}
    n_queries = 0
    n_skipped = 0

    for i in range(results_df.shape[0]):
        query_id = query_id_fn(results_df.iloc[i, 0])
        if limit_to_ids is not None and query_id not in limit_to_ids:
            continue
        query_labels = {lv: lookup(label_df, 'pdb_id_chain', query_id, lv) for lv in levels}
        if any(v is None for v in query_labels.values()):
            n_skipped += 1
            continue
        n_queries += 1

        r = {lv: max((label_df[lv] == query_labels[lv]).sum() - 1, 0) for lv in levels}
        tp   = {lv: 0   for lv in levels}
        prec = {lv: 0.0 for lv in levels}

        k = 0
        for target in results_df.iloc[i, 1:]:
            target_labels = {lv: lookup(label_df, 'pdb_id_chain', target, lv) for lv in levels}
            if any(v is None for v in target_labels.values()):
                continue
            k += 1
            for lv in levels:
                if query_labels[lv] == target_labels[lv]:
                    tp[lv] += 1
                    prec[lv] += tp[lv] / k
            if k >= top_k:
                break

        for lv in levels:
            patk[lv].append(tp[lv] / top_k)
            mapk[lv].append(prec[lv] / min(r[lv], top_k) if r[lv] > 0 else 0.0)
            ratk[lv].append(tp[lv] / r[lv] if r[lv] > 0 else 0.0)

    if n_skipped:
        print(f'WARNING: {n_skipped} queries skipped (not found in label_df).')
    return {
        'n_queries': n_queries,
        **{f'patk_{lv}': np.mean(patk[lv]) for lv in levels},
        **{f'map_{lv}':  np.mean(mapk[lv])  for lv in levels},
        **{f'ratk_{lv}': np.mean(ratk[lv])  for lv in levels},
    }


def read_gtalign_results(gtalign_results_dir):
    """Read gtalign results into a pandas dataframe (single-pass per file).

    Result lines are identified by a leading rank integer, e.g.:
        '     1 eval_structures/d1yl4r1.ent Chn:R  ...'
    The SCOPe domain ID is extracted from the path token (e.g. 'd1yl4r1').
    """
    files = glob.glob(os.path.join(gtalign_results_dir, '*.out'))
    rank_re = re.compile(r'^\s+(\d+)\s+\S')   # rank integer at line start
    path_re = re.compile(r'/(d\w+?)\.ent')     # SCOPe domain id in path

    results = []
    for file in files:
        tmp = [os.path.basename(file)]
        collected = 0
        with open(file) as fin:
            for line in fin:
                m = rank_re.match(line)
                if not m:
                    continue
                rank = int(m.group(1))
                if rank == 1 or rank > top_k+1:  # skip self-hit (rank 1) and anything beyond top_k
                    continue
                path_m = path_re.search(line)
                if not path_m:
                    print(f'WARNING: no SCOPe domain ID found at rank {rank} in {os.path.basename(file)} — empty string appended.')
                tmp.append(path_m.group(1) if path_m else '')
                collected += 1
                if collected >= top_k:
                    break
        results.append(tmp)

    return pd.DataFrame(results)

def read_usalign_results(usalign_results):
    """
    Read USalign results into a pandas dataframe.
    Making assumption that the top hit is itself since
    USalign does not return this data/score (self to self).
    """
    usalign_df = pd.read_csv(usalign_results, sep='\t')
    data = {}

    # dict of ordereddict:
    # {filename1: {filename2: tm1}}
    # {filename2: {filename1: tm2}}
    for i in range(usalign_df.shape[0]):
        scope_filename1 = os.path.splitext(os.path.basename(usalign_df.loc[i, '#PDBchain1']).split(':')[0])[0]
        scope_filename2 = os.path.splitext(os.path.basename(usalign_df.loc[i, 'PDBchain2']).split(':')[0])[0]
        tm1 = usalign_df.loc[i, 'TM1']
        tm2 = usalign_df.loc[i, 'TM2']

        if scope_filename1 in data.keys():
            tmp = data[scope_filename1]
            tmp[scope_filename2] = tm1
            data[scope_filename1] = tmp
        else:
            tmp = OrderedDict()
            tmp[scope_filename2] = tm1
            data[scope_filename1] = tmp
        if scope_filename2 in data.keys():
            tmp = data[scope_filename2]
            tmp[scope_filename1] = tm2
            data[scope_filename2] = tmp
        else:
            tmp = OrderedDict()
            tmp[scope_filename1] = tm2
            data[scope_filename2] = tmp
    
    ordered_data = []
    for key, inner_dict in data.items():
        sorted_inner_dict = sorted(inner_dict.items(), key=lambda item: item[1],
                                   reverse=True)
        tmp1 = sorted_inner_dict[:top_k]
        tmp2 = [key]
        tmp2.extend([prot for (prot,_) in tmp1])
        ordered_data.append(tmp2)

    return pd.DataFrame(ordered_data)
        
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluate proteogram vs. structural alignment methods.')
    parser.add_argument('--overwrite', action='store_true',
                        help='Recreate bad/good search image dirs without prompting.')
    parser.add_argument('--exclude_classes', '-x', default=None,
                        help='Comma-separated SCOPe class names to exclude from evaluation '
                             '(e.g. "g,h" to drop small/peptide classes).')
    args = parser.parse_args()

    # Files and folders
    config = read_yaml('config.yml')
    top_k = config['top_k']
    scope_level = config['scope_level']
    scope_eval_set = config['scope_eval_set']
    label_df_dir = os.path.dirname(scope_eval_set)
    proteogram_sim_results = config['proteogram_sim_results']
    gtalign_results_dir = config['gtalign_results_dir']
    usalign_results = config['usalign_results']
    search_images_dir = config['search_images_dir']
    save_bad_searches_dir = config['save_bad_searches_dir']
    save_good_searches_dir = config['save_good_searches_dir']

    scope_cla_handle = config['scope_cla_file']
    scope_des_handle = config['scope_des_file']
    scope_hie_handle = config['scope_hie_file']

    for d in (save_bad_searches_dir, save_good_searches_dir):
        if os.path.exists(d):
            if not args.overwrite:
                ans = input(f'{d} already exists. Recreate it? [y/N]: ').strip().lower()
                if ans != 'y':
                    print('Keeping existing directory. Pass --overwrite to skip this prompt.')
                else:
                    shutil.rmtree(d)
                    os.makedirs(d)
            else:
                shutil.rmtree(d)
                os.makedirs(d)
        else:
            os.makedirs(d)

    # Create a Scop object
    scop = Scop(cla_handle=open(scope_cla_handle, 'r'),
                des_handle=open(scope_des_handle, 'r'),
                hie_handle=open(scope_hie_handle, 'r'))

    # Use the scope_eval_set for a protein list (using the sid in the
    # first column to query local SCOPe info)
    scope_prots = []
    with open(scope_eval_set, 'r') as fin:
        for line in fin:
            try:
                spl_line = line.split()
                # Get a specific domain by its SCOP identifier (sid) found in scope_label_file
                scop_entry = scop.getDomainBySid(spl_line[0].replace('.ent',''))
                # Parse out info for our dataframe
                sccs = scop_entry.sccs
                sccs_spl = sccs.split('.')
                pdb_id_chain = scop_entry.sid  # SCOPe domain SID, e.g. 'd1jjua4'
                prot_file = f'{scop_entry.sid}.jpg'
                cls, fold, sfam, fam = sccs_spl[0], '.'.join(sccs_spl[:2]), '.'.join(sccs_spl[:3]), sccs
                #print(f'from label file | scope entry: {scop_entry} | family: {fam}')
                scope_prots.append([pdb_id_chain, prot_file, cls, fold, sfam, fam])
            except Exception as e:
                print(e)
    # Place data into a dataframe for easier access
    label_df = pd.DataFrame(scope_prots,
                            columns=['pdb_id_chain', 'proteogram_file', 'class', 'fold', 'superfamily', 'family'])

    if args.exclude_classes:
        excluded = {c.strip() for c in args.exclude_classes.split(',')}
        unknown = excluded - set(label_df['class'].unique())
        if unknown:
            print(f'WARNING: --exclude_classes named class(es) not found in data: {unknown}')
        before = len(label_df)
        label_df = label_df[~label_df['class'].isin(excluded)].reset_index(drop=True)
        print(f'Excluded {before - len(label_df)} entries for class(es): '
              + ', '.join(sorted(excluded - unknown)))

    print(label_df.nunique())
    print(f'label_df size: {len(label_df)}')

    label_df.to_csv(os.path.join(label_df_dir, 'scope_eval_set_labels.tsv'),
                    sep='\t', index=False)

    # Calculate Precision@K's and MAP@K's
    proteogram_res_df = pd.read_csv(proteogram_sim_results, sep='\t')
    precision_at_ks_fams = []
    precision_at_ks_sfams = []
    precision_at_ks_folds = []
    precision_at_ks_classes = []
    precision_at_ks_fams_map = []
    precision_at_ks_sfams_map = []
    precision_at_ks_folds_map = []
    precision_at_ks_classes_map = []
    recall_at_ks_fams = []
    recall_at_ks_sfams = []
    recall_at_ks_folds = []
    recall_at_ks_classes = []
    n_queries_skipped = 0
    proteogram_evaluated_ids = set()
    for i in range(proteogram_res_df.shape[0]):
        prot_file = os.path.basename(proteogram_res_df.iloc[i,0])
        query_fam   = lookup(label_df, 'proteogram_file', prot_file, 'family')
        query_sfam  = lookup(label_df, 'proteogram_file', prot_file, 'superfamily')
        query_fold  = lookup(label_df, 'proteogram_file', prot_file, 'fold')
        query_class = lookup(label_df, 'proteogram_file', prot_file, 'class')
        if any(v is None for v in (query_fam, query_sfam, query_fold, query_class)):
            n_queries_skipped += 1
            continue
        proteogram_evaluated_ids.add(os.path.splitext(prot_file)[0])
        
        # Relevant item counts in corpus (excluding self) for MAP@K normalisation
        r_fam   = max((label_df['family'] == query_fam).sum() - 1, 0)
        r_sfam  = max((label_df['superfamily'] == query_sfam).sum() - 1, 0)
        r_fold  = max((label_df['fold'] == query_fold).sum() - 1, 0)
        r_class = max((label_df['class'] == query_class).sum() - 1, 0)

        # For Precision calculation
        tp_fam = 0
        tp_sfam = 0
        tp_fold = 0
        tp_class = 0

        # For MAP@K calculation
        prec_at_k_fam = 0
        prec_at_k_sfam = 0
        prec_at_k_fold = 0
        prec_at_k_class = 0

        # Track number of valid results processed
        valid_results = 0

        # Iterate through results and find matches for family, superfamily,
        # fold and class SCOPe levels
        k = 0
        for target in proteogram_res_df.iloc[i,1:]:
            target_file  = os.path.basename(target.split(',')[0])
            if target_file == prot_file:  # skip self-hit
                continue
            target_fam   = lookup(label_df, 'proteogram_file', target_file, 'family')
            target_sfam  = lookup(label_df, 'proteogram_file', target_file, 'superfamily')
            target_fold  = lookup(label_df, 'proteogram_file', target_file, 'fold')
            target_class = lookup(label_df, 'proteogram_file', target_file, 'class')
            if any(v is None for v in (target_fam, target_sfam, target_fold, target_class)):
                continue
            k += 1
            valid_results += 1
            if query_fam == target_fam:
                tp_fam += 1
                prec_at_k_fam += (tp_fam / k)
            if query_sfam == target_sfam:
                tp_sfam += 1
                prec_at_k_sfam += (tp_sfam / k)
            if query_fold == target_fold:
                tp_fold += 1
                prec_at_k_fold += (tp_fold / k)
            if query_class == target_class:
                tp_class += 1
                prec_at_k_class += (tp_class / k)
            if k >= top_k:
                break

        # For Precision@K
        precision_at_ks_fams.append(tp_fam/top_k)
        precision_at_ks_sfams.append(tp_sfam/top_k)
        precision_at_ks_folds.append(tp_fold/top_k)
        precision_at_ks_classes.append(tp_class/top_k)

        # For MAP@K: normalise by min(R, K) where R = relevant items in corpus
        precision_at_ks_fams_map.append(prec_at_k_fam / min(r_fam, top_k) if r_fam > 0 else 0.0)
        precision_at_ks_sfams_map.append(prec_at_k_sfam / min(r_sfam, top_k) if r_sfam > 0 else 0.0)
        precision_at_ks_folds_map.append(prec_at_k_fold / min(r_fold, top_k) if r_fold > 0 else 0.0)
        precision_at_ks_classes_map.append(prec_at_k_class / min(r_class, top_k) if r_class > 0 else 0.0)

        # For Recall@K: tp / R
        recall_at_ks_fams.append(tp_fam / r_fam if r_fam > 0 else 0.0)
        recall_at_ks_sfams.append(tp_sfam / r_sfam if r_sfam > 0 else 0.0)
        recall_at_ks_folds.append(tp_fold / r_fold if r_fold > 0 else 0.0)
        recall_at_ks_classes.append(tp_class / r_class if r_class > 0 else 0.0)

        # Save images of those with no/full agreement at the configured scope_level
        stem = os.path.splitext(os.path.basename(prot_file))[0]
        # Use precision based on valid results found, not fixed top_k
        _tp_for_level = {'family': tp_fam, 'superfamily': tp_sfam,
                         'fold': tp_fold, 'class': tp_class}
        _query_for_level = {'family': query_fam, 'superfamily': query_sfam,
                            'fold': query_fold, 'class': query_class}
        tp_level = _tp_for_level.get(scope_level, tp_class)
        query_level = _query_for_level.get(scope_level, query_class)
        score_for_top_sims = tp_level / valid_results if valid_results > 0 else 0.0
        if score_for_top_sims <= 0.2:
            to_copy = f'{stem}_top_sims.jpg'
            shutil.copy(os.path.join(search_images_dir,
                                     to_copy),
                        os.path.join(save_bad_searches_dir, to_copy.replace('top_sims', f'top_sims_{query_level}_pk{score_for_top_sims:.4f}')))
        # Save images of those with complete agreement at scope_level (score == 1.0)
        if score_for_top_sims == 1.0:
            to_copy = f'{stem}_top_sims.jpg'
            shutil.copy(os.path.join(search_images_dir,
                                     to_copy),
                                     os.path.join(save_good_searches_dir, to_copy.replace('top_sims', f'top_sims_{query_level}_pk{score_for_top_sims:.4f}')))

    if n_queries_skipped:
        print(f'WARNING: {n_queries_skipped} proteogram queries skipped (not found in label_df).')

    proteogram_patk_fam = np.mean(precision_at_ks_fams)
    proteogram_patk_sfam = np.mean(precision_at_ks_sfams)
    proteogram_patk_fold = np.mean(precision_at_ks_folds)
    proteogram_patk_class = np.mean(precision_at_ks_classes)
    proteogram_map_fam = np.mean(precision_at_ks_fams_map)
    proteogram_map_sfam = np.mean(precision_at_ks_sfams_map)
    proteogram_map_fold = np.mean(precision_at_ks_folds_map)
    proteogram_map_class = np.mean(precision_at_ks_classes_map)
    proteogram_ratk_fam = np.mean(recall_at_ks_fams)
    proteogram_ratk_sfam = np.mean(recall_at_ks_sfams)
    proteogram_ratk_fold = np.mean(recall_at_ks_folds)
    proteogram_ratk_class = np.mean(recall_at_ks_classes)

    # Calculate Precision@K's and MAP@K's
    gtalign_res_df = read_gtalign_results(gtalign_results_dir)
    print(f'GTalign DataFrame rows: {gtalign_res_df.shape[0]}')
    gtalign_res_df.to_csv(os.path.join(gtalign_results_dir, 'combined_gtalign_results.tsv'),
                          sep='\t', index=False)
    gtalign_metrics = calc_patk_mapk(
        gtalign_res_df, label_df, top_k,
        query_id_fn=lambda x: x.replace('.ent.out', '').replace('.out', '')[:7],
        limit_to_ids=proteogram_evaluated_ids)

    usalign_res_df = read_usalign_results(usalign_results)
    usalign_metrics = calc_patk_mapk(usalign_res_df, label_df, top_k,
                                     limit_to_ids=proteogram_evaluated_ids)

    gt = gtalign_metrics
    us = usalign_metrics
    proteogram_n = len(precision_at_ks_fams)
    print(f'Proteins compared — Proteogram: {proteogram_n} | GTalign: {gt["n_queries"]} | USalign: {us["n_queries"]}')

    hdr  = f'{"Method":<15} | {"Class":>8} | {"Fold":>8} | {"Superfamily":>11} | {"Family":>8}'
    sep  = '-' * len(hdr)

    print(f'\nPrecision@K (K={top_k})')
    print(hdr)
    print(sep)
    print(f'{"GTalign":<15} | {gt["patk_class"]:>8.4f} | {gt["patk_fold"]:>8.4f} | {gt["patk_superfamily"]:>11.4f} | {gt["patk_family"]:>8.4f}')
    print(f'{"USalign":<15} | {us["patk_class"]:>8.4f} | {us["patk_fold"]:>8.4f} | {us["patk_superfamily"]:>11.4f} | {us["patk_family"]:>8.4f}')
    print(f'{"Proteogram":<15} | {proteogram_patk_class:>8.4f} | {proteogram_patk_fold:>8.4f} | {proteogram_patk_sfam:>11.4f} | {proteogram_patk_fam:>8.4f}')

    print(f'\nMAP@K (K={top_k})')
    print(hdr)
    print(sep)
    print(f'{"GTalign":<15} | {gt["map_class"]:>8.4f} | {gt["map_fold"]:>8.4f} | {gt["map_superfamily"]:>11.4f} | {gt["map_family"]:>8.4f}')
    print(f'{"USalign":<15} | {us["map_class"]:>8.4f} | {us["map_fold"]:>8.4f} | {us["map_superfamily"]:>11.4f} | {us["map_family"]:>8.4f}')
    print(f'{"Proteogram":<15} | {proteogram_map_class:>8.4f} | {proteogram_map_fold:>8.4f} | {proteogram_map_sfam:>11.4f} | {proteogram_map_fam:>8.4f}')

    print(f'\nRecall@K (K={top_k})')
    print(hdr)
    print(sep)
    print(f'{"GTalign":<15} | {gt["ratk_class"]:>8.4f} | {gt["ratk_fold"]:>8.4f} | {gt["ratk_superfamily"]:>11.4f} | {gt["ratk_family"]:>8.4f}')
    print(f'{"USalign":<15} | {us["ratk_class"]:>8.4f} | {us["ratk_fold"]:>8.4f} | {us["ratk_superfamily"]:>11.4f} | {us["ratk_family"]:>8.4f}')
    print(f'{"Proteogram":<15} | {proteogram_ratk_class:>8.4f} | {proteogram_ratk_fold:>8.4f} | {proteogram_ratk_sfam:>11.4f} | {proteogram_ratk_fam:>8.4f}')
