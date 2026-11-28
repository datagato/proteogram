"""
Evaluate proteogram approach vs. other aligners.

PRECISION@K AND MAP@K METRICS FOR CLASS-LEVEL EVALUATION:

Precision@K for Classes:
    P@K = (# of top-K results in same class as query) / K
        
    Explanation: Measures what fraction of the top K search results belong to the 
    same structural class as the query protein. Not rank-aware; treats all top-K 
    results equally. A score of 1.0 means all top-K results are in the correct class.

Capped Precision@K (a.k.a. normalized Precision@K):
    cP@K = (# of top-K results in same class as query) / min(R, K)

    where R = number of OTHER proteins in the corpus with the query's class.

    Explanation: Plain P@K can't reach 1.0 when R < K, so it penalizes queries
    from rare labels for something no method could do. Dividing by min(R, K)
    scores each query against what was actually findable: 1.0 means every
    findable relative is in the top K. Queries with R = 0 (no relative in the
    corpus) are undefined and are left out of the mean, and the number left out
    is reported. This is also the per-query score used to sort search images into
    good/bad folders.

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

Precision@1 (a.k.a. Hit@1):
    P@1 = 1 if the top (non-self) result is in the same class as the query, else 0,
    averaged over queries.

    Explanation: Nearest-neighbor annotation accuracy -- how often labeling a query
    by its single best hit gives the right answer. Unlike P@K it is not capped
    below 1.0 for queries whose label has fewer than K other members. Because it
    is a binary outcome per query, the --bootstrap significance test for it is an
    exact McNemar test on the discordant pairs (queries where exactly one of the
    two methods' top hit is correct) instead of Wilcoxon.

Remote Precision@1 and Remote Capped Precision@K (remote homology):
    Score only relatives the query does NOT share its next-finer group with:
        superfamily: same superfamily, different family
        fold:        same fold, different superfamily
        class:       same class, different fold
        family:      n/a (no finer level)
    Hits from the query's own finer group (e.g. its own family, at superfamily
    level) are removed from the ranked list -- counted neither right nor wrong
    -- and the list closes up, so the next K remaining hits are scored.
        rP@1  = 1 if the first remaining hit is in the query's group, else 0
        rcP@K = tp_remote / min(R_remote, K)
    where R_remote = corpus members in the query's group but outside its finer
    group. Queries with R_remote = 0 are left out (NaN), as for capped P@K.

    Explanation: with whole families held out, every test query still has its
    own family in the corpus, so plain superfamily scores are mostly family
    matching. These scores ask whether a method finds superfamily (or fold)
    relatives it can't get by near-duplicate family similarity -- the standard
    SCOP remote-homology setup. It needs ranked lists deeper than K, so all
    methods are scored over the same depth D (the shallowest method's list
    depth, or --remote_depth); a query whose list runs out before K remaining
    hits scores the hits it has and is counted as a short list.

Per-superfamily reporting:
    Pooled means weight every query equally, so one large superfamily can decide
    a comparison. The macro average first takes each metric's mean within each
    query superfamily, then averages those means with equal weight. Paired tests
    and (with --bootstrap) CIs use one value per superfamily. The report also
    shows the pooled margin after removing the superfamily that contributes most
    to it. With --train_list, results are also broken down by how many training
    proteins each query's superfamily had.

Query types (--query_types):
    A corpus from create_heldout_family_list.py --ood_level mixes novel-family
    queries (superfamily seen in training) with novel-superfamily or novel-fold
    queries (never seen). Every query searches the same corpus, and with
    --query_types each type also gets its own pooled and per-superfamily results.
    Pooled tables over the whole corpus mix the two settings.

Pertinent metric calculation info can be found at
https://weaviate.io/blog/retrieval-evaluation-metrics
"""
import argparse
import datetime
import json
import pandas as pd
import numpy as np
import os
import glob
import re
import shutil
import subprocess
import sys
from collections import OrderedDict

from Bio.SCOP import Scop

from proteogram.common import read_yaml

try:
    from scipy.stats import binomtest, wilcoxon
    _HAVE_SCIPY = True
except Exception:
    _HAVE_SCIPY = False


def lookup(df, filter_col, filter_val, value_col):
    """Return the first matching value or None (avoids iloc[0] on empty results)."""
    match = df.loc[df[filter_col] == filter_val, value_col]
    return match.iloc[0] if not match.empty else None


def image_stem(filename):
    """Basename without its extension.

    Proteogram lookups key on this rather than the full filename so a label
    table and a similarity-results file stay joinable across image formats:
    png is the current default, jpg for datasets built before the switch.
    """
    return os.path.splitext(os.path.basename(str(filename)))[0]


def capped_precision(tp, r, top_k):
    """Capped (normalized) Precision@K: tp / min(R, K); NaN when R = 0 (nothing
    findable). tp > R means R was counted over a smaller set than was searched --
    clipped to 1.0 and counted in CAPPED_OVERFLOW so the run can warn about it."""
    if r <= 0:
        return float('nan')
    if tp > r:
        CAPPED_OVERFLOW[0] += 1
    return min(tp / min(r, top_k), 1.0)


# Number of queries where tp exceeded R (see capped_precision).
CAPPED_OVERFLOW = [0]


def nanmean_or_nan(values):
    """Mean ignoring NaNs; NaN (without numpy's empty-slice warning) if all are NaN."""
    arr = np.asarray(values, dtype=float)
    return float(np.nanmean(arr)) if np.any(~np.isnan(arr)) else float('nan')


LEVELS = ('class', 'fold', 'superfamily', 'family')
# (key, table title, significance test used by --bootstrap)
METRICS = [('p1', 'Precision@1', 'McNemar exact test on discordant pairs'),
           ('patk', 'Precision@K', 'Wilcoxon signed-rank'),
           ('cpatk', 'Capped Precision@K', 'Wilcoxon signed-rank'),
           ('map', 'MAP@K', 'Wilcoxon signed-rank'),
           ('ratk', 'Recall@K', 'Wilcoxon signed-rank'),
           ('rp1', 'Remote Precision@1', 'McNemar exact test on discordant pairs'),
           ('rcpatk', 'Remote Capped Precision@K', 'Wilcoxon signed-rank')]
# Binary per-query metrics (McNemar) and metrics that are NaN for some queries.
BINARY_METRICS = ('p1', 'rp1')
NAN_METRICS = ('cpatk', 'rp1', 'rcpatk')
# Remote metrics at a level ignore hits that share the query's next-finer label.
REMOTE_FINER = {'class': 'fold', 'fold': 'superfamily', 'superfamily': 'family'}


def label_table(label_df, key):
    """{id: {level: label}} for fast per-hit lookups (key = pdb_id_chain or
    proteogram_stem)."""
    return (label_df.drop_duplicates(key).set_index(key)[list(LEVELS)]
            .to_dict('index'))


def label_counts(label_df):
    """{level: {label: corpus count}}, used for R and R_remote."""
    return {lv: label_df[lv].value_counts().to_dict() for lv in LEVELS}


def remote_r(q, counts, lv):
    """R_remote: corpus members in the query's group at lv but outside its
    next-finer group (the query itself is in its finer group, so it's excluded)."""
    fin = REMOTE_FINER[lv]
    return max(counts[lv].get(q[lv], 0) - counts[fin].get(q[fin], 0), 0)


def score_query(q, ranked, counts, top_k, remote_depth, remote_short):
    """Every per-query metric from one ranked hit list.

    q: the query's {level: label}. ranked: [(target_id, {level: label}), ...]
    in rank order, self and unlabeled targets already removed. The standard
    metrics use the first top_k hits; the remote metrics use the first
    remote_depth hits (None = all). remote_short[lv] is incremented when the
    list runs out before top_k remaining (non-excluded) hits.
    """
    top = ranked[:top_k]
    pq = {'top_hits': ','.join(t for t, _ in top)}
    for lv in LEVELS:
        r = max(counts[lv].get(q[lv], 0) - 1, 0)
        tp, prec = 0, 0.0
        for k, (_, t) in enumerate(top, 1):
            if t[lv] == q[lv]:
                tp += 1
                prec += tp / k
        pq[f'p1_{lv}'] = float(bool(top) and top[0][1][lv] == q[lv])
        pq[f'patk_{lv}'] = tp / top_k
        pq[f'cpatk_{lv}'] = capped_precision(tp, r, top_k)
        pq[f'map_{lv}'] = prec / min(r, top_k) if r > 0 else 0.0
        pq[f'ratk_{lv}'] = tp / r if r > 0 else 0.0

    pool = ranked if remote_depth is None else ranked[:remote_depth]
    for lv in LEVELS:
        pq[f'rp1_{lv}'] = pq[f'rcpatk_{lv}'] = float('nan')
        fin = REMOTE_FINER.get(lv)
        if fin is None:
            continue
        r_rem = remote_r(q, counts, lv)
        if r_rem == 0:
            continue
        tp, k, first = 0, 0, 0.0
        for _, t in pool:
            if t[fin] == q[fin]:      # own finer group: removed, list closes up
                continue
            k += 1
            if t[lv] == q[lv]:
                tp += 1
                if k == 1:
                    first = 1.0
            if k >= top_k:
                break
        if k < top_k:
            remote_short[lv] += 1
        pq[f'rp1_{lv}'] = first
        pq[f'rcpatk_{lv}'] = capped_precision(tp, r_rem, top_k)
    return pq


def list_depth(results_df):
    """Deepest ranked list in a results table (non-empty cells after column 0)."""
    if results_df.shape[0] == 0 or results_df.shape[1] < 2:
        return 0
    cells = results_df.iloc[:, 1:]
    return int((cells.notna() & (cells.astype(str) != '')).sum(axis=1).max())


def summarize_per_query(per_query, n_queries=None):
    """Mean of every per-query metric at every level, ignoring NaNs (capped P@K
    is NaN for R = 0 queries), plus how many queries each mean covers.

    Returns {'n_queries', 'per_query', '<metric>_<level>', '<metric>_n_<level>'}.
    """
    out = {'n_queries': len(per_query) if n_queries is None else n_queries,
           'per_query': per_query}
    for mkey, _, _ in METRICS:
        for lv in LEVELS:
            vals = np.array([pq[f'{mkey}_{lv}'] for pq in per_query.values()], dtype=float)
            out[f'{mkey}_{lv}'] = nanmean_or_nan(vals)
            out[f'{mkey}_n_{lv}'] = int(np.sum(~np.isnan(vals)))
    return out


def calc_patk_mapk(results_df, label_df, top_k, query_id_fn=None, limit_to_ids=None,
                   remote_depth=None):
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
        remote_depth: how deep into each ranked list the remote metrics may
            look (None = the whole list). Pass the same value for every method.

    Returns:
        summarize_per_query() output: means of every METRICS entry for each
        level, plus n_queries, per_query (which also holds each query's top
        hits) and remote_short ({level: queries whose list ran out before K
        remaining hits}). NaN per-query values (capped P@K with R = 0, remote
        metrics with R_remote = 0 or at family level) are left out of the means.
    """
    if query_id_fn is None:
        query_id_fn = lambda x: x

    labels_by_id = label_table(label_df, 'pdb_id_chain')
    counts = label_counts(label_df)
    max_depth = top_k if remote_depth is None else max(top_k, remote_depth)
    collect_all = remote_depth is None
    # Per-query metrics keyed by query_id (pdb_id_chain), for paired bootstrap.
    per_query = {}
    remote_short = {lv: 0 for lv in LEVELS}
    n_queries = 0
    n_skipped = 0

    for row in results_df.itertuples(index=False):
        query_id = query_id_fn(row[0])
        if limit_to_ids is not None and query_id not in limit_to_ids:
            continue
        q = labels_by_id.get(query_id)
        if q is None:
            n_skipped += 1
            continue
        n_queries += 1

        ranked = []
        for target in row[1:]:
            t = labels_by_id.get(target) if isinstance(target, str) else None
            if t is None or target == query_id:
                continue
            ranked.append((target, t))
            if not collect_all and len(ranked) >= max_depth:
                break
        per_query[query_id] = score_query(q, ranked, counts, top_k, remote_depth, remote_short)

    if n_skipped:
        print(f'WARNING: {n_skipped} queries skipped (not found in label_df).')
    out = summarize_per_query(per_query, n_queries=n_queries)
    out['remote_short'] = remote_short
    return out


def paired_bootstrap_delta(prot_vals, algn_vals, n_boot=10000, seed=0, ci=0.95):
    """Percentile bootstrap CI for the paired mean delta (prot - algn).

    prot_vals, algn_vals: 1-D arrays of the SAME per-query metric, aligned by
    query (index i is the same protein in both). Resampling the same indices for
    both preserves the pairing, so the CI reflects query-sampling variance in the
    *difference* -- i.e. how sensitive the Proteogram-vs-method margin is to which
    proteins happen to be in the test set.

    Returns (observed_delta, ci_low, ci_high).
    """
    prot_vals = np.asarray(prot_vals, dtype=float)
    algn_vals = np.asarray(algn_vals, dtype=float)
    diff = prot_vals - algn_vals
    observed = diff.mean()
    n = len(diff)
    rng = np.random.default_rng(seed)
    # Vectorised: (n_boot, n) index matrix -> mean of resampled diffs.
    idx = rng.integers(0, n, size=(n_boot, n))
    boot_means = diff[idx].mean(axis=1)
    lo = np.percentile(boot_means, (1 - ci) / 2 * 100)
    hi = np.percentile(boot_means, (1 + ci) / 2 * 100)
    return observed, lo, hi


def wilcoxon_pvalue(prot_vals, algn_vals):
    """Two-sided Wilcoxon signed-rank p-value on the paired per-query
    differences (a nonparametric test of whether the median delta is 0).
    Returns None if scipy is unavailable or the test is undefined (e.g. all
    per-query differences are exactly 0, common at class level)."""
    if not _HAVE_SCIPY:
        return None
    diff = np.asarray(prot_vals, dtype=float) - np.asarray(algn_vals, dtype=float)
    if np.allclose(diff, 0):
        return None
    try:
        return wilcoxon(diff, zero_method='wilcox', alternative='two-sided').pvalue
    except Exception:
        return None


def mcnemar_pvalue(prot_hits, algn_hits):
    """Exact two-sided McNemar p-value for paired binary outcomes (e.g. P@1).

    Only discordant queries carry information: b = Proteogram right & method
    wrong, c = Proteogram wrong & method right. Under H0 (equal accuracy) each
    discordant query is equally likely to fall either way, so b ~ Binomial(b+c,
    0.5); the exact binomial test is valid for any b+c, including small counts
    where the chi-square approximation is not. Returns (p, b, c); p is None if
    scipy is unavailable or there are no discordant pairs.
    """
    prot_hits = np.asarray(prot_hits, dtype=bool)
    algn_hits = np.asarray(algn_hits, dtype=bool)
    b = int(np.sum(prot_hits & ~algn_hits))
    c = int(np.sum(~prot_hits & algn_hits))
    if not _HAVE_SCIPY or b + c == 0:
        return None, b, c
    return binomtest(b, b + c, 0.5, alternative='two-sided').pvalue, b, c


def _sig_stars(p):
    if p is None:
        return 'n/a'
    if p < 1e-3:
        return '***'
    if p < 1e-2:
        return '**'
    if p < 5e-2:
        return '*'
    return 'ns'


def report_bootstrap(proteogram_per_query, method_per_query, top_k, n_boot, seed):
    """Print paired bootstrap deltas (Proteogram - method) with 95% CIs and
    p-values, for every method x level x metric. P@1 (binary per query) uses an
    exact McNemar test; the continuous metrics use Wilcoxon signed-rank.

    proteogram_per_query: {query_id: {metric_level: value}} for Proteogram.
    method_per_query: {method_name: {query_id: {metric_level: value}}}.

    Returns a list of row dicts (one per method x metric x level) with the
    delta, CI, test, p-value, number of pairs and, for P@1, discordant counts.
    """
    levels = LEVELS
    rows = []

    print('\n' + '=' * 78)
    print(f'PAIRED BOOTSTRAP: Proteogram - method  (delta [95% CI] sig)  '
          f'| {n_boot} resamples, seed {seed}')
    print('Significance on paired per-query values: McNemar for P@1, Wilcoxon otherwise '
          '(***<.001 **<.01 *<.05 ns; n/a = scipy missing or all-equal)')
    print('=' * 78)

    for mkey, mlabel, test_name in METRICS:
        if mkey not in BINARY_METRICS:
            mlabel += f' (K={top_k})'
        if mkey == 'cpatk':
            mlabel += ', R = 0 queries excluded'
        elif mkey in ('rp1', 'rcpatk'):
            mlabel += ', own finer group ignored, R_remote = 0 queries excluded'
        print(f'\n{mlabel}  [{test_name}]')
        hdr = f'{"vs Method":<12} | ' + ' | '.join(f'{lv:>22}' for lv in levels)
        print(hdr)
        print('-' * len(hdr))
        discordant = []
        for mname, mpq in method_per_query.items():
            common = sorted(set(proteogram_per_query) & set(mpq))
            cells = []
            counts = []
            for lv in levels:
                key = f'{mkey}_{lv}'
                prot = np.array([proteogram_per_query[q][key] for q in common], dtype=float)
                algn = np.array([mpq[q][key] for q in common], dtype=float)
                # Capped P@K / remote metrics are NaN where R (R_remote) = 0; R
                # comes from the shared label table, so both methods have NaN
                # at the same queries.
                defined = ~(np.isnan(prot) | np.isnan(algn))
                prot, algn = prot[defined], algn[defined]
                row = {'method': mname, 'metric': mkey, 'level': lv, 'test': test_name,
                       'n_pairs': int(len(prot)), 'delta': np.nan, 'ci_low': np.nan,
                       'ci_high': np.nan, 'p_value': np.nan, 'sig': 'n/a',
                       'proteogram_only_correct': np.nan, 'method_only_correct': np.nan}
                rows.append(row)
                if len(prot) == 0:
                    cells.append('n/a')
                    continue
                d, lo, hi = paired_bootstrap_delta(prot, algn, n_boot=n_boot, seed=seed)
                if mkey in BINARY_METRICS:
                    p, b, c = mcnemar_pvalue(prot, algn)
                    counts.append(f'{lv} {b}/{c}')
                    row.update(proteogram_only_correct=b, method_only_correct=c)
                else:
                    p = wilcoxon_pvalue(prot, algn)
                row.update(delta=d, ci_low=lo, ci_high=hi,
                           p_value=np.nan if p is None else p, sig=_sig_stars(p))
                cells.append(f'{d:+.3f}[{lo:+.3f},{hi:+.3f}]{_sig_stars(p):>3}')
            print(f'{mname:<12} | ' + ' | '.join(f'{c:>22}' for c in cells))
            if counts:
                discordant.append(f'  {mname}: ' + ', '.join(counts))
        if mkey in NAN_METRICS:
            n_by_level = ', '.join(
                f'{lv} {sum(1 for q in common if not np.isnan(proteogram_per_query[q][f"{mkey}_{lv}"]))}'
                for lv in levels)
            r_name = 'R' if mkey == 'cpatk' else 'R_remote'
            print(f'(paired on shared queries with {r_name} > 0: {n_by_level})')
        else:
            print(f'(paired on {len(common)} shared queries)')
        if discordant:
            print('Discordant pairs (Proteogram-only correct / method-only correct):')
            print('\n'.join(discordant))
    return rows


SUPERFAMILY_REPORT_METRICS = [
    ('rcpatk_superfamily', 'Remote capped P@K, superfamily'),
    ('rp1_superfamily', 'Remote P@1, superfamily'),
    ('map_superfamily', 'MAP@K, superfamily'),
    ('p1_superfamily', 'P@1, superfamily'),
    ('map_family', 'MAP@K, family'),
    ('p1_family', 'P@1, family'),
    ('rcpatk_fold', 'Remote capped P@K, fold'),
    ('rp1_fold', 'Remote P@1, fold'),
]
TRAIN_SIZE_BINS = [(1, 10), (11, 20), (21, 50), (51, 100), (101, None)]


def per_query_frame(methods, query_info):
    """Long table of the reported metrics: method, query_id, superfamily, metrics."""
    cols = [k for k, _ in SUPERFAMILY_REPORT_METRICS]
    rows = [{'method': name, 'query_id': q, 'superfamily': query_info[q]['superfamily'],
             **{c: pq[c] for c in cols}}
            for name, m in methods.items() for q, pq in m['per_query'].items()
            if q in query_info]
    return pd.DataFrame(rows)


def report_per_superfamily(methods, query_info, top_k, run_dir, n_boot=None, seed=0,
                           train_counts=None, prefix='', heading='PER-SUPERFAMILY (MACRO) RESULTS'):
    """Print and save macro-averaged (per-superfamily) comparisons.

    train_counts: optional {superfamily: training proteins} for the size breakdown.
    prefix: file-name prefix, used for the per-query-type reports.
    Returns {file name: description} for the files written.
    """
    frame = per_query_frame(methods, query_info)
    if frame.empty or 'Proteogram' not in set(frame['method']):
        return {}
    cols = [k for k, _ in SUPERFAMILY_REPORT_METRICS]
    by_sf = frame.groupby(['method', 'superfamily'])[cols].mean()
    table = by_sf.reset_index()
    n_q = frame[frame['method'] == 'Proteogram'].groupby('superfamily').size()
    table.insert(2, 'n_queries', table['superfamily'].map(n_q))
    if train_counts is not None:
        table.insert(3, 'n_train', table['superfamily'].map(train_counts).fillna(0).astype(int))
    written = {}
    table.to_csv(os.path.join(run_dir, f'{prefix}per_superfamily_metrics.tsv'), sep='\t',
                 index=False)
    written[f'{prefix}per_superfamily_metrics.tsv'] = ('each method\'s mean metrics per query superfamily, '
                                              'with query and training counts')

    print('\n' + '=' * 78)
    print(f'{heading}, K={top_k}: each superfamily has equal weight')
    print('delta = Proteogram - method; Wilcoxon on paired superfamily means'
          + ('; 95% bootstrap CI over superfamilies' if n_boot else ''))
    print('=' * 78)
    aligners = [n for n in methods if n != 'Proteogram']
    prot = by_sf.loc['Proteogram']
    prot_q = frame[frame['method'] == 'Proteogram'].set_index('query_id')
    rows = []
    for key, title in SUPERFAMILY_REPORT_METRICS:
        n_sf = int(prot[key].notna().sum())
        if n_sf == 0:
            continue
        means = ' | '.join(f'{n} {by_sf.loc[n][key].mean():.3f}' for n in methods)
        print(f'\n{title} ({n_sf} superfamilies): {means}')
        for name in aligners:
            other = by_sf.loc[name][key].reindex(prot.index)
            ok = prot[key].notna() & other.notna()
            a, b = prot.loc[ok, key].to_numpy(), other[ok].to_numpy()
            if len(a) == 0:
                continue
            p = wilcoxon_pvalue(a, b)
            lo = hi = np.nan
            if n_boot:
                _, lo, hi = paired_bootstrap_delta(a, b, n_boot=n_boot, seed=seed)
            macro = float((a - b).mean())
            # Pooled margin, and the same margin without its largest contributor.
            other_q = frame[frame['method'] == name].set_index('query_id')[key]
            diff = (prot_q[key] - other_q.reindex(prot_q.index)).dropna()
            pooled = float(diff.mean())
            by_group = diff.groupby(prot_q.loc[diff.index, 'superfamily']).sum()
            top = (by_group if pooled >= 0 else -by_group).idxmax()
            rest = diff[prot_q.loc[diff.index, 'superfamily'] != top]
            without = float(rest.mean()) if len(rest) else np.nan
            ci = f' [{lo:+.3f},{hi:+.3f}]' if n_boot else ''
            print(f'  vs {name:<9} macro {macro:+.3f}{ci} {_sig_stars(p):>3} | pooled {pooled:+.3f} | '
                  f'pooled without {top} ({int((prot_q.loc[diff.index, "superfamily"] == top).sum())} '
                  f'queries) {without:+.3f}')
            rows.append({'metric': key, 'method': name, 'n_superfamilies': int(len(a)),
                         'macro_proteogram': float(a.mean()), 'macro_method': float(b.mean()),
                         'macro_delta': macro, 'ci_low': lo, 'ci_high': hi,
                         'p_value': np.nan if p is None else p, 'sig': _sig_stars(p),
                         'pooled_delta': pooled, 'top_contributor': top,
                         'pooled_delta_without_top': without})
    pd.DataFrame(rows).to_csv(os.path.join(run_dir, f'{prefix}per_superfamily_tests.tsv'),
                              sep='\t', index=False)
    written[f'{prefix}per_superfamily_tests.tsv'] = ('macro delta, CI and Wilcoxon p per metric and method, '
                                            'plus the pooled delta without its top contributor')

    if train_counts is not None:
        size_rows = []
        print('\nMacro results by training proteins in the query\'s superfamily:')
        sf_train = pd.Series({sf: train_counts.get(sf, 0) for sf in prot.index})
        for lo_n, hi_n in TRAIN_SIZE_BINS:
            mask = sf_train >= lo_n if hi_n is None else sf_train.between(lo_n, hi_n)
            sfs = sf_train.index[mask]
            if len(sfs) == 0:
                continue
            label = f'>{lo_n - 1}' if hi_n is None else f'{lo_n}-{hi_n}'
            row = {'train_size': label, 'n_superfamilies': len(sfs),
                   'n_queries': int(n_q.reindex(sfs).sum())}
            for key in ('rcpatk_superfamily', 'map_superfamily', 'map_family'):
                for name in methods:
                    row[f'{name}_{key}'] = by_sf.loc[name].reindex(sfs)[key].mean()
            size_rows.append(row)
            print(f'  train {label:>7}: {len(sfs):3d} superfamilies, {row["n_queries"]:4d} queries | '
                  'remote capped P@K ' + ', '.join(
                      f'{n} {row[f"{n}_rcpatk_superfamily"]:.3f}' for n in methods))
        pd.DataFrame(size_rows).to_csv(os.path.join(run_dir, f'{prefix}per_train_size_metrics.tsv'),
                                       sep='\t', index=False)
        written[f'{prefix}per_train_size_metrics.tsv'] = ('macro metrics binned by superfamily '
                                                          'training size')
    return written


def report_by_query_type(methods, query_info, query_types, top_k, run_dir, n_boot=None, seed=0,
                         train_counts=None):
    """Pooled and per-superfamily results for each query type in one shared corpus.

    query_types: {query_id: 'novel_family' | 'novel_superfamily' | 'novel_fold'}.
    Training-size breakdowns apply only to novel_family, the type with training
    proteins in its superfamily. Returns {file name: description}.
    """
    missing = [q for q in query_info if q not in query_types]
    if missing:
        print(f'WARNING: {len(missing)} evaluated queries have no entry in --query_types and '
              f'are left out of the per-type results, e.g. {missing[:3]}.')
    written = {}
    rows = []
    order = ['novel_family', 'novel_superfamily', 'novel_fold']
    present = set(query_types.values())
    for qtype in order + sorted(present - set(order)):
        ids = {q for q, t in query_types.items() if t == qtype and q in query_info}
        if not ids:
            continue
        subset = OrderedDict()
        for name, m in methods.items():
            kept = {q: v for q, v in m['per_query'].items() if q in ids}
            if kept:
                subset[name] = {'per_query': kept}
        print('\n' + '#' * 78)
        print(f'QUERY TYPE: {qtype} ({len(ids)} queries, searched against the full corpus)')
        print('#' * 78)
        if 'Proteogram' not in subset:
            print('No Proteogram results for this query type.')
            continue
        prot = subset['Proteogram']['per_query']
        for key, title in SUPERFAMILY_REPORT_METRICS:
            shared = sorted(prot)
            values = np.array([prot[q][key] for q in shared], dtype=float)
            n = int((~np.isnan(values)).sum())
            if n == 0:
                continue
            means = ' | '.join(
                f'{name} {nanmean_or_nan([v["per_query"][q][key] for q in v["per_query"]]):.3f}'
                for name, v in subset.items())
            print(f'{title} (pooled, n={n}): {means}')
            for name, v in subset.items():
                if name == 'Proteogram':
                    continue
                common = sorted(set(prot) & set(v['per_query']))
                a = np.array([prot[q][key] for q in common], dtype=float)
                b = np.array([v['per_query'][q][key] for q in common], dtype=float)
                ok = ~(np.isnan(a) | np.isnan(b))
                a, b = a[ok], b[ok]
                if len(a) == 0:
                    continue
                if key.startswith(('p1_', 'rp1_')):
                    p, _, _ = mcnemar_pvalue(a, b)
                else:
                    p = wilcoxon_pvalue(a, b)
                rows.append({'query_type': qtype, 'metric': key, 'method': name,
                             'n_queries': int(len(a)), 'proteogram': float(a.mean()),
                             'method_value': float(b.mean()), 'delta': float((a - b).mean()),
                             'p_value': np.nan if p is None else p, 'sig': _sig_stars(p)})
                print(f'  vs {name:<9} {float((a - b).mean()):+.3f} {_sig_stars(p):>3}')
        written.update(report_per_superfamily(
            subset, query_info, top_k, run_dir, n_boot=n_boot, seed=seed,
            train_counts=train_counts if qtype == 'novel_family' else None,
            prefix=f'{qtype}_', heading=f'{qtype}: PER-SUPERFAMILY (MACRO) RESULTS'))
    pd.DataFrame(rows).to_csv(os.path.join(run_dir, 'query_type_summary.tsv'), sep='\t',
                              index=False)
    written['query_type_summary.tsv'] = ('pooled metrics and paired tests (McNemar for P@1, '
                                         'Wilcoxon otherwise) for each query type and method')
    return written


class Tee:
    """Write to several streams at once (console + report.txt)."""
    def __init__(self, *streams):
        self.streams = streams

    def write(self, text):
        for s in self.streams:
            s.write(text)

    def flush(self):
        for s in self.streams:
            s.flush()


def make_run_dir(base_dir, scope_level, top_k, prefix='eval'):
    """Create a new timestamped run folder under base_dir and return its path.
    A numeric suffix is added if the same second is already taken, so a run
    never writes into an existing folder."""
    stamp = datetime.datetime.now().strftime('%Y%m%d-%H%M%S')
    name = f'{prefix}_{stamp}_{scope_level}_k{top_k}'
    path = os.path.join(base_dir, name)
    n = 1
    while os.path.exists(path):
        n += 1
        path = os.path.join(base_dir, f'{name}_{n}')
    os.makedirs(path)
    return path


def print_metric_tables(methods, top_k, remote_depth=None):
    """Print one table per metric (rows = methods, columns = SCOPe levels).

    methods: OrderedDict {method name: summarize_per_query() output}.
    remote_depth: ranked-list depth the remote metrics used (for the title).
    """
    hdr = f'{"Method":<15} | {"Class":>8} | {"Fold":>8} | {"Superfamily":>11} | {"Family":>8}'
    sep = '-' * len(hdr)
    ref = methods.get('Proteogram') or next(reversed(methods.values()))
    depth_txt = 'full lists' if remote_depth is None else f'lists to depth {remote_depth}'
    for mkey, title, _ in METRICS:
        if mkey == 'p1':
            title += ' (top non-self hit correct)'
        elif mkey == 'cpatk':
            title += f' (K={top_k}; tp / min(R, K), queries with R = 0 excluded)'
        elif mkey == 'rp1':
            title += (f' (first hit outside the query\'s own family / superfamily / fold is in '
                      f'its superfamily / fold / class; {depth_txt})')
        elif mkey == 'rcpatk':
            title += (f' (K={top_k}; own finer group removed from the ranking, '
                      f'tp / min(R_remote, K); {depth_txt})')
        else:
            title += f' (K={top_k})'
        print(f'\n{title}')
        print(hdr)
        print(sep)
        for name, m in methods.items():
            cells = [f'{m[f"{mkey}_{lv}"]:.4f}' if not np.isnan(m[f'{mkey}_{lv}']) else 'n/a'
                     for lv in LEVELS]
            print(f'{name:<15} | {cells[0]:>8} | {cells[1]:>8} | {cells[2]:>11} | {cells[3]:>8}')
        if mkey in NAN_METRICS:
            label = '(n with R > 0)' if mkey == 'cpatk' else '(n, R_remote>0)'
            print(f'{label:<15} | {ref[f"{mkey}_n_class"]:>8d} | '
                  f'{ref[f"{mkey}_n_fold"]:>8d} | {ref[f"{mkey}_n_superfamily"]:>11d} | '
                  f'{ref[f"{mkey}_n_family"]:>8d}')
        if mkey == 'rcpatk':
            short = [f'{name}: ' + ', '.join(f'{lv} {m["remote_short"][lv]}'
                                             for lv in REMOTE_FINER)
                     for name, m in methods.items()
                     if any(m.get('remote_short', {}).values())]
            if short:
                print(f'Short lists (ran out before K={top_k} hits outside the query\'s finer '
                      f'group, scored on the hits available): ' + '; '.join(short))
        if mkey == 'cpatk':
            if CAPPED_OVERFLOW[0]:
                print(f'WARNING: {CAPPED_OVERFLOW[0]} query/level score(s) had more hits than '
                      f'R, so R was counted over a smaller set than was searched '
                      f'(scope_eval_set vs. the search corpus). Capped P@K was clipped to 1.0 '
                      f'there, and capped P@K/MAP@K are unreliable until R is counted over the '
                      f'full search corpus.')


def write_run_files(run_dir, methods, query_info, triage_rows, bootstrap_rows, scope_level):
    """Write the evaluation TSVs into run_dir; return {file name: description}.

    methods: OrderedDict {method name: summarize_per_query() output}.
    query_info: {query_id: {level: label, 'R_<level>': R}}.
    triage_rows: one dict per Proteogram query (see the main loop).
    bootstrap_rows: report_bootstrap() output, or None if --bootstrap was off.
    """
    written = {}
    level = scope_level if scope_level in LEVELS else 'class'
    metric_cols = [f'{mkey}_{lv}' for mkey, _, _ in METRICS for lv in LEVELS]

    def save(df, name, description):
        df.to_csv(os.path.join(run_dir, name), sep='\t', index=False)
        written[name] = description

    # Summary metrics, long format: one row per method x metric x level.
    titles = {mkey: title for mkey, title, _ in METRICS}
    save(pd.DataFrame([{'method': name, 'metric': mkey, 'metric_name': titles[mkey],
                        'level': lv, 'value': m[f'{mkey}_{lv}'],
                        'n_queries_scored': m[f'{mkey}_n_{lv}']}
                       for name, m in methods.items()
                       for mkey, _, _ in METRICS for lv in LEVELS]),
         'metrics_summary.tsv',
         'mean of each metric per method and SCOPe level, with the number of queries scored')

    # Every method's per-query metrics, labels and R.
    rows = []
    for name, m in methods.items():
        for q, pq in m['per_query'].items():
            rows.append({'method': name, 'query_id': q, **query_info.get(q, {}),
                         'top_hits': pq.get('top_hits', ''),
                         **{c: pq[c] for c in metric_cols}})
    per_query_df = pd.DataFrame(rows)
    save(per_query_df, 'per_query_metrics.tsv',
         'every metric for every query and method, with SCOPe labels, R per level and top hits')

    if bootstrap_rows is not None:
        save(pd.DataFrame(bootstrap_rows), 'bootstrap.tsv',
             'paired bootstrap delta (Proteogram - method) with 95% CI, test, p-value and, '
             'for P@1, discordant counts')

    # Proteogram search triage at scope_level, worst first within each file.
    triage = pd.DataFrame(triage_rows)
    if not triage.empty:
        triage = triage.sort_values(['capped_patk', 'R'], ascending=[True, False],
                                    na_position='last')
        save(triage, 'search_triage.tsv',
             f'every Proteogram query at {level} level: category, R, capped P@K, P@1, '
             f'AP@K, recall, top hits and their labels')
        for cat, desc in (('good', 'capped P@K = 1.0'), ('bad', 'capped P@K <= 0.2'),
                          ('no_relative', 'R = 0, no relative in the corpus')):
            save(triage[triage['category'] == cat], f'{cat}_searches.tsv',
                 f'Proteogram queries with {desc} (subset of search_triage.tsv)')

    if not per_query_df.empty and level in per_query_df.columns:
        # Per-label breakdown at scope_level: which labels each method struggles with.
        label_metrics = [f'p1_{level}', f'cpatk_{level}', f'map_{level}']
        if level in REMOTE_FINER:
            label_metrics += [f'rp1_{level}', f'rcpatk_{level}']
        stats = (per_query_df.groupby([level, 'method'])
                 [label_metrics].mean().unstack('method'))
        stats.columns = [f'{method}_{metric.rsplit("_", 1)[0]}' for metric, method in stats.columns]
        first = per_query_df[per_query_df['method'] == next(reversed(methods))]
        counts = first.groupby(level).agg(n_queries=('query_id', 'size'),
                                          R=(f'R_{level}', 'first'))
        per_label = counts.join(stats).reset_index().sort_values('n_queries', ascending=False)
        save(per_label, 'per_label_summary.tsv',
             f'per {level}: queries, R, and each method\'s mean P@1 / capped P@K / AP@K')

        # Side-by-side per query at scope_level, flagging P@1 disagreements
        # (the discordant pairs behind the McNemar test).
        wide = per_query_df.pivot_table(index='query_id', columns='method',
                                        values=[f'p1_{level}', f'cpatk_{level}'],
                                        aggfunc='first', dropna=False)
        wide.columns = [f'{method}_{metric.rsplit("_", 1)[0]}' for metric, method in wide.columns]
        hits = per_query_df.pivot_table(index='query_id', columns='method', values='top_hits',
                                        aggfunc='first')
        hits.columns = [f'{method}_top_hits' for method in hits.columns]
        p1_cols = [c for c in wide.columns if c.endswith('_p1')]
        comp = wide.join(hits)
        comp.insert(0, 'p1_disagree', comp[p1_cols].nunique(axis=1, dropna=True) > 1)
        info = pd.DataFrame.from_dict(query_info, orient='index')[[level, f'R_{level}']]
        comp = info.join(comp, how='inner').rename_axis('query_id').reset_index()
        save(comp.sort_values(['p1_disagree', level], ascending=[False, True]),
             'method_comparison.tsv',
             f'one row per query: each method\'s P@1, capped P@K and top hits at {level} level; '
             f'p1_disagree marks queries where the methods\' top hits differ in correctness')
    return written


def git_commit():
    """Current git commit (with '-dirty' if there are uncommitted changes), or None."""
    here = os.path.dirname(os.path.abspath(__file__))
    try:
        sha = subprocess.run(['git', 'rev-parse', 'HEAD'], cwd=here, capture_output=True,
                             text=True, check=True).stdout.strip()
        dirty = subprocess.run(['git', 'status', '--porcelain'], cwd=here, capture_output=True,
                               text=True, check=True).stdout.strip()
        return sha + ('-dirty' if dirty else '')
    except Exception:
        return None


def read_gtalign_results(gtalign_results_dir, top_k=None):
    """Read gtalign results into a pandas dataframe (single-pass per file).
    Keeps the top_k hits per query after the self-hit, or every hit if top_k
    is None.

    Result lines are identified by a leading rank integer, e.g.:
        '     1 eval_structures/d1yl4r1.ent Chn:R  ...'
    The SCOPe domain ID is extracted from the path token (e.g. 'd1yl4r1').
    """
    files = glob.glob(os.path.join(gtalign_results_dir, '*.out'))
    # A hit row is rank, structure path, then 'Chn:'. Requiring 'Chn:' skips
    # header lines that also start with a number ('  1575 structure(s)',
    # '  324887 total residues').
    rank_re = re.compile(r'^\s+(\d+)\s+\S+\s+Chn:')
    # SCOPe domain id in the path. Multi-chain domain ids contain a dot
    # (d1bom.1), so '.' is allowed; the lazy match stops at the final '.ent'.
    path_re = re.compile(r'/(d[\w.]+?)\.ent\b')

    results = []
    n_unparsed = 0
    unparsed_example = None
    for file in files:
        tmp = [os.path.basename(file)]
        collected = 0
        with open(file) as fin:
            for line in fin:
                m = rank_re.match(line)
                if not m:
                    continue
                rank = int(m.group(1))
                # skip self-hit (rank 1) and anything beyond top_k
                if rank == 1 or (top_k is not None and rank > top_k + 1):
                    continue
                path_m = path_re.search(line)
                if not path_m:
                    n_unparsed += 1
                    if unparsed_example is None:
                        unparsed_example = f'{os.path.basename(file)} rank {rank}: {line.strip()}'
                tmp.append(path_m.group(1) if path_m else '')
                collected += 1
                if top_k is not None and collected >= top_k:
                    break
        results.append(tmp)

    if n_unparsed:
        print(f'WARNING: {n_unparsed} GTalign hit row(s) had no SCOPe domain ID in the '
              f'path and were left blank (not scored). First: {unparsed_example}')
    return pd.DataFrame(results)

def _foldseek_id(raw):
    """Extract a 7-char SCOPe SID from a raw Foldseek query/target field.

    Foldseek may output any of:
      - 'd1a3wa3.ent'   (filename with extension)
      - 'd1a3wa3'       (stem only)
      - 'd1a3wa3A'      (stem + chain letter appended by createdb)
    SCOPe SIDs are always exactly 7 characters (e.g. 'd1a3wa3').
    """
    stem = os.path.splitext(os.path.basename(raw))[0]
    return stem[:7]


def read_foldseek_results(foldseek_results, top_k=None):
    """Read Foldseek easy-search results into a pandas DataFrame, keeping the
    top_k hits per query (every hit if top_k is None).

    Expects output produced with:
        --format-output "query,target,alntmscore"
    which yields a headerless three-column TSV.  Self-hits are present in
    Foldseek output and are skipped here (query == target after stem extraction).
    """
    fs_df = pd.read_csv(foldseek_results, sep='\t', header=None,
                        names=['query', 'target', 'score'])
    data = {}
    for _, row in fs_df.iterrows():
        q = _foldseek_id(row['query'])
        t = _foldseek_id(row['target'])
        score = float(row['score'])
        if q not in data:
            data[q] = OrderedDict()
        if t not in data[q] or score > data[q][t]:
            data[q][t] = score

    ordered_data = []
    for query_id, targets in data.items():
        sorted_targets = sorted(targets.items(), key=lambda x: x[1], reverse=True)
        filtered = [t for t, _ in sorted_targets if t != query_id]
        ordered_data.append([query_id] + (filtered if top_k is None else filtered[:top_k]))
    return pd.DataFrame(ordered_data)


def read_usalign_results(usalign_results, top_k=None):
    """
    Read USalign results into a pandas dataframe, keeping the top_k hits per
    query (every hit if top_k is None).
    Making assumption that the top hit is itself since
    USalign does not return this data/score (self to self).
    """
    usalign_df = pd.read_csv(usalign_results, sep='\t')
    n_before = usalign_df.shape[0]
    usalign_df = usalign_df.dropna(subset=['#PDBchain1', 'PDBchain2']).reset_index(drop=True)
    n_dropped = n_before - usalign_df.shape[0]
    if n_dropped:
        print(f'WARNING: {n_dropped} USalign rows skipped (missing PDBchain1/PDBchain2).')
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
        tmp1 = sorted_inner_dict if top_k is None else sorted_inner_dict[:top_k]
        tmp2 = [key]
        tmp2.extend([prot for (prot,_) in tmp1])
        ordered_data.append(tmp2)

    return pd.DataFrame(ordered_data)
        
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluate proteogram vs. structural alignment methods.')
    parser.add_argument('--overwrite', action='store_true',
                        help='No longer needed: every run writes to a new timestamped folder '
                             '(see --out_dir). Accepted and ignored for older command lines.')
    parser.add_argument('--config', default='config.yml',
                        help='Config file (default: config.yml in the working directory). Its '
                             'proteogram_sim_results names the Proteogram search to evaluate; a '
                             'copy goes into the run folder.')
    parser.add_argument('--out_dir', default=None,
                        help='Base folder for evaluation runs; each run creates '
                             'eval_<YYYYmmdd-HHMMSS>_<scope_level>_k<top_k>/ inside it with the '
                             'report, metric/bootstrap/per-query/triage TSVs, a copy of '
                             'config.yml and the sorted search images. Default: eval_output_dir '
                             'in config.yml, else eval_runs/ next to save_bad_searches_dir '
                             '(or next to proteogram_sim_results).')
    parser.add_argument('--exclude_classes', '-x', default=None,
                        help='Comma-separated SCOPe class names to exclude from evaluation '
                             '(e.g. "g,h" to drop small/peptide classes).')
    parser.add_argument('--bootstrap', action='store_true',
                        help='After the metric tables, report paired per-query bootstrap CIs '
                             'and p-values (exact McNemar for Precision@1, Wilcoxon signed-rank '
                             'for the other metrics) for the Proteogram-minus-alignment delta at '
                             'each level/metric. Since all methods score the same queries, this '
                             'tests whether the margin is significant and how sensitive it is to '
                             'which proteins are in the test set.')
    parser.add_argument('--n_boot', type=int, default=10000,
                        help='Number of bootstrap resamples for --bootstrap. Default: 10000.')
    parser.add_argument('--boot_seed', type=int, default=0,
                        help='RNG seed for --bootstrap resampling. Default: 0.')
    parser.add_argument('--remote_depth', type=int, default=None,
                        help='How deep into each ranked list the remote metrics (hits '
                             'outside the query\'s own family/superfamily/fold) may look. '
                             'Default: the shallowest method\'s list depth, so every method is '
                             'scored over the same depth. Lower it to compare at a fixed '
                             'depth; it is never raised above the shallowest list.')
    parser.add_argument('--train_list', default=None,
                        help='Training list for the model being evaluated. Adds a breakdown of '
                             'per-superfamily results by number of training proteins.')
    parser.add_argument('--query_types', default=None,
                        help='query_types TSV from create_heldout_family_list.py --ood_level. '
                             'Reports each query type (novel family, novel superfamily or fold) '
                             'separately. Set scope_eval_set to the matching eval_corpus list.')
    args = parser.parse_args()

    # Files and folders
    config = read_yaml(args.config)
    top_k = config['top_k']
    scope_level = config['scope_level']
    scope_eval_set = config['scope_eval_set']
    label_df_dir = os.path.dirname(scope_eval_set)
    proteogram_sim_results = config['proteogram_sim_results']
    gtalign_results_dir = config['gtalign_results_dir']
    usalign_results = config['usalign_results']
    foldseek_results = config.get('foldseek_results')
    search_images_dir = config['search_images_dir']

    scope_cla_handle = config['scope_cla_file']
    scope_des_handle = config['scope_des_file']
    scope_hie_handle = config['scope_hie_file']

    # ---- Timestamped run folder: nothing from an earlier run is overwritten ----
    base_out = args.out_dir or config.get('eval_output_dir') or os.path.join(
        os.path.dirname(os.path.normpath(
            config.get('save_bad_searches_dir') or os.path.abspath(proteogram_sim_results))),
        'eval_runs')
    run_dir = make_run_dir(base_out, scope_level, top_k)
    image_dirs = {'good': os.path.join(run_dir, 'search_images', 'good'),
                  'bad': os.path.join(run_dir, 'search_images', 'bad'),
                  'no_relative': os.path.join(run_dir, 'search_images', 'no_relative')}
    for d in image_dirs.values():
        os.makedirs(d)
    shutil.copy(args.config, os.path.join(run_dir, 'config.yml'))
    # Everything printed from here on also goes to report.txt.
    report_fh = open(os.path.join(run_dir, 'report.txt'), 'w')
    sys.stdout = Tee(sys.__stdout__, report_fh)
    started = datetime.datetime.now()
    print(f'Evaluation run folder: {run_dir}')
    print(f'Started {started:%Y-%m-%d %H:%M:%S}; command: {" ".join(sys.argv)}')
    if config.get('save_bad_searches_dir') or config.get('save_good_searches_dir'):
        print('Note: search images now go to search_images/{good,bad,no_relative} in the run '
              'folder; save_bad_searches_dir/save_good_searches_dir in config.yml are no '
              'longer written to.')
    triage_counts = {'good': 0, 'bad': 0, 'no_relative': 0}
    triage_rows = []
    query_info = {}  # query_id -> labels and R at each level, for the per-query TSV

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
                prot_file = f'{scop_entry.sid}.png'
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
    label_df.to_csv(os.path.join(run_dir, 'scope_eval_set_labels.tsv'), sep='\t', index=False)

    # Extension-insensitive join key for the proteogram lookups below. Added
    # after the write so the published TSV keeps its documented columns, and
    # derived from proteogram_file so a label table written before the png
    # switch still joins against png results.
    label_df['proteogram_stem'] = label_df['proteogram_file'].apply(image_stem)

    # Read every method's ranked lists in full first: the remote metrics look
    # past K (after removing the query's own family etc.), and every method is
    # scored over the same depth -- the shallowest method's list.
    proteogram_res_df = pd.read_csv(proteogram_sim_results, sep='\t')
    gtalign_res_df = read_gtalign_results(gtalign_results_dir)
    print(f'GTalign DataFrame rows: {gtalign_res_df.shape[0]}')
    usalign_res_df = read_usalign_results(usalign_results)
    foldseek_res_df = None
    if foldseek_results:
        fs_raw = pd.read_csv(foldseek_results, sep='\t', header=None,
                             names=['query', 'target', 'score'])
        scores = pd.to_numeric(fs_raw['score'], errors='coerce').dropna()
        print(f'Foldseek score column — n: {len(scores)}, '
              f'min: {scores.min():.4f}, max: {scores.max():.4f}, '
              f'mean: {scores.mean():.4f}, '
              f'n_zero: {(scores == 0).sum()}')
        print(f'Foldseek file columns: {fs_raw.shape[1]}  '
              f'sample row: {list(fs_raw.iloc[0])}')
        foldseek_res_df = read_foldseek_results(foldseek_results)

    # Proteogram lists include the self-hit, the aligners' don't.
    depths = {'Proteogram': max(list_depth(proteogram_res_df) - 1, 0),
              'GTalign': list_depth(gtalign_res_df), 'USalign': list_depth(usalign_res_df)}
    if foldseek_res_df is not None:
        depths['Foldseek'] = list_depth(foldseek_res_df)
    remote_depth = min(depths.values())
    if args.remote_depth is not None:
        remote_depth = min(remote_depth, args.remote_depth)
    print('Ranked-list depth per method: '
          + ', '.join(f'{name} {d}' for name, d in depths.items())
          + f'. Remote metrics use the first {remote_depth} hits of every method.')
    if remote_depth <= top_k:
        print(f'WARNING: remote metrics need ranked lists deeper than K={top_k}, but the '
              f'shallowest is {remote_depth}. Hits from the query\'s own family (or '
              f'superfamily) use up the list, so remote scores will be low for every method. '
              f'Rerun the shallowest search with more hits per query.')

    labels_by_stem = label_table(label_df, 'proteogram_stem')
    counts = label_counts(label_df)
    proteogram_remote_short = {lv: 0 for lv in LEVELS}
    max_depth = max(top_k, remote_depth)

    n_queries_skipped = 0
    proteogram_evaluated_ids = set()
    proteogram_per_query = {}  # {query_id: {metric_level: value}} for bootstrap
    for row in proteogram_res_df.itertuples(index=False):
        prot_file = image_stem(row[0])
        _q = labels_by_stem.get(prot_file)
        if _q is None:
            n_queries_skipped += 1
            continue
        proteogram_evaluated_ids.add(prot_file)  # already an extension-less stem

        # Ranked (target, labels) hits, self and unlabeled targets removed.
        ranked = []
        for target in row[1:]:
            if not isinstance(target, str):
                continue
            target_file = image_stem(target.split(',')[0])
            if target_file == prot_file:  # skip self-hit
                continue
            t = labels_by_stem.get(target_file)
            if t is None:
                continue
            ranked.append((target_file, t))
            if len(ranked) >= max_depth:
                break
        hits = ranked[:top_k]
        valid_results = len(hits)

        # Per-query metrics keyed by SCOPeID (== pdb_id_chain, so it pairs with
        # the alignment methods' per_query dicts). MAP@K and capped P@K
        # normalise by min(R, K), R = relevant items in the corpus; Recall@K by R.
        _qid = prot_file
        _r = {lv: max(counts[lv].get(_q[lv], 0) - 1, 0) for lv in LEVELS}
        _tp = {lv: sum(t[lv] == _q[lv] for _, t in hits) for lv in LEVELS}
        query_info[_qid] = {**{lv: _q[lv] for lv in LEVELS},
                            **{f'R_{lv}': int(_r[lv]) for lv in LEVELS},
                            **{f'R_remote_{lv}': remote_r(_q, counts, lv)
                               for lv in REMOTE_FINER}}
        proteogram_per_query[_qid] = score_query(_q, ranked, counts, top_k, remote_depth,
                                                 proteogram_remote_short)

        # Save search images sorted by capped precision at the configured
        # scope_level: tp / min(R, K), i.e. relative to what was findable, so a
        # query from a rare label isn't filed as "bad" just because it has fewer
        # than K relatives in the corpus. R = 0 queries had nothing to find and
        # go to their own folder. R is in the file name for context.
        stem = os.path.splitext(os.path.basename(prot_file))[0]
        triage_level = scope_level if scope_level in _tp else 'class'
        r_level = _r[triage_level]
        query_level = _q[triage_level]
        to_copy = f'{stem}_top_sims.jpg'
        score_for_top_sims = float('nan')
        if r_level == 0:
            category, tag = 'no_relative', f'top_sims_{query_level}_R0'
        else:
            score_for_top_sims = capped_precision(_tp[triage_level], r_level, top_k)
            tag = f'top_sims_{query_level}_R{r_level}_cpk{score_for_top_sims:.4f}'
            if score_for_top_sims <= 0.2:
                category = 'bad'
            elif score_for_top_sims == 1.0:
                category = 'good'
            else:
                category = 'mid'
        saved_image = ''
        if category in image_dirs:
            triage_counts[category] += 1
            saved_image = to_copy.replace('top_sims', tag)
            src = os.path.join(search_images_dir, to_copy)
            if os.path.exists(src):
                shutil.copy(src, os.path.join(image_dirs[category], saved_image))
            else:
                saved_image = ''
                print(f'WARNING: search image not found, not copied: {src}')
        pq = proteogram_per_query[_qid]
        triage_rows.append({
            'query_id': _qid, 'category': category, 'level': triage_level,
            'label': query_level, 'R': int(r_level), 'hits_in_top_k': _tp[triage_level],
            'n_results': valid_results, 'capped_patk': score_for_top_sims,
            'p1': pq[f'p1_{triage_level}'], 'map': pq[f'map_{triage_level}'],
            'ratk': pq[f'ratk_{triage_level}'],
            'top_hits': ','.join(t for t, _ in hits),
            'top_hit_labels': ','.join(str(lbls[triage_level]) for _, lbls in hits),
            **{lv: _q[lv] for lv in LEVELS},
            'image': saved_image})

    if n_queries_skipped:
        print(f'WARNING: {n_queries_skipped} proteogram queries skipped (not found in label_df).')
    print(f'Search images at {scope_level} level (capped P@K): {triage_counts["good"]} good '
          f'(= 1.0), {triage_counts["bad"]} bad (<= 0.2), {triage_counts["no_relative"]} with '
          f'no relative in the corpus (R = 0).')

    proteogram_metrics = summarize_per_query(proteogram_per_query)
    proteogram_metrics['remote_short'] = proteogram_remote_short

    # Calculate Precision@K's and MAP@K's
    gtalign_res_df.iloc[:, :top_k + 1].to_csv(
        os.path.join(gtalign_results_dir, 'combined_gtalign_results.tsv'),
        sep='\t', index=False)
    gtalign_metrics = calc_patk_mapk(
        gtalign_res_df, label_df, top_k,
        query_id_fn=lambda x: x.replace('.ent.out', '').replace('.out', '')[:7],
        limit_to_ids=proteogram_evaluated_ids, remote_depth=remote_depth)

    usalign_metrics = calc_patk_mapk(usalign_res_df, label_df, top_k,
                                     limit_to_ids=proteogram_evaluated_ids,
                                     remote_depth=remote_depth)

    if foldseek_res_df is not None:
        label_ids = set(label_df['pdb_id_chain'])
        fs_targets = [t for t in pd.unique(foldseek_res_df.iloc[:, 1:].values.ravel())
                      if isinstance(t, str)]
        n_matched = sum(1 for t in fs_targets if t in label_ids)
        print(f'Foldseek unique targets: {len(fs_targets)}, '
              f'found in label_df: {n_matched} ({100*n_matched/max(len(fs_targets),1):.1f}%)')
        foldseek_metrics = calc_patk_mapk(foldseek_res_df, label_df, top_k,
                                          limit_to_ids=proteogram_evaluated_ids,
                                          remote_depth=remote_depth)
    else:
        foldseek_metrics = None

    methods = OrderedDict([('GTalign', gtalign_metrics), ('USalign', usalign_metrics)])
    if foldseek_metrics:
        methods['Foldseek'] = foldseek_metrics
    methods['Proteogram'] = proteogram_metrics
    print('Proteins compared — ' + ' | '.join(
        f'{name}: {m["n_queries"]}' for name, m in methods.items())
          + ('' if foldseek_metrics else ' | Foldseek: N/A'))

    print_metric_tables(methods, top_k, remote_depth)

    bootstrap_rows = None
    if args.bootstrap:
        method_per_query = {name: m['per_query'] for name, m in methods.items()
                            if name != 'Proteogram'}
        bootstrap_rows = report_bootstrap(proteogram_per_query, method_per_query, top_k,
                                          n_boot=args.n_boot, seed=args.boot_seed)

    train_counts = None
    if args.train_list:
        train_counts = {}
        n_unknown = 0
        with open(args.train_list) as f:
            for line in f:
                sid = line.strip()
                if not sid:
                    continue
                try:
                    sfam = '.'.join(scop.getDomainBySid(sid).sccs.split('.')[:3])
                    train_counts[sfam] = train_counts.get(sfam, 0) + 1
                except Exception:
                    n_unknown += 1
        if n_unknown:
            print(f'WARNING: {n_unknown} --train_list ID(s) not found in SCOPe.')

    written = write_run_files(run_dir, methods, query_info, triage_rows, bootstrap_rows,
                              scope_level)
    written.update(report_per_superfamily(methods, query_info, top_k, run_dir,
                                          n_boot=args.n_boot if args.bootstrap else None,
                                          seed=args.boot_seed, train_counts=train_counts))
    if args.query_types:
        qt = pd.read_csv(args.query_types, sep='\t')
        written.update(report_by_query_type(
            methods, query_info, dict(zip(qt['SCOPeID'], qt['query_type'])), top_k, run_dir,
            n_boot=args.n_boot if args.bootstrap else None, seed=args.boot_seed,
            train_counts=train_counts))

    finished = datetime.datetime.now()
    run_info = {
        'started': started.isoformat(timespec='seconds'),
        'finished': finished.isoformat(timespec='seconds'),
        'command': sys.argv,
        'args': vars(args),
        'git_commit': git_commit(),
        'top_k': top_k,
        'scope_level': scope_level,
        'ranked_list_depth': depths,
        'remote_depth': remote_depth,
        'remote_short_lists': {name: m.get('remote_short') for name, m in methods.items()},
        'inputs': {'config': os.path.abspath(args.config),
                   'scope_eval_set': scope_eval_set,
                   'train_list': args.train_list,
                   'query_types': args.query_types,
                   'proteogram_sim_results': proteogram_sim_results,
                   'gtalign_results_dir': gtalign_results_dir,
                   'usalign_results': usalign_results,
                   'foldseek_results': foldseek_results,
                   'search_images_dir': search_images_dir},
        'n_queries': {name: m['n_queries'] for name, m in methods.items()},
        'n_proteogram_queries_skipped': n_queries_skipped,
        'search_image_counts': triage_counts,
        'capped_precision_overflow': CAPPED_OVERFLOW[0],
        'files': {**written,
                  'config.yml': 'copy of the config.yml used for this run',
                  'report.txt': 'everything printed to the console during the run',
                  'scope_eval_set_labels.tsv': 'SCOPe labels of the evaluation set',
                  'search_images/': 'search images sorted into good/bad/no_relative',
                  'run_info.json': 'this file'},
    }
    with open(os.path.join(run_dir, 'run_info.json'), 'w') as f:
        json.dump(run_info, f, indent=2, default=str)
    print(f'\nResults written to {run_dir}')
    sys.stdout = sys.__stdout__
    report_fh.close()
