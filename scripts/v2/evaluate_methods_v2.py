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
           ('ratk', 'Recall@K', 'Wilcoxon signed-rank')]


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
        summarize_per_query() output: p1_*, patk_*, cpatk_*, map_* and ratk_*
        means for each level, plus n_queries and per_query (which also holds
        each query's top hits). cpatk (capped P@K) is NaN per query when R = 0
        and its mean ignores those.
    """
    if query_id_fn is None:
        query_id_fn = lambda x: x

    levels = LEVELS
    # Per-query metrics keyed by query_id (pdb_id_chain), for paired bootstrap.
    per_query = {}
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
        hit1 = {lv: 0.0 for lv in levels}  # top valid hit matches the query
        hits = []

        k = 0
        for target in results_df.iloc[i, 1:]:
            target_labels = {lv: lookup(label_df, 'pdb_id_chain', target, lv) for lv in levels}
            if any(v is None for v in target_labels.values()):
                continue
            k += 1
            hits.append(target)
            if k == 1:
                hit1 = {lv: float(query_labels[lv] == target_labels[lv]) for lv in levels}
            for lv in levels:
                if query_labels[lv] == target_labels[lv]:
                    tp[lv] += 1
                    prec[lv] += tp[lv] / k
            if k >= top_k:
                break

        per_query[query_id] = {'top_hits': ','.join(map(str, hits))}
        for lv in levels:
            per_query[query_id][f'p1_{lv}'] = hit1[lv]
            per_query[query_id][f'patk_{lv}'] = tp[lv] / top_k
            per_query[query_id][f'cpatk_{lv}'] = capped_precision(tp[lv], r[lv], top_k)
            per_query[query_id][f'map_{lv}'] = (
                prec[lv] / min(r[lv], top_k) if r[lv] > 0 else 0.0)
            per_query[query_id][f'ratk_{lv}'] = tp[lv] / r[lv] if r[lv] > 0 else 0.0

    if n_skipped:
        print(f'WARNING: {n_skipped} queries skipped (not found in label_df).')
    return summarize_per_query(per_query, n_queries=n_queries)


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
        if mkey != 'p1':
            mlabel += f' (K={top_k})'
        if mkey == 'cpatk':
            mlabel += ', R = 0 queries excluded'
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
                # Capped P@K is NaN where R = 0; R comes from the shared label
                # table, so both methods have NaN at the same queries.
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
                if mkey == 'p1':
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
        if mkey == 'cpatk':
            n_by_level = ', '.join(
                f'{lv} {sum(1 for q in common if not np.isnan(proteogram_per_query[q][f"cpatk_{lv}"]))}'
                for lv in levels)
            print(f'(paired on shared queries with R > 0: {n_by_level})')
        else:
            print(f'(paired on {len(common)} shared queries)')
        if discordant:
            print('Discordant pairs (Proteogram-only correct / method-only correct):')
            print('\n'.join(discordant))
    return rows


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


def print_metric_tables(methods, top_k):
    """Print one table per metric (rows = methods, columns = SCOPe levels).

    methods: OrderedDict {method name: summarize_per_query() output}.
    """
    hdr = f'{"Method":<15} | {"Class":>8} | {"Fold":>8} | {"Superfamily":>11} | {"Family":>8}'
    sep = '-' * len(hdr)
    for mkey, title, _ in METRICS:
        if mkey == 'p1':
            title += ' (top non-self hit correct)'
        elif mkey == 'cpatk':
            title += f' (K={top_k}; tp / min(R, K), queries with R = 0 excluded)'
        else:
            title += f' (K={top_k})'
        print(f'\n{title}')
        print(hdr)
        print(sep)
        for name, m in methods.items():
            print(f'{name:<15} | {m[f"{mkey}_class"]:>8.4f} | {m[f"{mkey}_fold"]:>8.4f} | '
                  f'{m[f"{mkey}_superfamily"]:>11.4f} | {m[f"{mkey}_family"]:>8.4f}')
        if mkey == 'cpatk':
            ref = methods.get('Proteogram') or next(reversed(methods.values()))
            print(f'{"(n with R > 0)":<15} | {ref["cpatk_n_class"]:>8d} | '
                  f'{ref["cpatk_n_fold"]:>8d} | {ref["cpatk_n_superfamily"]:>11d} | '
                  f'{ref["cpatk_n_family"]:>8d}')
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
        stats = (per_query_df.groupby([level, 'method'])
                 [[f'p1_{level}', f'cpatk_{level}', f'map_{level}']].mean().unstack('method'))
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


def read_gtalign_results(gtalign_results_dir, top_k):
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


def read_foldseek_results(foldseek_results, top_k):
    """Read Foldseek easy-search results into a pandas DataFrame.

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
        ordered_data.append([query_id] + filtered[:top_k])
    return pd.DataFrame(ordered_data)


def read_usalign_results(usalign_results, top_k):
    """
    Read USalign results into a pandas dataframe.
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
        tmp1 = sorted_inner_dict[:top_k]
        tmp2 = [key]
        tmp2.extend([prot for (prot,_) in tmp1])
        ordered_data.append(tmp2)

    return pd.DataFrame(ordered_data)
        
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluate proteogram vs. structural alignment methods.')
    parser.add_argument('--overwrite', action='store_true',
                        help='No longer needed: every run writes to a new timestamped folder '
                             '(see --out_dir). Accepted and ignored for older command lines.')
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
    shutil.copy('config.yml', os.path.join(run_dir, 'config.yml'))
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

    # Calculate Precision@K's and MAP@K's
    proteogram_res_df = pd.read_csv(proteogram_sim_results, sep='\t')
    n_queries_skipped = 0
    proteogram_evaluated_ids = set()
    proteogram_per_query = {}  # {query_id: {metric_level: value}} for bootstrap
    for i in range(proteogram_res_df.shape[0]):
        prot_file = image_stem(proteogram_res_df.iloc[i,0])
        query_fam   = lookup(label_df, 'proteogram_stem', prot_file, 'family')
        query_sfam  = lookup(label_df, 'proteogram_stem', prot_file, 'superfamily')
        query_fold  = lookup(label_df, 'proteogram_stem', prot_file, 'fold')
        query_class = lookup(label_df, 'proteogram_stem', prot_file, 'class')
        if any(v is None for v in (query_fam, query_sfam, query_fold, query_class)):
            n_queries_skipped += 1
            continue
        proteogram_evaluated_ids.add(prot_file)  # already an extension-less stem
        
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
        # Labels of the top valid (non-self) hit, for Precision@1
        first_hit = None
        hits = []  # (target id, {level: label}) for the valid top-K hits

        # Iterate through results and find matches for family, superfamily,
        # fold and class SCOPe levels
        k = 0
        for target in proteogram_res_df.iloc[i,1:]:
            target_file  = image_stem(target.split(',')[0])
            if target_file == prot_file:  # skip self-hit
                continue
            target_fam   = lookup(label_df, 'proteogram_stem', target_file, 'family')
            target_sfam  = lookup(label_df, 'proteogram_stem', target_file, 'superfamily')
            target_fold  = lookup(label_df, 'proteogram_stem', target_file, 'fold')
            target_class = lookup(label_df, 'proteogram_stem', target_file, 'class')
            if any(v is None for v in (target_fam, target_sfam, target_fold, target_class)):
                continue
            k += 1
            valid_results += 1
            hits.append((target_file, {'family': target_fam, 'superfamily': target_sfam,
                                       'fold': target_fold, 'class': target_class}))
            if k == 1:
                first_hit = hits[0][1]
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

        # Per-query metrics keyed by SCOPeID (== pdb_id_chain, so it pairs with
        # the alignment methods' per_query dicts). MAP@K and capped P@K
        # normalise by min(R, K), R = relevant items in the corpus; Recall@K by R.
        _qid = os.path.splitext(prot_file)[0]
        _tp = {'family': tp_fam, 'superfamily': tp_sfam, 'fold': tp_fold, 'class': tp_class}
        _prec = {'family': prec_at_k_fam, 'superfamily': prec_at_k_sfam,
                 'fold': prec_at_k_fold, 'class': prec_at_k_class}
        _r = {'family': r_fam, 'superfamily': r_sfam, 'fold': r_fold, 'class': r_class}
        _q = {'family': query_fam, 'superfamily': query_sfam,
              'fold': query_fold, 'class': query_class}
        query_info[_qid] = {**{lv: _q[lv] for lv in LEVELS},
                            **{f'R_{lv}': int(_r[lv]) for lv in LEVELS}}
        proteogram_per_query[_qid] = {'top_hits': ','.join(t for t, _ in hits)}
        for lv in LEVELS:
            proteogram_per_query[_qid][f'p1_{lv}'] = float(
                first_hit is not None and first_hit[lv] == _q[lv])
            proteogram_per_query[_qid][f'patk_{lv}'] = _tp[lv] / top_k
            proteogram_per_query[_qid][f'cpatk_{lv}'] = capped_precision(_tp[lv], _r[lv], top_k)
            proteogram_per_query[_qid][f'map_{lv}'] = (
                _prec[lv] / min(_r[lv], top_k) if _r[lv] > 0 else 0.0)
            proteogram_per_query[_qid][f'ratk_{lv}'] = (
                _tp[lv] / _r[lv] if _r[lv] > 0 else 0.0)

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

    # Calculate Precision@K's and MAP@K's
    gtalign_res_df = read_gtalign_results(gtalign_results_dir, top_k)
    print(f'GTalign DataFrame rows: {gtalign_res_df.shape[0]}')
    gtalign_res_df.to_csv(os.path.join(gtalign_results_dir, 'combined_gtalign_results.tsv'),
                          sep='\t', index=False)
    gtalign_metrics = calc_patk_mapk(
        gtalign_res_df, label_df, top_k,
        query_id_fn=lambda x: x.replace('.ent.out', '').replace('.out', '')[:7],
        limit_to_ids=proteogram_evaluated_ids)

    usalign_res_df = read_usalign_results(usalign_results, top_k)
    usalign_metrics = calc_patk_mapk(usalign_res_df, label_df, top_k,
                                     limit_to_ids=proteogram_evaluated_ids)

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
        foldseek_res_df = read_foldseek_results(foldseek_results, top_k)
        label_ids = set(label_df['pdb_id_chain'])
        fs_targets = [t for t in pd.unique(foldseek_res_df.iloc[:, 1:].values.ravel())
                      if isinstance(t, str)]
        n_matched = sum(1 for t in fs_targets if t in label_ids)
        print(f'Foldseek unique targets: {len(fs_targets)}, '
              f'found in label_df: {n_matched} ({100*n_matched/max(len(fs_targets),1):.1f}%)')
        foldseek_metrics = calc_patk_mapk(foldseek_res_df, label_df, top_k,
                                          limit_to_ids=proteogram_evaluated_ids)
    else:
        foldseek_metrics = None

    methods = OrderedDict([('GTalign', gtalign_metrics), ('USalign', usalign_metrics)])
    if foldseek_metrics:
        methods['Foldseek'] = foldseek_metrics
    methods['Proteogram'] = proteogram_metrics
    print('Proteins compared — ' + ' | '.join(
        f'{name}: {m["n_queries"]}' for name, m in methods.items())
          + ('' if foldseek_metrics else ' | Foldseek: N/A'))

    print_metric_tables(methods, top_k)

    bootstrap_rows = None
    if args.bootstrap:
        method_per_query = {name: m['per_query'] for name, m in methods.items()
                            if name != 'Proteogram'}
        bootstrap_rows = report_bootstrap(proteogram_per_query, method_per_query, top_k,
                                          n_boot=args.n_boot, seed=args.boot_seed)

    written = write_run_files(run_dir, methods, query_info, triage_rows, bootstrap_rows,
                              scope_level)

    finished = datetime.datetime.now()
    run_info = {
        'started': started.isoformat(timespec='seconds'),
        'finished': finished.isoformat(timespec='seconds'),
        'command': sys.argv,
        'args': vars(args),
        'git_commit': git_commit(),
        'top_k': top_k,
        'scope_level': scope_level,
        'inputs': {'config': os.path.abspath('config.yml'),
                   'scope_eval_set': scope_eval_set,
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
