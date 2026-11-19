"""
GO-based evaluation of structure search: Proteogram vs. GTalign, USalign, Foldseek.

Uses the same config.yml inputs as evaluate_methods_v2.py (top_k, scope_eval_set,
proteogram_sim_results, gtalign_results_dir, usalign_results, foldseek_results,
embed_file) plus:
    annot_file     annotation TSV from create_scope_annotation_file.py (the same
                   key that script writes to); --annot_file overrides it
    go_obo_file    go-basic.obo (downloaded if missing)
Results go to a new timestamped folder (go_eval_<stamp>_<source>_k<K>/) with a
report, TSVs, a copy of config.yml and run_info.json.

GO TERMS
    --go_source uniprot (default) reads GOEvidence_MF/BP (term=evidence codes,
    from the updated annotation scripts) and evaluates each --evidence level:
        all           every annotation
        curated       anything but IEA (electronic, mostly InterPro-family based)
        experimental  EXP/IDA/IPI/IMP/IGI/IEP and high-throughput HTP/HDA/HMP/HGI/HEP
    --go_source pdbe reads the older GOTerms_MF/BP columns (no evidence codes;
    'all' only). Each protein's terms are propagated to all ancestors (is_a +
    part_of) in go-basic and the aspect root is dropped. The corpus is the
    scope_eval_set proteins; term frequencies are counted over its annotated
    members.
    GO is per PDB chain, so domains on the same chain share identical terms:
    hits on the query's own chain are always skipped and counted.

1. RETRIEVAL (all methods). For each query with >= 1 specific term, the top-K
   hits that are annotated, not the query and not on its chain are scored:
      GO P@1       top scored hit shares a specific term with the query
      GO precision fraction of scored hits sharing a specific term
      GO capped P  shared / min(R, scored hits); R = annotated corpus proteins
                   sharing a specific term with the query
      Jaccard      mean Jaccard of propagated informative term sets (graded)
      simGIC       IC-weighted Jaccard (Pesquita et al. 2008), graded
   "Specific" = annotated to <= --specific_max_freq of the corpus (default 5%);
   "informative" = <= --max_term_freq (default 50%) -- so near-universal
   ancestors don't make every pair match. Baselines per query: a random
   annotated protein (R / N), all annotated members of the query's SCOPe
   superfamily, and -- the "beyond SCOPe" check -- hits OUTSIDE the query's
   superfamily vs. a random protein outside it.

2. NEIGHBOR-BASED GO PREDICTION (all methods; CAFA-style). Each term's score
   is the (uniform or 1/rank weighted) fraction of scored top-K hits annotated
   with it. Evaluated against the query's propagated terms with the CAFA
   protein-centric metrics over thresholds 0.01..1.00:
      Fmax  max F1; precision averaged over queries with >= 1 prediction at
            the threshold (coverage reported), recall over all queries
      Smin  min sqrt(ru^2 + mi^2), ru/mi = mean remaining uncertainty /
            misinformation weighted by information accretion
            IA(t) = -log2 P(t | parents(t)) (Clark & Radivojac 2013)
   plus term-centric AUPR for terms with >= --min_term_positives positive
   queries. Baselines: Naive (each term at its corpus frequency, leave-one-out)
   and SCOPe_superfamily (term frequency among the query's superfamily members).

3. EMBEDDING-GO CORRELATION (Proteogram only). Spearman correlation between
   embedding cosine similarity and GO similarity (simGIC, Jaccard) over
   protein pairs (all, or a seeded sample of --max_pairs), overall and within
   each SCOPe relationship (same family / same superfamily / same fold / same
   class / different class). A positive correlation WITHIN a SCOPe stratum
   means the embedding tracks function beyond the SCOPe label.

Usage (from scripts/v2, like evaluate_methods_v2.py):
    python evaluate_go_v2.py --bootstrap
    python evaluate_go_v2.py --go_source pdbe --evidence all
"""
import argparse
import datetime
import json
import math
import os
import pickle
import shutil
import sys
from collections import Counter, OrderedDict, defaultdict

import numpy as np
import pandas as pd
import requests
from goatools.obo_parser import GODag
from scipy import sparse
from scipy.stats import spearmanr
from sklearn.metrics import average_precision_score

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import evaluate_methods_v2 as ev  # noqa: E402  (results readers, stats, run-folder helpers)

from proteogram.common import read_yaml  # noqa: E402
from proteogram.common.go_evidence import EXPERIMENTAL_CODES, parse_go_evidence  # noqa: E402


ROOTS = {'MF': 'GO:0003674', 'BP': 'GO:0008150', 'CC': 'GO:0005575'}
NAMESPACES = {'MF': 'molecular_function', 'BP': 'biological_process',
              'CC': 'cellular_component'}
THRESHOLDS = np.round(np.arange(0.01, 1.0001, 0.01), 2)
SCOPE_COLS = OrderedDict([('family', 'SCOPeFamily'), ('superfamily', 'SCOPeSuperfamily'),
                          ('fold', 'SCOPeFold'), ('class', 'SCOPeClass')])
_GO_OBO_URL = 'http://current.geneontology.org/ontology/go-basic.obo'
RETRIEVAL_METRICS = [('p1', 'GO P@1'), ('precision', 'GO precision'),
                     ('capped', 'GO capped P'), ('jaccard', 'Jaccard'),
                     ('simgic', 'simGIC'), ('out_sfam_precision', 'Outside-sfam precision')]


# ----------------------------------------------------------------------------
# Inputs
# ----------------------------------------------------------------------------

def download_if_missing(url, path):
    if os.path.exists(path):
        return
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    print(f'Downloading {os.path.basename(path)} from Gene Ontology...')
    r = requests.get(url, timeout=300)
    r.raise_for_status()
    with open(path, 'wb') as fh:
        fh.write(r.content)


def read_id_list(path):
    """SCOPe IDs from a list file (first token per line, extension stripped)."""
    ids = []
    with open(path) as f:
        for line in f:
            tok = line.split()
            if tok:
                ids.append(os.path.splitext(os.path.basename(tok[0]))[0])
    return ids


def load_annotations(path, id_col):
    df = pd.read_csv(path, sep='\t', dtype=str, low_memory=False).fillna('')
    if id_col not in df.columns:
        hint = (' This looks like a create_pdb_annotation_file.py (per-chain) file; point '
                'annot_file / --annot_file at the create_scope_annotation_file.py output '
                'for the scope_eval_set proteins.' if 'MonomerID' in df.columns else '')
        sys.exit(f'ERROR: id column {id_col!r} not in {path}.{hint}')
    return df.drop_duplicates(id_col).set_index(id_col)


def raw_terms(annot, aspect, source, evidence):
    """{protein id: set of GO ids} for one aspect and evidence level (unpropagated)."""
    out = {}
    if source == 'pdbe':
        col = f'GOTerms_{aspect}'
        if col not in annot.columns:
            sys.exit(f'ERROR: column {col} not in the annotation file.')
        for pid, value in annot[col].items():
            terms = {t for t in value.split('|') if t}
            if terms:
                out[pid] = terms
        return out
    col = f'GOEvidence_{aspect}'
    if col not in annot.columns:
        sys.exit(f'ERROR: column {col} not in the annotation file -- regenerate it with the '
                 f'updated create_scope_annotation_file.py, or use --go_source pdbe.')
    for pid, value in annot[col].items():
        codes_by_term = parse_go_evidence(value)
        if evidence == 'experimental':
            terms = {t for t, c in codes_by_term.items() if c & EXPERIMENTAL_CODES}
        elif evidence == 'curated':
            terms = {t for t, c in codes_by_term.items() if c - {'IEA'}}
        else:
            terms = set(codes_by_term)
        if terms:
            out[pid] = terms
    return out


def rows_to_hits(df, query_fn=lambda x: x):
    """Reader DataFrame (col 0 query, cols 1+ targets) -> {query: [targets]}."""
    hits = {}
    for row in df.itertuples(index=False):
        q = query_fn(str(row[0]))
        hits[q] = [str(t) for t in row[1:] if isinstance(t, str) and t]
    return hits


def read_proteogram_hits(path, k):
    """{query: top-k targets} from the Proteogram results TSV, self-hit removed."""
    df = pd.read_csv(path, sep='\t')
    hits = {}
    for row in df.itertuples(index=False):
        q = ev.image_stem(row[0])
        targets = []
        for cell in row[1:]:
            if not isinstance(cell, str):
                continue
            t = ev.image_stem(cell.split(',')[0])
            if t != q:
                targets.append(t)
            if len(targets) >= k:
                break
        hits[q] = targets
    return hits


def load_embeddings(path):
    """{protein id: unit-norm numpy vector} from measure_similarity_v2.py's pickle."""
    import torch  # noqa: F401  (needed to unpickle the stored tensors)
    with open(path, 'rb') as f:
        data = pickle.load(f)
    out = {}
    for name, vec in data.items():
        arr = vec.detach().cpu().numpy() if hasattr(vec, 'detach') else np.asarray(vec)
        arr = arr.astype(np.float32).ravel()
        norm = np.linalg.norm(arr)
        if norm > 0:
            out[ev.image_stem(name)] = arr / norm
    return out


# ----------------------------------------------------------------------------
# GO terms for one aspect + evidence level over the corpus
# ----------------------------------------------------------------------------

def direct_parents(term, namespace):
    parents = {p.id for p in term.parents}
    for p in getattr(term, 'relationship', {}).get('part_of', ()):
        if p.namespace == namespace:
            parents.add(p.id)
    return parents


class GOAspect:
    """Propagated term sets, frequencies, IC and IA for one aspect over the corpus."""

    def __init__(self, dag, aspect, raw, corpus, max_term_freq, specific_max_freq):
        self.aspect = aspect
        ns, root = NAMESPACES[aspect], ROOTS[aspect]
        memo = {}

        def ancestors(go_id):
            if go_id in memo:
                return memo[go_id]
            term = dag[go_id]
            closure = {term.id}
            for parent in direct_parents(term, ns):
                closure |= ancestors(parent)
            memo[go_id] = closure
            return closure

        self.n_dropped_terms = 0
        prop = {}
        for pid in corpus:
            closure = set()
            for t in raw.get(pid, ()):
                term = dag.get(t)
                if term is None or term.namespace != ns:
                    self.n_dropped_terms += 1   # obsolete, unknown, or wrong aspect
                    continue
                closure |= ancestors(term.id)
            closure.discard(root)
            if closure:
                prop[pid] = frozenset(closure)
        self.prop = prop
        self.ids = sorted(prop)
        self.index = {p: i for i, p in enumerate(self.ids)}
        self.n = len(self.ids)
        self.counts = Counter(t for s in prop.values() for t in s)
        self.freq = {t: c / self.n for t, c in self.counts.items()} if self.n else {}
        self.name = {t: dag[t].name for t in self.counts}

        informative = sorted(t for t, f in self.freq.items() if f <= max_term_freq)
        specific = sorted(t for t, f in self.freq.items() if f <= specific_max_freq)
        self.I = self._matrix(informative)
        self.S = self._matrix(specific)
        ic = np.array([-math.log2(self.freq[t]) for t in informative], dtype=float)
        self.I_ic = self.I.multiply(ic).tocsr() if len(informative) else self.I
        self.size = np.asarray(self.I.sum(axis=1)).ravel()
        self.weight = np.asarray(self.I_ic.sum(axis=1)).ravel()
        self.n_specific = np.asarray(self.S.sum(axis=1)).ravel()

        # Information accretion, IA(t) = -log2 P(t | all direct parents), from the corpus.
        term_prots = defaultdict(set)
        for p, s in prop.items():
            for t in s:
                term_prots[t].add(self.index[p])
        everyone = set(range(self.n))
        self.ia = {}
        for t, c in self.counts.items():
            parent_sets = [everyone if pa == root else term_prots.get(pa, set())
                           for pa in direct_parents(dag[t], ns)]
            n_par = len(set.intersection(*parent_sets)) if parent_sets else self.n
            self.ia[t] = -math.log2(c / n_par) if n_par else 0.0

    def _matrix(self, terms):
        col = {t: j for j, t in enumerate(terms)}
        rows, cols = [], []
        for p in self.ids:
            for t in self.prop[p]:
                j = col.get(t)
                if j is not None:
                    rows.append(self.index[p])
                    cols.append(j)
        return sparse.csr_matrix((np.ones(len(rows)), (rows, cols)),
                                 shape=(self.n, len(terms)))

    def similarities(self, qi):
        """Query row qi vs. every corpus protein: Jaccard, simGIC, shares-a-specific-term."""
        inter = np.asarray((self.I @ self.I[qi].T).todense()).ravel()
        union = self.size[qi] + self.size - inter
        jacc = np.divide(inter, union, out=np.zeros_like(inter), where=union > 0)
        inter_w = np.asarray((self.I_ic @ self.I[qi].T).todense()).ravel()
        union_w = self.weight[qi] + self.weight - inter_w
        gic = np.divide(inter_w, union_w, out=np.zeros_like(inter_w), where=union_w > 0)
        shared = np.asarray((self.S @ self.S[qi].T).todense()).ravel() > 0
        return jacc, gic, shared


# ----------------------------------------------------------------------------
# 1. Retrieval
# ----------------------------------------------------------------------------

def scored_hits(q, hits, go, chain_of, k):
    """Indices of a query's top-k hits that are annotated, not itself, not on its chain."""
    scored, n_unannotated, n_same_chain = [], 0, 0
    q_chain = chain_of.get(q)
    for h in hits[:k]:
        if h == q:
            continue
        if q_chain and chain_of.get(h) == q_chain:
            n_same_chain += 1
            continue
        hi = go.index.get(h)
        if hi is None:
            n_unannotated += 1
            continue
        scored.append(hi)
    return scored, n_unannotated, n_same_chain


def evaluate_retrieval(go, queries, method_hits, chain_of, sfam_of, k):
    chain_arr = np.array([chain_of.get(p, '') for p in go.ids], dtype=object)
    sfam_arr = np.array([sfam_of.get(p, '') for p in go.ids], dtype=object)
    rows = []
    for q in queries:
        qi = go.index.get(q)
        if qi is None or go.n_specific[qi] == 0:
            continue   # no specific term: relevance undefined for this query
        jacc, gic, shared = go.similarities(qi)
        others = np.ones(go.n, dtype=bool)
        others[qi] = False
        if chain_of.get(q):
            others &= chain_arr != chain_of[q]
        n_others = int(others.sum())
        r_total = int((shared & others).sum())
        q_sfam = sfam_of.get(q, '')
        same_sfam = others & (sfam_arr == q_sfam) if q_sfam else np.zeros(go.n, bool)
        outside = others & (sfam_arr != q_sfam) if q_sfam else np.zeros(go.n, bool)
        base = {
            'query_id': q, 'superfamily': q_sfam, 'R': r_total, 'n_corpus_others': n_others,
            'random_precision': r_total / n_others if n_others else np.nan,
            'random_jaccard': jacc[others].mean() if n_others else np.nan,
            'random_simgic': gic[others].mean() if n_others else np.nan,
            'same_sfam_precision': shared[same_sfam].mean() if same_sfam.any() else np.nan,
            'same_sfam_jaccard': jacc[same_sfam].mean() if same_sfam.any() else np.nan,
            'random_out_sfam_precision': shared[outside].mean() if outside.any() else np.nan,
        }
        for method, hits in method_hits.items():
            scored, n_unann, n_chain = scored_hits(q, hits.get(q, []), go, chain_of, k)
            row = {**base, 'method': method, 'n_scored_hits': len(scored),
                   'n_unannotated_hits': n_unann, 'n_same_chain_hits': n_chain}
            if scored:
                rel = shared[scored]
                out_hits = [h for h in scored if q_sfam and sfam_arr[h] != q_sfam]
                row.update(
                    p1=float(rel[0]), precision=float(rel.mean()),
                    capped=float(rel.sum() / min(r_total, len(scored))) if r_total else np.nan,
                    jaccard=float(jacc[scored].mean()), simgic=float(gic[scored].mean()),
                    n_out_sfam_hits=len(out_hits),
                    out_sfam_precision=float(shared[out_hits].mean()) if out_hits else np.nan)
            else:
                row.update(p1=np.nan, precision=np.nan, capped=np.nan, jaccard=np.nan,
                           simgic=np.nan, n_out_sfam_hits=0, out_sfam_precision=np.nan)
            rows.append(row)
    df = pd.DataFrame(rows)
    front = ['method', 'query_id', 'superfamily']
    return df[front + [c for c in df.columns if c not in front]] if not df.empty else df


def summarize_retrieval(per_query):
    out = []
    for method, g in per_query.groupby('method', sort=False):
        scored = g[g['n_scored_hits'] > 0]
        row = {'method': method, 'n_queries': len(g), 'n_queries_with_scored_hits': len(scored),
               'mean_scored_hits': g['n_scored_hits'].mean(),
               'unannotated_hits_skipped': int(g['n_unannotated_hits'].sum()),
               'same_chain_hits_skipped': int(g['n_same_chain_hits'].sum())}
        for col in ('p1', 'precision', 'capped', 'jaccard', 'simgic', 'out_sfam_precision'):
            row[col] = scored[col].mean()
        row['n_out_sfam_hits'] = int(g['n_out_sfam_hits'].sum())
        # Baselines over the queries this method scored (paired with its metrics).
        for col in ('random_precision', 'random_jaccard', 'random_simgic',
                    'same_sfam_precision', 'same_sfam_jaccard', 'random_out_sfam_precision'):
            row[col] = scored[col].mean()
        out.append(row)
    return pd.DataFrame(out)


def bootstrap_retrieval(per_query, n_boot, seed):
    rows = []
    prot = per_query[per_query['method'] == 'Proteogram'].set_index('query_id')
    for method in per_query['method'].unique():
        if method == 'Proteogram':
            continue
        other = per_query[per_query['method'] == method].set_index('query_id')
        common = prot.index.intersection(other.index)
        for metric, _ in RETRIEVAL_METRICS:
            a, b = prot.loc[common, metric].to_numpy(float), other.loc[common, metric].to_numpy(float)
            ok = ~(np.isnan(a) | np.isnan(b))
            a, b = a[ok], b[ok]
            row = {'method': method, 'metric': metric, 'n_pairs': int(len(a))}
            if len(a) == 0:
                rows.append({**row, 'sig': 'n/a'})
                continue
            d, lo, hi = ev.paired_bootstrap_delta(a, b, n_boot=n_boot, seed=seed)
            if metric == 'p1':
                p, bb, cc = ev.mcnemar_pvalue(a, b)
                row.update(test='McNemar exact', proteogram_only_correct=bb, method_only_correct=cc)
            else:
                p = ev.wilcoxon_pvalue(a, b)
                row['test'] = 'Wilcoxon signed-rank'
            row.update(delta=d, ci_low=lo, ci_high=hi,
                       p_value=np.nan if p is None else p, sig=ev._sig_stars(p),
                       proteogram_better=bool(d > 0))   # all retrieval metrics: higher is better
            rows.append(row)
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------
# 2. Neighbor-based GO prediction (CAFA-style)
# ----------------------------------------------------------------------------

def knn_predictions(go, queries, hits_by_query, chain_of, k, weighting):
    preds = {}
    for q in queries:
        scored, _, _ = scored_hits(q, hits_by_query.get(q, []), go, chain_of, k)
        if not scored:
            preds[q] = {}
            continue
        weights = [1.0 / (r + 1) if weighting == 'rank' else 1.0 for r in range(len(scored))]
        total = sum(weights)
        scores = defaultdict(float)
        for w, hi in zip(weights, scored):
            for t in go.prop[go.ids[hi]]:
                scores[t] += w / total
        preds[q] = dict(scores)
    return preds


def naive_predictions(go, queries, min_score):
    """Each term at its corpus frequency, leaving the query out."""
    n = go.n
    common = {t: c for t, c in go.counts.items() if c / n >= min_score / 2}
    preds = {}
    for q in queries:
        own = go.prop[q]
        preds[q] = {t: (c - (t in own)) / (n - 1) for t, c in common.items() if n > 1}
    return preds


def superfamily_predictions(go, queries, chain_of, sfam_of):
    """Term frequency among the query's annotated superfamily members (not itself/its chain)."""
    members = defaultdict(list)
    for p in go.ids:
        if sfam_of.get(p):
            members[sfam_of[p]].append(p)
    preds = {}
    for q in queries:
        mates = [p for p in members.get(sfam_of.get(q, ''), [])
                 if p != q and not (chain_of.get(q) and chain_of.get(p) == chain_of.get(q))]
        scores = defaultdict(float)
        for p in mates:
            for t in go.prop[p]:
                scores[t] += 1.0 / len(mates)
        preds[q] = dict(scores)
    return preds


def threshold_curves(go, queries, preds):
    """Per query x threshold: precision (NaN if nothing predicted), recall, ru, mi."""
    nq, nt = len(queries), len(THRESHOLDS)
    prec = np.full((nq, nt), np.nan)
    rec = np.zeros((nq, nt))
    ru = np.zeros((nq, nt))
    mi = np.zeros((nq, nt))
    for i, q in enumerate(queries):
        truth = go.prop[q]
        truth_ia = sum(go.ia.get(t, 0.0) for t in truth)
        pred = {t: s for t, s in preds.get(q, {}).items() if s >= THRESHOLDS[0] - 1e-9}
        if not pred:
            ru[i] = truth_ia
            continue
        terms = list(pred)
        s = np.array([pred[t] for t in terms])
        in_truth = np.array([t in truth for t in terms])
        ia = np.array([go.ia.get(t, 0.0) for t in terms])
        mask = s[None, :] >= THRESHOLDS[:, None] - 1e-9
        tp = (mask & in_truth).sum(axis=1)
        npred = mask.sum(axis=1)
        prec[i] = np.where(npred > 0, tp / np.maximum(npred, 1), np.nan)
        rec[i] = tp / len(truth)
        mi[i] = (mask & ~in_truth) @ ia
        ru[i] = truth_ia - (mask & in_truth) @ ia
    return prec, rec, ru, mi


def fmax_smin(curves, idx=None):
    prec, rec, ru, mi = curves
    if idx is not None:
        prec, rec, ru, mi = prec[idx], rec[idx], ru[idx], mi[idx]
    has_pred = ~np.isnan(prec)
    n_pred = has_pred.sum(axis=0)
    coverage = n_pred / len(prec)
    with np.errstate(invalid='ignore', divide='ignore'):
        # Precision averaged over queries with >= 1 prediction at each threshold.
        p = np.where(n_pred > 0, np.nansum(prec, axis=0) / np.maximum(n_pred, 1), np.nan)
        r = rec.mean(axis=0)
        f = np.where((p + r) > 0, 2 * p * r / (p + r), 0.0)
    f = np.nan_to_num(f, nan=0.0)
    best = int(np.argmax(f))
    s = np.sqrt(ru.mean(axis=0) ** 2 + mi.mean(axis=0) ** 2)
    best_s = int(np.argmin(s))
    return {'fmax': float(f[best]), 'fmax_threshold': float(THRESHOLDS[best]),
            'precision_at_fmax': float(p[best]) if not np.isnan(p[best]) else np.nan,
            'recall_at_fmax': float(r[best]), 'coverage_at_fmax': float(coverage[best]),
            'smin': float(s[best_s]), 'smin_threshold': float(THRESHOLDS[best_s])}


def term_aupr(go, queries, preds_by_method, min_positives):
    positives = Counter(t for q in queries for t in go.prop[q])
    terms = [t for t, c in positives.items() if c >= min_positives]
    rows = []
    for t in terms:
        y = np.array([t in go.prop[q] for q in queries])
        if y.all():
            continue
        row = {'go_id': t, 'name': go.name.get(t, ''), 'n_positive_queries': int(y.sum()),
               'corpus_freq': go.freq[t]}
        for method, preds in preds_by_method.items():
            s = np.array([preds.get(q, {}).get(t, 0.0) for q in queries])
            row[f'{method}_aupr'] = average_precision_score(y, s)
        rows.append(row)
    return pd.DataFrame(rows)


def evaluate_prediction(go, queries, method_hits, chain_of, sfam_of, k, weighting,
                        min_positives, n_boot, seed, do_bootstrap):
    preds = OrderedDict()
    for method, hits in method_hits.items():
        preds[method] = knn_predictions(go, queries, hits, chain_of, k, weighting)
    preds['Naive'] = naive_predictions(go, queries, THRESHOLDS[0])
    if any(sfam_of.values()):
        preds['SCOPe_superfamily'] = superfamily_predictions(go, queries, chain_of, sfam_of)
    curves = {m: threshold_curves(go, queries, p) for m, p in preds.items()}
    per_term = term_aupr(go, queries, preds, min_positives)

    summary = []
    for method in preds:
        row = {'method': method, 'n_queries': len(queries), **fmax_smin(curves[method])}
        col = f'{method}_aupr'
        row['macro_aupr'] = per_term[col].mean() if col in per_term else np.nan
        row['n_terms_aupr'] = len(per_term)
        summary.append(row)
    summary = pd.DataFrame(summary)

    boot_rows = []
    if do_bootstrap and 'Proteogram' in curves and len(queries) > 1:
        rng = np.random.default_rng(seed)
        idx = rng.integers(0, len(queries), size=(n_boot, len(queries)))
        samples = {m: [fmax_smin(c, i) for i in idx] for m, c in curves.items()}
        for metric in ('fmax', 'smin'):
            for method in preds:
                vals = np.array([s[metric] for s in samples[method]])
                summary.loc[summary['method'] == method, f'{metric}_ci_low'] = np.percentile(vals, 2.5)
                summary.loc[summary['method'] == method, f'{metric}_ci_high'] = np.percentile(vals, 97.5)
                if method == 'Proteogram':
                    continue
                prot = np.array([s[metric] for s in samples['Proteogram']])
                delta = prot - vals
                observed = (summary.loc[summary['method'] == 'Proteogram', metric].iloc[0]
                            - summary.loc[summary['method'] == method, metric].iloc[0])
                # Two-sided bootstrap p-value: how often the resampled delta crosses 0.
                p = min(1.0, 2 * min((delta <= 0).mean(), (delta >= 0).mean()))
                boot_rows.append({'method': method, 'metric': metric, 'delta': observed,
                                  'ci_low': np.percentile(delta, 2.5),
                                  'ci_high': np.percentile(delta, 97.5),
                                  'bootstrap_p': p, 'sig': ev._sig_stars(p),
                                  # Fmax: higher is better; Smin: lower is better.
                                  'proteogram_better': bool(observed > 0 if metric == 'fmax'
                                                            else observed < 0),
                                  'n_queries': len(queries)})
    return summary, per_term, pd.DataFrame(boot_rows)


# ----------------------------------------------------------------------------
# 3. Embedding-GO correlation (Proteogram)
# ----------------------------------------------------------------------------

def scope_relationships(ids, i, j, meta):
    """Closest shared SCOPe level for each pair (i[n], j[n]) of indices into ids."""
    relation = np.full(len(i), 'different class', dtype=object)
    assigned = np.zeros(len(i), dtype=bool)
    for level, col in SCOPE_COLS.items():   # finest first
        if col not in meta.columns:
            continue
        labels = meta.loc[ids, col].to_numpy(dtype=object)
        same = (labels[i] == labels[j]) & (labels[i] != '') & ~assigned
        relation[same] = f'same {level}'
        assigned |= same
    return relation


def embedding_correlation(go, emb, meta, chain_of, max_pairs, seed, chunk=50000):
    ids = [p for p in go.ids if p in emb]
    n = len(ids)
    if n < 3:
        return pd.DataFrame(), 0
    rng = np.random.default_rng(seed)
    total = n * (n - 1) // 2
    if total <= max_pairs:
        i, j = np.triu_indices(n, 1)
    else:
        draw = int(max_pairs * 1.3) + 10
        i, j = rng.integers(0, n, draw), rng.integers(0, n, draw)
        keep = i < j
        pairs = np.unique(np.stack([i[keep], j[keep]], axis=1), axis=0)
        pairs = pairs[rng.permutation(len(pairs))[:max_pairs]]
        i, j = pairs[:, 0], pairs[:, 1]
    chains = np.array([chain_of.get(p, '') for p in ids], dtype=object)
    keep = ~((chains[i] != '') & (chains[i] == chains[j]))
    i, j = i[keep], j[keep]

    E = np.stack([emb[p] for p in ids])
    gidx = np.array([go.index[p] for p in ids])
    cos, jac, gic = [], [], []
    for s in range(0, len(i), chunk):
        a, b = i[s:s + chunk], j[s:s + chunk]
        cos.append(np.einsum('ij,ij->i', E[a], E[b]))
        A, B = go.I[gidx[a]], go.I[gidx[b]]
        inter = np.asarray(A.multiply(B).sum(axis=1)).ravel()
        union = go.size[gidx[a]] + go.size[gidx[b]] - inter
        jac.append(np.divide(inter, union, out=np.zeros_like(inter), where=union > 0))
        inter_w = np.asarray(go.I_ic[gidx[a]].multiply(B).sum(axis=1)).ravel()
        union_w = go.weight[gidx[a]] + go.weight[gidx[b]] - inter_w
        gic.append(np.divide(inter_w, union_w, out=np.zeros_like(inter_w), where=union_w > 0))
    cos, jac, gic = np.concatenate(cos), np.concatenate(jac), np.concatenate(gic)
    relation = scope_relationships(ids, i, j, meta)

    rows = []
    strata = ['all'] + [f'same {lv}' for lv in SCOPE_COLS] + ['different class']
    for stratum in strata:
        mask = np.ones(len(cos), bool) if stratum == 'all' else relation == stratum
        m = int(mask.sum())
        row = {'stratum': stratum, 'n_pairs': m}
        if m >= 10:
            for name, go_sim in (('simgic', gic), ('jaccard', jac)):
                if np.ptp(cos[mask]) == 0 or np.ptp(go_sim[mask]) == 0:
                    row.update({f'rho_{name}': np.nan, f'p_{name}': np.nan})
                    continue   # constant input: correlation undefined
                rho, p = spearmanr(cos[mask], go_sim[mask])
                # Approximate 95% CI for Spearman rho (Fieller et al.: se = 1.06/sqrt(n-3)).
                se = 1.06 / math.sqrt(m - 3)
                z = math.atanh(max(min(rho, 0.999999), -0.999999)) if not np.isnan(rho) else np.nan
                row.update({f'rho_{name}': rho, f'p_{name}': p,
                            f'rho_{name}_ci_low': math.tanh(z - 1.96 * se) if not np.isnan(z) else np.nan,
                            f'rho_{name}_ci_high': math.tanh(z + 1.96 * se) if not np.isnan(z) else np.nan})
            row.update(mean_cosine=float(cos[mask].mean()), mean_simgic=float(gic[mask].mean()),
                       mean_jaccard=float(jac[mask].mean()))
        rows.append(row)
    return pd.DataFrame(rows), n


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------

def print_table(df, cols, title, fmt='{:.4f}'):
    print(f'\n{title}')
    if df.empty:
        print('  (nothing to report)')
        return
    widths = {c: max(len(c), 10) for c in cols}
    print('  ' + ' | '.join(f'{c:>{widths[c]}}' for c in cols))
    for _, r in df.iterrows():
        cells = []
        for c in cols:
            v = r.get(c, '')
            if isinstance(v, (float, np.floating)):
                v = 'n/a' if np.isnan(v) else fmt.format(v)
            cells.append(f'{str(v):>{widths[c]}}')
        print('  ' + ' | '.join(cells))


def main():
    parser = argparse.ArgumentParser(description='GO-based evaluation of structure search methods.')
    parser.add_argument('--annot_file', default=None,
                        help='Annotation TSV from create_scope_annotation_file.py, covering '
                             'the scope_eval_set proteins (default: annot_file in config.yml).')
    parser.add_argument('--go_source', choices=['uniprot', 'pdbe'], default='uniprot',
                        help="'uniprot': GOEvidence_* columns with evidence codes (default); "
                             "'pdbe': older GOTerms_* columns, no evidence codes.")
    parser.add_argument('--evidence', default='all,experimental',
                        help='Comma-separated evidence levels to evaluate (all, curated, '
                             'experimental); each is a separate analysis. Default: '
                             'all,experimental. --go_source pdbe supports only all.')
    parser.add_argument('--aspects', default='MF,BP', help='GO aspects. Default: MF,BP.')
    parser.add_argument('--k', type=int, default=None,
                        help='Hits per query to use (default: top_k in config.yml). Must not '
                             'exceed the depth of the alignment results files.')
    parser.add_argument('--methods', default=None,
                        help='Comma-separated subset of Proteogram,GTalign,USalign,Foldseek '
                             '(default: all with results configured).')
    parser.add_argument('--max_term_freq', type=float, default=0.5,
                        help='Terms annotated to more than this fraction of the corpus are '
                             'left out of Jaccard/simGIC. Default: 0.5.')
    parser.add_argument('--specific_max_freq', type=float, default=0.05,
                        help='A hit is relevant if it shares a term annotated to at most this '
                             'fraction of the corpus. Default: 0.05.')
    parser.add_argument('--knn_weight', choices=['uniform', 'rank'], default='uniform',
                        help='Neighbor weighting for GO prediction. Default: uniform.')
    parser.add_argument('--min_term_positives', type=int, default=10,
                        help='Minimum positive queries for a term to get an AUPR. Default: 10.')
    parser.add_argument('--max_pairs', type=int, default=200000,
                        help='Pairs sampled for the embedding-GO correlation. Default: 200000.')
    parser.add_argument('--no_embedding', action='store_true',
                        help='Skip the embedding-GO correlation.')
    parser.add_argument('--exclude_classes', '-x', default=None,
                        help='Comma-separated SCOPe classes to leave out of the corpus.')
    parser.add_argument('--id_column', default='SCOPeID',
                        help='ID column in the annotation file. Default: SCOPeID.')
    parser.add_argument('--bootstrap', action='store_true',
                        help='Paired bootstrap CIs and tests (Proteogram - each method).')
    parser.add_argument('--n_boot', type=int, default=1000, help='Bootstrap resamples. Default: 1000.')
    parser.add_argument('--boot_seed', type=int, default=0, help='Bootstrap/sampling seed.')
    parser.add_argument('--out_dir', default=None,
                        help='Base folder for runs (default: eval_output_dir in config.yml, '
                             'else eval_runs/ next to save_bad_searches_dir or the '
                             'Proteogram results).')
    args = parser.parse_args()

    config = read_yaml('config.yml')
    k = args.k or config['top_k']
    annot_file = args.annot_file or config.get('annot_file')
    if not annot_file:
        sys.exit('ERROR: pass --annot_file or set annot_file in config.yml.')
    annot_source = '--annot_file' if args.annot_file else 'annot_file in config.yml'
    evidence_levels = [e.strip() for e in args.evidence.split(',') if e.strip()]
    if args.go_source == 'pdbe' and evidence_levels != ['all']:
        print('Note: --go_source pdbe has no evidence codes; evaluating evidence level "all" only.')
        evidence_levels = ['all']
    bad = set(evidence_levels) - {'all', 'curated', 'experimental'}
    if bad:
        sys.exit(f'ERROR: unknown --evidence level(s): {sorted(bad)}')
    aspects = [a.strip().upper() for a in args.aspects.split(',') if a.strip()]
    if args.specific_max_freq > args.max_term_freq:
        sys.exit('ERROR: --specific_max_freq must be <= --max_term_freq.')

    proteogram_sim_results = config['proteogram_sim_results']
    base_out = args.out_dir or config.get('eval_output_dir') or os.path.join(
        os.path.dirname(os.path.normpath(
            config.get('save_bad_searches_dir') or os.path.abspath(proteogram_sim_results))),
        'eval_runs')
    run_dir = ev.make_run_dir(base_out, args.go_source, k, prefix='go_eval')
    shutil.copy('config.yml', os.path.join(run_dir, 'config.yml'))
    report_fh = open(os.path.join(run_dir, 'report.txt'), 'w')
    sys.stdout = ev.Tee(sys.__stdout__, report_fh)
    started = datetime.datetime.now()
    print(f'GO evaluation run folder: {run_dir}')
    print(f'Started {started:%Y-%m-%d %H:%M:%S}; command: {" ".join(sys.argv)}')
    print(f'K={k}, GO source={args.go_source}, evidence={evidence_levels}, aspects={aspects}')
    print(f'Annotation file ({annot_source}): {annot_file}')

    # ---- Inputs ----
    go_obo = config.get('go_obo_file', '../data/go/go-basic.obo')
    download_if_missing(_GO_OBO_URL, go_obo)
    dag = GODag(go_obo, optional_attrs={'relationship'}, prt=None)
    annot = load_annotations(annot_file, args.id_column)
    corpus = read_id_list(config['scope_eval_set'])
    missing = [p for p in corpus if p not in annot.index]
    corpus = [p for p in corpus if p in annot.index]
    if args.exclude_classes and 'SCOPeClass' in annot.columns:
        excluded = {c.strip() for c in args.exclude_classes.split(',')}
        corpus = [p for p in corpus if annot.at[p, 'SCOPeClass'] not in excluded]
    print(f'Corpus: {len(corpus)} scope_eval_set proteins in the annotation file '
          f'({len(missing)} not in it).')
    chain_of = annot['PDBAndChainId'].to_dict() if 'PDBAndChainId' in annot.columns else {}
    if not chain_of:
        print('WARNING: no PDBAndChainId column; same-chain hits cannot be excluded.')
    sfam_of = annot['SCOPeSuperfamily'].to_dict() if 'SCOPeSuperfamily' in annot.columns else {}

    wanted = None if not args.methods else {m.strip() for m in args.methods.split(',')}
    method_hits = OrderedDict()
    if wanted is None or 'Proteogram' in wanted:
        method_hits['Proteogram'] = read_proteogram_hits(proteogram_sim_results, k)
    if (wanted is None or 'GTalign' in wanted) and config.get('gtalign_results_dir'):
        method_hits['GTalign'] = rows_to_hits(
            ev.read_gtalign_results(config['gtalign_results_dir'], k),
            lambda x: x.replace('.ent.out', '').replace('.out', '')[:7])
    if (wanted is None or 'USalign' in wanted) and config.get('usalign_results'):
        method_hits['USalign'] = rows_to_hits(ev.read_usalign_results(config['usalign_results'], k))
    if (wanted is None or 'Foldseek' in wanted) and config.get('foldseek_results'):
        method_hits['Foldseek'] = rows_to_hits(ev.read_foldseek_results(config['foldseek_results'], k))
    for m, h in method_hits.items():
        depth = np.mean([len(v) for v in h.values()]) if h else 0
        print(f'{m}: {len(h)} queries with results (mean {depth:.1f} hits kept, K={k})')
        if h and depth < k * 0.9:
            print(f'WARNING: {m} results are shallower than K={k}; lower --k for a fair comparison.')
    corpus_set = set(corpus)
    queries = sorted(corpus_set.intersection(*[set(h) for h in method_hits.values()]))
    print(f'Queries with results from every method and in the corpus: {len(queries)}')

    embeddings = None
    if not args.no_embedding and 'Proteogram' in method_hits and config.get('embed_file'):
        try:
            embeddings = load_embeddings(config['embed_file'])
            print(f'Embeddings: {len(embeddings)} loaded from {config["embed_file"]}')
        except Exception as e:
            print(f'WARNING: could not load embeddings ({e}); skipping the correlation.')

    # ---- Analyses per evidence level and aspect ----
    tables = defaultdict(list)
    coverage_rows = []
    for evidence in evidence_levels:
        for aspect in aspects:
            tag = f'{aspect} / {evidence}'
            raw = raw_terms(annot, aspect, args.go_source, evidence)
            go = GOAspect(dag, aspect, raw, corpus, args.max_term_freq, args.specific_max_freq)
            q_annot = [q for q in queries if q in go.index]
            q_specific = [q for q in q_annot if go.n_specific[go.index[q]] > 0]
            coverage_rows.append({
                'evidence': evidence, 'aspect': aspect, 'corpus_size': len(corpus),
                'corpus_annotated': go.n, 'queries': len(queries), 'queries_annotated': len(q_annot),
                'queries_with_specific_term': len(q_specific),
                'distinct_terms_propagated': len(go.counts),
                'informative_terms': go.I.shape[1], 'specific_terms': go.S.shape[1],
                'median_terms_per_protein': float(np.median([len(s) for s in go.prop.values()]))
                if go.n else np.nan,
                'terms_dropped_unknown_or_obsolete': go.n_dropped_terms})
            print('\n' + '=' * 78)
            print(f'{tag}: {go.n}/{len(corpus)} corpus proteins annotated; '
                  f'{len(q_annot)} queries annotated, {len(q_specific)} with a specific term')
            print('=' * 78)
            if go.n < 3 or not q_annot:
                print('Too few annotated proteins; skipping.')
                continue
            label = {'evidence': evidence, 'aspect': aspect}

            per_query = evaluate_retrieval(go, q_specific, method_hits, chain_of, sfam_of, k)
            if not per_query.empty:
                summary = summarize_retrieval(per_query)
                print_table(summary, ['method', 'n_queries_with_scored_hits', 'mean_scored_hits',
                                      'p1', 'precision', 'capped', 'jaccard', 'simgic',
                                      'random_precision', 'same_sfam_precision',
                                      'out_sfam_precision', 'random_out_sfam_precision'],
                            f'GO retrieval ({tag})')
                tables['go_retrieval_summary'].append(summary.assign(**label))
                tables['go_retrieval_per_query'].append(per_query.assign(**label))
                if args.bootstrap and 'Proteogram' in method_hits:
                    boot = bootstrap_retrieval(per_query, args.n_boot, args.boot_seed)
                    print_table(boot, ['method', 'metric', 'n_pairs', 'delta', 'ci_low',
                                       'ci_high', 'p_value', 'sig', 'proteogram_better'],
                                f'GO retrieval, Proteogram - method ({tag})')
                    tables['go_retrieval_bootstrap'].append(boot.assign(**label))

            pred_summary, per_term, pred_boot = evaluate_prediction(
                go, q_annot, method_hits, chain_of, sfam_of, k, args.knn_weight,
                args.min_term_positives, args.n_boot, args.boot_seed, args.bootstrap)
            print_table(pred_summary, ['method', 'n_queries', 'fmax', 'fmax_threshold',
                                       'coverage_at_fmax', 'smin', 'macro_aupr', 'n_terms_aupr'],
                        f'GO prediction from neighbors ({tag})')
            tables['go_prediction_summary'].append(pred_summary.assign(**label))
            if not per_term.empty:
                tables['go_prediction_per_term'].append(per_term.assign(**label))
            if not pred_boot.empty:
                print_table(pred_boot, ['method', 'metric', 'delta', 'ci_low', 'ci_high',
                                        'bootstrap_p', 'sig', 'proteogram_better'],
                            f'GO prediction, Proteogram - method ({tag}; Smin: lower is better)')
                tables['go_prediction_bootstrap'].append(pred_boot.assign(**label))

            if embeddings:
                corr, n_emb = embedding_correlation(go, embeddings, annot, chain_of,
                                                    args.max_pairs, args.boot_seed)
                if not corr.empty:
                    print_table(corr, ['stratum', 'n_pairs', 'rho_simgic', 'rho_simgic_ci_low',
                                       'rho_simgic_ci_high', 'rho_jaccard', 'mean_cosine',
                                       'mean_simgic'],
                                f'Proteogram embedding cosine vs GO similarity ({tag}; '
                                f'{n_emb} proteins)')
                    tables['go_embedding_correlation'].append(
                        corr.assign(n_proteins=n_emb, **label))

    # ---- Write files ----
    descriptions = {
        'go_annotation_coverage': 'corpus/query annotation coverage per aspect and evidence level',
        'go_retrieval_summary': 'GO retrieval metrics and baselines per method',
        'go_retrieval_per_query': 'GO retrieval metrics per query and method',
        'go_retrieval_bootstrap': 'paired bootstrap Proteogram - method for GO retrieval',
        'go_prediction_summary': 'neighbor-based GO prediction: Fmax, Smin, macro AUPR (+CIs)',
        'go_prediction_per_term': 'per-term AUPR for every method and baseline',
        'go_prediction_bootstrap': 'paired bootstrap Proteogram - method for Fmax/Smin',
        'go_embedding_correlation': 'Spearman rho of embedding cosine vs GO similarity by '
                                    'SCOPe relationship',
    }
    written = {}
    pd.DataFrame(coverage_rows).to_csv(os.path.join(run_dir, 'go_annotation_coverage.tsv'),
                                       sep='\t', index=False)
    written['go_annotation_coverage.tsv'] = descriptions['go_annotation_coverage']
    for name, parts in tables.items():
        df = pd.concat(parts, ignore_index=True)
        front = ['evidence', 'aspect']
        df = df[front + [c for c in df.columns if c not in front]]
        df.to_csv(os.path.join(run_dir, f'{name}.tsv'), sep='\t', index=False)
        written[f'{name}.tsv'] = descriptions[name]

    run_info = {
        'started': started.isoformat(timespec='seconds'),
        'finished': datetime.datetime.now().isoformat(timespec='seconds'),
        'command': sys.argv, 'args': vars(args), 'git_commit': ev.git_commit(),
        'k': k, 'annot_file': os.path.abspath(annot_file), 'go_obo_file': os.path.abspath(go_obo),
        'go_obo_version': getattr(dag, 'version', None),
        'corpus_size': len(corpus), 'queries': len(queries),
        'methods': {m: len(h) for m, h in method_hits.items()},
        'files': {**written, 'config.yml': 'copy of the config.yml used',
                  'report.txt': 'console output', 'run_info.json': 'this file'},
    }
    with open(os.path.join(run_dir, 'run_info.json'), 'w') as f:
        json.dump(run_info, f, indent=2, default=str)
    print(f'\nResults written to {run_dir}')
    sys.stdout = sys.__stdout__
    report_fh.close()


if __name__ == '__main__':
    main()
