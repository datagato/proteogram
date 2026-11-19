"""GO annotations with evidence codes, via UniProt.

PDBe's GO mappings (graph-api/mappings/go) give chain-level GO terms but no
evidence codes, so experimental annotations can't be told apart from
electronic ones (IEA, mostly inferred from InterPro families -- which partly
re-measures sequence/family similarity in any GO-based evaluation). This module
gets the same chain -> UniProt link from SIFTS (PDBe graph-api/mappings/uniprot)
and then each accession's GO terms with their evidence codes from UniProt's
REST API (UniProt-GOA, the source PDBe's mappings are derived from).

Used by scripts/v2/create_scope_annotation_file.py and
scripts/v2/create_pdb_annotation_file.py. Added columns (see add_go_evidence_columns):
    UniProtAccessions   pipe-delimited accessions SIFTS maps to the chain
    GOEvidence_MF/BP/CC GO:0004672=IDA,IEA|GO:0005524=IEA  (term=evidence codes)
    GOTermsExp_MF/BP/CC pipe-delimited terms with at least one experimental code
"""
import time

import requests


# GO experimental evidence codes, including high-throughput ones (GO guidelines).
EXPERIMENTAL_CODES = frozenset({'EXP', 'IDA', 'IPI', 'IMP', 'IGI', 'IEP',
                                'HTP', 'HDA', 'HMP', 'HGI', 'HEP'})
ASPECTS = {'F': 'MF', 'P': 'BP', 'C': 'CC'}

_SIFTS_URL = 'https://www.ebi.ac.uk/pdbe/graph-api/mappings/uniprot/{}'
_UNIPROT_URL = 'https://rest.uniprot.org/uniprotkb/accessions'


def _get_with_retries(url, params=None, timeout=60, retries=4, delay=2.0):
    """GET with retries on connection errors, 429 and 5xx; returns the Response or None."""
    for attempt in range(retries):
        try:
            r = requests.get(url, params=params, timeout=timeout)
            if r.status_code == 404:
                return r
            if r.status_code == 429 or r.status_code >= 500:
                raise requests.HTTPError(f'HTTP {r.status_code}')
            return r
        except (requests.RequestException, ValueError):
            if attempt == retries - 1:
                return None
            time.sleep(delay * (2 ** attempt))
    return None


def fetch_sifts_uniprot(pdb_id, cache, api_delay=0.1):
    """{chain_id: sorted list of UniProt accessions} for a PDB entry (SIFTS), cached."""
    pdb_id = pdb_id.lower()
    if pdb_id in cache:
        return cache[pdb_id]
    chains = {}
    r = _get_with_retries(_SIFTS_URL.format(pdb_id), timeout=30)
    if r is not None and r.status_code == 200:
        try:
            entry = r.json().get(pdb_id, {})
            for acc, info in entry.get('UniProt', {}).items():
                for m in info.get('mappings', []):
                    chains.setdefault(m.get('chain_id'), set()).add(acc)
        except ValueError:
            pass
    cache[pdb_id] = {c: sorted(a) for c, a in chains.items()}
    time.sleep(api_delay)
    return cache[pdb_id]


def fetch_uniprot_go(accessions, batch_size=100, api_delay=0.2):
    """GO annotations with evidence codes for UniProt accessions, in batches.

    Returns {accession: {aspect ('MF'/'BP'/'CC'): {go_id: set(evidence codes)}}}.
    Accessions UniProt doesn't return (obsolete, merged) are absent from the
    result. UniProtKB entries carry UniProt-GOA's positive annotations
    (NOT-qualified annotations are not included).
    """
    result = {}
    accessions = sorted(set(a for a in accessions if a))
    for start in range(0, len(accessions), batch_size):
        batch = accessions[start:start + batch_size]
        url = _UNIPROT_URL
        params = {'accessions': ','.join(batch), 'fields': 'accession,go',
                  'format': 'json', 'size': len(batch)}
        while url:
            r = _get_with_retries(url, params=params)
            if r is None or r.status_code != 200:
                print(f'WARNING: UniProt GO request failed for {len(batch)} accession(s) '
                      f'starting {batch[0]}; their GO evidence columns will be empty.')
                break
            for entry in r.json().get('results', []):
                acc = entry.get('primaryAccession')
                per_aspect = result.setdefault(acc, {'MF': {}, 'BP': {}, 'CC': {}})
                for xref in entry.get('uniProtKBCrossReferences', []):
                    if xref.get('database') != 'GO':
                        continue
                    props = {p['key']: p['value'] for p in xref.get('properties', [])}
                    aspect = ASPECTS.get(props.get('GoTerm', ' ')[0])
                    code = props.get('GoEvidenceType', '').split(':')[0]
                    if aspect and code:
                        per_aspect[aspect].setdefault(xref['id'], set()).add(code)
            # Follow pagination if UniProt split the batch; params are in the next URL.
            url = r.links.get('next', {}).get('url')
            params = None
            time.sleep(api_delay)
    return result


def format_go_evidence(term_codes):
    """{go_id: codes} -> 'GO:1=IDA,IEA|GO:2=IEA' (sorted, stable)."""
    return '|'.join(f'{go}={",".join(sorted(codes))}'
                    for go, codes in sorted(term_codes.items()))


def parse_go_evidence(value):
    """'GO:1=IDA,IEA|GO:2=IEA' -> {go_id: set(codes)}; empty/NaN -> {}."""
    out = {}
    if not isinstance(value, str) or not value.strip():
        return out
    for item in value.split('|'):
        go, _, codes = item.partition('=')
        if go.strip():
            out[go.strip()] = {c for c in codes.split(',') if c}
    return out


def add_go_evidence_columns(annot_df, pdb_col='PDBId', chain_col='ChainId', api_delay=0.1):
    """Add UniProtAccessions, GOEvidence_* and GOTermsExp_* columns to annot_df.

    Chains map to accessions through SIFTS (one call per PDB entry, cached);
    GO comes from UniProt in batches. Terms from all accessions on a chain are
    merged. Returns the DataFrame with the new columns.
    """
    sifts_cache = {}
    chain_accs = []
    for pdb_id, chain in zip(annot_df[pdb_col], annot_df[chain_col]):
        mapping = fetch_sifts_uniprot(str(pdb_id), sifts_cache, api_delay=api_delay)
        chain_accs.append(mapping.get(str(chain), []))
    all_accs = {a for accs in chain_accs for a in accs}
    print(f'SIFTS: {sum(1 for a in chain_accs if a)} of {len(chain_accs)} chains mapped to '
          f'{len(all_accs)} UniProt accession(s); fetching GO with evidence codes...')
    go_by_acc = fetch_uniprot_go(all_accs)

    cols = {'UniProtAccessions': []}
    for aspect in ('MF', 'BP', 'CC'):
        cols[f'GOEvidence_{aspect}'] = []
        cols[f'GOTermsExp_{aspect}'] = []
    for accs in chain_accs:
        cols['UniProtAccessions'].append('|'.join(accs))
        for aspect in ('MF', 'BP', 'CC'):
            merged = {}
            for acc in accs:
                for go, codes in go_by_acc.get(acc, {}).get(aspect, {}).items():
                    merged.setdefault(go, set()).update(codes)
            cols[f'GOEvidence_{aspect}'].append(format_go_evidence(merged))
            cols[f'GOTermsExp_{aspect}'].append(
                '|'.join(sorted(g for g, c in merged.items() if c & EXPERIMENTAL_CODES)))
    for name, values in cols.items():
        annot_df[name] = values

    for aspect in ('MF', 'BP'):
        n_any = sum(1 for v in cols[f'GOEvidence_{aspect}'] if v)
        n_exp = sum(1 for v in cols[f'GOTermsExp_{aspect}'] if v)
        print(f'GO {aspect} with evidence codes: {n_any}/{len(annot_df)} rows; '
              f'experimental-only: {n_exp}/{len(annot_df)}')
    return annot_df
