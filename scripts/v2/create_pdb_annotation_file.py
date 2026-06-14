"""
Annotate per-chain monomer PDB files ({PDBID}_{CHAIN}.pdb) with:
  - GO Molecular_function, Biological_process, Cellular_component (chain-specific)
  - GO slim annotations via goslim_generic (comparable high-level labels)
  - RCSB entry-level metadata (deposit date, method, molecular weight, etc.)
  - UniProt transmembrane annotation (entry-level)

Input files must follow {PDBID}_{CHAIN}.pdb naming from split_pdb_into_chains.py.
GO data is fetched once per PDB entry and cached, so chains from the same
structure share one API call. A small delay is inserted between unique PDB
entries to respect API rate limits.
"""
import glob
import os
import time
import warnings

import pandas as pd
import requests
from Bio.PDB.PDBParser import PDBConstructionWarning, PDBParser
from Bio.PDB.Polypeptide import PPBuilder
from goatools.mapslim import mapslim as _mapslim
from goatools.obo_parser import GODag

from proteogram.common import read_yaml


warnings.filterwarnings("ignore", category=PDBConstructionWarning)

_API_DELAY = 0.1  # seconds between unique PDB ID API calls
_GO_OBO_URL = 'http://current.geneontology.org/ontology/go-basic.obo'
_GOSLIM_OBO_URL = 'http://current.geneontology.org/ontology/subsets/goslim_generic.obo'


def get_sequence(pdb_path):
    seq = ''
    try:
        p = PDBParser(PERMISSIVE=0)
        structure = p.get_structure('xyz', pdb_path)
        ppb = PPBuilder()
        for pp in ppb.build_peptides(structure):
            seq += str(pp.get_sequence())
    except Exception:
        pass
    return seq


def parse_pdb_id_and_chain(basename):
    """Extract PDB ID and chain from filenames like '5WSU_A.pdb'."""
    noext = basename.rsplit('.', 1)[0]   # '5WSU_A'
    parts = noext.split('_', 1)
    pdb_id = parts[0].upper()
    chain_id = parts[1] if len(parts) > 1 else ''
    return pdb_id, chain_id


def download_if_missing(url, path):
    if os.path.exists(path):
        return
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    print(f'Downloading {os.path.basename(path)} from Gene Ontology...')
    r = requests.get(url, timeout=120)
    r.raise_for_status()
    with open(path, 'wb') as fh:
        fh.write(r.content)


def load_go_dags(go_obo_path, goslim_obo_path):
    """Load full GO and GO slim DAGs, auto-downloading OBO files if absent."""
    download_if_missing(_GO_OBO_URL, go_obo_path)
    download_if_missing(_GOSLIM_OBO_URL, goslim_obo_path)
    go_dag = GODag(go_obo_path)
    goslim_dag = GODag(goslim_obo_path)
    return go_dag, goslim_dag


def map_to_go_slim(go_ids_str, go_dag, goslim_dag):
    """Map pipe-delimited specific GO IDs to their direct GO slim ancestors.

    Uses goslim_generic, which sits at a depth that is comparable across
    proteins without being too specific — suitable as a classification label.
    Returns a pipe-delimited string of slim GO IDs.
    """
    if not go_ids_str:
        return ''
    slim_terms = set()
    for go_id in go_ids_str.split('|'):
        go_id = go_id.strip()
        if not go_id or go_id not in go_dag:
            continue
        try:
            direct_anc, _ = _mapslim(go_id, go_dag, goslim_dag)
            slim_terms.update(direct_anc)
        except Exception:
            pass
    return '|'.join(sorted(slim_terms))


def fetch_go(pdb_id, cache):
    """Fetch GO annotations from PDBe for a PDB entry, cached by PDB ID."""
    if pdb_id in cache:
        return cache[pdb_id]
    try:
        r = requests.get(
            f'https://www.ebi.ac.uk/pdbe/graph-api/mappings/go/{pdb_id.lower()}',
            timeout=30)
        cache[pdb_id] = r.json()
    except Exception:
        cache[pdb_id] = {}
    time.sleep(_API_DELAY)
    return cache[pdb_id]


def extract_go_terms_for_chain(go_data, pdb_id, chain_id, category):
    """Return pipe-delimited GO IDs mapped to chain_id under the given category."""
    terms = []
    # PDBe keys the response by lowercase PDB ID
    pdb_entry = go_data.get(pdb_id.lower(), go_data.get(pdb_id.upper(), {}))
    for go_id, info in pdb_entry.get('GO', {}).items():
        if info['category'] == category:
            if any(m['chain_id'] == chain_id for m in info.get('mappings', [])):
                terms.append(go_id)
    return '|'.join(sorted(terms))


def fetch_rcsb_entry(pdb_id, cache):
    if pdb_id in cache:
        return cache[pdb_id]
    try:
        r = requests.get(
            f'https://data.rcsb.org/rest/v1/core/entry/{pdb_id}', timeout=30)
        cache[pdb_id] = r.json()
    except Exception:
        cache[pdb_id] = {}
    time.sleep(_API_DELAY)
    return cache[pdb_id]


def fetch_uniprot_features(pdb_id, cache):
    """Entry-level UniProt features for TM region annotation."""
    if pdb_id in cache:
        return cache[pdb_id]
    try:
        r = requests.get(
            f'https://data.rcsb.org/rest/v1/core/uniprot/{pdb_id}', timeout=30)
        features = r.json()[0].get('rcsb_uniprot_feature', [])
    except Exception:
        features = []
    cache[pdb_id] = features
    time.sleep(_API_DELAY)
    return features


if __name__ == '__main__':
    config = read_yaml('config.yml')
    monomers_dir = config['pdb_monomers_dir']
    annot_file = config['annot_file']
    fasta_style_file = config['fasta_style_file']
    limit_file = config.get('limit_file', '')
    go_obo_path = config.get('go_obo_file', '../data/go/go-basic.obo')
    goslim_obo_path = config.get('goslim_obo_file', '../data/go/goslim_generic.obo')

    go_dag, goslim_dag = load_go_dags(go_obo_path, goslim_obo_path)

    limit_to_these = set()
    if limit_file:
        with open(limit_file, 'r') as f:
            for line in f:
                limit_to_these.add(os.path.basename(line.strip()).rsplit('.', 1)[0])

    pdb_files = []
    for ext in ('*.pdb', '*.ent'):
        pdb_files.extend(
            glob.glob(os.path.join(monomers_dir, '**', ext), recursive=True))

    go_cache = {}
    rcsb_cache = {}
    uniprot_cache = {}
    annot_data = []
    for_fasta = {}

    for pdb_file in pdb_files:
        if not os.path.isfile(pdb_file):
            continue

        basename = os.path.basename(pdb_file)
        noext = basename.rsplit('.', 1)[0]

        if limit_to_these and noext not in limit_to_these:
            continue

        pdb_id, chain_id = parse_pdb_id_and_chain(basename)
        if not pdb_id or not chain_id:
            print(f'Skipping {basename}: cannot parse PDB ID / chain ID')
            continue

        pdb_id_chain = f'{pdb_id}_{chain_id}'
        proteogram_file = f'{pdb_id_chain}.jpg'
        seq = get_sequence(pdb_file)

        # GO annotations — one API call per PDB entry, shared across chains
        go_data = fetch_go(pdb_id, go_cache)
        go_mf = extract_go_terms_for_chain(go_data, pdb_id, chain_id, 'Molecular_function')
        go_bp = extract_go_terms_for_chain(go_data, pdb_id, chain_id, 'Biological_process')
        go_cc = extract_go_terms_for_chain(go_data, pdb_id, chain_id, 'Cellular_component')

        # GO slim — map specific terms to goslim_generic for comparable class labels
        go_slim_mf = map_to_go_slim(go_mf, go_dag, goslim_dag)
        go_slim_bp = map_to_go_slim(go_bp, go_dag, goslim_dag)
        go_slim_cc = map_to_go_slim(go_cc, go_dag, goslim_dag)

        # RCSB entry metadata — cached per PDB ID
        rcsb = fetch_rcsb_entry(pdb_id, rcsb_cache)
        deposit_date = rcsb.get('rcsb_accession_info', {}).get('deposit_date', '')
        exp_method = rcsb.get('rcsb_entry_info', {}).get('experimental_method', '')
        mol_weight = rcsb.get('rcsb_entry_info', {}).get('molecular_weight', '')
        disulfide_cnt = rcsb.get('rcsb_entry_info', {}).get('disulfide_bond_count', '')
        protein_entity_cnt = rcsb.get('rcsb_entry_info', {}).get('polymer_entity_count_protein', '')

        # UniProt TM regions — entry-level, cached per PDB ID
        uniprot_features = fetch_uniprot_features(pdb_id, uniprot_cache)
        # TODO: could be more precise by checking if the TM region maps to this chain, but many entries lack that detail, so we'll just flag the whole chain as TM if any TM region is present in the entry
        tm_regions = [f for f in uniprot_features if f.get('type') == 'TRANSMEMBRANE_REGION']
        is_tm = len(tm_regions) > 0
        tm_cnt = len(tm_regions)

        annot_data.append([
            noext,
            basename,
            proteogram_file,
            pdb_id,
            chain_id,
            pdb_id_chain,
            len(seq),
            deposit_date,
            exp_method,
            mol_weight,
            disulfide_cnt,
            protein_entity_cnt,
            is_tm,
            tm_cnt,
            go_mf,
            go_bp,
            go_cc,
            go_slim_mf,
            go_slim_bp,
            go_slim_cc,
            seq,
        ])
        for_fasta[f'>{pdb_id_chain}'] = seq

    annot_df = pd.DataFrame(annot_data, columns=[
        'MonomerID',
        'PDBFileName',
        'ProteogramFileName',
        'PDBId',
        'ChainId',
        'PDBAndChainId',
        'PDBSequenceLength',
        'PDBDepositDate',
        'PDBExperimentalMethod',
        'PDBMolecularWeight',
        'PDBDisulfideBond',
        'PDBProteinEntityCount',
        'PDBIsTransmembrane',
        'PDBTransmembraneRegionCounts',
        'GOTerms_MF',
        'GOTerms_BP',
        'GOTerms_CC',
        'GOSlim_MF',
        'GOSlim_BP',
        'GOSlim_CC',
        'PDBSequence',
    ])

    os.makedirs(os.path.dirname(os.path.abspath(annot_file)), exist_ok=True)
    try:
        annot_df.to_csv(annot_file, sep='\t', index=False)
    except Exception as e:
        out = os.path.join('.', os.path.basename(annot_file))
        print(f'Could not save to {annot_file}: {e}, saving to {out}')
        annot_df.to_csv(out, sep='\t', index=False)

    os.makedirs(os.path.dirname(os.path.abspath(fasta_style_file)), exist_ok=True)
    try:
        fasta_out = open(fasta_style_file, 'w')
    except Exception as e:
        fasta_out = open(os.path.join('.', os.path.basename(fasta_style_file)), 'w')
        print(f'Could not save fasta to {fasta_style_file}: {e}')
    for header, seq in for_fasta.items():
        fasta_out.write(header + '\n' + seq + '\n')
    fasta_out.close()

    print(f'Annotated {len(annot_df)} chains. Saved to {annot_file}')
