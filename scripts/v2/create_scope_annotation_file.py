"""
Using a list of SCOPe-named structure files, look up key annotations and
create a tab-delimited file to hold those annotations for lookup.  Uses
local SCOPe database files and RCSB PDB API as well as PDBe API calls.

GO data is fetched once per PDB entry and cached across SCOPe domains from
the same structure. Chain-specific GO terms (MF/BP/CC) are extracted by
filtering the PDBe mappings list by chain_id. Specific GO terms are also
mapped to goslim_generic for comparable high-level class labels.
"""
import glob
import os
import time
import warnings

import pandas as pd
import requests
from Bio.PDB.PDBParser import PDBConstructionWarning, PDBParser
from Bio.PDB.Polypeptide import PPBuilder
from Bio.SCOP import Scop
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
    limit_file = config['limit_file']
    go_obo_path = config.get('go_obo_file', '../data/go/go-basic.obo')
    goslim_obo_path = config.get('goslim_obo_file', '../data/go/goslim_generic.obo')
    structures_dir = config['scope_structures_dir']
    annot_file = config['annot_file']
    fasta_style_file = config['fasta_style_file']
    scope_cla_handle = config['scope_cla_file']
    scope_des_handle = config['scope_des_file']
    scope_hie_handle = config['scope_hie_file']

    limit_to_these_structs = set()
    if limit_file:
        with open(limit_file, 'r') as f:
            for line in f:
                limit_to_these_structs.add(
                    os.path.basename(line.strip()).replace('.ent', ''))

    go_dag, goslim_dag = load_go_dags(go_obo_path, goslim_obo_path)

    scop = Scop(cla_handle=open(scope_cla_handle, 'r'),
                des_handle=open(scope_des_handle, 'r'),
                hie_handle=open(scope_hie_handle, 'r'))

    pdb_files = glob.glob(os.path.join(structures_dir, '**', '*'), recursive=True)

    go_cache = {}
    rcsb_cache = {}
    uniprot_cache = {}
    annot_data = []
    for_fasta = {}

    for pdb_file in pdb_files:
        if not os.path.isfile(pdb_file):
            continue

        # SCOPe domain ID from filename: d{PDBID}{CHAIN}{domnum}
        bname = os.path.basename(pdb_file).rsplit('.', 1)[0]

        if limit_to_these_structs and bname not in limit_to_these_structs:
            continue

        seq = get_sequence(pdb_file)

        try:
            pdb_id = bname[1:5].upper()
            chain = bname[5].upper()
            pdb_id_chain = f'{pdb_id}_{chain}'
            prot_file = f'{pdb_id}_{chain}.jpg'
        except Exception:
            print(f'Problem with filename {os.path.basename(pdb_file)}')
            continue

        try:
            scop_entry = scop.getDomainBySid(bname)
            sccs = scop_entry.sccs
            sccs_spl = sccs.split('.')
            cls = sccs_spl[0]
            fold = '.'.join(sccs_spl[:2])
            sfam = '.'.join(sccs_spl[:3])
            fam = sccs
        except Exception:
            cls, fold, sfam, fam = '', '', '', ''

        # GO annotations — one API call per PDB entry, cached across SCOPe domains
        go_data = fetch_go(pdb_id, go_cache)
        go_mf = extract_go_terms_for_chain(go_data, pdb_id, chain, 'Molecular_function')
        go_bp = extract_go_terms_for_chain(go_data, pdb_id, chain, 'Biological_process')
        go_cc = extract_go_terms_for_chain(go_data, pdb_id, chain, 'Cellular_component')

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
        tm_regions = [f for f in uniprot_features if f.get('type') == 'TRANSMEMBRANE_REGION']
        is_tm = len(tm_regions) > 0
        tm_cnt = len(tm_regions)

        annot_data.append([
            bname,
            os.path.basename(pdb_file),
            prot_file,
            pdb_id,
            chain,
            pdb_id_chain,
            cls,
            fold,
            sfam,
            fam,
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

        fasta_style_id = f'>{pdb_id_chain}|{bname}|{fam}'
        for_fasta[fasta_style_id] = seq

    annot_df = pd.DataFrame(annot_data, columns=[
        'SCOPeID',
        'PDBFileName',
        'ProteogramFileName',
        'PDBId',
        'ChainId',
        'PDBAndChainId',
        'SCOPeClass',
        'SCOPeFold',
        'SCOPeSuperfamily',
        'SCOPeFamily',
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

    try:
        annot_df.to_csv(annot_file, sep='\t', index=False)
    except Exception as e:
        out = os.path.join('.', os.path.basename(annot_file))
        print(f'Problem saving to {annot_file}: {e}, saving to {out}')
        annot_df.to_csv(out, sep='\t', index=False)

    try:
        fasta_out = open(fasta_style_file, 'w')
    except Exception as e:
        fasta_out = open(os.path.join('.', os.path.basename(fasta_style_file)), 'w')
        print(f'Problem saving fasta to {fasta_style_file}: {e}')

    for header, seq in for_fasta.items():
        fasta_out.write(header + '\n' + seq + '\n')
    fasta_out.close()

    print(f'Annotated {len(annot_df)} SCOPe domains. Saved to {annot_file}')
