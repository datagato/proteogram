"""
Using a list of SCOPe-named structure files, look up key annotations and
create a tab-delimited file to hold those annotations for lookup.  Uses
local SCOPe database files and RCSB PDB API as well as PDBe API calls.
"""
import warnings
import glob
import os
import requests
import json
import pandas as pd

from Bio.PDB.PDBParser import PDBParser, PDBConstructionWarning
from Bio.PDB.Polypeptide import PPBuilder
from Bio.SCOP import Scop

from proteogram.common import read_yaml


# Ignore PDB construction warnings
warnings.filterwarnings("ignore", category=PDBConstructionWarning)

def get_sequence(pdb_path):
    """Get a protein sequence from a PDB file"""
    seq = ''
    try:
        p = PDBParser(PERMISSIVE=0)
        structure = p.get_structure('xyz', pdb_path)
        ppb = PPBuilder()
        for pp in ppb.build_peptides(structure):
            seq += pp.get_sequence()
    except Exception:
        seq = ''
    return seq

if __name__ == '__main__':

    config = read_yaml('config.yml')
    limit_file = config['limit_file']
    structures_dir = config['scope_structures_dir']
    annot_file = config['annot_file']
    fasta_style_file = config['fasta_style_file']
    scope_cla_handle = config['scope_cla_file']
    scope_des_handle = config['scope_des_file']
    scope_hie_handle = config['scope_hie_file']

    limit_to_these_structs = []
    if limit_file:
        with open(limit_file, 'r') as f:
            for line in f:
                limit_to_these_structs.append(
                    os.path.basename(line.strip()).replace('.ent',''))

    # Create a Scop object to get fold, superfamily and family
    scop = Scop(cla_handle=open(scope_cla_handle, 'r'),
                des_handle=open(scope_des_handle, 'r'),
                hie_handle=open(scope_hie_handle, 'r'))

    pdb_files = glob.glob(os.path.join(structures_dir, '**', '*'), recursive=True)

    annot_data = []
    for_fasta = {}
    for pdb_file in pdb_files:
        if not os.path.isfile(pdb_file):
            continue

        # Get the scope id from structure filename
        bname = os.path.basename(pdb_file).split('.')
        bname = '.'.join(bname[:-1])

        # Check if we have a restricted list and if pdb file is in it
        if limit_file:
            if bname not in limit_to_these_structs:
                continue

        # Sequence info
        seq = get_sequence(pdb_file)

        # SCOPe info
        try:
            pdb_id = bname[1:5].upper()
            chain = bname[5].upper()
            pdb_id_chain = pdb_id + '_' + chain
            prot_file = f'{pdb_id}_{chain}.jpg'
        except:
            print(f'Problem with filename {os.path.basename(pdb_file)}')
            continue
        try:
            scop_entry = scop.getDomainBySid(bname)
            # Parse out info for our dataframe
            sccs = scop_entry.sccs
            sccs_spl = sccs.split('.')
            cls, fold, sfam, fam = sccs_spl[0], '.'.join(sccs_spl[:2]), '.'.join(sccs_spl[:3]), sccs
        except:
            cls, fold, sfam, fam = '', '', '', ''

        # Gene Ontology
        try:
            go_response = requests.get(f'https://www.ebi.ac.uk/pdbe/graph-api/mappings/go/{pdb_id}')
            go_response = json.dumps(go_response.json())
        except:
            go_response = ''

        # RCSB Data API annotations for entry
        deposit_date = ''
        experimental_method = ''
        molecular_weight = ''
        disulfide_bond_count = ''
        entity_count = ''
        try:
            rcsb_response = requests.get(f'https://data.rcsb.org/rest/v1/core/entry/{pdb_id}')
            rcsb_response = rcsb_response.json()
            deposit_date = rcsb_response['rcsb_accession_info']['deposit_date']
            experimental_method = rcsb_response['rcsb_entry_info']['experimental_method']
            molecular_weight = rcsb_response['rcsb_entry_info']['molecular_weight']
            disulfide_bond_count = rcsb_response['rcsb_entry_info']['disulfide_bond_count']
            protein_entity_count = rcsb_response['rcsb_entry_info']['polymer_entity_count_protein']
        except:
            deposit_date = ''
            experimental_method = ''
            molecular_weight = ''
            disulfide_bond_count = ''
            protein_entity_count = ''

        # RCSB Data API annotations for uniprot - get if is transmembrane
        is_tm = False
        tm_cnts = 0
        try:
            rcsb_response = requests.get(f'https://data.rcsb.org/rest/v1/core/uniprot/{pdb_id}')
            rcsb_response = rcsb_response.json()
            # Get first entry and uniprot feature info (locations and sequence indices)
            rcsb_uniprot_features = rcsb_response[0]['rcsb_uniprot_feature']
        except Exception as excp:
            rcsb_uniprot_features = []
        
        for entry in rcsb_uniprot_features:
            # Get TM regions
            if 'type' in entry and entry['type'] == 'TRANSMEMBRANE_REGION':
                is_tm = True
                tm_cnts+=1

        row = [bname,
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
               experimental_method,
               molecular_weight,
               disulfide_bond_count,
               protein_entity_count,
               is_tm,
               tm_cnts,
               go_response,
               seq]
        
        annot_data.append(row)

        fasta_style_id = f'>{pdb_id_chain}|{bname}|{fam}'
        for_fasta[fasta_style_id] = seq

    annot_df = pd.DataFrame(annot_data)
    annot_df.columns = ['SCOPeID',
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
                        'PDBeGOAnnotation',
                        'PDBSequence']

    # Save the results
    try:
        annot_df.to_csv(
            os.path.join(annot_file),
            sep='\t',
            index=False)
    except Exception as e:
        print(f'Problem saving to specific location: {e}, so saving in cwd.')
        annot_df.to_csv(
            os.path.join('.', os.path.basename(annot_file)),
            sep='\t',
            index=False)

    try:
        f = open(fasta_style_file, 'w')
    except Exception as e:
        print(f'Problem saving to specific location: {e}, so saving in cwd.')
        f = open(os.path.join('.', os.path.basename(fasta_style_file), 'w'))

    # Write a fasta-style file with the protein sequences from all structures processed
    for entry in for_fasta:
        f.write(entry + '\n' + for_fasta[entry] + '\n')
    f.close()

