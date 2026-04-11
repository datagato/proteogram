"""
Copy structure files of certain amino acid length from one folder to another 
(mainly for running gtalign on a specific folder of structures).
"""
import os
import shutil
from time import time
import glob
from tqdm import tqdm
import warnings

from Bio.PDB.PDBParser import PDBParser, PDBConstructionWarning
from Bio.PDB.Polypeptide import PPBuilder


PROTEIN_LEN_LOWER_CUTOFF = 20
PROTEIN_LEN_UPPER_CUTOFF = 1e9

warnings.filterwarnings("ignore", category=PDBConstructionWarning)

def get_sequence(pdb_file):
    """Get a protein sequence from a PDB file"""
    p = PDBParser(PERMISSIVE=0)
    structure = p.get_structure('xyz', pdb_file)
    ppb = PPBuilder()
    seq = ''
    for pp in ppb.build_peptides(structure):
        seq += pp.get_sequence()
    return seq

if __name__ == '__main__':
    structure_dir = os.path.join('data',
                                  'human_proteome_scope_experiment',
                                  'human_proteome_scope_sampled')
    new_structure_dir = os.path.join('data',
                                          'human_proteome_scope_experiment',
                                          'human_proteome_scope_sampled_small100')

    # If not exists, make output dir
    if os.path.exists(new_structure_dir):
        shutil.rmtree(new_structure_dir)
    os.makedirs(new_structure_dir)

    start = time()
    pdb_files = glob.glob(os.path.join(structure_dir, '*.ent'))
    keyerrs = 0
    for pdb_file in tqdm(pdb_files):
        bname =  os.path.basename(pdb_file)
        chain_id =bname[1:5].upper()+':'+ bname[5].upper()
        try:
            sequence = get_sequence(pdb_file)
            if len(sequence) >= PROTEIN_LEN_LOWER_CUTOFF and \
                len(sequence) <= PROTEIN_LEN_UPPER_CUTOFF:
                shutil.copy(pdb_file, pdb_file.replace(structure_dir, new_structure_dir))
        except KeyError as e:
            keyerrs+=1
    print(f'Number of key errors = {keyerrs}')
