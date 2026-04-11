"""
Get the protein structure files from the PDB and parse out the sequences
with BioPython, saving the structures as format pdbXXXX.ent.

Also, these structure files (which have all chains) are further parsed
by chain and saved in the SCOPe format (<SID>.ent)

The all-chain structure files are mainly for DaliLite.  The single chain
SCOPe SID files will be used with gtalign and proteogram methods.
"""
from time import time
import pandas as pd
import glob
import os
import string
import shutil
import warnings

from Bio.PDB.PDBList import PDBList
from Bio import SeqIO, PDB

from proteogram.common.utils import split_by_chain_and_save, ChainSelect


warnings.filterwarnings("ignore", category=PDB.PDBParser.PDBConstructionWarning)

if __name__ == '__main__':
    # Files and folders (please make sure all folders exist already)
    scope_queries_list = 'queries-scope20840.lst'
    download_dir = './data/queries_scope_gtalign_experiment/downloaded_structures'
    scope_pdbs_dir = './data/queries_scope_gtalign_experiment/scope_pdbs_from_downloaded_structures'

    # Create some output directories (delete and recreate if exists)
    if os.path.exists(scope_pdbs_dir):
        shutil.rmtree(scope_pdbs_dir)
    os.makedirs(scope_pdbs_dir)

    # Open the SCOPe 2.08 proteins list
    df = pd.read_csv(scope_queries_list, sep='\t', header=None)
    print(df.shape)

    pdb_ids = []
    chain_ids = []
    scope_sids = []
    for i in range(df.shape[0]):
        pdb_id = str(df.iloc[i,0])[1:5].upper()
        chain_id = str(df.iloc[i,0])[5].upper()
        scope_sid = df.iloc[i,0][0:7]
        if chain_id in string.ascii_uppercase:
            pdb_ids.append(pdb_id)
            chain_ids.append(chain_id)
            scope_sids.append(scope_sid)


    # PDB list object to get quick access to the structure lists on the PDB or its mirrors
    pdb_list = PDBList(pdb=download_dir, verbose=False)
    # Download PDB structures from list
    start = time()
    pdb_list.download_pdb_files(list(set(pdb_ids)),
                                file_format='pdb',
                                pdir=download_dir)
    print(f'Getting the {len(pdb_ids)} PDB files took {time()-start} seconds')

    pdb_files = glob.glob(os.path.join(download_dir, '*.ent'))
    parser = PDB.PDBParser(PERMISSIVE=1)

    for i in range(len(pdb_files)):
        pdb_id = os.path.basename(pdb_files[i])[:-4][3:7].upper()
        # Get the indices of the pdb_id (bc there might be many)
        pdb_ids_idxs = [i for i, x in enumerate(pdb_ids) if x == pdb_id]
        # look at all chains associated with the pdb_id by using indices
        for idx in pdb_ids_idxs:
            chain_id = chain_ids[idx]
            sid = scope_sids[idx]
            split_by_chain_and_save(pdb_files[i], chain_id, sid, scope_pdbs_dir)
