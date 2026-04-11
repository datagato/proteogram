"""
Using list of protein structure files (PDB IDs) find the structures in 
the SCOPe 2.08 downloaded DB (https://scop.berkeley.edu/downloads/pdbstyle/)
move overlapping structure IDs to a separate folder.
"""
import glob
import os
from tqdm import tqdm
import random
import shutil

from Bio.SCOP import Scop


if __name__ == '__main__':
    # Files and folders (please make sure all folders exist already)
    pdb_id_file = 'human_proteome_pdb_ids_20241015.txt'
    output_dir = os.path.join('data', 'human_proteome_scope_sampled')
    scope_cla_handle = os.path.join('data', 'scope2.08', 'dir.cla.scope.2.08-stable.txt')
    scope_des_handle = os.path.join('data', 'scope2.08', 'dir.des.scope.2.08-stable.txt')
    scope_hie_handle = os.path.join('data', 'scope2.08', 'dir.hie.scope.2.08-stable.txt')
    scope_pdb_file_dir = os.path.join('data', 'scope2.08', 'pdbstyle-2.08')
    query_list_output_file = 'queries-scope-for-proteograms.lst'
    sample_size = 10000

    # Open pdb id list
    with open(pdb_id_file, 'r') as f:
        pdb_ids_from_rscb = [line.rstrip() for line in f]

    # Create a Scop object
    scop = Scop(cla_handle=open(scope_cla_handle, 'r'),
                des_handle=open(scope_des_handle, 'r'),
                hie_handle=open(scope_hie_handle, 'r'))
    
    # If the dir exists, delete it and then recreate it
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir)

    # Get all domains in SCOPe
    domains = scop.getDomains()
    lookup = {}
    pdb_ids = []
    sccs_ids = []
    sid_ids = []
    for dom in tqdm(domains):
        dom_dict = dom.__dict__
        try:
            pdb_id_spl = dom_dict['description'].split()
            pdb_id = pdb_id_spl[0].upper()
            sccs = dom_dict['sccs']
            sid = dom_dict['sid']
            # Is the pdb id from SCOPe in our list of interesting pdb ids
            if pdb_id in pdb_ids_from_rscb:
                pdb_ids.append(pdb_id)
                sid_ids.append(sid)
                sccs_ids.append(sccs)
        except Exception as e:
            print(e)

    print(f'{len(pdb_ids)} PDB IDs of interest found in SCOPe db')

    # The SCOPe 2.08 "pdbstyle" structure files (already downloaded)
    pdb_scope_files = glob.glob(os.path.join(scope_pdb_file_dir, '**', '*.ent'),
                                recursive=True)

    # Get a random subsample of structures from our original list
    # sid_ids_sampled = random.sample(sid_ids, sample_size)

    # Look in the downloaded SCOPe 2.08 structure database for our ids of interest
    #  and copy file into an output dir for further analysis (creating proteograms e.g.)
    with open(query_list_output_file, 'w') as f:
        for i in range(len(pdb_scope_files)):
            pdb_scope_file = pdb_scope_files[i]
            file_base = os.path.basename(pdb_scope_file)[:-4]
            if file_base in sid_ids:
                shutil.copy(pdb_scope_file, os.path.join(output_dir, file_base+'.ent'))
                f.write(file_base+'\n')


