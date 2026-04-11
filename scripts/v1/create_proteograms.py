import glob
import os
from time import time
import matplotlib.pyplot as plt
import warnings
from tqdm import tqdm

from Bio.PDB.PDBParser import PDBConstructionWarning

from proteogram.common.utils import read_yaml
from proteogram.v1.proteogram import Proteogram


# Seq length cutoffs for using domain
SEQUENCE_LEN_LOWER_CUTOFF = 20
SEQUENCE_LEN_UPPER_CUTOFF = 1e9

warnings.filterwarnings("ignore", category=PDBConstructionWarning)

if __name__ == '__main__':

    config = read_yaml('config.yml')
    structures_dir = config['scope_structures_dir']
    proteograms_output_dir = config['eval_proteograms_dir']
    # Only create proteograms for these structures in the input limit file
    limit_to_these_structs = []
    if 'limit_file' in config.keys():
        limit_file = config['limit_file']
        with open(limit_file, 'r') as f:
            for line in f:
                limit_to_these_structs.append(line.strip())
    else:
        limit_to_these_structs = None
    # If the output dir exists, don't recreate, otherwise make one
    if os.path.exists(proteograms_output_dir):
        print(f'Directory {proteograms_output_dir} exists, will use.')
    else:
        os.makedirs(proteograms_output_dir)

    start = time()
    pdb_files = glob.glob(os.path.join(structures_dir, '**', '*.ent'), recursive=True)
    keyerrs = 0
    seq_out_of_bounds = 0
    for pdb_file in tqdm(pdb_files):
        bname =  os.path.basename(pdb_file)
        if limit_to_these_structs:
            if bname not in limit_to_these_structs:
                continue
        # SCOPe naming format processed here
        chain_id =bname[1:5].upper()+':'+ bname[5].upper()
        proteogram = Proteogram(pdb_file,
                                atom_distance_cutoff=15,
                                sequence_len_lower_cutoff=SEQUENCE_LEN_LOWER_CUTOFF,
                                sequence_len_upper_cutoff=SEQUENCE_LEN_UPPER_CUTOFF)
        try:
            sequence = proteogram.get_sequence()
            if len(sequence) >= SEQUENCE_LEN_LOWER_CUTOFF and \
                len(sequence) < SEQUENCE_LEN_UPPER_CUTOFF:
                distance_map = proteogram.calc_distogram(sequence, chain_id[-1])
                hydro_map = proteogram.calc_hydrophobicity_map(sequence, distance_map)
                charge_map = proteogram.calc_charge_map(sequence, distance_map)
                final_data = proteogram.stack_data(distance_map, hydro_map, charge_map)
                image_file = os.path.join(proteograms_output_dir,
                                          f'{chain_id.replace(":","_")}.jpg')
                img = final_data.astype(float)
                plt.imsave(image_file, img)
            else:
                seq_out_of_bounds+=1
        except KeyError as e:
            keyerrs+=1
    print(f'Number of key errors = {keyerrs}')
    print(f'Number of sequences above or below the length limits = {seq_out_of_bounds}')

    print(f'Computation took {time()-start} seconds')

