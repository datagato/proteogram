"""
Create a training dataset with protein structures from SCOPe not in test set.
This script finds the SCOPe levels for a protein, creates proteograms, and
places the proteograms in folders according to a certain SCOPe level.
"""
import os
import pandas as pd
import numpy as np
import shutil
import random
import glob
from tqdm import tqdm
import matplotlib.pyplot as plt
import warnings

from Bio.SCOP import Scop
from Bio.PDB.PDBParser import PDBConstructionWarning

from proteogram.common.utils import read_yaml
from proteogram.v1.proteogram import Proteogram


# Distance cutoff for measuring possible residue interactions in Angstroms
SEQUENCE_LEN_LOWER_CUTOFF = 20
SEQUENCE_LEN_UPPER_CUTOFF = 1e9

warnings.filterwarnings("ignore", category=PDBConstructionWarning)

if __name__ == '__main__':

    config = read_yaml('config.yml')
    scope_eval_set = config['scope_eval_set']
    scope_structures_dir = config['scope_structures_dir']
    scope_cla_file = config['scope_cla_file']
    scope_des_file = config['scope_des_file']
    scope_hie_file = config['scope_hie_file']
    training_structures_dir = config['training_structures_dir']
    training_proteograms_dir = config['training_proteograms_dir']
    eval_structures_dir = config['eval_structures_dir']
    eval_proteograms_dir = config['eval_proteograms_dir']
    label_df_out = config['label_df_out']

    # All scope 2.08 structure files
    scope_struct_files =  glob.glob(os.path.join(scope_structures_dir,
                                                '**',
                                                '*.ent'),
                                    recursive=True)

    # Create a Scop object
    scop = Scop(cla_handle=open(scope_cla_file, 'r'),
                des_handle=open(scope_des_file, 'r'),
                hie_handle=open(scope_hie_file, 'r'))

    # scope_prots has the following structure - {file_path: (cls, fold, sfam, fam)}
    scope_prots = []
    for scope_struct_file in tqdm(scope_struct_files):
        sid = os.path.basename(scope_struct_file).replace('.ent', '')
        with open(scope_struct_file, 'r') as fin:
            try:
                # Get a specific domain by its SCOP identifier (sid)
                scop_entry = scop.getDomainBySid(sid)
                # Parse out info for our dataframe
                sccs = scop_entry.sccs
                sccs_spl = sccs.split('.')
                pdb_id = sid[1:5].upper()
                chain = sid[5].upper()
                cls, fold, sfam, fam = sccs_spl[0], '.'.join(sccs_spl[:2]), '.'.join(sccs_spl[:3]), sccs
                scope_prots.append([pdb_id, chain, scope_struct_file, cls, fold, sfam, fam])
            except Exception as e:
                print(e)
    # Place data into a dataframe for easier access
    label_df = pd.DataFrame(scope_prots,
                            columns=['pdb_id', 'chain', 'structure_file', 'class', 'fold', 'superfamily', 'family'])
    
    # Save annotations from SCOPe for the proteins in the 2.08 db
    label_df.to_csv(label_df_out, sep='\t', index=False)
    
    # Create output folders - if these already exist the following with throw an error
    os.makedirs(eval_structures_dir)
    os.makedirs(eval_proteograms_dir)
    os.makedirs(training_structures_dir)
    os.makedirs(training_proteograms_dir)

    file_dict = {str(k): [] for k in label_df['family'].unique()}
    for i in range(label_df.shape[0]):
        fam = str(label_df.loc[i,'family'])
        prot_file = label_df.loc[i, 'structure_file']
        if fam in file_dict:
            tmp = file_dict[fam]
            tmp.append(prot_file)
            file_dict[fam] = tmp
        else:
            file_dict[fam] = [prot_file]

    # Put 80% of the files found for the family into train folder and rest into val
    file_dict_train = {k: v[:int(len(v)*0.8)] for k, v in file_dict.items()}
    file_dict_val = {k: v[int(len(v)*0.8):] for k, v in file_dict.items()}
    train_cnt_limit = 1e9
    val_cnt_limit = 1e9
    pdb_files_final = []
    pdb_files_final_eval = []
    for fam in tqdm(file_dict_train.keys()):

        # Create dirs for families
        fam_train_dir = os.path.join(training_proteograms_dir, 'train', fam)
        if os.path.exists(fam_train_dir):
            shutil.rmtree(fam_train_dir)
        os.makedirs(fam_train_dir)
        fam_val_dir = os.path.join(training_proteograms_dir, 'val', fam)
        if os.path.exists(fam_val_dir):
            shutil.rmtree(fam_val_dir)
        os.makedirs(fam_val_dir)

        # Create a balanced eval dataset
        pdb_files_train = file_dict_train[fam]
        pdb_files_eval_set = []
        if len(pdb_files_train) > 0:
            num_sample = len(pdb_files_train)*0.005
            pdb_files_eval_set = np.random.choice(pdb_files_train, int(max(num_sample,1)))

        # Create proteograms and put into correct SCOPe level folder for
        # training model
        cnt_train_fam = 0
        for pdb_file in file_dict_train[fam]:
            proteogram = Proteogram(pdb_file,
                                    atom_distance_cutoff=15,
                                    sequence_len_lower_cutoff=SEQUENCE_LEN_LOWER_CUTOFF,
                                    sequence_len_upper_cutoff=SEQUENCE_LEN_UPPER_CUTOFF)
            sid = os.path.basename(pdb_file).replace('.ent','')
            if len(sid) > 6:
                sid = sid[:6]
            pdb_id = sid[1:5].upper()
            chain_id = sid[5].upper()
            try:
                sequence = proteogram.get_sequence()
                if len(sequence) >= SEQUENCE_LEN_LOWER_CUTOFF and \
                    len(sequence) < SEQUENCE_LEN_UPPER_CUTOFF:
                    distance_map = proteogram.calc_distogram(sequence, chain_id)
                    hydro_map = proteogram.calc_hydrophobicity_map(sequence, distance_map)
                    charge_map = proteogram.calc_charge_map(sequence, distance_map)
                    final_data = proteogram.stack_data(distance_map, hydro_map, charge_map)
                    if pdb_file not in pdb_files_eval_set:
                        image_file = os.path.join(fam_train_dir,
                                            f'{pdb_id}_{chain_id}.jpg')
                        img = final_data.astype(float)
                        plt.imsave(image_file, img)
                        cnt_train_fam+=1
                        pdb_files_final.append(pdb_file)
                    else:
                        # Add to eval dataset
                        image_file = os.path.join(config['eval_proteograms_dir'], f'{pdb_id}_{chain_id}.jpg')
                        img = final_data.astype(float)
                        plt.imsave(image_file, img)
                        shutil.copy(pdb_file, os.path.join(config['eval_structures_dir'], os.path.basename(pdb_file)))
                        pdb_files_final_eval.append(os.path.basename(pdb_file))
            except Exception as e: 
                print(f'problem with creating or saving a proteogram for {pdb_file}: {e}.')
            # Have as many images for this fam as we wish so it's not too imbalanced
            if cnt_train_fam == train_cnt_limit:
                break
        
        cnt_val_fam = 0
        for pdb_file in file_dict_val[fam]:
            proteogram = Proteogram(pdb_file,
                                    atom_distance_cutoff=15,
                                    sequence_len_lower_cutoff=SEQUENCE_LEN_LOWER_CUTOFF,
                                    sequence_len_upper_cutoff=SEQUENCE_LEN_UPPER_CUTOFF)
            sid = os.path.basename(pdb_file).replace('.ent','')
            if len(sid) > 6:
                sid = sid[:6]
            pdb_id = sid[1:5].upper()
            chain_id = sid[5].upper()
            try:
                sequence = proteogram.get_sequence()
                if len(sequence) >= SEQUENCE_LEN_LOWER_CUTOFF and \
                    len(sequence) < SEQUENCE_LEN_UPPER_CUTOFF:
                    distance_map = proteogram.calc_distogram(sequence, chain_id)
                    hydro_map = proteogram.calc_hydrophobicity_map(sequence, distance_map)
                    charge_map = proteogram.calc_charge_map(sequence, distance_map)
                    final_data = proteogram.stack_data(distance_map, hydro_map, charge_map)
                    image_file = os.path.join(fam_val_dir,
                                            f'{pdb_id}_{chain_id}.jpg')
                    img = final_data.astype(float)
                    plt.imsave(image_file, img)
                    cnt_val_fam+=1
                    pdb_files_final.append(pdb_file)
            except Exception as e:
                print(f'problem with creating or saving a proteogram for {pdb_file}: {e}.')
            if cnt_val_fam == val_cnt_limit:
                break
        
        # If empty folders, remove
        if len(glob.glob(os.path.join(fam_train_dir, '*.jpg'))) == 0:
            print(f'Empty dir: {fam_train_dir}')
            if os.path.exists(fam_train_dir):
                shutil.rmtree(fam_train_dir)
            if os.path.exists(fam_val_dir):
                shutil.rmtree(fam_val_dir)
        if len(glob.glob(os.path.join(fam_val_dir, '*.jpg'))) == 0:
            print(f'Empty dir {fam_val_dir}')
            if os.path.exists(fam_train_dir):
                shutil.rmtree(fam_train_dir)
            if os.path.exists(fam_val_dir):
                shutil.rmtree(fam_val_dir)

    # Copy structure files from final list above
    for pdb_file in pdb_files_final:
        bname = os.path.basename(pdb_file)
        shutil.copy(pdb_file, os.path.join(training_structures_dir, bname))

    with open(config['scope_eval_set'], 'w') as f:
        for pdb_file in pdb_files_final_eval:
            f.write(pdb_file + '\n')
