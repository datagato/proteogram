"""
Evaluate proteogram approach vs. gtalign.
Pertinent metric calculation info can be found at 
https://weaviate.io/blog/retrieval-evaluation-metrics
The precision@K metric measures how many of the retrieved items are relevant, but
this metric is not rank-aware.
The Mean Average Precision@K (MAP@K) metric measures the system's ability to 
return relevant items in the top K results while placing more relevant items at the top.
"""
import pandas as pd
import numpy as np
import os
import glob
import re
import shutil
from collections import OrderedDict

from Bio.SCOP import Scop

from proteogram.common.utils import read_yaml


def read_gtalign_results(gtalign_results_dir):
    """Read gtalign results into a pandads dataframe"""
    files = glob.glob(os.path.join(gtalign_results_dir, '*.out'))

    results = []
    for file in files:
        pdb_id_query = os.path.basename(file)
        pdb_id_query = pdb_id_query[1:5].upper() + '_' + pdb_id_query[5].upper()
        tmp = [pdb_id_query]
        idx = 0
        with open(file, 'r') as fin:
            i = 0
            for line in fin:
                if line[:7] == '     1 ':
                    idx = i
                    break
                i+=1
        with open(file, 'r') as fin:
            i = 0
            for line in fin:
                if i >= idx and i < (idx + top_k):
                    pdb_id = ''
                    try:
                        pdb_id = re.search('/d(.*?)\.ent', line).group(1).upper()
                        pdb_id = pdb_id[:4]+'_'+pdb_id[4]
                    except:
                        pass
                    tmp.append(pdb_id)
                i+=1
                if i >= (idx + top_k):
                    break
        results.append(tmp)

    return pd.DataFrame(results)

def convert_scopeid_pdbidchain(s):
    pdb_id = s[1:5].upper()
    chain = s[5].upper()
    return pdb_id+'_'+chain

def read_usalign_results(usalign_results):
    """
    Read USalign results into a pandads dataframe.
    Making assumption that the top hit is it's self since
    USalign does not return this data/score (self to self).
    """
    usalign_df = pd.read_csv(usalign_results, sep='\t')
    data = {}

    # dict of ordereddict:
    # {filename1: {filename2: tm1}}
    # {filename2: {filename1: tm2}}
    for i in range(usalign_df.shape[0]):
        scope_filename1 = os.path.basename(usalign_df.loc[i, '#PDBchain1']).split(':')[0]
        scope_filename2 = os.path.basename(usalign_df.loc[i, 'PDBchain2']).split(':')[0]
        tm1 = usalign_df.loc[i, 'TM1']
        tm2 = usalign_df.loc[i, 'TM2']

        if scope_filename1 in data.keys():
            tmp = data[scope_filename1]
            tmp[scope_filename2] = tm1
            data[scope_filename1] = tmp
        else:
            tmp = OrderedDict()
            tmp[scope_filename2] = tm1
            data[scope_filename1] = tmp
        if scope_filename2 in data.keys():
            tmp = data[scope_filename2]
            tmp[scope_filename1] = tm2
            data[scope_filename2] = tmp
        else:
            tmp = OrderedDict()
            tmp[scope_filename1] = tm2
            data[scope_filename2] = tmp
    
    ordered_data = []
    for key, inner_dict in data.items():
        sorted_inner_dict = sorted(inner_dict.items(), key=lambda item: item[1],
                                   reverse=True)
        tmp1 = sorted_inner_dict[:top_k-1]
        query = convert_scopeid_pdbidchain(key)
        tmp2 = [query, query]
        tmp2.extend([convert_scopeid_pdbidchain(prot) for (prot,_) in tmp1])
        ordered_data.append(tmp2)

    return pd.DataFrame(ordered_data)
        
if __name__ == '__main__':
    # Files and folders
    config = read_yaml('config.yml')
    top_k = config['top_k']
    scope_eval_set = config['scope_eval_set']
    proteogram_sim_results = config['proteogram_sim_results']
    gtalign_results_dir = config['gtalign_results_dir']
    usalign_results = config['usalign_results']
    search_images_dir = config['search_images_dir']
    save_bad_searches_dir = config['save_bad_searches_dir']
    save_good_searches_dir = config['save_good_searches_dir']

    scope_cla_handle = config['scope_cla_file']
    scope_des_handle = config['scope_des_file']
    scope_hie_handle = config['scope_hie_file']

    # # Create some output directories (delete and recreate if exists)
    # if os.path.exists(save_bad_searches_dir):
    #     shutil.rmtree(save_bad_searches_dir)
    # os.makedirs(save_bad_searches_dir)
  
    # if os.path.exists(save_good_searches_dir):
    #     shutil.rmtree(save_good_searches_dir)
    # os.makedirs(save_good_searches_dir)

    # Create a Scop object
    scop = Scop(cla_handle=open(scope_cla_handle, 'r'),
                des_handle=open(scope_des_handle, 'r'),
                hie_handle=open(scope_hie_handle, 'r'))

    # Use the scope_eval_set for a protein list (using the sid in the
    # first column to query local SCOPe info)
    scope_prots = []
    with open(scope_eval_set, 'r') as fin:
        for line in fin:
            try:
                spl_line = line.split()
                # Get a specific domain by its SCOP identifier (sid) found in scope_label_file
                scop_entry = scop.getDomainBySid(spl_line[0].replace('.ent',''))
                # Parse out info for our dataframe
                sccs = scop_entry.sccs
                sccs_spl = sccs.split('.')
                pdb_id = spl_line[0][1:5].upper()
                chain = spl_line[0][5].upper()
                pdb_id_chain = pdb_id+'_'+chain
                prot_file = f'{pdb_id}_{chain}.jpg'
                cls, fold, sfam, fam = sccs_spl[0], '.'.join(sccs_spl[:2]), '.'.join(sccs_spl[:3]), sccs
                #print(f'from label file | scope entry: {scop_entry} | family: {fam}')
                scope_prots.append([pdb_id, pdb_id_chain, prot_file, cls, fold, sfam, fam])
            except Exception as e:
                print(e)
    # Place data into a dataframe for easier access
    label_df = pd.DataFrame(scope_prots,
                            columns=['pdb_id', 'pdb_id_chain', 'proteogram_file', 'class', 'fold', 'superfamily', 'family'])
    print(label_df.nunique())

    label_df.to_csv(os.path.join('data', scope_eval_set.split('.')[0]+'_labels.tsv'),
                    sep='\t', index=False)

    # Calculate Precision@K's and MAP@K's
    proteogram_res_df = pd.read_csv(proteogram_sim_results, sep='\t')
    precision_at_ks_fams = []
    precision_at_ks_sfams = []
    precision_at_ks_folds = []
    precision_at_ks_classes = []
    precision_at_ks_fams_map = []
    precision_at_ks_sfams_map = []
    precision_at_ks_folds_map = []
    precision_at_ks_classes_map = []
    for i in range(proteogram_res_df.shape[0]):
        prot_file = os.path.basename(proteogram_res_df.iloc[i,0])
        try:
            query_fam = label_df.loc[label_df['proteogram_file'] == prot_file, 
                                         'family'].iloc[0]
            query_sfam = label_df.loc[label_df['proteogram_file'] == prot_file, 
                                          'superfamily'].iloc[0]
            query_fold = label_df.loc[label_df['proteogram_file'] == prot_file, 
                                          'fold'].iloc[0]
            query_class = label_df.loc[label_df['proteogram_file'] == prot_file,
                                          'class'].iloc[0]
        except Exception as e:
            print(e)
            continue
        
        # For Precision calculation
        tp_fam = 0
        tp_sfam = 0
        tp_fold = 0
        tp_class = 0

        # For MAP@K calculation
        prec_at_k_fam = 0
        prec_at_k_sfam = 0
        prec_at_k_fold = 0
        prec_at_k_class = 0

        # Iterate through results and find matches for family, superfamily,
        # fold and class SCOPe levels
        for rank, target in enumerate(proteogram_res_df.iloc[i,1:]):
            target_file = os.path.basename(target.split(',')[0])
            try:
                target_fam = label_df.loc[label_df['proteogram_file'] == target_file, 
                                          'family'].iloc[0]
            except Exception as e:
                print(f'problem with {target_file} and {query_fam}.')
                continue
            try:
                target_sfam = label_df.loc[label_df['proteogram_file'] == target_file, 
                                           'superfamily'].iloc[0]
            except Exception as e:
                print(f'problem with {target_file} and {query_sfam}.')
                continue
            try:
                target_fold = label_df.loc[label_df['proteogram_file'] == target_file, 
                                           'fold'].iloc[0]
            except Exception as e:
                print(f'problem with {target_file} and {query_fold}.')
                continue
            try:
                target_class = label_df.loc[label_df['proteogram_file'] == target_file,
                                           'class'].iloc[0]
            except Exception as e:
                print(f'problem with {target_file} and {query_class}.')
                continue
                
            if query_fam == target_fam:
                tp_fam+=1
                prec_at_k_fam += (tp_fam / (rank+1))
            if query_sfam == target_sfam:
                tp_sfam+=1
                prec_at_k_sfam += (tp_sfam / (rank+1))
            if query_fold == target_fold:
                tp_fold+=1
                prec_at_k_fold += (tp_fold / (rank+1))
            if query_class == target_class:
                tp_class+=1
                prec_at_k_class += (tp_class / (rank+1))
        
        # For Precision@K
        precision_at_ks_fams.append(tp_fam/top_k)
        precision_at_ks_sfams.append(tp_sfam/top_k)
        precision_at_ks_folds.append(tp_fold/top_k)
        precision_at_ks_classes.append(tp_class/top_k)

        # For MAP@K
        precision_at_ks_fams_map.append(prec_at_k_fam/tp_fam)
        precision_at_ks_sfams_map.append(prec_at_k_sfam/tp_sfam)
        precision_at_ks_folds_map.append(prec_at_k_fold/tp_fold)
        precision_at_ks_classes_map.append(prec_at_k_class/tp_class)

        # Save images of those with no family agreement (tp/top_k)
        pdb_id = os.path.basename(prot_file)[0:4]
        chain_id = os.path.basename(prot_file)[5]
        score_for_top_sims = tp_fold/top_k
        if score_for_top_sims == 0.2:
            to_copy = f'{pdb_id}_{chain_id}_top_sims.jpg'
            shutil.copy(os.path.join(search_images_dir,
                                     to_copy),
                        os.path.join(save_bad_searches_dir, to_copy.replace('top_sims', f'top_sims_{query_fold}_pk{score_for_top_sims:.4f}')))
        # Save images of those with complete family agreement (tp_fam/TOP_K is 1)
        if score_for_top_sims == 1.0:
            to_copy = f'{pdb_id}_{chain_id}_top_sims.jpg'
            shutil.copy(os.path.join(search_images_dir,
                                     to_copy),
                                     os.path.join(save_good_searches_dir, to_copy.replace('top_sims', f'top_sims_{query_fold}_pk{score_for_top_sims:.4f}')))

    proteogram_patk_fam = np.mean(precision_at_ks_fams)
    proteogram_patk_sfam = np.mean(precision_at_ks_sfams)
    proteogram_patk_fold = np.mean(precision_at_ks_folds)
    proteogram_patk_class = np.mean(precision_at_ks_classes)
    proteogram_map_fam = np.mean(precision_at_ks_fams_map)
    proteogram_map_sfam = np.mean(precision_at_ks_sfams_map)
    proteogram_map_fold = np.mean(precision_at_ks_folds_map)
    proteogram_map_class = np.mean(precision_at_ks_classes_map)

    # Calculate Precision@K's and MAP@K's
    gtalign_res_df = read_gtalign_results(gtalign_results_dir)
    # Save it
    gtalign_res_df.to_csv(os.path.join(gtalign_results_dir, 'combined_gtalign_results.tsv'),
            sep='\t',
            index=False)
    precision_at_ks_fams = []
    precision_at_ks_sfams = []
    precision_at_ks_folds = []
    precision_at_ks_classes =[]
    precision_at_ks_fams_map = []
    precision_at_ks_sfams_map = []
    precision_at_ks_folds_map = []
    precision_at_ks_classes_map = []
    for i in range(gtalign_res_df.shape[0]):
        pdb_id = gtalign_res_df.iloc[i,0].upper()
        try:
            query_fam = label_df.loc[label_df['pdb_id_chain'] == pdb_id, 
                                     'family'].iloc[0]
        except Exception as e:
            print(f'problem with {pdb_id} query_fam.')
        try:
            query_sfam = label_df.loc[label_df['pdb_id_chain'] == pdb_id, 
                                      'superfamily'].iloc[0]
        except Exception as e:
            print(f'problem with {pdb_id} query_sfam.')
        try:
            query_fold = label_df.loc[label_df['pdb_id_chain'] == pdb_id, 
                                      'fold'].iloc[0]
        except Exception as e:
            print(f'problem with {pdb_id} query_fold.')
        try:
            query_class = label_df.loc[label_df['pdb_id_chain'] == pdb_id,
                                      'class'].iloc[0]
        except Exception as e:
            print(f'problem with {pdb_id} query_class.')

        # For Precision calculation
        tp_fam = 0
        tp_sfam = 0
        tp_fold = 0
        tp_class = 0

        # For MAP@K calculation
        prec_at_k_fam = 0
        prec_at_k_sfam = 0
        prec_at_k_fold = 0
        prec_at_k_class = 0

        for rank, target in enumerate(gtalign_res_df.iloc[i,1:]):
            try:
                target_fam = label_df.loc[label_df['pdb_id_chain'] == target, 
                                            'family'].iloc[0]
            except Exception as e:
                # No similar proteins were found by gtalign for this index
                #print(e)
                target_fam = -1
            try:
                target_sfam = label_df.loc[label_df['pdb_id_chain'] == target, 
                                            'superfamily'].iloc[0]
            except Exception as e:
                # No similar proteins were found by gtalign for this index
                #print(e)
                targe_sfam = -1
            try:
                target_fold = label_df.loc[label_df['pdb_id_chain'] == target, 
                                            'fold'].iloc[0]
            except Exception as e:
                # No similar proteins were found by gtalign for this index
                #print(e)
                target_fold = -1
            try:
                target_class = label_df.loc[label_df['pdb_id_chain'] == target,
                                            'class'].iloc[0]
            except Exception as e:
                # No similar proteins were found by gtalign for this index
                #print(e)
                target_class = -1

            if query_fam == target_fam:
                tp_fam+=1
                prec_at_k_fam += (tp_fam / (rank+1))
            if query_sfam == target_sfam:
                tp_sfam+=1
                prec_at_k_sfam += (tp_sfam / (rank+1))
            if query_fold == target_fold:
                tp_fold+=1
                prec_at_k_fold += (tp_fold / (rank+1))
            if query_class == target_class:
                tp_class+=1
                prec_at_k_class += (tp_class / (rank+1))

        # For Precision@K
        precision_at_ks_fams.append(tp_fam/top_k)
        precision_at_ks_sfams.append(tp_sfam/top_k)
        precision_at_ks_folds.append(tp_fold/top_k)
        precision_at_ks_classes.append(tp_class/top_k)

        # For MAP@K
        precision_at_ks_fams_map.append(prec_at_k_fam/tp_fam)
        precision_at_ks_sfams_map.append(prec_at_k_sfam/tp_sfam)
        precision_at_ks_folds_map.append(prec_at_k_fold/tp_fold)
        precision_at_ks_classes_map.append(prec_at_k_class/tp_class)

    gtalign_patk_fam = np.mean(precision_at_ks_fams)
    gtalign_patk_sfam = np.mean(precision_at_ks_sfams)
    gtalign_patk_fold = np.mean(precision_at_ks_folds)
    gtalign_patk_class = np.mean(precision_at_ks_classes)
    gtalign_map_fam = np.mean(precision_at_ks_fams_map)
    gtalign_map_sfam = np.mean(precision_at_ks_sfams_map)
    gtalign_map_fold = np.mean(precision_at_ks_folds_map)
    gtalign_map_class = np.mean(precision_at_ks_classes_map) 

    # Calculate Precision@K's and MAP@K's for USalign
    usalign_res_df = read_usalign_results(usalign_results)

    precision_at_ks_fams = []
    precision_at_ks_sfams = []
    precision_at_ks_folds = []
    precision_at_ks_classes =[]
    precision_at_ks_fams_map = []
    precision_at_ks_sfams_map = []
    precision_at_ks_folds_map = []
    precision_at_ks_classes_map = []

    for i in range(usalign_res_df.shape[0]):
        pdb_id = usalign_res_df.iloc[i,0].upper()
        try:
            query_fam = label_df.loc[label_df['pdb_id_chain'] == pdb_id, 
                                     'family'].iloc[0]
        except Exception as e:
            print(f'problem with {pdb_id} query_fam.')
        try:
            query_sfam = label_df.loc[label_df['pdb_id_chain'] == pdb_id, 
                                      'superfamily'].iloc[0]
        except Exception as e:
            print(f'problem with {pdb_id} query_sfam.')
        try:
            query_fold = label_df.loc[label_df['pdb_id_chain'] == pdb_id, 
                                      'fold'].iloc[0]
        except Exception as e:
            print(f'problem with {pdb_id} query_fold.')
        try:
            query_class = label_df.loc[label_df['pdb_id_chain'] == pdb_id,
                                      'class'].iloc[0]
        except Exception as e:
            print(f'problem with {pdb_id} query_class.')

        # For Precision calculation
        tp_fam = 0
        tp_sfam = 0
        tp_fold = 0
        tp_class = 0

        # For MAP@K calculation
        prec_at_k_fam = 0
        prec_at_k_sfam = 0
        prec_at_k_fold = 0
        prec_at_k_class = 0

        for rank, target in enumerate(usalign_res_df.iloc[i,1:]):
            try:
                target_fam = label_df.loc[label_df['pdb_id_chain'] == target, 
                                            'family'].iloc[0]
            except Exception as e:
                # No similar proteins were found by gtalign for this index
                #print(e)
                target_fam = -1
            try:
                target_sfam = label_df.loc[label_df['pdb_id_chain'] == target, 
                                            'superfamily'].iloc[0]
            except Exception as e:
                # No similar proteins were found by gtalign for this index
                #print(e)
                targe_sfam = -1
            try:
                target_fold = label_df.loc[label_df['pdb_id_chain'] == target, 
                                            'fold'].iloc[0]
            except Exception as e:
                # No similar proteins were found by gtalign for this index
                #print(e)
                target_fold = -1
            try:
                target_class = label_df.loc[label_df['pdb_id_chain'] == target,
                                            'class'].iloc[0]
            except Exception as e:
                # No similar proteins were found by gtalign for this index
                #print(e)
                target_class = -1

            if query_fam == target_fam:
                tp_fam+=1
                prec_at_k_fam += (tp_fam / (rank+1))
            if query_sfam == target_sfam:
                tp_sfam+=1
                prec_at_k_sfam += (tp_sfam / (rank+1))
            if query_fold == target_fold:
                tp_fold+=1
                prec_at_k_fold += (tp_fold / (rank+1))
            if query_class == target_class:
                tp_class+=1
                prec_at_k_class += (tp_class / (rank+1))

        # For Precision@K
        precision_at_ks_fams.append(tp_fam/top_k)
        precision_at_ks_sfams.append(tp_sfam/top_k)
        precision_at_ks_folds.append(tp_fold/top_k)
        precision_at_ks_classes.append(tp_class/top_k)

        # For MAP@K
        precision_at_ks_fams_map.append(prec_at_k_fam/tp_fam)
        precision_at_ks_sfams_map.append(prec_at_k_sfam/tp_sfam)
        precision_at_ks_folds_map.append(prec_at_k_fold/tp_fold)
        precision_at_ks_classes_map.append(prec_at_k_class/tp_class)

    usalign_patk_fam = np.mean(precision_at_ks_fams)
    usalign_patk_sfam = np.mean(precision_at_ks_sfams)
    usalign_patk_fold = np.mean(precision_at_ks_folds)
    usalign_patk_class = np.mean(precision_at_ks_classes)
    usalign_map_fam = np.mean(precision_at_ks_fams_map)
    usalign_map_sfam = np.mean(precision_at_ks_sfams_map)
    usalign_map_fold = np.mean(precision_at_ks_folds_map)
    usalign_map_class = np.mean(precision_at_ks_classes_map) 

    print('Measure                       | Proteogram  | GTalign     | USalign    |')
    print(f'Precision@K for families      | {proteogram_patk_fam:.4f}      | {gtalign_patk_fam:.4f}      | {usalign_patk_fam:.4f}')
    print(f'Precision@K for superfamilies | {proteogram_patk_sfam:.4f}      | {gtalign_patk_sfam:.4f}      | {usalign_patk_sfam:.4f}')
    print(f'Precision@K for folds         | {proteogram_patk_fold:.4f}      | {gtalign_patk_fold:.4f}      | {usalign_patk_fold:.4f}')
    print(f'Precision@K for classes       | {proteogram_patk_class:.4f}      | {gtalign_patk_class:.4f}      | {usalign_patk_class:.4f}')
    print(f'MAP@K for families            | {proteogram_map_fam:.4f}      | {gtalign_map_fam:.4f}      | {usalign_map_fam:.4f}')
    print(f'MAP@K for superfamilies       | {proteogram_map_sfam:.4f}      | {gtalign_map_sfam:.4f}      | {usalign_map_sfam:.4f}')
    print(f'MAP@K for folds               | {proteogram_map_fold:.4f}      | {gtalign_map_fold:.4f}      | {usalign_map_fold:.4f}')
    print(f'MAP@K for classes             | {proteogram_map_class:.4f}      | {gtalign_map_class:.4f}      | {usalign_map_class:.4f}')
