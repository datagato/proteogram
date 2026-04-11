"""
Proteogram (image) search for a single structure
- Use a structure file from PDB
- Separate out chains/domains
- Save as separate domains in SCOPe naming format
- Convert domains to proteograms
- Search against a large DB of proteograms for similarity
"""
from time import time
import os
import pickle
import torch
import matplotlib.pyplot as plt

from proteogram.v1.image_similarity import Img2Vec
from proteogram.common.utils import read_yaml, split_by_chain_and_save
from proteogram.v1.proteogram import Proteogram


if __name__ == '__main__':
    torch.multiprocessing.set_start_method('spawn')
    # Run embedding vs loading saved embeddings (set to True if any 
    # changes to proteograms, but only if on CUDA-supported machine if
    # originally created with CUDA or CPU if only created on CPU).
    embed = True
    # Example downloaded structure path (experimental or predicted)
    #new_structure_path = './data/search_experiments/query/af3/AF-A0A3M6TU40-F1-model_v4.pdb'
    new_structure_path = './data/search_experiments/query/boltz_2025_0/boltz1-A0A3M6TU40_model_0.pdb'
    config = read_yaml('config.yml')
    top_k = config['top_k']
    model_file = config['model_file']
    embed_file = config['embed_file']
    embed_file_exists = bool(config['embed_file_exists'])
    results_file = config['proteogram_sim_results']
    dataset_dir = config['proteograms_dir_single_search']

    start = time()
    # Initialize Img2Vec with model from torchvision
    img_sim = Img2Vec(model_name_or_path=model_file,
                      dataset_dir=dataset_dir,
                      embed_file=embed_file,
                      weights='DEFAULT')
    print(f'Took {time()-start} seconds to initialize Img2Vec object.')

    # # Create dataset and create embeddings
    # start = time()
    # with torch.no_grad():
    #     if embed:
    #        img_sim.embed_dataset(str(dataset_dir))
    #        # Save embeddings
    #        with open(embed_file, 'wb') as pklout:
    #            pickle.dump(img_sim.dataset, pklout)
    #        print(f'Took {time()-start} seconds to create image embeddings.')
    #     else:
    #        with open(embed_file, 'rb') as pklin:
    #             img_sim.dataset = pickle.load(pklin)


    # Get chain "A" and make up sid
    pdb_id = os.path.basename(new_structure_path)[:-4]
    sid_id = 'dbo-a1_'
    chain_id = 'A'
    struct_dir = os.path.dirname(new_structure_path)
    scope_struct_path = struct_dir + os.sep + sid_id + '.ent'
    split_by_chain_and_save(new_structure_path,
                            chain_id=chain_id,
                            scope_sid=sid_id,
                            scope_pdbs_dir=struct_dir)

    # Create proteogram
    proteogram = Proteogram(new_structure_path)
    sequence = proteogram.get_sequence()
    distance_map = proteogram.calc_distogram(sequence, chain_id)
    hydro_map = proteogram.calc_hydrophobicity_map(sequence, distance_map)
    charge_map = proteogram.calc_charge_map(sequence, distance_map)
    final_data = proteogram.stack_data(distance_map, hydro_map, charge_map)
    image_file = os.path.join(struct_dir,
                                        f'{pdb_id}_{chain_id}.jpg')
    img = final_data.astype(float)
    plt.imsave(image_file, img)
        
    # Search to find similar images using cosine-similarity amongst embeddings
    # Save search results as images (TOP_K) to struct_dir
    start = time()
    if embed_file_exists == True:
        # We have embeddings saved as pytorch tensors to embed_file
        # Must have embed_file_exists set to True
        #img_sim.load_dataset()
        scores_n_arr, sim_calc_time = img_sim.similarities_new_image_scratch(image_path_query=image_file,
                                                            n=top_k,
                                                            save_result_images_dir=struct_dir,
                                                            use_prev_embeddings=True)
    else:
        # Need to embed the search dataset (database) and save it for future use
        # Must have embed_file_exists set to False
        scores_n_arr, sim_calc_time = img_sim.similarities_new_image_scratch(image_path_query=image_file,
                                                            n=top_k,
                                                            save_result_images_dir=struct_dir,
                                                            use_prev_embeddings=False)
    print(f'Took {sim_calc_time} seconds for similarity search.')
    print(scores_n_arr)
    
    print(f'Took {time()-start} seconds overall.')

    




