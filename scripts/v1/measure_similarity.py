"""
Proteogram (image) search
"""
from time import time
import glob
import os
import pandas as pd
import pickle
import shutil
import torch

from proteogram.v1.image_similarity import Img2Vec
from proteogram.common.utils import read_yaml


if __name__ == '__main__':
    # Run embedding vs loading saved embeddings (True if any 
    # changes to proteograms)
    embed = True

    config = read_yaml('config.yml')
    top_k = config['top_k']
    model_file = config['model_file']
    embed_file = config['embed_file']
    results_file = config['proteogram_sim_results']
    dataset_dir = config['proteograms_for_sim_dir']
    save_images_dir =config['search_images_dir']

    # If the output dir exists, don't recreate, otherwise make one
    if os.path.exists(save_images_dir):
        print(f'Directory {save_images_dir} exists, will use and may overwrite.')
    else:
        os.makedirs(save_images_dir)

    start = time()
    # Initialize Img2Vec with model from torchvision
    img_sim = Img2Vec(model_file, dataset_dir=dataset_dir, weights='DEFAULT', device='cuda')
    print(f'Took {time()-start} seconds to initialize Img2Vec object.')

    # Create dataset and create embeddings
    start = time()
    with torch.no_grad():
        if embed:
           img_sim.embed_dataset(str(dataset_dir))
           # Save embeddings
           with open(embed_file, 'wb') as pklout:
               pickle.dump(img_sim.dataset, pklout)
           print(f'Took {time()-start} seconds to create image embeddings.')
        else:
            if embed_file:
                with open(embed_file, 'rb') as pklin:
                    img_sim.dataset = pickle.load(pklin)
            
        # Search to find similar images using cosine-similarity amongst embeddings
        # Save search results as images (TOP_K) to save_images_dir
        start = time()
        sim_time = img_sim.similarities(n=top_k,
                                        save_result_images_dir=save_images_dir,
                                        use_prev_embeddings=True)
        
        print(f'Took {sim_time} seconds to calculate similarities / perform search.')
        print(f'Took {time()-start} seconds overall (including optional image result saving).')

        prot_files = glob.glob(os.path.join(dataset_dir, '**', '*.jpg'), recursive=True)

        # Create dataframe of results
        scores_tmp = [[''] * top_k] * len(prot_files)
        df_res = pd.DataFrame(scores_tmp, columns=[[str(i) for i in range(top_k)]])
        df_res['query_image'] = prot_files
        for i, image_path in enumerate(prot_files):
            try:
                scores = img_sim.sim_dict[image_path]
                # Locate the row to update and assign scores
                row_i = df_res[df_res['query_image'] == image_path].index[i]
                df_res.iloc[row_i,:top_k] = [f'{a},{b}' for (a, b) in scores]
            except KeyError as e:
                print(f'Key error for {e}')
        # Reorder cols
        df_res.drop('query_image', inplace=True, axis=1)
        df_res.insert(0, 'query_image', prot_files)
        # Write results to file
        df_res.to_csv(results_file, sep='\t', index=False)
    




