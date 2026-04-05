"""
Proteogram (image) search
"""
import argparse
from time import time
import glob
import os
import numpy as np
import pandas as pd
import pickle
import shutil
import torch
import torchvision.transforms as transforms
from PIL import Image

from proteogram.image_similarity_v2 import Img2Vec
from proteogram.utils import read_yaml


def pad_to_200(img, target=200, fill=128):
    """Pad a PIL image to target×target with gray (matching training script).

    Images smaller than target are center-padded; images larger are cropped
    from the top-left to target×target.
    """
    arr = np.array(img.convert('RGB'))
    H, W = arr.shape[0], arr.shape[1]

    def get_pad(curr, tgt):
        d = tgt - curr
        if d <= 0:
            return (0, 0)
        p1 = d // 2
        return (p1, d - p1)

    padding = (get_pad(H, target), get_pad(W, target), (0, 0))
    arr = np.pad(arr, padding, constant_values=fill)
    arr = arr[:target, :target, :]  # crop if oversized
    return Image.fromarray(arr.astype(np.uint8))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Proteogram image similarity search.')
    parser.add_argument('--exclude_classes', '-x', default=None,
                        help='Comma-separated SCOPe class names to exclude (e.g. "g,h").')
    parser.add_argument('--overwrite', action='store_true',
                        help='Recreate search_images_dir and overwrite embed/results files '
                             'without prompting.')
    parser.add_argument('--embed', action=argparse.BooleanOptionalAction, default=True,
                        help='Recompute and save embeddings (default: True). '
                             'Use --no-embed to load from embed_file instead.')
    args = parser.parse_args()

    # Run embedding vs loading saved embeddings
    embed = args.embed

    config = read_yaml('config.yml')
    top_k = config['top_k']
    model_file = config['model_file']
    embed_file = config['embed_file']
    results_file = config['proteogram_sim_results']
    dataset_dir = config['proteograms_for_sim_dir']
    save_images_dir = config['search_images_dir']

    def _confirm_overwrite(path, label, is_dir=False):
        """Prompt user to overwrite an existing file/dir; return True if proceeding."""
        if not os.path.exists(path):
            return True
        if args.overwrite:
            if is_dir:
                shutil.rmtree(path)
            else:
                os.remove(path)
            return True
        ans = input(f'{label} already exists at {path}. Overwrite? [y/N]: ').strip().lower()
        if ans == 'y':
            if is_dir:
                shutil.rmtree(path)
            else:
                os.remove(path)
            return True
        print(f'Keeping existing {label}. Pass --overwrite to skip this prompt.')
        return False

    if _confirm_overwrite(save_images_dir, 'search_images_dir', is_dir=True):
        os.makedirs(save_images_dir, exist_ok=True)

    _confirm_overwrite(embed_file, 'embed_file')
    _confirm_overwrite(results_file, 'results_file')

    prot_files = sorted(glob.glob(os.path.join(dataset_dir, '*.jpg')))

    if args.exclude_classes:
        excluded = {c.strip() for c in args.exclude_classes.split(',')}
        # Parse the CLA file to map SID -> SCOPe class (first component of SCCS)
        excluded_sids = set()
        with open(config['scope_cla_file']) as f:
            for line in f:
                if line.startswith('#') or not line.strip():
                    continue
                fields = line.split()
                if len(fields) >= 4:
                    sid, sccs = fields[0], fields[3]
                    if sccs.split('.')[0] in excluded:
                        excluded_sids.add(sid)
        before = len(prot_files)
        prot_files = [f for f in prot_files
                      if os.path.splitext(os.path.basename(f))[0] not in excluded_sids]
        print(f'Excluded {before - len(prot_files)} proteograms from class(es): '
              + ', '.join(sorted(excluded)))
        
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'Using device: {device}')

    start = time()
    # Initialize Img2Vec with model from torchvision
    img_sim = Img2Vec(model_file, dataset_dir=prot_files, weights='DEFAULT', device=device)
    # Override transform to match training: pad to 200x200 with gray rather than resize
    img_sim.transform = transforms.Compose([
        transforms.Lambda(pad_to_200),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    print(f'Took {time()-start} seconds to initialize Img2Vec object.')

    # Create dataset and create embeddings
    start = time()
    with torch.no_grad():
        if embed:
           img_sim.embed_dataset()
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
                                        pad_fn=pad_to_200)
        
        print(f'Took {sim_time} seconds to calculate similarities / perform search.')
        print(f'Took {time()-start} seconds overall (including optional image result saving).')

        # Create dataframe of results
        scores_tmp = [[''] * top_k] * len(prot_files)
        df_res = pd.DataFrame(scores_tmp, columns=[[str(i) for i in range(top_k)]])
        df_res['query_image'] = prot_files
        for i, image_path in enumerate(prot_files):
            try:
                scores = img_sim.sim_dict[image_path]
                df_res.iloc[i, :top_k] = [f'{a},{b}' for (a, b) in scores]
            except KeyError as e:
                print(f'Key error for {e}')
        # Reorder cols
        df_res.drop('query_image', inplace=True, axis=1)
        df_res.insert(0, 'query_image', prot_files)
        # Write results to file
        df_res.to_csv(results_file, sep='\t', index=False)
    




