"""
Given a single PDB file, create its v2 proteogram, embed it with the trained
ResNet18, and return the top-K most similar proteins from a pre-computed corpus
using cosine similarity.

Prerequisites:
  1. A trained model (.pt) set as `model_file` in config.yml.
  2. A pre-computed corpus embedding pickle set as `embed_file` in config.yml,
     produced by measure_similarity_v2.py.

Usage example:
    python query_similar_proteins.py --pdb_file /path/to/protein.pdb --chain_id A
"""
import argparse
import gc
import os
import pickle
import warnings

import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from Bio.PDB.PDBParser import PDBConstructionWarning

from proteogram.v2 import ProteogramV2, Img2Vec
from proteogram.common import read_yaml

warnings.filterwarnings("ignore", category=PDBConstructionWarning)


def pad_to_size(img, target=200, fill=128):
    """Pad a PIL image to target×target with gray, then crop if oversized."""
    import numpy as np
    from PIL import Image as PILImage
    import numpy as np
    arr = np.array(img.convert('RGB'))
    H, W = arr.shape[0], arr.shape[1]

    def _pad(curr, tgt):
        d = tgt - curr
        if d <= 0:
            return (0, 0)
        p1 = d // 2
        return (p1, d - p1)

    padding = (_pad(H, target), _pad(W, target), (0, 0))
    arr = __import__('numpy').pad(arr, padding, constant_values=fill)
    arr = arr[:target, :target, :]
    return PILImage.fromarray(arr.astype('uint8'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Create a v2 proteogram for a single PDB and find similar proteins.')
    parser.add_argument('--pdb_file', '-p', required=True,
                        help='Path to the query PDB file.')
    parser.add_argument('--chain_id', '-c', required=True,
                        help='Chain ID to extract from the PDB file (e.g. A).')
    parser.add_argument('--output_dir', '-o', default='.',
                        help='Directory to save the query proteogram and result image. '
                             'Default: current directory.')
    parser.add_argument('--top_k', '-k', type=int, default=None,
                        help='Number of top similar proteins to return. '
                             'Defaults to top_k in config.yml.')
    args = parser.parse_args()

    config = read_yaml('config.yml')
    top_k      = args.top_k or config['top_k']
    model_file = config['model_file']
    embed_file = config['embed_file']
    corpus_dir = config.get('proteograms_for_sim_dir')

    os.makedirs(args.output_dir, exist_ok=True)

    # --- Step 1: Create the proteogram from the query PDB ---
    use_gpu = torch.cuda.is_available()
    print(f'Creating proteogram for {args.pdb_file} (chain {args.chain_id})...')

    proteogram = ProteogramV2(
        pdb_path=args.pdb_file,
        output_dir=args.output_dir,
        chain_id=args.chain_id,
        calpha_atom_distance_cutoff=10,
        sequence_len_lower_cutoff=20,
        sequence_len_upper_cutoff=200,
        use_gpu=use_gpu,
    )

    if not proteogram.is_valid_chain():
        raise ValueError(
            f'Chain {args.chain_id} has {len(proteogram.sequence)} residues, '
            f'outside allowed range [{proteogram.sequence_len_lower_cutoff}, '
            f'{proteogram.sequence_len_upper_cutoff}].')

    final_data, err = proteogram.calculate_proteogram(subtract_solvent_energies=True)
    if err:
        print(f'Warning during proteogram calculation: {err}')
    if final_data is None:
        raise RuntimeError('Proteogram calculation returned no data.')

    query_name = os.path.splitext(os.path.basename(args.pdb_file))[0]
    query_jpg  = os.path.join(args.output_dir, f'{query_name}.jpg')
    plt.imsave(query_jpg, final_data.astype('uint8'))
    plt.close('all')
    del proteogram, final_data
    gc.collect()
    print(f'Saved query proteogram to {query_jpg}')

    # --- Step 2: Load corpus embeddings and embed the query image ---
    print(f'Loading corpus embeddings from {embed_file}...')
    with open(embed_file, 'rb') as f:
        corpus = pickle.load(f)
    print(f'Corpus size: {len(corpus)} proteins')

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    img_sim = Img2Vec(model_file, dataset_dir=[query_jpg], device=device)
    img_sim.transform = transforms.Compose([
        transforms.Lambda(lambda img: pad_to_size(img, target=200)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    img_sim.dataset = corpus

    print('Embedding query proteogram...')
    with torch.no_grad():
        query_vec = img_sim.embed_image(query_jpg)

    # --- Step 3: Cosine similarity against corpus ---
    print(f'Searching corpus for top {top_k} similar proteins...')
    cosine = nn.CosineSimilarity(dim=1)
    scores = []
    with torch.no_grad():
        for path, emb in corpus.items():
            sim = cosine(query_vec, emb)[0].item()
            scores.append((path, sim))

    scores.sort(key=lambda x: x[1], reverse=True)
    top_results = scores[:top_k]

    # --- Step 4: Print results and save result image ---
    print(f'\nTop {top_k} similar proteins:')
    for rank, (path, sim) in enumerate(top_results, 1):
        print(f'  {rank:>3}. {os.path.basename(path):<40}  cosine sim = {sim:.4f}')

    result_img_dir = os.path.join(args.output_dir, 'search_results')
    os.makedirs(result_img_dir, exist_ok=True)
    img_sim.save_images(query_jpg, result_img_dir, scores_n_arr=top_results, corpus_dir=corpus_dir)
    print(f'\nResult image saved to {result_img_dir}/')