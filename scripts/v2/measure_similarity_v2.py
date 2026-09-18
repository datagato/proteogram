"""
Proteogram (image) search
"""
import argparse
import functools
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

from proteogram.v2 import Img2Vec
from proteogram.common import read_yaml


def pad_to_size(img, target=200, fill=128):
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
    # FAISS search options
    parser.add_argument('--faiss', action='store_true',
                        help='Search with a FAISS ANN index instead of brute-force '
                             'cosine similarity. Pays off on large corpora; on a few '
                             'thousand proteograms the brute-force path is usually '
                             'faster. Needs faiss-cpu or faiss-gpu installed.')
    parser.add_argument('--faiss_pq', action='store_true',
                        help='Use a product-quantised (IVF-PQ) FAISS index, which '
                             'trades some recall for much lower memory. Worth it '
                             'above roughly 100k proteograms.')
    parser.add_argument('--faiss_top_k', type=int, default=None,
                        help='Depth of the FAISS ranking to write out. Defaults to the '
                             'whole corpus, which forces an exhaustive search and gives '
                             'up the ANN speedup; set it to the largest K you evaluate '
                             'at to keep the search approximate.')
    parser.add_argument('--faiss_index_file', type=str, default=None,
                        help='Where to save or load the FAISS index. Defaults to '
                             'embed_file with a .faiss extension.')
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
    # Fallback grid size / preprocessing mode from config -- used only when the
    # checkpoint doesn't self-describe them (see below). Defaults preserve the
    # legacy behavior (pad to 200) for older checkpoints.
    config_pad_size = config.get('search_grid_size', 200)
    config_resize = bool(config.get('search_resize', False))

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

    if not prot_files:
        raise ValueError(f'No proteogram .jpg files found under {dataset_dir!r}. '
                         'Config paths are resolved against the working directory, '
                         'so check them if running from inside scripts/v2/.')

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'Using device: {device}')

    start = time()
    # Initialize Img2Vec with model from torchvision
    img_sim = Img2Vec(model_file, dataset_dir=prot_files, weights='DEFAULT', device=device)

    # Report what the checkpoint self-describes, so a preprocessing mismatch is
    # visible rather than silent. Empty for legacy checkpoints (config is used).
    if img_sim.embedding_meta:
        print(f'Checkpoint meta: {img_sim.embedding_meta}')
    else:
        print('Checkpoint meta: (none — legacy checkpoint; using config fallbacks)')

    # Resolve the search-time grid size and preprocessing mode (pad vs resize)
    # to match how the model was trained. Prefer the checkpoint's own recorded
    # values (embedding checkpoints self-describe input_size/resize via meta) so
    # they can never silently mismatch training; fall back to config for legacy
    # checkpoints that don't carry them. `input_size` is the grid the model was
    # trained on (== max_image_size for older, pad-only checkpoints).
    meta_grid = img_sim.embedding_meta.get('input_size') or img_sim.embedding_meta.get('max_image_size')
    meta_resize = img_sim.embedding_meta.get('resize')
    if meta_grid:
        search_grid_size = meta_grid
        search_resize = bool(meta_resize) if meta_resize is not None else config_resize
        print(f'Using grid {search_grid_size} and '
              f'{"resize" if search_resize else "pad"} mode from checkpoint meta.')
        if search_grid_size != config_pad_size or search_resize != config_resize:
            print(f'  Note: overrides config (search_grid_size={config_pad_size}, '
                  f'search_resize={config_resize}).')
    else:
        search_grid_size = config_pad_size
        search_resize = config_resize
        print(f'No grid in checkpoint meta; using config '
              f'(search_grid_size={search_grid_size}, {"resize" if search_resize else "pad"} mode).')

    # Single preprocessing fn used both for the embedding transform and for
    # result-image saving, so display images match how embeddings were computed.
    if search_resize:
        def _prep_fn(img):
            return img.convert('RGB').resize((search_grid_size, search_grid_size))
    else:
        _prep_fn = functools.partial(pad_to_size, target=search_grid_size)

    # Override transform to match training. ViT-B/16 checkpoints (trained with
    # --model vit) always resize to a fixed 224x224 -- see
    # train_multiple_models_randomized_eval.py -- and hard-assert that input
    # shape, so they get their own fixed 224 resize regardless of the above.
    if img_sim._ft_is_vit:
        img_sim.transform = transforms.Compose([
            transforms.Lambda(lambda img: img.convert('RGB').resize((224, 224))),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
    else:
        img_sim.transform = transforms.Compose([
            transforms.Lambda(_prep_fn),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        print(f'{"Resizing" if search_resize else "Padding"} search images to '
              f'{search_grid_size}x{search_grid_size} (from checkpoint meta or config).')
    print(f'Took {time()-start} seconds to initialize Img2Vec object.')

    # Create dataset and create embeddings
    start = time()
    with torch.no_grad():
        if embed:
           img_sim.embed_dataset()
           if not img_sim.dataset:
               raise ValueError('embed_dataset() produced no embeddings. Check that the '
                                'proteogram files are readable and preprocessing worked.')
           # Save embeddings
           with open(embed_file, 'wb') as pklout:
               pickle.dump(img_sim.dataset, pklout)
           print(f'Took {time()-start} seconds to create image embeddings.')
        else:
            if embed_file:
                with open(embed_file, 'rb') as pklin:
                    img_sim.dataset = pickle.load(pklin)
            if not img_sim.dataset:
                raise ValueError(f'No embeddings in {embed_file!r}, rerun with --embed '
                                 'to recompute them.')
            
        # Search to find similar images using cosine-similarity amongst embeddings.
        # Save all corpus results (including self-hit) so Recall@K can be computed at
        # any K and with/without self-hit at eval time.
        # Image saving is done separately at top_k to avoid PIL's 65500px dimension limit.
        start = time()
        n_results = len(prot_files)  # all including self-hit

        if args.faiss:
            if args.faiss_index_file:
                faiss_index_file = args.faiss_index_file
            else:
                faiss_index_file = os.path.splitext(embed_file)[0] + '.faiss'
            if os.path.exists(faiss_index_file) and not args.overwrite:
                print(f'Loading existing FAISS index from {faiss_index_file}')
                img_sim.load_faiss_index(faiss_index_file)
            else:
                print(f'Building FAISS index (use_pq={args.faiss_pq})')
                img_sim.build_faiss_index(use_pq=args.faiss_pq)
                img_sim.save_faiss_index(faiss_index_file)
            # An IVF search only returns what sits in the cells it probes, so
            # asking for the full corpus ranking makes it scan every cell.
            # Ranking less deeply is what keeps the search approximate, and fast.
            faiss_top_k = min(args.faiss_top_k or n_results, n_results)
            sim_time = img_sim.similarities_faiss(n=faiss_top_k,
                                                  save_result_images_dir=None,
                                                  pad_fn=_prep_fn)
            if faiss_top_k < n_results:
                print(f'Ranked the top {faiss_top_k} of {n_results} results per query; '
                      f'metrics beyond K={faiss_top_k} cannot be computed from this run.')
        else:
            sim_time = img_sim.similarities(n=n_results,
                                            save_result_images_dir=None,
                                            pad_fn=_prep_fn)

        # Save top-k result images with padding
        full_sim_dict = {k: list(v) for k, v in img_sim.sim_dict.items()}
        for image_path in img_sim.sim_dict:
            img_sim.sim_dict[image_path] = full_sim_dict[image_path][:top_k]
            img_sim.save_images(os.path.join(dataset_dir, image_path), save_images_dir,
                                scores_n_arr=img_sim.sim_dict[image_path],
                                pad_fn=_prep_fn, corpus_dir=dataset_dir)
        img_sim.sim_dict = full_sim_dict  # restore all results for CSV

        print(f'Took {sim_time} seconds to calculate similarities / perform search.')
        print(f'Took {time()-start} seconds overall (including optional image result saving).')

        # Create dataframe of results
        # Width the table to the deepest ranking actually produced. The FAISS
        # path can return fewer than n_results per query, and blank trailing
        # cells read back as NaN, which evaluate_methods_v2.py cannot parse.
        n_cols = min((len(v) for v in img_sim.sim_dict.values()), default=n_results)
        scores_tmp = [[''] * n_cols] * len(prot_files)
        df_res = pd.DataFrame(scores_tmp, columns=[str(i) for i in range(n_cols)])
        df_res['query_image'] = prot_files
        for i, image_path in enumerate(prot_files):
            try:
                scores = img_sim.sim_dict[os.path.basename(image_path)]
                df_res.iloc[i, :n_cols] = [f'{a},{b}' for (a, b) in scores[:n_cols]]
            except KeyError as e:
                print(f'Key error for {e}')
        if n_cols < n_results:
            print(f'Wrote {n_cols} of {n_results} possible result columns, limited by '
                  f'the query with the fewest hits.')
        # Reorder cols
        df_res.drop('query_image', inplace=True, axis=1)
        df_res.insert(0, 'query_image', prot_files)
        # Write results to file
        df_res.to_csv(results_file, sep='\t', index=False)
    




