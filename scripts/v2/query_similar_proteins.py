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
    python query_similar_proteins.py --pdb_file /path/to/protein.pdb --chain_id A \
        --cg_method martini --annot_file scope_eval_set_labels.tsv

Note:
  - May need to choose a different version of PyTorch with the proper CUDA support if you encounter CUDA errors.
    - Try: pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
  - The output image will show the query proteogram and the top-K similar proteograms side by side, with cosine similarity scores annotated.
  - A performance report (timing for each pipeline step) is always printed at the
    end. Passing --annot_file additionally reports class/fold/superfamily/family
    agreement between the query and its top-K results, using a label file in the
    format produced by evaluate_methods_v2.py (scope_eval_set_labels.tsv).
"""
import argparse
import gc
import os
import pickle
import warnings
from time import perf_counter

import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import pandas as pd
import torch
torch.backends.cudnn.enabled = False
import torch.nn as nn
import torchvision.transforms as transforms
from Bio.PDB.PDBParser import PDBConstructionWarning

from proteogram.v2 import ProteogramV2, Img2Vec
from proteogram.common import read_yaml

warnings.filterwarnings("ignore", category=PDBConstructionWarning)

SCOPE_LEVELS = ('class', 'fold', 'superfamily', 'family')


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


def read_checkpoint_max_image_size(model_file):
    """Peek at a checkpoint's meta to recover the max_image_size it was trained
    with. Embedding checkpoints (--loss triplet_hierarchy) self-describe it in a
    {'state_dict':..., 'meta':...} dict; legacy bare-state_dict / ViT checkpoints
    don't, so this returns None for those (callers fall back to a default).

    Loaded separately/early here because both the query-length cutoff and the
    proteogram creation happen before Img2Vec (which also exposes this via
    img_sim.embedding_meta) is constructed. The extra checkpoint load is
    negligible next to the multi-minute MD simulation this script runs.
    """
    try:
        loaded = torch.load(model_file, weights_only=False, map_location='cpu')
    except Exception:
        return None
    if isinstance(loaded, dict) and 'meta' in loaded:
        return loaded['meta'].get('max_image_size')
    return None


def load_annotations(annot_file):
    """Load a SCOPe label file as produced by evaluate_methods_v2.py
    (scope_eval_set_labels.tsv): columns pdb_id_chain, proteogram_file,
    class, fold, superfamily, family. Delimiter is sniffed so plain CSV
    also works.
    """
    return pd.read_csv(annot_file, sep=None, engine='python')


def lookup_label_row(label_df, filename):
    """Find the annotation row for a proteogram path, matched by
    proteogram_file basename first, falling back to pdb_id_chain vs.
    the filename stem. Returns None if not found.
    """
    basename = os.path.basename(filename)
    stem = os.path.splitext(basename)[0]
    if 'proteogram_file' in label_df.columns:
        match = label_df.loc[label_df['proteogram_file'].apply(
            lambda x: os.path.basename(str(x))) == basename]
        if not match.empty:
            return match.iloc[0]
    if 'pdb_id_chain' in label_df.columns:
        match = label_df.loc[label_df['pdb_id_chain'] == stem]
        if not match.empty:
            return match.iloc[0]
    return None


def compute_agreement(label_df, query_jpg, top_results, levels=SCOPE_LEVELS):
    """Compute per-level structural agreement between the query and its
    top-K results.

    Returns a dict {level: (n_agree, n_labeled)} or None if the query
    itself has no annotation row.
    """
    query_row = lookup_label_row(label_df, query_jpg)
    if query_row is None:
        return None

    agreement = {lv: [0, 0] for lv in levels}
    for path, _sim in top_results:
        row = lookup_label_row(label_df, path)
        if row is None:
            continue
        for lv in levels:
            if lv not in label_df.columns:
                continue
            agreement[lv][1] += 1
            if row[lv] == query_row[lv]:
                agreement[lv][0] += 1
    return agreement


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
    parser.add_argument('--target_size', type=int, default=None,
                        help='Size (in pixels, square) the query proteogram is padded '
                             'to before embedding with a CNN/ResNet18 checkpoint. Must '
                             'match the --max_image_size used when the model was '
                             'trained. If omitted, taken from the checkpoint meta '
                             '(embedding checkpoints self-describe max_image_size), '
                             'falling back to 200 for legacy checkpoints. Ignored for '
                             'ViT-B/16 checkpoints, which always resize to 224x224.')
    parser.add_argument('--sequence_len_lower_cutoff', type=int, default=20,
                        help='Minimum chain length (residues) accepted for the query '
                             'protein. Default: 20.')
    parser.add_argument('--sequence_len_upper_cutoff', type=int, default=None,
                        help='Maximum chain length (residues) accepted for the query '
                             'protein. If omitted, taken from the checkpoint meta '
                             "max_image_size (a query longer than the model's training "
                             'size could not be embedded without cropping anyway), '
                             'falling back to 200 for legacy checkpoints.')
    parser.add_argument('--cg_method', choices=['martini', 'atomistic'], default=None,
                        help="Coarse-grained (CG) MD method for proteogram creation. "
                             "'martini' uses the Martini 3-inspired CG model (faster); "
                             "'atomistic' forces full atomistic simulation. Defaults to "
                             "cg_method in config.yml, or atomistic if unset there.")
    parser.add_argument('--annot_file', default=None,
                        help='Optional SCOPe label file (TSV/CSV) with columns '
                             'pdb_id_chain, proteogram_file, class, fold, superfamily, '
                             'family — as produced by evaluate_methods_v2.py '
                             '(scope_eval_set_labels.tsv). When provided, and the query '
                             'itself is found in it, a class/fold/superfamily/family '
                             'agreement report is printed for the top-K results. '
                             'Omit to skip agreement reporting (timings are always shown).')
    args = parser.parse_args()

    config = read_yaml('config.yml')
    top_k      = args.top_k or config['top_k']
    model_file = os.path.expanduser(config['model_file'])
    embed_file = os.path.expanduser(config['embed_file'])
    corpus_dir = config.get('proteograms_for_sim_dir')
    if corpus_dir:
        corpus_dir = os.path.expanduser(corpus_dir)

    if args.cg_method is None:
        cg_method = config.get('cg_method') or None
    else:
        cg_method = None if args.cg_method == 'atomistic' else args.cg_method

    # Recover the training max_image_size from the checkpoint meta (if present)
    # so the query pad target and length cutoff can match training automatically
    # rather than defaulting to 200 and silently cropping/rescaling the query.
    ckpt_max_image_size = read_checkpoint_max_image_size(model_file)

    # Query pad target: explicit --target_size > checkpoint meta > 200.
    if args.target_size is not None:
        target_size = args.target_size
        if ckpt_max_image_size and target_size != ckpt_max_image_size:
            print(f'Warning: --target_size {target_size} differs from checkpoint '
                  f'max_image_size {ckpt_max_image_size}; using the explicit value, '
                  f'but this will mismatch how the model was trained.')
    elif ckpt_max_image_size:
        target_size = ckpt_max_image_size
        print(f'Using query pad target {target_size} from checkpoint meta.')
    else:
        target_size = 200

    # Query length upper cutoff: explicit CLI > checkpoint meta > 200. A query
    # longer than the model's training size can't be embedded without cropping,
    # so the cutoff tracks the same value.
    if args.sequence_len_upper_cutoff is not None:
        seq_upper_cutoff = args.sequence_len_upper_cutoff
    elif ckpt_max_image_size:
        seq_upper_cutoff = ckpt_max_image_size
    else:
        seq_upper_cutoff = 200

    args.output_dir = os.path.expanduser(args.output_dir)
    os.makedirs(args.output_dir, exist_ok=True)

    timings = {}

    # --- Step 1: Create the proteogram from the query PDB ---
    try:
        from openmm import Platform
        _openmm_platforms = [Platform.getPlatform(i).getName() for i in range(Platform.getNumPlatforms())]
        use_gpu = 'CUDA' in _openmm_platforms
    except Exception:
        use_gpu = False
    print(f'Creating proteogram for {args.pdb_file} (chain {args.chain_id}, '
          f'cg_method={cg_method or "atomistic"})...')

    _t0 = perf_counter()
    proteogram = ProteogramV2(
        pdb_path=args.pdb_file,
        output_dir=args.output_dir,
        chain_id=args.chain_id,
        calpha_atom_distance_cutoff=10,
        sequence_len_lower_cutoff=args.sequence_len_lower_cutoff,
        sequence_len_upper_cutoff=seq_upper_cutoff,
        use_gpu=use_gpu,
        cg_method=cg_method,
    )

    if not proteogram.is_valid_chain():
        raise ValueError(
            f'Chain {args.chain_id} has {len(proteogram.sequence)} residues, '
            f'outside allowed range [{proteogram.sequence_len_lower_cutoff}, '
            f'{proteogram.sequence_len_upper_cutoff}].')

    final_data, err = proteogram.calculate_proteogram(subtract_solvent_energies=True)
    timings['Proteogram creation (MD + maps)'] = perf_counter() - _t0
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
    if not os.path.isfile(query_jpg):
        raise RuntimeError(f'Failed to save query proteogram — file not found after imsave: {query_jpg}')
    print(f'Saved query proteogram to {query_jpg}')

    # --- Step 2: Load corpus embeddings and embed the query image ---
    print(f'Loading corpus embeddings from {embed_file}...')
    _t0 = perf_counter()
    with open(embed_file, 'rb') as f:
        corpus = pickle.load(f)
    timings['Corpus embeddings load'] = perf_counter() - _t0
    print(f'Corpus size: {len(corpus)} proteins')

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    _t0 = perf_counter()
    img_sim = Img2Vec(model_file, dataset_dir=[query_jpg], device=device)
    # Override transform to match training. ViT-B/16 checkpoints (trained with
    # --model vit) always resize to a fixed 224x224 (never pad) -- see
    # train_multiple_models_randomized_eval.py -- while CNN/ResNet18 checkpoints
    # are trained on proteograms padded (gray) to 200x200. Using the wrong one
    # here causes a shape mismatch at inference (ViT hard-asserts 224x224 input).
    if img_sim._ft_is_vit:
        img_sim.transform = transforms.Compose([
            transforms.Lambda(lambda img: img.convert('RGB').resize((224, 224))),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
    else:
        img_sim.transform = transforms.Compose([
            transforms.Lambda(lambda img: pad_to_size(img, target=target_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
    img_sim.dataset = corpus
    timings['Img2Vec init'] = perf_counter() - _t0

    print('Embedding query proteogram...')
    _t0 = perf_counter()
    with torch.no_grad():
        query_vec = img_sim.embed_image(query_jpg)
    timings['Query embedding'] = perf_counter() - _t0

    # --- Step 3: Cosine similarity against corpus ---
    print(f'Searching corpus for top {top_k} similar proteins...')
    _t0 = perf_counter()
    cosine = nn.CosineSimilarity(dim=1)
    scores = []
    with torch.no_grad():
        for path, emb in corpus.items():
            sim = cosine(query_vec, emb.to(query_vec.device))[0].item()
            scores.append((path, sim))

    scores.sort(key=lambda x: x[1], reverse=True)

    # Exclude the query protein itself if it appears in the corpus.
    query_basename = os.path.basename(query_jpg)
    scores_filtered = [(p, s) for p, s in scores
                       if os.path.basename(p) != query_basename]
    top_results = scores_filtered[:top_k]
    timings['Similarity search'] = perf_counter() - _t0

    if corpus_dir is None:
        print('Warning: proteograms_for_sim_dir not set in config.yml — '
              'result images will not be shown. Set it to the directory '
              'containing the corpus proteogram JPG files.')

    # --- Step 4: Print results and save result image ---
    print(f'\nTop {top_k} similar proteins:')
    for rank, (path, sim) in enumerate(top_results, 1):
        print(f'  {rank:>3}. {os.path.basename(path):<40}  cosine sim = {sim:.4f}')

    result_img_dir = os.path.join(args.output_dir, 'search_results')
    os.makedirs(result_img_dir, exist_ok=True)
    _t0 = perf_counter()
    img_sim.save_images(query_jpg, result_img_dir, scores_n_arr=top_results, corpus_dir=corpus_dir,
                        pad_fn=lambda img: pad_to_size(img, target=target_size))
    timings['Result image saved'] = perf_counter() - _t0
    print(f'\nResult image saved to {result_img_dir}/')

    # --- Step 5: Optional structural agreement report (needs an annotation file) ---
    if args.annot_file:
        print(f'\nLoading annotations from {args.annot_file}...')
        _t0 = perf_counter()
        label_df = load_annotations(args.annot_file)
        agreement = compute_agreement(label_df, query_jpg, top_results)
        timings['Agreement calculation'] = perf_counter() - _t0

        print(f'\nStructural agreement within top {top_k} results:')
        if agreement is None:
            print(f'  Query {query_basename!r} not found in {args.annot_file} — '
                  f'skipping agreement report (query must be present in the '
                  f'annotation file to compute agreement).')
        else:
            for lv in SCOPE_LEVELS:
                n_agree, n_labeled = agreement[lv]
                if n_labeled == 0:
                    print(f'  {lv:<12} no labeled results found for this level')
                    continue
                pct = 100 * n_agree / n_labeled
                print(f'  {lv:<12} {n_agree}/{n_labeled} matched ({pct:.1f}%)')
    else:
        print('\nNo --annot_file provided — skipping structural agreement report '
              '(timings only).')

    # --- Performance report ---
    total = sum(timings.values())
    print('\n' + '=' * 60)
    print('PERFORMANCE REPORT')
    print('=' * 60)
    for step, dur in timings.items():
        print(f'  {step:<32} {dur:>8.2f}s')
    print('  ' + '-' * 42)
    print(f'  {"TOTAL":<32} {total:>8.2f}s')
    print('=' * 60)