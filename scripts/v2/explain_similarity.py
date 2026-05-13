"""Grad-CAM residue-pair importance maps for proteogram similarity pairs.

For each query, produces a 3-panel PNG figure (original proteogram | Grad-CAM
heatmap | overlay) showing which residue-pair interactions most influenced the
cosine similarity score.  A raw ``.npy`` heatmap array is also saved alongside
the figure for downstream analysis.

Because proteogram pixels directly encode pairwise residue interactions
(row i, column j → residue-i vs. residue-j interaction), the heatmap is
directly interpretable as a biological residue-pair attribution map.

Modes
-----
Single pair
    Explain one specific query→target pair.

    python explain_similarity.py \\
        --query /path/to/d3kfda_.jpg \\
        --target /path/to/d1yl4r1.jpg \\
        --output_dir gradcam_results/

Batch (top-K hits per query)
    Explain the top-K most similar proteins for every query in the eval set.

    python explain_similarity.py \\
        --batch \\
        --top_k 3 \\
        --output_dir gradcam_results/

Batch (worst-performing queries)
    Explain only the N queries that appear worst in the similarity results TSV
    (useful for diagnosing failure cases).

    python explain_similarity.py \\
        --batch \\
        --n_worst 50 \\
        --top_k 1 \\
        --output_dir gradcam_results/
"""

import argparse
import os
import pickle
import sys

import torch

# Allow running as a script without a full package install
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from proteogram.v2 import Img2Vec
from proteogram.v2.gradcam import GradCAM, _preprocess_image
from proteogram.common import read_yaml

import numpy as np
from PIL import Image


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_img2vec(config: dict, device: str) -> Img2Vec:
    """Initialise Img2Vec and load the corpus embeddings."""
    model_file = config['model_file']
    embed_file = config.get('embed_file')

    img_sim = Img2Vec(model_file, dataset_dir=[], device=device)

    if embed_file and os.path.exists(embed_file):
        with open(embed_file, 'rb') as fh:
            img_sim.dataset = pickle.load(fh)
        print(f'Loaded {len(img_sim.dataset):,} embeddings from {embed_file}')
    else:
        print(
            'WARNING: embed_file not found in config or on disk.  '
            'Corpus embeddings will not be loaded — batch mode unavailable.'
        )
    return img_sim


def _explain_pair(gcam: GradCAM,
                  query_path: str,
                  target_path: str,
                  output_dir: str,
                  query_sequence: str = None) -> None:
    """Run Grad-CAM for one query→target pair and save outputs."""
    if not os.path.exists(query_path):
        print(f'  SKIP: query not found: {query_path}')
        return
    if not os.path.exists(target_path):
        print(f'  SKIP: target not found: {target_path}')
        return

    query_name  = os.path.splitext(os.path.basename(query_path))[0]
    target_name = os.path.splitext(os.path.basename(target_path))[0]

    heatmap, cos_sim = gcam.compute_from_paths(query_path, target_path)

    query_img = np.array(Image.open(query_path).convert('RGB'))
    fig_path = gcam.save_figure(
        heatmap=heatmap,
        query_img=query_img,
        cos_sim=cos_sim,
        query_name=query_name,
        target_name=target_name,
        output_dir=output_dir,
        query_sequence=query_sequence,
    )
    npy_path = gcam.save_npy(heatmap, query_name, target_name, output_dir)
    print(f'  Saved: {fig_path}  (cos_sim={cos_sim:.4f})')
    print(f'  Saved: {npy_path}')


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description='Grad-CAM residue-pair importance maps for proteogram pairs.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # ── Single-pair mode ────────────────────────────────────────────────────
    single = parser.add_argument_group('Single-pair mode')
    single.add_argument(
        '--query', '-q', type=str, default=None,
        help='Path to the query proteogram JPG.',
    )
    single.add_argument(
        '--target', '-t', type=str, default=None,
        help='Path to the target proteogram JPG.',
    )
    single.add_argument(
        '--query_seq', type=str, default=None,
        help=(
            'One-letter amino acid sequence of the query protein.  '
            'When provided, adds residue tick labels to the figure axes.'
        ),
    )

    # ── Batch mode ──────────────────────────────────────────────────────────
    batch = parser.add_argument_group('Batch mode')
    batch.add_argument(
        '--batch', action='store_true',
        help=(
            'Explain the top-K hits for every query in the corpus '
            '(or the N worst queries when --n_worst is set).'
        ),
    )
    batch.add_argument(
        '--top_k', type=int, default=1,
        help='Number of top hits to explain per query.',
    )
    batch.add_argument(
        '--n_worst', type=int, default=None,
        help=(
            'Explain only the N queries with the lowest mean cosine score '
            '(i.e. worst-performing).  When omitted, all queries are explained.'
        ),
    )

    # ── Common ──────────────────────────────────────────────────────────────
    parser.add_argument(
        '--output_dir', '-o', type=str, default='gradcam_output',
        help='Directory to write Grad-CAM PNG figures and .npy files.',
    )
    parser.add_argument(
        '--no_npy', action='store_true',
        help='Skip saving raw .npy heatmap files.',
    )

    args = parser.parse_args()

    # ── Validate ────────────────────────────────────────────────────────────
    if not args.query and not args.batch:
        parser.error('Specify --query/--target for single-pair mode, or --batch for batch mode.')
    if args.query and not args.target:
        parser.error('--target is required when --query is provided.')

    config = read_yaml('config.yml')
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'Device: {device}')

    img_sim = _load_img2vec(config, device)
    gcam = GradCAM(embed_net=img_sim.embed, device=device)
    corpus_dir = config.get('proteograms_for_sim_dir', '')
    os.makedirs(args.output_dir, exist_ok=True)

    # ── Single-pair mode ────────────────────────────────────────────────────
    if args.query:
        print(f'\nExplaining pair: {args.query}  →  {args.target}')
        _explain_pair(
            gcam=gcam,
            query_path=args.query,
            target_path=args.target,
            output_dir=args.output_dir,
            query_sequence=args.query_seq,
        )
        return

    # ── Batch mode ──────────────────────────────────────────────────────────
    if not img_sim.dataset:
        print('ERROR: batch mode requires corpus embeddings.  Check embed_file in config.yml.')
        return

    # Build sim_dict if not already populated
    if not img_sim.sim_dict:
        results_file = config.get('proteogram_sim_results')
        if results_file and os.path.exists(results_file):
            import pandas as pd
            results_df = pd.read_csv(results_file, sep='\t')
            for _, row in results_df.iterrows():
                q_key = os.path.basename(row.iloc[0])
                hits = []
                for col_val in row.iloc[1:]:
                    if isinstance(col_val, str) and ',' in col_val:
                        fname, score = col_val.rsplit(',', 1)
                        hits.append((fname.strip(), float(score)))
                img_sim.sim_dict[q_key] = hits
            print(f'Loaded {len(img_sim.sim_dict):,} queries from {results_file}')
        else:
            print('No sim_dict and no results TSV found.  Running brute-force search ...')
            img_sim.embed_dataset()
            img_sim.similarities(n=args.top_k + 1)

    queries = list(img_sim.sim_dict.keys())

    # Optionally restrict to N worst-performing queries
    if args.n_worst is not None:
        def _mean_score(key):
            hits = img_sim.sim_dict.get(key, [])
            non_self = [s for _, s in hits if _ != key]
            return np.mean(non_self) if non_self else 1.0

        queries = sorted(queries, key=_mean_score)[:args.n_worst]
        print(f'Explaining {len(queries)} worst-performing queries.')
    else:
        print(f'Explaining all {len(queries):,} queries (top-{args.top_k} hits each).')

    done = 0
    for q_key in queries:
        q_path = os.path.join(corpus_dir, q_key)
        if not q_path.endswith('.jpg'):
            q_path += '.jpg'

        hits = img_sim.sim_dict.get(q_key, [])
        # Skip self-hit (score ≈ 1.0 at rank 0)
        non_self_hits = [(k, s) for k, s in hits if k != q_key]

        for target_key, _ in non_self_hits[:args.top_k]:
            t_path = os.path.join(corpus_dir, target_key)
            if not t_path.endswith('.jpg'):
                t_path += '.jpg'

            _explain_pair(
                gcam=gcam,
                query_path=q_path,
                target_path=t_path,
                output_dir=args.output_dir,
            )
            done += 1

    print(f'\nDone. {done} Grad-CAM maps saved to {os.path.abspath(args.output_dir)}')


if __name__ == '__main__':
    main()
