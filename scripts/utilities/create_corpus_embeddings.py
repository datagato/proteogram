"""
Create corpus embeddings for all JPGs found recursively across one or more
directories (e.g. both train/ and eval/ proteogram folders combined).

Saves a pickle of {filename: embedding_tensor} using filename-only keys so the
corpus is portable across machines. Use with query_similar_proteins.py.

Preprocessing is driven by the checkpoint's own metadata so it always matches
how the model was trained -- the same resolution logic as measure_similarity_v2.py:
  * grid size (pad/resize target): meta 'input_size', falling back to
    'max_image_size', then 200 for legacy checkpoints;
  * mode: meta 'resize' selects resize-to-grid vs. gray-pad-to-grid;
  * inclusion cutoff: meta 'max_image_size' -- proteograms larger than this in
    either dimension are excluded (matching ProteogramDataset at training time);
  * ViT-B/16 checkpoints always resize to a fixed 224x224 (never pad/filter).
CLI flags (--target_size, --resize/--no-resize, --max_image_size) override the
meta for legacy checkpoints that don't self-describe these values.

Note: filenames must be unique across all input directories. A warning is printed
if duplicates are detected — the last file with that name wins.

Usage:
    python create_corpus_embeddings.py \
        --model_file /path/to/model.pt \
        --embed_file /path/to/corpus_embeddings.pkl \
        --dirs /path/to/proteograms/train /path/to/proteograms/eval
"""
import argparse
import functools
import glob
import os
import pickle

import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image

from proteogram.v2 import Img2Vec


def pad_to_size(img, target=200, fill=128):
    """Pad a PIL image to target×target with gray, crop if oversized."""
    arr = np.array(img.convert('RGB'))
    H, W = arr.shape[0], arr.shape[1]

    def _pad(curr, tgt):
        d = tgt - curr
        if d <= 0:
            return (0, 0)
        p1 = d // 2
        return (p1, d - p1)

    padding = (_pad(H, target), _pad(W, target), (0, 0))
    arr = np.pad(arr, padding, constant_values=fill)
    arr = arr[:target, :target, :]
    return Image.fromarray(arr.astype('uint8'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Create corpus embeddings from JPGs across one or more directories.')
    parser.add_argument('--model_file', '-m', required=True,
                        help='Path to the trained model .pt file.')
    parser.add_argument('--embed_file', '-e', required=True,
                        help='Output path for the embeddings pickle.')
    parser.add_argument('--dirs', '-d', nargs='+', required=True,
                        help='One or more directories to search recursively for JPGs.')
    parser.add_argument('--target_size', type=int, default=None,
                        help='Override the square grid size (pixels) images are padded or '
                             'resized to before embedding. Normally taken from the checkpoint '
                             "meta ('input_size', falling back to 'max_image_size'); provide "
                             'this only for legacy checkpoints that lack meta (fallback 200). '
                             'Ignored for ViT-B/16 checkpoints, which always resize to 224x224.')
    parser.add_argument('--resize', action=argparse.BooleanOptionalAction, default=None,
                        help='Override preprocessing mode: --resize scales each proteogram to '
                             'the grid size, --no-resize gray-pads to it. Defaults to the '
                             "checkpoint meta 'resize', then pad mode for legacy checkpoints. "
                             'Ignored for ViT-B/16 checkpoints.')
    parser.add_argument('--max_image_size', type=int, default=None,
                        help='Override the inclusion cutoff (residues/pixels): proteograms larger '
                             'than this in either dimension are excluded, matching training. '
                             "Defaults to the checkpoint meta 'max_image_size', then the grid "
                             'size. In pad mode this equals the grid; in resize mode it can be '
                             'larger (bigger proteins are resized down to the grid).')
    args = parser.parse_args()

    # Collect all JPGs recursively across all input directories
    all_files = []
    for d in args.dirs:
        if not os.path.isdir(d):
            print(f'WARNING: {d} is not a directory, skipping.')
            continue
        found = glob.glob(os.path.join(d, '**', '*.jpg'), recursive=True)
        found += glob.glob(os.path.join(d, '**', '*.JPG'), recursive=True)
        print(f'  {d}: {len(found)} images')
        all_files.extend(found)

    if not all_files:
        raise RuntimeError('No JPG files found in the specified directories.')

    # Warn about basename collisions — last file with that name wins in the pickle
    basenames = [os.path.basename(f) for f in all_files]
    seen = set()
    dupes = {b for b in basenames if b in seen or seen.add(b)}
    if dupes:
        print(f'WARNING: {len(dupes)} duplicate filename(s) detected across directories. '
              f'Only the last occurrence of each will be kept. First few: '
              + ', '.join(sorted(dupes)[:5]))

    print(f'\nTotal images to embed: {len(all_files)}')

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'Using device: {device}')

    img_sim = Img2Vec(args.model_file, dataset_dir=all_files, device=device)

    # Report what the checkpoint self-describes, so a preprocessing mismatch is
    # visible rather than silent. Empty for legacy checkpoints (fallbacks used).
    if img_sim.embedding_meta:
        print(f'Checkpoint meta: {img_sim.embedding_meta}')
    else:
        print('Checkpoint meta: (none — legacy checkpoint; using fallbacks)')

    # Resolve grid size (pad/resize target), preprocessing mode, and the
    # residue/pixel inclusion cutoff to match how the model was trained. Prefer
    # the checkpoint's own recorded values (embedding checkpoints self-describe
    # input_size/resize/max_image_size via meta) so they can never silently
    # mismatch training; fall back to CLI flags / legacy defaults otherwise.
    # `input_size` is the grid the model was trained on (== max_image_size for
    # older, pad-only checkpoints).
    meta_grid = img_sim.embedding_meta.get('input_size') or img_sim.embedding_meta.get('max_image_size')
    meta_resize = img_sim.embedding_meta.get('resize')
    meta_cutoff = img_sim.embedding_meta.get('max_image_size')

    # Grid size: explicit --target_size > checkpoint meta > 200.
    if args.target_size is not None:
        grid_size = args.target_size
        if meta_grid and grid_size != meta_grid:
            print(f'Warning: --target_size {grid_size} differs from checkpoint '
                  f'input_size {meta_grid}; using the explicit value, but this '
                  f'will mismatch how the model was trained.')
    elif meta_grid:
        grid_size = meta_grid
        print(f'Using grid {grid_size} from checkpoint meta.')
    else:
        grid_size = 200

    # Preprocessing mode: explicit --resize/--no-resize > checkpoint meta > pad.
    if args.resize is not None:
        resize = args.resize
    elif meta_resize is not None:
        resize = bool(meta_resize)
    else:
        resize = False

    # Inclusion cutoff: explicit --max_image_size > checkpoint meta > grid size.
    if args.max_image_size is not None:
        cutoff = args.max_image_size
    elif meta_cutoff:
        cutoff = meta_cutoff
    else:
        cutoff = grid_size

    # Single preprocessing fn: resize-to-grid or gray-pad-to-grid.
    if resize:
        def _prep_fn(img):
            return img.convert('RGB').resize((grid_size, grid_size))
    else:
        _prep_fn = functools.partial(pad_to_size, target=grid_size)

    # ViT-B/16 checkpoints always resize to a fixed 224x224 (never pad) and
    # hard-assert that input shape -- see train_multiple_models_randomized_eval.py.
    # CNN/ResNet18 checkpoints use the resolved grid/mode. Using the wrong one
    # causes a shape mismatch at inference.
    if img_sim._ft_is_vit:
        img_sim.transform = transforms.Compose([
            transforms.Lambda(lambda img: img.convert('RGB').resize((224, 224))),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        print('ViT-B/16 checkpoint: resizing all images to 224x224 (no size filter).')
    else:
        img_sim.transform = transforms.Compose([
            transforms.Lambda(_prep_fn),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        print(f'{"Resizing" if resize else "Padding"} images to '
              f'{grid_size}x{grid_size} (cutoff={cutoff}).')

        # Exclude images larger than the cutoff (in either dimension) rather
        # than silently cropping them via pad_to_size -- matching how
        # ProteogramDataset excludes oversized images at training time
        # (train_multiple_models_randomized_eval.py), so the corpus never
        # contains embeddings of cropped, out-of-distribution proteograms. In
        # resize mode the cutoff can exceed the grid, so larger proteins are
        # kept and scaled down rather than excluded.
        kept, oversized = [], []
        for f in all_files:
            with Image.open(f) as im:
                w, h = im.size
            if w > cutoff or h > cutoff:
                oversized.append(f)
            else:
                kept.append(f)
        print(f'\nSize filter (cutoff={cutoff}): '
              f'{len(kept)} images kept, {len(oversized)} oversized images excluded.')
        if oversized:
            print('  First few excluded: '
                  + ', '.join(os.path.basename(f) for f in oversized[:5]))
        if not kept:
            raise RuntimeError(
                f'All {len(oversized)} images exceed cutoff={cutoff} -- '
                f'nothing left to embed. Pass a larger --max_image_size or check '
                f'the checkpoint meta.')
        all_files = kept
        img_sim.files = all_files

    print(f'Embedding {len(all_files)} images...')
    with torch.no_grad():
        img_sim.embed_dataset()

    os.makedirs(os.path.dirname(os.path.abspath(args.embed_file)), exist_ok=True)
    with open(args.embed_file, 'wb') as f:
        pickle.dump(img_sim.dataset, f)

    print(f'Saved {len(img_sim.dataset)} embeddings to {args.embed_file}')
