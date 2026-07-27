"""
This script trains a CNN on the Proteogram dataset, with options for architecture (from-scratch ConvNet, pretrained ResNet18, or pretrained ViT-B/16), hyperparameters (epochs, batch size, learning rate), and logging.  It includes early stopping based on validation loss, and saves the best model weights to disk.  The training and validation loss curves are plotted and saved to a file.  After training, the model is evaluated on a held-out test set, with per-class accuracies and a classification report printed to the console.

Requires a GPU for reasonable training time, especially for ResNet18 and ViT-B/16.  For reproducibility, a random seed controls the train/val/test split and all stochastic operations.

Data location and other parameters can be configured via command-line arguments or a config.yml file.  Command-line arguments take precedence over config.yml. The Proteogram dataset should be prepared in advance using the create_v2_proteograms.py script. The root directory ("training_data_dir" in config.yml) should contain all proteogram images in a flat layout (no train/eval subdirectories). A random seed (--seed) is used to reproducibly split images into train, validation, and held-out test sets. An annotation TSV file is also required, which should be specified via the --tsv_file argument or included in config.yml as "tsv_file".  The model weights will be saved to the path specified by "cnn_model_file_prefix" in config.yml, with a suffix indicating the architecture and hyperparameters.

Note on --model vit: ViT-B/16 uses fixed, learned positional embeddings sized for a 224x224 input split into 16x16 patches, unlike the CNN/ResNet18 paths which can train on padded, variable-size proteograms. Selecting --model vit therefore always forces --resize (never pad) and overrides --max_image_size to 224, regardless of what was passed on the command line.

Here is more information about the SCOPe dataset: https://scop.berkeley.edu

Note on --loss: three loss functions are available. 'ce' (default) is cross-entropy
with label smoothing, trained against a single --level (class/fold/superfamily/family).
'focal' is focal loss (--focal_gamma controls the focusing strength; gamma=0 reduces to
plain cross-entropy), useful when classes at the chosen --level are imbalanced or
dominated by easy examples. 'triplet_hierarchy' abandons single-level classification
entirely and trains a --model resnet18 embedding model with HierarchicalTripletLoss, a
batch-hard triplet loss whose margins (--triplet_margins) scale with SCOPe hierarchy
distance (same family/superfamily/fold/class/different class); batches are built by
HierarchicalPKSampler using P-K sampling (--triplet_p, --triplet_k) so the model learns
an embedding space for retrieval across the whole hierarchy at once rather than a
classifier for one level. See proteogram/v2/losses.py for the loss implementations.

 Usage example:
    python train_multiple_models_randomized_eval.py --model resnet18 --epochs 50 --batch_size 32 --lr 1e-4 --seed 42
    python train_multiple_models_randomized_eval.py --model vit --epochs 50 --batch_size 16 --lr 1e-4 --seed 42
"""
import copy
from sched import scheduler
import pandas as pd
import numpy as np
import glob
import os
import random
import matplotlib
import matplotlib.pyplot as plt
from PIL import Image
import argparse

from sklearn.metrics import classification_report, roc_auc_score
from sklearn.model_selection import train_test_split

import torch
from torch.utils.data import random_split, Dataset, DataLoader, Subset
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.models as tv_models
import torchvision.transforms as transforms

from proteogram.common import read_yaml
from proteogram.v2 import HierarchicalTripletLoss, HierarchicalPKSampler, SCOPE_LEVELS


matplotlib.use('agg')

class ProteogramDataset(Dataset):
    """SCOPe-based Proteograms dataset."""

    def __init__(self, tsv_file, root_dir, pad=True, level='class', new_size=128, transform=None,
                 names_to_labels=None, min_class_size=0, min_image_size=0, max_image_size=0,
                 exclude_classes=None, emit_hierarchy=False, min_family_size=0):
        """
        Arguments
        ---------
        tsv_file : string
            Path to the tsv file with annotations.
        root_dir : string
            Directory with all the images.
        level : string
            The scope category level [class, fold, superfamily, family]
        transform : callable, optional
            Optional transform to be applied on a sample.
        names_to_labels : dict, optional
            Pre-built name→integer mapping from the train dataset. When supplied
            (eval mode), samples whose class is absent from this mapping are
            dropped rather than creating a new mapping.
        min_class_size : int
            Classes with fewer than this many samples are excluded (train mode
            only, ignored when names_to_labels is supplied). Default: 0 (keep all).
        min_image_size : int
            Images whose width or height is below this threshold (in pixels) are
            excluded. Since proteograms are NxN where N = residue count, this is
            equivalent to filtering by sequence length. Default: 0 (keep all).
        max_image_size : int
            Images whose width or height exceeds this threshold (in pixels) are
            excluded. Prevents silent cropping of large proteins when padding mode
            is used. Default: 0 (keep all).
        exclude_classes : list of str, optional
            Class names to exclude entirely (train mode only; eval samples for
            excluded classes are automatically dropped via the names_to_labels
            mechanism). Default: None (keep all).
        emit_hierarchy : bool
            If True, also build per-sample hierarchy labels covering all four
            SCOPe levels (see SCOPE_LEVELS) and have __getitem__ return
            (image, hierarchy_label_tensor) instead of (image, label). Used by
            the --loss triplet_hierarchy path in train_multiple_models_
            randomized_eval.py; the `level`/`labels`/`names_to_labels`
            attributes are still built as normal so stratified splitting is
            unaffected. Default: False.
        min_family_size : int
            Only used when emit_hierarchy=True. Families (SCOPeFamily) with
            fewer than this many samples are excluded. Without this, sparse
            families force HierarchicalPKSampler's with-replacement fallback
            to fill --triplet_k with duplicate copies of the same image
            rather than distinct examples, giving the loss a trivial
            "identical to itself" positive instead of a real one. Default: 0
            (keep all).
        """
        self.annot_frame = pd.read_csv(tsv_file, sep='\t')
        self.root_dir = root_dir
        self.files = glob.glob(os.path.join(self.root_dir, '*.jpg'))

        if min_image_size > 0:
            small = []
            valid_files = []
            for f in self.files:
                w, h = Image.open(f).size
                if w < min_image_size or h < min_image_size:
                    small.append(os.path.basename(f))
                else:
                    valid_files.append(f)
            self.files = valid_files
            if small:
                print(f'WARNING: {len(small)} image(s) smaller than '
                      f'{min_image_size}x{min_image_size} px excluded. '
                      f'First few: {small[:5]}')

        if max_image_size > 0:
            large = []
            valid_files = []
            for f in self.files:
                w, h = Image.open(f).size
                if w > max_image_size or h > max_image_size:
                    large.append(os.path.basename(f))
                else:
                    valid_files.append(f)
            self.files = valid_files
            if large:
                print(f'WARNING: {len(large)} image(s) larger than '
                      f'{max_image_size}x{max_image_size} pixels excluded. '
                      f'First few: {large[:5]}')
        self.transform = transform
        self.pad = pad
        self.level = level # class, fold, superfamily or family
        self.new_size = new_size

        level_col = {'class': 'SCOPeClass', 'fold': 'SCOPeFold',
                     'superfamily': 'SCOPeSuperfamily', 'family': 'SCOPeFamily'}
        col = level_col.get(self.level, 'SCOPeFamily')

        # Look up annotation label for each image file
        self.label_names = []
        missing = []
        for file in self.files:
            bname = os.path.basename(file).replace('.jpg', '')
            row = self.annot_frame[self.annot_frame['SCOPeID'] == bname]
            if len(row) == 0:
                missing.append(bname)
                self.label_names.append(None)
            else:
                self.label_names.append(row.iloc[0][col])

        if missing:
            print(f'WARNING: {len(missing)} image(s) not found in TSV annotations '
                  f'and will be excluded. First few: {missing[:5]}')

        # Drop files with no annotation
        paired = [(f, l) for f, l in zip(self.files, self.label_names) if l is not None]
        if not paired:
            raise RuntimeError(
                f'No images matched any entry in the TSV SCOPeID column. '
                f'Check that image basenames (without .jpg) match the SCOPeID values.')
        self.files, self.label_names = zip(*paired)
        self.files, self.label_names = list(self.files), list(self.label_names)

        if names_to_labels is None:
            # Train mode: optionally exclude named classes
            if exclude_classes:
                excluded_set = set(exclude_classes)
                unknown = excluded_set - set(self.label_names)
                if unknown:
                    print(f'WARNING: --exclude_classes named class(es) not found in data: '
                          + ', '.join(sorted(unknown)))
                paired = [(f, l) for f, l in zip(self.files, self.label_names)
                          if l not in excluded_set]
                if not paired:
                    raise RuntimeError('All classes were excluded — check --exclude_classes.')
                print(f'Excluding {len(excluded_set - unknown)} named class(es): '
                      + ', '.join(sorted(excluded_set - unknown)))
                self.files, self.label_names = zip(*paired)
                self.files, self.label_names = list(self.files), list(self.label_names)

            # Train mode: optionally exclude classes below the minimum size threshold
            if min_class_size > 0:
                label_counts = {name: self.label_names.count(name)
                                for name in set(self.label_names)}
                excluded = {n for n, c in label_counts.items() if c < min_class_size}
                if excluded:
                    print(f'Excluding {len(excluded)} class(es) with < {min_class_size} samples: '
                          + ', '.join(f'{n} ({label_counts[n]})' for n in sorted(excluded)))
                    paired = [(f, l) for f, l in zip(self.files, self.label_names)
                              if l not in excluded]
                    if not paired:
                        raise RuntimeError('All classes were excluded — lower min_class_size.')
                    self.files, self.label_names = zip(*paired)
                    self.files, self.label_names = list(self.files), list(self.label_names)

            self.label_names_unique = set(self.label_names)
            self.names_to_labels = {name: i for i, name in enumerate(sorted(self.label_names_unique))}
            self.labels_to_names = {i: name for name, i in self.names_to_labels.items()}
        else:
            # Eval mode: reuse the train mapping; drop samples for unseen classes
            unknown = set(self.label_names) - set(names_to_labels.keys())
            if unknown:
                print(f'Dropping {len(unknown)} eval class(es) not in training set '
                      f'(excluded during training): {sorted(unknown)}')
                paired = [(f, l) for f, l in zip(self.files, self.label_names)
                          if l in names_to_labels]
                self.files, self.label_names = zip(*paired)
                self.files, self.label_names = list(self.files), list(self.label_names)
            self.label_names_unique = set(self.label_names)
            self.names_to_labels = names_to_labels
            self.labels_to_names = {v: k for k, v in names_to_labels.items()}

        self.labels = [self.names_to_labels[n] for n in self.label_names]

        self.emit_hierarchy = emit_hierarchy
        if emit_hierarchy:
            hierarchy_cols = {'class': 'SCOPeClass', 'fold': 'SCOPeFold',
                               'superfamily': 'SCOPeSuperfamily', 'family': 'SCOPeFamily'}
            # One row lookup per file, all four columns pulled from it -- SCOPE_LEVELS
            # order (coarse->fine) is what HierarchicalTripletLoss's nesting assumption
            # relies on: same family implies same superfamily/fold/class since all
            # four values come from the same annotation row.
            per_level_names = {lvl: [] for lvl in SCOPE_LEVELS}
            for file in self.files:
                bname = os.path.basename(file).replace('.jpg', '')
                row = self.annot_frame[self.annot_frame['SCOPeID'] == bname].iloc[0]
                for lvl in SCOPE_LEVELS:
                    per_level_names[lvl].append(row[hierarchy_cols[lvl]])

            if min_family_size > 0:
                family_names = per_level_names['family']
                family_counts = {name: family_names.count(name) for name in set(family_names)}
                keep_mask = [family_counts[name] >= min_family_size for name in family_names]
                n_excluded = len(keep_mask) - sum(keep_mask)
                if n_excluded:
                    n_families_excluded = sum(1 for c in family_counts.values() if c < min_family_size)
                    print(f'Excluding {n_excluded} image(s) from {n_families_excluded} '
                          f'family(ies) with < {min_family_size} members (--min_family_size).')
                self.files = [f for f, keep in zip(self.files, keep_mask) if keep]
                self.label_names = [l for l, keep in zip(self.label_names, keep_mask) if keep]
                self.labels = [l for l, keep in zip(self.labels, keep_mask) if keep]
                for lvl in SCOPE_LEVELS:
                    per_level_names[lvl] = [n for n, keep in zip(per_level_names[lvl], keep_mask) if keep]
                if not self.files:
                    raise RuntimeError('All samples were excluded by --min_family_size -- lower the threshold.')

            self.hierarchy_names_to_labels = {
                lvl: {name: i for i, name in enumerate(sorted(set(per_level_names[lvl])))}
                for lvl in SCOPE_LEVELS
            }
            self.hierarchy_labels = [
                tuple(self.hierarchy_names_to_labels[lvl][per_level_names[lvl][i]]
                      for lvl in SCOPE_LEVELS)
                for i in range(len(self.files))
            ]

    def get_pad(self, curr_size: int, target_size: int):
        d = target_size - curr_size
        if d <= 0: return (0, 0) # no need to pad
        p1 = d // 2
        p2 = d - p1
        return (p1, p2)

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        img_name = self.files[idx]
        if self.emit_hierarchy:
            label = torch.tensor(self.hierarchy_labels[idx], dtype=torch.int64)
        else:
            label = torch.tensor(self.labels[idx])
        if self.pad:
            image = plt.imread(img_name)
            H, W = image.shape[0], image.shape[1]
            DH, DW = self.new_size, self.new_size # desired height / width
            padding = (self.get_pad(H, DH), self.get_pad(W, DW), (0, 0))
            image = np.pad(image, padding, constant_values=128) # pad with gray
        else:
            image = Image.open(img_name).convert('RGB')
            image = image.resize((self.new_size, self.new_size))
            image = np.array(image)
        if self.transform:
            image = self.transform(image)
        return image, label


class ConvNet(nn.Module):
    """From-scratch CNN: 4 conv blocks (3→64→128→256→256) + GAP + FC.

    Uses BatchNorm after every conv layer for stable small-dataset training.
    Global Average Pooling makes the architecture input-size agnostic.
    Dropout (p=0.5) is applied in the FC layers only.
    """
    def __init__(self, num_classes):
        super().__init__()

        def _block(in_ch, out_ch):
            return nn.Sequential(
                nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2, 2),
            )

        self.block1 = _block(3,   64)
        self.block2 = _block(64,  128)
        self.block3 = _block(128, 256)
        self.block4 = _block(256, 256)
        self.gap = nn.AdaptiveAvgPool2d(1)  # (batch, 256, 1, 1)
        self.fc1 = nn.Linear(256, 128)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.gap(x).view(x.size(0), -1)    # flatten: (batch, 256)
        x = F.dropout(F.relu(self.fc1(x)), p=0.5, training=self.training)
        x = self.fc2(x)
        return x


def build_vit(num_classes, num_unfrozen_blocks=2):
    """Pretrained ViT-B/16 with the classification head replaced.

    ViT-B/16 uses fixed, learned positional embeddings sized for a 224x224
    input split into 16x16 patches, so it cannot accept the variable-size
    padded inputs the CNN/ResNet18 paths use -- the caller must resize
    proteograms to exactly 224x224 (see the --model == 'vit' handling in
    __main__, which forces --resize and overrides --max_image_size to 224).

    Only the last `num_unfrozen_blocks` of the 12 transformer encoder blocks
    are trainable (along with the final encoder LayerNorm and the new head);
    the patch embedding, class token, positional embedding, and earlier
    blocks are frozen. ViT-B/16 has ~86M parameters and, unlike a CNN, no
    convolutional inductive bias (locality/translation invariance) to help
    it generalise from a small dataset -- fully unfreezing the backbone
    overfits quickly on datasets of only a few thousand proteograms. Keeping
    most of the backbone frozen preserves the general ImageNet-pretrained
    features while limiting how much the model can memorise the (small)
    training set.
    """
    model = tv_models.vit_b_16(weights=tv_models.ViT_B_16_Weights.IMAGENET1K_V1)
    total_blocks = len(model.encoder.layers)
    num_unfrozen_blocks = max(0, min(num_unfrozen_blocks, total_blocks))
    trainable_block_prefixes = tuple(
        f'encoder.layers.encoder_layer_{i}'
        for i in range(total_blocks - num_unfrozen_blocks, total_blocks)
    )
    for name, param in model.named_parameters():
        if name.startswith('encoder.ln') or name.startswith(trainable_block_prefixes):
            continue
        param.requires_grad = False
    in_features = model.heads.head.in_features
    model.heads = nn.Sequential(
        nn.Dropout(0.5),
        nn.Linear(in_features, num_classes),
    )
    return model


def build_resnet18(num_classes, freeze_layers=('layer1',)):
    """Pretrained ResNet18 with the classification head replaced.

    Only the very first residual block (layer1) is frozen — proteograms encode
    distance-matrix geometry that looks nothing like ImageNet, so the backbone
    needs freedom to adapt.  Regularisation comes from AdamW weight decay
    rather than aggressive layer freezing.  A Dropout is inserted before the
    final linear layer for additional regularisation.
    """
    model = tv_models.resnet18(weights=tv_models.ResNet18_Weights.IMAGENET1K_V1)
    for name, param in model.named_parameters():
        if any(name.startswith(layer) for layer in freeze_layers):
            param.requires_grad = False
    in_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(0.5),
        nn.Linear(in_features, num_classes),
    )
    return model


def build_resnet18_embedding(embed_dim, freeze_layers=('layer1',)):
    """Pretrained ResNet18 with the classification head replaced by an
    embedding projection, for hierarchy-aware triplet training (see
    proteogram.v2.losses.HierarchicalTripletLoss). Same backbone/freezing
    policy as build_resnet18 -- only layer1 frozen, the rest fine-tuned.
    Cosine similarity is scale-invariant, so no explicit L2-normalize layer
    is baked into the model; normalization happens inside the loss and
    inside compute_val_patk instead.
    """
    model = tv_models.resnet18(weights=tv_models.ResNet18_Weights.IMAGENET1K_V1)
    for name, param in model.named_parameters():
        if any(name.startswith(layer) for layer in freeze_layers):
            param.requires_grad = False
    in_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(0.5),
        nn.Linear(in_features, embed_dim),
    )
    return model


def compute_val_patk(model, val_loader, device, k=5):
    """Retrieval Precision@K on a validation split, per SCOPe level, using the
    split itself as the search corpus (each query's own embedding excluded
    from its own top-K).

    Used both as the --loss triplet_hierarchy early-stopping metric (family
    level) and for per-level logging every epoch.

    Returns dict {level: mean P@K} keyed by SCOPE_LEVELS.
    """
    model.eval()
    embeddings, hierarchy = [], []
    with torch.no_grad():
        for data, hlabels in val_loader:
            data = data.to(device)
            emb = F.normalize(model(data), dim=1)
            embeddings.append(emb.cpu())
            hierarchy.append(hlabels)
    embeddings = torch.cat(embeddings, dim=0)
    hierarchy = torch.cat(hierarchy, dim=0)

    k = min(k, embeddings.size(0) - 1)
    sim = embeddings @ embeddings.T
    sim.fill_diagonal_(float('-inf'))  # exclude self-hit
    topk_idx = sim.topk(k, dim=1).indices  # (N, k)

    results = {}
    for level_idx, level in enumerate(SCOPE_LEVELS):
        query_vals = hierarchy[:, level_idx].unsqueeze(1)       # (N, 1)
        neighbor_vals = hierarchy[topk_idx, level_idx]          # (N, k)
        matches = (neighbor_vals == query_vals).float().mean(dim=1)
        results[level] = matches.mean().item()
    return results


def plot_triplet_curves(training_loss, val_patk_history, fig_path, level='family'):
    """Plot train triplet loss and validation Precision@K on twin y-axes
    (different scales/units, so a shared axis would be misleading)."""
    epochs = range(1, len(training_loss) + 1)
    val_metric = [d[level] for d in val_patk_history]
    fig, ax1 = plt.subplots()
    ax1.plot(epochs, training_loss, color='tab:blue', label='Train triplet loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Train loss', color='tab:blue')
    ax2 = ax1.twinx()
    ax2.plot(epochs, val_metric, color='tab:orange', label=f'Val P@K ({level})')
    ax2.set_ylabel(f'Val Precision@K ({level})', color='tab:orange')
    fig.suptitle('Hierarchical Triplet Training')
    fig.tight_layout()
    plt.savefig(fig_path, dpi=150)
    plt.close()
    print(f'Loss/P@K curve saved to {fig_path}')


def train_model_triplet(model, train_loader, val_loader, optimizer, criterion, epochs,
                        patience=None, device=torch.device('cpu'), eval_k=5,
                        early_stop_level='family'):
    """Train an embedding model with HierarchicalTripletLoss.

    Early stopping and best-weight selection use validation Precision@K at
    `early_stop_level` (default 'family', the level cross-entropy struggles
    with most -- see assets/tmp/future_opportunities.md) rather than a loss
    value: triplet loss magnitude isn't comparable across epochs/margin
    settings the way classification loss is, but P@K is the metric we
    actually care about downstream.

    `early_stop_level` may be one of SCOPE_LEVELS to optimize a single level,
    or 'mean' to optimize the average P@K across all four -- useful when
    family alone plateaus/declines early while fold and superfamily are still
    improving, which would otherwise restore a checkpoint that isn't actually
    best for the levels you care about.
    """
    model.to(device)
    criterion.to(device)
    training_loss = []
    val_patk_history = []

    best_val_patk = -1.0
    best_weights = None
    best_epoch = -1
    epochs_no_improve = 0

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5, mode='max')

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        running_pos_dist = 0.0
        running_neg_dist = 0.0
        running_valid_frac = 0.0
        n_batches = 0
        for data, hierarchy_labels in train_loader:
            data = data.to(device)
            hierarchy_labels = hierarchy_labels.to(device)
            optimizer.zero_grad()
            embeddings = model(data)
            loss, stats = criterion(embeddings, hierarchy_labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            running_pos_dist += stats.get('mean_hardest_pos_dist', float('nan'))
            running_neg_dist += stats.get('mean_hardest_neg_dist', float('nan'))
            running_valid_frac += stats.get('valid_anchor_frac', float('nan'))
            n_batches += 1
        train_loss = running_loss / max(n_batches, 1)
        training_loss.append(train_loss)
        # Diagnostic for embedding collapse: if pos/neg dist both -> 0, the
        # model is mapping everything to (nearly) the same point rather than
        # learning real structure -- the batch-hard loss can plateau at a
        # constant value (~base_margin + the largest active margin) even
        # though the embedding itself has stopped being useful.
        mean_pos_dist = running_pos_dist / max(n_batches, 1)
        mean_neg_dist = running_neg_dist / max(n_batches, 1)
        mean_valid_frac = running_valid_frac / max(n_batches, 1)

        val_patk = compute_val_patk(model, val_loader, device, k=eval_k)
        val_patk_history.append(val_patk)
        primary = (np.mean(list(val_patk.values())) if early_stop_level == 'mean'
                  else val_patk[early_stop_level])

        lr_before = optimizer.param_groups[0]['lr']
        scheduler.step(primary)
        lr_after = optimizer.param_groups[0]['lr']

        improved = primary > best_val_patk
        if improved:
            best_val_patk = primary
            if patience is not None:
                best_weights = copy.deepcopy(model.state_dict())
                best_epoch = epoch
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        suffix = ''
        if patience is not None:
            suffix = ' | *' if improved else f' | (no improvement {epochs_no_improve}/{patience})'
        if lr_after < lr_before:
            suffix += f' | LR reduced: {lr_before:.2e} → {lr_after:.2e}'
        patk_str = ' '.join(f'{lvl}={val_patk[lvl]:.3f}' for lvl in SCOPE_LEVELS)
        stats_str = (f'| pos_dist={mean_pos_dist:.3f} neg_dist={mean_neg_dist:.3f} '
                    f'valid_anchor_frac={mean_valid_frac:.2f}')
        # If train_loader's batch_sampler is a HierarchicalPKSampler with
        # stratification on, report how effective it's actually being:
        # - paired_frac: fraction of drawn families that share their stratum
        #   with another drawn family in the same batch, i.e. actually have a
        #   same-stratum-different-family negative to mine. This is the one
        #   that matters -- low means the chosen --triplet_stratify_level
        #   branches too shallowly (e.g. most superfamilies contain only one
        #   family) to produce useful pairs most of the time, even though...
        # - topup_frac stays low/zero -- it only measures running out of
        #   distinct strata entirely, a rarer, coarser failure mode.
        sampler = getattr(train_loader, 'batch_sampler', None)
        if sampler is not None and not np.isnan(getattr(sampler, 'paired_frac', float('nan'))):
            stats_str += f' paired_frac={sampler.paired_frac:.2f} topup_frac={sampler.topup_frac:.2f}'
        print(f'Epoch {epoch:>4d}: train loss: {train_loss:.6f}  val P@{eval_k}: {patk_str}  '
              + stats_str + suffix)

        if patience is not None and epochs_no_improve >= patience:
            print(f'Early stopping at epoch {epoch} — no val P@{eval_k} ({early_stop_level}) '
                  f'improvement for {patience} epochs.')
            break

    if patience is not None and best_weights is not None:
        print(f'Restoring best weights (val P@{eval_k} {early_stop_level}: {best_val_patk:.4f})')
        model.load_state_dict(best_weights)
    return model, training_loss, val_patk_history, best_epoch + 1


def train_model(model, train_loader, val_loader, optimizer, epochs,
                patience=None, device=torch.device('cpu')):
    """Train the ConvNet, tracking train and val loss each epoch.

    If `patience` is set, applies early stopping: training halts when val loss
    has not improved for that many consecutive epochs, and the best weights are
    restored. If `patience` is None, all epochs run and no weight restoration
    is performed.
    """
    model.to(device)
    training_loss = []
    val_loss_history = []

    best_val_loss = float('inf')
    best_weights = None
    best_epoch = -1
    epochs_no_improve = 0

    # LR scheduler uses its own patience (independent of early stopping patience)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)

    loaders = {'train': train_loader, 'val': val_loader}

    for epoch in range(epochs):
        epoch_losses = {}
        for phase in ['train', 'val']:
            model.train() if phase == 'train' else model.eval()
            running_loss = 0.0
            n_batches = 0
            with torch.set_grad_enabled(phase == 'train'):
                for data, target in loaders[phase]:
                    data, target = data.to(device), target.to(device)
                    optimizer.zero_grad()
                    output = model(data)
                    loss = loss_criteria(output, target)
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
                    running_loss += loss.item()
                    n_batches += 1
            epoch_losses[phase] = running_loss / n_batches
        lr_before = optimizer.param_groups[0]['lr']
        scheduler.step(epoch_losses['val'])
        lr_after = optimizer.param_groups[0]['lr']


        training_loss.append(epoch_losses['train'])
        val_loss_history.append(epoch_losses['val'])

        improved = epoch_losses['val'] < best_val_loss
        if improved:
            best_val_loss = epoch_losses['val']
            if patience is not None:
                best_weights = copy.deepcopy(model.state_dict())
                best_epoch = epoch
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        suffix = ''
        if patience is not None:
            suffix = ' | *' if improved else f' | (no improvement {epochs_no_improve}/{patience})'
        if lr_after < lr_before:
            suffix += f' | LR reduced: {lr_before:.2e} → {lr_after:.2e}'
        print(f'Epoch {epoch:>4d}: train loss: {epoch_losses["train"]:.6f}  '
              f'val loss: {epoch_losses["val"]:.6f}' + suffix)

        if patience is not None and epochs_no_improve >= patience:
            print(f'Early stopping at epoch {epoch} — no val loss improvement for {patience} epochs.')
            break

    if patience is not None and best_weights is not None:
        print(f'Restoring best weights (val loss: {best_val_loss:.6f})')
        model.load_state_dict(best_weights)
    return model, training_loss, val_loss_history, best_epoch + 1

def split_train_test(full_dataset, generator):
    """Split a PyTorch Dataset object into train and test sets"""
    total_size = len(full_dataset)
    train_size = int(total_size * 0.7)
    test_size = total_size - train_size
    train_dataset, test_dataset = random_split(
        full_dataset,
        [train_size, test_size],
        generator=generator
    )
    return train_dataset, test_dataset

def get_accuracies(model, test_loader, class_names, labels_to_names, device=torch.device('cpu')):
    """Accuracies per class, classification report, and AUC-ROC."""
    correct_pred = {classname: 0 for classname in class_names}
    total_pred = {classname: 0 for classname in class_names}
    total_correct = 0
    len_data = len(test_loader)
    y_pred = []
    y_test = []
    y_scores = []
    model.eval()
    model.to(device)
    with torch.no_grad():
        for (data, targets) in test_loader:
            data, targets = data.to(device), targets.to(device)
            outputs = model(data)
            probs = torch.softmax(outputs, dim=1)
            _, predictions = torch.max(outputs, 1)
            y_scores.append(probs.cpu().numpy())
            # collect the correct predictions for each class
            for label, prediction in zip(targets, predictions):
                if label == prediction:
                    total_correct += 1
                    correct_pred[labels_to_names[int(label)]] += 1
                total_pred[labels_to_names[int(label)]] += 1
                y_pred.append(labels_to_names[int(prediction)])
                y_test.append(labels_to_names[int(label)])

    # Print accuracy for each class
    for classname, correct_count in correct_pred.items():
        try:
            accuracy = 100 * float(correct_count) / total_pred[classname]
            print(f'Accuracy for class: {classname:5s} is {accuracy:.1f} %')
        except ZeroDivisionError:
            print(f'No samples in test set for class: {classname}')
    overall_accuracy_str = f"{100 * float(total_correct) / len_data:.1f}"
    print(f'Overall accuracy: {overall_accuracy_str} %')

    print('\nAdditional Classification Report:')
    print(classification_report(y_test, y_pred))

    name_to_int = {v: k for k, v in labels_to_names.items()}
    y_test_int = [name_to_int[n] for n in y_test]
    y_scores_arr = np.vstack(y_scores)
    if y_scores_arr.shape[1] == 2:
        # roc_auc_score's multi_class param is only for genuinely multiclass
        # targets -- with exactly 2 classes it dispatches to the binary case
        # regardless of multi_class, which requires a 1D score array (P of
        # the positive/greater-labeled class) and raises "y should be a 1d
        # array" if given the full (N, 2) softmax output instead. There's
        # only one ROC curve for binary, so macro == weighted here.
        auc_macro = auc_weighted = roc_auc_score(y_test_int, y_scores_arr[:, 1])
    else:
        auc_macro    = roc_auc_score(y_test_int, y_scores_arr, multi_class='ovr', average='macro')
        auc_weighted = roc_auc_score(y_test_int, y_scores_arr, multi_class='ovr', average='weighted')
    print(f'\nAUC-ROC (macro):    {auc_macro:.4f}')
    print(f'AUC-ROC (weighted): {auc_weighted:.4f}')

    return overall_accuracy_str


def view_pred_set(model, test_loader, num_preds, labels_to_names, fig_path):
    """Graph a set of predictions with labels and save plot."""
    predictions = []
    images = []
    labels = []
    with torch.no_grad():
        cnt = 1
        for (data, target) in test_loader:
            images.append(data)
            labels.append(target[0])
            outputs = model(data)
            _, predicted = torch.max(outputs, 1)
            predictions.append(predicted[0])

            if cnt == num_preds:
                break
            cnt += 1

    for i in range(num_preds):
        plot_row = max(1, int(num_preds/2))
        plt.subplot(2, plot_row, i + 1)
        img = images[i]
        npimg = img.numpy()
        npimg = np.squeeze(npimg, axis=0)
        npimg = npimg / 2 + 0.5
        plt.imshow(np.transpose(npimg, (1, 2, 0)))
        plt.axis('off')
        
        color = "green"
        pred = int(predictions[i].numpy())
        label = int(labels[i].numpy())
        name = labels_to_names[label]
        if label != pred:
            color = "red"
        plt.title(name, color=color)

    plt.suptitle('Objects Found by Model', size=20)
    plt.savefig(fig_path)

def load_model(model_path, classes, image_size):
    """Load the ConvNet model from disk."""
    model = ConvNet(classes, image_size)
    ConvNet.load_state_dict(torch.load(model_path))
    return model

def plot_losses(training_loss, val_loss, fig_path):
    """Plot training and validation loss curves on the same axes and save to file."""
    epochs = range(1, len(training_loss) + 1)
    plt.figure()
    plt.plot(epochs, training_loss, label='Train loss')
    plt.plot(epochs, val_loss, label='Val loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.tight_layout()
    plt.savefig(fig_path, dpi=150)
    plt.close()
    print(f'Loss curve saved to {fig_path}')

def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


class FocalLoss(nn.Module):
    """Multi-class focal loss: FL = -(1 - p_t)^gamma * log(p_t).

    gamma=0 reduces to standard cross-entropy. gamma=2 is the value used in
    the original RetinaNet paper and is a reasonable starting point.
    label_smoothing is applied to the underlying cross-entropy term.
    """
    def __init__(self, gamma=2.0, label_smoothing=0.0):
        super().__init__()
        self.gamma = gamma
        self.label_smoothing = label_smoothing

    def forward(self, inputs, targets):
        ce = F.cross_entropy(inputs, targets, label_smoothing=self.label_smoothing, reduction='none')
        pt = torch.exp(-ce)
        return ((1 - pt) ** self.gamma * ce).mean()


class TransformedSubset(Dataset):
    """Wraps a Subset and applies a transform, allowing train/eval subsets to use different augmentations."""
    def __init__(self, subset, transform):
        self.subset = subset
        self.transform = transform

    def __len__(self):
        return len(self.subset)

    def __getitem__(self, idx):
        image, label = self.subset[idx]
        if self.transform:
            image = self.transform(image)
        return image, label


if __name__ == '__main__':

    parser = argparse.ArgumentParser(
        description="Train CNN on Proteograms.")
    parser.add_argument('--data_dir', '-d',
                        type=str,
                        default=None,
                        help="Root directory containing all proteogram images (flat layout, "
                             "no train/eval subdirectories). Overrides training_data_dir in config.yml.")
    parser.add_argument("--epochs", "-e",
                        type=int,
                        help="Number of training epochs.")
    parser.add_argument("--batch_size", "-b",
                        type=int,
                        default=16,
                        help="Training batch size.")
    parser.add_argument("--lr", "-l",
                        type=float,
                        help="Training learning rate.")
    parser.add_argument('--model', '-m',
                        choices=['cnn', 'resnet18', 'vit'],
                        default='cnn',
                        help="Model architecture: 'cnn' (from-scratch 4-block ConvNet), "
                             "'resnet18' (pretrained ResNet18 fine-tuning), or 'vit' "
                             "(pretrained ViT-B/16 fine-tuning). Note: 'vit' forces --resize "
                             "and a fixed 224x224 input size (overriding --max_image_size), "
                             "since ViT-B/16's positional embeddings require a fixed input "
                             "shape and cannot use the padded/variable-size inputs the other "
                             "two architectures accept. Default: cnn.")
    parser.add_argument('--vit_unfrozen_blocks',
                        type=int,
                        default=2,
                        help="Number of trailing ViT-B/16 transformer encoder blocks (out of "
                             "12) to leave trainable during fine-tuning; the patch embedding, "
                             "class token, positional embedding, and earlier blocks are frozen. "
                             "Only used when --model vit. Lower values reduce overfitting risk "
                             "on small datasets at the cost of less backbone adaptation. "
                             "Default: 2.")
    parser.add_argument('--overwrite', '-o',
                        action='store_true',
                        help="Recreate / overwrite model")    
    parser.add_argument('--resize',
                        action='store_true',
                        help="Resize images to new_size instead of padding. "
                             "Default is to pad with gray, which preserves the "
                             "1-pixel-per-residue-pair semantic of proteograms.")
    parser.add_argument('--verbose', '-v',
                        action='store_true',
                        help="Verbose output and logging.")
    parser.add_argument('--tsv_file', '-t',
                        type=str,
                        default=None,
                        help="Path to the TSV annotations file. Defaults to "
                             "ProteogramData_SCOP_RCSB_PDBe_AnnotationsLookup.tsv "
                             "in the proteograms directory.")
    parser.add_argument('--patience',
                        type=int,
                        default=None,
                        help="Early stopping patience: stop after this many epochs "
                             "with no improvement in val loss. Omit to disable early stopping.")
    parser.add_argument('--seed',
                        type=int,
                        default=0,
                        help="Random seed used for the train/val/test split and all stochastic "
                             "operations. Change to produce a different reproducible split (default: 0).")
    parser.add_argument('--val_size',
                        type=float,
                        default=0.15,
                        help="Fraction of images to hold out as validation "
                             "set for early stopping (default: 0.15).")
    parser.add_argument('--test_size',
                        type=float,
                        default=0.15,
                        help="Fraction of images to hold out as the final held-out test set (default: 0.15).")
    parser.add_argument('--save_test_list',
                        action='store_true',
                        help="Flag to save the list of test set image filenames to a text file for later " "reference.")
    parser.add_argument('--max_image_size',
                        type=int,
                        default=200,
                        help="Pad proteogram images to this square size (pixels) and exclude any "
                             "images larger than this value. Equivalent to filtering by max sequence "
                             "length. Default: 200.")
    parser.add_argument('--exclude_classes', '-x',
                        type=str,
                        default=None,
                        help="Comma-separated list of SCOPe class names to exclude "
                             "from training and evaluation (e.g. 'j,h'). "
                             "Useful for removing very small or low-quality classes.")
    parser.add_argument('--level',
                        choices=['class', 'fold', 'superfamily', 'family'],
                        default='class',
                        help="SCOPe hierarchy level to use as the classification target. "
                             "'class' is the highest (broadest) level; 'family' is the lowest "
                             "(finest). Default: class.")
    parser.add_argument('--min_class_size',
                        type=int,
                        default=20,
                        help="Exclude classes with fewer than this many samples to avoid extreme "
                             "class imbalance. Default: 20.")
    parser.add_argument('--loss',
                        choices=['ce', 'focal', 'triplet_hierarchy'],
                        default='ce',
                        help="Loss function: 'ce' (cross-entropy with label smoothing), "
                             "'focal' (focal loss, good for hard/imbalanced examples), or "
                             "'triplet_hierarchy' (hierarchy-aware batch-hard triplet loss "
                             "with margins scaled by SCOPe hierarchy distance -- trains an "
                             "embedding model directly for retrieval instead of a classifier "
                             "for one --level; only supported with --model resnet18). Default: ce.")
    parser.add_argument('--focal_gamma',
                        type=float,
                        default=2.0,
                        help="Focusing parameter gamma for focal loss (ignored when --loss=ce). "
                             "gamma=0 reduces to cross-entropy; gamma=2 is the RetinaNet default. "
                             "Default: 2.0.")
    parser.add_argument('--embed_dim',
                        type=int,
                        default=128,
                        help="Embedding dimension for --loss triplet_hierarchy. Ignored otherwise. "
                             "Default: 128.")
    parser.add_argument('--triplet_p',
                        type=int,
                        default=16,
                        help="Number of distinct SCOPe families sampled per batch for --loss "
                             "triplet_hierarchy (P in P-K sampling). Ignored otherwise. Default: 16.")
    parser.add_argument('--triplet_k',
                        type=int,
                        default=4,
                        help="Samples per family per batch for --loss triplet_hierarchy (K in "
                             "P-K sampling; must be >= 2 so every anchor has an in-batch "
                             "same-family positive). Effective batch size is triplet_p * "
                             "triplet_k, overriding --batch_size. Ignored otherwise. Default: 4.")
    parser.add_argument('--triplet_base_margin',
                        type=float,
                        default=0.2,
                        help="Base triplet margin for --loss triplet_hierarchy. Ignored otherwise. "
                             "Default: 0.2.")
    parser.add_argument('--triplet_margins',
                        type=str,
                        default='0.0,0.2,0.4,0.6,0.8',
                        help="Comma-separated margins for SCOPe hierarchy distance 0..4 (same "
                             "family, same superfamily, same fold, same class, different class) "
                             "for --loss triplet_hierarchy. Ignored otherwise. "
                             "Default: 0.0,0.2,0.4,0.6,0.8.")
    parser.add_argument('--triplet_eval_k',
                        type=int,
                        default=5,
                        help="K for the validation Precision@K used as the early-stopping metric "
                             "with --loss triplet_hierarchy. Ignored otherwise. Default: 5.")
    parser.add_argument('--min_family_size',
                        type=int,
                        default=None,
                        help="For --loss triplet_hierarchy: exclude SCOPe families with fewer "
                             "than this many samples, so HierarchicalPKSampler's --triplet_k "
                             "positives are always distinct images rather than with-replacement "
                             "duplicates of the same image (a degenerate, near-zero-gradient "
                             "'positive'). Defaults to --triplet_k, the minimum needed to avoid "
                             "duplicates; pass a larger value for more headroom. Ignored otherwise.")
    parser.add_argument('--triplet_families_per_stratum',
                        type=int,
                        default=4,
                        help="For --loss triplet_hierarchy: target number of distinct families "
                             "to draw from each selected --triplet_stratify_level group (fold or "
                             "superfamily) when composing a batch, instead of drawing all "
                             "--triplet_p families uniformly at random. This guarantees batches "
                             "contain same-group-different-family negatives for "
                             "HierarchicalTripletLoss's mid-tier margins to mine, which random "
                             "family sampling rarely produces once there are thousands of families. "
                             "Ignored otherwise. Default: 4.")
    parser.add_argument('--triplet_stratify',
                        action='store_true',
                        help="For --loss triplet_hierarchy: compose batches by drawing "
                             "--triplet_families_per_stratum families from each of a handful of "
                             "--triplet_stratify_level groups first (see HierarchicalPKSampler), "
                             "instead of drawing all --triplet_p families uniformly at random. "
                             "Guarantees batches contain same-group-different-family negatives for "
                             "HierarchicalTripletLoss's mid-tier margins to mine. Off by default "
                             "(uniform random family sampling). Ignored otherwise.")
    parser.add_argument('--triplet_stratify_level',
                        choices=['fold', 'superfamily'],
                        default='fold',
                        help="For --loss triplet_hierarchy with --triplet_stratify: which SCOPe "
                             "level to stratify batch composition by. 'fold' guarantees "
                             "same-fold-different-family negatives (may or may not also share "
                             "superfamily). 'superfamily' guarantees same-superfamily-different-"
                             "family negatives directly -- a narrower, more targeted signal for "
                             "the superfamily/family margins specifically, at the cost of less "
                             "fold-level negative coverage. Ignored otherwise. Default: fold.")
    parser.add_argument('--triplet_early_stop_level',
                        choices=list(SCOPE_LEVELS) + ['mean'],
                        default='family',
                        help="For --loss triplet_hierarchy: which validation Precision@K to use "
                             "for early stopping and best-checkpoint selection. 'mean' averages "
                             "P@K across all four SCOPe levels -- useful when family alone "
                             "plateaus/declines early while fold/superfamily are still improving, "
                             "which would otherwise select a checkpoint that isn't actually best "
                             "for the levels you care about. Ignored otherwise. Default: family.")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

    if args.loss == 'triplet_hierarchy' and args.model != 'resnet18':
        raise ValueError("--loss triplet_hierarchy currently only supports --model resnet18.")
    if args.loss == 'triplet_hierarchy' and args.min_family_size is None:
        # Default to triplet_k: below this, HierarchicalPKSampler must sample a
        # family with replacement to fill a batch, producing duplicate-image
        # "positives" instead of genuinely different structures.
        args.min_family_size = args.triplet_k
        print(f'--min_family_size not set, defaulting to --triplet_k ({args.triplet_k})')

    config = read_yaml('config.yml')
    root_dir = args.data_dir or config['training_data_dir']
    # Get level from command line or config, with command line taking precedence. Default to 'class' if neither is provided.
    level = args.level or config.get('scope_level', 'class')

    if args.epochs:
        epochs = args.epochs
    elif 'num_epochs' in config:
        epochs = config['num_epochs']
    else:
        raise ValueError("Number of epochs must be specified via command line or config.yml")
    if args.lr:
        lr = args.lr
    elif 'learning_rate' in config:
        lr = config['learning_rate']
    else:
        raise ValueError("Learning rate must be specified via command line or config.yml")
    if args.batch_size:
        batch_size = args.batch_size
    elif 'batch_size' in config:
        batch_size = config['batch_size']
    else:
        raise ValueError("Batch size must be specified via command line or config.yml")
    # Write the resolved value back so every later use of args.batch_size (the
    # val/test DataLoaders) sees it too -- otherwise, when --batch_size is
    # omitted and only config.yml's batch_size is set, args.batch_size stays
    # None and DataLoader(..., batch_size=None) silently disables automatic
    # batching, yielding unbatched (C,H,W) samples instead of (N,C,H,W).
    args.batch_size = batch_size

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    image_resize = args.max_image_size

    # ViT-B/16 has fixed, learned positional embeddings sized for a 224x224 input
    # (16x16 patches), so it cannot accept the padded/variable-size images the
    # CNN and ResNet18 paths use. Always resize (never pad) to exactly 224x224
    # for the ViT path, regardless of what was passed via --resize/--max_image_size.
    if args.model == 'vit':
        if not args.resize:
            print('ViT-B/16 requires resizing (not padding) to a fixed input size; '
                  'forcing --resize.')
            args.resize = True
        if image_resize != 224:
            print(f'ViT-B/16 requires 224x224 input; overriding --max_image_size from '
                  f'{image_resize} to 224.')
            image_resize = 224
            args.max_image_size = 224

    # ResNet18 was trained with ImageNet normalisation (standardize input images to the same distribution as the data the model was pre-trained on)
    # so the pretrained feature detectors remain valid. The ConvNet was not pretrained, so it doesn't strictly require ImageNet normalisation, but applying the same normalisation to both models allows for a more controlled comparison.
    _imagenet_norm = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                          std=[0.229, 0.224, 0.225])
    _augment = [
        transforms.RandomApply([transforms.ColorJitter(brightness=0.1, contrast=0.1)], p=0.1),
        transforms.RandomApply([transforms.RandomAdjustSharpness(sharpness_factor=2)], p=0.1),
    ]

    if args.model == 'resnet18':
        transform_train = transforms.Compose([transforms.ToTensor()] + _augment + [_imagenet_norm])
        transform_eval  = transforms.Compose([transforms.ToTensor(), _imagenet_norm])
    else:
        transform_train = transforms.Compose([transforms.ToTensor()] + _augment + [_imagenet_norm])
        transform_eval  = transforms.Compose([transforms.ToTensor(), _imagenet_norm])
    
    tsv_file = args.tsv_file or os.path.join(
        root_dir, '..', 'ProteogramData_SCOP_RCSB_PDBe_AnnotationsLookup_AllSCOPe208.tsv')

    exclude_classes = [c.strip() for c in args.exclude_classes.split(',')] \
        if args.exclude_classes else None

    # Load the full dataset without transforms so each split can use its own augmentation policy.
    # https://docs.pytorch.org/docs/stable/notes/randomness.html
    g = torch.Generator()
    g.manual_seed(args.seed)

    full_dataset = ProteogramDataset(
        tsv_file=tsv_file,
        root_dir=root_dir,
        level=args.level,
        new_size=image_resize,
        pad=not args.resize,
        transform=None,
        # exclude classes with fewer than this many samples to avoid extreme class imbalance
        min_class_size=args.min_class_size,
        min_image_size=20,
        max_image_size=args.max_image_size,
        exclude_classes=exclude_classes,
        emit_hierarchy=(args.loss == 'triplet_hierarchy'),
        min_family_size=(args.min_family_size if args.loss == 'triplet_hierarchy' else 0))

    all_indices = list(range(len(full_dataset)))
    labels = full_dataset.labels

    train_val_indices, test_indices = train_test_split(
        all_indices,
        test_size=args.test_size,
        random_state=args.seed,
        stratify=[labels[i] for i in all_indices])

    # val fraction relative to the train+val pool
    val_fraction = args.val_size / (1.0 - args.test_size)
    train_indices, val_indices = train_test_split(
        train_val_indices,
        test_size=val_fraction,
        random_state=args.seed,
        stratify=[labels[i] for i in train_val_indices])

    train_split = TransformedSubset(Subset(full_dataset, train_indices), transform_train)
    val_split   = TransformedSubset(Subset(full_dataset, val_indices),   transform_eval)
    test_split  = TransformedSubset(Subset(full_dataset, test_indices),  transform_eval)
    print(f'Stratified split (seed={args.seed}): '
          f'{len(train_indices)} train / {len(val_indices)} val / {len(test_indices)} test (held-out)')

    # WeightedRandomSampler: oversample minority classes so each epoch sees
    # a balanced class distribution regardless of raw class frequencies.
    train_labels = [full_dataset.labels[i] for i in train_indices]
    label_counts = torch.bincount(torch.tensor(train_labels))
    class_weights = torch.where(
        label_counts > 0, 1.0 / label_counts.float(), torch.zeros_like(label_counts.float()))
    sample_weights = [class_weights[lbl].item() for lbl in train_labels]
    sampler = torch.utils.data.WeightedRandomSampler(
        weights=sample_weights, num_samples=len(sample_weights), replacement=True)

    class_names = full_dataset.label_names_unique

    if args.loss == 'triplet_hierarchy':
        if args.triplet_k < 2:
            raise ValueError(f'--triplet_k must be >= 2 so every anchor has an in-batch '
                              f'same-family positive (got {args.triplet_k}).')
        family_level_idx = SCOPE_LEVELS.index('family')
        stratify_level_idx = SCOPE_LEVELS.index(args.triplet_stratify_level)
        train_family_labels = [full_dataset.hierarchy_labels[i][family_level_idx]
                                for i in train_indices]
        train_stratum_labels = [full_dataset.hierarchy_labels[i][stratify_level_idx]
                                for i in train_indices]
        pk_sampler = HierarchicalPKSampler(train_family_labels, p=args.triplet_p,
                                           k=args.triplet_k,
                                           stratum_labels=train_stratum_labels if args.triplet_stratify else None,
                                           families_per_stratum=args.triplet_families_per_stratum,
                                           seed=args.seed)
        strat_note = (f'by {args.triplet_stratify_level}, '
                      f'{args.triplet_families_per_stratum} families/{args.triplet_stratify_level} target'
                      if args.triplet_stratify else 'disabled (pass --triplet_stratify to enable)')
        print(f'HierarchicalPKSampler: P={args.triplet_p} families x K={args.triplet_k} '
              f'samples/family = batch size {args.triplet_p * args.triplet_k}, '
              f'{len(pk_sampler)} batches/epoch (overrides --batch_size), stratification: {strat_note}')
        train_loader = DataLoader(train_split,
                                  batch_sampler=pk_sampler,
                                  worker_init_fn=seed_worker)
    else:
        train_loader = DataLoader(train_split,
                                  batch_size=args.batch_size,
                                  # Note, shuffle is mutually exclusive with sampler
                                  shuffle=True,
                                #   sampler=sampler,
                                  worker_init_fn=seed_worker)
    val_loader = DataLoader(val_split,
                            batch_size=args.batch_size,
                            shuffle=False,
                            worker_init_fn=seed_worker,
                            generator=g)
    test_loader = DataLoader(test_split,
                             batch_size=1 if args.loss != 'triplet_hierarchy' else args.batch_size,
                             shuffle=False,
                             worker_init_fn=seed_worker,
                             generator=g)

    num_classes = len(class_names)
    print(f'Number of classes: {num_classes}')

    output_dir = os.path.dirname(os.path.abspath(root_dir))

    if args.loss == 'triplet_hierarchy':
        # Embedding model: no classification head, trained directly for cosine
        # retrieval -- see proteogram/v2/losses.py for why (P@K is the actual
        # target, not per-level classification accuracy).
        model = build_resnet18_embedding(args.embed_dim)
        backbone_params = [p for n, p in model.named_parameters() if 'fc' not in n and p.requires_grad]
        head_params = list(model.fc.parameters())
        optimizer = optim.AdamW([
            {'params': backbone_params, 'lr': args.lr * 0.1},
            {'params': head_params,     'lr': args.lr},
        ], weight_decay=1e-3)
        print(f'ResNet18 (embedding, dim={args.embed_dim}): '
              f'backbone LR={args.lr * 0.1:.2e}, head LR={args.lr:.2e}')

        margins = tuple(float(x) for x in args.triplet_margins.split(','))
        loss_criteria = HierarchicalTripletLoss(margins=margins, base_margin=args.triplet_base_margin)
        print(f'Loss: HierarchicalTripletLoss(margins={margins}, base_margin={args.triplet_base_margin})')

        model, training_loss, val_patk_history, epochs_trained = train_model_triplet(
                            model,
                            train_loader=train_loader,
                            val_loader=val_loader,
                            optimizer=optimizer,
                            criterion=loss_criteria,
                            epochs=args.epochs,
                            patience=args.patience,
                            device=device,
                            eval_k=args.triplet_eval_k,
                            early_stop_level=args.triplet_early_stop_level)

        test_patk = compute_val_patk(model, test_loader, device, k=args.triplet_eval_k)
        print(f'\nHeld-out test set Precision@{args.triplet_eval_k}:')
        for lvl in SCOPE_LEVELS:
            print(f'  {lvl:12s}: {test_patk[lvl]:.4f}')

        suffix = (f'_{args.model}_lr{lr}_bs{args.triplet_p * args.triplet_k}_e{epochs_trained}_'
                  f'seed{args.seed}_max_image_size{args.max_image_size}_'
                  f'min_class_size{args.min_class_size}_loss-triplet_hierarchy_'
                  f'embeddim{args.embed_dim}_famPatK{test_patk["family"]:.3f}')

        # 'mean' isn't a per-epoch dict key (val_patk_history entries are keyed
        # by SCOPE_LEVELS only) -- fall back to plotting family in that case.
        plot_level = args.triplet_early_stop_level if args.triplet_early_stop_level != 'mean' else 'family'
        plot_triplet_curves(training_loss, val_patk_history, level=plot_level,
                            fig_path=os.path.join(output_dir, f'loss_curves{suffix}.png'))

        model_file = config.get('model_file_prefix', 'scope_proteogram_model') + suffix + '.pt'
        model_path = os.path.join(output_dir, model_file)

        if os.path.exists(model_path) and not args.overwrite:
            print(f'Model file {model_path} exists and overwrite not set, not saving model.')
        else:
            # Self-describing format -- state_dict alone is indistinguishable
            # from a classifier checkpoint (both are a bare fc.1.weight/fc.1.bias
            # Sequential(Dropout, Linear) head), so Img2Vec (proteogram/v2/
            # image_similarity.py) needs the meta to know fc is the trained
            # embedding projection, not a head to strip.
            torch.save({
                'state_dict': model.state_dict(),
                # max_image_size is recorded so search-time padding
                # (measure_similarity_v2.py) can match the training pad target
                # automatically -- a mismatch silently crops larger proteins
                # and reframes smaller ones at the wrong scale.
                'meta': {'kind': 'embedding', 'architecture': 'resnet18',
                         'embed_dim': args.embed_dim,
                         'max_image_size': args.max_image_size},
            }, model_path)
            print(f'Saved model to {model_path}')

        if args.save_test_list:
            save_list_name = f"test_list_for_model_{suffix}.lst"
            save_list_path = os.path.join(output_dir, save_list_name)
            os.makedirs(os.path.dirname(os.path.abspath(save_list_path)), exist_ok=True)
            with open(save_list_path, 'w') as f:
                for i in test_indices:
                    f.write(os.path.splitext(os.path.basename(full_dataset.files[i]))[0] + '\n')
            print(f'Test set file prefixes written to {save_list_path}')

        # No get_accuracies/view_pred_set: there's no classifier head or
        # class_names to report against in embedding mode.

    else:
        if args.model == 'resnet18':
            model = build_resnet18(num_classes)
            # Differential LR: lower rate for pretrained backbone, full rate for new head
            backbone_params = [p for n, p in model.named_parameters() if 'fc' not in n and p.requires_grad]
            head_params = list(model.fc.parameters())
            optimizer = optim.AdamW([
                {'params': backbone_params, 'lr': args.lr * 0.1},
                {'params': head_params,     'lr': args.lr},
            ], weight_decay=1e-3)
            print(f'ResNet18: backbone LR={args.lr * 0.1:.2e}, head LR={args.lr:.2e}')
        elif args.model == 'vit':
            model = build_vit(num_classes, num_unfrozen_blocks=args.vit_unfrozen_blocks)
            # Differential LR: lower rate for pretrained backbone, full rate for new head.
            # Only trainable backbone params (the unfrozen tail blocks + final LayerNorm)
            # are included -- frozen params have requires_grad=False and must be excluded
            # so AdamW doesn't try to step them.
            backbone_params = [p for n, p in model.named_parameters() if 'heads' not in n and p.requires_grad]
            head_params = list(model.heads.parameters())
            optimizer = optim.AdamW([
                {'params': backbone_params, 'lr': args.lr * 0.1},
                {'params': head_params,     'lr': args.lr},
            ], weight_decay=1e-3)
            print(f'ViT-B/16: backbone LR={args.lr * 0.1:.2e}, head LR={args.lr:.2e}, '
                  f'last {args.vit_unfrozen_blocks} of 12 encoder blocks trainable')
        else:
            model = ConvNet(num_classes)
            optimizer = optim.Adam(model.parameters(), lr=args.lr)
            print(f'ConvNet (from scratch): LR={args.lr:.2e}')

        if args.loss == 'focal':
            loss_criteria = FocalLoss(gamma=args.focal_gamma, label_smoothing=0.1)
            print(f'Loss: FocalLoss(gamma={args.focal_gamma}, label_smoothing=0.1)')
        else:
            loss_criteria = nn.CrossEntropyLoss(label_smoothing=0.1)
            print('Loss: CrossEntropyLoss(label_smoothing=0.1)')
        model, training_loss, val_loss, epochs_trained = train_model(model,
                            train_loader=train_loader,
                            val_loader=val_loader,
                            optimizer=optimizer,
                            epochs=args.epochs,
                            patience=args.patience,
                            device=device)

        overall_accuracy = get_accuracies(model,
                       test_loader,
                       class_names,
                       full_dataset.labels_to_names)

        loss_tag = f'focal_g{args.focal_gamma}' if args.loss == 'focal' else 'ce'
        model_tag = f'{args.model}_unfrozen{args.vit_unfrozen_blocks}' if args.model == 'vit' else args.model
        suffix = f'_{model_tag}_lr{lr}_bs{batch_size}_e{epochs_trained}_seed{args.seed}_max_image_size{args.max_image_size}_min_class_size{args.min_class_size}_level-{level}_loss{loss_tag}_acc{overall_accuracy}'

        plot_losses(training_loss, val_loss,
                    fig_path=os.path.join(output_dir, f'loss_curves{suffix}.png'))

        model_file = config.get('model_file_prefix', 'scope_proteogram_model') + suffix + '.pt'
        model_path = os.path.join(output_dir, model_file)

        # Save model
        if os.path.exists(model_path) and not args.overwrite:
            print(f'Model file {model_path} exists and overwrite not set, not saving model.')
        else:
            # Save only the model weights
            torch.save(model.state_dict(), model_path)
            print(f'Saved model to {model_path}')

        if args.save_test_list:
            save_list_name = f"test_list_for_model_{suffix}.lst"
            save_list_path = os.path.join(output_dir, save_list_name)
            os.makedirs(os.path.dirname(os.path.abspath(save_list_path)), exist_ok=True)
            with open(save_list_path, 'w') as f:
                for i in test_indices:
                    f.write(os.path.splitext(os.path.basename(full_dataset.files[i]))[0] + '\n')
            print(f'Test set file prefixes written to {save_list_path}')

        view_pred_set(model,
                      test_loader,
                      num_preds=10,
                      labels_to_names=full_dataset.labels_to_names,
                      fig_path=os.path.join(output_dir, f'sample_preds{suffix}.png'))

