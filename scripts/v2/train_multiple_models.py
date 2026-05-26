"""
This script trains a CNN on the Proteogram dataset, with options for architecture (from-scratch ConvNet or pretrained ResNet18), hyperparameters (epochs, batch size, learning rate), and logging.  It includes early stopping based on validation loss, and saves the best model weights to disk.  The training and validation loss curves are plotted and saved to a file.  After training, the model is evaluated on a held-out test set, with per-class accuracies and a classification report printed to the console.

Requires a GPU for reasonable training time, especially for ResNet18.  For reproducibility, random seeds are fixed and a deterministic data split is used.

Data location and other parameters can be configured via command-line arguments or a config.yml file.  Command-line arguments take precedence over config.yml. The Proteogram dataset should be prepared in advance using the create_v2_proteograms.py script. The root directory ("training_data_dir" in config.yml) should contain the "train" and "eval" subdirectories with the respective, representative images. An annotation TSV file is also required, which should be specified via the --tsv_file argument or included in config.yml as "tsv_file".  The model weights will be saved to the path specified by "cnn_model_file_prefix" in config.yml, with a suffix indicating the architecture and hyperparameters.

Here is more information about the SCOPe dataset: https://scop.berkeley.edu
 
 Usage example:
    python train_multiple_models.py --model resnet18 --epochs 50 --batch_size 32 --lr 1e-4
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

import torch
from torch.utils.data import random_split, Dataset, DataLoader
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.models as tv_models
import torchvision.transforms as transforms

from proteogram.common import read_yaml


matplotlib.use('agg')
# # For reproducibility, set seeds
torch.manual_seed(0)
random.seed(0)
np.random.seed(0)

class ProteogramDataset(Dataset):
    """SCOPe-based Proteograms dataset."""

    def __init__(self, tsv_file, root_dir, pad=True, level='class', new_size=128, transform=None,
                 names_to_labels=None, min_class_size=0, min_image_size=0, exclude_classes=None):
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
        exclude_classes : list of str, optional
            Class names to exclude entirely (train mode only; eval samples for
            excluded classes are automatically dropped via the names_to_labels
            mechanism). Default: None (keep all).
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
        label = torch.tensor(self.labels[idx])
        if self.pad:
            image = plt.imread(img_name)
            H, W = image.shape[0], image.shape[1]
            DH, DW = self.new_size, self.new_size # desired height / width
            padding = (self.get_pad(H, DH), self.get_pad(W, DW), (0, 0))
            image = np.pad(image, padding, constant_values=128) # pad with gray
            image = image[0:DH, 0:DW, :] # crop if needed
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


def scop_triplet_loss(embeddings, fold_labels, class_labels, margin=0.3):
    """Cosine-space triplet loss with SCOP-hierarchy pair selection.

    Positive: hardest same-fold protein (lowest cosine sim to anchor).
    Negative: hardest same-class, different-fold protein (highest cosine sim).
    Restricting negatives to the same class forces fold separation rather than
    just broad class separation.
    """
    emb = F.normalize(embeddings, dim=1)
    sim = emb @ emb.T
    B = emb.size(0)
    idx = torch.arange(B, device=emb.device)

    triplet_losses = []
    for i in range(B):
        same_fold = (fold_labels == fold_labels[i]) & (idx != i)
        hard_neg_mask = (class_labels == class_labels[i]) & (fold_labels != fold_labels[i])
        if same_fold.sum() == 0 or hard_neg_mask.sum() == 0:
            continue
        pos_sim = sim[i, same_fold].min()
        neg_sim = sim[i, hard_neg_mask].max()
        triplet_losses.append(F.relu(neg_sim - pos_sim + margin))

    if not triplet_losses:
        return torch.tensor(0.0, device=emb.device)
    return torch.stack(triplet_losses).mean()


class SCOPFoldSampler(torch.utils.data.Sampler):
    """Guarantee at least 2 proteins from the same fold in every batch.

    Without this, random batching on ~544 proteins / 241 folds produces many
    batches with no positive pairs, making triplet loss useless.
    Expects a random_split Subset whose underlying dataset has a `label_names`
    attribute (standard on ProteogramDataset).
    """
    def __init__(self, subset, batch_size):
        self.batch_size = batch_size
        dataset = subset.dataset
        local_fold_names = [dataset.label_names[i] for i in subset.indices]
        self.fold_to_local = {}
        for local_i, fname in enumerate(local_fold_names):
            self.fold_to_local.setdefault(fname, []).append(local_i)
        self.eligible_folds = [f for f, ids in self.fold_to_local.items() if len(ids) >= 2]

    def __len__(self):
        return sum(len(v) for v in self.fold_to_local.values() if len(v) >= 2)

    def __iter__(self):
        folds = self.eligible_folds.copy()
        random.shuffle(folds)
        batch = []
        for fold in folds:
            pair = random.sample(self.fold_to_local[fold], 2)
            batch.extend(pair)
            if len(batch) >= self.batch_size:
                yield batch[:self.batch_size]
                batch = []
        if len(batch) >= 2:
            yield batch


class WithClassLabel(Dataset):
    """Wraps a random_split Subset to return (image, fold_label, class_label).

    Used for triplet training: the DataLoader yields class labels alongside
    fold labels so scop_triplet_loss can select within-class negatives.
    """
    def __init__(self, subset, class_labels_by_orig_idx):
        self.subset = subset
        self.class_labels = class_labels_by_orig_idx

    def __len__(self):
        return len(self.subset)

    def __getitem__(self, idx):
        orig_idx = self.subset.indices[idx]
        image, fold_label = self.subset[idx]
        return image, fold_label, torch.tensor(self.class_labels[orig_idx])


class WithProteinId(Dataset):
    """Wraps a random_split Subset to return (image, label, protein_id).

    Used for ranking-loss training: the DataLoader yields the SCOPe domain ID
    (filename stem) alongside the image and label so that TM-score ground truth
    can be looked up per batch.
    """
    def __init__(self, subset):
        self.subset = subset

    def __len__(self):
        return len(self.subset)

    def __getitem__(self, idx):
        orig_idx = self.subset.indices[idx]
        image, label = self.subset[idx]
        filepath = self.subset.dataset.files[orig_idx]
        protein_id = os.path.splitext(os.path.basename(filepath))[0]
        return image, label, protein_id


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
                for batch in loaders[phase]:
                    data, target = batch[0].to(device), batch[1].to(device)
                    # batch[2] may be a tensor (class labels / triplet) or a
                    # tuple of strings (protein IDs for ranking loss)
                    if len(batch) == 3:
                        aux = batch[2]
                        class_target = aux.to(device) if isinstance(aux, torch.Tensor) else aux
                    else:
                        class_target = None
                    optimizer.zero_grad()
                    output = model(data)
                    loss = loss_criteria(output, target, class_target)
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
    # When patience is None, best_epoch is never updated (-1); use actual
    # epoch count instead so the saved filename reflects real training length.
    actual_epochs = best_epoch + 1 if (patience is not None and best_epoch >= 0) \
        else len(training_loss)
    return model, training_loss, val_loss_history, actual_epochs

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
    print(f'Overall accuracy: {(total_correct/len_data*100):.1f} %')

    print('\nAdditional Classification Report:')
    print(classification_report(y_test, y_pred))

    name_to_int = {v: k for k, v in labels_to_names.items()}
    y_test_int = [name_to_int[n] for n in y_test]
    y_scores_arr = np.vstack(y_scores)
    auc_macro    = roc_auc_score(y_test_int, y_scores_arr, multi_class='ovr', average='macro')
    auc_weighted = roc_auc_score(y_test_int, y_scores_arr, multi_class='ovr', average='weighted')
    print(f'\nAUC-ROC (macro):    {auc_macro:.4f}')
    print(f'AUC-ROC (weighted): {auc_weighted:.4f}')


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
        npimg = np.squeeze(npimg, axis=0)          # (3, H, W)
        # Reverse ImageNet normalisation so pixel values land in [0, 1]
        _mean = np.array([0.485, 0.456, 0.406])[:, None, None]
        _std  = np.array([0.229, 0.224, 0.225])[:, None, None]
        npimg = np.clip(npimg * _std + _mean, 0.0, 1.0)
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

if __name__ == '__main__':

    parser = argparse.ArgumentParser(
        description="Train CNN on Proteograms.")
    parser.add_argument('--data_dir', '-d',
                        type=str,
                        default=None,
                        help="Root directory containing 'train' and 'eval' subdirectories. "
                             "Overrides training_data_dir in config.yml.")
    parser.add_argument("--epochs", "-e",
                        type=int,
                        help="Number of training epochs.")
    parser.add_argument("--batch_size", "-b",
                        type=int,
                        help="Training batch size.")
    parser.add_argument("--lr", "-l",
                        type=float,
                        help="Training learning rate.")
    parser.add_argument('--model', '-m',
                        choices=['cnn', 'resnet18'],
                        default='cnn',
                        help="Model architecture: 'cnn' (from-scratch 4-block ConvNet) "
                             "or 'resnet18' (pretrained ResNet18 fine-tuning). Default: cnn.")
    parser.add_argument('--overwrite', '-o',
                        action='store_true',
                        help="Recreate / overwrite model")
    parser.add_argument('--min_class_size',
                        type=int,
                        default=20,
                        help="Minimum number of training samples a class must have to be "
                             "included. Classes below this threshold are excluded. "
                             "Default: 20. For fold-level or --hierarchical runs use 2 "
                             "since most folds have only 2-3 members.")    
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
    parser.add_argument('--val_size',
                        type=float,
                        default=0.15,
                        help="Fraction of train images to hold out as validation "
                             "set for early stopping (default: 0.15).")
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
    parser.add_argument('--triplet',
                        action='store_true',
                        help="Add cosine-space triplet loss (SCOP hierarchy aware) on top of "
                             "CrossEntropyLoss. Requires --model resnet18. Uses SCOPFoldSampler "
                             "to guarantee positive pairs per batch and loads class labels as "
                             "negative-mining anchors. Typical use: --level fold --triplet.")
    parser.add_argument('--ranking_loss',
                        action='store_true',
                        help="Add a physics-informed ListNet ranking loss on top of "
                             "CrossEntropyLoss. The ranking loss directly optimises cosine "
                             "similarities to match USalign TM-score order, bypassing the "
                             "classification objective. Requires --tm_score_file and "
                             "--model resnet18. Typical use: --level fold --ranking_loss "
                             "--tm_score_file /path/to/usalign_out.tsv")
    parser.add_argument('--tm_score_file',
                        type=str,
                        default=None,
                        help="Path to USalign all-vs-all TSV (columns: #PDBchain1, PDBchain2, "
                             "TM1, TM2). Required when --ranking_loss is set.")
    parser.add_argument('--ranking_weight',
                        type=float,
                        default=0.5,
                        help="Weight β for the ranking loss in: total = CE + β × ranking. "
                             "Default: 0.5. Higher values shift the model towards TM-score "
                             "alignment; lower values preserve classification accuracy.")
    parser.add_argument('--ranking_temperature',
                        type=float,
                        default=0.1,
                        help="Softmax temperature for TM-score GT distribution in ListNet "
                             "loss. Lower = sharper targets (default: 0.1).")
    args = parser.parse_args()

    config = read_yaml('config.yml')
    root_dir = args.data_dir or config['training_data_dir']
    # Get level from command line or config, with command line taking precedence. Default to 'class' if neither is provided.
    level = args.level or config.get('scope_level', 'class')

    if args.epochs is not None:
        epochs = args.epochs
    elif 'num_epochs' in config:
        epochs = config['num_epochs']
    else:
        raise ValueError("Number of epochs must be specified via command line or config.yml")
    if args.lr is not None:
        lr = args.lr
    elif 'learning_rate' in config:
        lr = config['learning_rate']
    else:
        raise ValueError("Learning rate must be specified via command line or config.yml")
    if args.batch_size is not None:
        batch_size = args.batch_size
    elif 'batch_size' in config:
        batch_size = config['batch_size']
    else:
        raise ValueError("Batch size must be specified via command line or config.yml")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    image_resize = 200  # pad to largest possible size; GAP makes the FC layer size-agnostic

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

    train_dataset = ProteogramDataset(
        tsv_file=tsv_file,
        root_dir=os.path.join(root_dir, 'train'),
        level=args.level,
        new_size=image_resize,
        pad=not args.resize,
        transform=transform_train,
        # exclude classes with certain number of samples to avoid extreme class imbalance and unreliable eval metrics; ignored if names_to_labels is supplied
        min_class_size=args.min_class_size,
        min_image_size=20,
        exclude_classes=exclude_classes)

    eval_dataset = ProteogramDataset(
        tsv_file=tsv_file,
        root_dir=os.path.join(root_dir, 'eval'),
        level=args.level,
        new_size=image_resize,
        pad=not args.resize,
        transform=transform_eval,
        names_to_labels=train_dataset.names_to_labels,
        min_image_size=20)

    # This is for reproducibility - https://docs.pytorch.org/docs/stable/notes/randomness.html
    g = torch.Generator()
    g.manual_seed(0)

    # Carve a validation split from the train folder (never touches eval/)
    n_total = len(train_dataset)
    n_val = max(1, int(n_total * args.val_size))
    n_train = n_total - n_val

    train_split, val_split = random_split(train_dataset, [n_train, n_val], generator=g)
    print(f'Split: {n_train} train / {n_val} val / {len(eval_dataset)} test (held-out)')

    # WeightedRandomSampler: oversample minority classes so each epoch sees
    # a balanced class distribution regardless of raw class frequencies.
    train_labels = [train_dataset.labels[i] for i in train_split.indices]
    label_counts = torch.bincount(torch.tensor(train_labels))
    class_weights = torch.where(
        label_counts > 0, 1.0 / label_counts.float(), torch.zeros_like(label_counts.float()))
    sample_weights = [class_weights[lbl].item() for lbl in train_labels]
    sampler = torch.utils.data.WeightedRandomSampler(
        weights=sample_weights, num_samples=len(sample_weights), replacement=True)

    class_names = train_dataset.label_names_unique

    if args.triplet and args.model == 'resnet18':
        # Build SCOPe class label list (parallel to train_dataset.files) for triplet negative mining
        _annot_df = pd.read_csv(tsv_file, sep='\t')
        _cls_sorted = sorted(_annot_df['SCOPeClass'].dropna().unique())
        _cls_to_int = {n: i for i, n in enumerate(_cls_sorted)}
        _cls_lookup = dict(zip(_annot_df['SCOPeID'], _annot_df['SCOPeClass']))
        class_labels_full = []
        for f in train_dataset.files:
            bname = os.path.basename(f).replace('.jpg', '')
            cls = _cls_lookup.get(bname, _cls_sorted[0])
            class_labels_full.append(_cls_to_int.get(cls, 0))
        print(f'Triplet mode: {len(_cls_sorted)} SCOPe classes for negative mining.')

        train_loader = DataLoader(
            WithClassLabel(train_split, class_labels_full),
            batch_sampler=SCOPFoldSampler(train_split, batch_size),
            worker_init_fn=seed_worker)
        val_loader = DataLoader(
            WithClassLabel(val_split, class_labels_full),
            batch_size=batch_size,
            shuffle=False,
            worker_init_fn=seed_worker,
            generator=g)
    elif args.ranking_loss and args.model == 'resnet18':
        if not args.tm_score_file:
            raise ValueError("--tm_score_file is required when --ranking_loss is set.")
        train_loader = DataLoader(
            WithProteinId(train_split),
            batch_size=batch_size,
            sampler=torch.utils.data.WeightedRandomSampler(
                weights=sample_weights, num_samples=len(sample_weights), replacement=True),
            worker_init_fn=seed_worker)
        val_loader = DataLoader(
            WithProteinId(val_split),
            batch_size=batch_size,
            shuffle=False,
            worker_init_fn=seed_worker,
            generator=g)
        print(f'Ranking-loss mode: loading TM-scores from {args.tm_score_file}')
    else:
        train_loader = DataLoader(train_split,
                                  batch_size=batch_size,
                                  shuffle=True,
                                  worker_init_fn=seed_worker)
        val_loader = DataLoader(val_split,
                                batch_size=batch_size,
                                shuffle=False,
                                worker_init_fn=seed_worker,
                                generator=g)

    test_loader = DataLoader(eval_dataset,
                             batch_size=1,
                             shuffle=False,
                             worker_init_fn=seed_worker,
                             generator=g)

    num_classes = len(class_names)

    if args.model == 'resnet18':
        model = build_resnet18(num_classes)
        # Differential LR: lower rate for pretrained backbone, full rate for new head
        backbone_params = [p for n, p in model.named_parameters() if 'fc' not in n and p.requires_grad]
        head_params = list(model.fc.parameters())
        optimizer = optim.AdamW([
            {'params': backbone_params, 'lr': lr * 0.1},
            {'params': head_params,     'lr': lr},
        ], weight_decay=1e-3)
        print(f'ResNet18: backbone LR={lr * 0.1:.2e}, head LR={lr:.2e}')
    else:
        model = ConvNet(num_classes)
        optimizer = optim.Adam(model.parameters(), lr=lr)
        print(f'ConvNet (from scratch): LR={lr:.2e}')

    _ce = nn.CrossEntropyLoss(label_smoothing=0.1)
    _TRIPLET_W = 0.5
    _emb_store = {}
    if args.triplet and args.model == 'resnet18':
        model.avgpool.register_forward_hook(
            lambda m, i, o: _emb_store.update({'emb': torch.flatten(o, 1)}))
        def loss_criteria(logits, targets, class_labels=None):
            ce = _ce(logits, targets)
            if class_labels is not None and 'emb' in _emb_store:
                return ce + _TRIPLET_W * scop_triplet_loss(
                    _emb_store['emb'], targets, class_labels)
            return ce
        print(f'Triplet loss enabled: CE + {_TRIPLET_W} × triplet (margin=0.3)')
    elif args.ranking_loss and args.model == 'resnet18':
        from proteogram.v2.ranking_loss import TmScoreRankingLoss
        _ranking = TmScoreRankingLoss(
            args.tm_score_file,
            temperature=args.ranking_temperature,
        )
        _RANKING_W = args.ranking_weight
        model.avgpool.register_forward_hook(
            lambda m, i, o: _emb_store.update({'emb': torch.flatten(o, 1)}))
        def loss_criteria(logits, targets, protein_ids=None):
            ce = _ce(logits, targets)
            if protein_ids is not None and 'emb' in _emb_store:
                ids = list(protein_ids)
                cov = _ranking.batch_coverage(ids)
                if cov > 0.0:
                    r_loss = _ranking.listnet_loss(_emb_store['emb'], ids)
                    return ce + _RANKING_W * r_loss
            return ce
        print(f'Ranking loss enabled: CE + {_RANKING_W} × ListNet  '
              f'(temperature={args.ranking_temperature})')
    else:
        def loss_criteria(logits, targets, class_labels=None):
            return _ce(logits, targets)
    model, training_loss, val_loss, epochs_trained = train_model(model,
                        train_loader=train_loader,
                        val_loader=val_loader,
                        optimizer=optimizer,
                        epochs=epochs,
                        patience=args.patience,
                        device=device)

    plot_losses(training_loss, val_loss,
                fig_path=os.path.join(root_dir, 'loss_curves.png'))

    model_file = config.get('model_file_prefix', 'scope_proteogram_model') \
        + f'_{args.model}_lr{lr}_bs{batch_size}_e{epochs_trained}.pt'
    model_path = os.path.join(root_dir, model_file)

    # Save model
    if os.path.exists(model_path) and not args.overwrite:
        print(f'Model file {model_path} exists and overwrite not set, not saving model.')
    else:
        # Save only the model weights
        torch.save(model.state_dict(), model_path)
        print(f'Saved model to {model_path}')

    get_accuracies(model,
                   test_loader,
                   class_names,
                   train_dataset.labels_to_names)

    view_pred_set(model,
                  test_loader,
                  num_preds=10,
                  labels_to_names=train_dataset.labels_to_names,
                  fig_path=os.path.join(root_dir, 'sample_preds.png'))

