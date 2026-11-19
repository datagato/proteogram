"""
Create held-out-family test and validation lists for
scripts/v2/train_multiple_models_randomized_eval.py (--test_list / --val_list).

A random stratified split puts members of the same family in both train and
test, so a model trained at superfamily level is mostly tested on families it
has already seen. This script instead holds out whole groups one level below the
training --level (families, when training at superfamily level): every held-out
protein comes from a family the model never saw, but its superfamily is still
represented in training by its other families. That's the classic SCOP
remote-homology setting -- recognizing a relative you've only seen distant
cousins of -- and it makes superfamily-level P@K a measure of generalization
rather than near-neighbor recall.

How groups are chosen:
    1. Eligibility mirrors ProteogramDataset in the training script, in the same
       order, so the pool matches what training sees: image size
       (--min_image_size..--max_image_size), annotation present, --exclude_classes,
       labels at --level with >= --min_class_size members, and (for triplet runs)
       families with >= --min_family_size members.
    2. Only labels with >= 2 holdout groups can contribute; holding out a label's
       only group would drop the label from training (that's the OOD case --
       see create_ood_test_list.py).
    3. Groups are visited in a seeded random order (--group_sampling uniform, or
       size_weighted to counter the drift toward small families) and assigned to
       test, then validation, until each reaches its fraction of the pool. A group
       is skipped if taking it would leave its label with no group in training,
       fewer than --min_train_per_label training proteins, or overshoot the target
       by more than 10%.

Outputs (in --out_dir, default: the parent of --data_dir, where training writes):
    heldout_<group level>_test_list_<tag>.lst   -> pass to --test_list
    heldout_<group level>_val_list_<tag>.lst    -> pass to --val_list
    heldout_<group level>_summary_<tag>.tsv     one row per held-out protein
where <tag> encodes level/min_class_size/max_image_size/seed, so
create_ood_test_list.py can read those settings from the test list's name.

Train is then everything eligible that is in neither list. Pass the same
--level/--min_class_size/--max_image_size/--exclude_classes (and
--min_family_size for triplet runs) to the training script, or some listed
proteins will be filtered out there (it warns with a count).

Usage example:
    python create_heldout_family_list.py \
        --data_dir ../data/.../proteograms_v2/ \
        --tsv_file ../data/.../ProteogramData_SCOP_RCSB_PDBe_AnnotationsLookup_AllSCOPe208.tsv \
        --level superfamily --max_image_size 500 --min_class_size 15 --seed 0

    python ../v2/train_multiple_models_randomized_eval.py --level superfamily \
        --max_image_size 500 --min_class_size 15 \
        --test_list <out_dir>/heldout_family_test_list_<tag>.lst \
        --val_list  <out_dir>/heldout_family_val_list_<tag>.lst ...
"""
import argparse
import os
import sys
from collections import Counter, defaultdict

import numpy as np
import pandas as pd
from PIL import Image

from proteogram.common import find_proteogram_images


LEVELS = ('class', 'fold', 'superfamily', 'family')
LEVEL_COLUMNS = {'class': 'SCOPeClass', 'fold': 'SCOPeFold',
                 'superfamily': 'SCOPeSuperfamily', 'family': 'SCOPeFamily'}
OVERSHOOT_TOLERANCE = 0.10


def prefix_of(path):
    return os.path.splitext(os.path.basename(path))[0]


def is_blank(value):
    return pd.isna(value) or str(value).strip() == ''


def keep_where(rows, mask):
    return [r for r, keep in zip(rows, mask) if keep]


def visit_order(groups, sampling, rng):
    """Seeded visiting order for candidate groups.

    'uniform' shuffles groups; 'size_weighted' draws each group with probability
    proportional to its size (Efraimidis-Spirakis keys u^(1/size)), so large
    families get held out about as often as their share of the proteins.
    """
    names = sorted(groups, key=str)
    if sampling == 'uniform':
        return [names[i] for i in rng.permutation(len(names))]
    keys = [rng.random() ** (1.0 / len(groups[g])) for g in names]
    return [g for _, g in sorted(zip(keys, names), key=lambda t: -t[0])]


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Write held-out-family test/val lists for "
                    "train_multiple_models_randomized_eval.py --test_list/--val_list.")
    parser.add_argument('--data_dir', '-d', required=True,
                        help="Proteogram directory used for training (flat layout).")
    parser.add_argument('--tsv_file', '-t', required=True,
                        help="Annotation TSV used for training.")
    parser.add_argument('--level', choices=LEVELS[:-1], default='superfamily',
                        help="SCOPe level the model is trained at. Default: superfamily.")
    parser.add_argument('--holdout_level', choices=LEVELS[1:], default=None,
                        help="Level of the groups to hold out; must be finer than --level. "
                             "Default: one level below --level (family for superfamily).")
    parser.add_argument('--max_image_size', type=int, required=True,
                        help="Residue/pixel cutoff, same as training.")
    parser.add_argument('--min_image_size', type=int, default=20,
                        help="Lower size cutoff (hard-coded to 20 in training). Default: 20.")
    parser.add_argument('--min_class_size', type=int, default=20,
                        help="Same as training: labels at --level with fewer members are "
                             "excluded. Default: 20 (the training default).")
    parser.add_argument('--min_family_size', type=int, default=0,
                        help="Same as training's --min_family_size (triplet_hierarchy runs "
                             "only); leave at 0 for ce/focal runs. Default: 0.")
    parser.add_argument('--exclude_classes', '-x', default=None,
                        help="Same as training: comma-separated labels at --level to exclude.")
    parser.add_argument('--test_frac', type=float, default=0.15,
                        help="Target fraction of eligible proteins held out for test. "
                             "Default: 0.15.")
    parser.add_argument('--val_frac', type=float, default=0.15,
                        help="Target fraction of eligible proteins held out for validation; "
                             "0 writes no val list. Default: 0.15.")
    parser.add_argument('--min_train_per_label', type=int, default=None,
                        help="Minimum proteins each label must keep in training after holdout. "
                             "Default: max(2, min_class_size // 2).")
    parser.add_argument('--group_sampling', choices=['uniform', 'size_weighted'],
                        default='uniform',
                        help="How held-out groups are drawn. 'uniform' tends to favor small "
                             "families; 'size_weighted' draws in proportion to size. "
                             "Default: uniform.")
    parser.add_argument('--seed', type=int, default=0, help="Random seed. Default: 0.")
    parser.add_argument('--out_dir', '-o', default=None,
                        help="Output directory. Default: the parent of --data_dir.")
    parser.add_argument('--overwrite', action='store_true',
                        help="Overwrite existing output files.")
    args = parser.parse_args()

    if args.holdout_level is None:
        args.holdout_level = LEVELS[LEVELS.index(args.level) + 1]
    if LEVELS.index(args.holdout_level) <= LEVELS.index(args.level):
        sys.exit(f'ERROR: --holdout_level ({args.holdout_level}) must be finer than '
                 f'--level ({args.level}).')
    for name in ('test_frac', 'val_frac'):
        if not 0.0 <= getattr(args, name) < 1.0:
            sys.exit(f'ERROR: --{name} must be in [0, 1).')
    if args.test_frac <= 0:
        sys.exit('ERROR: --test_frac must be > 0.')
    if args.test_frac + args.val_frac >= 1.0:
        sys.exit('ERROR: --test_frac + --val_frac must be < 1.')
    if args.min_train_per_label is None:
        args.min_train_per_label = max(2, args.min_class_size // 2)

    label_col = LEVEL_COLUMNS[args.level]
    group_col = LEVEL_COLUMNS[args.holdout_level]
    family_col = LEVEL_COLUMNS['family']

    tag = (f'level-{args.level}_holdout-{args.holdout_level}_seed{args.seed}_'
           f'max_image_size{args.max_image_size}_min_class_size{args.min_class_size}')
    if args.min_family_size:
        tag += f'_min_family_size{args.min_family_size}'
    out_dir = args.out_dir or os.path.dirname(os.path.abspath(args.data_dir.rstrip('/')))
    stem = f'heldout_{args.holdout_level}'
    paths = {'test': os.path.join(out_dir, f'{stem}_test_list_{tag}.lst'),
             'val': os.path.join(out_dir, f'{stem}_val_list_{tag}.lst'),
             'summary': os.path.join(out_dir, f'{stem}_summary_{tag}.tsv')}
    if args.val_frac == 0:
        del paths['val']
    existing = [p for p in paths.values() if os.path.exists(p)]
    if existing and not args.overwrite:
        sys.exit(f'ERROR: {existing[0]} exists; pass --overwrite to replace it.')

    print(f'Training level={args.level} ({label_col}); holding out {args.holdout_level} '
          f'groups ({group_col}); min_class_size={args.min_class_size}, image size '
          f'{args.min_image_size}..{args.max_image_size} px, seed={args.seed}')

    # ---- Eligibility, mirroring ProteogramDataset ----
    files = find_proteogram_images(args.data_dir)
    if not files:
        sys.exit(f'ERROR: no proteogram images found in {args.data_dir}')
    sized = []
    for f in files:
        w, h = Image.open(f).size
        if (args.min_image_size <= w <= args.max_image_size
                and args.min_image_size <= h <= args.max_image_size):
            sized.append(f)
    print(f'{len(files)} proteograms found; {len(files) - len(sized)} outside the size '
          f'range excluded -> {len(sized)}.')

    annot = pd.read_csv(args.tsv_file, sep='\t')
    needed = {'SCOPeID', label_col, group_col, family_col}
    missing_cols = needed - set(annot.columns)
    if missing_cols:
        sys.exit(f'ERROR: column(s) {sorted(missing_cols)} not found in {args.tsv_file}')
    annot = annot.drop_duplicates('SCOPeID', keep='first').set_index('SCOPeID')

    rows = []  # (prefix, label, group, family)
    n_unannotated = n_blank = 0
    for f in sized:
        p = prefix_of(f)
        if p not in annot.index:
            n_unannotated += 1
            continue
        r = annot.loc[p]
        if is_blank(r[label_col]) or is_blank(r[group_col]):
            n_blank += 1
            continue
        rows.append((p, r[label_col], r[group_col], r[family_col]))
    if n_unannotated:
        print(f'{n_unannotated} proteogram(s) not in the annotation TSV excluded.')
    if n_blank:
        print(f'{n_blank} proteogram(s) with a blank {label_col}/{group_col} excluded.')

    if args.exclude_classes:
        excluded = {c.strip() for c in args.exclude_classes.split(',')}
        before = len(rows)
        rows = [r for r in rows if r[1] not in excluded]
        print(f'--exclude_classes removed {before - len(rows)} proteogram(s).')

    label_counts = Counter(r[1] for r in rows)
    rows = [r for r in rows if label_counts[r[1]] >= args.min_class_size]
    n_small_labels = sum(1 for c in label_counts.values() if c < args.min_class_size)
    print(f'{n_small_labels} {args.level} label(s) under min_class_size excluded -> '
          f'{len(rows)} proteograms.')

    if args.min_family_size > 0:
        fam_counts = Counter(r[3] for r in rows)
        rows = keep_where(rows, [fam_counts[r[3]] >= args.min_family_size for r in rows])
        print(f'--min_family_size {args.min_family_size} -> {len(rows)} proteograms.')

    if not rows:
        sys.exit('ERROR: no eligible proteograms left after filtering.')

    # ---- Group structure ----
    groups = defaultdict(list)       # group -> prefixes
    group_label = {}                 # group -> label at --level
    for p, lbl, grp, _ in rows:
        groups[grp].append(p)
        if group_label.setdefault(grp, lbl) != lbl:
            sys.exit(f'ERROR: {args.holdout_level} {grp!r} appears under more than one '
                     f'{args.level} ({group_label[grp]!r}, {lbl!r}); check the annotation TSV.')
    train_left = Counter(r[1] for r in rows)                 # label -> proteins in train
    groups_left = Counter(group_label.values())              # label -> groups in train
    candidates = {g: m for g, m in groups.items() if groups_left[group_label[g]] >= 2}
    n_total = len(rows)
    print(f'{len(train_left)} {args.level} label(s), {len(groups)} {args.holdout_level} '
          f'group(s); {len(candidates)} group(s) in {len({group_label[g] for g in candidates})} '
          f'label(s) with >= 2 groups are candidates for holdout '
          f'({sum(map(len, candidates.values()))} proteograms).')

    # ---- Assign groups to test, then val ----
    rng = np.random.default_rng(args.seed)
    order = visit_order(candidates, args.group_sampling, rng)
    assigned = {}  # group -> split
    targets = [('test', args.test_frac), ('val', args.val_frac)]
    for split, frac in targets:
        if frac <= 0:
            continue
        target = round(frac * n_total)
        taken = 0
        skipped = Counter()
        for g in order:
            if taken >= target:
                break
            if g in assigned:
                continue
            size, lbl = len(groups[g]), group_label[g]
            if groups_left[lbl] < 2:
                skipped['last group of its label'] += 1
                continue
            if train_left[lbl] - size < args.min_train_per_label:
                skipped['min_train_per_label'] += 1
                continue
            if taken + size > target * (1 + OVERSHOOT_TOLERANCE):
                skipped['would overshoot target'] += 1
                continue
            assigned[g] = split
            taken += size
            groups_left[lbl] -= 1
            train_left[lbl] -= size
        note = f' (skipped: {dict(skipped)})' if skipped else ''
        print(f'{split}: {taken} proteins ({taken / n_total:.1%}, target {frac:.0%}) '
              f'from {sum(1 for s in assigned.values() if s == split)} '
              f'{args.holdout_level} group(s){note}')
        if taken < 0.9 * target:
            print(f'WARNING: {split} fell short of its target -- too few candidate groups '
                  f'under these constraints. Lower --min_train_per_label or the fractions.')

    if not any(s == 'test' for s in assigned.values()):
        sys.exit('ERROR: no groups could be held out for test.')

    # ---- Write outputs ----
    os.makedirs(out_dir, exist_ok=True)
    summary = []
    for split in ('test', 'val'):
        if split not in paths:
            continue
        held = [g for g in order if assigned.get(g) == split]
        prefixes = sorted(p for g in held for p in groups[g])
        with open(paths[split], 'w') as f:
            f.write(''.join(p + '\n' for p in prefixes))
        print(f'{split} list ({len(prefixes)} proteins) written to {paths[split]}')
        for g in held:
            for p in groups[g]:
                summary.append({'SCOPeID': p, 'split': split, group_col: g,
                                label_col: group_label[g], 'group_size': len(groups[g]),
                                'label_train_remaining': train_left[group_label[g]]})
    pd.DataFrame(summary).sort_values(['split', label_col, group_col, 'SCOPeID']).to_csv(
        paths['summary'], sep='\t', index=False)
    print(f'Summary written to {paths["summary"]}')

    # Size check: uniform group sampling favors small groups, which makes the
    # held-out set easier or harder in ways unrelated to novelty.
    held_sizes = [len(groups[g]) for g in assigned]
    cand_sizes = [len(m) for m in candidates.values()]
    print(f'{args.holdout_level} size, median (mean): held out {np.median(held_sizes):.0f} '
          f'({np.mean(held_sizes):.1f}) vs all candidates {np.median(cand_sizes):.0f} '
          f'({np.mean(cand_sizes):.1f}).')
    n_single = sum(1 for s in held_sizes if s == 1)
    if n_single:
        print(f'Note: {n_single} held-out group(s) have one member, so they have no '
              f'same-{args.holdout_level} hits for {args.holdout_level}-level P@K.')
