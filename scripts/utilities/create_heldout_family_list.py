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
       only group would drop the label from training (for that case, see
       --ood_level below).
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
where <tag> encodes level/min_class_size/max_image_size/seed.

Train is then everything eligible that is in neither list. Pass the same
--level/--min_class_size/--max_image_size/--exclude_classes (and
--min_family_size for triplet runs) to the training script, or some listed
proteins will be filtered out there (it warns with a count).

Leakage-screened three-list split (--clean_split):
    A plain family holdout can still leak: SCOPe's "automated matches" families
    (sccs ending in .0) are catch-all buckets, so a protein there can belong to a
    held-out curated family, and close homologs can sit on both sides of the split.
    With --clean_split:
    1. Only curated families (not .0) are held out; .0 proteins stay eligible for
       training, where only their superfamily/fold/class labels are used.
    2. blastp compares val and test with the whole pool. Val proteins with a hit
       (E <= --blast_evalue) to test are removed, as are train proteins with a hit
       to val or test. Test is never trimmed by the screen.
    3. Val and test are fixed once, using only labels that keep at least
       --eval_min_train screened training proteins (default: the largest
       --min_train_per_class). A label that falls short returns held-out families
       to training one at a time until it meets the floor.
    4. One train list is written per --min_train_per_class value, keeping labels
       with at least that many *training* proteins. Val and test are identical
       across these lists, so the thresholds can be compared directly.
    Extra outputs:
        heldout_<group level>_train_list_<tag>_min_train<N>.lst -> pass to --train_list
        heldout_<group level>_excluded_<tag>.tsv   each screened-out protein and why
        heldout_<group level>_labels_<tag>.tsv     per-label counts and threshold status

    python create_heldout_family_list.py --clean_split \
        --data_dir ../data/.../proteograms_v2/ --tsv_file ../data/.../annotations.tsv \
        --level superfamily --max_image_size 500 --min_train_per_class 5,10,20 --seed 0

    python ../v2/train_multiple_models_randomized_eval.py --level superfamily \
        --max_image_size 500 --train_list <out_dir>/heldout_family_train_list_<tag>_min_train10.lst \
        --test_list <out_dir>/heldout_family_test_list_<tag>.lst \
        --val_list  <out_dir>/heldout_family_val_list_<tag>.lst ...

Novel superfamily or fold holdout (--clean_split --ood_level superfamily|fold):
    Family holdout still trains on every test protein's superfamily. --ood_level
    also holds out whole superfamilies (or folds) that training never sees:
    1. Groups are drawn before the family holdout, at random with --seed and
       stratified by SCOPe class. A candidate needs >= 2 curated families (or, for
       folds, >= 2 superfamilies), >= --ood_min_curated curated proteins and at
       most --ood_max_size proteins in total.
    2. A novel superfamily must keep a sibling superfamily of its fold in
       training with >= --eval_min_train proteins after screening, so the setting
       is "new superfamily, known fold". Groups that lose this are discarded.
    3. All members leave training. Curated members become queries; .0 members are
       not used anywhere because their family labels are unreliable.
    4. The BLAST screen treats novel proteins like test proteins: train and val
       proteins with a hit to them are removed.
    Novel proteins and novel-family test proteins form one evaluation corpus, so
    every query type searches against the same distractors. Extra outputs:
        heldout_<group level>_ood_list_<tag>.lst          novel-group queries
        heldout_<group level>_eval_corpus_list_<tag>.lst  test + novel proteins;
            use as scope_eval_set and as the search/aligner set
        heldout_<group level>_query_types_<tag>.tsv       pass to evaluate_methods_v2.py
            --query_types
    Keep passing the test list (not the corpus) to training's --test_list: novel
    labels have no classifier output, so they are scored by retrieval only.

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
import subprocess
import sys
import tempfile
from collections import Counter, defaultdict
from multiprocessing.pool import ThreadPool

import numpy as np
import pandas as pd
from PIL import Image

from proteogram.common import find_proteogram_images


LEVELS = ('class', 'fold', 'superfamily', 'family')
LEVEL_COLUMNS = {'class': 'SCOPeClass', 'fold': 'SCOPeFold',
                 'superfamily': 'SCOPeSuperfamily', 'family': 'SCOPeFamily'}
OVERSHOOT_TOLERANCE = 0.10
# BLAST's length adjustment depends on how many sequences are searched, so a pair
# right at the cutoff can pass in the pool but fail in a smaller list. Searching
# at a looser E-value removes those borderline pairs too.
BLAST_SCREEN_MARGIN = 10


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


def parse_thresholds(text):
    """'5,10,20' -> [5, 10, 20] (sorted, unique, all >= 1)."""
    try:
        values = sorted({int(x) for x in text.split(',') if x.strip()})
    except ValueError:
        sys.exit(f'ERROR: --min_train_per_class must be comma-separated integers, got {text!r}.')
    if not values or values[0] < 1:
        sys.exit('ERROR: --min_train_per_class values must be >= 1.')
    return values


def is_automated_family(name):
    """SCOPe 'automated matches' families have an sccs ending in .0."""
    return str(name).endswith('.0')


def run_blast(queries, subjects, evalue, blastp, workers):
    """All-vs-all blastp of queries {id: seq} against subjects {id: seq}.

    Returns the set of (query, subject) hits, self-hits excluded. Uses -subject
    rather than makeblastdb, and short numeric FASTA names so dotted SCOPe IDs
    such as d1bom.1 can't be mangled by BLAST's defline parsing. E-values are
    computed against the full pool's residue count (-dbsize), so a pair's E-value
    doesn't change when the same proteins are re-searched in a smaller list.
    The search runs at evalue * BLAST_SCREEN_MARGIN to cover borderline pairs.
    """
    q_ids, s_ids = sorted(queries), sorted(subjects)
    q_name = {f'q{i}': x for i, x in enumerate(q_ids)}
    s_name = {f's{i}': x for i, x in enumerate(s_ids)}
    n_parts = max(1, min(workers, len(q_ids)))
    db_size = sum(len(seq) for seq in subjects.values())
    with tempfile.TemporaryDirectory() as tmp:
        subject_fa = os.path.join(tmp, 'subject.fa')
        with open(subject_fa, 'w') as f:
            f.write(''.join(f'>s{i}\n{subjects[x]}\n' for i, x in enumerate(s_ids)))

        def search(part):
            k, idx = part
            query_fa = os.path.join(tmp, f'query{k}.fa')
            with open(query_fa, 'w') as f:
                f.write(''.join(f'>q{i}\n{queries[q_ids[i]]}\n' for i in idx))
            done = subprocess.run([blastp, '-query', query_fa, '-subject', subject_fa,
                                   '-evalue', str(evalue * BLAST_SCREEN_MARGIN), '-dbsize', str(db_size),
                                   '-max_target_seqs', str(len(s_ids)),
                                   '-outfmt', '6 qseqid sseqid'],
                                  capture_output=True, text=True)
            if done.returncode != 0:
                raise RuntimeError(f'blastp failed: {done.stderr[-500:]}')
            return done.stdout

        parts = list(enumerate(np.array_split(np.arange(len(q_ids)), n_parts)))
        with ThreadPool(n_parts) as pool:
            outputs = pool.map(search, parts)
    hits = set()
    for out in outputs:
        for line in out.splitlines():
            q, t = line.split('\t')[:2]
            if q_name[q] != s_name[t]:
                hits.add((q_name[q], s_name[t]))
    return hits


def choose_novel_groups(rows, fold_of, class_of, level, frac, min_curated, max_size,
                        min_sibling, rng):
    """Pick whole superfamilies or folds to keep out of training.

    rows: (protein, superfamily, group, family). Returns
    ({group: curated proteins}, {group: automated-match proteins}, n_candidates).
    Selection is stratified by SCOPe class. A novel superfamily always leaves at
    least one other superfamily of its fold, with >= min_sibling proteins, outside
    the novel set.
    """
    members = defaultdict(list)
    fold_superfamilies = defaultdict(set)
    for r in rows:
        members[r[1] if level == 'superfamily' else fold_of[r[0]]].append(r)
        fold_superfamilies[fold_of[r[0]]].add(r[1])
    label_size = Counter(r[1] for r in rows)

    def has_sibling(group, fold, taken):
        return any(label_size[x] >= min_sibling
                   for x in fold_superfamilies[fold] - taken - {group})

    by_class = defaultdict(list)
    for g, ms in members.items():
        curated = [r for r in ms if not is_automated_family(r[3])]
        finer = {r[3] if level == 'superfamily' else r[1] for r in curated}
        if len(finer) < 2 or len(curated) < min_curated or len(ms) > max_size:
            continue
        if level == 'superfamily' and not has_sibling(g, fold_of[ms[0][0]], set()):
            continue
        by_class[class_of[ms[0][0]]].append(g)
    chosen = set()
    for cls in sorted(by_class, key=str):
        names = sorted(by_class[cls], key=str)
        target = int(round(frac * len(names)))
        taken = 0
        for i in rng.permutation(len(names)):
            if taken >= target:
                break
            g = names[i]
            if level == 'superfamily':
                if not has_sibling(g, fold_of[members[g][0][0]], chosen):
                    continue
            chosen.add(g)
            taken += 1
    curated = {g: sorted(r[0] for r in members[g] if not is_automated_family(r[3]))
               for g in chosen}
    automated = {g: sorted(r[0] for r in members[g] if is_automated_family(r[3]))
                 for g in chosen}
    return curated, automated, sum(len(v) for v in by_class.values())


def screen_split(assigned, groups, group_label, pool_labels, hits, eval_min_train,
                 novel_groups=None, novel_fold=None, label_fold=None):
    """Apply the homology screen and the training-count floor for held-out labels.

    Returns (assigned, test, val, train, removed_val, removed_train, released,
    novel_groups, lost_novel). While a held-out label has under eval_min_train
    screened training proteins, one of its held-out groups (val before test, then
    the largest) returns to the training pool and the screen is recomputed,
    because that group no longer excludes its homologs. `released` lists the
    returned groups. novel_groups {group: proteins} are screened like test. With
    novel_fold {group: fold}, a novel group whose fold has no training label with
    eval_min_train proteins is discarded and listed in `lost_novel`.
    """
    assigned = dict(assigned)
    novel_groups = dict(novel_groups or {})
    released = []
    lost_novel = []
    while True:
        novel = {p for members in novel_groups.values() for p in members}
        test = {p for g, s in assigned.items() if s == 'test' for p in groups[g]}
        val = {p for g, s in assigned.items() if s == 'val' for p in groups[g]}
        test_like = test | novel
        near_test = ({t for q, t in hits if q in test_like}
                     | {q for q, t in hits if t in test_like})
        removed_val = val & near_test
        kept_val = val - removed_val
        held = test_like | kept_val
        near_held = {t for q, t in hits if q in held} | {q for q, t in hits if t in held}
        candidates = set(pool_labels) - test - val
        removed_train = candidates & near_held
        train = candidates - removed_train
        counts = Counter(pool_labels[p] for p in train)
        short = {group_label[g] for g in assigned if counts[group_label[g]] < eval_min_train}
        if not short:
            if novel_fold:
                known = {label_fold[lbl] for lbl, c in counts.items() if c >= eval_min_train}
                unsupported = [g for g in novel_groups if novel_fold[g] not in known]
                if unsupported:
                    for g in unsupported:
                        del novel_groups[g]
                        lost_novel.append(g)
                    continue
            return (assigned, test, kept_val, train, removed_val, removed_train, released,
                    novel_groups, lost_novel)
        for lbl in sorted(short, key=str):
            options = [g for g in assigned if group_label[g] == lbl]
            g = min(options, key=lambda x: (assigned[x] != 'val', -len(groups[x]), str(x)))
            del assigned[g]
            released.append(g)


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
    parser.add_argument('--min_class_size', type=int, default=None,
                        help="Same as training: labels at --level with fewer members are "
                             "excluded. Default: 20 (the training default). Not used with "
                             "--clean_split, which applies --min_train_per_class instead.")
    parser.add_argument('--min_family_size', type=int, default=0,
                        help="Same as training's --min_family_size (triplet_hierarchy runs "
                             "only); leave at 0 for ce/focal runs. Default: 0.")
    parser.add_argument('--exclude_classes', '-x', default=None,
                        help="Same as training: comma-separated labels at --level to exclude.")
    parser.add_argument('--test_frac', type=float, default=None,
                        help="Target fraction of eligible proteins held out for test. "
                             "Default: 0.15.")
    parser.add_argument('--val_frac', type=float, default=None,
                        help="Target fraction of eligible proteins held out for validation; "
                             "0 writes no val list. Default: 0.15, or 0.10 with --clean_split.")
    parser.add_argument('--min_train_per_label', type=int, default=None,
                        help="Minimum proteins each label must keep in training after holdout. "
                             "Default: max(2, min_class_size // 2).")
    parser.add_argument('--group_sampling', choices=['uniform', 'size_weighted'],
                        default='uniform',
                        help="How held-out groups are drawn. 'uniform' tends to favor small "
                             "families; 'size_weighted' draws in proportion to size. "
                             "Default: uniform.")
    parser.add_argument('--clean_split', action='store_true',
                        help="Write a leakage-screened train/val/test split: hold out curated "
                             "families only and remove homologs across splits with blastp "
                             "(see the module docstring).")
    parser.add_argument('--min_train_per_class', default='10',
                        help="--clean_split: comma-separated minimum *training* proteins per "
                             "label; one train list is written per value. Default: 10.")
    parser.add_argument('--eval_min_train', type=int, default=None,
                        help="--clean_split: labels in val/test must keep at least this many "
                             "screened training proteins. Default: the largest "
                             "--min_train_per_class, so val/test are valid for every list.")
    parser.add_argument('--ood_level', choices=['superfamily', 'fold'], default=None,
                        help="--clean_split: also hold out whole superfamilies or folds that "
                             "training never sees, scored by retrieval. Default: off.")
    parser.add_argument('--ood_frac', type=float, default=0.30,
                        help="--ood_level: fraction of candidate groups to hold out, per "
                             "SCOPe class. Default: 0.30.")
    parser.add_argument('--ood_min_curated', type=int, default=5,
                        help="--ood_level: minimum curated proteins in a novel group. "
                             "Default: 5.")
    parser.add_argument('--ood_max_size', type=int, default=100,
                        help="--ood_level: maximum total proteins in a novel group, so one "
                             "group can't remove a large share of training. Default: 100.")
    parser.add_argument('--blast_evalue', type=float, default=1e-3,
                        help="--clean_split: no pair at or below this E-value may cross the "
                             "split; the search itself runs 10x looser to catch borderline "
                             "pairs. Default: 1e-3.")
    parser.add_argument('--blastp', default='blastp',
                        help="--clean_split: blastp executable. Default: blastp on PATH.")
    parser.add_argument('--blast_workers', type=int, default=os.cpu_count() or 1,
                        help="--clean_split: parallel blastp processes. Default: all CPUs.")
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
    thresholds = []
    if args.ood_level:
        if not args.clean_split:
            sys.exit('ERROR: --ood_level requires --clean_split.')
        if args.level != 'superfamily':
            sys.exit('ERROR: --ood_level requires --level superfamily.')
        if not 0.0 < args.ood_frac < 1.0:
            sys.exit('ERROR: --ood_frac must be in (0, 1).')
        if args.ood_min_curated < 2 or args.ood_max_size < args.ood_min_curated:
            sys.exit('ERROR: need 2 <= --ood_min_curated <= --ood_max_size.')
    if args.clean_split:
        if args.min_class_size is not None or args.min_train_per_label is not None:
            sys.exit('ERROR: --clean_split sets its limits with --min_train_per_class and '
                     '--eval_min_train; drop --min_class_size/--min_train_per_label.')
        if args.holdout_level != 'family':
            sys.exit('ERROR: --clean_split requires --holdout_level family.')
        thresholds = parse_thresholds(args.min_train_per_class)
        if args.eval_min_train is None:
            args.eval_min_train = thresholds[-1]
        if args.eval_min_train < thresholds[-1]:
            sys.exit('ERROR: --eval_min_train must be >= the largest --min_train_per_class, or '
                     'some val/test labels would be missing from that train list.')
        args.min_class_size = thresholds[0]
        args.min_train_per_label = args.eval_min_train
    elif args.min_class_size is None:
        args.min_class_size = 20
    test_default, val_default = (0.15, 0.10) if args.clean_split else (0.15, 0.15)
    args.test_frac = test_default if args.test_frac is None else args.test_frac
    args.val_frac = val_default if args.val_frac is None else args.val_frac
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
    if args.clean_split:
        tag = (f'level-{args.level}_holdout-{args.holdout_level}_seed{args.seed}_'
               f'max_image_size{args.max_image_size}_clean_blast{args.blast_evalue:g}_'
               f'eval_min_train{args.eval_min_train}')
        if args.min_family_size:
            tag += f'_min_family_size{args.min_family_size}'
        if args.ood_level:
            tag += f'_ood-{args.ood_level}{args.ood_frac:g}'
    out_dir = args.out_dir or os.path.dirname(os.path.abspath(args.data_dir.rstrip('/')))
    stem = f'heldout_{args.holdout_level}'
    paths = {'test': os.path.join(out_dir, f'{stem}_test_list_{tag}.lst'),
             'val': os.path.join(out_dir, f'{stem}_val_list_{tag}.lst'),
             'summary': os.path.join(out_dir, f'{stem}_summary_{tag}.tsv')}
    if args.val_frac == 0:
        del paths['val']
    if args.clean_split:
        paths['excluded'] = os.path.join(out_dir, f'{stem}_excluded_{tag}.tsv')
        paths['labels'] = os.path.join(out_dir, f'{stem}_labels_{tag}.tsv')
        if args.ood_level:
            paths['ood'] = os.path.join(out_dir, f'{stem}_ood_list_{tag}.lst')
            paths['corpus'] = os.path.join(out_dir, f'{stem}_eval_corpus_list_{tag}.lst')
            paths['query_types'] = os.path.join(out_dir, f'{stem}_query_types_{tag}.tsv')
        for n in thresholds:
            paths[f'train{n}'] = os.path.join(out_dir, f'{stem}_train_list_{tag}_min_train{n}.lst')
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
    if args.clean_split:
        needed.add('PDBSequence')
    if args.ood_level:
        needed |= {LEVEL_COLUMNS['class'], LEVEL_COLUMNS['fold']}
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

    sequences = {}
    if args.clean_split:
        # A protein that can't be screened can't be shown to be leakage-free.
        seq_of = annot['PDBSequence']
        sequences = {r[0]: str(seq_of[r[0]]).strip() for r in rows if not is_blank(seq_of[r[0]])}
        before = len(rows)
        rows = [r for r in rows if r[0] in sequences]
        if before != len(rows):
            print(f'{before - len(rows)} proteogram(s) with no sequence excluded '
                  f'(they cannot be homology-screened).')

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

    # ---- Novel superfamilies/folds, chosen before the family holdout ----
    novel_groups, novel_automated, novel_label, fold_of = {}, {}, {}, {}
    if args.ood_level:
        fold_of = {r[0]: annot.at[r[0], LEVEL_COLUMNS['fold']] for r in rows}
        class_of = {r[0]: annot.at[r[0], LEVEL_COLUMNS['class']] for r in rows}
        novel_label = {r[0]: r[1] for r in rows}
        # A separate stream, so the family draw doesn't depend on how many
        # random numbers this selection consumed.
        novel_groups, novel_automated, n_novel_candidates = choose_novel_groups(
            rows, fold_of, class_of, args.ood_level, args.ood_frac, args.ood_min_curated,
            args.ood_max_size, args.eval_min_train, np.random.default_rng([args.seed, 1]))
        if not novel_groups:
            sys.exit(f'ERROR: no novel {args.ood_level} could be selected; lower '
                     f'--ood_min_curated or raise --ood_frac/--ood_max_size.')
        leaving = {p for g in novel_groups for p in novel_groups[g] + novel_automated[g]}
        rows = [r for r in rows if r[0] not in leaving]
        print(f'Novel {args.ood_level}: {len(novel_groups)} of {n_novel_candidates} candidate '
              f'group(s) held out ({sum(map(len, novel_groups.values()))} curated proteins as '
              f'queries; {sum(map(len, novel_automated.values()))} automated-match members '
              f'unused) -> {len(rows)} proteograms remain for the family holdout.')

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
    candidates = {g: m for g, m in groups.items() if groups_left[group_label[g]] >= 2
                  and not (args.clean_split and is_automated_family(g))}
    if args.clean_split:
        n_auto = sum(len(m) for g, m in groups.items() if is_automated_family(g))
        print(f'{n_auto} of {len(rows)} proteins are in automated-match (.0) families; '
              f'those families are not eligible for holdout.')
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

    # ---- Homology screen and per-threshold train lists (--clean_split) ----
    final_members = {s: {p for g, sp in assigned.items() if sp == s for p in groups[g]}
                     for s in ('test', 'val')}
    train_lists = {}
    if args.clean_split:
        pool_labels = {r[0]: r[1] for r in rows}
        held = final_members['test'] | final_members['val']
        print(f'Screening {len(held)} held-out proteins against {len(pool_labels)} with blastp '
              f'(E <= {args.blast_evalue:g}, {args.blast_workers} workers)...')
        novel_all = {p for members in novel_groups.values() for p in members}
        hits = run_blast({p: sequences[p] for p in held | novel_all},
                         {p: sequences[p] for p in set(pool_labels) | novel_all},
                         args.blast_evalue, args.blastp, args.blast_workers)
        novel_fold, label_fold = None, None
        if args.ood_level == 'superfamily':
            novel_fold = {g: fold_of[m[0]] for g, m in novel_groups.items()}
            label_fold = {r[1]: fold_of[r[0]] for r in rows}
        (assigned, test_set, val_set, train_set, removed_val, removed_train, released,
         kept_novel, lost_novel) = screen_split(
            assigned, groups, group_label, pool_labels, hits, args.eval_min_train,
            novel_groups, novel_fold, label_fold)
        if not test_set:
            sys.exit('ERROR: no test proteins left after screening; lower --eval_min_train or '
                     'the holdout fractions.')
        final_members = {'test': test_set, 'val': val_set}
        novel_set = {p for members in kept_novel.values() for p in members}
        if args.ood_level:
            if not novel_set:
                sys.exit('ERROR: no novel groups left after screening; lower --eval_min_train.')
            lost_note = ''
            if args.ood_level == 'superfamily':
                lost_note = (f'; {len(lost_novel)} group(s) discarded because their fold kept '
                             f'no training superfamily with >= {args.eval_min_train} proteins')
            print(f'Novel {args.ood_level} after screening: {len(kept_novel)} group(s), '
                  f'{len(novel_set)} queries{lost_note}.')
        train_left = Counter(pool_labels[p] for p in train_set)
        eval_labels = {group_label[g] for g in assigned}
        dropped = {group_label[g] for g in released} - eval_labels
        print(f'Screen removed {len(removed_val)} val and {len(removed_train)} train protein(s). '
              f'{len(released)} held-out group(s) returned to train to keep >= '
              f'{args.eval_min_train} training proteins per val/test label; {len(dropped)} '
              f'label(s) left with no held-out group; BLAST -dbsize {sum(len(sequences[p]) for p in pool_labels)}.')
        print(f'After screening: {len(test_set)} test, {len(val_set)} val, '
              f'{len(train_set)} train candidates.')
        for n in thresholds:
            keep = {lbl for lbl, c in train_left.items() if c >= n}
            train_lists[n] = sorted(p for p in train_set if pool_labels[p] in keep)
            print(f'  --min_train_per_class {n}: {len(train_lists[n])} train proteins in '
                  f'{len(keep)} label(s); {len(train_left) - len(keep)} label(s) dropped.')

    # ---- Write outputs ----
    os.makedirs(out_dir, exist_ok=True)
    for n, members in train_lists.items():
        with open(paths[f'train{n}'], 'w') as f:
            f.write(''.join(p + '\n' for p in members))
        print(f'train list, min {n} per label ({len(members)} proteins) written to '
              f'{paths[f"train{n}"]}')
    if args.clean_split:
        excluded = ([{'SCOPeID': p, 'would_be_split': 'val', 'reason': 'blast_hit_to_test',
                      label_col: pool_labels[p]} for p in sorted(removed_val)]
                    + [{'SCOPeID': p, 'would_be_split': 'train',
                        'reason': 'blast_hit_to_val_or_test', label_col: pool_labels[p]}
                       for p in sorted(removed_train)])
        excluded += [{'SCOPeID': p, 'would_be_split': 'novel',
                      'reason': 'automated_member_of_novel_group', label_col: novel_label[p]}
                     for g in novel_groups for p in novel_automated[g]]
        excluded += [{'SCOPeID': p, 'would_be_split': 'novel',
                      'reason': 'novel_group_lost_known_fold', label_col: novel_label[p]}
                     for g in lost_novel for p in novel_groups[g]]
        pd.DataFrame(excluded, columns=['SCOPeID', 'would_be_split', 'reason', label_col]).to_csv(
            paths['excluded'], sep='\t', index=False)
        n_test = Counter(pool_labels[p] for p in final_members['test'])
        n_val = Counter(pool_labels[p] for p in final_members['val'])
        n_removed = Counter(pool_labels[p] for p in removed_train)
        n_novel = Counter(novel_label[p] for p in novel_set)
        pd.DataFrame([{label_col: lbl, 'train': train_left[lbl], 'val': n_val[lbl],
                       'test': n_test[lbl], 'novel': n_novel[lbl],
                       'train_removed_by_blast': n_removed[lbl],
                       'kept_out_of_eval': lbl in dropped,
                       **{f'in_min_train{n}': train_left[lbl] >= n for n in thresholds}}
                      for lbl in sorted(set(pool_labels.values()) | set(novel_label.values()),
                                        key=str)]).to_csv(
            paths['labels'], sep='\t', index=False)
        if args.ood_level:
            novel_type = f'novel_{args.ood_level}'
            with open(paths['ood'], 'w') as f:
                f.write(''.join(p + '\n' for p in sorted(novel_set)))
            corpus = sorted(final_members['test'] | novel_set)
            with open(paths['corpus'], 'w') as f:
                f.write(''.join(p + '\n' for p in corpus))
            pd.DataFrame([{'SCOPeID': p,
                           'query_type': novel_type if p in novel_set else 'novel_family',
                           'class': annot.at[p, LEVEL_COLUMNS['class']],
                           'fold': annot.at[p, LEVEL_COLUMNS['fold']],
                           'superfamily': annot.at[p, LEVEL_COLUMNS['superfamily']],
                           'family': annot.at[p, LEVEL_COLUMNS['family']]}
                          for p in corpus]).to_csv(paths['query_types'], sep='\t', index=False)
            info = annot.loc[corpus, [LEVEL_COLUMNS['fold'], LEVEL_COLUMNS['superfamily']]]
            fold_n = info.groupby(LEVEL_COLUMNS['fold']).size()
            pair_n = info.groupby([LEVEL_COLUMNS['fold'], LEVEL_COLUMNS['superfamily']]).size()
            n_remote_fold = sum(
                1 for p in novel_set
                if fold_n[info.at[p, LEVEL_COLUMNS['fold']]]
                > pair_n[(info.at[p, LEVEL_COLUMNS['fold']], info.at[p, LEVEL_COLUMNS['superfamily']])])
            print(f'Novel list ({len(novel_set)} proteins) written to {paths["ood"]}')
            print(f'Evaluation corpus ({len(corpus)} proteins) written to {paths["corpus"]}')
            print(f'Query types written to {paths["query_types"]}')
            print(f'{n_remote_fold} of {len(novel_set)} novel queries have a same-fold, '
                  f'different-superfamily relative in the corpus (remote fold is scored on these).')
        print(f'Excluded proteins written to {paths["excluded"]}')
        print(f'Per-label counts written to {paths["labels"]}')
    summary = []
    for split in ('test', 'val'):
        if split not in paths:
            continue
        held = [g for g in order if assigned.get(g) == split]
        prefixes = sorted(p for g in held for p in groups[g] if p in final_members[split])
        with open(paths[split], 'w') as f:
            f.write(''.join(p + '\n' for p in prefixes))
        print(f'{split} list ({len(prefixes)} proteins) written to {paths[split]}')
        for g in held:
            for p in groups[g]:
                if p not in final_members[split]:
                    continue
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
