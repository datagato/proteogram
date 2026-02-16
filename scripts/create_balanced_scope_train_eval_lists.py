#!/usr/bin/env python3
"""
Create balanced training and evaluation datasets from CD-HIT clustered sequences.

This script reads a FASTA-like .lst file from CD-HIT results and a TSV lookup
table (created with the "create_scope_lookup_table.py" script) with SCOPe 
annotations to create balanced training and evaluation datasets with N proteins from a specified SCOPe classification level.

Usage:
    python create_balanced_scope_eval_list.py \
        --lst-file path/to/cdhits_result.lst \
        --lookup-tsv path/to/annotations.tsv \
        --class-column SCOPeClass \
        --n-per-class 100 \
        --eval-fraction 0.2 \
        --train-output train_set.txt \
        --eval-output eval_set.txt \
        [--split-train 10] \
        [--seed 42]

Class column options:
    - SCOPeClass: Major structural class (e.g., a, b, c, d)
    - SCOPeFold: Fold level classification
    - SCOPeSuperfamily: Superfamily level classification  
    - SCOPeFamily: Family level classification
"""

import argparse
import math
import pandas as pd
import random
from collections import defaultdict
from pathlib import Path


def parse_lst_file(lst_file: str) -> list[str]:
    """
    Parse a FASTA-like .lst file and extract structure identifiers.
    
    Lines starting with ">" contain identifiers in the format:
    >d3wsed_:D\t247
    
    We extract the part before the ":" (e.g., "d3wsed_").
    
    Args:
        lst_file: Path to the .lst file
        
    Returns:
        List of structure identifiers
    """
    identifiers = []
    
    with open(lst_file, 'r') as f:
        for line in f:
            line = line.strip()
            if line.startswith('>'):
                # Remove the '>' and split by ':' to get the identifier
                # Format: >d3wsed_:D\t247
                header = line[1:]  # Remove '>'
                # Split by ':' and take the first part
                identifier = header.split(':')[0]
                identifiers.append(identifier)
    
    return identifiers


# Valid SCOPe classification columns
VALID_CLASS_COLUMNS = ['SCOPeClass', 'SCOPeFold', 'SCOPeSuperfamily', 'SCOPeFamily']


def load_lookup_table(tsv_file: str, class_column: str) -> pd.DataFrame:
    """
    Load the TSV lookup table with SCOPe annotations.
    
    Args:
        tsv_file: Path to the TSV file with SCOPe annotations
        class_column: Column name for classification level
        
    Returns:
        DataFrame with lookup table
    """
    df = pd.read_csv(tsv_file, sep='\t')
    
    if class_column not in df.columns:
        raise ValueError(f"TSV file must contain '{class_column}' column. "
                        f"Found columns: {list(df.columns)}")
    
    return df


def create_balanced_dataset(
    identifiers: list[str],
    lookup_df: pd.DataFrame,
    n_per_class: int,
    id_column: str = 'SCOPeID',
    class_column: str = 'SCOPeClass',
    seed: int | None = None
) -> tuple[dict[str, list[str]], dict[str, int]]:
    """
    Create a balanced dataset with N proteins from each SCOPe classification.
    
    Args:
        identifiers: List of structure identifiers from the .lst file
        lookup_df: DataFrame with SCOPe annotations
        n_per_class: Number of proteins to sample per classification
        id_column: Column name for the identifier in the lookup table
        class_column: Column name for the classification level
        seed: Random seed for reproducibility
        
    Returns:
        Tuple of (dict mapping class to list of identifiers, dict of class counts)
    """
    if seed is not None:
        random.seed(seed)
    
    # Create a set for faster lookup
    id_set = set(identifiers)
    
    # Group identifiers by SCOPe classification
    class_to_ids = defaultdict(list)
    
    for _, row in lookup_df.iterrows():
        scope_id = row[id_column]
        scope_class = row[class_column]
        
        if scope_id in id_set:
            class_to_ids[scope_class].append(scope_id)
    
    # Sample N from each class
    sampled_by_class = {}
    class_counts = {}
    
    for scope_class in sorted(class_to_ids.keys()):
        available_ids = class_to_ids[scope_class]
        n_available = len(available_ids)
        
        if n_available >= n_per_class:
            sampled = random.sample(available_ids, n_per_class)
        else:
            # Take all available if less than N
            sampled = available_ids
            print(f"Warning: Class '{scope_class}' has only {n_available} "
                  f"proteins (requested {n_per_class})")
        
        sampled_by_class[scope_class] = sampled
        class_counts[scope_class] = len(sampled)
    
    return sampled_by_class, class_counts


def split_train_eval(
    sampled_by_class: dict[str, list[str]],
    eval_fraction: float,
    seed: int | None = None
) -> tuple[list[str], list[str], dict[str, tuple[int, int]]]:
    """
    Split the balanced dataset into training and evaluation sets.
    
    The split is performed per-class to maintain balance in both sets.
    
    Args:
        sampled_by_class: Dict mapping class to list of identifiers
        eval_fraction: Fraction of data to use for evaluation (0.0-1.0)
        seed: Random seed for reproducibility
        
    Returns:
        Tuple of (train_ids, eval_ids, dict of (train_count, eval_count) per class)
    """
    if seed is not None:
        random.seed(seed)
    
    train_ids = []
    eval_ids = []
    split_counts = {}
    
    for scope_class in sorted(sampled_by_class.keys()):
        ids = sampled_by_class[scope_class].copy()
        random.shuffle(ids)
        
        n_eval = max(1, int(len(ids) * eval_fraction))
        n_train = len(ids) - n_eval
        
        class_eval = ids[:n_eval]
        class_train = ids[n_eval:]
        
        eval_ids.extend(class_eval)
        train_ids.extend(class_train)
        split_counts[scope_class] = (len(class_train), len(class_eval))
    
    return train_ids, eval_ids, split_counts


def split_into_files(
    ids: list[str],
    output_path: Path,
    n_splits: int
) -> list[Path]:
    """
    Split a list of identifiers into N separate files.
    
    Args:
        ids: List of identifiers to split
        output_path: Base path for output files (e.g., train.txt -> train_part01.txt)
        n_splits: Number of files to split into
        
    Returns:
        List of paths to created files
    """
    lines_per_file = math.ceil(len(ids) / n_splits)
    created_files = []
    
    base_name = output_path.stem
    suffix = output_path.suffix
    parent = output_path.parent
    
    for i in range(n_splits):
        start = i * lines_per_file
        end = min(start + lines_per_file, len(ids))
        chunk = ids[start:end]
        
        if not chunk:  # Skip empty chunks
            break
        
        out_file = parent / f"{base_name}_part{i+1:02d}{suffix}"
        with open(out_file, 'w') as f:
            for identifier in chunk:
                f.write(f"{identifier}\n")
        created_files.append(out_file)
    
    return created_files


def main():
    parser = argparse.ArgumentParser(
        description='Create a balanced evaluation dataset from CD-HIT results',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    parser.add_argument(
        '--lst-file', '-l',
        required=True,
        help='Path to the CD-HIT .lst file (FASTA-like format)'
    )
    
    parser.add_argument(
        '--lookup-tsv', '-t',
        required=True,
        help='Path to the TSV lookup table with SCOPe annotations'
    )
    
    parser.add_argument(
        '--class-column', '-c',
        choices=VALID_CLASS_COLUMNS,
        default='SCOPeClass',
        help='Classification column to use for balancing. '
             'Options: SCOPeClass, SCOPeFold, SCOPeSuperfamily, SCOPeFamily '
             '(default: SCOPeClass)'
    )
    
    parser.add_argument(
        '--n-per-class', '-n',
        type=int,
        default=100,
        help='Number of proteins to sample per class (default: 100)'
    )
    
    parser.add_argument(
        '--eval-fraction', '-e',
        type=float,
        default=0.2,
        help='Fraction of data to use for evaluation (default: 0.2)'
    )
    
    parser.add_argument(
        '--train-output',
        required=True,
        help='Path to output file for training set (one identifier per line)'
    )
    
    parser.add_argument(
        '--eval-output',
        required=True,
        help='Path to output file for evaluation set (one identifier per line)'
    )
    
    parser.add_argument(
        '--split-train',
        type=int,
        default=None,
        help='Split training set into N separate files for parallel processing '
             '(e.g., --split-train 10 creates train_part01.txt through train_part10.txt)'
    )
    
    parser.add_argument(
        '--id-column', '-i',
        default='SCOPeID',
        help='Column name for identifier in lookup table (default: SCOPeID)'
    )
    
    parser.add_argument(
        '--seed', '-s',
        type=int,
        default=None,
        help='Random seed for reproducibility'
    )
    
    args = parser.parse_args()
    
    # Parse the .lst file
    print(f"Reading identifiers from: {args.lst_file}")
    identifiers = parse_lst_file(args.lst_file)
    print(f"Found {len(identifiers)} identifiers in .lst file")
    
    # Load lookup table
    print(f"Loading lookup table from: {args.lookup_tsv}")
    lookup_df = load_lookup_table(args.lookup_tsv, args.class_column)
    print(f"Lookup table has {len(lookup_df)} entries")
    
    # Create balanced dataset
    print(f"\nCreating balanced dataset with {args.n_per_class} proteins per {args.class_column}...")
    sampled_by_class, class_counts = create_balanced_dataset(
        identifiers=identifiers,
        lookup_df=lookup_df,
        n_per_class=args.n_per_class,
        id_column=args.id_column,
        class_column=args.class_column,
        seed=args.seed
    )
    
    # Split into train and eval sets
    print(f"\nSplitting into train ({1-args.eval_fraction:.0%}) and eval ({args.eval_fraction:.0%}) sets...")
    train_ids, eval_ids, split_counts = split_train_eval(
        sampled_by_class=sampled_by_class,
        eval_fraction=args.eval_fraction,
        seed=args.seed
    )
    
    # Print summary
    print(f"\nClass distribution (total / train / eval):")
    for scope_class in sorted(class_counts.keys()):
        total = class_counts[scope_class]
        train_n, eval_n = split_counts[scope_class]
        print(f"  {scope_class}: {total} / {train_n} / {eval_n}")
    
    print(f"\nTotal selected: {sum(class_counts.values())}")
    print(f"  Training set: {len(train_ids)}")
    print(f"  Evaluation set: {len(eval_ids)}")
    
    # Save training set
    train_path = Path(args.train_output)
    train_path.parent.mkdir(parents=True, exist_ok=True)
    
    if args.split_train and args.split_train > 1:
        # Split training set into multiple files
        created_files = split_into_files(train_ids, train_path, args.split_train)
        print(f"\nSplit training set into {len(created_files)} files:")
        for f in created_files:
            print(f"  {f}")
    else:
        # Save as single file
        with open(train_path, 'w') as f:
            for identifier in train_ids:
                f.write(f"{identifier}\n")
        print(f"\nSaved training set to: {args.train_output}")
    
    # Save evaluation set
    eval_path = Path(args.eval_output)
    eval_path.parent.mkdir(parents=True, exist_ok=True)
    with open(eval_path, 'w') as f:
        for identifier in eval_ids:
            f.write(f"{identifier}\n")
    print(f"Saved evaluation set to: {args.eval_output}")


if __name__ == '__main__':
    main()
