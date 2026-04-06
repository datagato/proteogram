"""
Copy a subset of protein structure files from one directory to another,
selected by a list of file prefixes (one per line in a text file).

Each prefix is matched against filenames in the source directory: any file
whose basename starts with the prefix is copied. The match is prefix-only
(not a full filename) so it works regardless of extension (.ent, .pdb, etc.).

Usage example:
    python copy_structures_by_prefix.py \
        --prefix_file data/my_prefixes.txt \
        --src_dir data/all_structures \
        --dst_dir data/subset_structures
"""
import argparse
import glob
import os
import shutil
from tqdm import tqdm


def load_prefixes(prefix_file):
    with open(prefix_file) as f:
        return [line.strip() for line in f if line.strip()]


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Copy structure files matching a prefix list from src to dst.")
    parser.add_argument('--prefix_file', '-p',
                        required=True,
                        help="Text file with one prefix per line.")
    parser.add_argument('--src_dir', '-s',
                        required=True,
                        help="Source directory containing structure files.")
    parser.add_argument('--dst_dir', '-d',
                        required=True,
                        help="Destination directory (created if absent).")
    parser.add_argument('--overwrite', '-o',
                        action='store_true',
                        help="Overwrite destination directory if it exists.")
    args = parser.parse_args()

    prefixes = load_prefixes(args.prefix_file)
    print(f'Loaded {len(prefixes)} prefixes from {args.prefix_file}')

    if os.path.exists(args.dst_dir):
        if args.overwrite:
            shutil.rmtree(args.dst_dir)
            os.makedirs(args.dst_dir)
        else:
            print(f'Destination {args.dst_dir} already exists; files will be added/overwritten.')
    else:
        os.makedirs(args.dst_dir)

    all_files = [f for f in glob.glob(os.path.join(args.src_dir, '**', '*'), recursive=True)
                 if os.path.isfile(f)]
    prefix_set = set(prefixes)

    copied = 0
    not_found = []
    for prefix in tqdm(prefixes):
        matches = [f for f in all_files if os.path.basename(f).startswith(prefix)]
        if not matches:
            not_found.append(prefix)
            continue
        for src_file in matches:
            shutil.copy(src_file, os.path.join(args.dst_dir, os.path.basename(src_file)))
            copied += 1

    print(f'Copied {copied} file(s) to {args.dst_dir}')
    if not_found:
        print(f'WARNING: {len(not_found)} prefix(es) had no matching files: {not_found[:10]}')
