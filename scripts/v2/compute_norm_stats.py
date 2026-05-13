"""Compute global percentile normalisation statistics from a corpus of energy matrices.

Run this ONCE after generating a representative set of proteograms with
``--save_npy_matrices`` (a flag added to ``create_v2_proteograms.py``).

The script samples up to ``--max_samples`` per-channel ``.npy`` files, computes
the 1st and 99th percentile bounds across the full sample, and writes a
``norm_stats.json`` file that ``create_v2_proteograms.py --global_norm`` reads.

Usage
-----
    python compute_norm_stats.py \\
        --npy_dir /path/to/energy_matrices \\
        --out_file /path/to/norm_stats.json

Advanced usage
--------------
    python compute_norm_stats.py \\
        --npy_dir /path/to/energy_matrices \\
        --out_file /path/to/norm_stats.json \\
        --low_pct 0.5 \\
        --high_pct 99.5 \\
        --max_samples 10000
"""

import argparse
import glob
import os
import sys

import numpy as np
from tqdm import tqdm

# Allow running as a script without installing the package
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from proteogram.v2.normalisation import CHANNEL_NAMES, save_norm_stats


def _collect_channel_values(npy_dir: str,
                             channel: str,
                             max_samples: int,
                             seed: int = 42) -> np.ndarray:
    """Return a flat array of all non-zero upper-triangle values for one channel.

    Args:
        npy_dir:     Directory containing ``<protein_id>_<channel>.npy`` files.
        channel:     Channel name, e.g. ``"vdw_attractive"``.
        max_samples: Maximum number of matrices to load.
        seed:        Random seed for sub-sampling.

    Returns:
        1-D float64 array of sampled values, or empty array if no files found.
    """
    pattern = os.path.join(npy_dir, f'*_{channel}.npy')
    files = sorted(glob.glob(pattern))

    if not files:
        print(f'  WARNING: no files matching "{pattern}"', flush=True)
        return np.array([], dtype=np.float64)

    if len(files) > max_samples:
        rng = np.random.default_rng(seed=seed)
        files = list(rng.choice(files, size=max_samples, replace=False))

    all_vals = []
    for fpath in tqdm(files, desc=f'  {channel}', leave=False):
        arr = np.load(fpath)
        # Only include non-zero values (lower triangle is structurally zero)
        vals = arr[arr != 0].ravel().astype(np.float64)
        if vals.size:
            all_vals.append(vals)

    return np.concatenate(all_vals) if all_vals else np.array([], dtype=np.float64)


def compute_stats(npy_dir: str,
                  low_pct: float,
                  high_pct: float,
                  max_samples: int) -> dict:
    """Compute per-channel percentile bounds across a sampled corpus.

    Args:
        npy_dir:     Directory of per-channel ``.npy`` energy matrices.
        low_pct:     Lower percentile (default 1.0).
        high_pct:    Upper percentile (default 99.0).
        max_samples: Max matrices to sample per channel.

    Returns:
        Dict mapping channel name → ``{"p_low": float, "p_high": float}``,
        plus a ``"_meta"`` key with run information.
    """
    stats: dict = {}

    for channel in CHANNEL_NAMES:
        print(f'[{channel}] collecting values ...', flush=True)
        vals = _collect_channel_values(npy_dir, channel, max_samples)

        if vals.size == 0:
            print(f'  → no data found, using fallback [0.0, 255.0]')
            stats[channel] = {'p_low': 0.0, 'p_high': 255.0}
            continue

        p_low  = float(np.percentile(vals, low_pct))
        p_high = float(np.percentile(vals, high_pct))
        stats[channel] = {'p_low': p_low, 'p_high': p_high}
        print(
            f'  → p{low_pct}={p_low:.4f}  p{high_pct}={p_high:.4f}'
            f'  (N={vals.size:,})',
            flush=True,
        )

    stats['_meta'] = {
        'low_pct':   low_pct,
        'high_pct':  high_pct,
        'max_samples_per_channel': max_samples,
        'npy_dir':   os.path.abspath(npy_dir),
    }
    return stats


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            'Compute global percentile normalisation bounds from a corpus of '
            'per-channel .npy energy matrices and write norm_stats.json.'
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        '--npy_dir', '-d', required=True,
        help=(
            'Directory containing per-channel .npy files named '
            '<protein_id>_<channel>.npy  (produced by '
            'create_v2_proteograms.py --save_npy_matrices).'
        ),
    )
    parser.add_argument(
        '--out_file', '-o', required=True,
        help='Output path for norm_stats.json.',
    )
    parser.add_argument(
        '--low_pct', type=float, default=1.0,
        help='Lower percentile bound.',
    )
    parser.add_argument(
        '--high_pct', type=float, default=99.0,
        help='Upper percentile bound.',
    )
    parser.add_argument(
        '--max_samples', type=int, default=5000,
        help=(
            'Maximum number of energy matrices to sample per channel.  '
            'More = more accurate statistics, slower to run.'
        ),
    )
    args = parser.parse_args()

    if not os.path.isdir(args.npy_dir):
        parser.error(f'--npy_dir does not exist: {args.npy_dir}')
    if not (0 < args.low_pct < args.high_pct < 100):
        parser.error('Percentile bounds must satisfy 0 < low_pct < high_pct < 100')

    print(
        f'Computing norm stats\n'
        f'  npy_dir     = {args.npy_dir}\n'
        f'  percentiles = p{args.low_pct} / p{args.high_pct}\n'
        f'  max_samples = {args.max_samples} per channel\n',
        flush=True,
    )

    stats = compute_stats(
        npy_dir=args.npy_dir,
        low_pct=args.low_pct,
        high_pct=args.high_pct,
        max_samples=args.max_samples,
    )
    save_norm_stats(stats, args.out_file)
    print('\nDone.', flush=True)


if __name__ == '__main__':
    main()
