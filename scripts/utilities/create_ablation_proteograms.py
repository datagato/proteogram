#!/usr/bin/env python3
"""
Create ablation proteogram images.

For each of the 6 information channels (3 RGB channels x 2 triangles),
creates a set of images where only that channel/triangle combination
retains its original pixel values. All other pixels are replaced with
the per-channel dataset mean pixel value.

Proteogram channel mapping (from proteogram/v2/proteogram.py):
  Upper-right triangle: R=VdW-attractive, G=VdW-repulsive,  B=Distance
  Lower-left  triangle: R=ES-attractive,  G=ES-repulsive,   B=Hydrophobicity

Usage:
    python create_ablation_proteogram.py <input_dir> <output_dir>

Output:
    <output_dir>/
        upper_R_vdw_attractive/   (one ablated image per input image)
        upper_G_vdw_repulsive/
        upper_B_distance/
        lower_R_es_attractive/
        lower_G_es_repulsive/
        lower_B_hydrophobicity/
"""

import argparse
import sys
from pathlib import Path

import numpy as np
from PIL import Image


# (triangle, channel_index, feature_name)
ABLATION_TARGETS = [
    ("upper", 0, "vdw_attractive"),
    ("upper", 1, "vdw_repulsive"),
    ("upper", 2, "distance"),
    ("lower", 0, "es_attractive"),
    ("lower", 1, "es_repulsive"),
    ("lower", 2, "hydrophobicity"),
]

CHANNEL_NAMES = {0: "R", 1: "G", 2: "B"}
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".tiff", ".tif", ".bmp"}


def get_image_paths(input_dir: Path) -> list[Path]:
    paths = sorted([
        p for p in input_dir.iterdir()
        if p.suffix.lower() in IMAGE_EXTENSIONS
    ])
    if not paths:
        sys.exit(f"No image files found in {input_dir}")
    return paths


def compute_dataset_channel_means(image_paths: list[Path]) -> np.ndarray:
    """Compute mean pixel value per channel across all images in the dataset."""
    channel_sums = np.zeros(3, dtype=np.float64)
    channel_counts = np.zeros(3, dtype=np.int64)

    for path in image_paths:
        img = np.array(Image.open(path).convert("RGB"), dtype=np.float64)
        for c in range(3):
            channel_sums[c] += img[:, :, c].sum()
            channel_counts[c] += img[:, :, c].size

    return channel_sums / channel_counts


def make_triangle_masks(size: int) -> tuple[np.ndarray, np.ndarray]:
    """Return upper-right and lower-left boolean masks for a square image."""
    rows, cols = np.ogrid[:size, :size]
    upper_mask = cols > rows   # strict upper-right (diagonal excluded)
    lower_mask = cols < rows   # strict lower-left  (diagonal excluded)
    return upper_mask, lower_mask


def create_ablated_image(
    img: np.ndarray,
    triangle_mask: np.ndarray,
    channel_idx: int,
    channel_means: np.ndarray,
) -> np.ndarray:
    """
    Replace every pixel with its per-channel dataset mean, then restore
    the selected (triangle, channel) pixels from the original image.
    """
    ablated = np.empty_like(img, dtype=np.float64)
    for c in range(3):
        ablated[:, :, c] = channel_means[c]

    ablated[triangle_mask, channel_idx] = img[triangle_mask, channel_idx]

    return ablated.round().clip(0, 255).astype(np.uint8)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Produce 6 ablated proteogram image sets, one per information channel."
    )
    parser.add_argument("input_dir", help="Directory containing input proteogram images")
    parser.add_argument("output_dir", help="Root directory for ablated output images")
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)

    if not input_dir.is_dir():
        sys.exit(f"Input directory not found: {input_dir}")

    image_paths = get_image_paths(input_dir)
    print(f"Found {len(image_paths)} images in {input_dir}")

    print("Computing per-channel dataset means...")
    channel_means = compute_dataset_channel_means(image_paths)
    for c, name in CHANNEL_NAMES.items():
        print(f"  Channel {name}: mean = {channel_means[c]:.4f}")

    # Pre-create output subdirectories
    subdirs: dict[tuple[str, int], Path] = {}
    for triangle, channel_idx, feature_name in ABLATION_TARGETS:
        ch_name = CHANNEL_NAMES[channel_idx]
        subdir = output_dir / f"{triangle}_{ch_name}_{feature_name}"
        subdir.mkdir(parents=True, exist_ok=True)
        subdirs[(triangle, channel_idx)] = subdir

    skipped = 0
    for path in image_paths:
        img = np.array(Image.open(path).convert("RGB"), dtype=np.float64)
        h, w = img.shape[:2]
        if h != w:
            print(f"  Skipping {path.name}: not square ({h}x{w})")
            skipped += 1
            continue

        upper_mask, lower_mask = make_triangle_masks(h)
        masks = {"upper": upper_mask, "lower": lower_mask}

        for triangle, channel_idx, feature_name in ABLATION_TARGETS:
            ablated = create_ablated_image(img, masks[triangle], channel_idx, channel_means)
            out_path = subdirs[(triangle, channel_idx)] / path.name
            Image.fromarray(ablated, mode="RGB").save(out_path)

        print(f"  Processed: {path.name}")

    n_processed = len(image_paths) - skipped
    print(f"\nDone. Processed {n_processed} images ({skipped} skipped).")
    print(f"Output written to: {output_dir}")
    print("Subdirectories created:")
    for triangle, channel_idx, feature_name in ABLATION_TARGETS:
        ch_name = CHANNEL_NAMES[channel_idx]
        print(f"  {triangle}_{ch_name}_{feature_name}/")


if __name__ == "__main__":
    main()
