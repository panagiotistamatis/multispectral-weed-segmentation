"""
Calculate per-date mean/std statistics and class weights for the WeedsGalore dataset.

Usage:
    python -m wd.data.calculate_weedsgalore_stats --root path/to/weedsgalore-dataset

Output: prints a Python dict (copy-paste into weedsgalore_stats.py)
"""
import argparse
import json
import os

import numpy as np
from PIL import Image
from tqdm import tqdm

BANDS = ['R', 'G', 'B', 'NIR', 'RE']
NUM_CLASSES = 3  # 0=background, 1=crop, 2+=weed (collapsed to 2)


def read_split(splits_dir, split_name):
    """Read scene IDs from a split file (one ID per line)."""
    path = os.path.join(splits_dir, f"{split_name}.txt")
    with open(path, 'r') as f:
        return [line.strip() for line in f if line.strip()]


def scene_id_to_date(scene_id):
    """Extract date from scene_id, e.g. '2023-06-15_0472' -> '2023-06-15'."""
    return scene_id[:10]


def load_band(dataset_root, scene_id, band):
    """Load a single 16-bit PNG band and normalize to [0, 1] float32."""
    date = scene_id_to_date(scene_id)
    path = os.path.join(dataset_root, date, "images", f"{scene_id}_{band}.png")
    img = np.array(Image.open(path), dtype=np.float32) / 65535.0
    return img


def load_label(dataset_root, scene_id):
    """Load semantic label PNG."""
    date = scene_id_to_date(scene_id)
    path = os.path.join(dataset_root, date, "semantics", f"{scene_id}.png")
    label = np.array(Image.open(path), dtype=np.int64)
    # Collapse weed subclasses: anything > 2 becomes 2
    label[label > 2] = 2
    return label


def compute_ndvi(nir, red):
    """Compute NDVI = (NIR - R) / (NIR + R + eps)."""
    return (nir - red) / (nir + red + 1e-10)


def calculate_stats(dataset_root, scene_ids):
    """
    Calculate per-date, per-band statistics (mean, std, sum, sum_sq, count)
    and class pixel counts from given scene IDs.
    """
    # Group scenes by date
    date_scenes = {}
    for sid in scene_ids:
        d = scene_id_to_date(sid)
        date_scenes.setdefault(d, []).append(sid)

    all_bands = BANDS + ['NDVI']
    stats = {}

    for date, scenes in sorted(date_scenes.items()):
        print(f"\nProcessing date: {date} ({len(scenes)} scenes)")

        psum = {b: 0.0 for b in all_bands}
        psum_sq = {b: 0.0 for b in all_bands}
        count = 0

        for sid in tqdm(scenes, desc=date):
            # Load raw bands
            band_data = {}
            for band in BANDS:
                band_data[band] = load_band(dataset_root, sid, band)

            # Compute NDVI on-the-fly
            band_data['NDVI'] = compute_ndvi(band_data['NIR'], band_data['R'])

            # Accumulate stats
            pixels = band_data['R'].size  # H * W
            count += pixels

            for b in all_bands:
                psum[b] += band_data[b].sum()
                psum_sq[b] += (band_data[b] ** 2).sum()

        # Compute mean and std
        date_stats = {}
        for b in all_bands:
            mean = psum[b] / count
            std = np.sqrt((psum_sq[b] / count) - (mean ** 2))
            date_stats[b] = {
                'mean': float(mean),
                'std': float(std),
                'sum': float(psum[b]),
                'sum_sq': float(psum_sq[b]),
            }
        date_stats['count'] = int(count)
        stats[date] = date_stats

    return stats


def calculate_class_weights(dataset_root, scene_ids):
    """
    Count pixels per class and compute focal loss weights.
    Weights are inverse-frequency, normalized so that crop (class 1) = 1.0.
    """
    class_counts = np.zeros(NUM_CLASSES, dtype=np.int64)

    print("\nCounting class pixels...")
    for sid in tqdm(scene_ids, desc="class weights"):
        label = load_label(dataset_root, sid)
        for c in range(NUM_CLASSES):
            class_counts[c] += (label == c).sum()

    total = class_counts.sum()
    freq = class_counts / total

    # Normalize so crop (class 1) has weight 1.0
    weights = freq[1] / freq  # inverse frequency relative to crop

    print(f"\nClass counts: {dict(enumerate(class_counts.tolist()))}")
    print(f"Class frequencies: {dict(enumerate(freq.tolist()))}")
    print(f"Class weights (crop=1.0): {weights.tolist()}")

    return weights.tolist(), class_counts.tolist()


def main():
    parser = argparse.ArgumentParser(description="Calculate WeedsGalore stats")
    parser.add_argument("--root", type=str, required=True,
                        help="Path to weedsgalore-dataset root (containing date folders and splits/)")
    args = parser.parse_args()

    dataset_root = args.root
    splits_dir = os.path.join(dataset_root, "splits")

    # Use only training scenes for stats (same as WeedMap approach)
    train_ids = read_split(splits_dir, "train")
    print(f"Found {len(train_ids)} training scenes")

    # 1. Per-date band statistics
    stats = calculate_stats(dataset_root, train_ids)

    # 2. Class weights for focal loss
    weights, counts = calculate_class_weights(dataset_root, train_ids)

    # Print output for copy-paste into weedsgalore_stats.py
    print("\n" + "=" * 60)
    print("Copy the following into wd/data/weedsgalore_stats.py:")
    print("=" * 60)
    print(f"STATS = {json.dumps(stats, indent=4)}")
    print()
    print(f"CLASS_WEIGHTS = {weights}")
    print(f"CLASS_COUNTS = {counts}")


if __name__ == '__main__':
    main()
