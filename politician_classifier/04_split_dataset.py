"""
04_split_dataset.py
===================
Step 4: Split the cleaned raw dataset into train / val / test sets.

  - Reads from:  dataset/cleaned/<class>/
  - Writes to:   dataset/processed/train|val|test/<class>/
  - Ratios:      75% train | 15% val | 10% test  (configurable)
  - Stratified:  each class maintains the same split ratio
  - Images are COPIED (not moved) so cleaned/ stays intact

Usage:
  python 04_split_dataset.py
  python 04_split_dataset.py --train 0.8 --val 0.1 --test 0.1
  python 04_split_dataset.py --seed 123
"""

import argparse
import json
import os
import shutil
import sys
from pathlib import Path

from sklearn.model_selection import train_test_split
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent))
from config import (
    POLITICIANS,
    CLEANED_DIR,
    TRAIN_DIR,
    VAL_DIR,
    TEST_DIR,
    TRAIN_RATIO,
    VAL_RATIO,
    TEST_RATIO,
    STATS_DIR,
)
from utils import setup_logger, ensure_dirs, is_image_file

logger = setup_logger(__name__, "04_split.log")


# ──────────────────────────────────────────────────────────────────────────────
# PER-CLASS SPLIT
# ──────────────────────────────────────────────────────────────────────────────

def split_class(
    slug: str,
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
    seed: int,
) -> dict:
    """
    Copy images from raw/<slug> → processed/{train,val,test}/<slug>.

    Returns:
        Dict with counts for each split.
    """
    src_dir = os.path.join(CLEANED_DIR, slug)
    if not os.path.isdir(src_dir):
        logger.warning("[%s] Source directory not found (CLEANED_DIR) — skipping.", slug)
        return {}

    # Collect all valid images
    images = sorted(
        str(p) for p in Path(src_dir).iterdir()
        if p.is_file() and is_image_file(str(p))
    )

    if len(images) < 3:
        logger.warning("[%s] Too few images (%d) to split — skipping.", slug, len(images))
        return {}

    # Normalise ratios (guard against floating-point sum != 1)
    total = train_ratio + val_ratio + test_ratio
    train_r = train_ratio / total
    val_r   = val_ratio   / total

    # First split: train vs (val + test)
    train_imgs, temp_imgs = train_test_split(
        images,
        test_size=1.0 - train_r,
        random_state=seed,
        shuffle=True,
    )

    # Second split: val vs test (from the temp portion)
    val_fraction_of_temp = val_r / (1.0 - train_r)
    val_imgs, test_imgs = train_test_split(
        temp_imgs,
        test_size=1.0 - val_fraction_of_temp,
        random_state=seed,
        shuffle=True,
    )

    # Copy to destination folders
    splits = {
        "train": (train_imgs, TRAIN_DIR),
        "val":   (val_imgs,   VAL_DIR),
        "test":  (test_imgs,  TEST_DIR),
    }

    counts = {"class": slug}
    for split_name, (img_list, split_root) in splits.items():
        dst_dir = os.path.join(split_root, slug)
        ensure_dirs(dst_dir)
        for src_path in img_list:
            dst_path = os.path.join(dst_dir, Path(src_path).name)
            shutil.copy2(src_path, dst_path)
        counts[split_name] = len(img_list)
        logger.debug("[%s] %s: %d images", slug, split_name, len(img_list))

    counts["total"] = len(images)
    logger.info(
        "[%s] total=%d | train=%d | val=%d | test=%d",
        slug, len(images),
        counts["train"], counts["val"], counts["test"],
    )
    return counts


# ──────────────────────────────────────────────────────────────────────────────
# ENTRY POINT
# ──────────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Split processed images into train/val/test sets."
    )
    parser.add_argument("--train", type=float, default=TRAIN_RATIO,
                        help=f"Train ratio (default: {TRAIN_RATIO})")
    parser.add_argument("--val",   type=float, default=VAL_RATIO,
                        help=f"Val ratio (default: {VAL_RATIO})")
    parser.add_argument("--test",  type=float, default=TEST_RATIO,
                        help=f"Test ratio (default: {TEST_RATIO})")
    parser.add_argument("--seed",  type=int,   default=42,
                        help="Random seed for reproducibility (default: 42)")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ensure_dirs(TRAIN_DIR, VAL_DIR, TEST_DIR, STATS_DIR)

    # Validate ratios
    if not (0 < args.train < 1 and 0 < args.val < 1 and 0 < args.test < 1):
        logger.error("All ratios must be between 0 and 1 (exclusive).")
        sys.exit(1)

    targets = list(POLITICIANS.keys())

    logger.info("=" * 65)
    logger.info("  STEP 4 -- TRAIN / VAL / TEST SPLIT")
    logger.info("  Ratios: train=%.0f%% | val=%.0f%% | test=%.0f%% | seed=%d",
                args.train * 100, args.val * 100, args.test * 100, args.seed)
    logger.info("  Classes: %d", len(targets))
    logger.info("=" * 65)

    all_stats = []
    for slug in tqdm(targets, desc="Splitting classes", unit="class"):
        s = split_class(slug, args.train, args.val, args.test, args.seed)
        if s:
            all_stats.append(s)

    # ── Save stats ─────────────────────────────────────────────────────────
    stats_path = os.path.join(STATS_DIR, "04_split_stats.json")
    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump(all_stats, f, indent=2)

    # -- Summary table -----------------------------------------------------
    logger.info("\n" + "-" * 65)
    logger.info("  SPLIT SUMMARY")
    logger.info("  %-28s %7s %7s %7s %7s",
                "Class", "Total", "Train", "Val", "Test")
    logger.info("-" * 65)

    totals = {"total": 0, "train": 0, "val": 0, "test": 0}
    for s in all_stats:
        logger.info("  %-28s %7d %7d %7d %7d",
                    s["class"], s["total"], s["train"], s["val"], s["test"])
        for k in totals:
            totals[k] += s.get(k, 0)

    logger.info("-" * 65)
    logger.info("  %-28s %7d %7d %7d %7d",
                "TOTAL", totals["total"], totals["train"], totals["val"], totals["test"])
    logger.info("-" * 65)
    logger.info("  Stats saved -> %s", stats_path)
    logger.info("\n  Dataset ready at: dataset/processed/")


if __name__ == "__main__":
    main()
