"""
02_clean_dataset.py
===================
Step 2: Clean the raw dataset by removing:
  - Corrupted / unreadable images
  - Duplicate images (perceptual hashing)
  - Blurry / low-quality images (Laplacian variance)
  - Images that are too small (likely icons / placeholders)

Usage:
  python 02_clean_dataset.py
  python 02_clean_dataset.py --class imran_khan
  python 02_clean_dataset.py --blur-threshold 120
"""

import argparse
import json
import os
import sys
from pathlib import Path

import cv2
import imagehash
import numpy as np
from PIL import Image, UnidentifiedImageError
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent))
from config import (
    POLITICIANS,
    RAW_DIR,
    BLUR_THRESHOLD,
    HASH_THRESHOLD,
    MIN_IMAGE_BYTES,
    STATS_DIR,
)
from utils import setup_logger, ensure_dirs, is_image_file

logger = setup_logger(__name__, "02_clean.log")


# ──────────────────────────────────────────────────────────────────────────────
# PER-IMAGE CHECKS
# ──────────────────────────────────────────────────────────────────────────────

def is_corrupted(path: str) -> bool:
    """Return True if the image cannot be opened by Pillow."""
    try:
        with Image.open(path) as img:
            img.verify()            # full decode check
        return False
    except (UnidentifiedImageError, Exception):
        return True


def is_too_small(path: str, min_bytes: int = MIN_IMAGE_BYTES) -> bool:
    """Return True if the file is suspiciously small (likely an icon)."""
    return os.path.getsize(path) < min_bytes


def compute_blur_score(path: str) -> float:
    """
    Compute the Laplacian variance of a grayscale image.
    Lower = blurrier.
    """
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return 0.0
    return float(cv2.Laplacian(img, cv2.CV_64F).var())


def compute_phash(path: str) -> imagehash.ImageHash | None:
    """Compute perceptual hash for duplicate detection."""
    try:
        with Image.open(path) as img:
            return imagehash.phash(img)
    except Exception:
        return None


# ──────────────────────────────────────────────────────────────────────────────
# CLASS-LEVEL CLEANING
# ──────────────────────────────────────────────────────────────────────────────

def clean_class(slug: str, blur_threshold: float = BLUR_THRESHOLD) -> dict:
    """
    Clean all images for one politician class.

    Args:
        slug:           Politician slug key.
        blur_threshold: Laplacian variance threshold.

    Returns:
        Statistics dict.
    """
    class_dir = os.path.join(RAW_DIR, slug)
    if not os.path.isdir(class_dir):
        logger.warning("[%s] Raw directory not found — skipping.", slug)
        return {}

    images = [
        str(p) for p in Path(class_dir).iterdir()
        if p.is_file() and is_image_file(str(p))
    ]

    stats = {
        "class": slug,
        "initial": len(images),
        "removed_corrupted": 0,
        "removed_small": 0,
        "removed_blurry": 0,
        "removed_duplicate": 0,
        "final": 0,
    }

    seen_hashes: list[imagehash.ImageHash] = []

    for path in tqdm(images, desc=f"  Cleaning {slug}", unit="img", leave=False):
        removed = False

        # ── 1. File size check ────────────────────────────────────────────
        if is_too_small(path):
            os.remove(path)
            stats["removed_small"] += 1
            logger.debug("Removed (too small): %s", path)
            continue

        # ── 2. Corruption check ───────────────────────────────────────────
        if is_corrupted(path):
            os.remove(path)
            stats["removed_corrupted"] += 1
            logger.debug("Removed (corrupted): %s", path)
            continue

        # ── 3. Duplicate check ────────────────────────────────────────────
        phash = compute_phash(path)
        if phash is not None:
            if any(abs(phash - h) <= HASH_THRESHOLD for h in seen_hashes):
                os.remove(path)
                stats["removed_duplicate"] += 1
                logger.debug("Removed (duplicate): %s", path)
                removed = True
            else:
                seen_hashes.append(phash)

        if removed:
            continue

        # ── 4. Blur check ─────────────────────────────────────────────────
        blur = compute_blur_score(path)
        if blur < blur_threshold:
            os.remove(path)
            stats["removed_blurry"] += 1
            logger.debug("Removed (blurry, score=%.1f): %s", blur, path)
            continue

    # Final count
    remaining = [
        p for p in Path(class_dir).iterdir()
        if p.is_file() and is_image_file(str(p))
    ]
    stats["final"] = len(remaining)
    total_removed = (
        stats["removed_corrupted"]
        + stats["removed_small"]
        + stats["removed_blurry"]
        + stats["removed_duplicate"]
    )
    logger.info(
        "[%s] %d → %d (removed %d: corrupt=%d, small=%d, blur=%d, dup=%d)",
        slug,
        stats["initial"],
        stats["final"],
        total_removed,
        stats["removed_corrupted"],
        stats["removed_small"],
        stats["removed_blurry"],
        stats["removed_duplicate"],
    )
    return stats


# ──────────────────────────────────────────────────────────────────────────────
# ENTRY POINT
# ──────────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Clean raw images: remove corrupted, duplicate, blurry files."
    )
    parser.add_argument(
        "--class", dest="cls", default=None,
        help="Clean a single class by slug. Default: all.",
    )
    parser.add_argument(
        "--blur-threshold", type=float, default=BLUR_THRESHOLD,
        help=f"Laplacian blur threshold (default: {BLUR_THRESHOLD}).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ensure_dirs(STATS_DIR)

    targets = [args.cls] if args.cls else list(POLITICIANS.keys())

    logger.info("=" * 60)
    logger.info("  STEP 2 -- DATASET CLEANING")
    logger.info("  Classes: %d | Blur threshold: %.1f", len(targets), args.blur_threshold)
    logger.info("=" * 60)

    all_stats = []
    for slug in tqdm(targets, desc="Cleaning classes", unit="class"):
        s = clean_class(slug, blur_threshold=args.blur_threshold)
        if s:
            all_stats.append(s)

    # ── Save stats JSON ────────────────────────────────────────────────────
    stats_path = os.path.join(STATS_DIR, "02_cleaning_stats.json")
    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump(all_stats, f, indent=2)

    # -- Summary table -----------------------------------------------------
    logger.info("\n" + "-" * 70)
    logger.info("  CLEANING SUMMARY")
    logger.info("  %-25s %7s %7s %7s %7s %7s",
                "Class", "Before", "After", "Corr.", "Blur", "Dup.")
    logger.info("-" * 70)
    for s in all_stats:
        logger.info(
            "  %-25s %7d %7d %7d %7d %7d",
            s["class"],
            s["initial"],
            s["final"],
            s["removed_corrupted"] + s["removed_small"],
            s["removed_blurry"],
            s["removed_duplicate"],
        )
    logger.info("-" * 70)
    logger.info("  Stats saved → %s", stats_path)


if __name__ == "__main__":
    main()
