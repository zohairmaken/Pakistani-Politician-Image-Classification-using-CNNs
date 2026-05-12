"""
01_download_images.py
=====================
Step 1: Download raw images for each politician using icrawler.

Features:
  - Multiple search keywords per class for diversity
  - Google + Bing crawlers for redundancy
  - Automatic retry on failures
  - Multiprocessing support (one worker per politician)
  - Progress bars via tqdm
  - Skips already-downloaded images

Usage:
  python 01_download_images.py                   # download all classes
  python 01_download_images.py --class imran_khan  # single class
  python 01_download_images.py --workers 4       # parallel workers
"""

import argparse
import os
import sys
import time
import logging
from pathlib import Path
from multiprocessing import Pool, cpu_count

from icrawler.builtin import GoogleImageCrawler, BingImageCrawler
from tqdm import tqdm

# ── local imports ──────────────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).parent))
from config import (
    POLITICIANS,
    RAW_DIR,
    IMAGES_PER_KEYWORD,
    MAX_IMAGES_PER_CLASS,
    DOWNLOAD_TIMEOUT,
)
from utils import setup_logger, ensure_dirs

logger = setup_logger(__name__, "01_download.log")


# ──────────────────────────────────────────────────────────────────────────────
# HELPERS
# ──────────────────────────────────────────────────────────────────────────────

def _silent_logger() -> logging.Logger:
    """Return a logger that suppresses icrawler's verbose output."""
    log = logging.getLogger("icrawler")
    log.setLevel(logging.ERROR)
    return log


def _count_existing(folder: str) -> int:
    """Count already-downloaded images in a folder."""
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    return sum(1 for f in Path(folder).iterdir() if f.suffix.lower() in exts)


def download_with_retry(crawler_cls, keyword: str, save_dir: str,
                        num_images: int, retries: int = 3) -> int:
    """
    Attempt to download `num_images` for `keyword` using `crawler_cls`.
    Retries up to `retries` times on failure.

    Returns:
        Number of images successfully downloaded in this call.
    """
    before = _count_existing(save_dir)
    for attempt in range(1, retries + 1):
        try:
            crawler = crawler_cls(
                feeder_threads=1,
                parser_threads=1,
                downloader_threads=4,
                storage={"root_dir": save_dir},
                log_level=logging.ERROR,
            )
            crawler.crawl(
                keyword=keyword,
                max_num=num_images,
                min_size=(80, 80),        # skip tiny images
                file_idx_offset="auto",   # never overwrite existing files
            )
            break
        except Exception as exc:
            logger.warning(
                "Attempt %d/%d failed for '%s': %s", attempt, retries, keyword, exc
            )
            time.sleep(2 ** attempt)      # exponential back-off

    after = _count_existing(save_dir)
    return max(0, after - before)


# ──────────────────────────────────────────────────────────────────────────────
# CORE DOWNLOAD FUNCTION  (one politician at a time)
# ──────────────────────────────────────────────────────────────────────────────

def download_politician(slug: str) -> dict:
    """
    Download images for a single politician.

    Args:
        slug: Key in the POLITICIANS dict (e.g. 'imran_khan').

    Returns:
        Summary dict with counts and status.
    """
    info       = POLITICIANS[slug]
    name       = info["display_name"]
    keywords   = info["keywords"]
    save_dir   = os.path.join(RAW_DIR, slug)
    ensure_dirs(save_dir)

    existing = _count_existing(save_dir)
    if existing >= MAX_IMAGES_PER_CLASS:
        logger.info("[%s] Already has %d images — skipping download.", name, existing)
        return {"class": slug, "downloaded": 0, "total": existing, "skipped": True}

    needed    = MAX_IMAGES_PER_CLASS - existing
    per_kw    = min(IMAGES_PER_KEYWORD, needed // len(keywords) + 1)
    total_dl  = 0

    logger.info("[%s] Starting download — need %d more images (%d keywords).",
                name, needed, len(keywords))

    for kw in tqdm(keywords, desc=f"  {name}", unit="keyword", leave=False):
        if _count_existing(save_dir) >= MAX_IMAGES_PER_CLASS:
            break

        # Try Bing first (more stable), then Google as fallback
        for crawler_cls in (BingImageCrawler, GoogleImageCrawler):
            n = download_with_retry(crawler_cls, kw, save_dir, per_kw)
            total_dl += n
            if n > 0:
                break      # move to next keyword if Google worked

    total = _count_existing(save_dir)
    logger.info("[%s] Done — downloaded %d new images | total on disk: %d",
                name, total_dl, total)

    return {"class": slug, "downloaded": total_dl, "total": total, "skipped": False}


# ──────────────────────────────────────────────────────────────────────────────
# ENTRY POINT
# ──────────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download raw images for Pakistani politician classification."
    )
    parser.add_argument(
        "--class", dest="cls", default=None,
        help="Download a single class by slug (e.g. imran_khan). Default: all.",
    )
    parser.add_argument(
        "--workers", type=int, default=min(4, cpu_count()),
        help="Number of parallel download workers. Default: min(4, CPU count).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ensure_dirs(RAW_DIR)

    # Determine which classes to process
    if args.cls:
        if args.cls not in POLITICIANS:
            logger.error("Unknown class '%s'. Valid options: %s",
                         args.cls, list(POLITICIANS.keys()))
            sys.exit(1)
        targets = [args.cls]
    else:
        targets = list(POLITICIANS.keys())

    logger.info("=" * 60)
    logger.info("  STEP 1 -- IMAGE DOWNLOAD")
    logger.info("  Classes : %d | Workers : %d", len(targets), args.workers)
    logger.info("=" * 60)

    # ── parallel download ──────────────────────────────────────────────────
    if args.workers > 1 and len(targets) > 1:
        with Pool(processes=args.workers) as pool:
            results = list(
                tqdm(
                    pool.imap(download_politician, targets),
                    total=len(targets),
                    desc="Overall progress",
                    unit="class",
                )
            )
    else:
        results = []
        for slug in tqdm(targets, desc="Overall progress", unit="class"):
            results.append(download_politician(slug))

    # -- summary -----------------------------------------------------------
    logger.info("\n" + "-" * 60)
    logger.info("  DOWNLOAD SUMMARY")
    logger.info("-" * 60)
    total_images = 0
    for r in results:
        status = "SKIPPED" if r["skipped"] else f"+{r['downloaded']:3d} new"
        logger.info("  %-30s | %s | total: %d", r["class"], status, r["total"])
        total_images += r["total"]
    logger.info("-" * 60)
    logger.info("  Grand total images on disk: %d", total_images)
    logger.info("-" * 60)


if __name__ == "__main__":
    main()
