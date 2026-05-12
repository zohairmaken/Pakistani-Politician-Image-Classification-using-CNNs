import argparse
import os
import sys
import time
import logging
from pathlib import Path
from multiprocessing import Pool, cpu_count

from icrawler.builtin import GoogleImageCrawler, BingImageCrawler
from tqdm import tqdm

from src.utils.config_loader import load_config
from src.utils.helpers import setup_logger, ensure_dirs

def _count_existing(folder: str) -> int:
    """Count already-downloaded images in a folder."""
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    folder_path = Path(folder)
    if not folder_path.exists():
        return 0
    return sum(1 for f in folder_path.iterdir() if f.suffix.lower() in exts)

def download_with_retry(crawler_cls, keyword: str, save_dir: str,
                        num_images: int, logger: logging.Logger, retries: int = 3) -> int:
    """
    Attempt to download images with retries.
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
                min_size=(80, 80),
                file_idx_offset="auto",
            )
            break
        except Exception as exc:
            logger.warning(
                "Attempt %d/%d failed for '%s': %s", attempt, retries, keyword, exc
            )
            time.sleep(2 ** attempt)

    after = _count_existing(save_dir)
    return max(0, after - before)

class Downloader:
    def __init__(self, config: dict):
        self.config = config
        self.logger = setup_logger("downloader", config, "downloader.log")
        self.raw_dir = config['paths']['raw_dir']
        self.politicians = config['politicians']
        self.settings = config['download']
        
        ensure_dirs(self.raw_dir)

    def download_politician(self, slug: str) -> dict:
        info = self.politicians[slug]
        name = info["display_name"]
        keywords = info["keywords"]
        save_dir = os.path.join(self.raw_dir, slug)
        ensure_dirs(save_dir)

        existing = _count_existing(save_dir)
        max_images = self.settings['max_images_per_class']
        
        if existing >= max_images:
            self.logger.info("[%s] Already has %d images — skipping.", name, existing)
            return {"class": slug, "downloaded": 0, "total": existing, "skipped": True}

        needed = max_images - existing
        per_kw = min(self.settings['images_per_keyword'], needed // len(keywords) + 1)
        total_dl = 0

        self.logger.info("[%s] Starting download — need %d more images.", name, needed)

        for kw in tqdm(keywords, desc=f"  {name}", unit="keyword", leave=False):
            if _count_existing(save_dir) >= max_images:
                break

            for crawler_cls in (BingImageCrawler, GoogleImageCrawler):
                n = download_with_retry(crawler_cls, kw, save_dir, per_kw, self.logger)
                total_dl += n
                if n > 0:
                    break

        total = _count_existing(save_dir)
        return {"class": slug, "downloaded": total_dl, "total": total, "skipped": False}

    def run(self, target_class=None, workers=None):
        if workers is None:
            workers = min(self.settings['num_workers'], cpu_count())
            
        targets = [target_class] if target_class else list(self.politicians.keys())

        self.logger.info("=" * 60)
        self.logger.info("  IMAGE DOWNLOAD MODULE")
        self.logger.info("  Classes : %d | Workers : %d", len(targets), workers)
        self.logger.info("=" * 60)

        if workers > 1 and len(targets) > 1:
            with Pool(processes=workers) as pool:
                results = list(tqdm(pool.imap(self.download_politician, targets), total=len(targets)))
        else:
            results = [self.download_politician(slug) for slug in targets]

        return results

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--class", dest="cls", default=None)
    parser.add_argument("--workers", type=int, default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    downloader = Downloader(cfg)
    downloader.run(args.cls, args.workers)
