import os
import shutil
import json
import logging
from pathlib import Path
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from src.utils.helpers import setup_logger, ensure_dirs, is_image_file

class DatasetSplitter:
    def __init__(self, config: dict):
        self.config = config
        self.logger = setup_logger("splitter", config, "splitter.log")
        self.cleaned_dir = config['paths']['cleaned_dir']
        self.train_dir = config['paths']['train_dir']
        self.val_dir = config['paths']['val_dir']
        self.test_dir = config['paths']['test_dir']
        self.stats_dir = config['paths']['reports_dir']
        self.politicians = config['politicians']
        self.settings = config['training']
        
        ensure_dirs(self.train_dir, self.val_dir, self.test_dir, self.stats_dir)

    def split_class(self, slug: str, seed: int = None):
        src_dir = os.path.join(self.cleaned_dir, slug)
        if not os.path.isdir(src_dir):
            self.logger.warning("[%s] Source directory not found — skipping.", slug)
            return None

        images = sorted(str(p) for p in Path(src_dir).iterdir() if p.is_file() and is_image_file(str(p)))
        if len(images) < 3:
            self.logger.warning("[%s] Too few images (%d) to split.", slug, len(images))
            return None

        seed = seed or self.settings['seed']
        train_r = self.settings['train_ratio']
        val_r = self.settings['val_ratio']
        
        # Train vs (Val + Test)
        train_imgs, temp_imgs = train_test_split(images, test_size=1.0 - train_r, random_state=seed, shuffle=True)
        
        # Val vs Test
        val_fraction_of_temp = val_r / (1.0 - train_r)
        val_imgs, test_imgs = train_test_split(temp_imgs, test_size=1.0 - val_fraction_of_temp, random_state=seed, shuffle=True)

        splits = {"train": (train_imgs, self.train_dir), "val": (val_imgs, self.val_dir), "test": (test_imgs, self.test_dir)}
        counts = {"class": slug, "total": len(images)}
        
        for name, (img_list, root) in splits.items():
            dst = os.path.join(root, slug)
            ensure_dirs(dst)
            for src in img_list:
                shutil.copy2(src, os.path.join(dst, Path(src).name))
            counts[name] = len(img_list)
            
        return counts

    def run(self, seed=None):
        self.logger.info("=" * 60)
        self.logger.info("  DATASET SPLIT MODULE")
        self.logger.info("=" * 60)
        
        targets = list(self.politicians.keys())
        all_stats = []
        for s in tqdm(targets, desc="Overall Progress"):
            res = self.split_class(s, seed)
            if res: all_stats.append(res)
            
        stats_path = os.path.join(self.stats_dir, "split_report.json")
        with open(stats_path, "w") as f:
            json.dump(all_stats, f, indent=2)
            
        return all_stats

if __name__ == "__main__":
    from src.utils.config_loader import load_config
    cfg = load_config()
    splitter = DatasetSplitter(cfg)
    splitter.run()
