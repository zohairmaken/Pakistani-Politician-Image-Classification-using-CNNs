import os
import json
import math
import random
import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from tqdm import tqdm
from src.utils.helpers import setup_logger, ensure_dirs, is_image_file

class Visualizer:
    def __init__(self, config: dict):
        self.config = config
        self.logger = setup_logger("visualizer", config, "visualizer.log")
        self.paths = config['paths']
        self.politicians = config['politicians']
        self.colors = {"train": "#4C72B0", "val": "#55A868", "test": "#C44E52"}
        
        ensure_dirs(self.paths['reports_dir'])

    def collect_stats(self):
        splits = ["train", "val", "test"]
        split_dirs = {s: self.paths[f"{s}_dir"] for s in splits}
        stats = {}
        for slug in self.politicians:
            stats[slug] = {}
            for split in splits:
                d = os.path.join(split_dirs[split], slug)
                count = sum(1 for p in Path(d).iterdir() if p.is_file() and is_image_file(str(p))) if os.path.isdir(d) else 0
                stats[slug][split] = count
        return stats

    def plot_distribution(self, stats):
        classes = list(stats.keys())
        display = [self.politicians[c]["display_name"] for c in classes]
        splits = ["train", "val", "test"]
        x = np.arange(len(classes))
        width = 0.7 / len(splits)

        fig, ax = plt.subplots(figsize=(14, 7))
        fig.patch.set_facecolor("#0f1117")
        ax.set_facecolor("#0f1117")

        for i, split in enumerate(splits):
            counts = [stats[c].get(split, 0) for c in classes]
            ax.bar(x + (i - len(splits)/2 + 0.5) * width, counts, width, label=split.capitalize(), color=self.colors[split])

        ax.set_xticks(x)
        ax.set_xticklabels(display, rotation=40, ha="right", color="white")
        ax.set_ylabel("Image Count", color="white")
        ax.set_title("Dataset Distribution", color="white", fontsize=14)
        ax.legend()
        ax.tick_params(colors="white")
        
        out_path = os.path.join(self.paths['reports_dir'], "class_distribution.png")
        plt.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close()
        return out_path

    def run(self):
        stats = self.collect_stats()
        self.plot_distribution(stats)
        self.logger.info("Visualization complete.")

if __name__ == "__main__":
    from src.utils.config_loader import load_config
    cfg = load_config()
    viz = Visualizer(cfg)
    viz.run()
