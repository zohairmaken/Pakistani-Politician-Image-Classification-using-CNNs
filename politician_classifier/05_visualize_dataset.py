"""
05_visualize_dataset.py
=======================
Step 5 (Bonus): Generate rich visualisations of the final dataset.

Outputs (saved to stats/):
  1. class_distribution.png   — bar chart of images per class per split
  2. sample_grid_<class>.png  — 5x5 grid of sample face images per class
  3. dataset_summary.json     — complete statistics
  4. dataset_report.txt       — human-readable statistics report

Usage:
  python 05_visualize_dataset.py
  python 05_visualize_dataset.py --grid-size 4
  python 05_visualize_dataset.py --no-grids
"""

import argparse
import json
import math
import os
import random
import sys
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent))
from config import POLITICIANS, TRAIN_DIR, VAL_DIR, TEST_DIR, STATS_DIR
from utils import setup_logger, ensure_dirs, is_image_file

logger = setup_logger(__name__, "05_visualize.log")

SPLIT_COLORS = {"train": "#4C72B0", "val": "#55A868", "test": "#C44E52"}


def collect_stats(splits):
    split_dirs = {"train": TRAIN_DIR, "val": VAL_DIR, "test": TEST_DIR}
    stats = {}
    for slug in POLITICIANS:
        stats[slug] = {}
        for split in splits:
            d = os.path.join(split_dirs[split], slug)
            count = sum(1 for p in Path(d).iterdir()
                        if p.is_file() and is_image_file(str(p))) if os.path.isdir(d) else 0
            stats[slug][split] = count
    return stats


def plot_class_distribution(stats, splits, out_dir):
    classes = list(stats.keys())
    display = [POLITICIANS[c]["display_name"] for c in classes]
    n_s = len(splits)
    x = np.arange(len(classes))
    width = 0.7 / n_s

    fig, ax = plt.subplots(figsize=(max(14, len(classes) * 0.9), 7))
    fig.patch.set_facecolor("#0f1117")
    ax.set_facecolor("#0f1117")

    for i, split in enumerate(splits):
        counts = [stats[c].get(split, 0) for c in classes]
        bars = ax.bar(x + (i - n_s / 2 + 0.5) * width, counts, width,
                      label=split.capitalize(), color=SPLIT_COLORS[split],
                      edgecolor="#0f1117", alpha=0.92)
        for bar in bars:
            h = bar.get_height()
            if h > 0:
                ax.text(bar.get_x() + bar.get_width() / 2, h + 0.5,
                        str(int(h)), ha="center", va="bottom",
                        fontsize=7, color="white", fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels(display, rotation=40, ha="right", fontsize=9, color="white")
    ax.set_ylabel("Image Count", color="white", fontsize=11)
    ax.set_title("Pakistani Politicians Dataset — Class Distribution",
                 color="white", fontsize=14, fontweight="bold", pad=15)
    ax.legend(fontsize=10, facecolor="#1c1e26", edgecolor="none", labelcolor="white")
    ax.tick_params(colors="white")
    ax.spines[:].set_color("#333344")
    ax.yaxis.grid(True, color="#333344", linestyle="--", alpha=0.5)
    ax.set_axisbelow(True)

    plt.tight_layout()
    out_path = os.path.join(out_dir, "class_distribution.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
    logger.info("Saved class distribution chart -> %s", out_path)
    return out_path


def plot_sample_grid(slug, split, grid_n, out_dir):
    split_dirs = {"train": TRAIN_DIR, "val": VAL_DIR, "test": TEST_DIR}
    class_dir = os.path.join(split_dirs[split], slug)
    if not os.path.isdir(class_dir):
        return None
    all_imgs = [str(p) for p in Path(class_dir).iterdir()
                if p.is_file() and is_image_file(str(p))]
    if not all_imgs:
        return None

    random.seed(42)
    samples = random.sample(all_imgs, min(grid_n * grid_n, len(all_imgs)))
    cols = min(grid_n, len(samples))
    rows = math.ceil(len(samples) / cols)

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2, rows * 2))
    fig.patch.set_facecolor("#0f1117")

    if rows == 1 and cols == 1:
        axes = np.array([[axes]])
    elif rows == 1 or cols == 1:
        axes = np.array(axes).reshape(rows, cols)

    fig.suptitle(f"{POLITICIANS[slug]['display_name']}  |  {split.capitalize()} samples",
                 color="white", fontsize=12, fontweight="bold", y=1.01)

    for idx, ax in enumerate(axes.flat):
        ax.set_axis_off()
        if idx < len(samples):
            img = cv2.imread(samples[idx])
            if img is not None:
                ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

    plt.tight_layout(pad=0.3)
    out_path = os.path.join(out_dir, f"sample_grid_{slug}.png")
    plt.savefig(out_path, dpi=120, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
    return out_path


def save_summary(stats, out_dir):
    totals = {slug: sum(v.values()) for slug, v in stats.items()}
    summary = {
        "per_class": {
            slug: {
                "display_name": POLITICIANS[slug]["display_name"],
                "splits": split_counts,
                "total": totals[slug],
            }
            for slug, split_counts in stats.items()
        },
        "grand_totals": {
            "train": sum(s.get("train", 0) for s in stats.values()),
            "val":   sum(s.get("val",   0) for s in stats.values()),
            "test":  sum(s.get("test",  0) for s in stats.values()),
            "total": sum(totals.values()),
        },
    }
    json_path = os.path.join(out_dir, "dataset_summary.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    txt_path = os.path.join(out_dir, "dataset_report.txt")
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write("=" * 70 + "\n")
        f.write("  PAKISTANI POLITICIAN IMAGE CLASSIFICATION - DATASET REPORT\n")
        f.write("=" * 70 + "\n\n")
        f.write(f"{'Class':<30} {'Train':>7} {'Val':>7} {'Test':>7} {'Total':>7}\n")
        f.write("-" * 60 + "\n")
        for slug, data in summary["per_class"].items():
            sp = data["splits"]
            f.write(f"{data['display_name']:<30} "
                    f"{sp.get('train',0):>7} {sp.get('val',0):>7} "
                    f"{sp.get('test',0):>7} {data['total']:>7}\n")
        f.write("-" * 60 + "\n")
        gt = summary["grand_totals"]
        f.write(f"{'TOTAL':<30} {gt['train']:>7} {gt['val']:>7} "
                f"{gt['test']:>7} {gt['total']:>7}\n")
        f.write("=" * 70 + "\n")
    logger.info("Summary saved -> %s | %s", json_path, txt_path)


def parse_args():
    parser = argparse.ArgumentParser(description="Generate dataset visualisations.")
    parser.add_argument("--splits", nargs="+", default=["train", "val", "test"],
                        choices=["train", "val", "test"])
    parser.add_argument("--grid-size", type=int, default=5)
    parser.add_argument("--no-grids", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    ensure_dirs(STATS_DIR)

    logger.info("=" * 60)
    logger.info("  STEP 5 - DATASET VISUALISATION")
    logger.info("=" * 60)

    stats = collect_stats(args.splits)
    plot_class_distribution(stats, args.splits, STATS_DIR)

    if not args.no_grids:
        for slug in tqdm(POLITICIANS.keys(), desc="Sample grids", unit="class"):
            plot_sample_grid(slug, "train", args.grid_size, STATS_DIR)

    save_summary(stats, STATS_DIR)

    # Console summary
    print("\n" + "=" * 62)
    print("  DATASET STATISTICS")
    print("=" * 62)
    grand = {"train": 0, "val": 0, "test": 0}
    for slug, splits in stats.items():
        t, v, te = splits.get("train",0), splits.get("val",0), splits.get("test",0)
        grand["train"] += t; grand["val"] += v; grand["test"] += te
        print(f"  {POLITICIANS[slug]['display_name']:<28} "
              f"train={t:3d} val={v:3d} test={te:3d} total={t+v+te:3d}")
    print("-" * 62)
    print(f"  {'TOTAL':<28} train={grand['train']:3d} val={grand['val']:3d} "
          f"test={grand['test']:3d} total={sum(grand.values()):3d}")
    print("=" * 62)
    print(f"\n  All outputs saved to: {STATS_DIR}")


if __name__ == "__main__":
    main()
