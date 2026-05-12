"""
00_pad_to_minimum.py
====================
Pads any class in dataset/cleaned/ that has fewer than MIN_COUNT images
by generating augmented variants until the threshold is met.

Augmentations used:
  - Horizontal flip
  - Rotation ±15°
  - Brightness shift ±30%
  - Contrast jitter
  - Zoom crop (90-100%)
  - Salt & pepper noise
"""

import os, sys, random, cv2, numpy as np, time
from pathlib import Path
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent))
from config import CLEANED_DIR, POLITICIANS
from utils import setup_logger

logger = setup_logger(__name__, "00_pad.log")
MIN_COUNT = 80
random.seed(42)
np.random.seed(42)

# ─────────────────────────────────────────────────────────────────────────────
# AUGMENTATION FUNCTIONS
# ─────────────────────────────────────────────────────────────────────────────

def flip(img):
    return cv2.flip(img, 1)

def rotate(img, angle=None):
    a = angle if angle else random.uniform(-15, 15)
    h, w = img.shape[:2]
    M = cv2.getRotationMatrix2D((w//2, h//2), a, 1.0)
    return cv2.warpAffine(img, M, (w, h), borderMode=cv2.BORDER_REFLECT)

def brightness(img):
    factor = random.uniform(0.7, 1.3)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[:,:,2] = np.clip(hsv[:,:,2] * factor, 0, 255)
    return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)

def contrast(img):
    alpha = random.uniform(0.8, 1.2)
    return np.clip(img.astype(np.float32) * alpha, 0, 255).astype(np.uint8)

def zoom_crop(img):
    h, w = img.shape[:2]
    scale = random.uniform(0.88, 0.98)
    ch, cw = int(h*scale), int(w*scale)
    y1 = random.randint(0, h-ch)
    x1 = random.randint(0, w-cw)
    crop = img[y1:y1+ch, x1:x1+cw]
    return cv2.resize(crop, (w, h))

def noise(img):
    out = img.copy().astype(np.float32)
    amount = random.uniform(0.01, 0.03)
    num = int(amount * out.size)
    coords = [np.random.randint(0, i-1, num) for i in img.shape[:2]]
    out[coords[0], coords[1]] = 255
    coords = [np.random.randint(0, i-1, num) for i in img.shape[:2]]
    out[coords[0], coords[1]] = 0
    return np.clip(out, 0, 255).astype(np.uint8)

# Augmentation pipeline — cycle through for variety
AUG_FUNCS = [flip, rotate, brightness, contrast, zoom_crop, noise,
             lambda i: brightness(flip(i)),
             lambda i: rotate(contrast(i)),
             lambda i: zoom_crop(brightness(i)),
             lambda i: noise(rotate(i))]

def augment(img, idx):
    """Apply a deterministic augmentation based on index for reproducibility."""
    fn = AUG_FUNCS[idx % len(AUG_FUNCS)]
    return fn(img)

# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def pad_class(slug):
    class_dir = Path(CLEANED_DIR) / slug
    if not class_dir.exists():
        logger.warning("[%s] Cleaned dir not found. Skipping.", slug)
        return

    # Count only real images (not inside _rejected/)
    images = [f for f in class_dir.iterdir()
              if f.is_file() and f.suffix.lower() in {'.jpg','.jpeg','.png'}]
    current = len(images)

    if current >= MIN_COUNT:
        logger.info("[%s] Already has %d images (≥%d). No padding needed.", slug, current, MIN_COUNT)
        return

    needed = MIN_COUNT - current
    logger.info("[%s] Has %d images. Generating %d augmented variants...", slug, current, needed)

    generated = 0
    aug_idx   = 0
    source_images = images.copy()

    with tqdm(total=needed, desc=f"  Padding {slug.replace('_',' ').title()}", unit="img") as pbar:
        while generated < needed:
            src_path = source_images[generated % len(source_images)]
            img = cv2.imread(str(src_path))
            if img is None:
                aug_idx += 1
                continue

            aug = augment(img, aug_idx)
            out_name = f"aug_{generated:04d}_{int(time.time()*1000)}_{src_path.stem}.jpg"
            out_path = class_dir / out_name
            cv2.imwrite(str(out_path), aug, [cv2.IMWRITE_JPEG_QUALITY, 92])
            generated += 1
            aug_idx += 1
            pbar.update(1)

    total_after = len(list(class_dir.glob("*.jpg"))) + len(list(class_dir.glob("*.jpeg"))) + len(list(class_dir.glob("*.png")))
    logger.info("[%s] ✅ Padded: %d real + %d augmented = %d total", slug, current, generated, total_after)

def main():
    logger.info("=" * 60)
    logger.info("  AUGMENTATION PADDING — target: %d images per class", MIN_COUNT)
    logger.info("=" * 60)

    report = []
    for slug in tqdm(POLITICIANS.keys(), desc="Processing classes"):
        class_dir = Path(CLEANED_DIR) / slug
        before = len([f for f in class_dir.iterdir()
                      if f.is_file() and f.suffix.lower() in {'.jpg','.jpeg','.png'}]) if class_dir.exists() else 0
        pad_class(slug)
        after = len([f for f in class_dir.iterdir()
                     if f.is_file() and f.suffix.lower() in {'.jpg','.jpeg','.png'}]) if class_dir.exists() else 0
        report.append({"class": slug, "before": before, "after": after, "padded": after - before})

    logger.info("\n%s", "=" * 60)
    logger.info("  PADDING SUMMARY")
    logger.info("=" * 60)
    for r in report:
        flag = "✅" if r["after"] >= MIN_COUNT else "❌"
        logger.info("  %s %-30s %3d → %3d  (+%d)", flag, r["class"], r["before"], r["after"], r["padded"])

if __name__ == "__main__":
    main()
