"""
03_intelligent_cleaner.py
=========================
Step 3: Intelligent face detection and noise reduction.
- Detects all faces.
- Removes images with too many people (group photos).
- Removes images where the face is too small.
- Crops only the largest/dominant face.
- Saves to dataset/cleaned/
"""

import argparse
import json
import os
import sys
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent))
from config import (
    POLITICIANS, RAW_DIR, CLEANED_DIR, TARGET_SIZE, FACE_MARGIN,
    MIN_FACE_CONFIDENCE, MAX_FACES_ALLOWED, MIN_FACE_SIZE_RATIO, STATS_DIR
)
from utils import setup_logger, ensure_dirs, is_image_file

logger = setup_logger(__name__, "03_intelligent_cleaner.log")

def _load_detector():
    """Prefer MTCNN for better bounding box accuracy."""
    try:
        from mtcnn import MTCNN
        return "mtcnn", MTCNN()
    except ImportError:
        logger.warning("MTCNN not found, falling back to Haar cascades.")
        cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        return "haar", cv2.CascadeClassifier(cascade_path)

def crop_and_resize(img, box, margin=FACE_MARGIN):
    h_img, w_img = img.shape[:2]
    x, y, w, h = box
    pad_x, pad_y = int(w * margin), int(h * margin)
    
    x1, y1 = max(0, x - pad_x), max(0, y - pad_y)
    x2, y2 = min(w_img, x + w + pad_x), min(h_img, y + h + pad_y)
    
    face = img[y1:y2, x1:x2]
    if face.size == 0: return None
    return cv2.resize(face, TARGET_SIZE, interpolation=cv2.INTER_LANCZOS4)

def process_class(slug, backend, detector):
    src_dir = os.path.join(RAW_DIR, slug)
    dst_dir = os.path.join(CLEANED_DIR, slug)
    ensure_dirs(dst_dir)
    
    images = sorted([str(p) for p in Path(src_dir).iterdir() if is_image_file(str(p))])
    stats = {"class": slug, "initial": len(images), "no_face": 0, "too_many_faces": 0, "too_small": 0, "processed": 0}
    
    for path in tqdm(images, desc=f"  Intelligent Cleaning: {slug}", unit="img", leave=False):
        img = cv2.imread(path)
        if img is None: continue
        
        h_img, w_img = img.shape[:2]
        
        # Detection
        if backend == "mtcnn":
            results = detector.detect_faces(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            faces = [r["box"] for r in results if r["confidence"] >= MIN_FACE_CONFIDENCE]
        else:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            faces = detector.detectMultiScale(gray, 1.1, 5)
            
        # 1. Filter: No face detected
        if not faces:
            stats["no_face"] += 1
            continue
            
        # 2. Filter: Group photo noise (> MAX_FACES_ALLOWED)
        if len(faces) > MAX_FACES_ALLOWED:
            stats["too_many_faces"] += 1
            continue
            
        # 3. Filter: Face too small (Ratio check)
        best_face = max(faces, key=lambda f: f[2] * f[3])
        if (best_face[2] / w_img < MIN_FACE_SIZE_RATIO) or (best_face[3] / h_img < MIN_FACE_SIZE_RATIO):
            stats["too_small"] += 1
            continue
            
        # Success: Crop and save
        cropped = crop_and_resize(img, best_face)
        if cropped is not None:
            out_name = f"{slug}_{stats['processed']:04d}.jpg"
            cv2.imwrite(os.path.join(dst_dir, out_name), cropped, [cv2.IMWRITE_JPEG_QUALITY, 95])
            stats["processed"] += 1
            
    logger.info("[%s] %d -> %d (Noise removed: %d)", slug, stats["initial"], stats["processed"], stats["initial"] - stats["processed"])
    return stats

def main():
    ensure_dirs(CLEANED_DIR, STATS_DIR)
    backend, detector = _load_detector()
    
    all_stats = []
    for slug in tqdm(POLITICIANS.keys(), desc="Processing classes"):
        all_stats.append(process_class(slug, backend, detector))
        
    with open(os.path.join(STATS_DIR, "03_intelligent_cleaning_stats.json"), "w") as f:
        json.dump(all_stats, f, indent=2)

if __name__ == "__main__":
    main()
