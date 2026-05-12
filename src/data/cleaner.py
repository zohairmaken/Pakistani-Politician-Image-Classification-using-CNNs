import os
import json
import logging
import cv2
import imagehash
import numpy as np
from pathlib import Path
from PIL import Image, UnidentifiedImageError
from tqdm import tqdm
from src.utils.helpers import setup_logger, ensure_dirs, is_image_file

class DataCleaner:
    def __init__(self, config: dict):
        self.config = config
        self.logger = setup_logger("cleaner", config, "cleaner.log")
        self.raw_dir = config['paths']['raw_dir']
        self.cleaned_dir = config['paths']['cleaned_dir']
        self.stats_dir = config['paths']['reports_dir']
        self.settings = config['processing']
        self.politicians = config['politicians']
        
        ensure_dirs(self.cleaned_dir, self.stats_dir)
        self.backend, self.detector = self._load_detector()

    def _load_detector(self):
        try:
            from mtcnn import MTCNN
            return "mtcnn", MTCNN()
        except ImportError:
            self.logger.warning("MTCNN not found, falling back to Haar cascades.")
            cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
            return "haar", cv2.CascadeClassifier(cascade_path)

    def is_corrupted(self, path: str) -> bool:
        try:
            with Image.open(path) as img:
                img.verify()
            return False
        except:
            return True

    def compute_blur_score(self, path: str) -> float:
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if img is None: return 0.0
        return float(cv2.Laplacian(img, cv2.CV_64F).var())

    def crop_and_resize(self, img, box):
        h_img, w_img = img.shape[:2]
        x, y, w, h = box
        margin = self.settings['face_margin']
        pad_x, pad_y = int(w * margin), int(h * margin)
        
        x1, y1 = max(0, x - pad_x), max(0, y - pad_y)
        x2, y2 = min(w_img, x + w + pad_x), min(h_img, y + h + pad_y)
        
        face = img[y1:y2, x1:x2]
        if face.size == 0: return None
        target_size = tuple(self.settings['target_size'])
        return cv2.resize(face, target_size, interpolation=cv2.INTER_LANCZOS4)

    def clean_basic(self, slug: str):
        """Phase 1: Basic file cleaning (corruption, duplicates, blur)."""
        class_dir = os.path.join(self.raw_dir, slug)
        images = [str(p) for p in Path(class_dir).iterdir() if p.is_file() and is_image_file(str(p))]
        
        seen_hashes = []
        stats = {"removed_small": 0, "removed_corrupted": 0, "removed_duplicate": 0, "removed_blurry": 0}
        
        for path in tqdm(images, desc=f"  Basic Cleaning: {slug}", unit="img", leave=False):
            # Size check
            if os.path.getsize(path) < self.settings['min_image_bytes']:
                os.remove(path)
                stats["removed_small"] += 1
                continue
            
            # Corruption check
            if self.is_corrupted(path):
                os.remove(path)
                stats["removed_corrupted"] += 1
                continue
            
            # Duplicate check
            try:
                with Image.open(path) as img:
                    phash = imagehash.phash(img)
                if any(abs(phash - h) <= self.settings['hash_threshold'] for h in seen_hashes):
                    os.remove(path)
                    stats["removed_duplicate"] += 1
                    continue
                seen_hashes.append(phash)
            except: continue

            # Blur check
            if self.compute_blur_score(path) < self.settings['blur_threshold']:
                os.remove(path)
                stats["removed_blurry"] += 1
        
        return stats

    def clean_intelligent(self, slug: str):
        """Phase 2: Intelligent face detection and cropping."""
        src_dir = os.path.join(self.raw_dir, slug)
        dst_dir = os.path.join(self.cleaned_dir, slug)
        ensure_dirs(dst_dir)
        
        images = sorted([str(p) for p in Path(src_dir).iterdir() if is_image_file(str(p))])
        stats = {"no_face": 0, "too_many_faces": 0, "too_small": 0, "processed": 0}
        
        for path in tqdm(images, desc=f"  Face Extraction: {slug}", unit="img", leave=False):
            img = cv2.imread(path)
            if img is None: continue
            h_img, w_img = img.shape[:2]
            
            if self.backend == "mtcnn":
                results = self.detector.detect_faces(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
                faces = [r["box"] for r in results if r["confidence"] >= self.settings['min_face_confidence']]
            else:
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                faces = self.detector.detectMultiScale(gray, 1.1, 5)
            
            if not faces:
                stats["no_face"] += 1; continue
            if len(faces) > self.settings['max_faces_allowed']:
                stats["too_many_faces"] += 1; continue
            
            best_face = max(faces, key=lambda f: f[2] * f[3])
            if (best_face[2] / w_img < self.settings['min_face_size_ratio']):
                stats["too_small"] += 1; continue
                
            cropped = self.crop_and_resize(img, best_face)
            if cropped is not None:
                out_name = f"{slug}_{stats['processed']:04d}.jpg"
                cv2.imwrite(os.path.join(dst_dir, out_name), cropped, [cv2.IMWRITE_JPEG_QUALITY, 95])
                stats["processed"] += 1
        
        return stats

    def run(self, slug=None):
        targets = [slug] if slug else list(self.politicians.keys())
        all_stats = []
        
        self.logger.info("=" * 60)
        self.logger.info("  DATA CLEANING & FACE EXTRACTION")
        self.logger.info("=" * 60)

        for s in tqdm(targets, desc="Overall Progress"):
            basic = self.clean_basic(s)
            intel = self.clean_intelligent(s)
            combined = {"class": s, **basic, **intel}
            all_stats.append(combined)
            
        stats_path = os.path.join(self.stats_dir, "cleaning_report.json")
        with open(stats_path, "w") as f:
            json.dump(all_stats, f, indent=2)
            
        self.logger.info(f"Cleaning complete. Report saved to {stats_path}")
        return all_stats

if __name__ == "__main__":
    from src.utils.config_loader import load_config
    cfg = load_config()
    cleaner = DataCleaner(cfg)
    cleaner.run()
