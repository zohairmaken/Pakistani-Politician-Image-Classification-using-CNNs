import os
import random
import cv2
import numpy as np
import time
from pathlib import Path
from tqdm import tqdm
from src.utils.helpers import setup_logger, ensure_dirs, is_image_file

class DatasetPadder:
    def __init__(self, config: dict, min_count: int = 80):
        self.config = config
        self.logger = setup_logger("padder", config, "padding.log")
        self.cleaned_dir = config['paths']['cleaned_dir']
        self.politicians = config['politicians']
        self.min_count = min_count
        
        random.seed(config['training']['seed'])
        np.random.seed(config['training']['seed'])

    def augment(self, img, idx):
        # Deterministic cycle of augmentations
        funcs = [
            lambda i: cv2.flip(i, 1),
            lambda i: self.rotate(i),
            lambda i: self.brightness(i),
            lambda i: self.contrast(i),
            lambda i: self.zoom_crop(i),
            lambda i: self.noise(i)
        ]
        return funcs[idx % len(funcs)](img)

    def rotate(self, img):
        h, w = img.shape[:2]
        M = cv2.getRotationMatrix2D((w//2, h//2), random.uniform(-15, 15), 1.0)
        return cv2.warpAffine(img, M, (w, h), borderMode=cv2.BORDER_REFLECT)

    def brightness(self, img):
        factor = random.uniform(0.7, 1.3)
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.float32)
        hsv[:,:,2] = np.clip(hsv[:,:,2] * factor, 0, 255)
        return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)

    def contrast(self, img):
        alpha = random.uniform(0.8, 1.2)
        return np.clip(img.astype(np.float32) * alpha, 0, 255).astype(np.uint8)

    def zoom_crop(self, img):
        h, w = img.shape[:2]
        scale = random.uniform(0.88, 0.98)
        ch, cw = int(h*scale), int(w*scale)
        y1, x1 = random.randint(0, h-ch), random.randint(0, w-cw)
        return cv2.resize(img[y1:y1+ch, x1:x1+cw], (w, h))

    def noise(self, img):
        out = img.copy().astype(np.float32)
        num = int(random.uniform(0.01, 0.03) * out.size)
        for _ in range(2):
            coords = [np.random.randint(0, i-1, num) for i in img.shape[:2]]
            out[coords[0], coords[1]] = 255 if _ == 0 else 0
        return np.clip(out, 0, 255).astype(np.uint8)

    def pad_class(self, slug: str):
        class_dir = Path(self.cleaned_dir) / slug
        if not class_dir.exists(): return
        
        images = [f for f in class_dir.iterdir() if f.is_file() and is_image_file(str(f))]
        current = len(images)
        if current >= self.min_count: return
        
        needed = self.min_count - current
        self.logger.info(f"[{slug}] Padding {needed} images...")
        
        generated = 0
        while generated < needed:
            src_path = images[generated % len(images)]
            img = cv2.imread(str(src_path))
            if img is None: continue
            
            aug = self.augment(img, generated)
            out_name = f"aug_{generated:04d}_{src_path.stem}.jpg"
            cv2.imwrite(str(class_dir / out_name), aug, [cv2.IMWRITE_JPEG_QUALITY, 92])
            generated += 1

    def run(self):
        for slug in tqdm(self.politicians.keys(), desc="Padding Classes"):
            self.pad_class(slug)

if __name__ == "__main__":
    from src.utils.config_loader import load_config
    cfg = load_config()
    padder = DatasetPadder(cfg)
    padder.run()
