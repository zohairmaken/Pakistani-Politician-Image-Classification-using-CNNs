import os
import shutil
import json
import logging
from pathlib import Path
from deepface import DeepFace
from tqdm import tqdm
from src.utils.helpers import setup_logger, ensure_dirs, is_image_file

class IdentityVerifier:
    def __init__(self, config: dict):
        self.config = config
        self.logger = setup_logger("verifier", config, "verifier.log")
        self.cleaned_dir = config['paths']['cleaned_dir']
        self.ref_dir = config['paths']['references_dir']
        self.stats_dir = config['paths']['reports_dir']
        self.politicians = config['politicians']
        self.settings = config['processing']
        
        ensure_dirs(self.stats_dir)

    def verify_class(self, slug: str):
        src_dir = os.path.join(self.cleaned_dir, slug)
        ref_dir = os.path.join(self.ref_dir, slug)
        
        if not os.path.isdir(ref_dir) or not any(Path(ref_dir).iterdir()):
            self.logger.warning("[%s] No references found. Skipping verification.", slug)
            return None

        ref_images = [str(p) for p in Path(ref_dir).iterdir() if is_image_file(str(p))]
        target_images = [str(p) for p in Path(src_dir).iterdir() if is_image_file(str(p))]
        
        num_refs = len(ref_images)
        threshold = self.settings['identity_threshold'] if num_refs >= 3 else min(self.settings['identity_threshold'] + 0.15, 0.60)
        
        rejected_dir = os.path.join(src_dir, "_rejected")
        os.makedirs(rejected_dir, exist_ok=True)

        stats = {"class": slug, "initial": len(target_images), "verified": 0, "rejected": 0}
        
        for img_path in tqdm(target_images, desc=f"  Verifying: {slug}", unit="img", leave=False):
            is_match = False
            try:
                for ref_path in ref_images:
                    result = DeepFace.verify(
                        img1_path=img_path,
                        img2_path=ref_path,
                        model_name=self.settings['embedding_model'],
                        enforce_detection=False,
                        detector_backend='skip',
                        distance_metric='cosine',
                        silent=True
                    )
                    if result["distance"] <= threshold:
                        is_match = True
                        break
                        
                if is_match:
                    stats["verified"] += 1
                else:
                    shutil.move(img_path, os.path.join(rejected_dir, os.path.basename(img_path)))
                    stats["rejected"] += 1
            except Exception as e:
                self.logger.error("Error verifying %s: %s", img_path, e)
                shutil.move(img_path, os.path.join(rejected_dir, os.path.basename(img_path)))
                stats["rejected"] += 1
                
        return stats

    def run(self, slug=None):
        targets = [slug] if slug else list(self.politicians.keys())
        all_stats = []
        
        self.logger.info("=" * 60)
        self.logger.info("  IDENTITY VERIFICATION MODULE")
        self.logger.info("=" * 60)

        for s in tqdm(targets, desc="Overall Progress"):
            res = self.verify_class(s)
            if res: all_stats.append(res)
            
        stats_path = os.path.join(self.stats_dir, "verification_report.json")
        with open(stats_path, "w") as f:
            json.dump(all_stats, f, indent=2)
            
        return all_stats

if __name__ == "__main__":
    from src.utils.config_loader import load_config
    cfg = load_config()
    verifier = IdentityVerifier(cfg)
    verifier.run()
