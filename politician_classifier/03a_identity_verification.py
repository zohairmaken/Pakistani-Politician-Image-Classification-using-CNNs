"""
03a_identity_verification.py
============================
Step 3a: Verify facial identity using DeepFace embeddings.
- Requires 5-10 reference images per politician in the references/ folder.
- Compares every image in dataset/cleaned/ against the references.
- Discards images that don't match the target identity.
"""

import argparse
import os
import sys
import shutil
from pathlib import Path
from deepface import DeepFace
from tqdm import tqdm
import json

sys.path.insert(0, str(Path(__file__).parent))
from config import (
    POLITICIANS, CLEANED_DIR, REFERENCES_DIR, PROCESSED_DIR,
    IDENTITY_THRESHOLD, EMBEDDING_MODEL, STATS_DIR
)
from utils import setup_logger, ensure_dirs, is_image_file

logger = setup_logger(__name__, "03a_identity_verification.log")

def verify_class(slug):
    src_dir = os.path.join(CLEANED_DIR, slug)
    ref_dir = os.path.join(REFERENCES_DIR, slug)
    
    if not os.path.isdir(ref_dir) or not any(Path(ref_dir).iterdir()):
        logger.warning("[%s] No references found. Skipping — images kept as-is.", slug)
        return None

    ref_images = [str(p) for p in Path(ref_dir).iterdir() if is_image_file(str(p))]
    target_images = [str(p) for p in Path(src_dir).iterdir() if is_image_file(str(p))]
    
    # Safety: if only 1 reference, use lenient threshold to avoid mass false-rejection
    num_refs = len(ref_images)
    threshold = IDENTITY_THRESHOLD if num_refs >= 3 else min(IDENTITY_THRESHOLD + 0.15, 0.60)
    if num_refs < 3:
        logger.warning("[%s] Only %d reference(s) found — using lenient threshold %.2f", slug, num_refs, threshold)

    # Create a safe 'rejected' subfolder instead of permanently deleting
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
                    model_name=EMBEDDING_MODEL,
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
                # Move to _rejected/ instead of permanent delete
                shutil.move(img_path, os.path.join(rejected_dir, os.path.basename(img_path)))
                stats["rejected"] += 1
        except Exception as e:
            logger.error("Error verifying %s: %s", img_path, e)
            shutil.move(img_path, os.path.join(rejected_dir, os.path.basename(img_path)))
            stats["rejected"] += 1
            
    logger.info("[%s] Verified: %d | Rejected: %d (moved to _rejected/)", slug, stats["verified"], stats["rejected"])
    return stats

def main():
    if not os.path.exists(REFERENCES_DIR):
        logger.error("REFERENCES_DIR (%s) does not exist. Please create it and add clean faces for each class.", REFERENCES_DIR)
        return

    all_stats = []
    for slug in tqdm(POLITICIANS.keys(), desc="Verifying Identities"):
        s = verify_class(slug)
        if s: all_stats.append(s)
        
    with open(os.path.join(STATS_DIR, "03a_identity_stats.json"), "w") as f:
        json.dump(all_stats, f, indent=2)

if __name__ == "__main__":
    main()
