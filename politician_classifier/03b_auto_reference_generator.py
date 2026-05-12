"""
03b_auto_reference_generator.py
===============================
Bootstrap the reference dataset by finding the most common face in the raw data.
This solves the "search engine problem" where noise (logos, other people) is mixed in.
"""

import os
import sys
import shutil
import numpy as np
from pathlib import Path
from deepface import DeepFace
from tqdm import tqdm
import json

sys.path.insert(0, str(Path(__file__).parent))
from config import POLITICIANS, RAW_DIR, REFERENCES_DIR, EMBEDDING_MODEL, IDENTITY_THRESHOLD
from utils import setup_logger, ensure_dirs, is_image_file

logger = setup_logger(__name__, "03b_auto_ref.log")

def find_centroid_references(slug, num_seeds=20, num_refs=5):
    src_dir = os.path.join(RAW_DIR, slug)
    dst_dir = os.path.join(REFERENCES_DIR, slug)
    
    if os.path.exists(dst_dir) and any(Path(dst_dir).iterdir()):
        logger.info("[%s] References already exist. Skipping.", slug)
        return
        
    ensure_dirs(dst_dir)
    
    images = sorted([str(p) for p in Path(src_dir).iterdir() if is_image_file(str(p))])
    if not images:
        logger.warning("[%s] No images found to generate references.", slug)
        return
        
    # 1. Find the FIRST valid face as the Anchor (User's Heuristic)
    anchor_idx = -1
    valid_embeddings = []
    valid_paths = []
    
    logger.info("[%s] Searching for first valid face as anchor...", slug)
    
    for i, path in enumerate(images):
        try:
            # We use MTCNN to ensure high quality for the anchor
            embedding = DeepFace.represent(img_path=path, model_name=EMBEDDING_MODEL, enforce_detection=True, detector_backend='mtcnn')
            if anchor_idx == -1:
                anchor_idx = len(valid_embeddings)
                logger.info("[%s] Anchor found: %s", slug, os.path.basename(path))
            
            valid_embeddings.append(embedding[0]["embedding"])
            valid_paths.append(path)
            
            # Stop once we have enough seeds to build a pool
            if len(valid_embeddings) >= num_seeds:
                break
        except Exception:
            continue
            
    if anchor_idx == -1 or len(valid_embeddings) < 2:
        logger.warning("[%s] No valid faces found in first %d images.", slug, num_seeds)
        return

    # 2. Build a Pool around the Anchor
    anchor_emb = valid_embeddings[anchor_idx]
    pool_indices = [anchor_idx]
    
    for i in range(len(valid_embeddings)):
        if i == anchor_idx: continue
        
        # Cosine distance to anchor
        dist = 1 - (np.dot(anchor_emb, valid_embeddings[i]) / (np.linalg.norm(anchor_emb) * np.linalg.norm(valid_embeddings[i])))
        
        if dist <= IDENTITY_THRESHOLD:
            pool_indices.append(i)
            if len(pool_indices) >= num_refs:
                break

    # Save the pool as references
    for idx in pool_indices:
        src = valid_paths[idx]
        dst = os.path.join(dst_dir, os.path.basename(src))
        shutil.copy2(src, dst)
        
    logger.info("[%s] Generated %d references anchored by the first downloaded face.", slug, len(pool_indices))

def main():
    ensure_dirs(REFERENCES_DIR)
    for slug in tqdm(POLITICIANS.keys(), desc="Auto-Generating References"):
        find_centroid_references(slug)

if __name__ == "__main__":
    main()
