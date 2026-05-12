import os
import shutil
import numpy as np
from pathlib import Path
from deepface import DeepFace
from tqdm import tqdm
from src.utils.helpers import setup_logger, ensure_dirs, is_image_file

class AnchorGenerator:
    def __init__(self, config: dict):
        self.config = config
        self.logger = setup_logger("anchor_gen", config, "anchor_gen.log")
        self.raw_dir = config['paths']['raw_dir']
        self.ref_dir = config['paths']['references_dir']
        self.politicians = config['politicians']
        self.settings = config['processing']
        
        ensure_dirs(self.ref_dir)

    def generate_for_class(self, slug: str, num_seeds=20, num_refs=5):
        src_dir = os.path.join(self.raw_dir, slug)
        dst_dir = os.path.join(self.ref_dir, slug)
        
        if os.path.exists(dst_dir) and any(Path(dst_dir).iterdir()):
            self.logger.info("[%s] References exist. Skipping.", slug)
            return
            
        ensure_dirs(dst_dir)
        images = sorted([str(p) for p in Path(src_dir).iterdir() if is_image_file(str(p))])
        if not images: return
            
        valid_embeddings = []
        valid_paths = []
        anchor_idx = -1
        
        self.logger.info("[%s] Searching for anchor...", slug)
        for path in images[:num_seeds]:
            try:
                embedding = DeepFace.represent(
                    img_path=path, 
                    model_name=self.settings['embedding_model'], 
                    enforce_detection=True, 
                    detector_backend='mtcnn'
                )
                if anchor_idx == -1:
                    anchor_idx = len(valid_embeddings)
                
                valid_embeddings.append(embedding[0]["embedding"])
                valid_paths.append(path)
            except: continue
                
        if anchor_idx == -1: return

        anchor_emb = valid_embeddings[anchor_idx]
        pool = [anchor_idx]
        for i in range(len(valid_embeddings)):
            if i == anchor_idx: continue
            dist = 1 - (np.dot(anchor_emb, valid_embeddings[i]) / (np.linalg.norm(anchor_emb) * np.linalg.norm(valid_embeddings[i])))
            if dist <= self.settings['identity_threshold']:
                pool.append(i)
                if len(pool) >= num_refs: break

        for idx in pool:
            shutil.copy2(valid_paths[idx], os.path.join(dst_dir, os.path.basename(valid_paths[idx])))
        
        self.logger.info("[%s] Generated %d references.", slug, len(pool))

    def run(self):
        for slug in tqdm(self.politicians.keys(), desc="Generating Anchors"):
            self.generate_for_class(slug)

if __name__ == "__main__":
    from src.utils.config_loader import load_config
    cfg = load_config()
    gen = AnchorGenerator(cfg)
    gen.run()
