import argparse
import logging
from src.utils.config_loader import load_config
from src.utils.helpers import setup_logger
from src.data.downloader import Downloader
from src.data.cleaner import DataCleaner
from src.data.anchor_gen import AnchorGenerator
from src.data.verifier import IdentityVerifier
from src.preprocessing.padding import DatasetPadder
from src.data.splitter import DatasetSplitter
from src.training.trainer import Trainer
from src.evaluation.visualization import Visualizer

def run_pipeline(args):
    config = load_config(args.config)
    logger = setup_logger("pipeline", config, "pipeline.log")
    
    logger.info("Starting Full Project Transformation Pipeline")
    
    if args.step <= 1:
        logger.info(">> STEP 1: Downloading Images")
        Downloader(config).run()
        
    if args.step <= 2:
        logger.info(">> STEP 2: Cleaning and Face Extraction")
        DataCleaner(config).run()
        
    if args.step <= 3:
        logger.info(">> STEP 3: Identity Verification (including Anchor Gen)")
        AnchorGenerator(config).run()
        IdentityVerifier(config).run()
        
    if args.step <= 4:
        logger.info(">> STEP 4: Dataset Padding & Balancing")
        DatasetPadder(config).run()
        
    if args.step <= 5:
        logger.info(">> STEP 5: Dataset Splitting")
        DatasetSplitter(config).run()
        
    if args.step <= 6:
        logger.info(">> STEP 6: Visualization")
        Visualizer(config).run()
        
    if args.step <= 7:
        logger.info(">> STEP 7: Training & Evaluation")
        Trainer(config).train("ResNet50")
        Trainer(config).train("EfficientNetB0")

    logger.info("Pipeline Execution Complete!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pakistani Politician Classifier Pipeline")
    parser.add_argument("--config", default="configs/config.yaml", help="Path to config file")
    parser.add_argument("--step", type=int, default=1, help="Start from specific step (1-7)")
    args = parser.parse_args()
    
    run_pipeline(args)
