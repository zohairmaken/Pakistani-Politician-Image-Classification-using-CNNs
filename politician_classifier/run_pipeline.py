"""
run_pipeline.py
===============
Master script — runs the ADVANCED pipeline end-to-end.

Steps:
  1. Download raw images
  2. Clean (corrupted / duplicate / blurry)
  3. Intelligent Cleaning (isolate faces, remove group noise)
  3a. Identity Verification (DeepFace embedding matching)
  4. Train / Val / Test split
  5. Visualisation & statistics
  6. CNN Training (ResNet50, EfficientNetB0)
  7. Launch Dashboard
"""

import argparse
import subprocess
import sys
import time
import os
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from utils import setup_logger, ensure_dirs
from config import REFERENCES_DIR, CLEANED_DIR, PROCESSED_DIR, MODELS_DIR

logger = setup_logger("pipeline", "pipeline.log")
PYTHON = sys.executable

STEPS = {
    10: ("Download Images",         "01_download_images.py"),
    20: ("Initial Cleaning",        "02_clean_dataset.py"),
    30: ("Intelligent Cleaning",    "03_intelligent_cleaner.py"),
    40: ("Auto Reference Gen",      "03b_auto_reference_generator.py"),
    50: ("Identity Verification",   "03a_identity_verification.py"),
    60: ("Train/Val/Test Split",    "04_split_dataset.py"),
    70: ("Basic Visualisation",     "05_visualize_dataset.py"),
    80: ("CNN Training",            "06_train_models.py"),
}

def run_step(step_num: int, script: str, extra_args: list[str]) -> bool:
    cmd = [PYTHON, script] + extra_args
    label = STEPS[step_num][0]

    logger.info("")
    logger.info(">> STEP %d -- %s", step_num, label)
    logger.info("-" * 60)

    start = time.time()
    result = subprocess.run(cmd, cwd=Path(__file__).parent)
    elapsed = time.time() - start

    if result.returncode == 0:
        logger.info("V  Step completed in %.1fs\n", elapsed)
        return True
    else:
        logger.error("X  Step FAILED (exit code %d)", result.returncode)
        return False

def main():
    parser = argparse.ArgumentParser(description="Pakistani Politician AI Pipeline")
    parser.add_argument("--steps", nargs="+", type=int, default=[10, 20, 30, 40, 50, 60, 70, 80])
    parser.add_argument("--skip-dl", action="store_true", help="Skip download step")
    parser.add_argument("--dashboard", action="store_true", help="Launch Streamlit dashboard after training")
    args = parser.parse_args()

    ensure_dirs(REFERENCES_DIR, CLEANED_DIR, PROCESSED_DIR, MODELS_DIR)

    steps_to_run = sorted(args.steps)
    if args.skip_dl and 1 in steps_to_run:
        steps_to_run.remove(1)

    logger.info("=" * 65)
    logger.info("  PAKISTANI POLITICIAN ADVANCED AI PIPELINE")
    logger.info("  Steps: %s", steps_to_run)
    logger.info("=" * 65)

    for s in steps_to_run:
        success = run_step(s, STEPS[s][1], [])
        if not success:
            logger.error("Pipeline halted.")
            break

    if args.dashboard:
        logger.info("Launching Dashboard...")
        subprocess.run(["streamlit", "run", "dashboard.py"], cwd=Path(__file__).parent)

if __name__ == "__main__":
    main()
