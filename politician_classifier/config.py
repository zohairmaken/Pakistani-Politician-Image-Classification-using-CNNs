"""
config.py
=========
Central configuration file for the Pakistani Politician Image Classification pipeline.
"""

import os

# ─────────────────────────────────────────────────────────────────────────────
# PATHS
# ─────────────────────────────────────────────────────────────────────────────
BASE_DIR        = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR     = os.path.join(BASE_DIR, "dataset")
RAW_DIR         = os.path.join(DATASET_DIR, "raw")
CLEANED_DIR     = os.path.join(DATASET_DIR, "cleaned")   # Isolated face regions
PROCESSED_DIR   = os.path.join(DATASET_DIR, "processed") # Final split dataset
REFERENCES_DIR  = os.path.join(BASE_DIR, "references")    # Hand-picked clean faces
LOGS_DIR        = os.path.join(BASE_DIR, "logs")
STATS_DIR       = os.path.join(BASE_DIR, "stats")
MODELS_DIR      = os.path.join(BASE_DIR, "models")

TRAIN_DIR = os.path.join(PROCESSED_DIR, "train")
VAL_DIR   = os.path.join(PROCESSED_DIR, "val")
TEST_DIR  = os.path.join(PROCESSED_DIR, "test")

# ─────────────────────────────────────────────────────────────────────────────
# DOWNLOAD SETTINGS
# ─────────────────────────────────────────────────────────────────────────────
IMAGES_PER_KEYWORD = 200
MAX_IMAGES_PER_CLASS = 600
NUM_WORKERS          = 4
DOWNLOAD_TIMEOUT     = 10  # Added missing variable
MAX_FACES_ALLOWED = 1

# Additional Search Keywords for better variety
SEARCH_SUFFIXES = [
    "official portrait",
    "press conference face",
    "interview close up",
    "parliament session face",
    "speech face",
    "hd portrait",
    "profile photo"
]

# ─────────────────────────────────────────────────────────────────────────────
# IMAGE QUALITY & FACE SETTINGS
# ─────────────────────────────────────────────────────────────────────────────
TARGET_SIZE          = (224, 224)
BLUR_THRESHOLD       = 100.0
HASH_THRESHOLD       = 8
MIN_FACE_CONFIDENCE  = 0.95
FACE_MARGIN          = 0.20
MIN_IMAGE_BYTES      = 5_000

# ─────────────────────────────────────────────────────────────────────────────
# INTELLIGENT FILTERING
# ─────────────────────────────────────────────────────────────────────────────
MAX_FACES_ALLOWED    = 1      # STRICT: Discard images with more than 1 face (Group photos)
MIN_FACE_SIZE_RATIO  = 0.1    # Face must be at least 10% of image width/height
IDENTITY_THRESHOLD   = 0.40   # DeepFace distance threshold (lower is stricter)
EMBEDDING_MODEL      = "Facenet512" # Models: VGG-Face, Facenet, Facenet512, OpenFace, DeepFace, DeepID, ArcFace, Dlib, SFace

# ─────────────────────────────────────────────────────────────────────────────
# TRAINING PARAMETERS
# ─────────────────────────────────────────────────────────────────────────────
BATCH_SIZE           = 32
EPOCHS               = 25
LEARNING_RATE        = 1e-4
VAL_SPLIT            = 0.15
TEST_SPLIT           = 0.10
TRAIN_RATIO          = 0.75 # Added for split script
VAL_RATIO            = 0.15 # Added for split script
TEST_RATIO           = 0.10 # Added for split script
SEED                 = 42

# ─────────────────────────────────────────────────────────────────────────────
# POLITICIANS
# ─────────────────────────────────────────────────────────────────────────────
POLITICIANS = {
    "imran_khan": {
        "display_name": "Imran Khan",
        "keywords": ["Imran Khan PTI", "Imran Khan official portrait", "Imran Khan face"]
    },
    "nawaz_sharif": {
        "display_name": "Nawaz Sharif",
        "keywords": ["Nawaz Sharif PMLN", "Nawaz Sharif official portrait", "Nawaz Sharif face"]
    },
    "shahbaz_sharif": {
        "display_name": "Shahbaz Sharif",
        "keywords": ["Shahbaz Sharif Prime Minister", "Shahbaz Sharif face", "Shahbaz Sharif portrait"]
    },
    "maryam_nawaz": {
        "display_name": "Maryam Nawaz",
        "keywords": ["Maryam Nawaz Sharif", "Maryam Nawaz face", "Maryam Nawaz portrait"]
    },
    "bilawal_bhutto": {
        "display_name": "Bilawal Bhutto",
        "keywords": ["Bilawal Bhutto Zardari", "Bilawal Bhutto face", "Bilawal Bhutto portrait"]
    },
    "asif_ali_zardari": {
        "display_name": "Asif Ali Zardari",
        "keywords": ["Asif Ali Zardari", "Asif Zardari face", "Asif Zardari portrait"]
    },
    "fazlur_rehman": {
        "display_name": "Fazlur Rehman",
        "keywords": ["Maulana Fazlur Rehman", "Fazlur Rehman JUIF", "Fazlur Rehman face"]
    },
    "sheikh_rasheed": {
        "display_name": "Sheikh Rasheed",
        "keywords": ["Sheikh Rasheed Ahmed", "Sheikh Rasheed face", "Sheikh Rasheed portrait"]
    },
    "mohsin_naqvi": {
        "display_name": "Mohsin Naqvi",
        "keywords": ["Mohsin Naqvi face", "Mohsin Naqvi portrait", "Mohsin Naqvi PCB", "Mohsin Naqvi CM Punjab", "Mohsin Naqvi official"]
    },
    "hina_rabbani_khar": {
        "display_name": "Hina Rabbani Khar",
        "keywords": ["Hina Rabbani Khar face", "Hina Rabbani Khar portrait", "Hina Rabbani Khar MOFA", "Hina Rabbani Khar official"]
    },
    "murad_ali_shah": {
        "display_name": "Murad Ali Shah",
        "keywords": ["Murad Ali Shah face", "Murad Ali Shah portrait", "Murad Ali Shah Sindh", "Murad Ali Shah CM", "Murad Ali Shah official"]
    },
    "ali_amin_gandapur": {
        "display_name": "Ali Amin Gandapur",
        "keywords": ["Ali Amin Gandapur face", "Ali Amin Gandapur portrait", "Ali Amin Gandapur KPK", "Ali Amin Gandapur CM", "Ali Amin Gandapur official"]
    },
    "khawaja_asif": {
        "display_name": "Khawaja Asif",
        "keywords": ["Khawaja Asif face", "Khawaja Asif portrait", "Khawaja Asif PMLN"]
    },
    "attaullah_tarar": {
        "display_name": "Attaullah Tarar",
        "keywords": ["Attaullah Tarar face", "Attaullah Tarar portrait"]
    },
    "chaudhry_pervaiz_elahi": {
        "display_name": "Chaudhry Pervaiz Elahi",
        "keywords": ["Chaudhry Pervaiz Elahi face", "Chaudhry Pervaiz Elahi portrait"]
    },
    "ahmed_sharif_chaudhry": {
        "display_name": "Ahmed Sharif Chaudhry",
        "keywords": ["Major General Ahmed Sharif Chaudhry face", "Ahmed Sharif Chaudhry DG ISPR", "General Ahmed Sharif portrait", "ISPR spokesperson face"]
    },
}

# ─────────────────────────────────────────────────────────────────────────────
# LOGGING
# ─────────────────────────────────────────────────────────────────────────────
LOG_LEVEL = "INFO"
LOG_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
DATE_FORMAT = "%Y-%m-%d %H:%M:%S"
