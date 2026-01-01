"""
Constants and Configuration
============================
Central configuration for MER2LATEX project.
"""

from pathlib import Path

# ============================================
# CANONICAL INPUT FORMAT
# ============================================
TARGET_HEIGHT = 128
TARGET_WIDTH = 768
NUM_CHANNELS = 1
PAD_VALUE = 255

# ============================================
# DATASET STATISTICS (computed from EDA)
# ============================================
IM2LATEX_MEAN = 0.9449
IM2LATEX_STD = 0.1888
CROHME_MEAN = 0.9889
CROHME_STD = 0.1032

# ============================================
# MODEL SETTINGS
# ============================================
MAX_SEQ_LENGTH = 200
VOCAB_PATH = Path("/app/vocab.json")

# ============================================
# TRAINING SETTINGS
# ============================================
NUM_EPOCHS = 30
BATCH_SIZE = 4
LEARNING_RATE = 1e-4
WARMUP_STEPS = 1000
GRADIENT_CLIP = 1.0
LABEL_SMOOTHING = 0.1

# ============================================
# DATA PATHS
# ============================================
DATA_ROOT = Path("/app/data")

# IM2LATEX - actual structure
IM2LATEX_ROOT = DATA_ROOT / "IM2LATEX"
IM2LATEX_IMAGE_PATH = IM2LATEX_ROOT / "formula_images_processed" / "formula_images_processed"
IM2LATEX_TRAIN_CSV = IM2LATEX_ROOT / "im2latex_train.csv"
IM2LATEX_VAL_CSV = IM2LATEX_ROOT / "im2latex_validate.csv"
IM2LATEX_TEST_CSV = IM2LATEX_ROOT / "im2latex_test.csv"

# CROHME - processed from InkML
CROHME_ROOT = DATA_ROOT / "CROHME"
# CROHME processed data in data/preprocessed/crohme/
CROHME_PROCESSED_ROOT = DATA_ROOT / "preprocessed" / "crohme"
# Ground truth images and CSV (output from download_data.py)
CROHME_IMAGE_PATH = CROHME_PROCESSED_ROOT / "ground_truth" / "images"
CROHME_CSV_PATH = CROHME_PROCESSED_ROOT / "ground_truth" / "dataset.csv"

# Output paths for preprocessed data
OUTPUT_PATH = DATA_ROOT / "preprocessed"
OUTPUT_PATH.mkdir(exist_ok=True, parents=True)

IM2LATEX_OUTPUT_PATH = OUTPUT_PATH / "im2latex"
CROHME_OUTPUT_PATH = OUTPUT_PATH / "crohme"
IM2LATEX_OUTPUT_PATH.mkdir(exist_ok=True, parents=True)
CROHME_OUTPUT_PATH.mkdir(exist_ok=True, parents=True)

# ============================================
# MODEL CHECKPOINT PATHS
# ============================================
CHECKPOINT_ROOT = Path("/app/checkpoints")
CHECKPOINT_ROOT.mkdir(exist_ok=True, parents=True)

LOGS_ROOT = Path("/app/logs")
LOGS_ROOT.mkdir(exist_ok=True, parents=True)