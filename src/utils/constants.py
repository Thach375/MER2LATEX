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
# DATASET STATISTICS
# ============================================
IM2LATEX_MEAN = 0.9449
IM2LATEX_STD = 0.1888
CROHME_MEAN = 0.9889
CROHME_STD = 0.1032

# ============================================
# DATA PATHS
# ============================================
DATA_ROOT = Path("/app/data")

# IM2LATEX
IM2LATEX_ROOT = DATA_ROOT / "IM2LATEX"
IM2LATEX_IMAGE_PATH = IM2LATEX_ROOT / "image"
IM2LATEX_LABEL_PATH = IM2LATEX_ROOT / "label"

# CROHME
CROHME_ROOT = DATA_ROOT / "CROHME"
CROHME_IMAGE_PATH = CROHME_ROOT / "ground_truth" / "images"
CROHME_CSV_PATH = CROHME_ROOT / "ground_truth" / "dataset.csv"

# Output
OUTPUT_ROOT = DATA_ROOT / "preprocessed"
IM2LATEX_OUTPUT_PATH = OUTPUT_ROOT / "im2latex"
CROHME_OUTPUT_PATH = OUTPUT_ROOT / "crohme"

# Output paths for preprocessed data
OUTPUT_PATH = Path("/app/data/preprocessed")
OUTPUT_PATH.mkdir(exist_ok=True, parents=True)

IM2LATEX_OUTPUT_PATH = OUTPUT_PATH / "im2latex"
CROHME_OUTPUT_PATH = OUTPUT_PATH / "crohme"
IM2LATEX_OUTPUT_PATH.mkdir(exist_ok=True)
CROHME_OUTPUT_PATH.mkdir(exist_ok=True)