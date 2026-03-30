from pathlib import Path

# General paths
# Input file location
DRIVE_PATH = "./sinasc.parquet.gzip"
LOCAL_PATH = Path("../data/processed/sinasc.parquet.gzip").resolve()

# Model folder
MODEL_PATH = Path("../models/ml_model.pkl").resolve()

# Log file
LOG_FILE = Path("../logs/ensemble.log").resolve()

# Algorithms
# General
RANDOM_STATE = 42
SAMPLING_STRATEGY = 1.0
N_JOBS = -1

# StratifiedKFold
N_SPLITS = 5

# HPO
SCORING = ["recall", "precision", "f1"]
REFIT = "f1"
