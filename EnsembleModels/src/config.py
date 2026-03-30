from pathlib import Path

# Drive file
DRIVE_PATH = "./sinasc.parquet.gzip"
LOCAL_PATH = Path("../data/processed/sinasc.parquet.gzip").resolve()

# Model folder
MODEL_PATH = Path("../models/ml_model.pkl").resolve()

# Log file
LOG_FILE = Path("../logs/ensemble.log").resolve()

# StratifiedKFold
N_SPLITS = 5

# Models
RANDOM_STATE = 42
SAMPLING_STRATEGY = 1.0
N_JOBS = -1

# HPO
CV = 1
SCORING = ["recall", "precision", "f1"]
REFIT = "f1"
