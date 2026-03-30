from pathlib import Path

# File paths for internal use
# These are same for both Docker and local environments
LOG_FILE = Path("../logs/datapipeline.log").resolve()

MODEL_DIR = Path("../models/").resolve()
SPLINK_MODEL = MODEL_DIR / "splink.json"

DATA_DIR = Path("../data/").resolve()
RAW_DIR = DATA_DIR / "raw"
INTERIM_DIR = DATA_DIR / "interim"
PROCESSED_DIR = DATA_DIR / "processed"

RAW_SINASC = RAW_DIR / "DN.parquet.gzip"
RAW_SIM = RAW_DIR / "DO.parquet.gzip"

INTERIM_SINASC = INTERIM_DIR / "DN.parquet.gzip"
INTERIM_SIM = INTERIM_DIR / "DO.parquet.gzip"

PROCESSED_SINASC = PROCESSED_DIR / "sinasc.parquet.gzip"

# dataset.py
# Extracts the datasets related to the following years
SINASC_YEARS = [2020, 2021, 2022, 2023]
SIM_YEARS = [2020, 2021, 2022, 2023, 2024]

# features.py
# Filters the databases using the lists below:
SINASC_COLUMNS = [
    # expected for linkage:
    "DTNASC",
    "CODMUNNASC",  # as CODMUN
    "PARTO",
    "SEXO",
    "PESO",
    "SEMAGESTAC",
    "RACACOR",
    "GRAVIDEZ",
    # In addition to :
    # unique_id (created during execution)
    # expected for ensemble (output):
    "APGAR5",
    "CONSPRENAT",
    "IDANOMAL",
    "IDADEMAE",
    "TPROBSON",  # as catTPROBSON
    # In addition to :
    # RACACOR, catPESO, catSEMAGESTAC, target (as OBITO)
]
SIM_COLUMNS = [
    # Used for row filter
    "IDADE",
    # expected for linkage:
    "DTNASC",
    "CODMUNNATU",
    "PARTO",
    "SEXO",
    "PESO",
    "SEMAGESTAC",
    "RACACOR",
    "GRAVIDEZ",
]

# linkage module
# All columns needed by splink and output combined
PRE_LINKAGE_COLS = [
    "DTNASC",
    "CODMUN",
    "PARTO",
    "SEXO",
    "PESO",
    "SEMAGESTAC",
    "RACACOR",
    "GRAVIDEZ",
    "unique_id",
    "APGAR5",
    "CONSPRENAT",
    "IDANOMAL",
    "IDADEMAE",
    "catTPROBSON",
    "catSEMAGESTAC",
    "catPESO",
]

# Threshold used by splink to filter matchs from non-matches
LINKAGE_THRESHOLD = -3.5

# Columns used to filter the datasets in memory right before linkage
LINKAGE_COLUMNS = [
    "unique_id",
    "DTNASC",
    "CODMUN",
    "PARTO",
    "SEXO",
    "PESO",
    "SEMAGESTAC",
    "RACACOR",
    "GRAVIDEZ",
]

# Output required by ensemble models
ENSEMBLE_COLUMNS = [
    "catSEMAGESTAC",
    "catPESO",
    "APGAR5",
    "CONSPRENAT",
    "IDANOMAL",
    "RACACOR",
    "catTPROBSON",
    "IDADEMAE",
    "OBITO",
]


def create_folders():
    """Ensures data/ has the correct folder structure inside before execution."""
    for folder in [RAW_DIR, INTERIM_DIR, PROCESSED_DIR]:
        folder.mkdir(parents=True, exist_ok=True)
