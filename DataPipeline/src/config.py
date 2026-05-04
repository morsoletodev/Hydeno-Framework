from pathlib import Path
from typing import List, Dict

from pydantic import BaseModel, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict, YamlConfigSettingsSource


class GlobalConfig(BaseSettings):
    """Info related to file and folder paths, file extention and compression algo used."""

    model_config = SettingsConfigDict(yaml_file="config/global.yaml")

    f_log: Path
    f_model: Path

    raw_dir: Path
    interim_dir: Path
    processed_dir: Path

    base_ext: str
    comp: str | None = None

    @model_validator(mode="before")
    @classmethod
    def assemble_data(cls, data: dict) -> dict:
        # Build file paths
        for file in data.keys():
            if "f_" in file:
                data[file] = Path(data[file]).resolve()

        # Build data/ related paths
        base_dir = Path(data.pop("base_data_dir")).resolve()
        for folder in ["raw_dir", "interim_dir", "processed_dir"]:
            if folder in data:
                data[folder] = base_dir / data[folder]

        return data

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls,
        init_settings,
        env_settings,
        dotenv_settings,
        file_secret_settings,
    ):
        return (YamlConfigSettingsSource(settings_cls),)


LOG_FILE = Path("../logs/datapipeline.log").resolve()

MODEL_DIR = Path("../models/").resolve()
SPLINK_MODEL = MODEL_DIR / "splink.json"

DATA_DIR = Path("../data/").resolve()
RAW_DIR = DATA_DIR / "raw"
INTERIM_DIR = DATA_DIR / "interim"
PROCESSED_DIR = DATA_DIR / "processed"

# Dataset info
RAW_SINASC = RAW_DIR / "DN.parquet.gzip"
RAW_SIM = RAW_DIR / "DO.parquet.gzip"

INTERIM_SINASC = INTERIM_DIR / "DN.parquet.gzip"
INTERIM_SIM = INTERIM_DIR / "DO.parquet.gzip"

PROCESSED_SINASC = PROCESSED_DIR / "sinasc.parquet.gzip"


class Dataset(BaseModel):
    dataset_name: str
    years: List[int]

    columns: List[str]

    raw: Path
    interim: Path
    processed: Path

    @model_validator(mode="before")
    @classmethod
    def assemble_data(cls, data: dict) -> dict:
        global_c = GlobalConfig()

        # Build years
        data["years"] = list(range(data.pop("start_year"), data.pop("final_year")))

        # Build file paths for raw, interim and processed folders
        name = data.pop("dataset_filename")
        filename = (
            f"{name}.{global_c.base_ext}.{global_c.comp}"
            if global_c.comp
            else f"{name}.{global_c.base_ext}"
        )
        for file in ["raw", "interim", "processed"]:
            data[file] = getattr(global_c, f"{file}_dir") / filename

        return data


class DatasetConfig(BaseSettings):
    model_config = SettingsConfigDict(yaml_file="config/dataset.yaml")

    datasets: Dict[str, Dataset]

    pre_linkage_cols: List[str]

    linkage_threshold: float
    linkage_columns: List[str]

    ensemble_columns: List[str]

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls,
        init_settings,
        env_settings,
        dotenv_settings,
        file_secret_settings,
    ):
        return (YamlConfigSettingsSource(settings_cls),)


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
    """Ensures the correct folder structure exists before execution."""

    config = GlobalConfig()

    for folder in [
        config.raw_dir,
        config.interim_dir,
        config.processed_dir,
        config.f_log.parent,
        config.f_model.parent,
    ]:
        folder.mkdir(parents=True, exist_ok=True)
