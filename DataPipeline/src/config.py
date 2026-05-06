from pathlib import Path
from typing import Literal

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


class Transformations(BaseModel):
    drop_duplicates: bool = True

    filter: dict[str, dict[Literal["min", "max"], int | float]] | None = None
    rename: dict[str, str] | None = None

    binary_groups: dict[str, list[int | float | str]] | None = None
    thresholds: dict[str, int | float | list[int] | list[float]] | None = None


class Dataset(BaseModel):
    dataset_name: str
    years: list[int]

    columns: list[str]

    raw: Path
    interim: Path
    processed: Path

    transformations: Transformations

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

    datasets: dict[str, Dataset]

    pre_linkage_cols: list[str]

    ensemble_columns: list[str]

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


class LinkConfig(BaseSettings):
    model_config = SettingsConfigDict(yaml_file="config/link.yaml")

    link_type: Literal["link_only", "link_and_dedupe", "dedupe_only"]

    blocking_rules: dict[str, list[str]]

    comparisons: dict[str, list[int] | list[float] | None]

    link_threshold: float

    link_columns: list[str]

    recall: float

    max_pairs: float

    training_seed: int

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
