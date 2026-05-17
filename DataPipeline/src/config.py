from pathlib import Path
from typing import Literal
from functools import lru_cache

from pydantic import BaseModel, model_validator, Field, computed_field
from pydantic_settings import BaseSettings, SettingsConfigDict, YamlConfigSettingsSource


class GlobalConfig(BaseSettings):
    """Info related to file and folder paths, file extention and compression algo used."""

    model_config = SettingsConfigDict(yaml_file="config/global.yaml")

    f_log: Path
    f_splink_model: Path

    dir_data: Path = Field(exclude=True)
    dir_raw: Path = Path("raw")
    dir_interim: Path = Path("interim")
    dir_processed: Path = Path("processed")

    ext: str
    comp: str | None = None
    merge_at_end: bool = True

    @model_validator(mode="after")
    def assemble_data(self) -> "GlobalConfig":
        # Build file paths
        self.f_log = self.f_log.resolve()
        self.f_splink_model = self.f_splink_model.resolve()

        # Build data/ related paths
        self.dir_data = self.dir_data.resolve()
        self.dir_raw = self.dir_data / self.dir_raw
        self.dir_interim = self.dir_data / self.dir_interim
        self.dir_processed = self.dir_data / self.dir_processed

        return self

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


class FilterBounds(BaseModel):
    min: int | float | None = None
    max: int | float | None = None


class Transformations(BaseModel):
    drop_duplicates: bool = True

    filter: dict[str, FilterBounds] | None = None
    rename: dict[str, str] | None = None

    binary_groups: dict[str, list[int | float | str]] | None = None
    thresholds: dict[str, int | float | list[int] | list[float]] | None = None


class Dataset(BaseModel):
    file_name: str

    start_year: int = Field(exclude=True)
    final_year: int = Field(exclude=True)

    columns: list[str]
    transformations: Transformations

    @computed_field
    def years(self) -> list[int]:
        return list(range(self.start_year, self.final_year))


class DatasetConfig(BaseSettings):
    model_config = SettingsConfigDict(yaml_file="config/dataset.yaml")

    datasets: dict[str, Dataset]

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

    seed: int

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


@lru_cache(maxsize=1)
def get_global_config() -> GlobalConfig:
    return GlobalConfig()


@lru_cache(maxsize=1)
def get_dataset_config() -> DatasetConfig:
    return DatasetConfig()


@lru_cache(maxsize=1)
def get_link_config() -> LinkConfig:
    return LinkConfig()


def get_dataset_paths(ds: Dataset) -> dict[str, Path]:
    gc = get_global_config()

    name = (
        f"{ds.file_name}.{gc.ext}.{gc.comp}" if gc.comp else f"{ds.file_name}.{gc.ext}"
    )

    return {
        "raw": gc.dir_raw / name,
        "interim": gc.dir_interim / name,
        "processed": gc.dir_processed / name,
    }


def create_folders():
    """Ensures the correct folder structure exists before execution."""

    config = get_global_config()

    for folder in [
        config.dir_raw,
        config.dir_interim,
        config.dir_processed,
        config.f_log.parent,
        config.f_splink_model.parent,
    ]:
        folder.mkdir(parents=True, exist_ok=True)
