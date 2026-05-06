import logging
import sys
from pathlib import Path

import pandas as pd
import numpy as np

from . import config as conf
from .config import GlobalConfig, DatasetConfig

logger = logging.getLogger(__name__)


def _filter(
    df: pd.DataFrame, filters: dict[str, dict[str, int | float]]
) -> pd.DataFrame:
    for col, bounds in filters.items():
        if (min := bounds.get("min")) is not None:
            df = df.loc[df[col] >= min]
        if (max := bounds.get("max")) is not None:
            df = df.loc[df[col] <= max]
    return df.reset_index(drop=True)


def _binary_group(
    df: pd.DataFrame, bgroups: dict[str, list[int | float | str]]
) -> pd.DataFrame:
    for col, group in bgroups.items():
        df[f"b{col}"] = np.where(df[col].isin(group), 1, 0)
    return df


def _threshold(
    df: pd.DataFrame, thres: dict[str, int | float | list[int] | list[float]]
) -> pd.DataFrame:
    for col, intervals in thres.items():
        if isinstance(intervals, list):
            intervals = [(df[col].min() - 1), *intervals, df[col].max()]
            print(intervals)

            df[f"t{col}"] = pd.cut(
                df[col], intervals, labels=range(len(intervals) - 1)
            ).astype(df[col].dtype)
        else:
            df[f"t{col}"] = (df[col] >= intervals).astype(int)
    return df


def _process(
    f_input: Path,
    f_output: Path,
    dataset_name: str,
    trans: conf.Transformations,
    read_columns: list[str] | None = None,
):
    df = pd.read_parquet(f_input, columns=read_columns)
    logger.info(f"{dataset_name} loaded")

    # Row Filter
    if trans.drop_duplicates:
        df = df.drop_duplicates(keep="first")
        logger.info("Drop duplicates: done")
    if trans.filter:
        df = _filter(df, trans.filter)
        logger.info("Filter: done")

    # Linkage Requirements
    df["unique_id"] = range(df.shape[0])
    logger.info("Linkage requirements: done")
    if trans.rename:
        df = df.rename(columns=trans.rename)
        logger.info("Rename: done")

    # Feature Engineering
    if trans.binary_groups:
        df = _binary_group(df, trans.binary_groups)
        logger.info("Binary group: done")
    if trans.thresholds:
        df = _threshold(df, trans.thresholds)
        logger.info("Threshold: done")

    df.to_parquet(f_output, compression=GlobalConfig().comp)
    logger.info(f"{dataset_name} saved")


def process_data():
    """Orchestrate data processing"""
    logger.info("[Process Start]")

    datasets = DatasetConfig().datasets

    try:
        for dataset, items in datasets.items():
            _process(
                items.raw, items.interim, dataset, items.transformations, items.columns
            )
    except Exception as error:
        logger.critical(
            "(!) Process service failed unexpectedly: %s", error, exc_info=True
        )
        sys.exit(1)

    logger.info("[Process End]")
