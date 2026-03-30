import logging
import sys

import pandas as pd
import numpy as np

from . import config as conf

logger = logging.getLogger(__name__)


def process_sinasc():
    """Filter, create linkage requirements and create categories for SINASC."""
    df = pd.read_parquet(conf.RAW_SINASC, columns=conf.SINASC_COLUMNS)
    logger.info("SINASC loaded")

    # Row filter
    df = df.drop_duplicates(keep="first")
    df = df.loc[(df["PESO"] >= 350) & (df["PESO"] <= 6500)]
    logger.info("Dataset filtered")

    # Linkage requirements
    df["unique_id"] = range(df.shape[0])
    df = df.rename(columns={"CODMUNNASC": "CODMUN"})
    logger.info("Linkage requirements met")

    # Categories creation
    df["catTPROBSON"] = np.where(df["TPROBSON"].isin([1, 2, 3, 4, 6]), 0, 1)
    df["catPESO"] = (df["PESO"] >= 2500).astype(int)
    df["catSEMAGESTAC"] = (df["SEMAGESTAC"] >= 37).astype(int)
    logger.info("Categorical columns created")

    # File save
    df.to_parquet(conf.INTERIM_SINASC, compression="gzip")
    logger.info("Data saved")


def process_sim():
    """Filter and create linkage requirements for SIM"""
    df = pd.read_parquet(conf.RAW_SIM, columns=conf.SIM_COLUMNS)
    logger.info("SIM loaded")

    # Row filter
    df = df.drop_duplicates(keep="first")
    df = df.loc[df["IDADE"] < 229, :]  # IDADE menor que 29 dias
    df = df.drop(columns=["IDADE"])
    df = df.loc[(df["PESO"] >= 350) & (df["PESO"] <= 6500), :]
    logger.info("Dataset filtered")

    # Linkage requirements
    df["unique_id"] = range(df.shape[0])
    df = df.rename(columns={"CODMUNNATU": "CODMUN"})
    logger.info("Linkage requirements met")

    # File save
    df.to_parquet(conf.INTERIM_SIM, compression="gzip")
    logger.info("Data saved")


def process_data():
    """Orchestrate SIM and SINASC processing"""
    logger.info("[Process Start]")

    try:
        process_sinasc()
        process_sim()
    except Exception as error:
        logger.critical(
            "(!) Process service failed unexpectedly: %s", error, exc_info=True
        )
        sys.exit(1)

    logger.info("[Process End]")
