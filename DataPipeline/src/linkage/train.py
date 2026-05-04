import logging
import sys

import pandas as pd
from splink import block_on

from .. import config as conf
from .model import create_linker

logger = logging.getLogger(__name__)


def train_splink():
    """Creates a splink model."""
    logger.info("[Linkage Model Training Start]")

    try:
        linker = create_linker()
        logger.info("Linker object created")

        df_sim = pd.read_parquet(conf.INTERIM_SIM, columns=conf.LINKAGE_COLUMNS)
        df_sinasc = pd.read_parquet(conf.INTERIM_SINASC, columns=conf.LINKAGE_COLUMNS)
        deterministic_rules = [
            block_on("DTNASC", "CODMUN"),
        ]

        # Model training
        recall_test = len(df_sim) / len(df_sinasc)
        linker.training.estimate_probability_two_random_records_match(
            deterministic_rules, recall=recall_test
        )
        logger.info("Probability estimated")

        linker.training.estimate_u_using_random_sampling(max_pairs=1e7, seed=42)
        logger.info("U estimated")

        training_blocking_rule = block_on("DTNASC", "CODMUN")
        linker.training.estimate_parameters_using_expectation_maximisation(
            training_blocking_rule
        )
        logger.info("M estimated")

        linker.misc.save_model_to_json(conf.SPLINK_MODEL, overwrite=True)
        logger.info("Model saved")

    except Exception as error:
        logger.critical(
            "(!) Process service failed unexpectedly: %s", error, exc_info=True
        )
        sys.exit(1)

    logger.info("[Linkage Model Training End]")
