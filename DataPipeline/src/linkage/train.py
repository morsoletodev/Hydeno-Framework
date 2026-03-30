import logging
import sys

import pandas as pd
from splink import block_on, Linker, DuckDBAPI, SettingsCreator
from splink.comparison_library import CustomComparison
import splink.comparison_level_library as cll

from .. import config as conf

logger = logging.getLogger(__name__)


def train_splink():
    """Creates a splink model."""
    logger.info("[Linkage Model Training Start]")

    try:
        df_sim = pd.read_parquet(conf.INTERIM_SIM, columns=conf.LINKAGE_COLUMNS)
        df_sinasc = pd.read_parquet(conf.INTERIM_SINASC, columns=conf.LINKAGE_COLUMNS)

        # Threshold definition
        # Each threshold is used to define a score for a given row-pair
        # if the sum of all scores is equal or more the a given threshold
        # then the pair is considered a true match
        peso_threshold = CustomComparison(
            output_column_name="Weight Threshold",
            comparison_levels=[
                cll.NullLevel("PESO"),
                cll.ExactMatchLevel("PESO"),
                cll.AbsoluteDifferenceLevel("PESO", difference_threshold=5),
                cll.AbsoluteDifferenceLevel("PESO", difference_threshold=10),
                cll.ElseLevel(),
            ],
        )

        gestacao_threshold = CustomComparison(
            output_column_name="Gestational Age Threshold",
            comparison_levels=[
                cll.NullLevel("SEMAGESTAC"),
                cll.ExactMatchLevel("SEMAGESTAC"),
                cll.AbsoluteDifferenceLevel("SEMAGESTAC", difference_threshold=1),
                cll.ElseLevel(),
            ],
        )

        racacor_threshold = CustomComparison(
            output_column_name="Skin Color Threshold",
            comparison_levels=[
                cll.NullLevel("RACACOR"),
                cll.ExactMatchLevel("RACACOR"),
                cll.ElseLevel(),
            ],
        )

        gravidez_threshold = CustomComparison(
            output_column_name="Pregnancy Threshold",
            comparison_levels=[
                cll.NullLevel("GRAVIDEZ"),
                cll.ExactMatchLevel("GRAVIDEZ"),
                cll.ElseLevel(),
            ],
        )

        parto_threshold = CustomComparison(
            output_column_name="Childbirth Threshold",
            comparison_levels=[
                cll.NullLevel("PARTO"),
                cll.ExactMatchLevel("PARTO"),
                cll.ElseLevel(),
            ],
        )

        sexo_threshold = CustomComparison(
            output_column_name="Sex Threshold",
            comparison_levels=[
                cll.NullLevel("SEXO"),
                cll.ExactMatchLevel("SEXO"),
                cll.ElseLevel(),
            ],
        )
        logger.info("Comparison rules created")

        # Columns used for Blocking
        # This is used to reduce computational resources needed to link two databases
        deterministic_rules = [
            block_on("DTNASC", "CODMUN"),
        ]
        logger.info("Deterministic rules created")

        linker = Linker(
            [df_sinasc, df_sim],
            SettingsCreator(
                link_type="link_only",  # used when more than 1 database is present
                comparisons=[
                    peso_threshold,
                    gestacao_threshold,
                    racacor_threshold,
                    gravidez_threshold,
                    parto_threshold,
                    sexo_threshold,
                ],
                blocking_rules_to_generate_predictions=deterministic_rules,
                retain_intermediate_calculation_columns=True,
            ),
            db_api=DuckDBAPI(),
        )
        logger.info("Linker object created")

        # Model training
        recall_test = len(df_sim) / len(df_sinasc)
        linker.training.estimate_probability_two_random_records_match(
            deterministic_rules, recall=recall_test
        )
        logger.info("Probability estimated")

        linker.training.estimate_u_using_random_sampling(max_pairs=1e7)
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
