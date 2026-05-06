import logging
import sys

from ..config import LinkConfig, GlobalConfig
from .model import create_linker, _blocking_rules

logger = logging.getLogger(__name__)


def train_splink():
    """Creates a splink model."""
    logger.info("[Linkage Model Training Start]")

    try:
        linker = create_linker()
        logger.info("Linker object created")

        link_c = LinkConfig()

        block_rules = _blocking_rules(link_c.blocking_rules)

        linker.training.estimate_probability_two_random_records_match(
            block_rules, recall=link_c.recall
        )
        logger.info("Probability estimated")

        linker.training.estimate_u_using_random_sampling(
            max_pairs=link_c.max_pairs, seed=link_c.training_seed
        )
        logger.info("U estimated")

        for rule in block_rules:
            linker.training.estimate_parameters_using_expectation_maximisation(rule)
        logger.info("M estimated")

        linker.misc.save_model_to_json(GlobalConfig().f_model, overwrite=True)
        logger.info("Model saved")

    except Exception as error:
        logger.critical(
            "(!) Process service failed unexpectedly: %s", error, exc_info=True
        )
        sys.exit(1)

    logger.info("[Linkage Model Training End]")
