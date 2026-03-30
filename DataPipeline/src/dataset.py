import logging
import sys

from etlsus import pipeline

from . import config as conf

logger = logging.getLogger(__name__)


def get_dataset():
    """Extracts databases required."""
    logger.info("[Acquire Start]")

    try:
        pipeline(
            dataset="SINASC",
            data_dir=str(conf.RAW_DIR),
            years_to_extract=conf.SINASC_YEARS,
            merge_at_end=True,
        )
        logger.info("SINASC extracted")

        pipeline(
            dataset="SIM",
            data_dir=str(conf.RAW_DIR),
            years_to_extract=conf.SIM_YEARS,
            merge_at_end=True,
        )
        logger.info("SIM extracted")
    except Exception as error:
        logger.critical(
            "(!) Acquire service failed unexpectedly: %s", error, exc_info=True
        )
        sys.exit(1)

    logger.info("[Acquire End]")
