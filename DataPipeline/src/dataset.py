import logging

import typer
from etlsus import pipeline

from .config import get_dataset_config, get_global_config

logger = logging.getLogger(__name__)


def get_dataset():
    """Extracts databases required."""
    logger.info("[Acquire Start]")

    datasets = get_dataset_config().datasets
    global_c = get_global_config()

    try:
        for name, configs in datasets.items():
            pipeline(
                dataset=name,
                data_dir=str(global_c.dir_raw),
                years_to_extract=configs.years,
                merge_at_end=global_c.merge_at_end,
            )
            logger.info("%s extracted", configs.file_name)

    except Exception as error:
        logger.critical(
            "(!) Acquire service failed unexpectedly: %s", error, exc_info=True
        )
        raise typer.Exit(code=1)

    logger.info("[Acquire End]")
