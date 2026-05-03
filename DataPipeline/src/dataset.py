import logging
import sys

from etlsus import pipeline

from .config import DatasetConfig, GlobalConfig

logger = logging.getLogger(__name__)


def get_dataset():
    """Extracts databases required."""
    logger.info("[Acquire Start]")

    dataset_c = DatasetConfig()

    try:
        for dataset, configs in dataset_c.datasets.items():
            pipeline(
                dataset=dataset,
                data_dir=str(GlobalConfig().raw_dir),
                years_to_extract=configs.years,
                merge_at_end=True,
            )
            logger.info(f"{dataset} extracted")

    except Exception as error:
        logger.critical(
            "(!) Acquire service failed unexpectedly: %s", error, exc_info=True
        )
        sys.exit(1)

    logger.info("[Acquire End]")
