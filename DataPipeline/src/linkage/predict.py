import logging
import sys

import pandas as pd
from splink import DuckDBAPI, Linker

from .. import config as conf
from .train import train_splink

logger = logging.getLogger(__name__)


def predict_splink():
    """Perform linkage between SIM and SINASC"""
    logger.info("[Linkage Start]")

    try:
        df_sim = pd.read_parquet(conf.INTERIM_SIM, columns=conf.LINKAGE_COLUMNS)
        df_sinasc = pd.read_parquet(conf.INTERIM_SINASC, columns=conf.PRE_LINKAGE_COLS)

        # Executes training if needed
        if not conf.SPLINK_MODEL.exists():
            logger.info("ModelFileNotFound: training model instead")
            train_splink()
            logger.info("Resuming Linkage")

        linker = Linker(
            [df_sinasc[conf.LINKAGE_COLUMNS], df_sim],
            conf.SPLINK_MODEL,
            db_api=DuckDBAPI(),
        )

        # Infer possible links
        df_links = linker.inference.predict()
        logger.info("Inference done")

        # Filters for distinct ids with match weight equal or higher than threshold
        sql = f"""
        SELECT DISTINCT unique_id_l
        FROM {df_links.physical_name}
        WHERE match_weight >= {conf.LINKAGE_THRESHOLD}
        """
        id_sinasc = linker.misc.query_sql(sql)["unique_id_l"]
        logger.info("SINASC ids found: %s", id_sinasc.shape)

        # Creates target column named `OBITO`
        df_sinasc["OBITO"] = df_sinasc["unique_id"].isin(id_sinasc).astype(int)
        logger.info("'OBITO' column created")

        # Saves data
        df_sinasc[conf.ENSEMBLE_COLUMNS].to_parquet(
            conf.PROCESSED_SINASC, compression="gzip"
        )
        logger.info("Output file for ensemble models created")

    except Exception as error:
        logger.critical(
            "(!) Linkage service failed unexpectedly: %s", error, exc_info=True
        )
        sys.exit(1)

    logger.info("[Linkage End]")
