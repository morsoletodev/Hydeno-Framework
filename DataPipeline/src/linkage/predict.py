import logging
import sys

import pandas as pd
from splink import DuckDBAPI, Linker

from ..config import LinkConfig, DatasetConfig, GlobalConfig
from .train import train_splink
from .model import _dfs

logger = logging.getLogger(__name__)


def predict_splink():
    """Perform linkage between SIM and SINASC"""
    logger.info("[Linkage Start]")

    dc = DatasetConfig()
    lc = LinkConfig()
    gc = GlobalConfig()

    try:
        # Executes training if needed
        if not gc.f_model.exists():
            logger.info("ModelFileNotFound: training model instead")
            train_splink()
            logger.info("Resuming Linkage")

        linker = Linker(
            _dfs(lc.link_columns, dc.datasets),
            gc.f_model,
            db_api=DuckDBAPI(),
        )

        # Infer possible links
        df_links = linker.inference.predict()
        logger.info("Inference done")

        # Filters for distinct ids with match weight equal or higher than threshold
        sql = f"""
        SELECT DISTINCT unique_id_l
        FROM {df_links.physical_name}
        WHERE match_weight >= {lc.link_threshold}
        """
        id_sinasc = linker.misc.query_sql(sql)["unique_id_l"]
        logger.info("SINASC ids found: %s", id_sinasc.shape)

        # Creates target column named `OBITO`
        df_sinasc = pd.read_parquet(
            dc.datasets["SINASC"].interim, columns=dc.pre_linkage_cols
        )
        df_sinasc["OBITO"] = df_sinasc["unique_id"].isin(id_sinasc).astype(int)
        logger.info("'OBITO' column created")

        # Saves data
        df_sinasc[dc.ensemble_columns].to_parquet(
            dc.datasets["SINASC"].processed, compression="gzip"
        )
        logger.info("Output file for ensemble models created")

    except Exception as error:
        logger.critical(
            "(!) Linkage service failed unexpectedly: %s", error, exc_info=True
        )
        sys.exit(1)

    logger.info("[Linkage End]")
