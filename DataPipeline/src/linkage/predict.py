import logging

import pandas as pd
from splink import DuckDBAPI, Linker
import typer

from ..config import (
    get_dataset_paths,
    get_link_config,
    get_dataset_config,
    get_global_config,
)
from .model import _dfs

logger = logging.getLogger(__name__)


def predict_splink():
    """Performs linkage between SIM and SINASC"""
    logger.info("[Linkage Start]")

    dc = get_dataset_config()
    lc = get_link_config()
    gc = get_global_config()
    db_api = DuckDBAPI()
    sinasc_paths = get_dataset_paths(dc.datasets["SINASC"])

    try:
        if not gc.f_splink_model.exists():
            raise FileNotFoundError("%s doesn't exists", gc.f_splink_model)

        linker = Linker(
            _dfs(db_api, lc.link_columns, dc.datasets),
            gc.f_splink_model,
            db_api=db_api,
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
        logger.info("SINASC ids found: %s", len(id_sinasc))

        # Creates target column named `OBITO`
        df_sinasc = pd.read_parquet(
            sinasc_paths["interim"],
            columns=[col for col in dc.ensemble_columns if col != "OBITO"]
            + ["unique_id"],
        )
        df_sinasc["OBITO"] = df_sinasc["unique_id"].isin(id_sinasc).astype(int)
        logger.info("'OBITO' column created")

        # Saves data
        df_sinasc.to_parquet(sinasc_paths["processed"], compression="gzip")
        logger.info("Output file for ensemble models created")

    except Exception as error:
        logger.critical(
            "(!) Linkage service failed unexpectedly: %s", error, exc_info=True
        )
        raise typer.Exit(code=1)

    logger.info("[Linkage End]")
