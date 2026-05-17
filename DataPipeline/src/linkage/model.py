from typing import Any

import pandas as pd
from splink import block_on, Linker, SettingsCreator, DuckDBAPI
from splink.internals.blocking_rule_creator import BlockingRuleCreator
from splink.comparison_library import CustomComparison
import splink.comparison_level_library as cll

from ..config import get_dataset_config, get_dataset_paths, get_link_config, Dataset


def _dfs(
    db_api: DuckDBAPI, link_columns: list[str], datasets: dict[str, Dataset]
) -> list[pd.DataFrame]:
    table_names = []
    for ds in datasets.values():
        view_name = f"{ds.file_name}_view"

        select_cols = ", ".join(f'"{col}"' for col in link_columns)

        safe_path = str(get_dataset_paths(ds)["interim"]).replace("'", "''")
        sql = f"CREATE VIEW {view_name} AS SELECT {select_cols} FROM read_parquet('{safe_path}')"
        db_api._con.execute(sql)
        table_names.append(view_name)

    return table_names


def _comparisons(comparisons: dict[str, list[Any]]) -> list[CustomComparison]:
    rets = []
    for col, thresholds in comparisons.items():
        column_name = f"{col.capitalize()} Threshold"
        levels = [
            cll.NullLevel(col),
            cll.ExactMatchLevel(col),
            *(
                cll.AbsoluteDifferenceLevel(col, difference_threshold=x)
                for x in (thresholds or [])
            ),
            cll.ElseLevel(),
        ]

        rets.append(
            CustomComparison(
                output_column_name=column_name,
                comparison_levels=levels,
            )
        )
    return rets


def _blocking_rules(blocking_rules: dict[str, list[str]]) -> list[BlockingRuleCreator]:
    return [block_on(*x) for x in blocking_rules.values()]


def create_linker(db_api: DuckDBAPI) -> Linker:
    dc = get_dataset_config()
    lc = get_link_config()

    return Linker(
        _dfs(db_api, lc.link_columns, dc.datasets),
        SettingsCreator(
            link_type=lc.link_type,
            comparisons=_comparisons(lc.comparisons),
            blocking_rules_to_generate_predictions=_blocking_rules(lc.blocking_rules),
        ),
        db_api=db_api,
    )
