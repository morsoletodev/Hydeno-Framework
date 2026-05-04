from typing import Any

import pandas as pd
from splink import block_on, Linker, SettingsCreator, DuckDBAPI
from splink.internals.blocking_rule_creator import BlockingRuleCreator
from splink.comparison_library import CustomComparison
import splink.comparison_level_library as cll

from ..config import DatasetConfig, LinkConfig, Dataset


def _dfs(link_columns: list[str], datasets: dict[str, Dataset]) -> list[pd.DataFrame]:
    return [pd.read_parquet(x.interim, columns=link_columns) for x in datasets.values()]


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


def create_linker() -> Linker:
    dc = DatasetConfig()
    lc = LinkConfig()

    return Linker(
        _dfs(lc.link_columns, dc.datasets),
        SettingsCreator(
            link_type=lc.link_type,
            comparisons=_comparisons(lc.comparisons),
            blocking_rules_to_generate_predictions=_blocking_rules(lc.blocking_rules),
        ),
        db_api=DuckDBAPI(),
    )
