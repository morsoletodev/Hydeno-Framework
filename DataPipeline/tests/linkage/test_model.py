from io import BytesIO

import pandas as pd
from splink import Linker
from splink.comparison_library import CustomComparison
import splink.comparison_level_library as cll


from src.linkage.model import _dfs, _comparisons, _blocking_rules, create_linker


def test_dfs_success(mocker):
    """Validates size and dataframes within returned object"""
    # Param 1: link_columns
    link_columns = ["col_1", "col_2"]

    # Param 2: datasets
    df_1 = pd.DataFrame({"col_1": [0, 1], "col_2": [2, 3]})
    df_2 = pd.DataFrame({"col_1": [0, 2], "col_2": [1, 3]})

    datasets = {
        1: mocker.MagicMock(interim=BytesIO(df_1.to_parquet())),
        2: mocker.MagicMock(interim=BytesIO(df_2.to_parquet())),
    }

    # Function call
    result = _dfs(link_columns, datasets)

    # Assert
    assert len(result) == 2
    pd.testing.assert_frame_equal(result[0], df_1)
    pd.testing.assert_frame_equal(result[1], df_2)


def test_comparisons_success():
    """Validates param handling and comparisons created"""
    # Param 1: comparisons
    comparisons = {"col_1": None, "col_2": [1]}

    # Function call
    result = _comparisons(comparisons)

    # Expected Comparisons
    expected_comp = [
        CustomComparison(
            output_column_name="Col_1 Threshold",
            comparison_levels=[
                cll.NullLevel("col_1"),
                cll.ExactMatchLevel("col_1"),
                cll.ElseLevel(),
            ],
        ),
        CustomComparison(
            output_column_name="Col_2 Threshold",
            comparison_levels=[
                cll.NullLevel("col_2"),
                cll.ExactMatchLevel("col_2"),
                cll.AbsoluteDifferenceLevel("col_2", difference_threshold=1),
                cll.ElseLevel(),
            ],
        ),
    ]

    # Assert
    assert len(result) == 2
    for comp, exp_comp in zip(result, expected_comp):
        assert (
            comp.get_comparison("duckdb").human_readable_description
            == exp_comp.get_comparison("duckdb").human_readable_description
        )


def test_blocking_rules_success(mocker):
    """Validates function logic for handling parameters"""
    # Mocker
    mock_block_on = mocker.patch("src.linkage.model.block_on")

    # Param 1: blocking_rules
    blocking_rules = {"rule_1": ["col_1"]}

    # Function call
    _blocking_rules(blocking_rules)

    # Assert
    mock_block_on.assert_called_with("col_1")


def test_create_linker_success(mocker):
    """Validates Linker object with minimal configs"""
    # Mocker
    mock_link = mocker.patch("src.linkage.model.LinkConfig")
    mock_link.return_value = mocker.MagicMock(
        link_columns=["col_1", "col_2"],
        link_type="link_only",
        comparisons={"col_1": None, "col_2": [1]},
        blocking_rules={"rule_1": ["col_1"]},
    )

    mock_dataset = mocker.patch("src.linkage.model.DatasetConfig")
    df_1 = pd.DataFrame({"col_1": [0, 1], "col_2": [2, 3]})
    df_2 = pd.DataFrame({"col_1": [0, 2], "col_2": [1, 3]})
    mock_dataset.return_value = mocker.MagicMock(
        datasets={
            1: mocker.MagicMock(interim=BytesIO(df_1.to_parquet())),
            2: mocker.MagicMock(interim=BytesIO(df_2.to_parquet())),
        }
    )

    # Function call
    res = create_linker()

    # Assert
    assert isinstance(res, Linker)
