from src.dataset import get_dataset


def test_get_dataset_success(mocker):
    mock_pipeline = mocker.patch("src.dataset.pipeline")
    mock_dataset_config = mocker.patch("src.dataset.DatasetConfig")
    mock_global_config = mocker.patch("src.dataset.GlobalConfig")

    mock_configs = mocker.MagicMock()
    mock_configs.years = [2020, 2021]

    mock_dataset_config.return_value.datasets.items.return_value = [
        ("dataset_1", mock_configs)
    ]
    mock_global_config.return_value.raw_dir = "mock/path"

    get_dataset()

    mock_pipeline.assert_called_with(
        dataset="dataset_1",
        data_dir="mock/path",
        years_to_extract=[2020, 2021],
        merge_at_end=True,
    )
