from pathlib import Path

import pytest
import yaml

import src.config as conf


@pytest.fixture
def global_data():
    with open("config/global.yaml") as f:
        yield yaml.safe_load(f)


def test_globalconfig_file_loading(global_data):
    global_c = conf.GlobalConfig()

    assert global_c.base_ext == global_data["base_ext"]
    assert global_c.comp == global_data["comp"]


def test_globalconfig_assemble_data_folder_paths(global_data):
    global_c = conf.GlobalConfig()

    base_dir = Path(global_data["base_data_dir"]).resolve()

    assert global_c.raw_dir == (base_dir / global_data["raw_dir"])
    assert global_c.interim_dir == (base_dir / global_data["interim_dir"])
    assert global_c.processed_dir == (base_dir / global_data["processed_dir"])


def test_globalconfig_assemble_data_file_paths(global_data):
    global_c = conf.GlobalConfig()

    for file in global_data.keys():
        if "f_" in file:
            expected_filepath = Path(global_data[file]).resolve()

            print(global_c.f_log)

            assert expected_filepath == getattr(global_c, file)


@pytest.fixture
def dataset_data():
    with open("config/dataset.yaml") as f:
        yield yaml.safe_load(f)


def test_datasetconfig_file_loading(dataset_data):
    data_c = conf.DatasetConfig()

    assert data_c.pre_linkage_cols == dataset_data["pre_linkage_cols"]
    assert data_c.linkage_threshold == dataset_data["linkage_threshold"]
    assert data_c.linkage_columns == dataset_data["linkage_columns"]
    assert data_c.ensemble_columns == dataset_data["ensemble_columns"]


def test_datasetconfig_assemble_data_years(dataset_data):
    data_c = conf.DatasetConfig()
    ds = dataset_data["datasets"]

    for i in ds.keys():
        exp_years = list(
            range(
                ds[i]["start_year"],
                ds[i]["final_year"],
            )
        )

        assert data_c.datasets[i].years == exp_years


def test_datasetconfig_assemble_data_file_paths(dataset_data, global_data):
    data_c = conf.DatasetConfig()
    ds = dataset_data["datasets"]
    exp_ext = f".{global_data['base_ext']}.{global_data['comp']}"
    global_c = conf.GlobalConfig()

    for i in ds.keys():
        for folder in ["raw", "interim", "processed"]:
            exp_name = f"{ds[i]['dataset_filename']}{exp_ext}"
            exp_path = getattr(global_c, f"{folder}_dir") / exp_name

            assert getattr(data_c.datasets[i], folder) == exp_path
            assert getattr(data_c.datasets[i], folder).name == exp_name
