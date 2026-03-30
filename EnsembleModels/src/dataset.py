from typing import Literal

import pandas as pd

from . import config as conf


def get_dataset(
    drop_duplicates: bool = False,
    drop_na: bool = False,
    dataset_location: Literal["colab", "local"] = "local",
):
    if dataset_location == "colab":
        path = conf.DRIVE_PATH
    else:
        path = conf.LOCAL_PATH
    df = pd.read_parquet(path)

    if drop_duplicates:
        df = df.drop_duplicates(ignore_index=True)

    if drop_na:
        df = df.dropna(axis="index", ignore_index=True)

    X = df.drop(columns=["OBITO"])
    y = df["OBITO"]

    return X, y
