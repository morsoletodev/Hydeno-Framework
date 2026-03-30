from pickle import dump
import logging

from imblearn.pipeline import Pipeline
from sklearn.model_selection import (
    StratifiedKFold,
    GridSearchCV,
    train_test_split,
    PredefinedSplit,
)
from sklearn.utils.class_weight import compute_sample_weight
import numpy as np

from . import config as conf
from .algorithms import get_balance, get_ensemble
from .report import fold_evaluation_report, mean_evaluation_report

logger = logging.getLogger(__name__)


def setup_pipeline(ensemble: str, sampler: str | None = None, **kwargs):
    if sampler == "sample_weight" or not sampler:
        return get_ensemble(ensemble)

    return Pipeline(
        [
            (sampler, get_balance(sampler)),
            (ensemble, get_ensemble(ensemble, **kwargs)),
        ]
    )


def run_pipeline(
    X,
    y,
    ensemble: str,
    sampler: str | None = None,
    save_model: bool = False,
    **kwargs,
):
    logger.info("[Pipeline Start]")

    skf = StratifiedKFold(
        n_splits=conf.N_SPLITS,
    )

    list_results = []
    logger.info(f"Pipeline consists of {ensemble=} and {sampler=}")
    logger.info(f"Accuracy | Precision | Recall | F1-score")

    for train_index, test_index in skf.split(X, y):
        X_train = X.iloc[train_index, :]
        X_test = X.iloc[test_index, :]
        y_train = y[train_index]
        y_test = y[test_index]

        pipeline = setup_pipeline(ensemble, sampler, **kwargs)

        if sampler == "sample_weight":
            sample_weight = compute_sample_weight(class_weight="balanced", y=y_train)
            pipeline.fit(X_train, y_train, sample_weight=sample_weight)
        else:
            pipeline.fit(X_train, y_train)

        if save_model:
            with open(conf.MODEL_PATH, "wb") as f:
                dump(pipeline, f, protocol=5)

        y_pred = pipeline.predict(X_test)

        list_results.append(fold_evaluation_report(y_test, y_pred))

    mean_evaluation_report(list_results)

    logger.info("[Pipeline End]")


def hpo_pipeline(
    X,
    y,
    ensemble: str,
    sampler: str,
    param_grid: dict,
    **kwargs,
):
    logger.info("[HPO Start]")
    logger.info(f"HPO consists of {ensemble=} and {sampler=}")

    pipeline = setup_pipeline(ensemble, sampler)
    updated_param_grid = {f"{ensemble}__{k}": v for k, v in param_grid.items()}

    indices = np.arange(len(X))
    train_idx, test_idx = train_test_split(
        indices, test_size=0.2, stratify=y, random_state=42
    )

    test_fold = np.zeros(len(X))
    test_fold[train_idx] = -1
    test_fold[test_idx] = 0

    ps = PredefinedSplit(test_fold)

    grid_search = GridSearchCV(
        pipeline,
        updated_param_grid,
        cv=ps,
        scoring=conf.SCORING,
        refit=conf.REFIT,
        verbose=2,
    )

    grid_search.fit(X, y, **kwargs)

    logger.info("[HPO End]")

    return grid_search.best_params_, grid_search.best_score_
