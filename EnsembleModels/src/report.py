import logging

import sklearn.metrics as metr

from . import config as conf

logger = logging.getLogger(__name__)


def fold_evaluation_report(y_true, y_pred):
    acc = metr.accuracy_score(y_true, y_pred)
    pre = metr.precision_score(y_true, y_pred)
    rec = metr.recall_score(y_true, y_pred)
    f1s = metr.f1_score(y_true, y_pred)

    print("-" * 80)
    print(f"Accuracy: {acc:.4} | Precision: {pre:.4}")
    print(f"Recall:   {rec:.4} | F1-score:  {f1s:.4}")

    # logger.info("Metrics found:")
    logger.info(
        f"{acc:.4f} | {pre:.4f} | {rec:.4f} | {f1s:.4f}",
    )

    return acc, pre, rec, f1s


def mean_evaluation_report(metrics):
    means = [sum(metric) / conf.N_SPLITS for metric in zip(*metrics)]

    print(f"\nMeans over {conf.N_SPLITS} folds:")
    print(f"Accuracy: {means[0]:.4} | Precision: {means[1]:.4}")
    print(f"Recall:   {means[2]:.4} | F1-score:  {means[3]:.4}")

    logger.info("Final metrics:")
    logger.info(
        f"Accuracy: {means[0]:.4f} | Precision: {means[1]:.4f} | Recall: {means[2]:.4f} | F1-score {means[3]:.4f}"
    )
