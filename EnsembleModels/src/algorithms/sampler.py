import warnings

from imblearn.under_sampling import ClusterCentroids, EditedNearestNeighbours
from imblearn.over_sampling import SMOTE, ADASYN
from imblearn.combine import SMOTEENN

from .. import config as conf


def get_balance(sampler: str):
    """Returns the corresponding sampler instance."""

    if sampler == "cc":
        return ClusterCentroids(
            sampling_strategy=conf.SAMPLING_STRATEGY,
            random_state=conf.RANDOM_STATE,
        )
    elif sampler == "enn":
        return EditedNearestNeighbours(
            n_jobs=conf.N_JOBS,
        )
    elif sampler == "smote":
        return SMOTE(
            sampling_strategy=conf.SAMPLING_STRATEGY,
            random_state=conf.RANDOM_STATE,
        )
    elif sampler == "adasyn":
        return ADASYN(
            sampling_strategy=conf.SAMPLING_STRATEGY,
            random_state=conf.RANDOM_STATE,
        )
    elif sampler == "smoteenn":
        smote = get_balance("smote")
        enn = get_balance("enn")

        return SMOTEENN(
            sampling_strategy=conf.SAMPLING_STRATEGY,
            random_state=conf.RANDOM_STATE,
            smote=smote,
            enn=enn,
            n_jobs=conf.N_JOBS,
        )
    else:
        warn = f"Warning: No balance algorithm found for {sampler}"
        warnings.warn(warn)
        return None
