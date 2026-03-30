import warnings

from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier

from .. import config as conf


def get_ensemble(ensemble: str, **kwargs):
    """Returns the corresponding ensemble instance."""

    if ensemble == "rlc":
        return RandomForestClassifier(
            random_state=conf.RANDOM_STATE,
            n_jobs=conf.N_JOBS,
            **kwargs,
        )
    elif ensemble == "xgbc":
        return XGBClassifier(**kwargs)
    elif ensemble == "lgbc":
        return LGBMClassifier(
            verbosity=-1,
            **kwargs,
        )
    elif ensemble == "cbc":
        return CatBoostClassifier(
            logging_level="Silent",
            random_state=conf.RANDOM_STATE,
            **kwargs,
        )
    else:
        warn = f"Warning: No ensemble algorithm found for {ensemble}"
        warnings.warn(warn)
        return None
