import logging

from . import config as conf
from .dataset import get_dataset
from .pipeline import run_pipeline, hpo_pipeline

format = "%(asctime)s %(name)-12s %(levelname)-8s %(message)s"
logging.basicConfig(filename=conf.LOG_FILE, level=logging.INFO, format=format)

__all__ = [
    "get_dataset",
    "run_pipeline",
    "hpo_pipeline",
]
