import logging

from .cli import app
from .config import create_folders, GlobalConfig

logger = logging.getLogger(__name__)


if __name__ == "__main__":
    """Program entrance."""

    create_folders()

    format = "%(asctime)s %(name)-12s %(levelname)-8s %(message)s"
    logging.basicConfig(
        filename=GlobalConfig().f_log, level=logging.INFO, format=format
    )

    app()
