import logging

from .cli import app
from . import config as conf

logger = logging.getLogger(__name__)


if __name__ == "__main__":
    """Program entrance."""

    # logging setup
    format = "%(asctime)s %(name)-12s %(levelname)-8s %(message)s"
    logging.basicConfig(filename=conf.LOG_FILE, level=logging.INFO, format=format)

    # data/ folder creation
    conf.create_folders()

    # app execution
    app()
