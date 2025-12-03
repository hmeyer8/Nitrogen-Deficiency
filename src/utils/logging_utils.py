import logging
from datetime import datetime
from pathlib import Path

from src.config import INTERIM_DIR


def setup_logger(name="nitrogen-pipeline"):
    log_path = INTERIM_DIR / f"{name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    formatter = logging.Formatter(
        "%(asctime)s — %(levelname)s — %(message)s"
    )

    file_handler = logging.FileHandler(log_path)
    file_handler.setFormatter(formatter)

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)

    return logger
