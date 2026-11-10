import logging
import os


def setup_logging() -> None:
    level = os.getenv("LOG_LEVEL", "INFO")
    logging.basicConfig(level=level, format="%(asctime)s %(levelname)-7s %(name)s - %(message)s")
    logging.getLogger("src").setLevel(level)  # also when Prefect owns the root handlers
