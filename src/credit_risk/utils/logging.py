"""Logging utilities."""

import logging
import sys


def setup_logging(level: str = "INFO", format: str | None = None) -> logging.Logger:
    if format is None:
        format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

    logger = logging.getLogger("credit_risk")
    logger.setLevel(getattr(logging, level))

    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setLevel(getattr(logging, level))
        formatter = logging.Formatter(format)
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    return logger


def get_logger(name: str = "credit_risk") -> logging.Logger:
    return logging.getLogger(name)
