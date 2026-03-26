"""Logging utilities."""

import logging
import sys

from credit_risk.config.settings import LoggingConfig


def setup_logging(config: LoggingConfig | None = None) -> logging.Logger:
    if config is None:
        config = LoggingConfig()

    logger = logging.getLogger("credit_risk")
    logger.setLevel(getattr(logging, config.level))

    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setLevel(getattr(logging, config.level))
        formatter = logging.Formatter(config.format)
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    return logger


def get_logger(name: str = "credit_risk") -> logging.Logger:
    return logging.getLogger(name)
