# ibkr_trading_bot/core/utils/logging_setup.py
from __future__ import annotations

import logging

__all__ = ["get_logger"]

def get_logger(name: str = "ibkr") -> logging.Logger:
    """Return a configured logger.
    - Safe to call multiple times; handlers won't duplicate.
    - Defaults to INFO level and simple console formatting.
    """
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter("[%(levelname)s] %(message)s"))
        logger.addHandler(handler)
    if logger.level == logging.NOTSET:
        logger.setLevel(logging.INFO)
    return logger
