"""
Nastavení logování pro projekt.
"""

import logging


def get_logger(name: str) -> logging.Logger:
    """
    Vrací nakonfigurovaný logger.
    """
    logger = logging.getLogger(name)
    if not logger.handlers:
        logger.setLevel(logging.INFO)

        # Console handler
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)

        # Formát logu
        formatter = logging.Formatter(
            '[%(asctime)s] [%(levelname)s] %(name)s: %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        ch.setFormatter(formatter)

        logger.addHandler(ch)

    return logger

logger = get_logger("IBKR_BOT")
