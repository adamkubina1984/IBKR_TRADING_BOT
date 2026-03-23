from __future__ import annotations

from ibkr_trading_bot.gui import tab_model_manager as _legacy

DEFAULT_MODEL_DIR = _legacy.DEFAULT_MODEL_DIR
ModelRecord = _legacy.ModelRecord
discover_models = _legacy.discover_models


class ModelManagerTab(_legacy.ModelManagerTab):
    def __init__(self, *args, **kwargs):
        _legacy.DEFAULT_MODEL_DIR = DEFAULT_MODEL_DIR
        super().__init__(*args, **kwargs)


__all__ = [
    "DEFAULT_MODEL_DIR",
    "ModelManagerTab",
    "ModelRecord",
    "discover_models",
]
