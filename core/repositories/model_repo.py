from pathlib import Path

import joblib

from ibkr_trading_bot.utils.io_helpers import ensure_dir_exists


class ModelRepository:
    def __init__(self, logger):
        self.log = logger
        self.base = Path('model')

    def save(self, model, symbol: str = 'GC', timeframe: str = '1 min', name: str = 'best_model'):
        fname = f"{name}_{symbol}_{timeframe.replace(' ', '')}.joblib"
        ensure_dir_exists(self.base)
        path = self.base/fname
        joblib.dump(model, path)
        self.log.info(f"Saved model {path}")
        return path

    def load_latest(self, symbol: str = 'GC', timeframe: str = '1 min'):
        # Nejjednodušší verze: hledat best_model_* a vrátit poslední
        pattern = f"best_model_{symbol}_{timeframe.replace(' ', '')}.joblib"
        path = self.base/pattern
        if path.exists():
            return joblib.load(path)
        raise FileNotFoundError(f"Model not found: {pattern}")
