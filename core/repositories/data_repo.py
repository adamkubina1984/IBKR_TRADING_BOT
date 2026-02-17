from pathlib import Path

import pandas as pd

from ibkr_trading_bot.utils.io_helpers import ensure_dir_exists, load_latest_data, save_data


class DataRepository:
    def __init__(self, logger):
        self.log = logger

    def save_raw(self, params, df: pd.DataFrame):
        sym = params.symbol
        tf = getattr(params, 'timeframe', getattr(params, 'bar_size', '1 min')).replace(' ', '')
        file = Path('data')/f"{sym}_{tf}.csv"
        ensure_dir_exists(file.parent)
        save_data(df, file)
        self.log.info(f"Saved raw data to {file}")

    def load_latest_data(self, symbol: str, timeframe: str):
        # Delegace na existující helper pro kompatibilitu
        return load_latest_data(symbol=symbol, timeframe=timeframe)
