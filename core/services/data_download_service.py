from dataclasses import dataclass

import pandas as pd

from ibkr_trading_bot.core.models.bars import IBKR_BAR_SIZE, RESAMPLE_RULE, Timeframe
from ibkr_trading_bot.utils.download_ibkr_data import download_data


@dataclass
class DataDownloadParams:
    symbol: str
    duration: str  # např. "5 D"
    timeframe: Timeframe = Timeframe.M1
    exchange: str = "COMEX"
    what_to_show: str = "TRADES"

class DataDownloadService:
    def __init__(self, data_repo, logger):
        self.repo = data_repo
        self.log = logger

    def download(self, p: DataDownloadParams) -> pd.DataFrame:
        bar_size = IBKR_BAR_SIZE[p.timeframe]
        try:
            df = download_data(symbol=p.symbol, duration=p.duration, bar_size=bar_size,
                               exchange=p.exchange, what_to_show=p.what_to_show)
        except Exception as e:
            self.log.warning(f"Native download failed ({bar_size}): {e}")
            if p.timeframe != Timeframe.M1:
                # fallback na 1m a resample
                df1 = download_data(symbol=p.symbol, duration=p.duration, bar_size=IBKR_BAR_SIZE[Timeframe.M1],
                                    exchange=p.exchange, what_to_show=p.what_to_show)
                rule = RESAMPLE_RULE[p.timeframe]
                df = self._resample(df1, rule)
            else:
                raise
        self.repo.save_raw(p, df)
        return df

    def _resample(self, df: pd.DataFrame, rule: str) -> pd.DataFrame:
        # Očekává sloupce: open, high, low, close, volume a datetime index
        ohlc = df.resample(rule, label='right', closed='right').agg({
            'open':'first','high':'max','low':'min','close':'last','volume':'sum'
        }).dropna()
        return ohlc
