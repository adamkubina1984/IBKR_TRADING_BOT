import logging
import os

import pandas as pd
from dotenv import load_dotenv
from tvDatafeed import Interval, TvDatafeed

log = logging.getLogger(__name__)

def _pick_interval(*cands):
    for name in cands:
        if hasattr(Interval, name):
            return getattr(Interval, name)
    raise AttributeError(f"Interval not found in {cands}")

TF_MAP = {
    "5 min":  _pick_interval("in_5_minute", "in_5_min"),
    "15 min": _pick_interval("in_15_minute", "in_15_min"),
    "30 min": _pick_interval("in_30_minute", "in_30_min"),
    "1 hour": _pick_interval("in_1_hour", "in_60_minute"),
}

class TradingViewClient:
    def __init__(self, username: str | None = None, password: str | None = None):
        # pokusíme se načíst .env (projektový i nadřazený)
        load_dotenv(r"C:\Users\adamk\Můj disk\Trader\.env")
        load_dotenv()  # pro případ, že spouštíš z adresáře se .env

        user = username or os.getenv("TV_USERNAME")
        pwd  = password or os.getenv("TV_PASSWORD")

        try:
            if user and pwd:
                log.info("[TV] Using LOGIN (env credentials found).")
                self._tv = TvDatafeed(user, pwd)
            else:
                log.warning("[TV] NO-LOGIN mode (no credentials). Data may be limited.")
                self._tv = TvDatafeed()
        except Exception as e:
            log.error(f"[TV] Login failed ({e}). Falling back to NO-LOGIN.")
            self._tv = TvDatafeed()

    def get_history(self, symbol: str, exchange: str, timeframe_label: str, limit: int = 60) -> pd.DataFrame:
        interval = TF_MAP[timeframe_label]
        df = self._tv.get_hist(symbol=symbol, exchange=exchange, interval=interval, n_bars=int(limit))
        if df is None or df.empty:
            log.error("[TV] No data returned (check symbol/exchange or login).")
            return pd.DataFrame(columns=["time","open","high","low","close","volume"])
        df = df.reset_index().rename(columns={"datetime": "time"})
        df["time"] = pd.to_datetime(df["time"], utc=True, errors="coerce")  # ← DŮLEŽITÉ
        return df[["time","open","high","low","close","volume"]]
