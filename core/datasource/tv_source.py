import os
import time
from dataclasses import dataclass

import pandas as pd

try:
    from tvDatafeed import Interval, TvDatafeed
except Exception:
    TvDatafeed = None
    Interval = None

# Map GUI bar-size labels to tvDatafeed Interval attributes
INTERVAL_MAP = {
    '1 min': 'in_1_minute',
    '5 mins': 'in_5_minute',
    '15 mins': 'in_15_minute',
    '30 mins': 'in_30_minute',
    '1 hour': 'in_1_hour',
}

def _load_tv_credentials():
    # 1) environment variables
    u = os.getenv("TV_USER", None)
    p = os.getenv("TV_PASS", None)
    if u and p:
        return u, p
    # 2) config_tv.yaml in project root (sibling of 'ibkr_trading_bot')
    #    format:
    #    tv:
    #      username: "name"
    #      password: "pass"
    try:
        import yaml
        here = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
        cfg_path = os.path.join(here, "config_tv.yaml")
        if os.path.exists(cfg_path):
            with open(cfg_path, encoding="utf-8") as f:
                data = yaml.safe_load(f) or {}
                tv = (data.get("tv") or {})
                u = tv.get("username") or None
                p = tv.get("password") or None
                if u and p:
                    return u, p
    except Exception:
        pass
    return None, None

@dataclass
class TVConfig:
    username: str | None = None
    password: str | None = None

class TradingViewSource:
    def __init__(self, cfg: TVConfig):
        if TvDatafeed is None:
            raise ImportError("tvDatafeed is not installed. Install fork from GitHub (see README_TV.txt).")
        # Try explicit cfg first, then env/config file
        user = cfg.username
        pwd  = cfg.password
        if not user or not pwd:
            eu, ep = _load_tv_credentials()
            user = user or eu
            pwd  = pwd or ep
        # Use positional args (some forks don't support keywords)
        self.tv = TvDatafeed(user, pwd)

    def _interval(self, bar_size: str):
        key = INTERVAL_MAP.get(bar_size, 'in_1_hour')
        try:
            return getattr(Interval, key)
        except Exception:
            return getattr(Interval, 'in_15_minute')

    def _get_hist_retry(self, symbol, exchange, interval, n_bars, retries=3, delay=3):
        last_err = None
        for _ in range(retries):
            try:
                df = self.tv.get_hist(symbol=symbol, exchange=exchange, interval=interval, n_bars=n_bars)
                if df is not None and len(df) > 0:
                    return df
            except Exception as e:
                last_err = e
            time.sleep(delay)
        if last_err:
            raise last_err
        return None

    def get_hist(self, symbol: str, exchange: str, bar_size: str, n_bars: int = 1000) -> pd.DataFrame:
        df = self._get_hist_retry(symbol, exchange, self._interval(bar_size), n_bars)
        if df is None or len(df) == 0:
            # As a fallback, try with GC1!/COMEX if user typed different exchange accidentally
            if symbol == "GOLD" and exchange != "TVC":
                df = self._get_hist_retry("GOLD!", "TVC", self._interval(bar_size), n_bars)
        if df is None or len(df) == 0:
            return pd.DataFrame(columns=['datetime','open','high','low','close','volume'])
        df = df.reset_index().rename(columns={'index':'datetime'})
        df = df[['datetime','open','high','low','close','volume']].copy()
        # -> převedu na UTC a odstraním tzinfo (naivní UTC), aby se dál už nelokalizovalo podruhé
        df['datetime'] = pd.to_datetime(df['datetime'], utc=True).dt.tz_convert(None)
        return df

    def get_last_closed(self, symbol: str, exchange: str, bar_size: str):
        df = self.get_hist(symbol, exchange, bar_size, n_bars=2)
        if len(df) < 2:
            return None
        row = df.iloc[-2]
        return {
            'time': row['datetime'].to_pydatetime(),
            'open': float(row['open']),
            'high': float(row['high']),
            'low':  float(row['low']),
            'close': float(row['close']),
            'volume': int(row['volume'] or 0),
        }
