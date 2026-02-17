# features/indicators.py

"""
Výpočet technických indikátorů pro obchodní data.
"""

import numpy as np
import pandas as pd


def calculate_ema(df, span: int, column: str = "close") -> pd.Series:
    """Exponenciální klouzavý průměr (EMA) – s konverzí na numeriku."""
    s = pd.to_numeric(df[column], errors="coerce")
    ema = s.ewm(span=span, adjust=False).mean()
    return pd.Series(ema, index=df.index, name=f"ema_{span}")

def calculate_rsi(df, period: int = 14, column: str = "close") -> pd.Series:
    """Relative Strength Index (RSI) – Wilderovo vyhlazení, robustní proti dělení nulou."""
    delta = df[column].diff()
    gain = delta.clip(lower=0)
    loss = (-delta.clip(upper=0))

    # Wilder: exponenciální klouzavý průměr s alpha=1/period (ekvivalent RMA)
    roll_up = gain.ewm(alpha=1/period, adjust=False, min_periods=period).mean()
    roll_down = loss.ewm(alpha=1/period, adjust=False, min_periods=period).mean()

    rs = roll_up / roll_down.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    return pd.Series(rsi, index=df.index, name="rsi")

def calculate_atr(df, period: int = 14) -> pd.Series:
    """Average True Range (ATR)"""
    high_low = df['high'] - df['low']
    high_close = np.abs(df['high'] - df['close'].shift())
    low_close = np.abs(df['low'] - df['close'].shift())

    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    atr = tr.rolling(window=period, min_periods=period).mean()
    return atr

def calculate_macd(df, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.DataFrame:
    """MACD a signální linie"""
    ema_fast = calculate_ema(df, fast)
    ema_slow = calculate_ema(df, slow)
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    histogram = macd_line - signal_line

    return pd.DataFrame({
        "macd": macd_line,
        "signal": signal_line,
        "hist": histogram
    }, index=df.index)

def calculate_bollinger_bands(df, period: int = 20, std_dev: float = 2.0, column: str = "close") -> pd.DataFrame:
    """Bollingerova pásma"""
    sma = df[column].rolling(window=period, min_periods=period).mean()
    std = df[column].rolling(window=period, min_periods=period).std()
    upper_band = sma + std_dev * std
    lower_band = sma - std_dev * std

    return pd.DataFrame({
        "bb_upper": upper_band,
        "bb_middle": sma,
        "bb_lower": lower_band
    }, index=df.index)

def calculate_williams_r(df, period: int = 14) -> pd.Series:
    """Williams %R – robustní proti dělení nulou, min_periods=period."""
    high_max = df['high'].rolling(window=period, min_periods=period).max()
    low_min  = df['low'].rolling(window=period, min_periods=period).min()
    denom = (high_max - low_min).replace(0, np.nan)
    wr = -100 * (high_max - df['close']) / denom
    return pd.Series(wr, index=df.index, name="williams_r")

def calculate_stochastic(df, k_period: int = 14, d_period: int = 3) -> pd.DataFrame:
    """Stochastic Oscillator (%K a %D) – robustní proti dělení nulou, min_periods."""
    low_min  = df['low'].rolling(window=k_period, min_periods=k_period).min()
    high_max = df['high'].rolling(window=k_period, min_periods=k_period).max()
    denom = (high_max - low_min).replace(0, np.nan)
    percent_k = 100 * (df['close'] - low_min) / denom
    percent_k = percent_k.clip(lower=0, upper=100)
    percent_d = percent_k.rolling(window=d_period, min_periods=d_period).mean()

    return pd.DataFrame({
        "stoch_k": percent_k,
        "stoch_d": percent_d
    }, index=df.index)

