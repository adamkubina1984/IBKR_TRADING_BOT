"""
Jednotkové testy pro výpočet technických indikátorů, patternů a rolling statistik.
"""

import pandas as pd
import numpy as np

from features.indicators import (
    calculate_ema, calculate_rsi, calculate_atr,
    calculate_macd, calculate_bollinger_bands,
    calculate_williams_r, calculate_stochastic
)

from features.candlestick_patterns import detect_candlestick_patterns
from features.rolling_stats import (
    calculate_rolling_stats,
    calculate_price_change,
    calculate_breakouts
)


def create_test_df(n=100):
    """Vytvoří jednoduchý testovací DataFrame s OHLC daty."""
    np.random.seed(0)
    data = {
        "datetime": pd.date_range("2023-01-01", periods=n, freq="1min"),
        "open": np.random.rand(n) * 100,
    }
    data["high"] = data["open"] + np.random.rand(n) * 2
    data["low"] = data["open"] - np.random.rand(n) * 2
    data["close"] = data["open"] + np.random.randn(n)
    data["volume"] = np.random.randint(100, 1000, size=n)

    df = pd.DataFrame(data)
    df.set_index("datetime", inplace=True)
    return df


def test_indicators():
    df = create_test_df()

    # EMA
    ema = calculate_ema(df, span=10)
    assert isinstance(ema, pd.Series)
    assert not ema.isna().all()

    # RSI
    rsi = calculate_rsi(df)
    assert isinstance(rsi, pd.Series)

    # ATR
    atr = calculate_atr(df)
    assert isinstance(atr, pd.Series)

    # MACD
    macd = calculate_macd(df)
    assert all(col in macd.columns for col in ["macd", "macd_signal", "macd_hist"])

    # Bollinger Bands
    bb = calculate_bollinger_bands(df)
    assert all(col in bb.columns for col in ["bb_upper", "bb_middle", "bb_lower"])

    # Williams %R
    wr = calculate_williams_r(df)
    assert isinstance(wr, pd.Series)

    # Stochastic
    stoch = calculate_stochastic(df)
    assert "stoch_k" in stoch.columns and "stoch_d" in stoch.columns


def test_patterns():
    df = create_test_df()
    df = detect_candlestick_patterns(df)

    for pattern in [
        "is_doji", "is_spinning_top", "is_marubozu",
        "is_pin_bar", "is_bullish_engulfing", "is_bearish_engulfing"
    ]:
        assert pattern in df.columns
        assert df[pattern].dtype == bool


def test_rolling_stats():
    df = create_test_df()

    # Rolling průměry
    df_stats = calculate_rolling_stats(df, windows=[5, 10])
    assert "roll_mean_5" in df_stats.columns
    assert "roll_std_10" in df_stats.columns

    # Změna ceny
    df_change = calculate_price_change(df)
    assert "price_change_1" in df_change.columns

    # Breakouty
    df_breakout = calculate_breakouts(df, window=10)
    assert "breakout_high" in df_breakout.columns
    assert df_breakout["breakout_high"].dtype == bool


if __name__ == "__main__":
    test_indicators()
    test_patterns()
    test_rolling_stats()
    print("✅ Všechny testy proběhly úspěšně.")
