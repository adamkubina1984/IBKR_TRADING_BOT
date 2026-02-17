# tests/test_features.py
# ---------------------------------------------------
# Popis:
# - Ověření, že feature_engineering normalizuje aliasy sloupců (Adj Close -> close)
#   a že compute_all_features vrátí požadované featury bez NaN v klíčových sloupcích.

import numpy as np
import pandas as pd

from ibkr_trading_bot.features.feature_engineering import _normalize_ohlcv, compute_all_features


def test_alias_mapping_adj_close_to_close():
    df = pd.DataFrame({
        "timestamp": pd.date_range("2024-01-01", periods=30, freq="5min", tz="UTC"),
        "Open": np.linspace(100, 110, 30),
        "High": np.linspace(101, 111, 30),
        "Low":  np.linspace( 99, 109, 30),
        "Adj Close": np.linspace(100, 110, 30),
        "Volume": 1000,
    })
    norm = _normalize_ohlcv(df)
    for c in ["open", "high", "low", "close", "volume"]:
        assert c in norm.columns

def test_compute_all_features_outputs_core_cols_without_nan():
    n = 200
    df = pd.DataFrame({
        "timestamp": pd.date_range("2024-01-01", periods=n, freq="5min", tz="UTC"),
        "open": 100 + np.cumsum(np.random.randn(n))*0.1,
        "high": 100 + np.cumsum(np.random.randn(n))*0.1 + 0.2,
        "low":  100 + np.cumsum(np.random.randn(n))*0.1 - 0.2,
        "close":100 + np.cumsum(np.random.randn(n))*0.1,
        "volume": 1000,
    })
    df = df.sort_values("timestamp").set_index("timestamp")
    feats = compute_all_features(df)
    # klíčové sloupce, které se dropují na začátku kvůli oknům – vrátil by se čistý DF:
    for c in ["rsi", "atr", "macd_macd", "macd_signal", "macd_hist"]:
        assert c in feats.columns
        assert not feats[c].isna().any()
