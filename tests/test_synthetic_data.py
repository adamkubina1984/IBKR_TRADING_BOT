# tests/test_synthetic_data.py
"""
Test syntetických OHLC dat:
- tvar sloupců,
- žádné NaN,
- high >= max(open, close), low <= min(open, close),
- délka = n_samples.
"""

import pandas as pd

from data.generate_synthetic import generate_synthetic_data


def test_synthetic_ohlc_shape_and_consistency():
    n = 1000
    df = generate_synthetic_data(n_samples=n, noise_level=0.06)

    # 1) sloupce a délka
    expected_cols = ["timestamp", "open", "high", "low", "close", "volume"]
    assert list(df.columns) == expected_cols
    assert len(df) == n

    # 2) datové typy / NaN
    assert not df.isna().any().any()
    for col in ["open", "high", "low", "close"]:
        assert pd.api.types.is_float_dtype(df[col])
    assert pd.api.types.is_integer_dtype(df["volume"])

    # 3) ohlc konzistence
    assert (df["high"] >= df[["open", "close"]].max(axis=1)).all()
    assert (df["low"]  <= df[["open", "close"]].min(axis=1)).all()
