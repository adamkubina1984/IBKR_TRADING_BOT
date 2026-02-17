"""
Výpočet rolling statistik pro obchodní data.
"""

import pandas as pd


def calculate_rolling_stats(df: pd.DataFrame, windows=[20, 50, 100]) -> pd.DataFrame:
    """
    Přidá rolling průměry, odchylky, minima, maxima pro zadaná okna.
    """
    df = df.copy()
    for window in windows:
        df[f'roll_mean_{window}'] = df['close'].rolling(window=window).mean()
        df[f'roll_std_{window}'] = df['close'].rolling(window=window).std()
        df[f'roll_min_{window}'] = df['close'].rolling(window=window).min()
        df[f'roll_max_{window}'] = df['close'].rolling(window=window).max()
    return df


def calculate_price_change(df: pd.DataFrame, periods=[1, 5, 10]) -> pd.DataFrame:
    """
    Přidá procentuální změnu ceny za daný počet period.
    """
    df = df.copy()
    for p in periods:
        df[f'price_change_{p}'] = df['close'].pct_change(periods=p)
    return df


def calculate_breakouts(df: pd.DataFrame, window: int = 20) -> pd.DataFrame:
    """
    Přidá boolean sloupce, pokud zavírací cena prorazila nad/pod poslední high/low.
    """
    df = df.copy()
    df['breakout_high'] = df['close'] > df['high'].rolling(window=window).max().shift(1)
    df['breakout_low'] = df['close'] < df['low'].rolling(window=window).min().shift(1)
    return df
