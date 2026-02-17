"""
Detekce svíčkových patternů pro obchodní data.
"""

import pandas as pd


def is_doji(row, threshold=0.1) -> bool:
    """Doji – open a close jsou téměř stejné"""
    return abs(row['close'] - row['open']) <= threshold * (row['high'] - row['low'])

def is_spinning_top(row, threshold=0.3) -> bool:
    """Spinning Top – krátké tělo, dlouhé knoty"""
    body = abs(row['close'] - row['open'])
    upper = row['high'] - max(row['close'], row['open'])
    lower = min(row['close'], row['open']) - row['low']
    total_range = row['high'] - row['low']

    if total_range == 0:
        return False  # ochrana proti dělení nulou

    return (body / total_range) < threshold and upper > body and lower > body

def is_marubozu(row, threshold=0.05) -> bool:
    """Marubozu – žádné knoty"""
    upper = row['high'] - max(row['close'], row['open'])
    lower = min(row['close'], row['open']) - row['low']
    return upper < threshold and lower < threshold

def is_pin_bar(row, body_threshold=0.25, wick_ratio=2.5) -> bool:
    """Pin Bar – dlouhý knot, malé tělo"""
    body = abs(row['close'] - row['open'])
    upper_wick = row['high'] - max(row['close'], row['open'])
    lower_wick = min(row['close'], row['open']) - row['low']
    total_range = row['high'] - row['low']

    # horní nebo dolní pin bar
    upper_condition = upper_wick > wick_ratio * body and body / total_range < body_threshold
    lower_condition = lower_wick > wick_ratio * body and body / total_range < body_threshold
    return upper_condition or lower_condition

def is_bullish_engulfing(prev, curr) -> bool:
    """Bullish Engulfing – aktuální tělo překryje předchozí medvědí"""
    return prev['close'] < prev['open'] and curr['close'] > curr['open'] and \
           curr['close'] > prev['open'] and curr['open'] < prev['close']

def is_bearish_engulfing(prev, curr) -> bool:
    """Bearish Engulfing – aktuální tělo překryje předchozí býčí"""
    return prev['close'] > prev['open'] and curr['close'] < curr['open'] and \
           curr['open'] > prev['close'] and curr['close'] < prev['open']


def detect_candlestick_patterns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Vypočítá svíčkové patterny a přidá je jako boolean sloupce.
    """
    df = df.copy()

    df['is_doji'] = df.apply(is_doji, axis=1)
    df['is_spinning_top'] = df.apply(is_spinning_top, axis=1)
    df['is_marubozu'] = df.apply(is_marubozu, axis=1)
    df['is_pin_bar'] = df.apply(is_pin_bar, axis=1)

    df['is_bullish_engulfing'] = False
    df['is_bearish_engulfing'] = False

    for i in range(1, len(df)):
        prev = df.iloc[i - 1]
        curr = df.iloc[i]
        df.at[df.index[i], 'is_bullish_engulfing'] = is_bullish_engulfing(prev, curr)
        df.at[df.index[i], 'is_bearish_engulfing'] = is_bearish_engulfing(prev, curr)

    return df
