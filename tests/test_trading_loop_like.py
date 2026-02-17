# tests/test_trading_loop_like.py
# ---------------------------------------------------
# Popis:
# - Minimální test konzistence: počet obchodů z calculate_metrics odpovídá
#   počtu změn pozice (entry/exit) / 2 v jednoduchém scénáři.

import numpy as np
import pandas as pd

from ibkr_trading_bot.utils.metrics import calculate_metrics


def _df(prices):
    ts = pd.date_range("2024-01-01", periods=len(prices), freq="5min", tz="UTC")
    return pd.DataFrame({
        "timestamp": ts,
        "open": prices,
        "high": np.maximum(prices, np.array(prices)+0.2),
        "low":  np.minimum(prices, np.array(prices)-0.2),
        "close": prices,
        "volume": 1000
    })

def _positions_from_signals(y_pred):
    arr = np.asarray(y_pred, dtype=float)
    pos = (arr > 0).astype(int) - (arr < 0).astype(int)
    return pos

def test_num_trades_matches_position_flips_in_simple_case():
    prices = [100,101,102,101,100,101,99,100]
    df = _df(prices)
    y_pred = [0,1,1,0,-1,-1,0,0]  # long trade + short trade
    out = calculate_metrics(y_true=[0]*len(y_pred), y_pred=y_pred, df=df)

    # hrubý odhad počtu obchodů ze změn pozic
    pos = _positions_from_signals(y_pred)
    flips = (pos[1:] != pos[:-1]).sum()
    # každé "vstup/exit" = obchod; flip 1→-1 podobně = 2 změny
    approx_trades = flips // 2 + flips % 2  # zjednodušeně; v testu by mělo vyjít přesně 2

    assert out["num_trades"] == 2
    assert approx_trades == 2
