# ibkr_trading_bot/labels/triple_barrier.py
from __future__ import annotations

import numpy as np
import pandas as pd


def make_triple_barrier_labels(
    df: pd.DataFrame,
    horizon: int = 12,
    take_profit_bps: float = 60.0,
    stop_loss_bps: float = 40.0,
    fee_per_trade: float = 0.0,
    slippage_bps: float = 0.0,
    price_col: str = "close",
) -> pd.Series:
    """
    Jednoduchá triple-barrier/horizon implementace:
      - sleduje max. horizon barů dopředu
      - pokud je dřív dosažen TP než SL -> 1, opačně -> 0
      - jinak rozhodne sign (konec - vstup) po zohlednění nákladů

    Výstup: Series binární (1 = long, 0 = short) zarovnaná s df.index.
    """
    if price_col not in df.columns:
        raise ValueError(f"Sloupec '{price_col}' v datech chybí.")

    close = df[price_col].astype(float).to_numpy()
    n = len(close)
    y = np.zeros(n, dtype=int)

    # náklady (zjednodušeně)
    roundtrip_cost = (slippage_bps / 1e4) * 2  # vstup + výstup
    fee_cost = 0.0 if fee_per_trade == 0 else (fee_per_trade / np.maximum(close, 1e-12))

    for i in range(n):
        j_end = min(i + horizon, n - 1)
        if j_end <= i:
            y[i] = 0
            continue

        entry = close[i]
        tp = entry * (1.0 + take_profit_bps / 1e4)
        sl = entry * (1.0 - stop_loss_bps / 1e4)

        path = close[i + 1 : j_end + 1]
        if path.size == 0:
            y[i] = 0
            continue

        hit_tp = np.any(path >= tp)
        hit_sl = np.any(path <= sl)

        if hit_tp and not hit_sl:
            y[i] = 1
        elif hit_sl and not hit_tp:
            y[i] = 0
        else:
            # fallback: porovnej konec s náklady
            pnl = (close[j_end] - entry) / max(entry, 1e-12)
            pnl -= roundtrip_cost
            # fee_cost může být v penězích; pro jednoduchost jej vyjádříme v relativu
            if isinstance(fee_cost, np.ndarray):
                pnl -= float(fee_cost[i])
            else:
                pnl -= fee_cost
            y[i] = 1 if pnl > 0 else 0

    return pd.Series(y, index=df.index, name="target")


def make_triple_barrier_labels_ternary(
    df: pd.DataFrame,
    horizon: int = 12,
    take_profit_bps: float = 60.0,
    stop_loss_bps: float = 40.0,
    price_col: str = "close",
) -> pd.Series:
    """
    Ternární triple‑barrier labels:
      - 1  => TP hit (long)
      - -1 => SL hit (short)
      - 0  => neither within horizon (flat)

    Výstup: Series s indexem shodným s `df.index`.
    """
    if price_col not in df.columns:
        raise ValueError(f"Sloupec '{price_col}' v datech chybí.")

    close = df[price_col].astype(float).to_numpy()
    n = len(close)
    y = np.zeros(n, dtype=int)

    for i in range(n):
        j_end = min(i + horizon, n - 1)
        if j_end <= i:
            y[i] = 0
            continue

        entry = close[i]
        tp = entry * (1.0 + take_profit_bps / 1e4)
        sl = entry * (1.0 - stop_loss_bps / 1e4)

        path = close[i + 1 : j_end + 1]
        if path.size == 0:
            y[i] = 0
            continue

        # determine first hit (if any)
        hit_tp_idx = None
        hit_sl_idx = None
        for k, p in enumerate(path, start=1):
            if hit_tp_idx is None and p >= tp:
                hit_tp_idx = k
            if hit_sl_idx is None and p <= sl:
                hit_sl_idx = k
            if hit_tp_idx is not None or hit_sl_idx is not None:
                break

        if hit_tp_idx is not None and (hit_sl_idx is None or hit_tp_idx <= hit_sl_idx):
            y[i] = 1
        elif hit_sl_idx is not None and (hit_tp_idx is None or hit_sl_idx < hit_tp_idx):
            y[i] = -1
        else:
            y[i] = 0

    return pd.Series(y, index=df.index, name="target")
