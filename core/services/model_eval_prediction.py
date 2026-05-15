from __future__ import annotations

import numpy as np
import pandas as pd

from ibkr_trading_bot.core.services.signal_policy import (
    apply_confidence_entry_threshold,
    apply_exit_confidence_threshold,
    normalize_signal_array,
)


def apply_confidence_threshold_eval(raw_pred, confidence, threshold):
    return apply_confidence_entry_threshold(raw_pred, confidence, threshold)


def apply_exit_threshold_eval(y_pred: np.ndarray, confidence: np.ndarray, exit_thr: float) -> np.ndarray:
    return apply_exit_confidence_threshold(y_pred, confidence, exit_thr)


def normalize_pred_eval(arr) -> np.ndarray:
    return normalize_signal_array(arr)


def safe_close_series_eval(df: pd.DataFrame | None):
    if not isinstance(df, pd.DataFrame):
        return None
    for column in ["close", "Close", "CLOSE", "adj_close", "Adj Close"]:
        if column in df.columns:
            return pd.to_numeric(df[column], errors="coerce")
    return None