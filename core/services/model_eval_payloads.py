from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd


@dataclass
class EvaluationPayload:
    X_current: pd.DataFrame | np.ndarray
    y_true_current: np.ndarray
    df_current: pd.DataFrame
    close_series: pd.Series | None
    confidence_arr: np.ndarray
    y_pred_raw: np.ndarray
    y_pred_used: np.ndarray
    results: dict[str, Any]
    scope_info: dict[str, Any]
    threshold_source: str
    thr_short: float
    thr_long: float
    entry_threshold: float
    exit_threshold: float
    exit_policy: str


@dataclass
class AutoThresholdPayload:
    best_entry: float
    best_exit: float
    best_score: float
    best_metrics: dict[str, Any] | None