from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd

from ibkr_trading_bot.core.services.model_eval_payloads import EvaluationPayload
from ibkr_trading_bot.core.services.model_eval_prepared_data import PreparedEvaluationData


def run_model_evaluation(
    *,
    model: Any,
    metadata: dict[str, Any] | None,
    data_path: str | Path | None = None,
    prepared_data: PreparedEvaluationData | None = None,
    scope_mode: str,
    fee_per_trade: float,
    entry_threshold: float,
    exit_threshold: float,
    exit_policy: str | None = None,
    progress_cb=None,
    load_prepared_data: Callable[..., PreparedEvaluationData],
    apply_scope_fn: Callable[..., tuple[Any, Any, Any, dict[str, Any]]],
    coerce_features_fn: Callable[..., Any],
    align_features_fn: Callable[..., pd.DataFrame],
    resolve_thresholds_fn: Callable[[dict[str, Any]], tuple[float, float, str]],
    extract_directional_probabilities_fn: Callable[..., tuple[np.ndarray, np.ndarray, np.ndarray]],
    ternary_proba_to_signal_fn: Callable[..., np.ndarray],
    resolve_exit_policy_fn: Callable[..., str],
    recalc_metrics_fn: Callable[..., tuple[np.ndarray, dict[str, Any]]],
    safe_close_series_fn: Callable[[pd.DataFrame | None], pd.Series | None],
    default_exit_policy: str,
) -> EvaluationPayload:
    if model is None:
        raise ValueError("Nejprve vyber model (.pkl).")
    if not hasattr(model, "predict"):
        raise AttributeError("Nacteny objekt nema metodu `.predict`.")

    prepared = (
        prepared_data
        if prepared_data is not None
        else load_prepared_data(data_path, metadata=metadata or {}, progress_cb=progress_cb)
    )
    X, y_true, df_for_metrics, scope_info = apply_scope_fn(
        prepared.X_full,
        prepared.y_true_full,
        prepared.df_for_metrics_full,
        scope_mode,
        metadata or {},
    )
    X = coerce_features_fn(X, model, metadata or {})
    if y_true is None:
        raise ValueError("Po priprave datasetu chybi cilova promenna (target/y).")

    if callable(progress_cb):
        progress_cb("Vyhodnoceni: pocitam predikce...")
    proba = None
    X_pred = align_features_fn(model, X)
    if hasattr(model, "predict_proba"):
        try:
            proba = model.predict_proba(X_pred)
        except Exception:
            proba = None

    thr_short, thr_long, threshold_source = resolve_thresholds_fn(metadata or {})
    if proba is None or proba.ndim != 2 or int(proba.shape[1]) != 3:
        raise ValueError("Tab 3 vyzaduje ternarni model s predict_proba (3 tridy: short/neutral/long).")

    prob_short, _prob_hold, prob_long = extract_directional_probabilities_fn(
        proba,
        getattr(model, "classes_", None),
        label_map=(metadata or {}).get("class_to_dir"),
    )
    y_pred_raw = ternary_proba_to_signal_fn(prob_short, prob_long, thr_short, thr_long)
    confidence_arr = np.max(proba, axis=1)
    policy_name = resolve_exit_policy_fn(
        exit_policy if exit_policy is not None else (metadata or {}),
        default=default_exit_policy,
    )
    y_pred_used, results = recalc_metrics_fn(
        y_pred_raw=np.asarray(y_pred_raw),
        confidence_arr=np.asarray(confidence_arr),
        y_true_current=np.asarray(y_true),
        df_current=df_for_metrics,
        fee_per_trade=float(fee_per_trade),
        entry_threshold=float(entry_threshold),
        exit_threshold=float(exit_threshold),
        exit_policy=policy_name,
        progress_cb=progress_cb,
    )

    return EvaluationPayload(
        X_current=X,
        y_true_current=np.asarray(y_true),
        df_current=df_for_metrics,
        close_series=safe_close_series_fn(df_for_metrics),
        confidence_arr=np.asarray(confidence_arr),
        y_pred_raw=np.asarray(y_pred_raw),
        y_pred_used=np.asarray(y_pred_used),
        results=results,
        scope_info=scope_info,
        threshold_source=threshold_source,
        thr_short=float(thr_short),
        thr_long=float(thr_long),
        entry_threshold=float(entry_threshold),
        exit_threshold=float(exit_threshold),
        exit_policy=policy_name,
    )