from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from ibkr_trading_bot.core.services.auto_threshold_search import run_auto_threshold_search
from ibkr_trading_bot.core.services.evaluation_service import EvaluationService
from ibkr_trading_bot.core.services.model_eval_helpers import (
    _build_holdout_metrics_payload,
    _model_threshold_snapshot,
    _ranking_metrics_summary,
    finite_or_none,
    get_tab5_holdout_base_payload,
    get_tab5_holdout_optimized_payload,
    normalize_path,
    pick_metric,
    resolve_ternary_thresholds_eval,
    safe_float,
    utc_now_iso,
)
from ibkr_trading_bot.core.services.model_eval_features import (
    align_X_for_model_eval,
    coerce_features_for_model_eval,
)
from ibkr_trading_bot.core.services.model_eval_execution import (
    run_model_evaluation as _run_model_evaluation_impl,
)
from ibkr_trading_bot.core.services.model_eval_input import extract_X_y_eval
from ibkr_trading_bot.core.services.model_eval_loader import (
    LoadedPredictor,
    extract_predictor_from_object,
    load_predictor_with_merged_meta,
)
from ibkr_trading_bot.core.services.model_eval_payloads import (
    AutoThresholdPayload,
    EvaluationPayload,
)
from ibkr_trading_bot.core.services.model_eval_prediction import (
    safe_close_series_eval,
)
from ibkr_trading_bot.core.services.model_eval_prepared_data import (
    PreparedEvaluationData,
    load_prepared_evaluation_data,
    prepared_evaluation_cache_key,
)
from ibkr_trading_bot.core.services.model_eval_ranking import (
    TAB5_HOLDOUT_RANKING_BY_POLICY_KEY,
    TAB5_HOLDOUT_RANKING_KEY,
    TAB5_HOLDOUT_RANKING_SCHEMA_VERSION,
    build_tab5_holdout_ranking_payload,
    get_tab5_holdout_ranking,
    is_tab5_holdout_ranking_stale,
    ranking_status_from_error_message,
    set_tab5_holdout_ranking,
)
from ibkr_trading_bot.core.services.model_eval_search import (
    recalculate_metrics_from_predictions as _recalculate_metrics_from_predictions_impl,
    run_auto_threshold_search_from_context as _run_auto_threshold_search_from_context_impl,
)
from ibkr_trading_bot.core.services.model_eval_scope import apply_eval_scope
from ibkr_trading_bot.core.services.dataset_service import DatasetService
from ibkr_trading_bot.core.services.signal_policy import (
    DEFAULT_EXIT_POLICY,
    apply_entry_exit_thresholds,
    extract_directional_probabilities,
    resolve_exit_policy_setting,
    ternary_proba_to_signal,
)

__all__ = [
    "AutoThresholdPayload",
    "DatasetService",
    "EvaluationPayload",
    "EvaluationService",
    "LoadedPredictor",
    "PreparedEvaluationData",
    "TAB5_HOLDOUT_RANKING_BY_POLICY_KEY",
    "TAB5_HOLDOUT_RANKING_KEY",
    "TAB5_HOLDOUT_RANKING_SCHEMA_VERSION",
    "apply_eval_scope",
    "build_tab5_holdout_ranking_payload",
    "extract_X_y_eval",
    "extract_predictor_from_object",
    "finite_or_none",
    "get_tab5_holdout_base_payload",
    "get_tab5_holdout_optimized_payload",
    "get_tab5_holdout_ranking",
    "is_tab5_holdout_ranking_stale",
    "load_prepared_evaluation_data",
    "load_predictor_with_merged_meta",
    "normalize_path",
    "prepared_evaluation_cache_key",
    "ranking_status_from_error_message",
    "recalculate_metrics_from_predictions",
    "run_auto_threshold_search_from_context",
    "run_model_evaluation",
    "set_tab5_holdout_ranking",
    "utc_now_iso",
]

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
) -> EvaluationPayload:
    return _run_model_evaluation_impl(
        model=model,
        metadata=metadata,
        data_path=data_path,
        prepared_data=prepared_data,
        scope_mode=scope_mode,
        fee_per_trade=float(fee_per_trade),
        entry_threshold=float(entry_threshold),
        exit_threshold=float(exit_threshold),
        exit_policy=exit_policy,
        progress_cb=progress_cb,
        load_prepared_data=load_prepared_evaluation_data,
        apply_scope_fn=apply_eval_scope,
        coerce_features_fn=coerce_features_for_model_eval,
        align_features_fn=align_X_for_model_eval,
        resolve_thresholds_fn=resolve_ternary_thresholds_eval,
        extract_directional_probabilities_fn=extract_directional_probabilities,
        ternary_proba_to_signal_fn=ternary_proba_to_signal,
        resolve_exit_policy_fn=resolve_exit_policy_setting,
        recalc_metrics_fn=recalculate_metrics_from_predictions,
        safe_close_series_fn=safe_close_series_eval,
        default_exit_policy=DEFAULT_EXIT_POLICY,
    )


def recalculate_metrics_from_predictions(
    *,
    y_pred_raw,
    confidence_arr,
    y_true_current,
    df_current,
    fee_per_trade: float,
    entry_threshold: float,
    exit_threshold: float,
    exit_policy: str = DEFAULT_EXIT_POLICY,
    progress_cb=None,
) -> tuple[np.ndarray, dict[str, Any]]:
    return _recalculate_metrics_from_predictions_impl(
        y_pred_raw=y_pred_raw,
        confidence_arr=confidence_arr,
        y_true_current=y_true_current,
        df_current=df_current,
        fee_per_trade=float(fee_per_trade),
        entry_threshold=float(entry_threshold),
        exit_threshold=float(exit_threshold),
        exit_policy=exit_policy,
        progress_cb=progress_cb,
        apply_thresholds=apply_entry_exit_thresholds,
        evaluation_service_factory=lambda: EvaluationService(None, None, None),
    )


def run_auto_threshold_search_from_context(
    *,
    y_pred_raw,
    confidence_arr,
    y_true_current,
    df_current,
    fee_per_trade: float,
    current_entry: float,
    current_exit: float,
    exit_policy: str = DEFAULT_EXIT_POLICY,
    progress_cb=None,
    should_run=None,
) -> AutoThresholdPayload:
    return _run_auto_threshold_search_from_context_impl(
        y_pred_raw=y_pred_raw,
        confidence_arr=confidence_arr,
        y_true_current=y_true_current,
        df_current=df_current,
        fee_per_trade=float(fee_per_trade),
        current_entry=float(current_entry),
        current_exit=float(current_exit),
        exit_policy=exit_policy,
        progress_cb=progress_cb,
        should_run=should_run,
        apply_thresholds=apply_entry_exit_thresholds,
        evaluation_service_factory=lambda: EvaluationService(None, None, None),
        auto_threshold_runner=run_auto_threshold_search,
        pick_metric_fn=pick_metric,
        safe_float_fn=safe_float,
    )


