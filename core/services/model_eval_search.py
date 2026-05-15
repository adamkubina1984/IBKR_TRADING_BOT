from __future__ import annotations

from typing import Any, Callable

import numpy as np

from ibkr_trading_bot.core.services.model_eval_payloads import AutoThresholdPayload
from ibkr_trading_bot.core.services.signal_policy import DEFAULT_EXIT_POLICY


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
    apply_thresholds: Callable[..., np.ndarray],
    evaluation_service_factory: Callable[[], Any],
) -> tuple[np.ndarray, dict[str, Any]]:
    if callable(progress_cb):
        progress_cb("Prepocitavam metriky...")
    y_pred_used = apply_thresholds(
        y_pred_raw,
        confidence_arr,
        entry_threshold,
        exit_threshold,
        exit_policy=exit_policy,
    )

    results = evaluation_service_factory().calculate_metrics(
        y_true=np.asarray(y_true_current),
        y_pred=y_pred_used,
        df=df_current,
        fee_per_trade=float(fee_per_trade),
        slippage_bps=0.0,
        rolling_window=200,
        annualize_sharpe=False,
    )
    if not isinstance(results, dict) or not results:
        raise ValueError("Vypocet metrik vratil prazdny vysledek.")
    return np.asarray(y_pred_used), results


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
    apply_thresholds: Callable[..., np.ndarray],
    evaluation_service_factory: Callable[[], Any],
    auto_threshold_runner: Callable[..., Any],
    pick_metric_fn: Callable[..., Any],
    safe_float_fn: Callable[[Any], float | None],
) -> AutoThresholdPayload:
    def evaluate_pair(entry_thr: float, exit_thr: float) -> tuple[float, dict[str, Any]]:
        y = apply_thresholds(
            y_pred_raw,
            confidence_arr,
            float(entry_thr),
            float(exit_thr),
            exit_policy=exit_policy,
        )
        metrics = evaluation_service_factory().calculate_metrics(
            y_true=np.asarray(y_true_current),
            y_pred=y,
            df=df_current,
            fee_per_trade=float(fee_per_trade),
            slippage_bps=0.0,
            rolling_window=200,
            annualize_sharpe=False,
        )
        profit = pick_metric_fn(metrics, "profit_net", "profit_gross", "profit")
        score = safe_float_fn(profit)
        return (score if score is not None else float("-inf")), metrics

    def pick_metric_for_search(metrics: dict[str, Any] | None, *metric_names: str):
        metric_name_set = {str(name) for name in metric_names}
        if "max_dd" in metric_name_set:
            return pick_metric_fn(metrics, "max_dd", "max_drawdown_net", "max_drawdown")
        if metric_name_set.intersection({"trades", "num_trades"}):
            return pick_metric_fn(metrics, "num_trades", "trades")
        raise KeyError(", ".join(str(name) for name in metric_names))

    result = auto_threshold_runner(
        current_entry=float(current_entry),
        current_exit=float(current_exit),
        evaluate_pair=evaluate_pair,
        pick_metric=pick_metric_for_search,
        progress_cb=progress_cb,
        should_run=should_run,
    )
    return AutoThresholdPayload(
        best_entry=float(result.best_entry),
        best_exit=float(result.best_exit),
        best_score=float(result.best_score),
        best_metrics=result.best_metrics,
    )