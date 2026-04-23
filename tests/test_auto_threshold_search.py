import numpy as np
import pytest

from ibkr_trading_bot.core.services.auto_threshold_search import (
    is_better_auto_threshold_candidate,
    run_auto_threshold_search,
)
from ibkr_trading_bot.core.services.model_eval_service import run_auto_threshold_search_from_context


def _pick_metric(metrics, *keys: str):
    metrics = metrics or {}
    for key in keys:
        if key in metrics:
            return metrics[key]
    return None


def test_auto_threshold_search_is_deterministic():
    def evaluate_pair(entry: float, exit_thr: float):
        score = 100.0 - (((entry - 0.34) ** 2) + ((exit_thr - 0.34) ** 2)) * 1000.0
        metrics = {
            "max_dd": 50.0 + abs(entry - 0.34) + abs(exit_thr - 0.34),
            "trades": 100.0 - ((abs(entry - 0.34) + abs(exit_thr - 0.34)) * 10.0),
        }
        return score, metrics

    result = run_auto_threshold_search(
        current_entry=0.50,
        current_exit=0.50,
        evaluate_pair=evaluate_pair,
        pick_metric=_pick_metric,
    )

    assert result.best_entry == pytest.approx(0.34)
    assert result.best_exit == pytest.approx(0.34)
    assert result.best_score == pytest.approx(100.0)


def test_auto_threshold_tie_break_prefers_more_trades_when_score_dd_distance_tie():
    is_better = is_better_auto_threshold_candidate(
        cand_score=10.0,
        cand_metrics={"max_dd": 25.0, "trades": 9.0},
        cand_entry=0.20,
        cand_exit=0.21,
        best_score=10.0,
        best_metrics={"max_dd": 25.0, "trades": 4.0},
        best_entry=0.21,
        best_exit=0.20,
        current_entry=0.205,
        current_exit=0.205,
        pick_metric=_pick_metric,
    )

    assert is_better is True


def test_auto_threshold_search_refines_all_best_coarse_regions():
    def evaluate_pair(entry: float, exit_thr: float):
        is_left_coarse = np.isclose(entry, 0.20) and np.isclose(exit_thr, 0.20)
        is_right_coarse = np.isclose(entry, 0.80) and np.isclose(exit_thr, 0.80)
        is_right_fine_peak = np.isclose(entry, 0.83) and np.isclose(exit_thr, 0.83)
        if is_right_fine_peak:
            return 11.0, {"max_dd": 5.0, "trades": 8.0}
        if is_left_coarse or is_right_coarse:
            return 10.0, {"max_dd": 5.0, "trades": 5.0}
        return -100.0, {"max_dd": 50.0, "trades": 0.0}

    result = run_auto_threshold_search(
        current_entry=0.20,
        current_exit=0.20,
        evaluate_pair=evaluate_pair,
        pick_metric=_pick_metric,
    )

    assert result.best_entry == pytest.approx(0.83)
    assert result.best_exit == pytest.approx(0.83)
    assert result.best_score == pytest.approx(11.0)


def test_run_auto_threshold_search_from_context_accepts_trade_count_fallback_keys(monkeypatch):
    monkeypatch.setattr(
        "ibkr_trading_bot.core.services.model_eval_service.EvaluationService.calculate_metrics",
        lambda self, **kwargs: {
            "profit_net": 10.0,
            "max_drawdown": -5.0,
            "num_trades": 7,
        },
    )

    result = run_auto_threshold_search_from_context(
        y_pred_raw=np.array([1, 0, -1], dtype=int),
        confidence_arr=np.array([0.8, 0.7, 0.9], dtype=float),
        y_true_current=np.array([1, 0, -1], dtype=int),
        df_current=None,
        fee_per_trade=0.0,
        current_entry=0.30,
        current_exit=0.40,
        exit_policy="hold_until_opposite",
    )

    assert result.best_entry >= 0.0
    assert result.best_exit >= 0.0
