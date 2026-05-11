import json
import uuid
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from ibkr_trading_bot.core.services.model_training_service import training_profile_for_mode
from ibkr_trading_bot.model import train_models
from ibkr_trading_bot.model.feature_stability import (
    compute_feature_stability_score,
    filter_unstable_features,
)
from ibkr_trading_bot.model.train_models import train_and_evaluate_model


def _small_training_frame(n_rows: int = 96) -> pd.DataFrame:
    idx = pd.Series(range(n_rows), dtype=float)
    target = (((idx.astype(int) // 3) % 2) == 1).astype(int)
    return pd.DataFrame(
        {
            "timestamp": pd.date_range("2026-01-01", periods=n_rows, freq="5min", tz="UTC"),
            "f_signal": target + (0.15 * (idx / max(1.0, float(n_rows - 1)))),
            "f_aux": (idx % 11).astype(float) / 10.0,
            "f_trend": idx / max(1.0, float(n_rows - 1)),
            "target": target,
        }
    )


def _single_candidate_rf_grid() -> dict[str, list]:
    return {
        "clf__n_estimators": [40],
        "clf__max_depth": [4],
        "clf__min_samples_leaf": [1],
        "clf__n_jobs": [1],
    }


def _single_candidate_svm_grid() -> dict[str, list]:
    return {
        "clf__C": [1.0],
        "clf__gamma": ["scale"],
    }


def _workspace_tmp_dir(prefix: str) -> Path:
    path = Path(".codex_test_tmp") / f"{prefix}_{uuid.uuid4().hex}"
    path.mkdir(parents=True, exist_ok=True)
    return path


def _load_result_meta(result: dict[str, object]) -> dict[str, object]:
    output_path = Path(str(result["output_path"]))
    meta_path = output_path.with_name(f"{output_path.stem}_meta.json")
    return json.loads(meta_path.read_text(encoding="utf-8"))


def test_compute_feature_stability_score_handles_normal_and_edge_cases():
    scores = compute_feature_stability_score(
        {
            "stable": {"mean": 0.40, "std": 0.10, "min": 0.30, "max": 0.50, "folds_present": 3},
            "volatile": {"mean": 0.05, "std": 0.20, "min": 0.00, "max": 0.30, "folds_present": 3},
            "non_positive_mean": {"mean": 0.0, "std": 0.01, "min": 0.0, "max": 0.02, "folds_present": 3},
            "non_finite_std": {"mean": 0.20, "std": float("inf"), "min": 0.10, "max": 0.30, "folds_present": 3},
            "too_few_folds": {"mean": 0.20, "std": 0.01, "min": 0.19, "max": 0.21, "folds_present": 1},
        }
    )

    assert scores["stable"] == pytest.approx(0.75)
    assert scores["volatile"] == pytest.approx(0.0)
    assert scores["non_positive_mean"] == pytest.approx(0.0)
    assert scores["non_finite_std"] == pytest.approx(0.0)
    assert scores["too_few_folds"] == pytest.approx(0.0)


def test_filter_unstable_features_preserves_order_and_safe_fallback():
    feature_stability = {
        "feat_a": {"mean": 0.40, "std": 0.04, "min": 0.35, "max": 0.44, "folds_present": 3},
        "feat_b": {"mean": 0.30, "std": 0.20, "min": 0.05, "max": 0.45, "folds_present": 3},
    }

    kept = filter_unstable_features(
        feature_stability,
        ["missing_feat", "feat_b", "feat_a"],
        0.50,
    )
    fallback = filter_unstable_features(feature_stability, ["feat_b", "missing_feat"], 0.95)

    assert kept == ["feat_a"]
    assert fallback == ["feat_b", "missing_feat"]


def test_train_and_evaluate_model_writes_feature_stability_filter_meta(monkeypatch):
    out_dir = _workspace_tmp_dir("feature_stability_threshold_meta")
    monkeypatch.setattr(train_models, "_model_dir", lambda: out_dir)

    def _fake_collect_fold_feature_importances_for_params(**kwargs):
        return [
            {"f_aux": 0.24, "f_trend": 0.76},
            {"f_aux": 0.26, "f_trend": 0.12},
            {"f_aux": 0.25, "f_trend": 0.63},
        ]

    monkeypatch.setattr(
        train_models,
        "_collect_fold_feature_importances_for_params",
        _fake_collect_fold_feature_importances_for_params,
    )

    result = train_and_evaluate_model(
        df=_small_training_frame(n_rows=120),
        estimator_name="rf",
        param_grid=_single_candidate_rf_grid(),
        n_splits=3,
        embargo=2,
        holdout_bars=12,
        mc_enabled=False,
        annualize_sharpe=True,
        feature_stability_threshold=0.60,
    )

    meta = _load_result_meta(result)

    assert meta["feature_stability_threshold"] == pytest.approx(0.60)
    assert meta["feature_stability_filter_applied"] is True
    assert meta["feature_stability_filter_fallback_reason"] is None
    assert meta["trained_features"] == ["f_aux"]
    assert meta["features_kept_by_stability"] == ["f_aux"]
    assert meta["features_removed_by_stability"] == ["f_trend"]
    assert set(meta["feature_stability"]) == {"f_aux", "f_trend"}
    assert set(meta["feature_stability_score"]) == {"f_aux", "f_trend"}
    assert meta["feature_stability_score"]["f_aux"] > 0.60
    assert meta["feature_stability_score"]["f_trend"] < 0.60


def test_train_and_evaluate_model_reverts_stability_filter_when_refit_fails(monkeypatch):
    out_dir = _workspace_tmp_dir("feature_stability_threshold_refit_revert")
    monkeypatch.setattr(train_models, "_model_dir", lambda: out_dir)

    def _fake_collect_fold_feature_importances_for_params(**kwargs):
        return [
            {"f_aux": 0.24, "f_trend": 0.76},
            {"f_aux": 0.26, "f_trend": 0.12},
            {"f_aux": 0.25, "f_trend": 0.63},
        ]

    original_fit_estimator = train_models._fit_estimator

    def _fit_estimator_fail_on_filtered_refit(estimator, X, y, sample_weight=None):
        columns = list(getattr(X, "columns", []))
        if columns == ["f_aux"]:
            raise RuntimeError("forced refit failure for filtered features")
        return original_fit_estimator(estimator, X, y, sample_weight=sample_weight)

    monkeypatch.setattr(
        train_models,
        "_collect_fold_feature_importances_for_params",
        _fake_collect_fold_feature_importances_for_params,
    )
    monkeypatch.setattr(
        train_models,
        "_fit_estimator",
        _fit_estimator_fail_on_filtered_refit,
    )

    result = train_and_evaluate_model(
        df=_small_training_frame(n_rows=120),
        estimator_name="rf",
        param_grid=_single_candidate_rf_grid(),
        n_splits=3,
        embargo=2,
        holdout_bars=12,
        mc_enabled=False,
        annualize_sharpe=True,
        feature_stability_threshold=0.60,
    )

    meta = _load_result_meta(result)

    assert meta["feature_stability_threshold"] == pytest.approx(0.60)
    assert meta["feature_stability_filter_applied"] is False
    assert meta["feature_stability_filter_fallback_reason"] == (
        "stability_filter_refit_failed_reverted_to_original_features"
    )
    assert meta["features_removed_by_stability"] == []
    assert meta["features_kept_by_stability"] == meta["trained_features"]
    assert set(meta["trained_features"]) == {"f_aux", "f_trend"}


def test_train_and_evaluate_model_unsupported_estimator_stability_filter_is_noop(monkeypatch):
    out_dir = _workspace_tmp_dir("feature_stability_threshold_unsupported")
    monkeypatch.setattr(train_models, "_model_dir", lambda: out_dir)

    result = train_and_evaluate_model(
        df=_small_training_frame(n_rows=120),
        estimator_name="svm",
        param_grid=_single_candidate_svm_grid(),
        n_splits=3,
        embargo=2,
        holdout_bars=12,
        mc_enabled=False,
        annualize_sharpe=True,
        feature_stability_threshold=0.50,
    )

    meta = _load_result_meta(result)

    assert meta["feature_stability"] == {}
    assert meta["feature_stability_threshold"] == pytest.approx(0.50)
    assert meta["feature_stability_filter_applied"] is False
    assert meta["feature_stability_filter_fallback_reason"] is not None
    assert meta["features_removed_by_stability"] == []
    assert meta["features_kept_by_stability"] == meta["trained_features"]
    assert set(meta["feature_stability_score"]) == set(meta["trained_features"])
    assert all(score == 0.0 for score in meta["feature_stability_score"].values())


def test_training_profile_for_mode_sets_feature_stability_thresholds():
    assert training_profile_for_mode("quick")["feature_stability_threshold"] is None


def test_predict_proba_suppresses_feature_name_warning():
    class _WarningEstimator:
        feature_names_in_ = np.array(["feat_a", "feat_b"], dtype=object)

        def predict_proba(self, X):
            warnings.warn(
                "X does not have valid feature names, but LGBMClassifier was fitted with feature names",
                UserWarning,
            )
            return np.array([[0.25, 0.75]], dtype=float)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        proba = train_models._predict_proba(_WarningEstimator(), np.array([[1.0, 2.0]]))

    assert proba is not None
    assert np.asarray(proba).tolist() == pytest.approx([0.75])
    assert not any("valid feature names" in str(w.message) for w in caught)
    assert training_profile_for_mode("standard")["feature_stability_threshold"] == pytest.approx(0.40)
    assert training_profile_for_mode("strict")["feature_stability_threshold"] == pytest.approx(0.50)


def test_predict_labels_for_metrics_suppresses_feature_name_warning():
    class _WarningEstimator:
        feature_names_in_ = np.array(["feat_a", "feat_b"], dtype=object)

        def predict_proba(self, X):
            warnings.warn(
                "X does not have valid feature names, but LGBMClassifier was fitted with feature names",
                UserWarning,
            )
            return np.array([[0.40, 0.60]], dtype=float)

    X = pd.DataFrame({"feat_a": [1.0], "feat_b": [2.0]})
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        y_pred, n_signals, proba = train_models._predict_labels_for_metrics(_WarningEstimator(), X)

    assert y_pred.tolist() == [1]
    assert n_signals == 1
    assert np.asarray(proba).tolist() == pytest.approx([0.60])
    assert not any("valid feature names" in str(w.message) for w in caught)


def test_training_profile_for_mode_raises_trade_floor_and_strict_top_n():
    assert training_profile_for_mode("standard")["quality_min_trades"] == 60
    assert training_profile_for_mode("strict")["quality_min_trades"] == 60
    assert training_profile_for_mode("strict")["candidate_top_n"] == 10


def test_training_profile_for_mode_supports_new_workflow_profiles():
    explore = training_profile_for_mode("explore")
    refine = training_profile_for_mode("refine")
    refresh = training_profile_for_mode("refresh")

    assert explore["profile_name"] == "explore"
    assert refine["profile_name"] == "refine"
    assert refresh["profile_name"] == "refresh"
    assert explore["quality_min_trades"] < refine["quality_min_trades"]
    assert refresh["max_param_candidates"] == 12
    assert refresh["candidate_chain_enabled"] is False
    assert refresh["quality_min_side_prediction_share"] == pytest.approx(0.05)
    assert refresh["quality_min_side_prediction_count"] == 10


def test_quality_gate_rejects_one_sided_holdout_predictions():
    holdout_metrics = {
        "per_class_3": {
            "-1": {"f1": 0.30, "recall": 0.25},
            "1": {"f1": 0.32, "recall": 0.28},
        },
        "f1_macro_3": 0.31,
        "profit_net": 120.0,
        "num_trades": 80,
        "num_trades_short": 40,
        "num_trades_long": 40,
    }
    baseline_metrics = {
        "per_class_3": {
            "-1": {"f1": 0.10},
            "1": {"f1": 0.10},
        },
        "profit_net": 0.0,
    }

    passed, reasons = train_models._quality_gate_vs_baseline_ternary(
        holdout_metrics,
        baseline_metrics,
        min_trades=20,
        y_pred=np.array(([-1] * 95) + ([1] * 5)),
        min_side_prediction_share=0.10,
        min_side_prediction_count=10,
    )

    assert passed is False
    assert "long_predictions_too_few(5<10)" in reasons
    assert "long_prediction_share_too_low(0.0500<0.1000)" in reasons


def test_quality_gate_allows_balanced_holdout_predictions_when_other_metrics_pass():
    holdout_metrics = {
        "per_class_3": {
            "-1": {"f1": 0.30, "recall": 0.25},
            "1": {"f1": 0.32, "recall": 0.28},
        },
        "f1_macro_3": 0.31,
        "profit_net": 120.0,
        "num_trades": 80,
        "num_trades_short": 40,
        "num_trades_long": 40,
    }
    baseline_metrics = {
        "per_class_3": {
            "-1": {"f1": 0.10},
            "1": {"f1": 0.10},
        },
        "profit_net": 0.0,
    }

    passed, reasons = train_models._quality_gate_vs_baseline_ternary(
        holdout_metrics,
        baseline_metrics,
        min_trades=20,
        y_pred=np.array(([-1] * 55) + ([1] * 45)),
        min_side_prediction_share=0.10,
        min_side_prediction_count=10,
    )

    assert passed is True
    assert reasons == []


def test_candidate_priority_score_penalizes_nearly_one_sided_predictions():
    base = {
        "cv_score": 0.45,
        "f1_macro_3": 0.22,
        "profit_net": 80.0,
        "sharpe": 0.15,
        "pf": 1.12,
        "rec_short": 0.18,
        "rec_long": 0.17,
        "n_dir_pred_mean": 100.0,
    }
    balanced = {
        **base,
        "n_short_pred_mean": 50.0,
        "n_long_pred_mean": 50.0,
    }
    skewed = {
        **base,
        "n_short_pred_mean": 95.0,
        "n_long_pred_mean": 5.0,
    }

    balanced_score = train_models._candidate_priority_score(balanced, "balanced")
    skewed_score = train_models._candidate_priority_score(skewed, "balanced")

    assert balanced_score > skewed_score


def test_build_holdout_chunk_diagnostics_summarizes_regime_slices(monkeypatch):
    monkeypatch.setattr(train_models, "HAS_CALC_METRICS", True)

    def _fake_calculate_metrics(*, y_true, y_pred, df, **kwargs):
        arr = np.asarray(y_pred, dtype=int)
        return {
            "profit_net": float(arr.sum()),
            "sharpe": float(arr.mean()) if arr.size > 0 else 0.0,
            "num_trades": int(np.sum(arr != 0)),
            "num_trades_short": int(np.sum(arr < 0)),
            "num_trades_long": int(np.sum(arr > 0)),
            "per_class_3": {
                "-1": {"f1": 0.10},
                "1": {"f1": 0.30},
            },
        }

    monkeypatch.setattr(train_models, "calculate_metrics", _fake_calculate_metrics)

    df_hold = pd.DataFrame(
        {
            "timestamp": pd.date_range("2026-01-01", periods=6, freq="5min", tz="UTC"),
            "close": [100, 101, 102, 103, 104, 105],
        }
    )
    y_true = np.array([-1, -1, 0, 0, 1, 1])
    y_pred = np.array([-1, -1, 0, 1, 1, 1])

    chunks = train_models._build_holdout_chunk_diagnostics(
        y_true,
        y_pred,
        df_hold,
        fee_per_trade=0.0,
        slippage_bps=0.0,
        annualize_sharpe=True,
        max_chunks=3,
    )

    assert len(chunks) == 3
    assert chunks[0]["start_timestamp"] == "2026-01-01T00:00:00+00:00"
    assert chunks[2]["end_timestamp"] == "2026-01-01T00:25:00+00:00"
    assert chunks[0]["prediction_balance"]["n_short"] == 2
    assert chunks[0]["prediction_balance"]["n_long"] == 0
    assert chunks[1]["prediction_balance"]["n_hold"] == 1
    assert chunks[1]["prediction_balance"]["n_long"] == 1
    assert chunks[2]["profit_net"] == 2.0
    assert chunks[2]["directional_f1"] == pytest.approx(0.20)
