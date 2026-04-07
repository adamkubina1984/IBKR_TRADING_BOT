import json
import uuid
from pathlib import Path

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
    assert training_profile_for_mode("standard")["feature_stability_threshold"] == pytest.approx(0.40)
    assert training_profile_for_mode("strict")["feature_stability_threshold"] == pytest.approx(0.50)
