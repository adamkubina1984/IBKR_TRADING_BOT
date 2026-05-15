# tests/test_training_cli.py
import json
import logging
import os
import sys
import uuid
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import sklearn

from ibkr_trading_bot.data.generate_synthetic import generate_synthetic_data
from ibkr_trading_bot.features.feature_engineering import compute_all_features
from ibkr_trading_bot.model import train_models
from ibkr_trading_bot.model.evaluate_models import _predict_with_thresholds, evaluate_model_once
from ibkr_trading_bot.model.train_models import train_and_evaluate_model, train_simple_model


def _small_training_frame(n_rows: int = 96) -> pd.DataFrame:
    idx = np.arange(n_rows, dtype=float)
    target = (((idx.astype(int) // 3) % 2) == 1).astype(int)
    return pd.DataFrame(
        {
            "timestamp": pd.date_range("2026-01-01", periods=n_rows, freq="5min", tz="UTC"),
            "f_signal": target + (0.15 * np.sin(idx / 3.0)),
            "f_aux": np.cos(idx / 5.0),
            "f_trend": idx / max(1.0, float(n_rows - 1)),
            "target": target,
        }
    )


def _small_ternary_training_frame(n_rows: int = 240) -> pd.DataFrame:
    idx = np.arange(n_rows, dtype=float)
    target = ((idx.astype(int) % 3) - 1).astype(int)
    short_signal = (target == -1).astype(float)
    long_signal = (target == 1).astype(float)
    return pd.DataFrame(
        {
            "timestamp": pd.date_range("2026-01-01", periods=n_rows, freq="5min", tz="UTC"),
            "f_short": short_signal + (0.10 * np.sin(idx / 4.0)),
            "f_long": long_signal + (0.10 * np.cos(idx / 5.0)),
            "f_trend": idx / max(1.0, float(n_rows - 1)),
            "target": target,
        }
    )


def _single_candidate_grid() -> dict[str, list]:
    return {
        "clf__C": [1.0],
        "clf__gamma": ["scale"],
    }


def _single_candidate_rf_grid() -> dict[str, list]:
    return {
        "clf__n_estimators": [40],
        "clf__max_depth": [4],
        "clf__min_samples_leaf": [1],
        "clf__n_jobs": [1],
    }


def _multi_candidate_rf_grid() -> dict[str, list]:
    return {
        "clf__n_estimators": [40],
        "clf__max_depth": [2, 4],
        "clf__min_samples_leaf": [1, 2],
        "clf__n_jobs": [1],
    }


def _workspace_tmp_dir(prefix: str) -> Path:
    path = Path(".codex_test_tmp") / f"{prefix}_{uuid.uuid4().hex}"
    path.mkdir(parents=True, exist_ok=True)
    return path


def _load_result_meta(result: dict[str, object]) -> tuple[Path, dict[str, object], dict[str, object]]:
    output_path = Path(str(result["output_path"]))
    meta_path = output_path.with_name(f"{output_path.stem}_meta.json")
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    search_plan = result["search_plan"]
    return output_path, meta, search_plan


def _small_hgbt_grid() -> dict[str, list]:
    return {
        "clf__max_depth": [3, 4],
        "clf__learning_rate": [0.03, 0.06],
        "clf__max_iter": [120],
        "clf__l2_regularization": [0.1, 0.5],
    }


def _small_lgb_grid() -> dict[str, list]:
    return {
        "clf__n_estimators": [80, 120],
        "clf__max_depth": [4],
        "clf__learning_rate": [0.03, 0.05],
        "clf__num_leaves": [15, 31],
        "clf__min_child_samples": [40],
        "clf__reg_lambda": [0.5, 1.0],
        "clf__subsample": [0.8],
    }


def test_train_and_evaluate_smoke():
    tmp_path = Path(".codex_test_tmp") / f"training_cli_{uuid.uuid4().hex}"
    tmp_path.mkdir(parents=True, exist_ok=True)
    # 1) syntetická data
    df = generate_synthetic_data(n_samples=500, noise_level=0.05)
    # 2) featury
    feats = compute_all_features(df)
    features_csv = tmp_path / "features.csv"
    feats.to_csv(features_csv, index=False)

    # 3) train
    model_out = tmp_path / "model.joblib"
    path_model = train_simple_model(features_csv=str(features_csv), model_out=str(model_out))
    assert os.path.exists(path_model)
    meta_path = model_out.with_name(model_out.stem + "_meta.json")
    assert meta_path.exists()
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    assert meta["sklearn_version"] == sklearn.__version__
    assert meta["python_version"] == sys.version.split()[0]

    # 4) evaluate
    results_csv = tmp_path / "results.csv"
    path_results = evaluate_model_once(features_csv=str(features_csv), model_path=str(model_out), results_out=str(results_csv))
    assert os.path.exists(path_results)

    # 5) results sanity
    res = pd.read_csv(results_csv)
    assert not res.empty
    for col in ["model_path", "profit", "f1", "accuracy", "num_trades"]:
        assert col in res.columns


def test_train_and_evaluate_defaults_to_grid_search_backend(monkeypatch):
    out_dir = _workspace_tmp_dir("search_backend_grid")
    monkeypatch.setattr(train_models, "_model_dir", lambda: out_dir)

    result = train_and_evaluate_model(
        df=_small_training_frame(),
        estimator_name="svm",
        param_grid=_single_candidate_grid(),
        n_splits=3,
        embargo=2,
        holdout_bars=12,
        mc_enabled=False,
        annualize_sharpe=True,
    )

    output_path, meta, search_plan = _load_result_meta(result)

    assert output_path.exists()
    assert result["best_params"] == meta["best_params"]
    assert search_plan["search_backend_requested"] == "grid"
    assert search_plan["search_backend_used"] == "grid"
    assert search_plan["search_backend_fallback_reason"] is None
    assert meta["search_plan"]["search_backend_used"] == "grid"
    assert meta["search_backend"] == "grid"
    assert meta["optuna_best_params"] is None


def test_train_and_evaluate_writes_feature_stability(monkeypatch):
    out_dir = _workspace_tmp_dir("feature_stability")
    monkeypatch.setattr(train_models, "_model_dir", lambda: out_dir)

    result = train_and_evaluate_model(
        df=_small_training_frame(n_rows=120),
        estimator_name="rf",
        param_grid=_single_candidate_rf_grid(),
        n_splits=3,
        embargo=2,
        holdout_bars=12,
        mc_enabled=False,
        annualize_sharpe=True,
    )

    _, meta, _ = _load_result_meta(result)

    assert meta["feature_importance"]
    assert "feature_stability" in meta
    assert set(meta["feature_stability"]) == set(meta["trained_features"])

    for feature_name in meta["trained_features"]:
        stats = meta["feature_stability"][feature_name]
        assert set(stats) == {"mean", "std", "min", "max", "folds_present"}
        assert all(isinstance(stats[key], float) for key in ("mean", "std", "min", "max"))
        assert isinstance(stats["folds_present"], int)
        assert stats["min"] <= stats["mean"] <= stats["max"]
        assert stats["std"] >= 0.0
        assert 1 <= stats["folds_present"] <= 3


def test_train_and_evaluate_collects_feature_stability_only_for_best_params(monkeypatch):
    out_dir = _workspace_tmp_dir("feature_stability_best_params")
    monkeypatch.setattr(train_models, "_model_dir", lambda: out_dir)

    calls: list[dict[str, object]] = []

    def _fake_collect_fold_feature_importances_for_params(**kwargs):
        params = dict(kwargs["params"])
        feature_names = list(kwargs["feature_names"])
        calls.append(params)
        if not feature_names:
            return []
        weight = 1.0 / float(len(feature_names))
        return [{str(name): float(weight) for name in feature_names}]

    monkeypatch.setattr(
        train_models,
        "_collect_fold_feature_importances_for_params",
        _fake_collect_fold_feature_importances_for_params,
    )

    result = train_and_evaluate_model(
        df=_small_training_frame(n_rows=132),
        estimator_name="rf",
        param_grid=_multi_candidate_rf_grid(),
        n_splits=3,
        embargo=2,
        holdout_bars=12,
        mc_enabled=False,
        annualize_sharpe=True,
    )

    _, meta, _ = _load_result_meta(result)

    assert len(calls) == 1
    assert calls[0] == result["best_params"]
    assert meta["feature_stability"]
    assert set(meta["feature_stability"]) == set(meta["trained_features"])


class _DummyTernaryEvalModel:
    def predict_proba(self, X):
        n = len(X)
        short = np.full(n, 0.7, dtype=float)
        hold = np.full(n, 0.2, dtype=float)
        long = np.full(n, 0.1, dtype=float)
        return np.column_stack([short, hold, long])


def test_predict_with_thresholds_returns_signed_labels_for_ternary_models():
    X = pd.DataFrame({"f": [0.0, 1.0, 2.0]})

    y_pred = _predict_with_thresholds(
        _DummyTernaryEvalModel(),
        X,
        {"ternary_threshold_short": 0.5, "ternary_threshold_long": 0.5},
    )

    assert np.array_equal(y_pred, np.array([-1, -1, -1]))


def test_threshold_calibration_split_reserves_embargo_gap_for_label_lookahead():
    df_train = _small_ternary_training_frame(n_rows=210)

    df_core, df_calib, selection, effective_embargo = train_models._select_threshold_calibration_split(
        df_train,
        is_ternary=True,
        threshold_calibration_enabled=True,
        threshold_calibration_pct=0.20,
        threshold_calibration_min_bars=24,
        threshold_calibration_max_bars=60,
        threshold_calibration_train_min_guard=120,
        embargo=3,
        label_lookahead_bars=12,
    )

    assert effective_embargo == 12
    assert df_calib is not None
    assert len(df_calib) == selection["applied_bars"]
    assert len(df_core) == selection["train_core_bars"]
    assert selection["gap_bars"] == 12
    assert selection["embargo_bars"] == 12
    assert selection["label_lookahead_bars"] == 12
    assert selection["embargo_respects_label_lookahead"] is True
    assert selection["no_overlap"] is True
    assert selection["train_core_bars"] + selection["gap_bars"] + selection["applied_bars"] == selection["train_full_bars"]


def test_train_and_evaluate_records_gapped_threshold_calibration_split(monkeypatch):
    out_dir = _workspace_tmp_dir("threshold_calibration_split")
    monkeypatch.setattr(train_models, "_model_dir", lambda: out_dir)

    result = train_and_evaluate_model(
        df=_small_ternary_training_frame(n_rows=360),
        estimator_name="rf",
        param_grid=_single_candidate_rf_grid(),
        n_splits=3,
        embargo=2,
        holdout_bars=36,
        mc_enabled=False,
        annualize_sharpe=True,
        quality_gate_enabled=False,
        label_lookahead_bars=12,
        threshold_calibration_enabled=True,
        threshold_calibration_pct=0.10,
        threshold_calibration_min_bars=24,
        threshold_calibration_max_bars=48,
        threshold_calibration_train_min_guard=120,
    )

    _, meta, _ = _load_result_meta(result)
    selection = dict(meta["threshold_calibration_selection"])

    assert meta["effective_embargo"] == 12
    assert meta["n_threshold_calibration_bars"] == selection["applied_bars"]
    assert selection["embargo_bars"] == 12
    assert selection["label_lookahead_bars"] == 12
    assert selection["embargo_respects_label_lookahead"] is True
    assert selection["gap_bars"] == 12
    assert selection["no_overlap"] is True
    assert selection["train_core_bars"] + selection["gap_bars"] + selection["applied_bars"] == selection["train_full_bars"]


def test_candidate_chain_invalidates_on_criterion_mismatch(monkeypatch):
    out_dir = _workspace_tmp_dir("candidate_chain_criterion_mismatch")
    monkeypatch.setattr(train_models, "_model_dir", lambda: out_dir)

    name_prefix = "chain_case"
    chain_path = train_models._chain_shortlist_path(name_prefix, "rf")
    signature_hash, signature_payload = train_models._build_chain_signature(
        name_prefix=name_prefix,
        estimator_name="rf",
        meta_extra={},
        holdout_mode="bars",
        holdout_pct=None,
        holdout_min_bars=0,
        holdout_max_bars=None,
        holdout_bars=12,
        label_lookahead_bars=0,
        is_ternary=False,
    )
    chain_payload = {
        "signature_hash": signature_hash,
        "signature": signature_payload,
        "modes": {
            "quick": {
                "criterion": "balanced",
                "candidate_count_seen": 2,
                "candidates": [
                    {
                        "params": {
                            "clf__n_estimators": 40,
                            "clf__max_depth": 4,
                            "clf__min_samples_leaf": 1,
                            "clf__n_jobs": 1,
                        },
                        "cv_score": 0.42,
                        "profit_net": 10.0,
                    },
                    {
                        "params": {
                            "clf__n_estimators": 40,
                            "clf__max_depth": 2,
                            "clf__min_samples_leaf": 1,
                            "clf__n_jobs": 1,
                        },
                        "cv_score": 0.38,
                        "profit_net": 8.0,
                    },
                ],
            }
        },
    }
    chain_path.write_text(json.dumps(chain_payload, ensure_ascii=True, indent=2), encoding="utf-8")

    result = train_and_evaluate_model(
        df=_small_training_frame(n_rows=120),
        estimator_name="rf",
        param_grid={
            "clf__n_estimators": [40],
            "clf__max_depth": [2, 4],
            "clf__min_samples_leaf": [1],
            "clf__n_jobs": [1],
        },
        n_splits=3,
        embargo=2,
        holdout_bars=12,
        mc_enabled=False,
        annualize_sharpe=True,
        name_prefix=name_prefix,
        training_mode="standard",
        candidate_chain_enabled=True,
        candidate_selection_criterion="profit_first",
        candidate_top_n=2,
        candidate_fresh_ratio=0.30,
    )

    _, meta, _ = _load_result_meta(result)
    chain_info = dict(meta["search_plan"]["candidate_chain"])

    assert chain_info["source_mode"] == "quick"
    assert chain_info["signature_match"] is True
    assert chain_info["source_criterion"] == "balanced"
    assert chain_info["invalid_reason"] == "criterion_mismatch"
    assert chain_info["reuse_decision"] == "fresh_sampling"
    assert chain_info["used"] is False
    assert chain_info["carry_count"] == 0
    assert chain_info["reranked_with_current_criterion"] is False

def test_train_and_evaluate_accepts_canonical_training_mode_alias(monkeypatch):
    out_dir = _workspace_tmp_dir("training_mode_alias")
    monkeypatch.setattr(train_models, "_model_dir", lambda: out_dir)

    result = train_and_evaluate_model(
        df=_small_training_frame(n_rows=120),
        estimator_name="rf",
        param_grid=_single_candidate_rf_grid(),
        n_splits=3,
        embargo=2,
        holdout_bars=12,
        mc_enabled=False,
        annualize_sharpe=True,
        training_mode="explore",
        candidate_chain_enabled=True,
    )

    _, meta, _ = _load_result_meta(result)
    chain_info = dict(meta["search_plan"]["candidate_chain"])

    assert chain_info["mode"] == "quick"
    assert chain_info["source_mode"] is None


def test_train_and_evaluate_rejects_unknown_training_mode():
    with pytest.raises(ValueError, match="Unsupported training_mode"):
        train_and_evaluate_model(
            df=_small_training_frame(n_rows=96),
            estimator_name="rf",
            param_grid=_single_candidate_rf_grid(),
            n_splits=3,
            embargo=2,
            holdout_bars=12,
            mc_enabled=False,
            annualize_sharpe=True,
            training_mode="mystery_mode",
        )


def test_finalize_threshold_tuning_outcome_records_fallback_reason_and_adjustments():
    threshold_tuning = {
        "selected_mode_base": "score_grid",
        "selected_mode": "shared_threshold_asymmetry_guard",
        "rebalance_adjustment": {
            "before_thresholds": {"short": 0.42, "long": 0.58},
            "after_thresholds": {"short": 0.47, "long": 0.55},
            "reverted": False,
        },
        "threshold_cap": {
            "before": {"short": 0.47, "long": 0.55},
            "after": {"short": 0.46, "long": 0.55},
        },
        "threshold_gap_guard": {
            "accepted": True,
        },
        "threshold_asymmetry_guard": {
            "accepted": True,
        },
    }

    outcome = train_models._finalize_threshold_tuning_outcome(dict(threshold_tuning))

    assert outcome["primary_search_mode"] == "score_grid"
    assert outcome["final_selected_mode"] == "shared_threshold_asymmetry_guard"
    assert outcome["fallback_reason"] == "shared_threshold_asymmetry_guard"
    assert outcome["adjustments_applied"] == [
        "rebalance",
        "threshold_cap",
        "threshold_gap_guard",
        "threshold_asymmetry_guard",
        "shared_threshold_asymmetry_guard",
    ]


def test_finalize_threshold_tuning_outcome_preserves_nonfallback_primary_mode():
    threshold_tuning = {
        "selected_mode_base": "quantile_override",
        "selected_mode": "rebalance(quantile_override)",
        "rebalance_adjustment": {
            "before_thresholds": {"short": 0.31, "long": 0.63},
            "after_thresholds": {"short": 0.35, "long": 0.59},
            "reverted": False,
        },
    }

    outcome = train_models._finalize_threshold_tuning_outcome(dict(threshold_tuning))

    assert outcome["primary_search_mode"] == "quantile_override"
    assert outcome["final_selected_mode"] == "rebalance(quantile_override)"
    assert outcome["fallback_reason"] is None
    assert outcome["adjustments_applied"] == ["rebalance"]


def test_train_and_evaluate_persists_threshold_tuning_outcome_metadata(monkeypatch):
    out_dir = _workspace_tmp_dir("threshold_tuning_outcome")
    monkeypatch.setattr(train_models, "_model_dir", lambda: out_dir)

    result = train_and_evaluate_model(
        df=_small_ternary_training_frame(n_rows=360),
        estimator_name="rf",
        param_grid=_single_candidate_rf_grid(),
        n_splits=3,
        embargo=2,
        holdout_bars=36,
        mc_enabled=False,
        annualize_sharpe=True,
        quality_gate_enabled=False,
        label_lookahead_bars=12,
        threshold_calibration_enabled=True,
        threshold_calibration_pct=0.10,
        threshold_calibration_min_bars=24,
        threshold_calibration_max_bars=48,
        threshold_calibration_train_min_guard=120,
    )

    _, meta, _ = _load_result_meta(result)
    threshold_tuning = dict(meta["threshold_tuning"])

    assert threshold_tuning["primary_search_mode"] in {"score_grid", "quantile_override"}
    assert threshold_tuning["final_selected_mode"] == threshold_tuning["selected_mode"]
    assert "fallback_reason" in threshold_tuning
    assert isinstance(threshold_tuning["adjustments_applied"], list)
    assert all(isinstance(item, str) for item in threshold_tuning["adjustments_applied"])
    if threshold_tuning["fallback_reason"] is not None:
        assert isinstance(threshold_tuning["fallback_reason"], str)


def test_apply_single_fallback_ternary_thresholds_keeps_primary_when_guardrails_pass():
    y_true = np.asarray([0, 0, 2, 2, 1, 1, 0, 2, 1, 0, 2, 1], dtype=int)
    oof_short = np.asarray([0.82, 0.78, 0.10, 0.12, 0.15, 0.20, 0.75, 0.18, 0.12, 0.80, 0.16, 0.10], dtype=float)
    oof_long = np.asarray([0.08, 0.12, 0.82, 0.78, 0.20, 0.15, 0.18, 0.76, 0.10, 0.14, 0.80, 0.12], dtype=float)

    ts, tl, tuning = train_models._apply_single_fallback_ternary_thresholds(
        y_true_mapped=y_true,
        oof_short=oof_short,
        oof_long=oof_long,
        thr_short=0.60,
        thr_long=0.60,
        estimator_name="rf",
        min_side_floor=2,
        max_side_dominance=0.72,
        min_side_recall=0.10,
    )

    assert ts == pytest.approx(0.60)
    assert tl == pytest.approx(0.60)
    assert tuning["guardrail_reasons"] == []
    assert tuning["oof_selected_final"]["n_short"] >= 2.0
    assert tuning["oof_selected_final"]["n_long"] >= 2.0
    assert "selected_mode" not in tuning
    assert "fallback_reason" not in tuning


def test_apply_single_fallback_ternary_thresholds_uses_explicit_fallback_when_guardrails_fail():
    y_true = np.asarray([0, 0, 2, 2, 1, 1, 0, 2, 1, 0, 2, 1], dtype=int)
    oof_short = np.asarray([0.82, 0.78, 0.10, 0.12, 0.15, 0.20, 0.75, 0.18, 0.12, 0.80, 0.16, 0.10], dtype=float)
    oof_long = np.asarray([0.08, 0.12, 0.82, 0.78, 0.20, 0.15, 0.18, 0.76, 0.10, 0.14, 0.80, 0.12], dtype=float)

    ts, tl, tuning = train_models._apply_single_fallback_ternary_thresholds(
        y_true_mapped=y_true,
        oof_short=oof_short,
        oof_long=oof_long,
        thr_short=0.95,
        thr_long=0.03,
        estimator_name="rf",
        min_side_floor=2,
        max_side_dominance=0.72,
        min_side_recall=0.10,
    )

    assert tuning["selected_mode"] == "data_driven_fallback"
    assert tuning["fallback_reason"] == "primary_thresholds_failed_guardrails"
    assert "threshold_gap_above_limit" in tuning["guardrail_reasons"]
    assert tuning["data_driven_fallback"]["guardrail_reasons"] == tuning["guardrail_reasons"]
    assert tuning["oof_selected_final"]["n_short"] >= 2.0
    assert tuning["oof_selected_final"]["n_long"] >= 2.0
    assert ts < 0.95
    assert tl >= 0.03


def test_train_and_evaluate_unsupported_estimator_writes_empty_feature_stability(monkeypatch):
    out_dir = _workspace_tmp_dir("feature_stability_unsupported")
    monkeypatch.setattr(train_models, "_model_dir", lambda: out_dir)

    result = train_and_evaluate_model(
        df=_small_training_frame(n_rows=120),
        estimator_name="svm",
        param_grid=_single_candidate_grid(),
        n_splits=3,
        embargo=2,
        holdout_bars=12,
        mc_enabled=False,
        annualize_sharpe=True,
    )

    _, meta, _ = _load_result_meta(result)

    assert "feature_stability" in meta
    assert meta["feature_stability"] == {}


def test_train_and_evaluate_falls_back_to_grid_when_optuna_missing(monkeypatch, caplog):
    out_dir = _workspace_tmp_dir("search_backend_fallback")
    monkeypatch.setattr(train_models, "_model_dir", lambda: out_dir)
    monkeypatch.setattr(train_models, "HAS_OPTUNA", False, raising=False)
    monkeypatch.setattr(train_models, "optuna", None, raising=False)
    caplog.set_level(logging.WARNING)

    result = train_and_evaluate_model(
        df=_small_training_frame(),
        estimator_name="svm",
        param_grid=_single_candidate_grid(),
        n_splits=3,
        embargo=2,
        holdout_bars=12,
        mc_enabled=False,
        annualize_sharpe=True,
        search_backend="optuna",
    )

    output_path, meta, search_plan = _load_result_meta(result)

    assert output_path.exists()
    assert search_plan["search_backend_requested"] == "optuna"
    assert search_plan["search_backend_used"] == "grid"
    assert search_plan["search_backend_fallback_reason"] == "optuna_not_available"
    assert meta["search_plan"]["search_backend_used"] == "grid"
    assert "search_backend='optuna' requested but Optuna is not available" in caplog.text
    assert "falling back to 'grid'" in caplog.text


def test_train_and_evaluate_falls_back_to_grid_when_optuna_estimator_unsupported(monkeypatch, caplog):
    out_dir = _workspace_tmp_dir("search_backend_unsupported")
    monkeypatch.setattr(train_models, "_model_dir", lambda: out_dir)
    monkeypatch.setattr(train_models, "HAS_OPTUNA", True, raising=False)
    monkeypatch.setattr(train_models, "optuna", object(), raising=False)
    caplog.set_level(logging.WARNING)

    result = train_and_evaluate_model(
        df=_small_training_frame(),
        estimator_name="svm",
        param_grid=_single_candidate_grid(),
        n_splits=3,
        embargo=2,
        holdout_bars=12,
        mc_enabled=False,
        annualize_sharpe=True,
        search_backend="optuna",
        optuna_trials=4,
        optuna_timeout_seconds=30,
    )

    _, meta, search_plan = _load_result_meta(result)

    assert search_plan["search_backend_requested"] == "optuna"
    assert search_plan["search_backend_used"] == "grid"
    assert search_plan["search_backend_fallback_reason"] == "optuna_estimator_not_supported"
    assert meta["search_backend"] == "grid"
    assert "unsupported estimator 'svm'" in caplog.text


@pytest.mark.skipif(not train_models.HAS_OPTUNA, reason="Optuna is not installed")
def test_train_and_evaluate_optuna_hgbt_returns_best_params(monkeypatch):
    out_dir = _workspace_tmp_dir("search_backend_optuna_hgbt")
    monkeypatch.setattr(train_models, "_model_dir", lambda: out_dir)

    result = train_and_evaluate_model(
        df=_small_training_frame(n_rows=132),
        estimator_name="hgbt",
        param_grid=_small_hgbt_grid(),
        n_splits=3,
        embargo=2,
        holdout_bars=18,
        mc_enabled=False,
        annualize_sharpe=True,
        search_backend="optuna",
        optuna_trials=4,
        optuna_timeout_seconds=60,
    )

    _, meta, search_plan = _load_result_meta(result)

    assert result["best_params"]
    assert search_plan["search_backend_used"] == "optuna"
    assert search_plan["optuna_completed_trials"] >= 1
    assert meta["search_backend"] == "optuna"
    assert meta["optuna_trials"] == 4
    assert meta["optuna_best_params"] == result["best_params"]
    assert meta["optuna_best_score"] == pytest.approx(float(result["best_score"]))


@pytest.mark.skipif(
    not (train_models.HAS_OPTUNA and getattr(train_models, "HAS_LGB", False)),
    reason="Optuna or LightGBM is not installed",
)
def test_train_and_evaluate_optuna_lgb_returns_best_params(monkeypatch):
    out_dir = _workspace_tmp_dir("search_backend_optuna_lgb")
    monkeypatch.setattr(train_models, "_model_dir", lambda: out_dir)

    result = train_and_evaluate_model(
        df=_small_training_frame(n_rows=132),
        estimator_name="lgb",
        param_grid=_small_lgb_grid(),
        n_splits=3,
        embargo=2,
        holdout_bars=18,
        mc_enabled=False,
        annualize_sharpe=True,
        search_backend="optuna",
        optuna_trials=4,
        optuna_timeout_seconds=60,
    )

    _, meta, search_plan = _load_result_meta(result)

    assert result["best_params"]
    assert search_plan["search_backend_used"] == "optuna"
    assert search_plan["optuna_completed_trials"] >= 1
    assert meta["search_backend"] == "optuna"
    assert meta["optuna_trials"] == 4
    assert meta["optuna_best_params"] == result["best_params"]
    assert meta["optuna_best_score"] == pytest.approx(float(result["best_score"]))
