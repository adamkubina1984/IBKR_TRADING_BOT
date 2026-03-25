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
from ibkr_trading_bot.model.evaluate_models import evaluate_model_once
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


def _single_candidate_grid() -> dict[str, list]:
    return {
        "clf__C": [1.0],
        "clf__gamma": ["scale"],
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
