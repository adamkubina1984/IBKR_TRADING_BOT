import json
import logging
from pathlib import Path

import joblib
import numpy as np

from ibkr_trading_bot.core.services.model_service import load_model_with_meta, runtime_sklearn_version


class DummyPredictor:
    def __init__(self):
        self.classes_ = np.array([0, 1, 2])
        self.feature_names_in_ = np.array(["f_open", "f_close"], dtype=object)

    def predict(self, X):
        return np.zeros(len(X), dtype=int)


def _write_dummy_model(tmp_path, sklearn_version: str | None):
    model_path = tmp_path / "dummy_model.pkl"
    joblib.dump({"model": DummyPredictor()}, model_path)
    meta = {
        "created_at": "2026-03-06T12:00:00",
        "trained_features": ["f_open", "f_close"],
        "classes": [0, 1, 2],
        "ternary_threshold_short": 0.4,
        "ternary_threshold_long": 0.6,
    }
    if sklearn_version is not None:
        meta["sklearn_version"] = sklearn_version
    meta_path = model_path.with_name(model_path.stem + "_meta.json")
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return model_path


def test_load_model_with_matching_sklearn_version_has_no_warning(caplog, tmp_path):
    caplog.set_level(logging.WARNING)
    model_path = _write_dummy_model(tmp_path, runtime_sklearn_version())

    loaded = load_model_with_meta(model_path)

    assert loaded.version_warning is None
    assert "scikit-learn mismatch" not in caplog.text


def test_load_model_with_mismatched_sklearn_version_warns(caplog, tmp_path):
    caplog.set_level(logging.WARNING)
    runtime_version = runtime_sklearn_version() or "unknown"
    mismatched_version = "9.9.9" if runtime_version != "9.9.9" else "0.0.1"
    model_path = _write_dummy_model(tmp_path, mismatched_version)

    loaded = load_model_with_meta(model_path)

    assert loaded.version_warning is not None
    assert mismatched_version in loaded.version_warning
    assert runtime_version in loaded.version_warning
    assert "scikit-learn mismatch" in caplog.text


def test_load_model_with_meta_recovers_missing_sidecar(tmp_path):
    model_path = Path(tmp_path) / "refresh_model.pkl"
    joblib.dump(
        {
            "model": DummyPredictor(),
            "created_at": "20260421_131050",
            "estimator_name": "hgbt",
            "features": ["f_open", "f_close"],
            "ternary_threshold_short": 0.41,
            "ternary_threshold_long": 0.63,
            "metrics_holdout": {"profit_net": 12.5, "sharpe": 0.7},
            "training_mode": "refresh",
        },
        model_path,
    )

    loaded = load_model_with_meta(model_path)
    meta_path = model_path.with_name(model_path.stem + "_meta.json")

    assert meta_path.exists()
    assert loaded.meta["trained_features"] == ["f_open", "f_close"]
    assert loaded.meta["model_classes"] == [0, 1, 2]
    assert loaded.meta["ternary_threshold_short"] == 0.41
    assert loaded.meta["ternary_threshold_long"] == 0.63
    assert loaded.meta["metrics"]["profit_net"] == 12.5
