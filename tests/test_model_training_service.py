import json
from pathlib import Path

import pandas as pd
import pytest

from ibkr_trading_bot.core.services import model_training_service as model_training_service_module


def test_run_training_job_reports_trade_count_from_num_trades(monkeypatch, tmp_path):
    csv_path = tmp_path / "features.csv"
    csv_path.write_text("timestamp,close\n2026-01-01T00:00:00Z,100\n", encoding="utf-8")

    prepared = pd.DataFrame(
        {
            "timestamp": pd.date_range("2026-01-01", periods=32, freq="5min", tz="UTC"),
            "close": range(32),
        }
    )

    monkeypatch.setattr(
        model_training_service_module.DatasetService,
        "prepare_from_csv",
        lambda self, *args, **kwargs: prepared.copy(),
    )
    monkeypatch.setattr(model_training_service_module, "compute_holdout_bars", lambda **kwargs: 8)
    monkeypatch.setattr(model_training_service_module, "name_and_meta_from_csv", lambda *args, **kwargs: ("demo", {}))
    monkeypatch.setattr(model_training_service_module, "_model_dir", lambda: str(tmp_path))
    monkeypatch.setattr(
        model_training_service_module,
        "_result_metrics_from_meta",
        lambda meta: {
            "profit_net": 42.0,
            "sharpe": 1.2,
            "pf": 1.8,
            "num_trades": 9,
            "trades": [{"pnl": 1.0}],
            "num_trades_short": 4,
            "num_trades_long": 5,
        },
    )

    def _fake_train_and_evaluate_model(*args, **kwargs):
        model_path = tmp_path / "demo_rf_20260420_120000.pkl"
        model_path.write_bytes(b"model")
        meta_path = model_path.with_name(model_path.stem + "_meta.json")
        meta_path.write_text(json.dumps({"search_plan": {}, "quality_gate": {}}), encoding="utf-8")
        return {"output_path": str(model_path)}

    monkeypatch.setattr(model_training_service_module, "train_and_evaluate_model", _fake_train_and_evaluate_model)

    result = model_training_service_module.run_training_job(
        csv_path=str(csv_path),
        holdout_pct=0.2,
        holdout_min_bars=8,
        holdout_max_bars=32,
        phase="standard",
        estimator_name="rf",
        criterion="balanced",
        horizon=8,
        tp_bps=40.0,
        sl_bps=40.0,
        candidate_top_n=5,
        candidate_fresh_ratio=0.3,
        training_profile={"n_splits": 3},
    )

    assert result["trades"] == 9
    assert result["num_trades_short"] == 4
    assert result["num_trades_long"] == 5


def test_run_training_job_records_canonical_workflow_mode_for_legacy_phase(monkeypatch, tmp_path):
    csv_path = tmp_path / "features.csv"
    csv_path.write_text("timestamp,close\n2026-01-01T00:00:00Z,100\n", encoding="utf-8")

    prepared = pd.DataFrame(
        {
            "timestamp": pd.date_range("2026-01-01", periods=32, freq="5min", tz="UTC"),
            "close": range(32),
        }
    )

    monkeypatch.setattr(
        model_training_service_module.DatasetService,
        "prepare_from_csv",
        lambda self, *args, **kwargs: prepared.copy(),
    )
    monkeypatch.setattr(model_training_service_module, "compute_holdout_bars", lambda **kwargs: 8)
    monkeypatch.setattr(model_training_service_module, "name_and_meta_from_csv", lambda *args, **kwargs: ("demo", {}))
    monkeypatch.setattr(model_training_service_module, "_model_dir", lambda: str(tmp_path))

    captured: dict[str, object] = {}

    def _fake_train_and_evaluate_model(*args, **kwargs):
        captured["training_mode"] = kwargs.get("training_mode")
        captured["meta_extra"] = dict(kwargs.get("meta_extra") or {})
        model_path = tmp_path / "demo_rf_20260420_120000.pkl"
        model_path.write_bytes(b"model")
        meta_path = model_path.with_name(model_path.stem + "_meta.json")
        meta_path.write_text(json.dumps({"search_plan": {}, "quality_gate": {}}), encoding="utf-8")
        return {"output_path": str(model_path)}

    monkeypatch.setattr(model_training_service_module, "train_and_evaluate_model", _fake_train_and_evaluate_model)

    result = model_training_service_module.run_training_job(
        csv_path=str(csv_path),
        holdout_pct=0.2,
        holdout_min_bars=8,
        holdout_max_bars=32,
        phase="strict",
        estimator_name="rf",
        criterion="balanced",
        horizon=8,
        tp_bps=40.0,
        sl_bps=40.0,
        candidate_top_n=5,
        candidate_fresh_ratio=0.3,
        training_profile=model_training_service_module.training_profile_for_mode("strict"),
    )

    meta_extra = dict(captured["meta_extra"])
    profile = dict(meta_extra["training_profile"])

    assert captured["training_mode"] == "strict"
    assert meta_extra["training_mode"] == "strict"
    assert meta_extra["training_mode_requested"] == "strict"
    assert meta_extra["training_mode_compatibility"] == "strict"
    assert meta_extra["workflow_mode"] == "refresh"
    assert profile["compatibility_mode"] == "strict"
    assert profile["workflow_mode"] == "refresh"
    assert profile["training_mode_requested"] == "strict"
    assert profile["training_mode"] == "strict"
    assert result["phase"] == "strict"
    assert result["workflow_mode"] == "refresh"
    assert result["compatibility_mode"] == "strict"
    assert result["runtime_training_mode"] == "strict"


def test_run_training_job_maps_canonical_phase_to_compatibility_training_mode(monkeypatch, tmp_path):
    csv_path = tmp_path / "features.csv"
    csv_path.write_text("timestamp,close\n2026-01-01T00:00:00Z,100\n", encoding="utf-8")

    prepared = pd.DataFrame(
        {
            "timestamp": pd.date_range("2026-01-01", periods=32, freq="5min", tz="UTC"),
            "close": range(32),
        }
    )

    monkeypatch.setattr(
        model_training_service_module.DatasetService,
        "prepare_from_csv",
        lambda self, *args, **kwargs: prepared.copy(),
    )
    monkeypatch.setattr(model_training_service_module, "compute_holdout_bars", lambda **kwargs: 8)
    monkeypatch.setattr(model_training_service_module, "name_and_meta_from_csv", lambda *args, **kwargs: ("demo", {}))
    monkeypatch.setattr(model_training_service_module, "_model_dir", lambda: str(tmp_path))

    captured: dict[str, object] = {}

    def _fake_train_and_evaluate_model(*args, **kwargs):
        captured["training_mode"] = kwargs.get("training_mode")
        captured["meta_extra"] = dict(kwargs.get("meta_extra") or {})
        model_path = tmp_path / "demo_rf_20260420_120000.pkl"
        model_path.write_bytes(b"model")
        meta_path = model_path.with_name(model_path.stem + "_meta.json")
        meta_path.write_text(json.dumps({"search_plan": {}, "quality_gate": {}}), encoding="utf-8")
        return {"output_path": str(model_path)}

    monkeypatch.setattr(model_training_service_module, "train_and_evaluate_model", _fake_train_and_evaluate_model)

    result = model_training_service_module.run_training_job(
        csv_path=str(csv_path),
        holdout_pct=0.2,
        holdout_min_bars=8,
        holdout_max_bars=32,
        phase="explore",
        estimator_name="rf",
        criterion="balanced",
        horizon=8,
        tp_bps=40.0,
        sl_bps=40.0,
        candidate_top_n=5,
        candidate_fresh_ratio=0.3,
        training_profile=model_training_service_module.training_profile_for_mode("explore"),
    )

    meta_extra = dict(captured["meta_extra"])
    profile = dict(meta_extra["training_profile"])

    assert captured["training_mode"] == "quick"
    assert meta_extra["training_mode"] == "explore"
    assert meta_extra["training_mode_requested"] == "explore"
    assert meta_extra["training_mode_compatibility"] == "quick"
    assert meta_extra["workflow_mode"] == "explore"
    assert profile["training_mode"] == "quick"
    assert profile["training_mode_requested"] == "explore"
    assert profile["workflow_mode"] == "explore"
    assert profile["compatibility_mode"] == "quick"
    assert result["phase"] == "explore"
    assert result["workflow_mode"] == "explore"
    assert result["compatibility_mode"] == "quick"
    assert result["runtime_training_mode"] == "quick"


def test_run_training_job_rejects_unknown_phase(tmp_path):
    csv_path = tmp_path / "features.csv"
    csv_path.write_text("timestamp,close\n2026-01-01T00:00:00Z,100\n", encoding="utf-8")

    with pytest.raises(ValueError, match="Unsupported training mode"):
        model_training_service_module.run_training_job(
            csv_path=str(csv_path),
            holdout_pct=0.2,
            holdout_min_bars=8,
            holdout_max_bars=32,
            phase="mystery_mode",
            estimator_name="rf",
            criterion="balanced",
            horizon=8,
            tp_bps=40.0,
            sl_bps=40.0,
            candidate_top_n=5,
            candidate_fresh_ratio=0.3,
            training_profile={"n_splits": 3},
        )