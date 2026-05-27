import csv
import json
import uuid
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QApplication

from ibkr_trading_bot.core.services import model_eval_service as model_eval_runtime
from ibkr_trading_bot.core.services.model_service import read_sidecar_model_meta, write_sidecar_model_meta
from ibkr_trading_bot.data.generate_synthetic import generate_synthetic_data
from ibkr_trading_bot.gui import tab_model_ranking as tab_model_ranking_module
from ibkr_trading_bot.gui.tab_model_ranking import (
    COL_BIAS,
    COL_MODE,
    COL_NOTE,
    COL_PROFIT,
    COL_PROFIT_OPT,
    COL_STABILITY,
    ModelRankingTab,
    discover_ranking_models,
)


class DummyRankingPredictor:
    def __init__(self):
        self.classes_ = np.array([0, 1, 2])

    def predict(self, X):
        n = len(X)
        out = np.zeros(n, dtype=int)
        out[n // 3 : (2 * n) // 3] = 1
        out[(2 * n) // 3 :] = 2
        return out

    def predict_proba(self, X):
        n = len(X)
        lin = np.linspace(0.0, 1.0, n, dtype=float)
        short = 0.15 + 0.70 * (1.0 - lin)
        neutral = np.full(n, 0.20, dtype=float)
        long = 0.15 + 0.70 * lin
        arr = np.column_stack([short, neutral, long])
        arr = arr / arr.sum(axis=1, keepdims=True)
        return arr


@pytest.fixture(scope="module")
def qapp():
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    return app


def _ranking_payload(
    csv_path: Path,
    *,
    fee: float,
    profit_h: float,
    trades_h: float,
    metadata: dict[str, object] | None = None,
) -> dict[str, object]:
    return model_eval_runtime.build_tab5_holdout_ranking_payload(
        data_path=csv_path,
        fee_per_trade=fee,
        metadata=metadata,
        entry_threshold=0.55,
        exit_threshold=0.60,
        metrics={"profit_net": profit_h, "max_dd": -10.0, "trades": trades_h},
        status="ok",
    )


def _write_ranked_model(
    tmp_path: Path,
    name: str,
    *,
    csv_path: Path,
    n_total_bars: int,
    estimator: str = "hgbt",
    horizon: int = 8,
    tp_bps: float = 50.0,
    sl_bps: float = 50.0,
    base_profit: float = 100.0,
    sharpe: float = 0.02,
    optimized_profit: float = 500.0,
    trades_h: float = 120.0,
    fee: float = 0.25,
    ranking_status: str = "ok",
    instrument: str = "GC",
    exchange: str = "COMEX",
    timeframe: str = "5m",
    training_mode: str | None = None,
    candidate_selection_criterion: str | None = None,
    user_note: str | None = None,
    trained_features: list[str] | None = None,
    feature_stability: dict[str, object] | None = None,
    feature_stability_score: dict[str, float] | None = None,
    feature_stability_threshold: float | None = None,
    feature_stability_filter_applied: bool = False,
    features_kept_by_stability: list[str] | None = None,
    features_removed_by_stability: list[str] | None = None,
    per_class_3: dict[str, object] | None = None,
) -> Path:
    model_path = tmp_path / name
    model_path.write_bytes(b"model")
    meta = {
        "created_at": "2026-03-16T10:00:00",
        "created_at_iso": "2026-03-16T10:00:00",
        "instrument": instrument,
        "exchange": exchange,
        "timeframe": timeframe,
        "n_total_bars": int(n_total_bars),
        "estimator_name": estimator,
        "label_horizon_bars": int(horizon),
        "label_take_profit_bps": float(tp_bps),
        "label_stop_loss_bps": float(sl_bps),
        "metrics_holdout": {
            "profit_net": float(base_profit),
            "sharpe": float(sharpe),
        },
        "trained_features": list(trained_features or []),
    }
    if training_mode:
        meta["training_mode"] = str(training_mode)
    if candidate_selection_criterion:
        meta["candidate_selection_criterion"] = str(candidate_selection_criterion)
    if user_note:
        meta["model_ranking_note"] = str(user_note)
    if feature_stability is not None:
        meta["feature_stability"] = dict(feature_stability)
    if feature_stability_score is not None:
        meta["feature_stability_score"] = dict(feature_stability_score)
    if per_class_3 is not None:
        meta["metrics_holdout"]["per_class_3"] = dict(per_class_3)
    if feature_stability_threshold is not None:
        meta["feature_stability_threshold"] = float(feature_stability_threshold)
    if feature_stability_filter_applied:
        meta["feature_stability_filter_applied"] = True
    if features_kept_by_stability is not None:
        meta["features_kept_by_stability"] = list(features_kept_by_stability)
    if features_removed_by_stability is not None:
        meta["features_removed_by_stability"] = list(features_removed_by_stability)
    if ranking_status == "ok":
        meta[model_eval_runtime.TAB5_HOLDOUT_RANKING_KEY] = _ranking_payload(
            csv_path,
            fee=fee,
            profit_h=optimized_profit,
            trades_h=trades_h,
            metadata=meta,
        )
    else:
        meta[model_eval_runtime.TAB5_HOLDOUT_RANKING_KEY] = {
            "status": ranking_status,
            "error": "bad",
        }
    write_sidecar_model_meta(model_path, meta)
    return model_path


def test_tab5_holdout_ranking_persistence_and_stale_detection():
    tmp_path = Path(".codex_test_tmp") / f"tab5_rank_meta_{uuid.uuid4().hex}"
    tmp_path.mkdir(parents=True, exist_ok=True)
    csv_path = tmp_path / "features.csv"
    csv_path.write_text("a,b\n1,2\n3,4\n", encoding="utf-8")
    model_path = tmp_path / "model.pkl"
    model_path.write_bytes(b"model")

    meta = {
        "created_at": "2026-03-16T10:00:00",
        "trained_features": ["feat_a", "feat_b"],
        "class_to_dir": {0: "SHORT", 1: "HOLD", 2: "LONG"},
        "ternary_threshold_short": 0.45,
        "ternary_threshold_long": 0.55,
    }
    meta[model_eval_runtime.TAB5_HOLDOUT_RANKING_KEY] = model_eval_runtime.build_tab5_holdout_ranking_payload(
        data_path=csv_path,
        fee_per_trade=0.25,
        metadata=meta,
        base_entry_threshold=0.34,
        base_exit_threshold=0.41,
        base_metrics={"profit_net": 12.5, "max_dd": -5.0, "trades": 9},
        optimized_entry_threshold=0.36,
        optimized_exit_threshold=0.43,
        optimized_metrics={"profit_net": 18.5, "max_dd": -4.0, "trades": 11},
        entry_threshold=0.36,
        exit_threshold=0.43,
        metrics={"profit_net": 18.5, "max_dd": -4.0, "trades": 11},
        status="ok",
    )
    meta_path = write_sidecar_model_meta(model_path, meta)

    reloaded = read_sidecar_model_meta(model_path)
    ranking = reloaded[model_eval_runtime.TAB5_HOLDOUT_RANKING_KEY]

    assert ranking["status"] == "ok"
    assert ranking["schema_version"] == model_eval_runtime.TAB5_HOLDOUT_RANKING_SCHEMA_VERSION
    assert ranking["threshold_source"] == "model"
    assert ranking["label_mode"] == "ternary_mapped"
    assert ranking["feature_contract"] == {"count": 2, "signature": ranking["feature_contract"]["signature"]}
    assert isinstance(ranking["feature_contract"]["signature"], str)
    assert ranking["entry_threshold"] == 0.36
    assert ranking["exit_threshold"] == 0.43
    assert ranking["profit_h"] == 18.5
    assert ranking["base_profit_h"] == 12.5
    assert ranking["holdout_base"]["entry_threshold"] == 0.34
    assert ranking["holdout_base"]["profit_h"] == 12.5
    assert ranking["holdout_optimized"]["entry_threshold"] == 0.36
    assert ranking["holdout_optimized"]["profit_h"] == 18.5
    assert model_eval_runtime.is_tab5_holdout_ranking_stale(
        reloaded,
        data_path=csv_path,
        fee_per_trade=0.25,
        model_path=model_path,
        meta_path=meta_path,
    ) is False
    assert model_eval_runtime.is_tab5_holdout_ranking_stale(
        reloaded,
        data_path=csv_path,
        fee_per_trade=0.30,
        model_path=model_path,
        meta_path=meta_path,
    ) is True

    csv_path.write_text("a,b\n1,2\n3,4\n5,6\n", encoding="utf-8")
    assert model_eval_runtime.is_tab5_holdout_ranking_stale(
        reloaded,
        data_path=csv_path,
        fee_per_trade=0.25,
        model_path=model_path,
        meta_path=meta_path,
    ) is True

    feature_changed = dict(reloaded)
    feature_changed["trained_features"] = ["feat_a", "feat_c"]
    assert model_eval_runtime.is_tab5_holdout_ranking_stale(
        feature_changed,
        data_path=csv_path,
        fee_per_trade=0.25,
        model_path=model_path,
        meta_path=meta_path,
    ) is True

    label_changed = dict(reloaded)
    label_changed["class_to_dir"] = {-1: "SHORT", 0: "HOLD", 1: "LONG"}
    assert model_eval_runtime.is_tab5_holdout_ranking_stale(
        label_changed,
        data_path=csv_path,
        fee_per_trade=0.25,
        model_path=model_path,
        meta_path=meta_path,
    ) is True


def test_set_and_get_tab5_holdout_ranking_by_policy_bucket():
    meta = {}

    stored = model_eval_runtime.set_tab5_holdout_ranking(
        meta,
        {"status": "ok"},
        exit_policy="flat_on_neutral",
    )

    assert stored["exit_policy"] == "flat_on_weak_signal"
    assert model_eval_runtime.get_tab5_holdout_ranking(meta, exit_policy="flat_on_neutral") == stored
    assert model_eval_runtime.get_tab5_holdout_ranking(meta, exit_policy="hold_until_opposite") is None


def test_policy_specific_lookup_rejects_legacy_ranking_without_exit_policy(tmp_path: Path):
    csv_path = tmp_path / "features.csv"
    csv_path.write_text("a,b\n1,2\n3,4\n", encoding="utf-8")

    ranking = model_eval_runtime.build_tab5_holdout_ranking_payload(
        data_path=csv_path,
        fee_per_trade=0.25,
        entry_threshold=0.34,
        exit_threshold=0.41,
        metrics={"profit_net": 12.5, "max_dd": -5.0, "trades": 9},
        status="ok",
    )
    ranking.pop("exit_policy", None)
    meta = {model_eval_runtime.TAB5_HOLDOUT_RANKING_KEY: ranking}

    assert model_eval_runtime.get_tab5_holdout_ranking(meta, exit_policy="flat_on_weak_signal") is None
    assert model_eval_runtime.is_tab5_holdout_ranking_stale(
        meta,
        data_path=csv_path,
        fee_per_trade=0.25,
    ) is True


def test_discover_ranking_models_sorts_optimized_then_fallback_then_error():
    tmp_path = Path(".codex_test_tmp") / f"tab5_rank_sort_{uuid.uuid4().hex}"
    tmp_path.mkdir(parents=True, exist_ok=True)

    paths = {
        "optimized.pkl": {
            "created_at": "2026-03-16T09:00:00",
            "metrics_holdout": {"profit_net": 5.0, "sharpe": 0.1},
            model_eval_runtime.TAB5_HOLDOUT_RANKING_KEY: {
                "status": "ok",
                "profit_h": 25.0,
            },
        },
        "fallback.pkl": {
            "created_at": "2026-03-16T08:00:00",
            "metrics_holdout": {"profit_net": 20.0, "sharpe": 0.2},
        },
        "error.pkl": {
            "created_at": "2026-03-16T07:00:00",
            "metrics_holdout": {"profit_net": 30.0, "sharpe": 0.3},
            model_eval_runtime.TAB5_HOLDOUT_RANKING_KEY: {
                "status": "error",
                "error": "boom",
            },
        },
    }

    for filename, meta in paths.items():
        model_path = tmp_path / filename
        model_path.write_bytes(b"model")
        write_sidecar_model_meta(model_path, meta)

    records = discover_ranking_models(tmp_path)

    assert [record.model_path.name for record in records] == ["optimized.pkl", "fallback.pkl", "error.pkl"]


def test_ranking_task_writes_tab5_holdout_ranking_fields(monkeypatch):
    tmp_path = Path(".codex_test_tmp") / f"tab5_rank_task_{uuid.uuid4().hex}"
    tmp_path.mkdir(parents=True, exist_ok=True)

    raw = generate_synthetic_data(n_samples=260, noise_level=0.05)
    features_path = tmp_path / "synthetic_ohlcv.csv"
    raw.to_csv(features_path, index=False)
    prepared = model_eval_runtime.load_prepared_evaluation_data(
        features_path,
        metadata={"classes": [0, 1, 2]},
    )
    trained_features = [str(name) for name in list(prepared.X_full.columns[:3])]
    monkeypatch.setattr(
        model_eval_runtime,
        "run_auto_threshold_search_from_context",
        lambda **kwargs: model_eval_runtime.AutoThresholdPayload(
            best_entry=float(kwargs["current_entry"]),
            best_exit=float(kwargs["current_exit"]),
            best_score=0.0,
            best_metrics=None,
        ),
    )

    model_path = tmp_path / "dummy_ternary.pkl"
    joblib.dump({"model": DummyRankingPredictor()}, model_path)
    write_sidecar_model_meta(
        model_path,
        {
            "created_at": "2026-03-16T10:00:00",
            "classes": [0, 1, 2],
            "trained_features": trained_features,
            "ternary_threshold_short": 0.45,
            "ternary_threshold_long": 0.45,
            "metrics_holdout": {"profit_net": 1.0},
        },
    )

    result = ModelRankingTab._task_compute_rankings(
        model_paths=[str(model_path)],
        data_path=str(features_path),
        fee_per_trade=0.0,
        current_entry=0.60,
        current_exit=0.70,
        full_recompute=True,
    )

    meta = read_sidecar_model_meta(model_path)
    ranking = meta[model_eval_runtime.TAB5_HOLDOUT_RANKING_KEY]

    assert result["updated"] == 1
    assert result["failures"] == 0
    assert ranking["status"] == "ok"
    assert ranking["schema_version"] == model_eval_runtime.TAB5_HOLDOUT_RANKING_SCHEMA_VERSION
    assert ranking["scope"] == "holdout"
    assert ranking["csv_path"].endswith("synthetic_ohlcv.csv")
    assert ranking["threshold_source"] == "model"
    assert ranking["label_mode"] == "ternary_mapped"
    assert ranking["feature_contract"]["count"] == 3
    assert ranking["model_thresholds"]["short"] == pytest.approx(0.45)
    assert ranking["model_thresholds"]["long"] == pytest.approx(0.45)
    assert ranking["holdout_base"]["entry_threshold"] == pytest.approx(0.60)
    assert ranking["holdout_base"]["exit_threshold"] == pytest.approx(0.70)
    assert ranking["base_profit_h"] == pytest.approx(ranking["holdout_base"]["profit_h"])
    assert ranking["holdout_optimized"]["source"] == "holdout_auto_threshold_search"
    assert ranking["profit_h"] == pytest.approx(ranking["holdout_optimized"]["profit_h"])
    assert isinstance(ranking["entry_threshold"], float)
    assert isinstance(ranking["exit_threshold"], float)
    assert "profit_h" in ranking
    assert "max_dd_h" in ranking
    assert "trades_h" in ranking


def test_manual_recompute_skips_fresh_ok_models(monkeypatch, qapp):
    tmp_path = Path(".codex_test_tmp") / f"tab5_rank_manual_{uuid.uuid4().hex}"
    tmp_path.mkdir(parents=True, exist_ok=True)
    csv_path = tmp_path / "features.csv"
    csv_path.write_text("a,b\n1,2\n3,4\n", encoding="utf-8")

    fresh_model = tmp_path / "fresh.pkl"
    fresh_model.write_bytes(b"fresh")
    fresh_meta = {"created_at": "2026-03-16T10:00:00"}
    fresh_meta[model_eval_runtime.TAB5_HOLDOUT_RANKING_KEY] = model_eval_runtime.build_tab5_holdout_ranking_payload(
        data_path=csv_path,
        fee_per_trade=0.25,
        entry_threshold=0.34,
        exit_threshold=0.41,
        metrics={"profit_net": 12.5, "max_dd": -5.0, "trades": 9},
        status="ok",
    )
    write_sidecar_model_meta(fresh_model, fresh_meta)

    pending_model = tmp_path / "pending.pkl"
    pending_model.write_bytes(b"pending")
    write_sidecar_model_meta(
        pending_model,
        {
            "created_at": "2026-03-16T09:00:00",
            "metrics_holdout": {"profit_net": 1.0},
        },
    )

    monkeypatch.setattr("ibkr_trading_bot.gui.tab_model_ranking.DEFAULT_MODEL_DIR", tmp_path)

    tab = ModelRankingTab()
    try:
        captured = {}
        context = {
            "data_path": str(csv_path),
            "fee_per_trade": 0.25,
            "entry_threshold": 0.60,
            "exit_threshold": 0.70,
        }
        monkeypatch.setattr(tab, "_current_eval_context", lambda: context)
        monkeypatch.setattr(
            tab,
            "_start_batch_worker",
            lambda **kwargs: captured.update(kwargs),
        )

        tab._on_recompute_profit_opt_clicked()

        assert [record.model_path.name for record in captured["records"]] == ["pending.pkl"]
        assert captured["context"] == context
        assert captured["full_recompute"] is False
    finally:
        tab.close()


def test_recompute_profit_opt_uses_only_checked_pending_models(monkeypatch, qapp):
    tmp_path = Path(".codex_test_tmp") / f"tab5_rank_checked_manual_{uuid.uuid4().hex}"
    tmp_path.mkdir(parents=True, exist_ok=True)
    csv_path = tmp_path / "features.csv"
    csv_path.write_text("a,b\n1,2\n3,4\n", encoding="utf-8")

    fresh_model = tmp_path / "fresh.pkl"
    fresh_model.write_bytes(b"fresh")
    fresh_meta = {"created_at": "2026-03-16T10:00:00"}
    fresh_meta[model_eval_runtime.TAB5_HOLDOUT_RANKING_KEY] = model_eval_runtime.build_tab5_holdout_ranking_payload(
        data_path=csv_path,
        fee_per_trade=0.25,
        entry_threshold=0.34,
        exit_threshold=0.41,
        metrics={"profit_net": 12.5, "max_dd": -5.0, "trades": 9},
        status="ok",
    )
    write_sidecar_model_meta(fresh_model, fresh_meta)

    pending_model = tmp_path / "pending.pkl"
    pending_model.write_bytes(b"pending")
    write_sidecar_model_meta(
        pending_model,
        {
            "created_at": "2026-03-16T09:00:00",
            "metrics_holdout": {"profit_net": 1.0},
        },
    )

    other_pending_model = tmp_path / "other_pending.pkl"
    other_pending_model.write_bytes(b"other")
    write_sidecar_model_meta(
        other_pending_model,
        {
            "created_at": "2026-03-16T08:00:00",
            "metrics_holdout": {"profit_net": 2.0},
        },
    )

    monkeypatch.setattr("ibkr_trading_bot.gui.tab_model_ranking.DEFAULT_MODEL_DIR", tmp_path)

    tab = ModelRankingTab()
    try:
        captured = {}
        context = {
            "data_path": str(csv_path),
            "fee_per_trade": 0.25,
            "entry_threshold": 0.60,
            "exit_threshold": 0.70,
        }
        monkeypatch.setattr(tab, "_current_eval_context", lambda: context)
        monkeypatch.setattr(
            tab,
            "_start_batch_worker",
            lambda **kwargs: captured.update(kwargs),
        )

        for row in range(tab.tbl.rowCount()):
            model_item = tab.tbl.item(row, 0)
            check_item = tab.tbl.item(row, tab_model_ranking_module.COL_CHECK)
            if model_item is None or check_item is None:
                continue
            if model_item.text() == "pending.pkl":
                check_item.setCheckState(Qt.Checked)
                break
        qapp.processEvents()

        tab._on_recompute_profit_opt_clicked()

        assert [record.model_path.name for record in captured["records"]] == ["pending.pkl"]
        assert captured["context"] == context
        assert captured["full_recompute"] is False
    finally:
        tab.close()


def test_incremental_recompute_skips_fresh_ok_models(monkeypatch, qapp):
    tmp_path = Path(".codex_test_tmp") / f"tab5_rank_incremental_{uuid.uuid4().hex}"
    tmp_path.mkdir(parents=True, exist_ok=True)
    csv_path = tmp_path / "features.csv"
    csv_path.write_text("a,b\n1,2\n3,4\n", encoding="utf-8")

    fresh_model = tmp_path / "fresh.pkl"
    fresh_model.write_bytes(b"fresh")
    fresh_meta = {"created_at": "2026-03-16T10:00:00"}
    fresh_meta[model_eval_runtime.TAB5_HOLDOUT_RANKING_KEY] = model_eval_runtime.build_tab5_holdout_ranking_payload(
        data_path=csv_path,
        fee_per_trade=0.25,
        entry_threshold=0.34,
        exit_threshold=0.41,
        metrics={"profit_net": 12.5, "max_dd": -5.0, "trades": 9},
        status="ok",
    )
    write_sidecar_model_meta(fresh_model, fresh_meta)

    pending_model = tmp_path / "pending.pkl"
    pending_model.write_bytes(b"pending")
    write_sidecar_model_meta(
        pending_model,
        {
            "created_at": "2026-03-16T09:00:00",
            "metrics_holdout": {"profit_net": 1.0},
        },
    )

    monkeypatch.setattr("ibkr_trading_bot.gui.tab_model_ranking.DEFAULT_MODEL_DIR", tmp_path)

    tab = ModelRankingTab()
    try:
        captured = {}
        context = {
            "data_path": str(csv_path),
            "fee_per_trade": 0.25,
            "entry_threshold": 0.60,
            "exit_threshold": 0.70,
        }
        monkeypatch.setattr(tab, "_current_eval_context", lambda: context)
        monkeypatch.setattr(
            tab,
            "_start_batch_worker",
            lambda **kwargs: captured.update(kwargs),
        )

        tab._start_incremental_if_needed()

        assert [record.model_path.name for record in captured["records"]] == ["pending.pkl"]
        assert captured["context"] == context
        assert captured["full_recompute"] is False
    finally:
        tab.close()


def test_ranking_task_prepares_datasets_per_model_contract(monkeypatch, tmp_path: Path):
    model_a = tmp_path / "model_a.pkl"
    model_b = tmp_path / "model_b.pkl"
    model_c = tmp_path / "model_c.pkl"
    data_path = tmp_path / "dataset.csv"
    for path in (model_a, model_b, model_c):
        path.write_bytes(b"model")
    data_path.write_text("timestamp,open,high,low,close,volume\n1,1,1,1,1,1\n", encoding="utf-8")

    metas = {
        str(model_a): {
            "trained_features": ["feat_a"],
            "class_to_dir": {0: "SHORT", 1: "HOLD", 2: "LONG"},
            "label_horizon_bars": 8,
            "label_take_profit_bps": 50.0,
            "label_stop_loss_bps": 40.0,
        },
        str(model_b): {
            "trained_features": ["feat_a"],
            "class_to_dir": {0: "SHORT", 1: "HOLD", 2: "LONG"},
            "label_horizon_bars": 8,
            "label_take_profit_bps": 50.0,
            "label_stop_loss_bps": 40.0,
        },
        str(model_c): {
            "trained_features": ["feat_a"],
            "class_to_dir": {0: "SHORT", 1: "HOLD", 2: "LONG"},
            "label_horizon_bars": 12,
            "label_take_profit_bps": 60.0,
            "label_stop_loss_bps": 45.0,
        },
    }
    load_calls: list[tuple[int, float, float]] = []
    search_calls: list[dict[str, object]] = []
    recalc_calls: list[dict[str, object]] = []

    def fake_load_predictor_with_merged_meta(model_path):
        meta = dict(metas[str(model_path)])
        return model_eval_runtime.LoadedPredictor(
            predictor=DummyRankingPredictor(),
            metadata=meta,
            model_path=str(model_path),
        )

    def fake_load_prepared_evaluation_data(data_path, metadata=None, progress_cb=None):
        load_calls.append(
            (
                int((metadata or {}).get("label_horizon_bars") or 0),
                float((metadata or {}).get("label_take_profit_bps") or 0.0),
                float((metadata or {}).get("label_stop_loss_bps") or 0.0),
            )
        )
        return model_eval_runtime.PreparedEvaluationData(
            data_path=str(data_path),
            X_full=pd.DataFrame({"feat_a": [1.0]}),
            y_true_full=np.asarray([1]),
            df_for_metrics_full=pd.DataFrame({"close": [100.0]}),
        )

    def fake_run_model_evaluation(**kwargs):
        return model_eval_runtime.EvaluationPayload(
            X_current=pd.DataFrame({"feat_a": [1.0]}),
            y_true_current=np.asarray([1]),
            df_current=pd.DataFrame({"close": [100.0]}),
            close_series=pd.Series([100.0]),
            confidence_arr=np.asarray([0.9]),
            y_pred_raw=np.asarray([1]),
            y_pred_used=np.asarray([1]),
            results={"profit_net": 10.0, "max_dd": -1.0, "trades": 1},
            scope_info={"mode": "holdout", "applied_rows": 1, "total_rows": 1},
            threshold_source="model",
            thr_short=0.45,
            thr_long=0.55,
            entry_threshold=0.6,
            exit_threshold=0.7,
            exit_policy="hold_to_flip",
        )

    monkeypatch.setattr(model_eval_runtime, "load_predictor_with_merged_meta", fake_load_predictor_with_merged_meta)
    monkeypatch.setattr(model_eval_runtime, "load_prepared_evaluation_data", fake_load_prepared_evaluation_data)
    monkeypatch.setattr(model_eval_runtime, "run_model_evaluation", fake_run_model_evaluation)
    monkeypatch.setattr(
        model_eval_runtime,
        "run_auto_threshold_search_from_context",
        lambda **kwargs: (
            search_calls.append(kwargs),
            model_eval_runtime.AutoThresholdPayload(
                best_entry=0.61,
                best_exit=0.72,
                best_score=12.0,
                best_metrics={"profit_net": 12.0, "trades": 1},
            ),
        )[1],
    )
    monkeypatch.setattr(
        model_eval_runtime,
        "recalculate_metrics_from_predictions",
        lambda **kwargs: (
            recalc_calls.append(kwargs),
            (np.asarray([1]), {"profit_net": 12.0, "max_dd": -1.0, "trades": 1}),
        )[1],
    )
    monkeypatch.setattr(tab_model_ranking_module, "read_sidecar_model_meta", lambda path: {})
    monkeypatch.setattr(tab_model_ranking_module, "write_sidecar_model_meta", lambda path, meta: None)

    result = ModelRankingTab._task_compute_rankings(
        model_paths=[str(model_a), str(model_b), str(model_c)],
        data_path=str(data_path),
        fee_per_trade=0.25,
        current_entry=0.6,
        current_exit=0.7,
        full_recompute=True,
    )

    assert result["updated"] == 3
    assert result["failures"] == 0
    assert load_calls == [
        (8, 50.0, 40.0),
        (12, 60.0, 45.0),
    ]
    assert len(search_calls) == 3
    assert len(recalc_calls) == 3
    assert all(kwargs["exit_policy"] == "hold_to_flip" for kwargs in search_calls)
    assert all(kwargs["exit_policy"] == "hold_to_flip" for kwargs in recalc_calls)


def test_recompute_profit_opt_uses_only_visible_pending_models_when_filter_is_active(monkeypatch, qapp):
    tmp_path = Path(".codex_test_tmp") / f"tab5_rank_recompute_filtered_{uuid.uuid4().hex}"
    tmp_path.mkdir(parents=True, exist_ok=True)
    csv_path = tmp_path / "features.csv"
    csv_path.write_text("a,b\n1,2\n3,4\n", encoding="utf-8")

    _write_ranked_model(
        tmp_path,
        "tight_pending.pkl",
        csv_path=csv_path,
        n_total_bars=44268,
        ranking_status="error",
        per_class_3={"-1": {"recall": 0.31}, "1": {"recall": 0.29}},
    )
    _write_ranked_model(
        tmp_path,
        "wide_pending.pkl",
        csv_path=csv_path,
        n_total_bars=44268,
        ranking_status="error",
        per_class_3={"-1": {"recall": 0.61}, "1": {"recall": 0.09}},
    )

    monkeypatch.setattr("ibkr_trading_bot.gui.tab_model_ranking.DEFAULT_MODEL_DIR", tmp_path)

    tab = ModelRankingTab()
    try:
        captured = {}
        context = {
            "data_path": str(csv_path),
            "fee_per_trade": 0.25,
            "entry_threshold": 0.60,
            "exit_threshold": 0.70,
        }
        monkeypatch.setattr(tab, "_current_eval_context", lambda: context)
        monkeypatch.setattr(tab, "_start_batch_worker", lambda **kwargs: captured.update(kwargs))

        tab.set_bias_filter("tight")
        qapp.processEvents()
        tab._on_recompute_profit_opt_clicked()

        assert [record.model_path.name for record in captured["records"]] == ["tight_pending.pkl"]
        assert captured["context"] == context
        assert captured["full_recompute"] is False
    finally:
        tab.close()


def test_recompute_profit_opt_uses_only_mode_filtered_pending_models(monkeypatch, qapp):
    tmp_path = Path(".codex_test_tmp") / f"tab5_rank_recompute_mode_filter_{uuid.uuid4().hex}"
    tmp_path.mkdir(parents=True, exist_ok=True)
    csv_path = tmp_path / "features.csv"
    csv_path.write_text("a,b\n1,2\n3,4\n", encoding="utf-8")

    _write_ranked_model(
        tmp_path,
        "filtered_pending.pkl",
        csv_path=csv_path,
        n_total_bars=44268,
        ranking_status="error",
        training_mode="explore",
    )
    _write_ranked_model(
        tmp_path,
        "other_pending.pkl",
        csv_path=csv_path,
        n_total_bars=44268,
        ranking_status="error",
        training_mode="refresh",
    )

    monkeypatch.setattr("ibkr_trading_bot.gui.tab_model_ranking.DEFAULT_MODEL_DIR", tmp_path)

    tab = ModelRankingTab()
    try:
        captured = {}
        context = {
            "data_path": str(csv_path),
            "fee_per_trade": 0.25,
            "entry_threshold": 0.60,
            "exit_threshold": 0.70,
        }
        monkeypatch.setattr(tab, "_current_eval_context", lambda: context)
        monkeypatch.setattr(tab, "_start_batch_worker", lambda **kwargs: captured.update(kwargs))

        tab.set_mode_filter(["Explore"])
        qapp.processEvents()
        tab._on_recompute_profit_opt_clicked()

        assert [record.model_path.name for record in captured["records"]] == ["filtered_pending.pkl"]
        assert captured["context"] == context
        assert captured["full_recompute"] is False
    finally:
        tab.close()


def test_recompute_profit_opt_ignores_row_selection_when_filter_is_active(monkeypatch, qapp):
    tmp_path = Path(".codex_test_tmp") / f"tab5_rank_recompute_row_selection_{uuid.uuid4().hex}"
    tmp_path.mkdir(parents=True, exist_ok=True)
    csv_path = tmp_path / "features.csv"
    csv_path.write_text("a,b\n1,2\n3,4\n", encoding="utf-8")

    _write_ranked_model(
        tmp_path,
        "selected_pending.pkl",
        csv_path=csv_path,
        n_total_bars=44268,
        ranking_status="error",
        per_class_3={"-1": {"recall": 0.31}, "1": {"recall": 0.29}},
    )
    _write_ranked_model(
        tmp_path,
        "other_filtered_pending.pkl",
        csv_path=csv_path,
        n_total_bars=44268,
        ranking_status="error",
        per_class_3={"-1": {"recall": 0.32}, "1": {"recall": 0.30}},
    )
    _write_ranked_model(
        tmp_path,
        "filtered_out_pending.pkl",
        csv_path=csv_path,
        n_total_bars=44268,
        ranking_status="error",
        per_class_3={"-1": {"recall": 0.61}, "1": {"recall": 0.09}},
    )

    monkeypatch.setattr("ibkr_trading_bot.gui.tab_model_ranking.DEFAULT_MODEL_DIR", tmp_path)

    tab = ModelRankingTab()
    try:
        captured = {}
        context = {
            "data_path": str(csv_path),
            "fee_per_trade": 0.25,
            "entry_threshold": 0.60,
            "exit_threshold": 0.70,
        }
        monkeypatch.setattr(tab, "_current_eval_context", lambda: context)
        monkeypatch.setattr(tab, "_start_batch_worker", lambda **kwargs: captured.update(kwargs))

        tab.set_bias_filter("tight")
        qapp.processEvents()
        selected_row = None
        for row in range(tab.tbl.rowCount()):
            item = tab.tbl.item(row, 0)
            if item is not None and item.text() == "selected_pending.pkl":
                selected_row = row
                break
        assert selected_row is not None
        tab.tbl.selectRow(selected_row)
        qapp.processEvents()

        tab._on_recompute_profit_opt_clicked()

        assert {record.model_path.name for record in captured["records"]} == {
            "selected_pending.pkl",
            "other_filtered_pending.pkl",
        }
        assert captured["context"] == context
        assert captured["full_recompute"] is False
    finally:
        tab.close()


def test_recompute_profit_opt_prefers_checked_models_over_filtered_scope(monkeypatch, qapp):
    tmp_path = Path(".codex_test_tmp") / f"tab5_rank_recompute_checked_{uuid.uuid4().hex}"
    tmp_path.mkdir(parents=True, exist_ok=True)
    csv_path = tmp_path / "features.csv"
    csv_path.write_text("a,b\n1,2\n3,4\n", encoding="utf-8")

    _write_ranked_model(
        tmp_path,
        "checked_pending.pkl",
        csv_path=csv_path,
        n_total_bars=44268,
        ranking_status="error",
        per_class_3={"-1": {"recall": 0.31}, "1": {"recall": 0.29}},
    )
    _write_ranked_model(
        tmp_path,
        "other_filtered_pending.pkl",
        csv_path=csv_path,
        n_total_bars=44268,
        ranking_status="error",
        per_class_3={"-1": {"recall": 0.32}, "1": {"recall": 0.30}},
    )
    _write_ranked_model(
        tmp_path,
        "filtered_out_pending.pkl",
        csv_path=csv_path,
        n_total_bars=44268,
        ranking_status="error",
        per_class_3={"-1": {"recall": 0.61}, "1": {"recall": 0.09}},
    )

    monkeypatch.setattr("ibkr_trading_bot.gui.tab_model_ranking.DEFAULT_MODEL_DIR", tmp_path)

    tab = ModelRankingTab()
    try:
        captured = {}
        context = {
            "data_path": str(csv_path),
            "fee_per_trade": 0.25,
            "entry_threshold": 0.60,
            "exit_threshold": 0.70,
        }
        monkeypatch.setattr(tab, "_current_eval_context", lambda: context)
        monkeypatch.setattr(tab, "_start_batch_worker", lambda **kwargs: captured.update(kwargs))

        tab.set_bias_filter("tight")
        qapp.processEvents()

        for row in range(tab.tbl.rowCount()):
            model_item = tab.tbl.item(row, 0)
            check_item = tab.tbl.item(row, tab_model_ranking_module.COL_CHECK)
            if model_item is None or check_item is None:
                continue
            if model_item.text() == "checked_pending.pkl":
                check_item.setCheckState(Qt.Checked)
                break
        qapp.processEvents()

        tab._on_recompute_profit_opt_clicked()

        assert [record.model_path.name for record in captured["records"]] == ["checked_pending.pkl"]
        assert captured["context"] == context
        assert captured["full_recompute"] is False
    finally:
        tab.close()


def test_context_change_does_not_auto_start_ranking_batch(monkeypatch, qapp):
    tmp_path = Path(".codex_test_tmp") / f"tab5_rank_context_change_{uuid.uuid4().hex}"
    tmp_path.mkdir(parents=True, exist_ok=True)
    csv_path = tmp_path / "features.csv"
    csv_path.write_text("a,b\n1,2\n3,4\n", encoding="utf-8")

    pending_model = tmp_path / "pending.pkl"
    pending_model.write_bytes(b"pending")
    write_sidecar_model_meta(
        pending_model,
        {
            "created_at": "2026-03-16T09:00:00",
            "metrics_holdout": {"profit_net": 1.0},
            "instrument": "GC",
            "exchange": "COMEX",
            "timeframe": "5m",
            "n_total_bars": 2,
            "estimator_name": "hgbt",
            "label_horizon_bars": 8,
            "label_take_profit_bps": 50.0,
            "label_stop_loss_bps": 50.0,
        },
    )

    monkeypatch.setattr("ibkr_trading_bot.gui.tab_model_ranking.DEFAULT_MODEL_DIR", tmp_path)

    tab = ModelRankingTab()
    try:
        calls = {"auto": 0}
        monkeypatch.setattr(tab, "_refresh_list", lambda force=False: False)
        monkeypatch.setattr(
            tab,
            "_current_eval_context",
            lambda: {
                "data_path": str(csv_path),
                "fee_per_trade": 0.25,
                "entry_threshold": 0.55,
                "exit_threshold": 0.60,
            },
        )
        monkeypatch.setattr(tab, "_start_incremental_if_needed", lambda: calls.__setitem__("auto", calls["auto"] + 1))
        tab._last_context_fingerprint = None

        tab._tick()

        assert calls["auto"] == 0
        assert "klikni 'Prepocitat neaktualni (H opt)'" in tab.lbl_status.text()
    finally:
        tab.close()


def test_ranking_table_shows_training_mode_and_persists_note(monkeypatch, qapp):
    tmp_path = Path(".codex_test_tmp") / f"tab5_rank_note_{uuid.uuid4().hex}"
    tmp_path.mkdir(parents=True, exist_ok=True)
    csv_path = tmp_path / "features.csv"
    csv_path.write_text("a,b\n1,2\n3,4\n", encoding="utf-8")

    model_path = _write_ranked_model(
        tmp_path,
        "annotated.pkl",
        csv_path=csv_path,
        n_total_bars=44268,
        training_mode="standard",
        user_note="puvodni poznamka",
    )

    monkeypatch.setattr("ibkr_trading_bot.gui.tab_model_ranking.DEFAULT_MODEL_DIR", tmp_path)

    tab = ModelRankingTab()
    try:
        assert tab.tbl.item(0, COL_MODE).text() == "Refine"
        assert tab.tbl.item(0, COL_NOTE).text() == "puvodni poznamka"

        note_item = tab.tbl.item(0, COL_NOTE)
        assert note_item is not None
        note_item.setText("  kandidat k revizi  ")
        qapp.processEvents()

        reloaded = read_sidecar_model_meta(model_path)
        assert reloaded["model_ranking_note"] == "kandidat k revizi"
        assert tab.tbl.item(0, COL_NOTE).text() == "kandidat k revizi"
    finally:
        tab.close()


def test_ranking_table_prefers_v3_base_and_optimized_payload_values(monkeypatch, qapp):
    tmp_path = Path(".codex_test_tmp") / f"tab5_rank_payload_{uuid.uuid4().hex}"
    tmp_path.mkdir(parents=True, exist_ok=True)
    csv_path = tmp_path / "features.csv"
    csv_path.write_text("a,b\n1,2\n3,4\n", encoding="utf-8")

    model_path = _write_ranked_model(
        tmp_path,
        "payload_pref.pkl",
        csv_path=csv_path,
        n_total_bars=44268,
        base_profit=999.0,
        optimized_profit=555.0,
        trades_h=77.0,
    )
    meta = read_sidecar_model_meta(model_path)
    meta[model_eval_runtime.TAB5_HOLDOUT_RANKING_KEY] = model_eval_runtime.build_tab5_holdout_ranking_payload(
        data_path=csv_path,
        fee_per_trade=0.25,
        metadata=meta,
        base_entry_threshold=0.52,
        base_exit_threshold=0.63,
        base_metrics={"profit_net": 12.0, "trades": 90, "max_dd": -3.0},
        optimized_entry_threshold=0.41,
        optimized_exit_threshold=0.57,
        optimized_metrics={"profit_net": 34.0, "trades": 70, "max_dd": -2.0},
        entry_threshold=0.41,
        exit_threshold=0.57,
        metrics={"profit_net": 34.0, "trades": 70, "max_dd": -2.0},
        status="ok",
    )
    write_sidecar_model_meta(model_path, meta)

    monkeypatch.setattr("ibkr_trading_bot.gui.tab_model_ranking.DEFAULT_MODEL_DIR", tmp_path)

    tab = ModelRankingTab()
    try:
        assert tab.tbl.item(0, COL_PROFIT_OPT).text() == "34.00"
        assert tab.tbl.item(0, COL_PROFIT).text() == "12.00"
    finally:
        tab.close()


def test_ranking_table_shows_stability_score_and_tooltip(monkeypatch, qapp):
    tmp_path = Path(".codex_test_tmp") / f"tab5_rank_stability_{uuid.uuid4().hex}"
    tmp_path.mkdir(parents=True, exist_ok=True)
    csv_path = tmp_path / "features.csv"
    csv_path.write_text("a,b\n1,2\n3,4\n", encoding="utf-8")

    _write_ranked_model(
        tmp_path,
        "stable.pkl",
        csv_path=csv_path,
        n_total_bars=44268,
        trained_features=["feat_a", "feat_b", "feat_c"],
        feature_stability={
            "feat_a": {"mean": 0.90, "std": 0.09, "folds_present": 5},
            "feat_b": {"mean": 0.60, "std": 0.12, "folds_present": 5},
            "feat_c": {"mean": 0.40, "std": 0.24, "folds_present": 5},
        },
        feature_stability_score={
            "feat_a": 0.80,
            "feat_b": 0.60,
            "feat_c": 0.40,
        },
    )

    monkeypatch.setattr("ibkr_trading_bot.gui.tab_model_ranking.DEFAULT_MODEL_DIR", tmp_path)

    tab = ModelRankingTab()
    try:
        item = tab.tbl.item(0, COL_STABILITY)
        assert item is not None
        assert tab.tbl.horizontalHeaderItem(COL_STABILITY).text() == "Stability"
        assert item.text() == "0.600"
        assert "Top 5 stable features" in item.toolTip()
        assert "feat_a: 0.800" in item.toolTip()
        assert "Features: 3 -> 3" in item.toolTip()
    finally:
        tab.close()


def test_ranking_table_shows_bias_score_column(monkeypatch, qapp):
    tmp_path = Path(".codex_test_tmp") / f"tab5_rank_bias_{uuid.uuid4().hex}"
    tmp_path.mkdir(parents=True, exist_ok=True)
    csv_path = tmp_path / "features.csv"
    csv_path.write_text("a,b\n1,2\n3,4\n", encoding="utf-8")

    _write_ranked_model(
        tmp_path,
        "biased.pkl",
        csv_path=csv_path,
        n_total_bars=44268,
        per_class_3={
            "-1": {"recall": 0.61},
            "1": {"recall": 0.09},
        },
    )

    monkeypatch.setattr("ibkr_trading_bot.gui.tab_model_ranking.DEFAULT_MODEL_DIR", tmp_path)

    tab = ModelRankingTab()
    try:
        item = tab.tbl.item(0, COL_BIAS)
        assert item is not None
        assert tab.tbl.horizontalHeaderItem(COL_BIAS).text() == "Bias S-L"
        assert item.text() == "+0.520"
        assert "SHORT recall: 0.610" in item.toolTip()
        assert "LONG recall: 0.090" in item.toolTip()
    finally:
        tab.close()


def test_ranking_bias_filter_applies_absolute_threshold(monkeypatch, qapp):
    tmp_path = Path(".codex_test_tmp") / f"tab5_rank_bias_filter_{uuid.uuid4().hex}"
    tmp_path.mkdir(parents=True, exist_ok=True)
    csv_path = tmp_path / "features.csv"
    csv_path.write_text("a,b\n1,2\n3,4\n", encoding="utf-8")

    _write_ranked_model(
        tmp_path,
        "tight.pkl",
        csv_path=csv_path,
        n_total_bars=44268,
        per_class_3={"-1": {"recall": 0.31}, "1": {"recall": 0.29}},
    )
    _write_ranked_model(
        tmp_path,
        "moderate.pkl",
        csv_path=csv_path,
        n_total_bars=44268,
        per_class_3={"-1": {"recall": 0.40}, "1": {"recall": 0.20}},
    )
    _write_ranked_model(
        tmp_path,
        "wide.pkl",
        csv_path=csv_path,
        n_total_bars=44268,
        per_class_3={"-1": {"recall": 0.61}, "1": {"recall": 0.09}},
    )

    monkeypatch.setattr("ibkr_trading_bot.gui.tab_model_ranking.DEFAULT_MODEL_DIR", tmp_path)

    tab = ModelRankingTab()
    try:
        assert tab.tbl.rowCount() == 3

        tab.set_bias_filter("tight")
        qapp.processEvents()
        assert tab.tbl.rowCount() == 1
        assert tab.tbl.item(0, 0).text() == "tight.pkl"

        tab.set_bias_filter("moderate")
        qapp.processEvents()
        assert tab.tbl.rowCount() == 2
        assert {tab.tbl.item(row, 0).text() for row in range(tab.tbl.rowCount())} == {
            "tight.pkl",
            "moderate.pkl",
        }

        tab.set_bias_filter(None)
        qapp.processEvents()
        assert tab.tbl.rowCount() == 3
    finally:
        tab.close()


def test_ranking_bias_column_sorts_by_absolute_value(monkeypatch, qapp):
    tmp_path = Path(".codex_test_tmp") / f"tab5_rank_bias_sort_{uuid.uuid4().hex}"
    tmp_path.mkdir(parents=True, exist_ok=True)
    csv_path = tmp_path / "features.csv"
    csv_path.write_text("a,b\n1,2\n3,4\n", encoding="utf-8")

    _write_ranked_model(
        tmp_path,
        "large_negative.pkl",
        csv_path=csv_path,
        n_total_bars=44268,
        per_class_3={"-1": {"recall": 0.10}, "1": {"recall": 0.60}},
    )
    _write_ranked_model(
        tmp_path,
        "small_positive.pkl",
        csv_path=csv_path,
        n_total_bars=44268,
        per_class_3={"-1": {"recall": 0.31}, "1": {"recall": 0.29}},
    )
    _write_ranked_model(
        tmp_path,
        "medium_positive.pkl",
        csv_path=csv_path,
        n_total_bars=44268,
        per_class_3={"-1": {"recall": 0.40}, "1": {"recall": 0.20}},
    )

    monkeypatch.setattr("ibkr_trading_bot.gui.tab_model_ranking.DEFAULT_MODEL_DIR", tmp_path)

    tab = ModelRankingTab()
    try:
        tab.tbl.sortItems(COL_BIAS, Qt.AscendingOrder)
        qapp.processEvents()

        assert [tab.tbl.item(row, 0).text() for row in range(tab.tbl.rowCount())] == [
            "small_positive.pkl",
            "medium_positive.pkl",
            "large_negative.pkl",
        ]
    finally:
        tab.close()


def test_ranking_stability_filter_applies_thresholds(monkeypatch, qapp):
    tmp_path = Path(".codex_test_tmp") / f"tab5_rank_stability_filter_{uuid.uuid4().hex}"
    tmp_path.mkdir(parents=True, exist_ok=True)
    csv_path = tmp_path / "features.csv"
    csv_path.write_text("a,b\n1,2\n3,4\n", encoding="utf-8")

    _write_ranked_model(
        tmp_path,
        "good.pkl",
        csv_path=csv_path,
        n_total_bars=44268,
        optimized_profit=610.0,
        trained_features=["feat_a"],
        feature_stability={"feat_a": {"mean": 0.55, "std": 0.05, "folds_present": 5}},
        feature_stability_score={"feat_a": 0.45},
    )
    _write_ranked_model(
        tmp_path,
        "excellent.pkl",
        csv_path=csv_path,
        n_total_bars=44268,
        optimized_profit=620.0,
        trained_features=["feat_a"],
        feature_stability={"feat_a": {"mean": 0.70, "std": 0.07, "folds_present": 5}},
        feature_stability_score={"feat_a": 0.55},
    )
    _write_ranked_model(
        tmp_path,
        "edge.pkl",
        csv_path=csv_path,
        n_total_bars=44268,
        optimized_profit=630.0,
        trained_features=["feat_a"],
        feature_stability={"feat_a": {"mean": 0.50, "std": 0.10, "folds_present": 5}},
        feature_stability_score={"feat_a": 0.40},
    )
    _write_ranked_model(
        tmp_path,
        "legacy.pkl",
        csv_path=csv_path,
        n_total_bars=44268,
        optimized_profit=640.0,
        trained_features=["feat_a"],
    )

    monkeypatch.setattr("ibkr_trading_bot.gui.tab_model_ranking.DEFAULT_MODEL_DIR", tmp_path)

    tab = ModelRankingTab()
    try:
        assert tab.tbl.rowCount() == 4

        tab.set_stability_filter("good")
        qapp.processEvents()
        assert tab.tbl.rowCount() == 2
        assert {tab.tbl.item(row, 0).text() for row in range(tab.tbl.rowCount())} == {
            "good.pkl",
            "excellent.pkl",
        }

        tab.set_stability_filter("excellent")
        qapp.processEvents()
        assert tab.tbl.rowCount() == 1
        assert tab.tbl.item(0, 0).text() == "excellent.pkl"

        tab.set_stability_filter(None)
        qapp.processEvents()
        assert tab.tbl.rowCount() == 4
    finally:
        tab.close()


def test_ranking_mode_filter_applies_selected_modes(monkeypatch, qapp):
    tmp_path = Path(".codex_test_tmp") / f"tab5_rank_mode_filter_{uuid.uuid4().hex}"
    tmp_path.mkdir(parents=True, exist_ok=True)
    csv_path = tmp_path / "features.csv"
    csv_path.write_text("a,b\n1,2\n3,4\n", encoding="utf-8")

    _write_ranked_model(
        tmp_path,
        "alpha.pkl",
        csv_path=csv_path,
        n_total_bars=44268,
        optimized_profit=610.0,
        training_mode="explore",
    )
    _write_ranked_model(
        tmp_path,
        "beta.pkl",
        csv_path=csv_path,
        n_total_bars=44268,
        optimized_profit=620.0,
        training_mode="refresh",
    )
    _write_ranked_model(
        tmp_path,
        "gamma.pkl",
        csv_path=csv_path,
        n_total_bars=44268,
        optimized_profit=630.0,
        training_mode="explore",
    )

    monkeypatch.setattr("ibkr_trading_bot.gui.tab_model_ranking.DEFAULT_MODEL_DIR", tmp_path)

    tab = ModelRankingTab()
    try:
        assert tab.tbl.rowCount() == 3

        tab.set_mode_filter(["Explore"])
        qapp.processEvents()

        assert tab.tbl.rowCount() == 2
        assert {tab.tbl.item(row, 0).text() for row in range(tab.tbl.rowCount())} == {
            "alpha.pkl",
            "gamma.pkl",
        }
        assert tab._has_active_filters() is True

        tab.set_mode_filter(None)
        qapp.processEvents()

        assert tab.tbl.rowCount() == 3
    finally:
        tab.close()


def test_compare_action_prefers_checked_scope_over_selected_rows(monkeypatch, qapp):
    tmp_path = Path(".codex_test_tmp") / f"tab5_compare_scope_{uuid.uuid4().hex}"
    tmp_path.mkdir(parents=True, exist_ok=True)
    csv_path = tmp_path / "features.csv"
    csv_path.write_text("a,b\n1,2\n3,4\n", encoding="utf-8")

    _write_ranked_model(
        tmp_path,
        "selected_a.pkl",
        csv_path=csv_path,
        n_total_bars=44268,
        optimized_profit=510.0,
    )
    _write_ranked_model(
        tmp_path,
        "selected_b.pkl",
        csv_path=csv_path,
        n_total_bars=44268,
        optimized_profit=520.0,
    )
    _write_ranked_model(
        tmp_path,
        "checked_c.pkl",
        csv_path=csv_path,
        n_total_bars=44268,
        optimized_profit=530.0,
    )
    _write_ranked_model(
        tmp_path,
        "checked_d.pkl",
        csv_path=csv_path,
        n_total_bars=44268,
        optimized_profit=540.0,
    )

    monkeypatch.setattr("ibkr_trading_bot.gui.tab_model_ranking.DEFAULT_MODEL_DIR", tmp_path)

    tab = ModelRankingTab()
    try:
        selected_records = [record for record in tab.records if record.model_path.name in {"selected_a.pkl", "selected_b.pkl"}]
        checked_records = [record for record in tab.records if record.model_path.name in {"checked_c.pkl", "checked_d.pkl"}]
        captured: dict[str, object] = {}

        monkeypatch.setattr(
            tab,
            "_current_eval_context",
            lambda: {
                "data_path": str(csv_path),
                "fee_per_trade": 0.25,
                "entry_threshold": 0.55,
                "exit_threshold": 0.60,
            },
        )
        monkeypatch.setattr(tab, "_selected_records", lambda: list(selected_records))
        monkeypatch.setattr(tab, "_checked_records", lambda: list(checked_records))
        monkeypatch.setattr(
            tab,
            "_show_comparison_dialog",
            lambda records, context: captured.update(
                {
                    "records": [record.model_path.name for record in records],
                    "context": dict(context),
                }
            ),
        )

        tab._on_compare_selected_clicked()

        assert captured["records"] == ["checked_d.pkl", "checked_c.pkl"]
        assert captured["context"] == {
            "data_path": str(csv_path),
            "fee_per_trade": 0.25,
            "entry_threshold": 0.55,
            "exit_threshold": 0.60,
        }
        assert "oteviram porovnani pro 2 modelu (zatrzene)" in tab.lbl_status.text()
    finally:
        tab.close()


def test_compare_filtered_uses_filtered_scope_even_when_checked_exists(monkeypatch, qapp):
    tmp_path = Path(".codex_test_tmp") / f"tab5_compare_filtered_{uuid.uuid4().hex}"
    tmp_path.mkdir(parents=True, exist_ok=True)
    csv_path = tmp_path / "features.csv"
    csv_path.write_text("a,b\n1,2\n3,4\n", encoding="utf-8")

    _write_ranked_model(
        tmp_path,
        "alpha.pkl",
        csv_path=csv_path,
        n_total_bars=44268,
        estimator="lgb",
        horizon=12,
        tp_bps=40.0,
        sl_bps=30.0,
        optimized_profit=610.0,
    )
    _write_ranked_model(
        tmp_path,
        "beta.pkl",
        csv_path=csv_path,
        n_total_bars=44268,
        estimator="hgbt",
        horizon=16,
        tp_bps=40.0,
        sl_bps=30.0,
        optimized_profit=620.0,
    )
    _write_ranked_model(
        tmp_path,
        "balanced.pkl",
        csv_path=csv_path,
        n_total_bars=44268,
        estimator="lgb",
        horizon=12,
        tp_bps=40.0,
        sl_bps=30.0,
        candidate_selection_criterion="balanced",
        optimized_profit=630.0,
    )
    _write_ranked_model(
        tmp_path,
        "outside.pkl",
        csv_path=csv_path,
        n_total_bars=44268,
        estimator="rf",
        horizon=8,
        tp_bps=50.0,
        sl_bps=50.0,
        optimized_profit=640.0,
    )

    shortlist_dir = tmp_path / "auto_search"
    shortlist_dir.mkdir(parents=True, exist_ok=True)
    shortlist_path = shortlist_dir / "sample_shortlist.json"
    shortlist_path.write_text(
        json.dumps(
            {
                "version": 1,
                "mode": "refine",
                "created_at": "2026-05-27T12:00:00+00:00",
                "dataset_signature": {
                    "instrument": "GC",
                    "exchange": "COMEX",
                    "timeframe": "5m",
                    "n_total_bars": 44268,
                },
                "candidates": [
                    {"model": "lgb", "criterion": "balanced", "horizon": 12, "tp_bps": 40.0, "sl_bps": 30.0},
                    {"model": "hgbt", "criterion": "balanced", "horizon": 16, "tp_bps": 40.0, "sl_bps": 30.0},
                ],
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    monkeypatch.setattr("ibkr_trading_bot.gui.tab_model_ranking.DEFAULT_MODEL_DIR", tmp_path)

    tab = ModelRankingTab()
    try:
        context = {
            "data_path": str(csv_path),
            "fee_per_trade": 0.25,
            "entry_threshold": 0.55,
            "exit_threshold": 0.60,
        }
        captured: dict[str, object] = {}
        monkeypatch.setattr(tab, "_current_eval_context", lambda: context)
        monkeypatch.setattr(
            tab,
            "_show_comparison_dialog",
            lambda records, ctx: captured.update(
                {
                    "records": [record.model_path.name for record in records],
                    "context": dict(ctx),
                }
            ),
        )

        for row in range(tab.tbl.rowCount()):
            model_item = tab.tbl.item(row, 0)
            check_item = tab.tbl.item(row, tab_model_ranking_module.COL_CHECK)
            if model_item is None or check_item is None:
                continue
            if model_item.text() == "outside.pkl":
                check_item.setCheckState(Qt.Checked)
                break
        qapp.processEvents()

        tab.set_shortlist_filter(str(shortlist_path))
        qapp.processEvents()

        tab._on_compare_filtered_clicked()

        assert set(captured["records"]) == {"alpha.pkl", "beta.pkl", "balanced.pkl"}
        assert captured["context"] == context
        assert "oteviram porovnani pro 3 modelu (filtrovane)" in tab.lbl_status.text()
    finally:
        tab.close()


def test_compare_filtered_falls_back_to_latest_shortlist_without_active_filter(monkeypatch, qapp):
    tmp_path = Path(".codex_test_tmp") / f"tab5_compare_latest_shortlist_{uuid.uuid4().hex}"
    tmp_path.mkdir(parents=True, exist_ok=True)
    csv_path = tmp_path / "features.csv"
    csv_path.write_text("a,b\n1,2\n3,4\n", encoding="utf-8")

    _write_ranked_model(
        tmp_path,
        "latest_a.pkl",
        csv_path=csv_path,
        n_total_bars=44268,
        estimator="lgb",
        horizon=12,
        tp_bps=40.0,
        sl_bps=30.0,
        optimized_profit=610.0,
    )
    _write_ranked_model(
        tmp_path,
        "latest_b.pkl",
        csv_path=csv_path,
        n_total_bars=44268,
        estimator="hgbt",
        horizon=16,
        tp_bps=40.0,
        sl_bps=30.0,
        optimized_profit=620.0,
    )
    _write_ranked_model(
        tmp_path,
        "old_only.pkl",
        csv_path=csv_path,
        n_total_bars=44268,
        estimator="rf",
        horizon=8,
        tp_bps=50.0,
        sl_bps=50.0,
        optimized_profit=630.0,
    )

    shortlist_dir = tmp_path / "auto_search"
    shortlist_dir.mkdir(parents=True, exist_ok=True)
    (shortlist_dir / "older_shortlist.json").write_text(
        json.dumps(
            {
                "version": 1,
                "mode": "refine",
                "created_at": "2026-05-26T12:00:00+00:00",
                "dataset_signature": {
                    "instrument": "GC",
                    "exchange": "COMEX",
                    "timeframe": "5m",
                    "n_total_bars": 44268,
                },
                "candidates": [
                    {"model": "rf", "criterion": "balanced", "horizon": 8, "tp_bps": 50.0, "sl_bps": 50.0}
                ],
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    latest_shortlist_path = shortlist_dir / "latest_shortlist.json"
    latest_shortlist_path.write_text(
        json.dumps(
            {
                "version": 1,
                "mode": "refine",
                "created_at": "2026-05-27T12:00:00+00:00",
                "dataset_signature": {
                    "instrument": "GC",
                    "exchange": "COMEX",
                    "timeframe": "5m",
                    "n_total_bars": 44268,
                },
                "candidates": [
                    {"model": "lgb", "criterion": "balanced", "horizon": 12, "tp_bps": 40.0, "sl_bps": 30.0},
                    {"model": "hgbt", "criterion": "balanced", "horizon": 16, "tp_bps": 40.0, "sl_bps": 30.0},
                ],
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    monkeypatch.setattr("ibkr_trading_bot.gui.tab_model_ranking.DEFAULT_MODEL_DIR", tmp_path)

    tab = ModelRankingTab()
    try:
        captured: dict[str, object] = {}
        context = {
            "data_path": str(csv_path),
            "fee_per_trade": 0.25,
            "entry_threshold": 0.55,
            "exit_threshold": 0.60,
        }
        monkeypatch.setattr(tab, "_current_eval_context", lambda: context)
        monkeypatch.setattr(
            tab,
            "_show_comparison_dialog",
            lambda records, ctx: captured.update(
                {
                    "records": [record.model_path.name for record in records],
                    "context": dict(ctx),
                }
            ),
        )

        assert tab._has_active_filters() is False
        assert tab._active_shortlist_artifact() is None
        assert tab._latest_shortlist_artifact() is not None
        assert tab._latest_shortlist_artifact().path == latest_shortlist_path

        tab._on_compare_filtered_clicked()

        assert set(captured["records"]) == {"latest_a.pkl", "latest_b.pkl"}
        assert captured["context"] == context
        assert "oteviram porovnani pro 2 modelu (posledni shortlist)" in tab.lbl_status.text()
    finally:
        tab.close()


def test_latest_shortlist_prefill_does_not_activate_filtered_scope(monkeypatch, qapp):
    tmp_path = Path(".codex_test_tmp") / f"tab5_latest_shortlist_scope_{uuid.uuid4().hex}"
    tmp_path.mkdir(parents=True, exist_ok=True)
    csv_path = tmp_path / "features.csv"
    csv_path.write_text("a,b\n1,2\n3,4\n", encoding="utf-8")

    _write_ranked_model(
        tmp_path,
        "alpha.pkl",
        csv_path=csv_path,
        n_total_bars=44268,
        estimator="lgb",
        horizon=12,
        tp_bps=40.0,
        sl_bps=30.0,
        ranking_status="error",
    )
    _write_ranked_model(
        tmp_path,
        "beta.pkl",
        csv_path=csv_path,
        n_total_bars=44268,
        estimator="hgbt",
        horizon=16,
        tp_bps=40.0,
        sl_bps=30.0,
        ranking_status="error",
    )

    shortlist_dir = tmp_path / "auto_search"
    shortlist_dir.mkdir(parents=True, exist_ok=True)
    (shortlist_dir / "latest_shortlist.json").write_text(
        json.dumps(
            {
                "version": 1,
                "mode": "refine",
                "created_at": "2026-05-27T12:00:00+00:00",
                "dataset_signature": {
                    "instrument": "GC",
                    "exchange": "COMEX",
                    "timeframe": "5m",
                    "n_total_bars": 44268,
                },
                "candidates": [
                    {"model": "lgb", "criterion": "balanced", "horizon": 12, "tp_bps": 40.0, "sl_bps": 30.0},
                    {"model": "hgbt", "criterion": "balanced", "horizon": 16, "tp_bps": 40.0, "sl_bps": 30.0},
                ],
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    monkeypatch.setattr("ibkr_trading_bot.gui.tab_model_ranking.DEFAULT_MODEL_DIR", tmp_path)

    tab = ModelRankingTab()
    try:
        assert tab._has_active_filters() is False
        assert tab._active_shortlist_artifact() is None
        assert "pripraven posledni" in tab.lbl_shortlist.text()

        candidate_records, _empty_message, _up_to_date_message = tab._recompute_profit_opt_target()
        assert {record.model_path.name for record in candidate_records} == {"alpha.pkl", "beta.pkl"}
    finally:
        tab.close()


def test_shortlist_filter_limits_table_to_matching_candidates(monkeypatch, qapp):
    tmp_path = Path(".codex_test_tmp") / f"tab5_shortlist_filter_{uuid.uuid4().hex}"
    tmp_path.mkdir(parents=True, exist_ok=True)
    csv_path = tmp_path / "features.csv"
    csv_path.write_text("a,b\n1,2\n3,4\n", encoding="utf-8")

    _write_ranked_model(
        tmp_path,
        "alpha.pkl",
        csv_path=csv_path,
        n_total_bars=44268,
        estimator="lgb",
        horizon=12,
        tp_bps=40.0,
        sl_bps=30.0,
        optimized_profit=610.0,
    )
    _write_ranked_model(
        tmp_path,
        "beta.pkl",
        csv_path=csv_path,
        n_total_bars=44268,
        estimator="hgbt",
        horizon=16,
        tp_bps=40.0,
        sl_bps=30.0,
        optimized_profit=620.0,
    )
    _write_ranked_model(
        tmp_path,
        "gamma.pkl",
        csv_path=csv_path,
        n_total_bars=44268,
        estimator="rf",
        horizon=8,
        tp_bps=50.0,
        sl_bps=50.0,
        optimized_profit=630.0,
    )

    shortlist_dir = tmp_path / "auto_search"
    shortlist_dir.mkdir(parents=True, exist_ok=True)
    shortlist_path = shortlist_dir / "sample_shortlist.json"
    shortlist_path.write_text(
        json.dumps(
            {
                "version": 1,
                "mode": "refine",
                "created_at": "2026-05-27T12:00:00+00:00",
                "dataset_signature": {
                    "instrument": "GC",
                    "exchange": "COMEX",
                    "timeframe": "5m",
                    "n_total_bars": 44268,
                },
                "candidates": [
                    {"model": "lgb", "criterion": "balanced", "horizon": 12, "tp_bps": 40.0, "sl_bps": 30.0},
                    {"model": "hgbt", "criterion": "balanced", "horizon": 16, "tp_bps": 40.0, "sl_bps": 30.0},
                ],
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    monkeypatch.setattr("ibkr_trading_bot.gui.tab_model_ranking.DEFAULT_MODEL_DIR", tmp_path)

    tab = ModelRankingTab()
    try:
        assert tab.cmb_shortlist_filter.count() == 2

        tab.set_shortlist_filter(str(shortlist_path))
        qapp.processEvents()

        assert tab.tbl.rowCount() == 3
        assert {tab.tbl.item(row, 0).text() for row in range(tab.tbl.rowCount())} == {
            "alpha.pkl",
            "beta.pkl",
            "balanced.pkl",
        }
        assert tab._has_active_filters() is True
        assert "sample_shortlist.json" in tab.lbl_shortlist.text()
        assert "kandidati 2" in tab.lbl_shortlist.text()
    finally:
        tab.close()


def test_shortlist_filter_respects_candidate_selection_criterion(monkeypatch, qapp):
    tmp_path = Path(".codex_test_tmp") / f"tab5_shortlist_criterion_{uuid.uuid4().hex}"
    tmp_path.mkdir(parents=True, exist_ok=True)
    csv_path = tmp_path / "features.csv"
    csv_path.write_text("a,b\n1,2\n3,4\n", encoding="utf-8")

    _write_ranked_model(
        tmp_path,
        "balanced.pkl",
        csv_path=csv_path,
        n_total_bars=44268,
        estimator="lgb",
        horizon=12,
        tp_bps=40.0,
        sl_bps=30.0,
        candidate_selection_criterion="balanced",
        optimized_profit=610.0,
    )
    _write_ranked_model(
        tmp_path,
        "profit_first.pkl",
        csv_path=csv_path,
        n_total_bars=44268,
        estimator="lgb",
        horizon=12,
        tp_bps=40.0,
        sl_bps=30.0,
        candidate_selection_criterion="profit_first",
        optimized_profit=620.0,
    )

    shortlist_dir = tmp_path / "auto_search"
    shortlist_dir.mkdir(parents=True, exist_ok=True)
    shortlist_path = shortlist_dir / "criterion_shortlist.json"
    shortlist_path.write_text(
        json.dumps(
            {
                "version": 1,
                "mode": "refine",
                "created_at": "2026-05-27T12:00:00+00:00",
                "dataset_signature": {
                    "instrument": "GC",
                    "exchange": "COMEX",
                    "timeframe": "5m",
                    "n_total_bars": 44268,
                },
                "candidates": [
                    {
                        "model": "lgb",
                        "criterion": "profit_first",
                        "horizon": 12,
                        "tp_bps": 40.0,
                        "sl_bps": 30.0,
                    }
                ],
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    monkeypatch.setattr("ibkr_trading_bot.gui.tab_model_ranking.DEFAULT_MODEL_DIR", tmp_path)

    tab = ModelRankingTab()
    try:
        tab.set_shortlist_filter(str(shortlist_path))
        qapp.processEvents()

        assert tab.tbl.rowCount() == 1
        assert tab.tbl.item(0, 0).text() == "profit_first.pkl"
    finally:
        tab.close()


def test_check_filtered_requires_active_filter(monkeypatch, qapp):
    tmp_path = Path(".codex_test_tmp") / f"tab5_check_filtered_guard_{uuid.uuid4().hex}"
    tmp_path.mkdir(parents=True, exist_ok=True)
    csv_path = tmp_path / "features.csv"
    csv_path.write_text("a,b\n1,2\n3,4\n", encoding="utf-8")

    _write_ranked_model(
        tmp_path,
        "alpha.pkl",
        csv_path=csv_path,
        n_total_bars=44268,
        optimized_profit=610.0,
    )
    _write_ranked_model(
        tmp_path,
        "beta.pkl",
        csv_path=csv_path,
        n_total_bars=44268,
        optimized_profit=620.0,
    )

    monkeypatch.setattr("ibkr_trading_bot.gui.tab_model_ranking.DEFAULT_MODEL_DIR", tmp_path)

    tab = ModelRankingTab()
    try:
        assert tab._checked_records() == []

        tab._on_check_filtered_clicked()

        assert tab._checked_records() == []
        assert "nejprve zapni shortlist nebo jiny filtr" in tab.lbl_status.text()
    finally:
        tab.close()


def test_check_filtered_replaces_checked_scope_with_visible_shortlist_rows(monkeypatch, qapp):
    tmp_path = Path(".codex_test_tmp") / f"tab5_check_filtered_scope_{uuid.uuid4().hex}"
    tmp_path.mkdir(parents=True, exist_ok=True)
    csv_path = tmp_path / "features.csv"
    csv_path.write_text("a,b\n1,2\n3,4\n", encoding="utf-8")

    _write_ranked_model(
        tmp_path,
        "alpha.pkl",
        csv_path=csv_path,
        n_total_bars=44268,
        estimator="lgb",
        horizon=12,
        tp_bps=40.0,
        sl_bps=30.0,
        optimized_profit=610.0,
    )
    _write_ranked_model(
        tmp_path,
        "beta.pkl",
        csv_path=csv_path,
        n_total_bars=44268,
        estimator="hgbt",
        horizon=16,
        tp_bps=40.0,
        sl_bps=30.0,
        optimized_profit=620.0,
    )
    _write_ranked_model(
        tmp_path,
        "balanced.pkl",
        csv_path=csv_path,
        n_total_bars=44268,
        estimator="lgb",
        horizon=12,
        tp_bps=40.0,
        sl_bps=30.0,
        candidate_selection_criterion="balanced",
        optimized_profit=630.0,
    )
    _write_ranked_model(
        tmp_path,
        "outside.pkl",
        csv_path=csv_path,
        n_total_bars=44268,
        estimator="rf",
        horizon=8,
        tp_bps=50.0,
        sl_bps=50.0,
        optimized_profit=640.0,
    )

    shortlist_dir = tmp_path / "auto_search"
    shortlist_dir.mkdir(parents=True, exist_ok=True)
    shortlist_path = shortlist_dir / "sample_shortlist.json"
    shortlist_path.write_text(
        json.dumps(
            {
                "version": 1,
                "mode": "refine",
                "created_at": "2026-05-27T12:00:00+00:00",
                "dataset_signature": {
                    "instrument": "GC",
                    "exchange": "COMEX",
                    "timeframe": "5m",
                    "n_total_bars": 44268,
                },
                "candidates": [
                    {"model": "lgb", "criterion": "balanced", "horizon": 12, "tp_bps": 40.0, "sl_bps": 30.0},
                    {"model": "hgbt", "criterion": "balanced", "horizon": 16, "tp_bps": 40.0, "sl_bps": 30.0},
                ],
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    monkeypatch.setattr("ibkr_trading_bot.gui.tab_model_ranking.DEFAULT_MODEL_DIR", tmp_path)

    tab = ModelRankingTab()
    try:
        for row in range(tab.tbl.rowCount()):
            model_item = tab.tbl.item(row, 0)
            check_item = tab.tbl.item(row, tab_model_ranking_module.COL_CHECK)
            if model_item is None or check_item is None:
                continue
            if model_item.text() == "outside.pkl":
                check_item.setCheckState(Qt.Checked)
                break
        qapp.processEvents()

        assert {record.model_path.name for record in tab._checked_records()} == {"outside.pkl"}

        tab.set_shortlist_filter(str(shortlist_path))
        qapp.processEvents()

        tab._on_check_filtered_clicked()

        assert {record.model_path.name for record in tab._checked_records()} == {
            "alpha.pkl",
            "beta.pkl",
            "balanced.pkl",
        }
        assert "zatrzeno 3 filtrovanych modelu" in tab.lbl_status.text()

        checked_scope, scope_source = tab._comparison_target_records()
        assert scope_source == "checked"
        assert {record.model_path.name for record in checked_scope} == {
            "alpha.pkl",
            "beta.pkl",
            "balanced.pkl",
        }

        tab._on_clear_checked_clicked()

        assert tab._checked_records() == []
        assert "zatrzeni vymazano" in tab.lbl_status.text()
    finally:
        tab.close()


def test_comparison_snapshots_expose_ranking_context_and_metadata(monkeypatch, qapp):
    tmp_path = Path(".codex_test_tmp") / f"tab5_compare_snapshot_{uuid.uuid4().hex}"
    tmp_path.mkdir(parents=True, exist_ok=True)
    csv_path = tmp_path / "features.csv"
    csv_path.write_text("a,b\n1,2\n3,4\n", encoding="utf-8")

    _write_ranked_model(
        tmp_path,
        "alpha.pkl",
        csv_path=csv_path,
        n_total_bars=44268,
        training_mode="standard",
        user_note="kandidat",
        horizon=8,
        tp_bps=40.0,
        sl_bps=30.0,
        optimized_profit=610.0,
        trades_h=77.0,
    )
    _write_ranked_model(
        tmp_path,
        "beta.pkl",
        csv_path=csv_path,
        n_total_bars=44268,
        training_mode="refresh",
        horizon=16,
        tp_bps=35.0,
        sl_bps=25.0,
        optimized_profit=620.0,
        trained_features=["feat_a", "feat_b", "feat_c"],
        feature_stability={
            "feat_a": {"mean": 0.90, "std": 0.09, "folds_present": 5},
            "feat_b": {"mean": 0.60, "std": 0.12, "folds_present": 5},
            "feat_c": {"mean": 0.40, "std": 0.24, "folds_present": 5},
        },
        feature_stability_score={
            "feat_a": 0.80,
            "feat_b": 0.60,
            "feat_c": 0.40,
        },
        per_class_3={
            "-1": {"recall": 0.61},
            "1": {"recall": 0.09},
        },
    )

    monkeypatch.setattr("ibkr_trading_bot.gui.tab_model_ranking.DEFAULT_MODEL_DIR", tmp_path)

    tab = ModelRankingTab()
    try:
        context = {
            "data_path": str(csv_path),
            "fee_per_trade": 0.25,
            "entry_threshold": 0.55,
            "exit_threshold": 0.60,
        }

        snapshots = tab._comparison_snapshots(list(tab.records), context)
        snapshot_map = {snapshot.model_name: snapshot for snapshot in snapshots}

        alpha = snapshot_map["alpha.pkl"]
        assert alpha.values["mode"] == "Refine"
        assert alpha.values["status"] == "ok"
        assert alpha.values["freshness"] == "aktualni"
        assert alpha.values["optimized_profit"] == "610.00"
        assert alpha.values["trades"] == "77.00"
        assert alpha.values["horizon"] == "8"
        assert alpha.values["tp_bps"] == "40.00"
        assert alpha.values["sl_bps"] == "30.00"
        assert alpha.values["note"] == "kandidat"

        beta = snapshot_map["beta.pkl"]
        assert beta.values["mode"] == "Refresh"
        assert beta.values["optimized_profit"] == "620.00"
        assert beta.values["bias"] == "+0.520"
        assert beta.values["stability"] == "0.600"
        assert beta.values["horizon"] == "16"
        assert beta.values["tp_bps"] == "35.00"
        assert beta.values["sl_bps"] == "25.00"
    finally:
        tab.close()


def test_compare_export_writes_csv_and_json(monkeypatch, qapp):
    tmp_path = Path(".codex_test_tmp") / f"tab5_compare_export_{uuid.uuid4().hex}"
    tmp_path.mkdir(parents=True, exist_ok=True)
    csv_path = tmp_path / "features.csv"
    csv_path.write_text("a,b\n1,2\n3,4\n", encoding="utf-8")

    _write_ranked_model(
        tmp_path,
        "alpha.pkl",
        csv_path=csv_path,
        n_total_bars=44268,
        training_mode="standard",
        optimized_profit=610.0,
        trades_h=77.0,
    )
    _write_ranked_model(
        tmp_path,
        "beta.pkl",
        csv_path=csv_path,
        n_total_bars=44268,
        training_mode="refresh",
        optimized_profit=620.0,
        trades_h=88.0,
    )

    monkeypatch.setattr("ibkr_trading_bot.gui.tab_model_ranking.DEFAULT_MODEL_DIR", tmp_path)

    tab = ModelRankingTab()
    try:
        context = {
            "data_path": str(csv_path),
            "fee_per_trade": 0.25,
            "entry_threshold": 0.55,
            "exit_threshold": 0.60,
        }
        snapshots = tab._comparison_snapshots(list(tab.records), context)

        csv_out = tmp_path / "compare_export.csv"
        json_out = tmp_path / "compare_export.json"
        out_paths = [str(csv_out), str(json_out)]

        monkeypatch.setattr(
            tab_model_ranking_module.QFileDialog,
            "getSaveFileName",
            lambda *args, **kwargs: (out_paths.pop(0), ""),
        )

        tab._export_comparison_snapshots(snapshots, context, file_format="csv")
        tab._export_comparison_snapshots(snapshots, context, file_format="json")

        with csv_out.open("r", encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle))
        assert len(rows) == 2
        assert rows[0]["model_name"] == "beta.pkl"
        assert rows[0]["data_path"] == str(csv_path)
        assert rows[0]["scope_mode"] == "holdout"
        assert rows[0]["optimized_profit"] == "620.00"

        payload = json.loads(json_out.read_text(encoding="utf-8"))
        assert payload["comparison_context"]["data_path"] == str(csv_path)
        assert payload["comparison_context"]["scope_mode"] == "holdout"
        assert payload["models"][0]["model_name"] == "beta.pkl"
        assert payload["models"][0]["optimized_profit"] == "620.00"
        assert "compare export ulozen" in tab.lbl_status.text()
    finally:
        tab.close()


def test_ranking_stability_detail_dialog_opens_without_crash(monkeypatch, qapp):
    tmp_path = Path(".codex_test_tmp") / f"tab5_rank_stability_detail_{uuid.uuid4().hex}"
    tmp_path.mkdir(parents=True, exist_ok=True)
    csv_path = tmp_path / "features.csv"
    csv_path.write_text("a,b\n1,2\n3,4\n", encoding="utf-8")

    _write_ranked_model(
        tmp_path,
        "stable_detail.pkl",
        csv_path=csv_path,
        n_total_bars=44268,
        trained_features=["feat_a", "feat_b"],
        feature_stability={
            "feat_a": {"mean": 0.90, "std": 0.09, "folds_present": 5},
            "feat_b": {"mean": 0.55, "std": 0.22, "folds_present": 5},
            "feat_c": {"mean": 0.20, "std": 0.16, "folds_present": 5},
        },
        feature_stability_score={
            "feat_a": 0.90,
            "feat_b": 0.60,
            "feat_c": 0.20,
        },
        feature_stability_threshold=0.50,
        feature_stability_filter_applied=True,
        features_kept_by_stability=["feat_a", "feat_b"],
        features_removed_by_stability=["feat_c"],
    )

    monkeypatch.setattr("ibkr_trading_bot.gui.tab_model_ranking.DEFAULT_MODEL_DIR", tmp_path)

    tab = ModelRankingTab()
    try:
        detail = tab._build_feature_stability_detail(tab.records[0])
        assert detail.average_score == pytest.approx((0.90 + 0.60 + 0.20) / 3.0)
        assert detail.original_feature_count == 3
        assert detail.filtered_feature_count == 2
        assert [row.feature_name for row in detail.rows] == ["feat_a", "feat_b", "feat_c"]

        opened = {}

        def fake_exec(self):
            opened["title"] = self.windowTitle()
            return 0

        monkeypatch.setattr(tab_model_ranking_module.FeatureStabilityDetailDialog, "exec", fake_exec)

        tab._on_cell_clicked(0, COL_STABILITY)
        qapp.processEvents()

        assert opened["title"].endswith("stable_detail.pkl")
    finally:
        tab.close()


def test_ranking_batch_finished_reenables_recompute_button(qapp):
    tab = ModelRankingTab()
    try:
        tab._ranking_request_id = 3
        tab._ranking_worker = object()
        tab._batch_total = 2
        tab.btn_recompute_profit_opt.setEnabled(False)
        tab.btn_stop_recompute.setEnabled(True)

        tab._on_batch_result(
            3,
            {
                "updated": 2,
                "failures": 0,
                "requested": 2,
                "full_recompute": False,
            },
        )
        tab._on_batch_finished(3)

        assert tab._ranking_worker is None
        assert tab.btn_recompute_profit_opt.isEnabled() is True
        assert tab.btn_stop_recompute.isEnabled() is False
        assert "hotovo 2/2" in tab.lbl_batch_progress.text()
        assert "prepocet dokoncen" in tab.lbl_status.text()
    finally:
        tab.close()


def test_ranking_timer_runs_only_while_tab_is_active(monkeypatch, qapp, tmp_path):
    monkeypatch.setattr("ibkr_trading_bot.gui.tab_model_ranking.DEFAULT_MODEL_DIR", tmp_path)

    tab = ModelRankingTab()
    try:
        assert tab.timer.isActive() is False

        tab.on_tab_activated()

        assert tab.timer.isActive() is True

        tab.on_tab_deactivated()

        assert tab.timer.isActive() is False
    finally:
        tab.close()


def test_batch_progress_text_keeps_model_context_for_nested_status(qapp):
    tab = ModelRankingTab()
    try:
        tab._batch_total = 5
        tab._ranking_worker = object()

        tab._on_batch_progress_text("Ranking 2/5: demo.pkl")

        assert tab.lbl_status.text() == "Status: Ranking 2/5: demo.pkl"
        assert "hotovo 1/5" in tab.lbl_batch_progress.text()
        assert "demo.pkl" in tab.lbl_batch_progress.text()

        tab._on_batch_progress_text("Vyhodnoceni: pripravuji dataset...")

        assert "model 2/5 (demo.pkl)" in tab.lbl_status.text()
        assert "Vyhodnoceni: pripravuji dataset..." in tab.lbl_status.text()
        assert "hotovo 1/5" in tab.lbl_batch_progress.text()
    finally:
        tab.close()


def test_stop_recompute_requests_worker_stop_and_reports_partial_progress(qapp):
    class DummyWorker:
        def __init__(self):
            self.stop_calls = 0

        def stop(self):
            self.stop_calls += 1

    tab = ModelRankingTab()
    try:
        worker = DummyWorker()
        tab._ranking_request_id = 7
        tab._ranking_worker = worker
        tab._batch_total = 5
        tab._batch_completed = 2
        tab._batch_current_index = 3
        tab._batch_current_model = "demo.pkl"
        tab.btn_stop_recompute.setEnabled(True)

        tab._on_stop_recompute_clicked()

        assert worker.stop_calls == 1
        assert tab._batch_cancel_requested is True
        assert tab.btn_stop_recompute.isEnabled() is False
        assert "zastavuji prepocet" in tab.lbl_status.text()

        tab._on_batch_finished(7)

        assert tab._ranking_worker is None
        assert tab.btn_recompute_profit_opt.isEnabled() is True
        assert tab.btn_stop_recompute.isEnabled() is False
        assert "prepocet zastaven" in tab.lbl_status.text()
        assert "2/5" in tab.lbl_batch_progress.text()
    finally:
        tab.close()
