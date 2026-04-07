import uuid
from pathlib import Path

import joblib
import numpy as np
import pytest
from PySide6.QtWidgets import QApplication

from ibkr_trading_bot.core.services import model_eval_service as model_eval_runtime
from ibkr_trading_bot.core.services.model_training_service import dataset_snapshot_signature_from_csv
from ibkr_trading_bot.core.services.model_service import read_sidecar_model_meta, write_sidecar_model_meta
from ibkr_trading_bot.data.generate_synthetic import generate_synthetic_data
from ibkr_trading_bot.gui import tab_model_ranking as tab_model_ranking_module
from ibkr_trading_bot.gui.tab_model_ranking import (
    COL_MODE,
    COL_NOTE,
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


def _ranking_payload(csv_path: Path, *, fee: float, profit_h: float, trades_h: float) -> dict[str, object]:
    return model_eval_runtime.build_tab5_holdout_ranking_payload(
        data_path=csv_path,
        fee_per_trade=fee,
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
    user_note: str | None = None,
    trained_features: list[str] | None = None,
    feature_stability: dict[str, object] | None = None,
    feature_stability_score: dict[str, float] | None = None,
    feature_stability_threshold: float | None = None,
    feature_stability_filter_applied: bool = False,
    features_kept_by_stability: list[str] | None = None,
    features_removed_by_stability: list[str] | None = None,
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
    if user_note:
        meta["model_ranking_note"] = str(user_note)
    if feature_stability is not None:
        meta["feature_stability"] = dict(feature_stability)
    if feature_stability_score is not None:
        meta["feature_stability_score"] = dict(feature_stability_score)
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

    meta = {"created_at": "2026-03-16T10:00:00"}
    meta[model_eval_runtime.TAB5_HOLDOUT_RANKING_KEY] = model_eval_runtime.build_tab5_holdout_ranking_payload(
        data_path=csv_path,
        fee_per_trade=0.25,
        entry_threshold=0.34,
        exit_threshold=0.41,
        metrics={"profit_net": 12.5, "max_dd": -5.0, "trades": 9},
        status="ok",
    )
    meta_path = write_sidecar_model_meta(model_path, meta)

    reloaded = read_sidecar_model_meta(model_path)
    ranking = reloaded[model_eval_runtime.TAB5_HOLDOUT_RANKING_KEY]

    assert ranking["status"] == "ok"
    assert ranking["entry_threshold"] == 0.34
    assert ranking["exit_threshold"] == 0.41
    assert ranking["profit_h"] == 12.5
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


def test_ranking_task_writes_tab5_holdout_ranking_fields():
    tmp_path = Path(".codex_test_tmp") / f"tab5_rank_task_{uuid.uuid4().hex}"
    tmp_path.mkdir(parents=True, exist_ok=True)

    raw = generate_synthetic_data(n_samples=260, noise_level=0.05)
    features_path = tmp_path / "synthetic_ohlcv.csv"
    raw.to_csv(features_path, index=False)

    model_path = tmp_path / "dummy_ternary.pkl"
    joblib.dump({"model": DummyRankingPredictor()}, model_path)
    write_sidecar_model_meta(
        model_path,
        {
            "created_at": "2026-03-16T10:00:00",
            "classes": [0, 1, 2],
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
    assert ranking["scope"] == "holdout"
    assert ranking["csv_path"].endswith("synthetic_ohlcv.csv")
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

        tab._on_recompute_all_clicked()

        assert [record.model_path.name for record in captured["records"]] == ["pending.pkl"]
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
        assert "klikni 'Prepocitat Profit(H)'" in tab.lbl_status.text()
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
        assert tab.tbl.item(0, COL_MODE).text() == "standard"
        assert tab.tbl.item(0, COL_NOTE).text() == "puvodni poznamka"

        note_item = tab.tbl.item(0, COL_NOTE)
        assert note_item is not None
        note_item.setText("  favorit do strictu  ")
        qapp.processEvents()

        reloaded = read_sidecar_model_meta(model_path)
        assert reloaded["model_ranking_note"] == "favorit do strictu"
        assert tab.tbl.item(0, COL_NOTE).text() == "favorit do strictu"
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


def test_strict_shortlist_filters_and_dedupes_candidates():
    tmp_path = Path(".codex_test_tmp") / f"tab5_rank_strict_shortlist_{uuid.uuid4().hex}"
    tmp_path.mkdir(parents=True, exist_ok=True)
    csv_path = tmp_path / "GC_5m_features.csv"
    csv_path.write_text("timestamp,open,high,low,close,volume\n1,1,1,1,1,1\n", encoding="utf-8")
    snapshot_signature = dataset_snapshot_signature_from_csv(csv_path, 44268)
    assert snapshot_signature is not None

    _write_ranked_model(
        tmp_path,
        "good_a.pkl",
        csv_path=csv_path,
        n_total_bars=44268,
        estimator="hgbt",
        horizon=8,
        tp_bps=50.0,
        sl_bps=50.0,
        base_profit=140.0,
        sharpe=0.02,
        optimized_profit=640.0,
    )
    _write_ranked_model(
        tmp_path,
        "good_b.pkl",
        csv_path=csv_path,
        n_total_bars=44268,
        estimator="lgb",
        horizon=12,
        tp_bps=50.0,
        sl_bps=60.0,
        base_profit=-15.0,
        sharpe=0.01,
        optimized_profit=590.0,
    )
    _write_ranked_model(
        tmp_path,
        "duplicate_lower.pkl",
        csv_path=csv_path,
        n_total_bars=44268,
        estimator="hgbt",
        horizon=8,
        tp_bps=50.0,
        sl_bps=50.0,
        base_profit=100.0,
        sharpe=0.015,
        optimized_profit=610.0,
    )
    _write_ranked_model(
        tmp_path,
        "rescued_only.pkl",
        csv_path=csv_path,
        n_total_bars=44268,
        estimator="xgb",
        horizon=8,
        tp_bps=40.0,
        sl_bps=40.0,
        base_profit=-20.0,
        sharpe=0.0,
        optimized_profit=700.0,
    )
    _write_ranked_model(
        tmp_path,
        "other_snapshot.pkl",
        csv_path=csv_path,
        n_total_bars=45840,
        estimator="rf",
        horizon=8,
        tp_bps=50.0,
        sl_bps=50.0,
        base_profit=90.0,
        sharpe=0.02,
        optimized_profit=520.0,
    )
    _write_ranked_model(
        tmp_path,
        "stale_fee.pkl",
        csv_path=csv_path,
        n_total_bars=44268,
        estimator="et",
        horizon=8,
        tp_bps=50.0,
        sl_bps=50.0,
        base_profit=95.0,
        sharpe=0.02,
        optimized_profit=510.0,
        fee=0.99,
    )
    _write_ranked_model(
        tmp_path,
        "status_error.pkl",
        csv_path=csv_path,
        n_total_bars=44268,
        ranking_status="error",
    )

    records = discover_ranking_models(tmp_path)
    shortlist = ModelRankingTab._build_strict_shortlist(
        records=records,
        context={"data_path": str(csv_path), "fee_per_trade": 0.25},
        current_snapshot_signature=snapshot_signature,
        limit=5,
    )

    selected_names = [candidate.record.model_path.name for candidate in shortlist["selected"]]
    rejected_reasons = {rejection.record.model_path.name: rejection.reason for rejection in shortlist["rejected"]}

    assert selected_names == ["good_a.pkl", "good_b.pkl"]
    assert shortlist["selected"][0].tier_label == "A"
    assert shortlist["selected"][1].tier_label == "B"
    assert "duplicitni konfigurace" in rejected_reasons["duplicate_lower.pkl"]
    assert rejected_reasons["rescued_only.pkl"] == "Profit(H) < 0 a Sharpe(H) <= 0"
    assert rejected_reasons["other_snapshot.pkl"] == "model nepatri do aktualniho dataset snapshotu"
    assert rejected_reasons["stale_fee.pkl"] == "ranking je stale pro aktualni CSV/fee"
    assert rejected_reasons["status_error.pkl"] == "ranking nema status ok"


def test_strict_shortlist_limits_to_top5_after_tiering():
    tmp_path = Path(".codex_test_tmp") / f"tab5_rank_strict_top5_{uuid.uuid4().hex}"
    tmp_path.mkdir(parents=True, exist_ok=True)
    csv_path = tmp_path / "GC_5m_features.csv"
    csv_path.write_text("timestamp,open,high,low,close,volume\n1,1,1,1,1,1\n", encoding="utf-8")
    snapshot_signature = dataset_snapshot_signature_from_csv(csv_path, 44268)
    assert snapshot_signature is not None

    for idx, profit in enumerate([650.0, 620.0, 610.0, 600.0, 590.0, 580.0], start=1):
        _write_ranked_model(
            tmp_path,
            f"candidate_{idx}.pkl",
            csv_path=csv_path,
            n_total_bars=44268,
            estimator=f"model{idx}",
            horizon=8 + idx,
            tp_bps=40.0 + idx,
            sl_bps=50.0 + idx,
            base_profit=100.0 - idx,
            sharpe=0.02,
            optimized_profit=profit,
        )

    records = discover_ranking_models(tmp_path)
    shortlist = ModelRankingTab._build_strict_shortlist(
        records=records,
        context={"data_path": str(csv_path), "fee_per_trade": 0.25},
        current_snapshot_signature=snapshot_signature,
        limit=5,
    )

    selected_names = [candidate.record.model_path.name for candidate in shortlist["selected"]]
    rejected_reasons = {rejection.record.model_path.name: rejection.reason for rejection in shortlist["rejected"]}

    assert len(selected_names) == 5
    assert selected_names == [f"candidate_{idx}.pkl" for idx in range(1, 6)]
    assert rejected_reasons["candidate_6.pkl"] == "mimo Top 5 po serazeni"


def test_strict_shortlist_excludes_already_strict_models():
    tmp_path = Path(".codex_test_tmp") / f"tab5_rank_strict_mode_filter_{uuid.uuid4().hex}"
    tmp_path.mkdir(parents=True, exist_ok=True)
    csv_path = tmp_path / "GC_5m_features.csv"
    csv_path.write_text("timestamp,open,high,low,close,volume\n1,1,1,1,1,1\n", encoding="utf-8")
    snapshot_signature = dataset_snapshot_signature_from_csv(csv_path, 44268)
    assert snapshot_signature is not None

    _write_ranked_model(
        tmp_path,
        "already_strict.pkl",
        csv_path=csv_path,
        n_total_bars=44268,
        optimized_profit=650.0,
        base_profit=150.0,
        sharpe=0.03,
        training_mode="strict",
    )
    _write_ranked_model(
        tmp_path,
        "fresh_standard.pkl",
        csv_path=csv_path,
        n_total_bars=44268,
        optimized_profit=620.0,
        base_profit=140.0,
        sharpe=0.02,
        training_mode="standard",
    )

    records = discover_ranking_models(tmp_path)
    shortlist = ModelRankingTab._build_strict_shortlist(
        records=records,
        context={"data_path": str(csv_path), "fee_per_trade": 0.25},
        current_snapshot_signature=snapshot_signature,
        limit=5,
    )

    selected_names = [candidate.record.model_path.name for candidate in shortlist["selected"]]
    rejected_reasons = {rejection.record.model_path.name: rejection.reason for rejection in shortlist["rejected"]}

    assert selected_names == ["fresh_standard.pkl"]
    assert rejected_reasons["already_strict.pkl"] == "model uz je strict"


def test_strict_batch_writes_provenance_to_new_models(monkeypatch):
    tmp_path = Path(".codex_test_tmp") / f"tab5_rank_strict_batch_{uuid.uuid4().hex}"
    tmp_path.mkdir(parents=True, exist_ok=True)
    csv_path = tmp_path / "GC_5m_features.csv"
    csv_path.write_text("timestamp,open,high,low,close,volume\n1,1,1,1,1,1\n", encoding="utf-8")
    counter = {"idx": 0}

    def fake_run_training_job(**kwargs):
        counter["idx"] += 1
        model_path = tmp_path / f"strict_result_{counter['idx']}.pkl"
        model_path.write_bytes(b"model")
        meta = {
            "created_at": "2026-03-16T11:00:00",
            "estimator_name": kwargs["estimator_name"],
            "label_horizon_bars": kwargs["horizon"],
            "label_take_profit_bps": kwargs["tp_bps"],
            "label_stop_loss_bps": kwargs["sl_bps"],
        }
        write_sidecar_model_meta(model_path, meta)
        return {
            "status": "ok",
            "model_path": str(model_path),
            "meta_path": str(model_path.with_name(model_path.stem + "_meta.json")),
            "profit_net": 123.0,
            "sharpe": 0.02,
            "pf": 1.1,
            "trades": 42,
        }

    monkeypatch.setattr("ibkr_trading_bot.gui.tab_model_ranking.run_training_job", fake_run_training_job)

    result = ModelRankingTab._task_run_strict_batch(
        jobs=[
            {
                "source_model_path": str(tmp_path / "source_a.pkl"),
                "source_rank_position": 1,
                "estimator_name": "hgbt",
                "criterion": "balanced",
                "horizon": 8,
                "tp_bps": 50.0,
                "sl_bps": 50.0,
                "strict_source_metrics": {"profit_h_opt": 640.0, "profit_h": 140.0, "sharpe_h": 0.02, "trades_h": 120.0},
            },
            {
                "source_model_path": str(tmp_path / "source_b.pkl"),
                "source_rank_position": 2,
                "estimator_name": "lgb",
                "criterion": "profit_first",
                "horizon": 12,
                "tp_bps": 40.0,
                "sl_bps": 60.0,
                "strict_source_metrics": {"profit_h_opt": 590.0, "profit_h": 90.0, "sharpe_h": 0.015, "trades_h": 110.0},
            },
        ],
        training_csv_path=str(csv_path),
        holdout_pct=0.10,
        holdout_min_bars=1000,
        holdout_max_bars=6000,
        candidate_top_n=5,
        candidate_fresh_ratio=0.30,
        batch_id="strict_test_batch",
    )

    assert result["created"] == 2
    assert result["rejected"] == 0
    assert result["failures"] == 0

    meta_a = read_sidecar_model_meta(tmp_path / "strict_result_1.pkl")
    meta_b = read_sidecar_model_meta(tmp_path / "strict_result_2.pkl")

    assert meta_a["training_mode"] == "strict"
    assert meta_a["strict_source_model_path"].endswith("source_a.pkl")
    assert meta_a["strict_source_rank_position"] == 1
    assert meta_a["strict_batch_id"] == "strict_test_batch"
    assert meta_a["strict_trigger"] == "ranking_top5"
    assert meta_a["strict_source_metrics"]["profit_h_opt"] == 640.0

    assert meta_b["training_mode"] == "strict"
    assert meta_b["strict_source_model_path"].endswith("source_b.pkl")
    assert meta_b["strict_source_rank_position"] == 2
