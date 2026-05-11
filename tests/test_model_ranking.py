import uuid
from pathlib import Path

import joblib
import numpy as np
import pytest
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QApplication

from ibkr_trading_bot.core.services import model_eval_service as model_eval_runtime
from ibkr_trading_bot.core.services.model_service import read_sidecar_model_meta, write_sidecar_model_meta
from ibkr_trading_bot.data.generate_synthetic import generate_synthetic_data
from ibkr_trading_bot.gui import tab_model_ranking as tab_model_ranking_module
from ibkr_trading_bot.gui.tab_model_ranking import (
    COL_BIAS,
    COL_CHECK,
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
    if per_class_3 is not None:
        meta["metrics_holdout"]["per_class_3"] = per_class_3

    if ranking_status == "ok":
        ranking = _ranking_payload(csv_path, fee=fee, profit_h=optimized_profit, trades_h=trades_h)
    else:
        ranking = model_eval_runtime.build_tab5_holdout_ranking_payload(
            data_path=csv_path,
            fee_per_trade=fee,
            entry_threshold=0.55,
            exit_threshold=0.60,
            metrics=None,
            status=ranking_status,
            error="test",
        )
    meta[model_eval_runtime.TAB5_HOLDOUT_RANKING_KEY] = ranking

    write_sidecar_model_meta(model_path, meta)
    return model_path


def test_tick_context_change_does_not_auto_recompute(monkeypatch, qapp):
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


def test_tick_auto_recompute_targets_only_new_models(monkeypatch, qapp):
    tmp_path = Path(".codex_test_tmp") / f"tab5_rank_tick_new_only_{uuid.uuid4().hex}"
    tmp_path.mkdir(parents=True, exist_ok=True)
    csv_path = tmp_path / "features.csv"
    csv_path.write_text("a,b\n1,2\n3,4\n", encoding="utf-8")

    _write_ranked_model(
        tmp_path,
        "old_pending.pkl",
        csv_path=csv_path,
        n_total_bars=44268,
        ranking_status="error",
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
        monkeypatch.setattr(tab, "_current_eval_context", lambda: context)

        captured = {}

        def _capture_incremental(*, candidate_records=None):
            captured["records"] = list(candidate_records or [])

        monkeypatch.setattr(tab, "_start_incremental_if_needed", _capture_incremental)

        _write_ranked_model(
            tmp_path,
            "new_pending.pkl",
            csv_path=csv_path,
            n_total_bars=44268,
            ranking_status="error",
        )

        tab._tick()

        assert [record.model_path.name for record in captured["records"]] == ["new_pending.pkl"]
        assert "records" in captured
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


