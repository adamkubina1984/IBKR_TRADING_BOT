import json

import pytest
from PySide6.QtCore import QItemSelectionModel
from PySide6.QtWidgets import QAbstractItemView, QMessageBox, QWidget

from ibkr_trading_bot.gui import tab_model_ranking as tab_model_ranking_module
from ibkr_trading_bot.gui.main_window import MainWindow

from ._gui_test_helpers import StubEvalRankingTab, StubEvalTab, StubLiveTab, StubTrainTab, ranking_payload, write_model


def test_model_ranking_single_click_does_not_apply_and_double_click_switches_and_evaluates(monkeypatch, qapp, tmp_path):
    model_path = write_model(
        tmp_path,
        "dummy_ternary.pkl",
        ranking={
            "status": "ok",
            "entry_threshold": 0.31,
            "exit_threshold": 0.44,
            "profit_h": 2.5,
        },
        profit_net=2.0,
    )

    monkeypatch.setattr(tab_model_ranking_module, "DEFAULT_MODEL_DIR", tmp_path)
    monkeypatch.setattr(MainWindow, "_create_data_tab", lambda self: QWidget())
    monkeypatch.setattr(MainWindow, "_create_train_tab", lambda self: QWidget())
    monkeypatch.setattr(MainWindow, "_create_eval_tab", lambda self: StubEvalTab())
    monkeypatch.setattr(MainWindow, "_create_live_tab", lambda self: StubLiveTab())

    window = MainWindow()
    try:
        window._ensure_tab_loaded(2)
        ranking = window.tab_model_ranking
        assert ranking is not None
        assert window.tab_eval is not None
        assert ranking.btn_delete.isHidden()
        assert ranking.btn_load.isHidden()

        window.tab_eval.model_path = None
        window.tab_eval.et_spin.setValue(0.61)
        window.tab_eval.ext_spin.setValue(0.72)
        if window.tab_live is not None:
            window.tab_live.le_model_path.setText("")

        ranking.tbl.setCurrentCell(0, 0)
        qapp.processEvents()

        assert window.tab_eval.model_path is None
        assert window.tab_live is None or window.tab_live.le_model_path.text() == ""
        assert window.tab_eval.eval_calls == 0

        ranking._on_cell_double_clicked(0, 0)
        qapp.processEvents()

        assert window.tabs.currentIndex() == 3
        assert window.tab_eval.model_path == str(model_path)
        assert window.tab_eval.et_spin.value() == pytest.approx(0.31)
        assert window.tab_eval.ext_spin.value() == pytest.approx(0.44)
        assert window.tab_eval.eval_calls == 1
        assert window.tab_live is not None
        assert window.tab_live.le_model_path.text() == str(model_path)
    finally:
        window.close()


def test_model_ranking_delete_shortcut_deletes_multiple_models(monkeypatch, qapp, tmp_path):
    write_model(tmp_path, "model_a.pkl", profit_net=1.0)
    write_model(tmp_path, "model_b.pkl", profit_net=2.0)

    monkeypatch.setattr(tab_model_ranking_module, "DEFAULT_MODEL_DIR", tmp_path)
    monkeypatch.setattr(MainWindow, "_create_data_tab", lambda self: QWidget())
    monkeypatch.setattr(MainWindow, "_create_train_tab", lambda self: QWidget())
    monkeypatch.setattr(MainWindow, "_create_eval_tab", lambda self: QWidget())
    monkeypatch.setattr(MainWindow, "_create_live_tab", lambda self: StubLiveTab())
    monkeypatch.setattr(QMessageBox, "question", lambda *args, **kwargs: QMessageBox.Yes)

    window = MainWindow()
    try:
        window._ensure_tab_loaded(2)
        ranking = window.tab_model_ranking
        assert ranking is not None
        assert ranking.tbl.selectionMode() == QAbstractItemView.ExtendedSelection

        ranking.tbl.setCurrentCell(0, 0)
        qapp.processEvents()

        selection_model = ranking.tbl.selectionModel()
        assert selection_model is not None
        selection_model.select(
            ranking.tbl.model().index(1, 0),
            QItemSelectionModel.Select | QItemSelectionModel.Rows,
        )
        qapp.processEvents()

        ranking._delete_shortcut.activated.emit()
        qapp.processEvents()

        assert sorted(path.name for path in tmp_path.glob("*.pkl")) == []
        assert sorted(path.name for path in tmp_path.glob("*_meta.json")) == []
        assert ranking.tbl.rowCount() == 0
        assert len(ranking.records) == 0
    finally:
        window.close()


def test_model_ranking_strict_top5_uses_preview_and_starts_batch(monkeypatch, qapp, tmp_path):
    csv_path = tmp_path / "GC_5m_features.csv"
    csv_path.write_text("timestamp,open,high,low,close,volume\n1,1,1,1,1,1\n", encoding="utf-8")

    for name, estimator, opt_profit, base_profit, sharpe, horizon, tp_bps, sl_bps in [
        ("candidate_a.pkl", "hgbt", 640.0, 140.0, 0.02, 8, 50.0, 50.0),
        ("candidate_b.pkl", "lgb", 590.0, -10.0, 0.01, 12, 40.0, 60.0),
        ("rescued_only.pkl", "xgb", 700.0, -20.0, 0.0, 8, 40.0, 40.0),
    ]:
        model_path = write_model(
            tmp_path,
            name,
            ranking=ranking_payload(csv_path, fee=0.25, profit_h=opt_profit, trades_h=120.0),
            profit_net=base_profit,
        )
        meta_path = model_path.with_name(model_path.stem + "_meta.json")
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        meta.update(
            {
                "instrument": "GC",
                "exchange": "COMEX",
                "timeframe": "5m",
                "n_total_bars": 44268,
                "estimator_name": estimator,
                "label_horizon_bars": horizon,
                "label_take_profit_bps": tp_bps,
                "label_stop_loss_bps": sl_bps,
            }
        )
        meta["metrics_holdout"]["sharpe"] = sharpe
        meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    monkeypatch.setattr(tab_model_ranking_module, "DEFAULT_MODEL_DIR", tmp_path)
    monkeypatch.setattr(MainWindow, "_create_data_tab", lambda self: QWidget())
    monkeypatch.setattr(MainWindow, "_create_train_tab", lambda self: StubTrainTab(str(csv_path), n_total_bars=44268))
    monkeypatch.setattr(MainWindow, "_create_eval_tab", lambda self: StubEvalRankingTab(str(csv_path)))
    monkeypatch.setattr(MainWindow, "_create_live_tab", lambda self: StubLiveTab())
    monkeypatch.setattr(QMessageBox, "question", lambda *args, **kwargs: QMessageBox.Yes)

    window = MainWindow()
    try:
        window._ensure_tab_loaded(2)
        ranking = window.tab_model_ranking
        assert ranking is not None

        captured = {}
        monkeypatch.setattr(
            ranking,
            "_start_strict_worker",
            lambda **kwargs: captured.update(kwargs),
        )

        ranking._on_strict_top5_clicked()
        qapp.processEvents()

        assert [candidate.record.model_path.name for candidate in captured["selected"]] == [
            "candidate_a.pkl",
            "candidate_b.pkl",
        ]
        assert captured["selected"][0].tier_label == "A"
        assert captured["selected"][1].tier_label == "B"
        assert captured["training_context"]["csv_path"].endswith("GC_5m_features.csv")
    finally:
        window.close()


def test_model_ranking_strict_top5_requires_training_csv(monkeypatch, qapp, tmp_path):
    monkeypatch.setattr(tab_model_ranking_module, "DEFAULT_MODEL_DIR", tmp_path)
    monkeypatch.setattr(MainWindow, "_create_data_tab", lambda self: QWidget())
    monkeypatch.setattr(MainWindow, "_create_train_tab", lambda self: StubTrainTab(None))
    monkeypatch.setattr(MainWindow, "_create_eval_tab", lambda self: StubEvalRankingTab(str(tmp_path / "missing.csv")))
    monkeypatch.setattr(MainWindow, "_create_live_tab", lambda self: StubLiveTab())

    warnings = []
    monkeypatch.setattr(QMessageBox, "warning", lambda *args, **kwargs: warnings.append(args[2]) or QMessageBox.Ok)

    window = MainWindow()
    try:
        window._ensure_tab_loaded(2)
        ranking = window.tab_model_ranking
        assert ranking is not None

        ranking._on_strict_top5_clicked()
        qapp.processEvents()

        assert warnings
        assert "Tab 2" in warnings[0]
    finally:
        window.close()