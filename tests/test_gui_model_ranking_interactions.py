import json

import pytest
from PySide6.QtCore import QItemSelectionModel, Qt
from PySide6.QtTest import QTest
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


def test_model_ranking_delete_uses_visible_filtered_records(monkeypatch, qapp, tmp_path):
    visible_model = write_model(
        tmp_path,
        "visible.pkl",
        ranking={"status": "error"},
        profit_net=1.0,
    )
    hidden_model = write_model(
        tmp_path,
        "hidden.pkl",
        ranking={"status": "error"},
        profit_net=2.0,
    )

    for model_path, rec_short, rec_long in [
        (visible_model, 0.31, 0.29),
        (hidden_model, 0.61, 0.09),
    ]:
        meta_path = model_path.with_name(model_path.stem + "_meta.json")
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        meta.setdefault("metrics_holdout", {})["per_class_3"] = {
            "-1": {"recall": rec_short},
            "1": {"recall": rec_long},
        }
        meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    monkeypatch.setattr(tab_model_ranking_module, "DEFAULT_MODEL_DIR", tmp_path)
    monkeypatch.setattr(MainWindow, "_create_data_tab", lambda self: QWidget())
    monkeypatch.setattr(MainWindow, "_create_train_tab", lambda self: QWidget())
    monkeypatch.setattr(MainWindow, "_create_eval_tab", lambda self: StubEvalTab())
    monkeypatch.setattr(MainWindow, "_create_live_tab", lambda self: StubLiveTab())
    monkeypatch.setattr(QMessageBox, "question", lambda *args, **kwargs: QMessageBox.Yes)

    window = MainWindow()
    try:
        window._ensure_tab_loaded(2)
        ranking = window.tab_model_ranking
        assert ranking is not None

        ranking.set_bias_filter("tight")
        qapp.processEvents()
        assert ranking.tbl.rowCount() == 1
        assert ranking.tbl.item(0, 0).text() == "visible.pkl"

        ranking.tbl.setCurrentCell(0, 0)
        qapp.processEvents()
        ranking._on_delete_selected()
        qapp.processEvents()

        assert not visible_model.exists()
        assert not visible_model.with_name("visible_meta.json").exists()
        assert hidden_model.exists()
        assert hidden_model.with_name("hidden_meta.json").exists()
        assert ranking.tbl.rowCount() == 0
    finally:
        window.close()


def test_model_ranking_delete_key_uses_sorted_visible_row(monkeypatch, qapp, tmp_path):
    alpha_model = write_model(tmp_path, "alpha.pkl", profit_net=1.0)
    omega_model = write_model(tmp_path, "omega.pkl", profit_net=9.0)

    monkeypatch.setattr(tab_model_ranking_module, "DEFAULT_MODEL_DIR", tmp_path)
    monkeypatch.setattr(MainWindow, "_create_data_tab", lambda self: QWidget())
    monkeypatch.setattr(MainWindow, "_create_train_tab", lambda self: QWidget())
    monkeypatch.setattr(MainWindow, "_create_eval_tab", lambda self: StubEvalTab())
    monkeypatch.setattr(MainWindow, "_create_live_tab", lambda self: StubLiveTab())
    monkeypatch.setattr(QMessageBox, "question", lambda *args, **kwargs: QMessageBox.Yes)

    window = MainWindow()
    try:
        window._ensure_tab_loaded(2)
        ranking = window.tab_model_ranking
        assert ranking is not None

        ranking.tbl.sortItems(0, Qt.DescendingOrder)
        qapp.processEvents()

        assert ranking.tbl.item(0, 0).text() == "omega.pkl"
        ranking.tbl.setFocus()
        ranking.tbl.setCurrentCell(0, 0)
        qapp.processEvents()

        QTest.keyClick(ranking.tbl, Qt.Key_Delete)
        qapp.processEvents()

        assert not omega_model.exists()
        assert not omega_model.with_name("omega_meta.json").exists()
        assert alpha_model.exists()
        assert alpha_model.with_name("alpha_meta.json").exists()
    finally:
        window.close()


