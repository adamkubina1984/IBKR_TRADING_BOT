import numpy as np
import pandas as pd
import pytest
from PySide6.QtWidgets import QWidget

from ibkr_trading_bot.gui import tab_model_ranking as tab_model_ranking_module
from ibkr_trading_bot.gui import tab_model_training as tab_model_training_module
from ibkr_trading_bot.gui.main_window import MainWindow
from ibkr_trading_bot.gui.tab_model_training import ModelTrainingTab

from ._gui_test_helpers import StubLiveTab


def test_main_window_tab_order_hides_legacy_manager(monkeypatch, qapp, tmp_path):
    monkeypatch.setattr(tab_model_ranking_module, "DEFAULT_MODEL_DIR", tmp_path)
    monkeypatch.setattr(MainWindow, "_create_data_tab", lambda self: QWidget())
    monkeypatch.setattr(MainWindow, "_create_train_tab", lambda self: QWidget())
    monkeypatch.setattr(MainWindow, "_create_live_tab", lambda self: StubLiveTab())

    window = MainWindow()
    try:
        labels = [window.tabs.tabText(i) for i in range(window.tabs.count())]
        assert labels == [
            "1) Stazeni dat z TreadingView",
            "2) Trenovani modelu",
            "3) Model Ranking",
            "4) Kontrola modelu",
            "5) Live trading bot",
        ]
        assert window.tabs.count() == 5
        assert all("Legacy" not in label for label in labels)
        assert window.tab_model_manager is None

        window._ensure_tab_loaded(2)
        window._ensure_tab_loaded(3)
        window._ensure_tab_loaded(4)

        assert window.tab_model_ranking is not None
        assert window.tab_eval is not None
        assert window.tab_live is not None
    finally:
        window.close()


def test_model_training_tab_hides_manual_training_controls(qapp):
    tab = ModelTrainingTab()
    try:
        assert tab.cmb_model.isHidden()
        assert tab.cmb_training_mode.isHidden()
        assert tab.cmb_candidate_criterion.isHidden()
        assert tab.btn_train.isHidden()
        assert tab.prog.isHidden()
        assert tab.tbl.isHidden()
        assert not tab.btn_auto_search.isHidden()
        assert not tab.cmb_search_backend.isHidden()
    finally:
        tab.close()


def test_model_training_tab_propagates_search_backend_profile(monkeypatch, qapp):
    captured: dict[str, object] = {}

    class _Signal:
        def connect(self, _cb):
            return None

    class StubTrainWorker:
        def __init__(self, **kwargs):
            captured.update(kwargs)
            self.progress = _Signal()
            self.phase = _Signal()
            self.model_ready = _Signal()
            self.error = _Signal()
            self.finished = _Signal()

        def start(self):
            return None

        def isRunning(self):
            return False

    monkeypatch.setattr(tab_model_training_module, "TrainWorker", StubTrainWorker)

    tab = ModelTrainingTab()
    try:
        n_rows = 96
        idx = np.arange(n_rows, dtype=float)
        target = (((idx.astype(int) // 3) % 2) == 1).astype(int)
        tab.csv_path = "data/processed/GC_5m_features.csv"
        tab.dataset = pd.DataFrame(
            {
                "timestamp": pd.date_range("2026-01-01", periods=n_rows, freq="5min", tz="UTC"),
                "f_signal": target + (0.1 * np.sin(idx / 3.0)),
                "f_aux": np.cos(idx / 5.0),
                "f_trend": idx / max(1.0, float(n_rows - 1)),
                "target": target,
            }
        )
        tab.cmb_search_backend.setCurrentText("optuna")
        tab.spn_optuna_trials.setValue(17)
        tab.spn_optuna_timeout.setValue(222)

        tab.run_training()

        profile = captured.get("training_profile")
        assert isinstance(profile, dict)
        assert profile["search_backend"] == "optuna"
        assert profile["optuna_trials"] == 17
        assert profile["optuna_timeout_seconds"] == 222
    finally:
        tab.close()


def test_model_training_tab_logs_holdout_chunk_summary(qapp):
    tab = ModelTrainingTab()
    try:
        tab._log_holdout_key_metrics(
            {
                "metrics": {
                    "profit_net": 12.5,
                    "num_trades": 10,
                    "num_trades_short": 4,
                    "num_trades_long": 6,
                    "pf": 1.1,
                    "max_dd": -5.0,
                    "per_class_3": {
                        "-1": {"recall": 0.20},
                        "1": {"recall": 0.30},
                    },
                },
                "quality_gate": {
                    "holdout_chunks": [
                        {
                            "chunk_index": 1,
                            "profit_net": -4.0,
                            "num_trades": 3,
                            "prediction_balance": {"n_short": 3, "n_long": 0},
                        },
                        {
                            "chunk_index": 2,
                            "profit_net": 1.5,
                            "num_trades": 2,
                            "prediction_balance": {"n_short": 1, "n_long": 1},
                        },
                        {
                            "chunk_index": 3,
                            "profit_net": 15.0,
                            "num_trades": 5,
                            "prediction_balance": {"n_short": 0, "n_long": 5},
                        },
                    ]
                },
            }
        )

        text = tab.log.toPlainText()
        assert "INFO Holdout chunks:" in text
        assert "c1 pnl=-4.00 trades=3 S/L=3/0" in text
        assert "c3 pnl=15.00 trades=5 S/L=0/5" in text
    finally:
        tab.close()


def test_model_training_tab_logs_reject_holdout_chunk_summary(tmp_path, qapp):
    tab = ModelTrainingTab()
    try:
        diag_path = tmp_path / "diag.json"
        diag_path.write_text(
            __import__("json").dumps(
                {
                    "metrics_holdout": {
                        "profit_net": -12.0,
                        "sharpe": -0.2,
                        "num_trades": 8,
                        "num_trades_short": 7,
                        "num_trades_long": 1,
                        "per_class_3": {
                            "-1": {"recall": 0.4},
                            "1": {"recall": 0.1},
                        },
                    },
                    "quality_gate": {
                        "reasons": ["long_prediction_share_too_low(0.0500<0.1000)"],
                        "holdout_chunks": [
                            {
                                "chunk_index": 1,
                                "profit_net": -8.0,
                                "num_trades": 4,
                                "prediction_balance": {"n_short": 4, "n_long": 0},
                            },
                            {
                                "chunk_index": 2,
                                "profit_net": -4.0,
                                "num_trades": 4,
                                "prediction_balance": {"n_short": 3, "n_long": 1},
                            },
                        ],
                    },
                    "mc_summary": {"sharpe": {"p50": -0.15}},
                    "threshold_tuning": {"selected_mode": "holdout_recent"},
                    "ternary_threshold_short": 0.55,
                    "ternary_threshold_long": 0.60,
                }
            ),
            encoding="utf-8",
        )

        tab._log_reject_summary_from_diag_meta(f"QUALITY_GATE_REJECT: failed | diag_meta={diag_path.as_posix()}")

        text = tab.log.toPlainText()
        assert "INFO Reject summary:" in text
        assert "INFO Reject holdout chunks:" in text
        assert "c1 pnl=-8.00 trades=4 S/L=4/0" in text
    finally:
        tab.close()