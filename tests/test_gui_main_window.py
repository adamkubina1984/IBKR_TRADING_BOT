import numpy as np
import pandas as pd
import pytest
from pathlib import Path
from PySide6.QtWidgets import QWidget

from ibkr_trading_bot.gui import tab_model_ranking as tab_model_ranking_module
from ibkr_trading_bot.gui import tab_model_training as tab_model_training_module
from ibkr_trading_bot.gui.main_window import MainWindow
from ibkr_trading_bot.gui.tab_model_training import ModelTrainingTab

from ._gui_test_helpers import StubLiveTab


class _HookedTab(QWidget):
    def __init__(self):
        super().__init__()
        self.events: list[str] = []

    def on_tab_activated(self):
        self.events.append("activated")

    def on_tab_deactivated(self):
        self.events.append("deactivated")


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


def test_main_window_notifies_loaded_tabs_about_visibility_changes(monkeypatch, qapp, tmp_path):
    monkeypatch.setattr(tab_model_ranking_module, "DEFAULT_MODEL_DIR", tmp_path)
    data_tab = _HookedTab()
    train_tab = _HookedTab()
    monkeypatch.setattr(MainWindow, "_create_data_tab", lambda self: data_tab)
    monkeypatch.setattr(MainWindow, "_create_train_tab", lambda self: train_tab)
    monkeypatch.setattr(MainWindow, "_create_model_ranking_tab", lambda self: QWidget())
    monkeypatch.setattr(MainWindow, "_create_eval_tab", lambda self: QWidget())
    monkeypatch.setattr(MainWindow, "_create_live_tab", lambda self: QWidget())

    window = MainWindow()
    try:
        assert data_tab.events == ["activated"]
        assert train_tab.events == []

        window.tabs.setCurrentIndex(1)
        qapp.processEvents()

        assert data_tab.events == ["activated", "deactivated"]
        assert train_tab.events == ["activated"]
    finally:
        window.close()


def test_model_training_tab_hides_manual_training_controls(qapp):
    tab = ModelTrainingTab()
    try:
        assert tab.cmb_model.isHidden()
        assert tab.cmb_training_mode.isHidden()
        assert [tab.cmb_training_mode.itemText(i) for i in range(tab.cmb_training_mode.count())] == ["quick", "standard"]
        assert tab.cmb_candidate_criterion.isHidden()
        assert tab.btn_train.isHidden()
        assert tab.prog.isHidden()
        assert tab.tbl.isHidden()
        assert not tab.btn_auto_search.isHidden()
        assert not tab.cmb_search_backend.isHidden()
    finally:
        tab.close()


def test_model_training_tab_restores_last_selected_csv_on_activation(monkeypatch, qapp, tmp_path):
    class _DummySettings:
        _store: dict[str, object] = {}

        def __init__(self, *args, **kwargs):
            pass

        def value(self, key, default=None):
            return self._store.get(str(key), default)

        def setValue(self, key, value):
            self._store[str(key)] = value

        def sync(self):
            return None

    csv_path = tmp_path / "features.csv"
    csv_path.write_text("timestamp,close\n2026-01-01T00:00:00Z,1\n", encoding="utf-8")

    dataset = pd.DataFrame(
        {
            "timestamp": pd.date_range("2026-01-01", periods=8, freq="5min", tz="UTC"),
            "feature_a": np.linspace(0.0, 1.0, 8),
            "target": [0, 1, 0, 1, 0, 1, 0, 1],
        }
    )

    _DummySettings._store = {}
    monkeypatch.setattr(tab_model_training_module, "QSettings", _DummySettings)
    monkeypatch.setattr(
        tab_model_training_module.DatasetService,
        "prepare_from_csv",
        lambda self, path, **kwargs: dataset.copy(),
    )
    monkeypatch.setattr(tab_model_training_module, "read_dataset_sidecar_meta", lambda path: {})
    monkeypatch.setattr(ModelTrainingTab, "_log_dataset_audit", lambda self, df: None)

    first = ModelTrainingTab()
    try:
        assert first._load_csv_path(str(csv_path), persist=True) is True
        assert first.csv_path == str(csv_path.resolve())
    finally:
        first.close()

    second = ModelTrainingTab()
    try:
        assert second.dataset is None
        assert second.lbl_csv.text().endswith("features.csv")

        second.on_tab_activated()

        assert second.csv_path == str(csv_path.resolve())
        assert second.dataset is not None
        assert second.btn_auto_search.isEnabled() is True
    finally:
        second.close()


def test_model_training_tab_prefers_more_advanced_legacy_fast_checkpoint_for_refine(monkeypatch, qapp, tmp_path):
    monkeypatch.setattr(tab_model_training_module, "_model_dir", lambda: tmp_path.as_posix())
    state_dir = tmp_path / "auto_search"
    state_dir.mkdir(parents=True, exist_ok=True)

    refine_state = state_dir / "tv_GC_COMEX_5m_sample_refine_state.json"
    refine_state.write_text(
        """
{
  "version": 2,
  "csv_path": "C:\\tmp\\tv_GC_COMEX_5m_sample.csv",
  "workflow_mode": "refine",
  "spec": {"version": 2, "workflow_mode": "refine"},
  "phase": "refine",
  "queue": [],
  "queue_idx": 0,
  "results": [],
  "stopped": true,
  "completed": false
}
        """.strip(),
        encoding="utf-8",
    )

    legacy_fast_state = state_dir / "tv_GC_COMEX_5m_sample_fast_state.json"
    legacy_fast_state.write_text(
        """
{
  "version": 1,
  "spec": {"search_profile": "fast"},
  "phase": "quick",
  "quick_queue": [],
  "quick_idx": 157,
  "results": [],
  "stopped": true,
  "completed": false
}
        """.strip(),
        encoding="utf-8",
    )

    tab = tab_model_training_module.ModelTrainingTab()
    try:
        tab.csv_path = str((tmp_path / "tv_GC_COMEX_5m_sample.csv").resolve())
        assert tab._auto_search_state_path("Refine") == legacy_fast_state
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
        meta_extra = captured.get("meta_extra")
        assert isinstance(profile, dict)
        assert isinstance(meta_extra, dict)
        assert profile["search_backend"] == "optuna"
        assert profile["optuna_trials"] == 17
        assert profile["optuna_timeout_seconds"] == 222
        assert profile["training_mode"] == "standard"
        assert profile["workflow_mode"] == "refine"
        assert profile["compatibility_mode"] == "standard"
        assert profile["runtime_training_mode"] == "standard"
        assert profile["training_mode_requested"] == "standard"
        assert meta_extra["workflow_mode"] == "refine"
        assert meta_extra["training_mode"] == "standard"
        assert meta_extra["training_mode_requested"] == "standard"
        assert meta_extra["training_mode_compatibility"] == "standard"
        assert meta_extra["training_mode_runtime"] == "standard"
    finally:
        tab.close()


def test_model_training_tab_passes_refresh_overrides_to_auto_worker(monkeypatch, qapp, tmp_path):
    captured: dict[str, object] = {}

    class _Signal:
        def connect(self, _cb):
            return None

    class StubAutoWorker:
        def __init__(self, **kwargs):
            captured.update(kwargs)
            self.message = _Signal()
            self.result = _Signal()
            self.finished_state = _Signal()
            self.error = _Signal()
            self.finished = _Signal()

        def start(self):
            return None

        def isRunning(self):
            return False

    monkeypatch.setattr(tab_model_training_module, "AutoSearchWorker", StubAutoWorker)
    monkeypatch.setattr(tab_model_training_module, "_model_dir", lambda: tmp_path.as_posix())

    source_artifact = tmp_path / "approved_refresh_source_shortlist.json"
    source_artifact.write_text(
        __import__("json").dumps(
            {
                "version": 1,
                "mode": "refine",
                "dataset_signature": {
                    "instrument": "GC",
                    "exchange": "COMEX",
                    "timeframe": "5m",
                    "n_total_bars": 64,
                },
                "candidates": [
                    {
                        "candidate_id": "lgb_h12_tp50_sl50_balanced",
                        "model": "lgb",
                        "criterion": "balanced",
                        "horizon": 12,
                        "tp_bps": 50.0,
                        "sl_bps": 50.0,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    target_csv = tmp_path / "tv_GC_COMEX_5m_refresh_target.csv"
    target_csv.write_text("timestamp,close\n2026-01-01T00:00:00Z,1\n", encoding="utf-8")

    tab = ModelTrainingTab()
    try:
        tab.csv_path = str((tmp_path / "tv_GC_COMEX_5m_source_dataset.csv").resolve())
        tab.dataset = pd.DataFrame(
            {
                "timestamp": pd.date_range("2026-01-01", periods=32, freq="5min", tz="UTC"),
                "feature_a": np.linspace(0.0, 1.0, 32),
                "target": [0, 1] * 16,
            }
        )
        tab.cmb_auto_search_profile.setCurrentText("Refresh")
        tab._set_refresh_source_artifact_path(source_artifact.as_posix(), persist=False)
        tab._set_refresh_target_csv_path(target_csv.as_posix(), persist=False)

        tab.run_auto_search()

        assert captured["search_profile"] == "refresh"
        assert captured["source_artifact_path"] == source_artifact.resolve().as_posix()
        assert captured["refresh_csv_path"] == str(target_csv.resolve())
        assert str(captured["state_path"]).endswith(
            "approved_refresh_source_shortlist__to__tv_GC_COMEX_5m_refresh_target_refresh_state.json"
        )
    finally:
        tab.close()


def test_model_training_tab_passes_refine_source_artifact_to_auto_worker(monkeypatch, qapp, tmp_path):
    captured: dict[str, object] = {}

    class _Signal:
        def connect(self, _cb):
            return None

    class StubAutoWorker:
        def __init__(self, **kwargs):
            captured.update(kwargs)
            self.message = _Signal()
            self.result = _Signal()
            self.finished_state = _Signal()
            self.error = _Signal()
            self.finished = _Signal()

        def start(self):
            return None

        def isRunning(self):
            return False

    monkeypatch.setattr(tab_model_training_module, "AutoSearchWorker", StubAutoWorker)
    monkeypatch.setattr(tab_model_training_module, "_model_dir", lambda: tmp_path.as_posix())

    source_artifact = tmp_path / "approved_refine_source_region_summary.json"
    source_artifact.write_text(
        __import__("json").dumps(
            {
                "version": 1,
                "mode": "explore",
                "source_csv_path": str((tmp_path / "tv_GC_COMEX_5m_source_dataset.csv").resolve()),
                "dataset_signature": {
                    "instrument": "GC",
                    "exchange": "COMEX",
                    "timeframe": "5m",
                    "n_total_bars": 32,
                },
                "approved_regions": [
                    {
                        "region_id": "lgb_h12_tp50_sl50",
                        "models": ["lgb"],
                        "horizon_values": [12],
                        "tp_bps_min": 50.0,
                        "tp_bps_max": 55.0,
                        "sl_bps_min": 50.0,
                        "sl_bps_max": 55.0,
                        "criteria": ["balanced"],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    tab = ModelTrainingTab()
    try:
        tab.csv_path = str((tmp_path / "tv_GC_COMEX_5m_source_dataset.csv").resolve())
        tab.dataset = pd.DataFrame(
            {
                "timestamp": pd.date_range("2026-01-01", periods=32, freq="5min", tz="UTC"),
                "feature_a": np.linspace(0.0, 1.0, 32),
                "target": [0, 1] * 16,
            }
        )
        tab.cmb_auto_search_profile.setCurrentText("Refine")
        tab._set_refine_source_artifact_path(source_artifact.as_posix(), persist=False)

        tab.run_auto_search()

        assert captured["search_profile"] == "refine"
        assert captured["source_artifact_path"] == source_artifact.resolve().as_posix()
        assert str(captured["state_path"]).endswith("approved_refine_source_region_summary_refine_state.json")
    finally:
        tab.close()


def test_model_training_tab_blocks_incompatible_refine_source_artifact(monkeypatch, qapp, tmp_path):
    captured: dict[str, object] = {}

    class _Signal:
        def connect(self, _cb):
            return None

    class StubAutoWorker:
        def __init__(self, **kwargs):
            captured.update(kwargs)
            self.message = _Signal()
            self.result = _Signal()
            self.finished_state = _Signal()
            self.error = _Signal()
            self.finished = _Signal()

        def start(self):
            return None

        def isRunning(self):
            return False

    monkeypatch.setattr(tab_model_training_module, "AutoSearchWorker", StubAutoWorker)
    monkeypatch.setattr(tab_model_training_module, "_model_dir", lambda: tmp_path.as_posix())

    source_artifact = tmp_path / "incompatible_refine_region_summary.json"
    source_artifact.write_text(
        __import__("json").dumps(
            {
                "version": 1,
                "mode": "explore",
                "source_csv_path": str((tmp_path / "tv_GC_COMEX_5m_other_dataset.csv").resolve()),
                "dataset_signature": {
                    "instrument": "GC",
                    "exchange": "COMEX",
                    "timeframe": "5m",
                    "n_total_bars": 128,
                },
                "approved_regions": [
                    {
                        "region_id": "lgb_h12_tp50_sl50",
                        "models": ["lgb"],
                        "horizon_values": [12],
                        "tp_bps_min": 50.0,
                        "tp_bps_max": 55.0,
                        "sl_bps_min": 50.0,
                        "sl_bps_max": 55.0,
                        "criteria": ["balanced"],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    tab = ModelTrainingTab()
    try:
        tab.csv_path = str((tmp_path / "tv_GC_COMEX_5m_source_dataset.csv").resolve())
        tab.dataset = pd.DataFrame(
            {
                "timestamp": pd.date_range("2026-01-01", periods=32, freq="5min", tz="UTC"),
                "feature_a": np.linspace(0.0, 1.0, 32),
                "target": [0, 1] * 16,
            }
        )
        tab.cmb_auto_search_profile.setCurrentText("Refine")
        tab._set_refine_source_artifact_path(source_artifact.as_posix(), persist=False)

        tab.run_auto_search()

        assert captured == {}
        assert "ERROR Workflow: Refine source artifact" in tab.log.toPlainText()
    finally:
        tab.close()


def test_model_training_tab_blocks_incompatible_refresh_target_dataset(monkeypatch, qapp, tmp_path):
    captured: dict[str, object] = {}

    class _Signal:
        def connect(self, _cb):
            return None

    class StubAutoWorker:
        def __init__(self, **kwargs):
            captured.update(kwargs)
            self.message = _Signal()
            self.result = _Signal()
            self.finished_state = _Signal()
            self.error = _Signal()
            self.finished = _Signal()

        def start(self):
            return None

        def isRunning(self):
            return False

    monkeypatch.setattr(tab_model_training_module, "AutoSearchWorker", StubAutoWorker)
    monkeypatch.setattr(tab_model_training_module, "_model_dir", lambda: tmp_path.as_posix())

    source_artifact = tmp_path / "approved_refresh_source_shortlist.json"
    source_artifact.write_text(
        __import__("json").dumps(
            {
                "version": 1,
                "mode": "refine",
                "dataset_signature": {
                    "instrument": "GC",
                    "exchange": "COMEX",
                    "timeframe": "5m",
                    "n_total_bars": 64,
                },
                "candidates": [
                    {
                        "candidate_id": "lgb_h12_tp50_sl50_balanced",
                        "model": "lgb",
                        "criterion": "balanced",
                        "horizon": 12,
                        "tp_bps": 50.0,
                        "sl_bps": 50.0,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    target_csv = tmp_path / "tv_GC_COMEX_15m_refresh_target.csv"
    target_csv.write_text("timestamp,close\n2026-01-01T00:00:00Z,1\n", encoding="utf-8")

    tab = ModelTrainingTab()
    try:
        tab.csv_path = str((tmp_path / "tv_GC_COMEX_5m_source_dataset.csv").resolve())
        tab.dataset = pd.DataFrame(
            {
                "timestamp": pd.date_range("2026-01-01", periods=32, freq="5min", tz="UTC"),
                "feature_a": np.linspace(0.0, 1.0, 32),
                "target": [0, 1] * 16,
            }
        )
        tab.cmb_auto_search_profile.setCurrentText("Refresh")
        tab._set_refresh_source_artifact_path(source_artifact.as_posix(), persist=False)
        tab._set_refresh_target_csv_path(target_csv.as_posix(), persist=False)

        tab.run_auto_search()

        assert captured == {}
        assert "ERROR Workflow: Refresh source artifact neodpovida target datasetu" in tab.log.toPlainText()
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