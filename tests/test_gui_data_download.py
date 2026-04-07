import json

import numpy as np
import pandas as pd
import pytest
from PySide6.QtWidgets import QFileDialog, QMessageBox

from ibkr_trading_bot.gui import tab_data_download as tab_data_download_module
from ibkr_trading_bot.gui.tab_data_download import DataDownloadTab

from ._gui_test_helpers import StubCanvas


def test_data_download_tab_defaults_ibkr_mode_to_fut(monkeypatch, qapp):
    monkeypatch.setattr(tab_data_download_module, "FigureCanvas", StubCanvas)
    tab = DataDownloadTab()
    try:
        assert tab.cmb_ibkr_mode.currentText() == "FUT"
    finally:
        tab.close()


def test_data_download_roll_chain_update_rebuilds_canonical_dataset(monkeypatch, qapp, tmp_path):
    monkeypatch.setattr(tab_data_download_module, "FigureCanvas", StubCanvas)
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(tab_data_download_module, "RAW_DIR", str(raw_dir))

    tab = DataDownloadTab()
    try:
        processed_dir = tmp_path / "processed"
        processed_dir.mkdir(parents=True, exist_ok=True)
        csv_path = processed_dir / "GC_5m_rollchain_2bars_20250101_20250101_20260324_010101.csv"
        pd.DataFrame(
            {
                "date": pd.date_range("2025-01-01 00:00:00", periods=2, freq="5min"),
                "open": [1.0, 2.0],
                "high": [1.2, 2.2],
                "low": [0.8, 1.8],
                "close": [1.1, 2.1],
                "volume": [10.0, 11.0],
                "source_expiry": ["202504", "202504"],
                "roll_segment_id": [1, 1],
            }
        ).to_csv(csv_path, index=False)
        meta_path = csv_path.with_name(csv_path.stem + "_meta.json")
        meta_path.write_text(
            json.dumps(
                {
                    "dataset_kind": "gc_roll_chain",
                    "canonical": True,
                    "instrument": "GC",
                    "bar_size": "5 mins",
                    "expiries_used": ["202504", "202506"],
                    "source_contracts": [
                        {"expiry": "202504", "csv_path": str(raw_dir / "GC_202504_5m_old.csv")},
                        {"expiry": "202506", "csv_path": str(raw_dir / "GC_202506_5m_old.csv")},
                    ],
                    "quality_gate_passed": True,
                    "quality_gate_reasons": [],
                },
                indent=2,
            ),
            encoding="utf-8",
        )

        captured: dict[str, object] = {}

        def fake_builder(**kwargs):
            captured.update(kwargs)
            return {
                "csv_path": str(processed_dir / "GC_5m_rollchain_3bars_20250101_20250102_20260324_020202.csv"),
                "meta_path": str(processed_dir / "GC_5m_rollchain_3bars_20250101_20250102_20260324_020202_meta.json"),
                "chart_df": pd.DataFrame(
                    {
                        "date": pd.date_range("2025-01-01 00:00:00", periods=3, freq="5min"),
                        "open": [1.0, 2.0, 3.0],
                        "high": [1.2, 2.2, 3.2],
                        "low": [0.8, 1.8, 2.8],
                        "close": [1.1, 2.1, 3.1],
                        "volume": [10.0, 11.0, 12.0],
                    }
                ),
                "status_text": "rebuild ok",
                "quality_gate_passed": True,
                "quality_gate_reasons": [],
                "meta": {},
            }

        monkeypatch.setattr(tab_data_download_module, "update_gc_roll_chain_latest_contract", fake_builder)

        payload = tab._task_update_ibkr_csv(str(csv_path), "FUT", "209912")

        assert payload.operation == "update_ibkr_roll_chain"
        assert payload.chart_df is None
        assert payload.status_ok is True
        assert payload.output_path == str(processed_dir / "GC_5m_rollchain_3bars_20250101_20250102_20260324_020202.csv")
        assert captured["expiries"] == ["202504", "202506"]
        assert captured["bar_size"] == "5 mins"
        assert captured["output_dir"] == str(processed_dir.resolve())
        assert captured["raw_dir"] == str(raw_dir)
        assert captured["preferred_contract_paths"] == {
            "202504": str(raw_dir / "GC_202504_5m_old.csv"),
            "202506": str(raw_dir / "GC_202506_5m_old.csv"),
        }
        assert pd.Timestamp(captured["start_date"]) == pd.Timestamp("2025-01-01 00:00:00")
    finally:
        tab.close()


def test_data_download_roll_chain_update_requires_canonical_meta(monkeypatch, qapp, tmp_path):
    monkeypatch.setattr(tab_data_download_module, "FigureCanvas", StubCanvas)

    tab = DataDownloadTab()
    try:
        csv_path = tmp_path / "GC_5m_rollchain_2bars_20250101_20250101_20260324_010101.csv"
        pd.DataFrame(
            {
                "date": pd.date_range("2025-01-01 00:00:00", periods=2, freq="5min"),
                "open": [1.0, 2.0],
                "high": [1.2, 2.2],
                "low": [0.8, 1.8],
                "close": [1.1, 2.1],
                "volume": [10.0, 11.0],
            }
        ).to_csv(csv_path, index=False)

        with pytest.raises(ValueError, match="canonical _meta.json"):
            tab._task_update_ibkr_csv(str(csv_path), "FUT", "202506")
    finally:
        tab.close()


def test_data_download_update_dialog_prefers_processed_dir(monkeypatch, qapp, tmp_path):
    monkeypatch.setattr(tab_data_download_module, "FigureCanvas", StubCanvas)
    raw_dir = tmp_path / "raw"
    processed_dir = tmp_path / "processed"
    raw_dir.mkdir(parents=True, exist_ok=True)
    processed_dir.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(tab_data_download_module, "RAW_DIR", str(raw_dir))
    monkeypatch.setattr(tab_data_download_module, "PROCESSED_DIR", str(processed_dir))

    captured: dict[str, object] = {}

    def fake_get_open_file_name(parent, title, start_dir, file_filter):
        captured["start_dir"] = start_dir
        return "", ""

    monkeypatch.setattr(QFileDialog, "getOpenFileName", fake_get_open_file_name)

    tab = DataDownloadTab()
    try:
        tab.on_update_ibkr_csv()
        assert captured["start_dir"] == str(processed_dir)
    finally:
        tab.close()


def test_data_download_payload_plots_only_when_auto_plot_enabled(monkeypatch, qapp):
    monkeypatch.setattr(tab_data_download_module, "FigureCanvas", StubCanvas)
    tab = DataDownloadTab()
    try:
        plotted: list[int] = []
        monkeypatch.setattr(tab, "_plot_candles", lambda df: plotted.append(int(len(df))))
        df = pd.DataFrame(
            {
                "open": np.arange(10, dtype=float),
                "high": np.arange(10, dtype=float) + 1.0,
                "low": np.arange(10, dtype=float) - 1.0,
                "close": np.arange(10, dtype=float) + 0.5,
                "volume": np.ones(10, dtype=float),
            },
            index=pd.date_range("2025-01-01 00:00:00", periods=10, freq="5min"),
        )

        tab._apply_download_payload(
            tab_data_download_module.DownloadTaskPayload(
                operation="download_ibkr",
                chart_df=None,
                status_text="saved",
                status_ok=True,
                auto_plot=False,
            )
        )
        tab._apply_download_payload(
            tab_data_download_module.DownloadTaskPayload(
                operation="load_csv",
                chart_df=df,
                status_text="loaded",
                status_ok=True,
                auto_plot=True,
            )
        )

        assert plotted == [10]
    finally:
        tab.close()


def test_data_download_tab_init_does_not_eagerly_init_tradingview_client(monkeypatch, qapp):
    calls: list[dict[str, object]] = []

    class StubTradingViewClient:
        def __init__(self, *args, **kwargs):
            calls.append(dict(kwargs))

    monkeypatch.setattr(tab_data_download_module, "FigureCanvas", StubCanvas)
    monkeypatch.setattr(tab_data_download_module, "TradingViewClient", StubTradingViewClient)

    tab = DataDownloadTab()
    try:
        assert calls == []
    finally:
        tab.close()


def test_data_download_shutdown_stops_running_worker(monkeypatch, qapp):
    class StubSignal:
        def __init__(self):
            self.disconnected = 0

        def disconnect(self, *_args, **_kwargs):
            self.disconnected += 1

    class StubWorker:
        def __init__(self):
            self.progress_text = StubSignal()
            self.finished = StubSignal()
            self.stopped = False
            self.wait_calls: list[int] = []
            self.terminated = False
            self.running = True

        def stop(self):
            self.stopped = True

        def isRunning(self):
            return self.running

        def wait(self, wait_ms):
            self.wait_calls.append(int(wait_ms))
            self.running = False
            return True

        def terminate(self):
            self.terminated = True

    monkeypatch.setattr(tab_data_download_module, "FigureCanvas", StubCanvas)
    tab = DataDownloadTab()
    try:
        worker = StubWorker()
        tab._task_worker = worker

        tab.shutdown()

        assert tab._task_worker is None
        assert worker.stopped is True
        assert worker.wait_calls == [2000]
        assert worker.terminated is False
        assert worker.progress_text.disconnected == 1
        assert worker.finished.disconnected == 1
    finally:
        tab.close()