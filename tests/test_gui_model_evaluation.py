import pandas as pd
import pytest

from ibkr_trading_bot.gui import tab_model_evaluation as tab_model_evaluation_module
from ibkr_trading_bot.gui.tab_model_evaluation import AutoThresholdPayload, ModelEvaluationTab


def test_model_evaluation_auto_threshold_result_updates_spinboxes(monkeypatch, qapp):
    tab = ModelEvaluationTab()
    try:
        tab.et_spin.setValue(0.61)
        tab.ext_spin.setValue(0.72)
        tab._auto_request_id = 7
        calls = []
        monkeypatch.setattr(tab, "_run_params_recalc", lambda: calls.append("recalc"))

        payload = AutoThresholdPayload(
            best_entry=0.34,
            best_exit=0.34,
            best_score=123.0,
            best_metrics={"profit_net": 123.0, "trades": 2},
        )
        tab._on_auto_threshold_result(7, payload)
        qapp.processEvents()

        assert tab.et_spin.value() == pytest.approx(0.34)
        assert tab.ext_spin.value() == pytest.approx(0.34)
        assert tab._pending_auto_threshold_dialog == payload
        assert calls == ["recalc"]
    finally:
        tab.close()


def test_model_evaluation_extracts_trade_dataframe_from_shared_results(qapp):
    tab = ModelEvaluationTab()
    try:
        results = {
            "trades": [
                {
                    "entry_time": "2026-03-18 10:00:00",
                    "exit_time": "2026-03-18 10:05:00",
                    "direction": "LONG",
                    "entry_price": 100.0,
                    "exit_price": 101.5,
                    "pnl": 1.5,
                    "pnl_net": 1.2,
                    "exit_reason": "exit_to_flat",
                }
            ]
        }

        df = tab._extract_trades_df(results)

        assert isinstance(df, pd.DataFrame)
        assert list(df.columns[:6]) == ["entry_time", "exit_time", "direction", "entry_price", "exit_price", "pnl"]
        assert df.iloc[0]["direction"] == "LONG"
        assert df.iloc[0]["pnl_net"] == 1.2
    finally:
        tab.close()


def test_model_evaluation_restores_last_data_csv_path(monkeypatch, qapp, tmp_path):
    class _DummySettings:
        _store: dict[str, str] = {}

        def __init__(self, *args, **kwargs):
            pass

        def value(self, key, default=None):
            return self._store.get(str(key), default)

        def setValue(self, key, value):
            self._store[str(key)] = str(value)

        def sync(self):
            return None

    _DummySettings._store = {}
    monkeypatch.setattr(tab_model_evaluation_module, "QSettings", _DummySettings)

    csv_path = tmp_path / "dataset.csv"
    csv_path.write_text("timestamp,open,high,low,close,volume\n1,1,1,1,1,1\n", encoding="utf-8")

    first = ModelEvaluationTab()
    try:
        first.set_data_path(str(csv_path))
        assert first.data_path is not None
    finally:
        first.close()

    second = ModelEvaluationTab()
    try:
        assert second.data_path == str(csv_path.resolve())
        assert str(csv_path.resolve()) in second.data_label.text()
    finally:
        second.close()