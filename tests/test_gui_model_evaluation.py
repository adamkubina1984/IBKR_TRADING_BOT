import json

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


def test_model_evaluation_task_wrappers_forward_exit_policy(monkeypatch):
    calls: dict[str, dict] = {}

    monkeypatch.setattr(
        tab_model_evaluation_module.model_eval_runtime,
        "run_model_evaluation",
        lambda **kwargs: calls.setdefault("eval", kwargs),
    )
    monkeypatch.setattr(
        tab_model_evaluation_module.model_eval_runtime,
        "recalculate_metrics_from_predictions",
        lambda **kwargs: calls.setdefault("recalc", kwargs),
    )
    monkeypatch.setattr(
        tab_model_evaluation_module.model_eval_runtime,
        "run_auto_threshold_search_from_context",
        lambda **kwargs: calls.setdefault("search", kwargs),
    )

    ModelEvaluationTab._task_run_full_evaluation(
        model=object(),
        metadata={},
        data_path="demo.csv",
        scope_mode="holdout",
        fee_per_trade=0.0,
        entry_threshold=0.6,
        exit_threshold=0.7,
        exit_policy="legacy_flat_exit",
    )
    ModelEvaluationTab._task_recalculate_metrics(
        y_pred_raw=[1],
        confidence_arr=[0.9],
        y_true_current=[1],
        df_current=pd.DataFrame({"close": [100.0]}),
        fee_per_trade=0.0,
        entry_threshold=0.6,
        exit_threshold=0.7,
        exit_policy="legacy_flat_exit",
    )
    ModelEvaluationTab._task_auto_threshold_search(
        y_pred_raw=[1],
        confidence_arr=[0.9],
        y_true_current=[1],
        df_current=pd.DataFrame({"close": [100.0]}),
        fee_per_trade=0.0,
        current_entry=0.6,
        current_exit=0.7,
        exit_policy="legacy_flat_exit",
    )

    assert calls["eval"]["exit_policy"] == "legacy_flat_exit"
    assert calls["recalc"]["exit_policy"] == "legacy_flat_exit"
    assert calls["search"]["exit_policy"] == "legacy_flat_exit"


def test_model_evaluation_workers_use_active_exit_policy(monkeypatch, qapp):
    created: list[dict[str, object]] = []

    class _DummySignal:
        def connect(self, _callback):
            return None

    class _StubTaskWorker:
        def __init__(self, _fn, **kwargs):
            created.append(kwargs)
            self.progress_text = _DummySignal()
            self.result = _DummySignal()
            self.error = _DummySignal()
            self.finished = _DummySignal()

        def start(self):
            return None

    monkeypatch.setattr(tab_model_evaluation_module, "TaskWorker", _StubTaskWorker)

    tab = ModelEvaluationTab()
    try:
        tab.model_metadata = {"user_settings": {"exit_policy": "legacy_flat"}}
        tab.loaded_model = object()
        tab.model_path = "demo.pkl"
        tab.data_path = "demo.csv"
        tab.y_pred_raw = pd.Series([1]).to_numpy()
        tab.confidence_arr = pd.Series([0.9]).to_numpy()
        tab.y_true_current = pd.Series([1]).to_numpy()
        tab.df_current = pd.DataFrame({"close": [100.0]})

        tab._start_evaluation_worker()
        tab._run_params_recalc()
        tab._start_auto_threshold_worker()

        assert len(created) == 3
        assert all(kwargs["exit_policy"] == "flat_on_weak_signal" for kwargs in created)
        assert "flat_on_weak_signal" in tab.lbl_exit_policy.text()
    finally:
        tab.close()


def test_model_evaluation_save_model_settings_persists_active_exit_policy(monkeypatch, qapp, tmp_path):
    monkeypatch.setattr(tab_model_evaluation_module.QMessageBox, "information", lambda *args, **kwargs: None)

    tab = ModelEvaluationTab()
    try:
        model_path = tmp_path / "demo.pkl"
        tab.model_path = str(model_path)
        tab.model_metadata = {"user_settings": {"exit_policy": "legacy_flat"}}
        tab.et_spin.setValue(0.61)
        tab.ext_spin.setValue(0.72)
        monkeypatch.setattr(tab, "_resolve_ternary_thresholds", lambda: (0.31, 0.69))

        tab._on_save_model_settings()

        meta_path = model_path.with_name("demo_meta.json")
        saved = json.loads(meta_path.read_text(encoding="utf-8"))

        assert saved["user_settings"]["entry_threshold"] == pytest.approx(0.61)
        assert saved["user_settings"]["exit_threshold"] == pytest.approx(0.72)
        assert saved["user_settings"]["exit_policy"] == "flat_on_weak_signal"
        assert saved["exit_policy"] == "flat_on_weak_signal"
    finally:
        tab.close()