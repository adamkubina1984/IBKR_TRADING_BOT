import pytest

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