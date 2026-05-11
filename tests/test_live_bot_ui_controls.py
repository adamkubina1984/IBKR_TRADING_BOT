"""Smoke-test pro UI control workflow Tab 5 (Start/Reset/Obchodovat)."""
import pytest
import pandas as pd
from PySide6.QtCore import QCoreApplication

from ._gui_test_helpers import StubCanvas


@pytest.fixture(autouse=True)
def _patch_default_live_controller(monkeypatch, tmp_path):
    from ibkr_trading_bot.core.services.live_service_controller import LiveServiceController
    from ibkr_trading_bot.gui import tab_live_bot as tab_live_bot_module

    def _build_controller(self):
        return LiveServiceController(
            strategy_id="tab5-live-test",
            instrument=(self.config.symbol or "GOLD").strip(),
            exchange=(self.config.exchange or "TVC").strip(),
            timeframe=(self.config.bar_size or "5 min").strip(),
            entry_threshold=float(self._curr_entry_thr),
            exit_threshold=float(self._curr_exit_thr),
            use_ma_alignment=bool(self.config.use_and_ensemble),
            freshness_timeout_sec=max(60, int(self.config.max_fresh_age_min) * 60),
            session_root=tmp_path,
        )

    monkeypatch.setattr(tab_live_bot_module.LiveBotWidget, "_build_live_service_controller", _build_controller)


def test_live_session_control_start_reset_obchodovat(monkeypatch, qapp):
    """Smoke-test pro hlavní UI workflow: trading toggle a reset lifecycle."""
    from ibkr_trading_bot.gui import tab_live_bot as tab_live_bot_module

    monkeypatch.setattr(tab_live_bot_module, "FigureCanvas", StubCanvas)
    tab = tab_live_bot_module.LiveBotWidget()
    try:
        # Počáteční stav: trading vypnut, runtime state čist
        assert tab._trading_enabled is False
        assert tab._live_pos == 0
        assert len(tab._trades) == 0
        assert tab._open_trade is None
        assert "Arm trading" in tab.btn_trade.text()
        assert "OFF" in tab.btn_trade.text()

        # Zapni obchodování
        tab._on_toggle_trading(True)
        QCoreApplication.processEvents()
        assert tab._trading_enabled is True
        assert "Arm trading" in tab.btn_trade.text()
        assert "ON" in tab.btn_trade.text()
        assert tab.lbl_mode.text() == "Mode: LIVE"

        # Vypni obchodování
        tab._on_toggle_trading(False)
        QCoreApplication.processEvents()
        assert tab._trading_enabled is False
        assert "Arm trading" in tab.btn_trade.text()
        assert "OFF" in tab.btn_trade.text()
        assert tab.lbl_mode.text() == "Mode: OBSERVE"

        # Simuluj reset - mělo by vyčistit stav
        tab._reset_runtime_state(full_reset=True)
        QCoreApplication.processEvents()
        assert tab._live_pos == 0
        assert tab._open_trade is None
        assert len(tab._trades) == 0
        assert tab._trading_enabled is False
    finally:
        tab.close()


def test_live_reset_clears_trades_and_position(monkeypatch, qapp):
    """Verify reset clears runtime state completely."""
    from ibkr_trading_bot.gui import tab_live_bot as tab_live_bot_module

    monkeypatch.setattr(tab_live_bot_module, "FigureCanvas", StubCanvas)
    tab = tab_live_bot_module.LiveBotWidget()
    try:
        # Set some state manually
        tab._live_pos = 1
        tab._trades = [{"entry": 100.0, "exit": 101.0, "pnl": 1.0}]
        tab._open_trade = {"direction": "LONG", "entry_price": 100.0}
        assert len(tab._trades) == 1
        assert tab._live_pos == 1

        # Reset
        tab._reset_runtime_state(full_reset=True)
        QCoreApplication.processEvents()

        # Verify state is cleared
        assert tab._live_pos == 0
        assert len(tab._trades) == 0
        assert tab._open_trade is None
    finally:
        tab.close()


def test_start_clears_stale_runtime_state_before_new_session(monkeypatch, qapp):
    """New Start session must clear previous bars/trades even when model path is changed."""
    from ibkr_trading_bot.gui import tab_live_bot as tab_live_bot_module

    class _WarmServiceStub:
        def __init__(self, base_service, config=None, **kwargs):
            self.base = base_service
            self.state = "WARM-UP"
            self.config = config

        def start(self, symbol, exchange, timeframe):
            return None

    monkeypatch.setattr(tab_live_bot_module, "FigureCanvas", StubCanvas)
    monkeypatch.setattr(tab_live_bot_module, "LiveWarmupService", _WarmServiceStub)

    tab = tab_live_bot_module.LiveBotWidget()
    try:
        # Simulate stale state from previous run
        tab._bars = [{
            "time": pd.Timestamp("2026-04-24T10:00:00Z"),
            "open": 100.0,
            "high": 101.0,
            "low": 99.0,
            "close": 100.5,
            "volume": 10.0,
        }]
        tab._bar_index = {int(pd.Timestamp("2026-04-24T10:00:00Z").value): 0}
        tab._trades = [{"entry": 100.0, "exit": 101.0, "pnl": 1.0}]
        tab._live_pos = 1
        tab._open_trade = {"direction": "LONG", "entry_price": 100.0}
        tab.live_df = pd.DataFrame(
            [{
                "timestamp": pd.Timestamp("2026-04-24T10:00:00Z"),
                "open": 100.0,
                "high": 101.0,
                "low": 99.0,
                "close": 100.5,
                "volume": 10.0,
            }]
        )

        # Keep start local to this test (no real threads/network)
        monkeypatch.setattr(tab, "_load_models", lambda: True)
        monkeypatch.setattr(tab, "_apply_tf_presets", lambda: None)
        monkeypatch.setattr(tab, "_start_worker", lambda: None)
        monkeypatch.setattr(tab, "_start_warmup_worker", lambda: None)

        tab._on_start()
        QCoreApplication.processEvents()

        assert len(tab._bars) == 0
        assert tab._bar_index == {}
        assert len(tab.live_df) == 0
        assert tab._live_pos == 0
        assert tab._open_trade is None
        assert len(tab._trades) == 0
        assert tab.lbl_mode.text() == "Mode: OBSERVE"
    finally:
        tab.close()


def test_start_schedules_warmup_in_background(monkeypatch, qapp):
    from ibkr_trading_bot.gui import tab_live_bot as tab_live_bot_module

    class _Signal:
        def connect(self, callback):
            return None

        def disconnect(self):
            return None

    scheduled = {"workers": []}

    class _WorkerStub:
        def __init__(self, task_fn, *args, **kwargs):
            self.task_fn = task_fn
            self.args = args
            self.kwargs = kwargs
            self.progress_text = _Signal()
            self.result = _Signal()
            self.error = _Signal()
            self.finished = _Signal()
            self.started = False
            scheduled["workers"].append(self)

        def start(self):
            self.started = True

        def stop(self):
            return None

        def isRunning(self):
            return False

        def wait(self, ms):
            return True

        def deleteLater(self):
            return None

    class _WarmServiceMustNotRunInline:
        def __init__(self, *args, **kwargs):
            raise AssertionError("LiveWarmupService should not be constructed inline in _on_start")

    monkeypatch.setattr(tab_live_bot_module, "FigureCanvas", StubCanvas)
    monkeypatch.setattr(tab_live_bot_module, "TaskWorker", _WorkerStub)
    monkeypatch.setattr(tab_live_bot_module, "LiveWarmupService", _WarmServiceMustNotRunInline)

    tab = tab_live_bot_module.LiveBotWidget()
    try:
        monkeypatch.setattr(tab, "_load_models", lambda: True)
        monkeypatch.setattr(tab, "_apply_tf_presets", lambda: None)
        monkeypatch.setattr(tab, "_sync_ib_config_from_ui", lambda *args, **kwargs: None)
        monkeypatch.setattr(tab, "_reset_runtime_state", lambda *args, **kwargs: None)
        monkeypatch.setattr(tab, "_start_worker", lambda: None)
        tab.live_controller = None

        tab._on_start()

        assert len(scheduled["workers"]) == 1
        assert scheduled["workers"][0].task_fn is tab_live_bot_module._task_run_live_warmup
        assert scheduled["workers"][0].started is True
        assert tab._warmup_worker is scheduled["workers"][0]
    finally:
        tab._warmup_worker = None
        tab.close()


def test_warmup_result_starts_stream_without_bootstrap_fallback_when_snapshot_is_ready(monkeypatch, qapp):
    from ibkr_trading_bot.gui import tab_live_bot as tab_live_bot_module

    monkeypatch.setattr(tab_live_bot_module, "FigureCanvas", StubCanvas)
    tab = tab_live_bot_module.LiveBotWidget()
    try:
        payload = tab_live_bot_module.LiveWarmupPayload(
            warm_service=type("_Warm", (), {"config": None, "state": "LIVE"})(),
            adapter=object(),
        )
        state = {"launch_called": 0, "bootstrap_called": 0}

        monkeypatch.setattr(tab, "_seed_snapshot_from_warmup_history", lambda adapter: True)
        monkeypatch.setattr(tab, "_sync_live_mode_label", lambda: None)
        monkeypatch.setattr(tab, "_launch_stream_worker", lambda: state.__setitem__("launch_called", state["launch_called"] + 1))
        monkeypatch.setattr(tab, "_start_worker", lambda: state.__setitem__("bootstrap_called", state["bootstrap_called"] + 1))

        tab._warmup_request_id = 3
        tab._bootstrap_done = True
        tab._warmup_done = False

        tab._on_warmup_result(3, payload)

        assert state["launch_called"] == 1
        assert state["bootstrap_called"] == 0
    finally:
        tab.close()
