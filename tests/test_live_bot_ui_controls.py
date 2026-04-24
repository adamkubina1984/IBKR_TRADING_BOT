"""Smoke-test pro UI control workflow Tab 5 (Start/Reset/Obchodovat)."""
import pandas as pd
from PySide6.QtCore import QCoreApplication

from ._gui_test_helpers import StubCanvas


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
        assert "OFF" in tab.btn_trade.text()

        # Zapni obchodování
        tab._on_toggle_trading(True)
        QCoreApplication.processEvents()
        assert tab._trading_enabled is True
        assert "ON" in tab.btn_trade.text()

        # Vypni obchodování
        tab._on_toggle_trading(False)
        QCoreApplication.processEvents()
        assert tab._trading_enabled is False
        assert "OFF" in tab.btn_trade.text()

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
        monkeypatch.setattr(tab, "_seed_snapshot_from_warmup_history", lambda adapter: False)

        tab._on_start()
        QCoreApplication.processEvents()

        assert len(tab._bars) == 0
        assert tab._bar_index == {}
        assert len(tab.live_df) == 0
        assert tab._live_pos == 0
        assert tab._open_trade is None
        assert len(tab._trades) == 0
    finally:
        tab.close()
