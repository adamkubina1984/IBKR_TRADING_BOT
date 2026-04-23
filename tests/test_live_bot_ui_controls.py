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
