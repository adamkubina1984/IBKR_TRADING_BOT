from __future__ import annotations

from ibkr_trading_bot.core.services.live import ExecutionJournal, RuntimeStateStore
from ibkr_trading_bot.core.services.live_trading_execution_service import LiveTradingExecutionConfig, LiveTradingExecutionService


def test_live_service_writes_decisions_and_closed_trades_to_journal(tmp_path):
    service = LiveTradingExecutionService(
        LiveTradingExecutionConfig(strategy_id="journal-test", instrument="GC", timeframe="5m", min_bars_for_health=10),
        journal=ExecutionJournal(tmp_path / "execution.journal.jsonl"),
        state_store=RuntimeStateStore(tmp_path / "runtime.state.json"),
    )
    service.arm_trading(baseline_profit_per_bar=0.8, actor="tester")
    service.process_closed_bar("2026-04-29T12:30:00Z", 100.0, model_direction="LONG", confidence=1.0)
    service.process_closed_bar("2026-04-29T12:35:00Z", 101.0, model_direction="SHORT", confidence=1.0)

    events = service.journal.read_events()
    event_types = [event.event_type for event in events]

    assert event_types[:2] == ["armed", "baseline_captured"]
    assert event_types.count("decision") == 2
    assert "trade_closed" in event_types
    trade_closed = next(event for event in events if event.event_type == "trade_closed")
    assert float(trade_closed.payload["pnl"]) == 1.0
    assert int(trade_closed.payload["side"]) == 1