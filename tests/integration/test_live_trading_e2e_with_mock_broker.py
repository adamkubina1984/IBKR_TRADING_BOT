from __future__ import annotations

from ibkr_trading_bot.core.services.live import BrokerOrder, BrokerPosition, ExecutionJournal, RuntimeStateStore
from ibkr_trading_bot.core.services.live_trading_execution_service import LiveTradingExecutionConfig, LiveTradingExecutionService


def _stop_order() -> BrokerOrder:
    return BrokerOrder(
        order_id="ord-stop",
        symbol="GC",
        action="SELL",
        order_type="STP",
        total_quantity=1.0,
        remaining_quantity=1.0,
        status="Submitted",
        stop_price=99.0,
    )


def _take_profit_order() -> BrokerOrder:
    return BrokerOrder(
        order_id="ord-tp",
        symbol="GC",
        action="SELL",
        order_type="LMT",
        total_quantity=1.0,
        remaining_quantity=1.0,
        status="Submitted",
        limit_price=103.0,
    )


def test_live_service_e2e_with_mock_broker_restart_and_reconcile(tmp_path):
    service = LiveTradingExecutionService(
        LiveTradingExecutionConfig(
            strategy_id="e2e-mock-broker",
            instrument="GC",
            timeframe="5m",
            min_bars_for_health=10,
        ),
        journal=ExecutionJournal(tmp_path / "execution.journal.jsonl"),
        state_store=RuntimeStateStore(tmp_path / "runtime.state.json"),
    )
    service.arm_trading(baseline_profit_per_bar=0.8, actor="tester")

    first = service.process_closed_bar(
        "2026-04-29T12:30:00Z",
        100.0,
        model_direction="LONG",
        confidence=1.0,
    )
    assert [intent.action for intent in first.order_intents] == ["ENTRY_LONG"]

    stable = service.process_closed_bar(
        "2026-04-29T12:35:00Z",
        100.2,
        model_direction="LONG",
        confidence=1.0,
        broker_positions=[BrokerPosition(account="DU123456", symbol="GC", quantity=1.0, side="LONG", avg_cost=100.0)],
        open_orders=[_stop_order(), _take_profit_order()],
    )
    assert stable.status.mode == "LIVE"
    assert stable.reconciliation_report is not None
    assert stable.reconciliation_report.status == "OK"

    flip = service.process_closed_bar(
        "2026-04-29T12:40:00Z",
        101.0,
        model_direction="SHORT",
        confidence=1.0,
        broker_positions=[BrokerPosition(account="DU123456", symbol="GC", quantity=1.0, side="LONG", avg_cost=100.0)],
        open_orders=[_stop_order(), _take_profit_order()],
    )
    assert [intent.action for intent in flip.order_intents] == ["FLIP_TO_SHORT"]
    assert flip.trade_result is not None
    assert float(flip.trade_result.closed_trade.pnl) == 1.0

    restored = LiveTradingExecutionService(
        LiveTradingExecutionConfig(
            strategy_id="e2e-mock-broker",
            instrument="GC",
            timeframe="5m",
            min_bars_for_health=10,
        ),
        journal=ExecutionJournal(tmp_path / "execution.journal.jsonl"),
        state_store=RuntimeStateStore(tmp_path / "runtime.state.json"),
    )
    assert restored.status.runtime_state.position.side == "SHORT"
    assert restored.status.mode == "LIVE"