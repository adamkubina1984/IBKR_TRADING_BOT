from __future__ import annotations

from ibkr_trading_bot.core.services.live import BrokerExecution, BrokerOrder, BrokerPosition, PositionState, RuntimeState
from ibkr_trading_bot.core.services.live.position_reconciler import PositionReconciler


def _base_state() -> RuntimeState:
    return RuntimeState(
        session_id="session-1",
        instrument="GC",
        position=PositionState(side="LONG", quantity=1.0, avg_price=100.0, opened_at="2026-04-29T12:30:00Z"),
        protective_order_ids=("ord-stop", "ord-tp"),
        broker_order_ids=("ord-stop", "ord-tp"),
        pending_order_ids=("ord-stop", "ord-tp"),
    )


def _stop_order(order_id: str = "ord-stop") -> BrokerOrder:
    return BrokerOrder(
        order_id=order_id,
        symbol="GC",
        action="SELL",
        order_type="STP",
        total_quantity=1.0,
        remaining_quantity=1.0,
        status="Submitted",
        stop_price=99.0,
    )


def _take_profit_order(order_id: str = "ord-tp") -> BrokerOrder:
    return BrokerOrder(
        order_id=order_id,
        symbol="GC",
        action="SELL",
        order_type="LMT",
        total_quantity=1.0,
        remaining_quantity=1.0,
        status="Submitted",
        limit_price=103.0,
    )


def test_duplicate_broker_positions_require_safe_stop():
    reconciler = PositionReconciler()
    report = reconciler.reconcile(
        _base_state(),
        broker_positions=[
            BrokerPosition(account="DU1", symbol="GC", quantity=1.0, side="LONG", avg_cost=100.0),
            BrokerPosition(account="DU1", symbol="GC", quantity=1.0, side="LONG", avg_cost=100.1),
        ],
        open_orders=[_stop_order(), _take_profit_order()],
        fills=[],
    )

    assert report.status == "SAFE_STOP"
    assert "MULTIPLE_BROKER_POSITIONS" in report.safe_stop_reasons


def test_duplicate_protective_stops_require_safe_stop():
    reconciler = PositionReconciler()
    report = reconciler.reconcile(
        _base_state(),
        broker_positions=[BrokerPosition(account="DU1", symbol="GC", quantity=1.0, side="LONG", avg_cost=100.0)],
        open_orders=[_stop_order("ord-stop-a"), _stop_order("ord-stop-b"), _take_profit_order()],
        fills=[],
    )

    assert report.status == "SAFE_STOP"
    assert "DUPLICATE_PROTECTIVE_STOP" in report.safe_stop_reasons


def test_unmatched_fill_requires_safe_stop():
    reconciler = PositionReconciler()
    report = reconciler.reconcile(
        _base_state(),
        broker_positions=[BrokerPosition(account="DU1", symbol="GC", quantity=1.0, side="LONG", avg_cost=100.0)],
        open_orders=[_stop_order(), _take_profit_order()],
        fills=[BrokerExecution(execution_id="fill-1", order_id="ord-unknown", symbol="GC", side="BOT", quantity=1.0)],
    )

    assert report.status == "SAFE_STOP"
    assert "UNMATCHED_BROKER_FILL" in report.safe_stop_reasons