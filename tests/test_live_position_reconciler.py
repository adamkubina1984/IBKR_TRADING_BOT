from __future__ import annotations

from ibkr_trading_bot.core.services.live import BrokerOrder, BrokerPosition, PositionState, RuntimeState
from ibkr_trading_bot.core.services.live.position_reconciler import PositionReconciler


def _base_state() -> RuntimeState:
    return RuntimeState(
        session_id="session-1",
        instrument="GC",
        position=PositionState(side="LONG", quantity=1.0, avg_price=100.0, opened_at="2026-04-29T12:30:00Z"),
        protective_order_ids=("ord-stop", "ord-tp"),
        broker_order_ids=("ord-stop", "ord-tp"),
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


def test_reconcile_restart_with_open_position_and_complete_bracket_is_ok():
    reconciler = PositionReconciler()
    report = reconciler.reconcile(
        _base_state(),
        broker_positions=[BrokerPosition(account="DU123456", symbol="GC", quantity=1.0, side="LONG", avg_cost=100.0)],
        open_orders=[_stop_order(), _take_profit_order()],
        fills=[],
    )

    assert report.status == "OK"
    assert report.safe_stop_reasons == ()
    assert report.protective_orders.stop_order_id == "ord-stop"
    assert report.protective_orders.take_profit_order_id == "ord-tp"


def test_reconcile_missing_take_profit_requires_safe_stop():
    reconciler = PositionReconciler()
    report = reconciler.reconcile(
        _base_state(),
        broker_positions=[BrokerPosition(account="DU123456", symbol="GC", quantity=1.0, side="LONG", avg_cost=100.0)],
        open_orders=[_stop_order()],
        fills=[],
    )

    assert report.status == "SAFE_STOP"
    assert "MISSING_PROTECTIVE_TAKE_PROFIT" in report.safe_stop_reasons


def test_reconcile_orphan_working_order_requires_safe_stop():
    reconciler = PositionReconciler()
    orphan = BrokerOrder(
        order_id="ord-orphan",
        symbol="GC",
        action="BUY",
        order_type="LMT",
        total_quantity=1.0,
        remaining_quantity=1.0,
        status="Submitted",
        limit_price=99.5,
    )

    report = reconciler.reconcile(
        _base_state(),
        broker_positions=[BrokerPosition(account="DU123456", symbol="GC", quantity=1.0, side="LONG", avg_cost=100.0)],
        open_orders=[_stop_order(), _take_profit_order(), orphan],
        fills=[],
    )

    assert report.status == "SAFE_STOP"
    assert "ORPHAN_WORKING_ORDER" in report.safe_stop_reasons