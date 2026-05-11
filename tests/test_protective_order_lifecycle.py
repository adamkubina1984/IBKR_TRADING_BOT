from __future__ import annotations

from ibkr_trading_bot.core.services.live import BrokerOrder, PositionState, ProtectiveOrderRequest, ProtectiveOrdersManager


def _short_position() -> PositionState:
    return PositionState(side="SHORT", quantity=1.0, avg_price=100.0, opened_at="2026-04-29T12:30:00Z")


def _short_stop_order(stop_price: float) -> BrokerOrder:
    return BrokerOrder(
        order_id="ord-stop",
        symbol="GC",
        action="BUY",
        order_type="STP",
        total_quantity=1.0,
        remaining_quantity=1.0,
        status="Submitted",
        stop_price=stop_price,
    )


def _short_take_profit_order(limit_price: float) -> BrokerOrder:
    return BrokerOrder(
        order_id="ord-tp",
        symbol="GC",
        action="BUY",
        order_type="LMT",
        total_quantity=1.0,
        remaining_quantity=1.0,
        status="Submitted",
        limit_price=limit_price,
    )


def test_short_position_creates_missing_protective_bracket():
    manager = ProtectiveOrdersManager()
    plan = manager.plan(
        _short_position(),
        working_orders=[],
        request=ProtectiveOrderRequest(stop_loss_price=101.0, take_profit_price=97.0),
    )

    assert [instruction.action for instruction in plan.instructions] == ["CREATE", "CREATE"]
    assert [instruction.side for instruction in plan.instructions] == ["BUY", "BUY"]


def test_short_position_only_tightens_existing_protective_orders():
    manager = ProtectiveOrdersManager()
    tighter = manager.plan(
        _short_position(),
        working_orders=[_short_stop_order(101.5), _short_take_profit_order(96.5)],
        request=ProtectiveOrderRequest(stop_loss_price=101.0, take_profit_price=97.0),
    )
    idempotent = manager.plan(
        _short_position(),
        working_orders=[_short_stop_order(101.0), _short_take_profit_order(97.0)],
        request=ProtectiveOrderRequest(stop_loss_price=101.0, take_profit_price=97.0),
    )
    looser = manager.plan(
        _short_position(),
        working_orders=[_short_stop_order(101.0), _short_take_profit_order(97.0)],
        request=ProtectiveOrderRequest(stop_loss_price=101.5, take_profit_price=96.5),
    )

    assert [instruction.action for instruction in tighter.instructions] == ["AMEND", "AMEND"]
    assert tighter.instructions[0].stop_price == 101.0
    assert tighter.instructions[1].limit_price == 97.0
    assert idempotent.instructions == ()
    assert looser.instructions == ()