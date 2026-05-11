from __future__ import annotations

from ibkr_trading_bot.core.services.live import (
    BrokerOrder,
    PositionState,
    ProtectiveOrderRequest,
    ProtectiveOrdersManager,
)


def _long_position() -> PositionState:
    return PositionState(side="LONG", quantity=1.0, avg_price=100.0, opened_at="2026-04-29T12:30:00Z")


def _stop_order(stop_price: float) -> BrokerOrder:
    return BrokerOrder(
        order_id="ord-stop",
        symbol="GC",
        action="SELL",
        order_type="STP",
        total_quantity=1.0,
        remaining_quantity=1.0,
        status="Submitted",
        stop_price=stop_price,
    )


def _take_profit_order(limit_price: float) -> BrokerOrder:
    return BrokerOrder(
        order_id="ord-tp",
        symbol="GC",
        action="SELL",
        order_type="LMT",
        total_quantity=1.0,
        remaining_quantity=1.0,
        status="Submitted",
        limit_price=limit_price,
    )


def test_plan_creates_missing_protective_orders_for_open_position():
    manager = ProtectiveOrdersManager()
    plan = manager.plan(
        _long_position(),
        working_orders=[],
        request=ProtectiveOrderRequest(stop_loss_price=99.0, take_profit_price=104.0),
    )

    assert [instruction.action for instruction in plan.instructions] == ["CREATE", "CREATE"]
    assert [instruction.role for instruction in plan.instructions] == ["STOP_LOSS", "TAKE_PROFIT"]
    assert plan.current_stop_order_id is None
    assert plan.current_take_profit_order_id is None


def test_plan_only_tightens_existing_orders_and_is_idempotent():
    manager = ProtectiveOrdersManager()
    tighter_plan = manager.plan(
        _long_position(),
        working_orders=[_stop_order(99.0), _take_profit_order(105.0)],
        request=ProtectiveOrderRequest(stop_loss_price=99.5, take_profit_price=104.0),
    )

    assert [instruction.action for instruction in tighter_plan.instructions] == ["AMEND", "AMEND"]
    assert tighter_plan.instructions[0].stop_price == 99.5
    assert tighter_plan.instructions[1].limit_price == 104.0

    idempotent_plan = manager.plan(
        _long_position(),
        working_orders=[_stop_order(99.5), _take_profit_order(104.0)],
        request=ProtectiveOrderRequest(stop_loss_price=99.5, take_profit_price=104.0),
    )
    looser_plan = manager.plan(
        _long_position(),
        working_orders=[_stop_order(99.5), _take_profit_order(104.0)],
        request=ProtectiveOrderRequest(stop_loss_price=99.0, take_profit_price=105.0),
    )

    assert idempotent_plan.instructions == ()
    assert looser_plan.instructions == ()