from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Literal

from .broker_dtos import BrokerOrder
from .runtime_state import PositionState

ProtectiveInstructionAction = Literal["CREATE", "AMEND"]
ProtectiveOrderRole = Literal["STOP_LOSS", "TAKE_PROFIT"]

_CLOSED_ORDER_STATUSES = {"FILLED", "CANCELLED", "API_CANCELLED", "INACTIVE"}


def _is_open_order(order: BrokerOrder) -> bool:
    return (order.status or "").upper() not in _CLOSED_ORDER_STATUSES


def _closing_action(position_side: str) -> str | None:
    normalized = position_side.strip().upper()
    if normalized == "LONG":
        return "SELL"
    if normalized == "SHORT":
        return "BUY"
    return None


def _protective_role(position_side: str, order: BrokerOrder) -> ProtectiveOrderRole | None:
    closing_action = _closing_action(position_side)
    if closing_action is None or (order.action or "").upper() != closing_action:
        return None
    order_type = (order.order_type or "").upper()
    if order.stop_price is not None and order_type in {"STP", "STP LMT", "TRAIL", "TRAIL LIMIT"}:
        return "STOP_LOSS"
    if order.limit_price is not None and order_type in {"LMT", "MIT", "LIT"}:
        return "TAKE_PROFIT"
    return None


def _normalize_tif(value: str) -> str:
    text = str(value or "GTC").strip().upper()
    return text or "GTC"


def _is_tighter_stop(position_side: str, current_price: float, desired_price: float) -> bool:
    if position_side == "LONG":
        return desired_price > current_price
    if position_side == "SHORT":
        return desired_price < current_price
    return False


def _is_tighter_take_profit(position_side: str, current_price: float, desired_price: float) -> bool:
    if position_side == "LONG":
        return desired_price < current_price
    if position_side == "SHORT":
        return desired_price > current_price
    return False


@dataclass(frozen=True)
class ProtectiveOrderRequest:
    stop_loss_price: float | None = None
    take_profit_price: float | None = None
    quantity: float | None = None
    tif: str = "GTC"
    outside_rth: bool = False

    def __post_init__(self) -> None:
        if self.stop_loss_price is not None:
            object.__setattr__(self, "stop_loss_price", float(self.stop_loss_price))
        if self.take_profit_price is not None:
            object.__setattr__(self, "take_profit_price", float(self.take_profit_price))
        if self.quantity is not None:
            quantity = float(self.quantity)
            if quantity <= 0.0:
                raise ValueError("quantity must be positive when provided.")
            object.__setattr__(self, "quantity", quantity)
        object.__setattr__(self, "tif", _normalize_tif(self.tif))


@dataclass(frozen=True)
class ProtectiveOrderInstruction:
    action: ProtectiveInstructionAction
    role: ProtectiveOrderRole
    side: str
    order_type: str
    quantity: float
    tif: str
    outside_rth: bool
    order_id: str | None = None
    stop_price: float | None = None
    limit_price: float | None = None
    reason: str = ""


@dataclass(frozen=True)
class ProtectiveOrdersPlan:
    instructions: tuple[ProtectiveOrderInstruction, ...] = field(default_factory=tuple)
    current_stop_order_id: str | None = None
    current_take_profit_order_id: str | None = None


class ProtectiveOrdersManager:
    def plan(
        self,
        position: PositionState,
        working_orders: Iterable[BrokerOrder],
        request: ProtectiveOrderRequest,
    ) -> ProtectiveOrdersPlan:
        if position.side == "FLAT":
            return ProtectiveOrdersPlan()

        candidate_orders = [order for order in working_orders if _is_open_order(order)]
        current_stop_order = None
        current_take_profit_order = None
        for order in candidate_orders:
            role = _protective_role(position.side, order)
            if role == "STOP_LOSS" and current_stop_order is None:
                current_stop_order = order
            elif role == "TAKE_PROFIT" and current_take_profit_order is None:
                current_take_profit_order = order

        quantity = float(request.quantity if request.quantity is not None else position.quantity)
        side = _closing_action(position.side)
        instructions: list[ProtectiveOrderInstruction] = []

        if request.stop_loss_price is not None:
            if current_stop_order is None:
                instructions.append(
                    ProtectiveOrderInstruction(
                        action="CREATE",
                        role="STOP_LOSS",
                        side=side or "SELL",
                        order_type="STP",
                        quantity=quantity,
                        tif=request.tif,
                        outside_rth=request.outside_rth,
                        stop_price=request.stop_loss_price,
                        reason="missing_protective_stop",
                    )
                )
            elif current_stop_order.stop_price is not None and _is_tighter_stop(
                position.side,
                float(current_stop_order.stop_price),
                float(request.stop_loss_price),
            ):
                instructions.append(
                    ProtectiveOrderInstruction(
                        action="AMEND",
                        role="STOP_LOSS",
                        side=side or "SELL",
                        order_type=current_stop_order.order_type or "STP",
                        quantity=quantity,
                        tif=request.tif,
                        outside_rth=request.outside_rth,
                        order_id=current_stop_order.order_id,
                        stop_price=request.stop_loss_price,
                        reason="tighten_protective_stop",
                    )
                )

        if request.take_profit_price is not None:
            if current_take_profit_order is None:
                instructions.append(
                    ProtectiveOrderInstruction(
                        action="CREATE",
                        role="TAKE_PROFIT",
                        side=side or "SELL",
                        order_type="LMT",
                        quantity=quantity,
                        tif=request.tif,
                        outside_rth=request.outside_rth,
                        limit_price=request.take_profit_price,
                        reason="missing_protective_take_profit",
                    )
                )
            elif current_take_profit_order.limit_price is not None and _is_tighter_take_profit(
                position.side,
                float(current_take_profit_order.limit_price),
                float(request.take_profit_price),
            ):
                instructions.append(
                    ProtectiveOrderInstruction(
                        action="AMEND",
                        role="TAKE_PROFIT",
                        side=side or "SELL",
                        order_type=current_take_profit_order.order_type or "LMT",
                        quantity=quantity,
                        tif=request.tif,
                        outside_rth=request.outside_rth,
                        order_id=current_take_profit_order.order_id,
                        limit_price=request.take_profit_price,
                        reason="tighten_protective_take_profit",
                    )
                )

        return ProtectiveOrdersPlan(
            instructions=tuple(instructions),
            current_stop_order_id=current_stop_order.order_id if current_stop_order is not None else None,
            current_take_profit_order_id=(
                current_take_profit_order.order_id if current_take_profit_order is not None else None
            ),
        )