from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Literal

from .broker_dtos import BrokerExecution, BrokerOrder, BrokerPosition
from .runtime_state import RuntimeState

ReconciliationSeverity = Literal["WARNING", "SAFE_STOP"]
ReconciliationStatus = Literal["OK", "SAFE_STOP"]

_CLOSED_ORDER_STATUSES = {"FILLED", "CANCELLED", "API_CANCELLED", "INACTIVE"}


def _normalize_text(value: object) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _matches_instrument(instrument: str | None, *candidates: object) -> bool:
    normalized_instrument = _normalize_text(instrument)
    if normalized_instrument is None:
        return True
    needle = normalized_instrument.upper()
    for candidate in candidates:
        text = _normalize_text(candidate)
        if text is None:
            continue
        haystack = text.upper()
        if haystack == needle or haystack.startswith(needle):
            return True
    return False


def _is_open_order(order: BrokerOrder) -> bool:
    return (order.status or "").upper() not in _CLOSED_ORDER_STATUSES


def _closing_action_for_position(side: str) -> str | None:
    normalized = side.strip().upper()
    if normalized == "LONG":
        return "SELL"
    if normalized == "SHORT":
        return "BUY"
    return None


def _protective_role_for_order(position_side: str, order: BrokerOrder) -> str | None:
    closing_action = _closing_action_for_position(position_side)
    if closing_action is None or (order.action or "").upper() != closing_action:
        return None
    order_type = (order.order_type or "").upper()
    if order.stop_price is not None and order_type in {"STP", "STP LMT", "TRAIL", "TRAIL LIMIT"}:
        return "STOP_LOSS"
    if order.limit_price is not None and order_type in {"LMT", "MIT", "LIT"}:
        return "TAKE_PROFIT"
    return None


@dataclass(frozen=True)
class ReconciliationIssue:
    code: str
    severity: ReconciliationSeverity
    message: str


@dataclass(frozen=True)
class ProtectiveOrdersSnapshot:
    stop_order_id: str | None = None
    take_profit_order_id: str | None = None


@dataclass(frozen=True)
class ReconciliationReport:
    status: ReconciliationStatus
    issues: tuple[ReconciliationIssue, ...] = ()
    broker_position: BrokerPosition | None = None
    protective_orders: ProtectiveOrdersSnapshot = field(default_factory=ProtectiveOrdersSnapshot)
    unmatched_fill_order_ids: tuple[str, ...] = ()

    @property
    def safe_stop_reasons(self) -> tuple[str, ...]:
        return tuple(issue.code for issue in self.issues if issue.severity == "SAFE_STOP")


class PositionReconciler:
    def __init__(self, *, quantity_tolerance: float = 1e-9) -> None:
        self.quantity_tolerance = float(quantity_tolerance)

    def reconcile(
        self,
        state: RuntimeState,
        broker_positions: Iterable[BrokerPosition],
        open_orders: Iterable[BrokerOrder],
        fills: Iterable[BrokerExecution],
    ) -> ReconciliationReport:
        issues: list[ReconciliationIssue] = []
        known_order_ids = set(state.pending_order_ids) | set(state.broker_order_ids) | set(state.protective_order_ids)

        matching_positions = [
            position
            for position in broker_positions
            if position.quantity > self.quantity_tolerance
            and _matches_instrument(state.instrument, position.symbol, position.local_symbol)
        ]
        if len(matching_positions) > 1:
            issues.append(
                ReconciliationIssue(
                    code="MULTIPLE_BROKER_POSITIONS",
                    severity="SAFE_STOP",
                    message="Multiple live broker positions match the local instrument.",
                )
            )
        broker_position = matching_positions[0] if matching_positions else None

        if state.position.side == "FLAT":
            if broker_position is not None:
                issues.append(
                    ReconciliationIssue(
                        code="ORPHAN_BROKER_POSITION",
                        severity="SAFE_STOP",
                        message="Broker reports an open position while local runtime state is flat.",
                    )
                )
        else:
            if broker_position is None:
                issues.append(
                    ReconciliationIssue(
                        code="LOCAL_POSITION_MISSING_AT_BROKER",
                        severity="SAFE_STOP",
                        message="Local runtime state expects an open position, but broker is flat.",
                    )
                )
            else:
                if broker_position.side != state.position.side:
                    issues.append(
                        ReconciliationIssue(
                            code="BROKER_POSITION_SIDE_MISMATCH",
                            severity="SAFE_STOP",
                            message="Broker position side does not match local runtime state.",
                        )
                    )
                if abs(broker_position.quantity - state.position.quantity) > self.quantity_tolerance:
                    issues.append(
                        ReconciliationIssue(
                            code="BROKER_POSITION_QUANTITY_MISMATCH",
                            severity="SAFE_STOP",
                            message="Broker position quantity does not match local runtime state.",
                        )
                    )

        working_orders = [
            order
            for order in open_orders
            if _is_open_order(order)
            and (_matches_instrument(state.instrument, order.symbol) or order.order_id in known_order_ids)
        ]
        stop_orders: list[BrokerOrder] = []
        take_profit_orders: list[BrokerOrder] = []
        for order in working_orders:
            role = _protective_role_for_order(state.position.side, order)
            if role == "STOP_LOSS":
                stop_orders.append(order)
                continue
            if role == "TAKE_PROFIT":
                take_profit_orders.append(order)
                continue
            if order.order_id not in known_order_ids:
                issues.append(
                    ReconciliationIssue(
                        code="ORPHAN_WORKING_ORDER",
                        severity="SAFE_STOP",
                        message="Broker has a working order that is not tracked by runtime state.",
                    )
                )

        if state.position.side != "FLAT":
            if len(stop_orders) == 0:
                issues.append(
                    ReconciliationIssue(
                        code="MISSING_PROTECTIVE_STOP",
                        severity="SAFE_STOP",
                        message="Open broker position is missing a protective stop order.",
                    )
                )
            elif len(stop_orders) > 1:
                issues.append(
                    ReconciliationIssue(
                        code="DUPLICATE_PROTECTIVE_STOP",
                        severity="SAFE_STOP",
                        message="Multiple protective stop orders are working for one position.",
                    )
                )
            if len(take_profit_orders) == 0:
                issues.append(
                    ReconciliationIssue(
                        code="MISSING_PROTECTIVE_TAKE_PROFIT",
                        severity="SAFE_STOP",
                        message="Open broker position is missing a protective take-profit order.",
                    )
                )
            elif len(take_profit_orders) > 1:
                issues.append(
                    ReconciliationIssue(
                        code="DUPLICATE_PROTECTIVE_TAKE_PROFIT",
                        severity="SAFE_STOP",
                        message="Multiple protective take-profit orders are working for one position.",
                    )
                )

        unmatched_fill_order_ids = tuple(
            fill.order_id
            for fill in fills
            if fill.order_id is not None
            and _matches_instrument(state.instrument, fill.symbol)
            and fill.order_id not in known_order_ids
        )
        if unmatched_fill_order_ids:
            issues.append(
                ReconciliationIssue(
                    code="UNMATCHED_BROKER_FILL",
                    severity="SAFE_STOP",
                    message="Broker fills exist that are not mapped to local runtime order ids.",
                )
            )

        status: ReconciliationStatus = "SAFE_STOP" if any(
            issue.severity == "SAFE_STOP" for issue in issues
        ) else "OK"
        return ReconciliationReport(
            status=status,
            issues=tuple(issues),
            broker_position=broker_position,
            protective_orders=ProtectiveOrdersSnapshot(
                stop_order_id=stop_orders[0].order_id if len(stop_orders) == 1 else None,
                take_profit_order_id=take_profit_orders[0].order_id if len(take_profit_orders) == 1 else None,
            ),
            unmatched_fill_order_ids=unmatched_fill_order_ids,
        )