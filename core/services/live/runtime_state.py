from __future__ import annotations

import json
import os
import tempfile
import threading
from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from .execution_journal import ExecutionEvent, ExecutionJournal


def _utcnow_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _normalize_timestamp(value: Any) -> str | None:
    if value in {None, ""}:
        return None
    if isinstance(value, datetime):
        parsed = value
    else:
        text = str(value).strip()
        if not text:
            return None
        try:
            parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
        except ValueError:
            return text
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc).isoformat()


def _normalize_text(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _append_event_id(existing: tuple[str, ...], event_id: str, *, max_items: int = 256) -> tuple[str, ...]:
    merged = tuple([*existing, event_id])
    if len(merged) <= max_items:
        return merged
    return merged[-max_items:]


def _append_unique_text(existing: tuple[str, ...], value: Any) -> tuple[str, ...]:
    normalized = _normalize_text(value)
    if normalized is None:
        return existing
    if normalized in existing:
        return existing
    return tuple([*existing, normalized])


def _remove_text(existing: tuple[str, ...], value: Any) -> tuple[str, ...]:
    normalized = _normalize_text(value)
    if normalized is None:
        return existing
    return tuple(item for item in existing if item != normalized)


def _normalize_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    text = str(value or "").strip().lower()
    return text in {"1", "true", "yes", "on", "armed"}


def _is_protective_order(payload: dict[str, Any]) -> bool:
    if _normalize_bool(payload.get("is_protective")):
        return True
    role = str(payload.get("order_role", "")).strip().upper()
    return role in {"PROTECTIVE", "SL", "TP", "STOP_LOSS", "TAKE_PROFIT"}


@dataclass(frozen=True)
class BaselineState:
    profit_per_bar: float
    window_bars: int | None = None
    captured_at: str | None = None
    source: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "profit_per_bar", float(self.profit_per_bar))
        object.__setattr__(self, "captured_at", _normalize_timestamp(self.captured_at))
        object.__setattr__(self, "source", _normalize_text(self.source))
        if self.window_bars is not None:
            window_bars = int(self.window_bars)
            if window_bars <= 0:
                raise ValueError("window_bars must be positive when provided.")
            object.__setattr__(self, "window_bars", window_bars)

    def to_dict(self) -> dict[str, Any]:
        return {
            "profit_per_bar": self.profit_per_bar,
            "window_bars": self.window_bars,
            "captured_at": self.captured_at,
            "source": self.source,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "BaselineState":
        return cls(
            profit_per_bar=float(payload["profit_per_bar"]),
            window_bars=payload.get("window_bars"),
            captured_at=payload.get("captured_at"),
            source=payload.get("source"),
        )


@dataclass(frozen=True)
class OperatorAction:
    action: str
    occurred_at: str = field(default_factory=_utcnow_iso)
    actor: str | None = None
    reason: str | None = None

    def __post_init__(self) -> None:
        action = str(self.action).strip().upper()
        if not action:
            raise ValueError("action must not be empty.")
        object.__setattr__(self, "action", action)
        object.__setattr__(self, "occurred_at", _normalize_timestamp(self.occurred_at) or _utcnow_iso())
        object.__setattr__(self, "actor", _normalize_text(self.actor))
        object.__setattr__(self, "reason", _normalize_text(self.reason))

    def to_dict(self) -> dict[str, Any]:
        return {
            "action": self.action,
            "occurred_at": self.occurred_at,
            "actor": self.actor,
            "reason": self.reason,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "OperatorAction":
        return cls(
            action=str(payload["action"]),
            occurred_at=str(payload.get("occurred_at", _utcnow_iso())),
            actor=payload.get("actor"),
            reason=payload.get("reason"),
        )


def _append_operator_action(
    existing: tuple[OperatorAction, ...],
    action: OperatorAction,
    *,
    max_items: int = 64,
) -> tuple[OperatorAction, ...]:
    merged = tuple([*existing, action])
    if len(merged) <= max_items:
        return merged
    return merged[-max_items:]


@dataclass(frozen=True)
class PositionState:
    side: str = "FLAT"
    quantity: float = 0.0
    avg_price: float | None = None
    opened_at: str | None = None

    def __post_init__(self) -> None:
        side = self.side.strip().upper() or "FLAT"
        if side not in {"FLAT", "LONG", "SHORT"}:
            raise ValueError("side must be FLAT, LONG, or SHORT.")
        quantity = abs(float(self.quantity))
        avg_price = None if self.avg_price is None else float(self.avg_price)
        opened_at = _normalize_timestamp(self.opened_at)
        if side == "FLAT":
            quantity = 0.0
            avg_price = None
            opened_at = None
        elif quantity <= 0.0:
            raise ValueError("Non-flat positions require a positive quantity.")
        object.__setattr__(self, "side", side)
        object.__setattr__(self, "quantity", quantity)
        object.__setattr__(self, "avg_price", avg_price)
        object.__setattr__(self, "opened_at", opened_at)

    @classmethod
    def flat(cls) -> "PositionState":
        return cls()

    def to_dict(self) -> dict[str, Any]:
        return {
            "side": self.side,
            "quantity": self.quantity,
            "avg_price": self.avg_price,
            "opened_at": self.opened_at,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "PositionState":
        return cls(
            side=str(payload.get("side", "FLAT")),
            quantity=float(payload.get("quantity", 0.0)),
            avg_price=payload.get("avg_price"),
            opened_at=payload.get("opened_at"),
        )


@dataclass(frozen=True)
class RuntimeState:
    version: int = 2
    session_id: str | None = None
    strategy_id: str | None = None
    instrument: str | None = None
    timeframe: str | None = None
    updated_at: str = field(default_factory=_utcnow_iso)
    armed: bool = False
    baseline: BaselineState | None = None
    last_seen_bar_at: str | None = None
    last_processed_closed_bar_at: str | None = None
    last_decision: str | None = None
    position: PositionState = field(default_factory=PositionState.flat)
    pending_order_ids: tuple[str, ...] = ()
    broker_order_ids: tuple[str, ...] = ()
    protective_order_ids: tuple[str, ...] = ()
    operator_actions: tuple[OperatorAction, ...] = ()
    applied_event_ids: tuple[str, ...] = ()
    last_applied_sequence: int = 0
    extra: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.version < 1:
            raise ValueError("version must be >= 1.")
        if self.last_applied_sequence < 0:
            raise ValueError("last_applied_sequence must be >= 0.")
        object.__setattr__(self, "updated_at", _normalize_timestamp(self.updated_at) or _utcnow_iso())
        object.__setattr__(self, "armed", bool(self.armed))
        if self.baseline is not None and not isinstance(self.baseline, BaselineState):
            object.__setattr__(self, "baseline", BaselineState.from_dict(dict(self.baseline)))
        object.__setattr__(self, "last_seen_bar_at", _normalize_timestamp(self.last_seen_bar_at))
        object.__setattr__(self, "last_processed_closed_bar_at", _normalize_timestamp(self.last_processed_closed_bar_at))
        object.__setattr__(self, "last_decision", _normalize_text(self.last_decision))
        object.__setattr__(self, "pending_order_ids", tuple(str(item) for item in self.pending_order_ids))
        object.__setattr__(self, "broker_order_ids", tuple(str(item) for item in self.broker_order_ids))
        object.__setattr__(self, "protective_order_ids", tuple(str(item) for item in self.protective_order_ids))
        normalized_actions = tuple(
            item if isinstance(item, OperatorAction) else OperatorAction.from_dict(dict(item))
            for item in self.operator_actions
        )
        object.__setattr__(self, "operator_actions", normalized_actions)
        object.__setattr__(self, "applied_event_ids", tuple(str(item) for item in self.applied_event_ids))
        object.__setattr__(self, "extra", dict(self.extra))

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "session_id": self.session_id,
            "strategy_id": self.strategy_id,
            "instrument": self.instrument,
            "timeframe": self.timeframe,
            "updated_at": self.updated_at,
            "armed": self.armed,
            "baseline": self.baseline.to_dict() if self.baseline is not None else None,
            "last_seen_bar_at": self.last_seen_bar_at,
            "last_processed_closed_bar_at": self.last_processed_closed_bar_at,
            "last_decision": self.last_decision,
            "position": self.position.to_dict(),
            "pending_order_ids": list(self.pending_order_ids),
            "broker_order_ids": list(self.broker_order_ids),
            "protective_order_ids": list(self.protective_order_ids),
            "operator_actions": [item.to_dict() for item in self.operator_actions],
            "applied_event_ids": list(self.applied_event_ids),
            "last_applied_sequence": self.last_applied_sequence,
            "extra": self.extra,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "RuntimeState":
        return cls(
            version=int(payload.get("version", 1)),
            session_id=payload.get("session_id"),
            strategy_id=payload.get("strategy_id"),
            instrument=payload.get("instrument"),
            timeframe=payload.get("timeframe"),
            updated_at=str(payload.get("updated_at", _utcnow_iso())),
            armed=bool(payload.get("armed", False)),
            baseline=(BaselineState.from_dict(payload["baseline"]) if payload.get("baseline") else None),
            last_seen_bar_at=payload.get("last_seen_bar_at"),
            last_processed_closed_bar_at=payload.get("last_processed_closed_bar_at"),
            last_decision=payload.get("last_decision"),
            position=PositionState.from_dict(payload.get("position") or {}),
            pending_order_ids=tuple(payload.get("pending_order_ids") or ()),
            broker_order_ids=tuple(payload.get("broker_order_ids") or ()),
            protective_order_ids=tuple(payload.get("protective_order_ids") or ()),
            operator_actions=tuple(
                OperatorAction.from_dict(item) for item in payload.get("operator_actions") or ()
            ),
            applied_event_ids=tuple(payload.get("applied_event_ids") or ()),
            last_applied_sequence=int(payload.get("last_applied_sequence", 0)),
            extra=dict(payload.get("extra") or {}),
        )


def apply_execution_event(state: RuntimeState, event: ExecutionEvent) -> RuntimeState:
    if event.sequence is not None and event.sequence <= state.last_applied_sequence:
        return state
    if event.event_id in state.applied_event_ids:
        return state

    payload = dict(event.payload)
    pending_order_ids = list(state.pending_order_ids)
    broker_order_ids = state.broker_order_ids
    protective_order_ids = state.protective_order_ids
    operator_actions = state.operator_actions
    position = state.position
    last_seen_bar_at = state.last_seen_bar_at
    last_processed_closed_bar_at = state.last_processed_closed_bar_at
    last_decision = state.last_decision
    armed = state.armed
    baseline = state.baseline
    session_id = state.session_id or _normalize_text(payload.get("session_id"))
    strategy_id = state.strategy_id or event.strategy_id or _normalize_text(payload.get("strategy_id"))
    instrument = state.instrument or event.instrument or _normalize_text(payload.get("instrument"))
    timeframe = state.timeframe or _normalize_text(payload.get("timeframe"))
    extra = dict(state.extra)

    if event.event_type == "heartbeat":
        last_seen_bar_at = _normalize_timestamp(payload.get("bar_timestamp") or event.occurred_at)
        last_processed_closed_bar_at = _normalize_timestamp(payload.get("closed_bar_timestamp")) or last_processed_closed_bar_at
    elif event.event_type == "decision":
        last_decision = _normalize_text(payload.get("decision") or payload.get("signal"))
        if last_decision is not None:
            last_decision = last_decision.upper()
        last_processed_closed_bar_at = (
            _normalize_timestamp(payload.get("closed_bar_timestamp") or payload.get("bar_timestamp"))
            or last_processed_closed_bar_at
        )
    elif event.event_type == "order_submitted":
        order_id = _normalize_text(payload.get("order_id"))
        if order_id and order_id not in pending_order_ids:
            pending_order_ids.append(order_id)
        broker_order_ids = _append_unique_text(broker_order_ids, order_id)
        if _is_protective_order(payload):
            protective_order_ids = _append_unique_text(protective_order_ids, order_id)
    elif event.event_type in {"order_cancelled", "order_rejected"}:
        order_id = _normalize_text(payload.get("order_id"))
        pending_order_ids = [current for current in pending_order_ids if current != order_id]
        broker_order_ids = _remove_text(broker_order_ids, order_id)
        protective_order_ids = _remove_text(protective_order_ids, order_id)
    elif event.event_type == "fill":
        order_id = _normalize_text(payload.get("order_id"))
        pending_order_ids = [current for current in pending_order_ids if current != order_id]
        broker_order_ids = _remove_text(broker_order_ids, order_id)
        protective_order_ids = _remove_text(protective_order_ids, order_id)
        if "resulting_position" in payload:
            position = PositionState.from_dict(payload["resulting_position"])
    elif event.event_type == "position_snapshot":
        position = PositionState.from_dict(payload)
    elif event.event_type == "position_closed":
        order_id = _normalize_text(payload.get("order_id"))
        pending_order_ids = [current for current in pending_order_ids if current != order_id]
        broker_order_ids = _remove_text(broker_order_ids, order_id)
        protective_order_ids = ()
        position = PositionState.flat()
    elif event.event_type == "armed":
        armed = True
        operator_actions = _append_operator_action(
            operator_actions,
            OperatorAction(
                action="ARM",
                occurred_at=event.occurred_at,
                actor=payload.get("actor"),
                reason=payload.get("reason"),
            ),
        )
    elif event.event_type == "disarmed":
        armed = False
        operator_actions = _append_operator_action(
            operator_actions,
            OperatorAction(
                action="DISARM",
                occurred_at=event.occurred_at,
                actor=payload.get("actor"),
                reason=payload.get("reason"),
            ),
        )
    elif event.event_type in {"baseline_captured", "baseline_updated"}:
        baseline = BaselineState.from_dict(payload)
    elif event.event_type == "operator_action":
        operator_actions = _append_operator_action(
            operator_actions,
            OperatorAction(
                action=str(payload.get("action", "UNKNOWN")),
                occurred_at=payload.get("occurred_at") or event.occurred_at,
                actor=payload.get("actor"),
                reason=payload.get("reason"),
            ),
        )
    else:
        extra["last_event_type"] = event.event_type

    return replace(
        state,
        session_id=session_id,
        strategy_id=strategy_id,
        instrument=instrument,
        timeframe=timeframe,
        updated_at=_utcnow_iso(),
        armed=armed,
        baseline=baseline,
        last_seen_bar_at=last_seen_bar_at,
        last_processed_closed_bar_at=last_processed_closed_bar_at,
        last_decision=last_decision,
        position=position,
        pending_order_ids=tuple(pending_order_ids),
        broker_order_ids=broker_order_ids,
        protective_order_ids=protective_order_ids,
        operator_actions=operator_actions,
        applied_event_ids=_append_event_id(state.applied_event_ids, event.event_id),
        last_applied_sequence=max(state.last_applied_sequence, int(event.sequence or 0)),
        extra=extra,
    )


def replay_execution_events(
    state: RuntimeState | None,
    events: Iterable[ExecutionEvent],
) -> RuntimeState:
    restored = state or RuntimeState()
    for event in sorted(events, key=lambda item: int(item.sequence or 0)):
        restored = apply_execution_event(restored, event)
    return restored


class RuntimeStateStore:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path).expanduser().resolve()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()

    def save(self, state: RuntimeState) -> RuntimeState:
        persisted = replace(state, updated_at=_utcnow_iso())
        payload = json.dumps(persisted.to_dict(), ensure_ascii=True, indent=2, sort_keys=True)
        with self._lock:
            tmp_fd, tmp_name = tempfile.mkstemp(
                dir=self.path.parent,
                prefix=f"{self.path.stem}.",
                suffix=".tmp",
            )
            try:
                with os.fdopen(tmp_fd, "w", encoding="utf-8", newline="\n") as handle:
                    handle.write(payload)
                    handle.flush()
                    os.fsync(handle.fileno())
                os.replace(tmp_name, self.path)
            finally:
                if os.path.exists(tmp_name):
                    os.unlink(tmp_name)
        return persisted

    def load(self) -> RuntimeState | None:
        if not self.path.exists():
            return None
        payload = json.loads(self.path.read_text(encoding="utf-8"))
        return RuntimeState.from_dict(payload)

    def restore(self, journal: ExecutionJournal | None = None) -> RuntimeState:
        state = self.load() or RuntimeState()
        if journal is None:
            return state
        return replay_execution_events(state, journal.read_events())