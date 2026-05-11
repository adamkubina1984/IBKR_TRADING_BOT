from __future__ import annotations

import json
import os
import threading
import uuid
from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def _utcnow_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _normalize_timestamp(value: Any) -> str:
    if isinstance(value, datetime):
        parsed = value
    else:
        text = str(value or "").strip()
        if not text:
            return _utcnow_iso()
        try:
            parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
        except ValueError:
            return text
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc).isoformat()


@dataclass(frozen=True)
class ExecutionEvent:
    event_type: str
    occurred_at: str = field(default_factory=_utcnow_iso)
    payload: dict[str, Any] = field(default_factory=dict)
    event_id: str = field(default_factory=lambda: uuid.uuid4().hex)
    sequence: int | None = None
    instrument: str | None = None
    strategy_id: str | None = None
    correlation_id: str | None = None

    def __post_init__(self) -> None:
        event_type = self.event_type.strip()
        if not event_type:
            raise ValueError("event_type must not be empty.")
        if not self.event_id.strip():
            raise ValueError("event_id must not be empty.")
        if self.sequence is not None and self.sequence < 1:
            raise ValueError("sequence must be positive when provided.")
        object.__setattr__(self, "event_type", event_type)
        object.__setattr__(self, "event_id", self.event_id.strip())
        object.__setattr__(self, "occurred_at", _normalize_timestamp(self.occurred_at))
        object.__setattr__(self, "payload", dict(self.payload))

    def to_dict(self) -> dict[str, Any]:
        return {
            "event_id": self.event_id,
            "sequence": self.sequence,
            "event_type": self.event_type,
            "occurred_at": self.occurred_at,
            "instrument": self.instrument,
            "strategy_id": self.strategy_id,
            "correlation_id": self.correlation_id,
            "payload": self.payload,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "ExecutionEvent":
        return cls(
            event_id=str(payload.get("event_id") or uuid.uuid4().hex),
            sequence=payload.get("sequence"),
            event_type=str(payload["event_type"]),
            occurred_at=str(payload.get("occurred_at", _utcnow_iso())),
            instrument=payload.get("instrument"),
            strategy_id=payload.get("strategy_id"),
            correlation_id=payload.get("correlation_id"),
            payload=dict(payload.get("payload") or {}),
        )


class ExecutionJournal:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path).expanduser().resolve()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._next_sequence = self._discover_next_sequence()

    def append(self, event: ExecutionEvent) -> ExecutionEvent:
        with self._lock:
            persisted = replace(event, sequence=self._next_sequence)
            self._next_sequence += 1
            with self.path.open("a", encoding="utf-8", newline="\n") as handle:
                handle.write(json.dumps(persisted.to_dict(), ensure_ascii=True, sort_keys=True))
                handle.write("\n")
                handle.flush()
                os.fsync(handle.fileno())
            return persisted

    def append_event(self, event_type: str, *, payload: dict[str, Any] | None = None, **kwargs: Any) -> ExecutionEvent:
        return self.append(ExecutionEvent(event_type=event_type, payload=payload or {}, **kwargs))

    def read_events(self) -> list[ExecutionEvent]:
        if not self.path.exists():
            return []
        restored: list[ExecutionEvent] = []
        with self.path.open("r", encoding="utf-8") as handle:
            for line in handle:
                text = line.strip()
                if not text:
                    continue
                restored.append(ExecutionEvent.from_dict(json.loads(text)))
        return restored

    def last_sequence(self) -> int:
        return self._next_sequence - 1

    def _discover_next_sequence(self) -> int:
        if not self.path.exists():
            return 1
        last_seen = 0
        with self.path.open("r", encoding="utf-8") as handle:
            for line in handle:
                text = line.strip()
                if not text:
                    continue
                payload = json.loads(text)
                last_seen = int(payload.get("sequence") or last_seen)
        return last_seen + 1