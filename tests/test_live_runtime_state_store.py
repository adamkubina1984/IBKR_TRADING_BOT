from __future__ import annotations

from ibkr_trading_bot.core.services.live import (
    BaselineState,
    ExecutionEvent,
    ExecutionJournal,
    OperatorAction,
    PositionState,
    RuntimeState,
    RuntimeStateStore,
)


def test_runtime_state_store_round_trip(tmp_path):
    store = RuntimeStateStore(tmp_path / "runtime.state.json")
    state = RuntimeState(
        session_id="session-1",
        strategy_id="demo-strategy",
        instrument="GC",
        timeframe="5m",
        armed=True,
        baseline=BaselineState(
            profit_per_bar=0.42,
            window_bars=600,
            captured_at="2026-04-29T12:25:00Z",
            source="paper-baseline",
        ),
        last_processed_closed_bar_at="2026-04-29T12:30:00Z",
        last_decision="LONG",
        position=PositionState(
            side="LONG",
            quantity=1.0,
            avg_price=101.25,
            opened_at="2026-04-29T12:30:00Z",
        ),
        pending_order_ids=("ord-1",),
        broker_order_ids=("ord-1", "ord-sl-1"),
        protective_order_ids=("ord-sl-1",),
        operator_actions=(
            OperatorAction(action="ARM", occurred_at="2026-04-29T12:29:00Z", actor="tester", reason="enable paper"),
        ),
        applied_event_ids=("evt-1",),
        last_applied_sequence=1,
        extra={"note": "checkpoint"},
    )

    saved = store.save(state)
    loaded = store.load()

    assert loaded == saved


def test_runtime_state_restore_is_idempotent_across_restarts(tmp_path):
    store = RuntimeStateStore(tmp_path / "runtime.state.json")
    journal = ExecutionJournal(tmp_path / "execution.journal.jsonl")

    base_state = store.save(RuntimeState(session_id="session-1", strategy_id="demo-strategy", instrument="GC"))
    assert base_state.last_applied_sequence == 0

    journal.append(
        ExecutionEvent(
            event_type="armed",
            event_id="evt-1",
            payload={"actor": "tester", "reason": "paper on"},
        )
    )
    journal.append(
        ExecutionEvent(
            event_type="baseline_captured",
            event_id="evt-2",
            payload={
                "profit_per_bar": 0.42,
                "window_bars": 600,
                "captured_at": "2026-04-29T12:30:00Z",
                "source": "paper-baseline",
            },
        )
    )
    journal.append(
        ExecutionEvent(
            event_type="decision",
            event_id="evt-3",
            payload={"decision": "LONG", "closed_bar_timestamp": "2026-04-29T12:35:00Z"},
        )
    )
    journal.append(
        ExecutionEvent(event_type="order_submitted", event_id="evt-4", payload={"order_id": "ord-1"})
    )
    journal.append(
        ExecutionEvent(
            event_type="order_submitted",
            event_id="evt-5",
            payload={"order_id": "ord-sl-1", "is_protective": True},
        )
    )
    journal.append(
        ExecutionEvent(
            event_type="fill",
            event_id="evt-6",
            payload={
                "order_id": "ord-1",
                "resulting_position": {
                    "side": "LONG",
                    "quantity": 1.0,
                    "avg_price": 100.5,
                    "opened_at": "2026-04-29T12:31:00Z",
                },
            },
        )
    )

    restored_once = store.restore(journal)
    checkpointed = store.save(restored_once)
    restored_twice = store.restore(journal)

    assert restored_once.armed is True
    assert restored_once.baseline is not None
    assert restored_once.baseline.window_bars == 600
    assert restored_once.last_decision == "LONG"
    assert restored_once.last_processed_closed_bar_at == "2026-04-29T12:35:00+00:00"
    assert restored_once.position.side == "LONG"
    assert restored_once.position.avg_price == 100.5
    assert restored_once.pending_order_ids == ("ord-sl-1",)
    assert restored_once.broker_order_ids == ("ord-sl-1",)
    assert restored_once.protective_order_ids == ("ord-sl-1",)
    assert restored_once.operator_actions[-1].action == "ARM"
    assert restored_once.last_applied_sequence == 6
    assert restored_twice == checkpointed