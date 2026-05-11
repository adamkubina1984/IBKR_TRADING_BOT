from __future__ import annotations

from ibkr_trading_bot.core.services.live import ExecutionEvent, ExecutionJournal


def test_execution_journal_appends_and_restores_events(tmp_path):
    journal_path = tmp_path / "execution.journal.jsonl"
    journal = ExecutionJournal(journal_path)

    first = journal.append(ExecutionEvent(event_type="decision", event_id="evt-1", payload={"decision": "LONG"}))
    second = journal.append(
        ExecutionEvent(
            event_type="fill",
            event_id="evt-2",
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

    restored = ExecutionJournal(journal_path).read_events()

    assert first.sequence == 1
    assert second.sequence == 2
    assert [event.sequence for event in restored] == [1, 2]
    assert restored[0].payload["decision"] == "LONG"
    assert restored[1].payload["resulting_position"]["avg_price"] == 100.5