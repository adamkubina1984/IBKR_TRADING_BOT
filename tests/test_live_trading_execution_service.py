from __future__ import annotations

import json

import pytest

from ibkr_trading_bot.core.services.live import ExecutionJournal, RuntimeStateStore
from ibkr_trading_bot.core.services.live_trading_execution_service import (
    LiveTradingExecutionConfig,
    LiveTradingExecutionService,
)


def _make_service(tmp_path, **config_overrides) -> LiveTradingExecutionService:
    config = LiveTradingExecutionConfig(
        strategy_id="demo-strategy",
        instrument="GC",
        timeframe="5m",
        warning_alpha=1.0,
        **config_overrides,
    )
    return LiveTradingExecutionService(
        config,
        journal=ExecutionJournal(tmp_path / "execution.journal.jsonl"),
        state_store=RuntimeStateStore(tmp_path / "runtime.state.json"),
    )


def test_service_enforces_one_position_max_on_repeated_entry_signal(tmp_path):
    service = _make_service(tmp_path, min_bars_for_health=10)
    service.arm_trading(baseline_profit_per_bar=0.8, actor="tester")

    first = service.process_closed_bar(
        "2026-04-29T12:30:00Z",
        100.0,
        model_direction="LONG",
        confidence=1.0,
    )
    second = service.process_closed_bar(
        "2026-04-29T12:35:00Z",
        101.0,
        model_direction="LONG",
        confidence=1.0,
    )

    assert [intent.action for intent in first.order_intents] == ["ENTRY_LONG"]
    assert second.order_intents == ()
    assert second.status.runtime_state.position.side == "LONG"
    assert second.status.mode == "LIVE"


def test_service_safe_stops_when_live_profit_per_bar_turns_negative(tmp_path):
    service = _make_service(tmp_path, min_bars_for_health=2)
    service.arm_trading(baseline_profit_per_bar=0.4, actor="tester")

    service.process_closed_bar("2026-04-29T12:30:00Z", 100.0, model_direction="LONG", confidence=1.0)
    result = service.process_closed_bar("2026-04-29T12:35:00Z", 99.0, model_direction="SHORT", confidence=1.0)

    assert result.status.mode == "SAFE_STOP"
    assert result.status.last_safe_stop_reason == "LIVE_PROFIT_PER_BAR_BELOW_ZERO"


def test_service_safe_stops_when_live_profit_per_bar_drops_below_half_baseline(tmp_path):
    service = _make_service(tmp_path, min_bars_for_health=2)
    service.arm_trading(baseline_profit_per_bar=1.0, actor="tester")

    service.process_closed_bar("2026-04-29T12:30:00Z", 100.0, model_direction="LONG", confidence=1.0)
    result = service.process_closed_bar("2026-04-29T12:35:00Z", 100.8, model_direction="SHORT", confidence=1.0)

    assert result.status.live_profit_per_bar == pytest.approx(0.4)
    assert result.status.mode == "SAFE_STOP"
    assert result.status.last_safe_stop_reason == "LIVE_PROFIT_PER_BAR_BELOW_BASELINE_FLOOR"


def test_service_warning_hysteresis_enters_and_clears_on_ewma_thresholds(tmp_path):
    service = _make_service(tmp_path, rolling_window_bars=2, min_bars_for_health=2)
    service.arm_trading(baseline_profit_per_bar=0.6, actor="tester")

    service.process_closed_bar("2026-04-29T12:30:00Z", 100.0, model_direction="LONG", confidence=1.0)
    warning = service.process_closed_bar("2026-04-29T12:35:00Z", 100.8, model_direction="SHORT", confidence=1.0)
    assert warning.status.mode == "WARNING"
    assert warning.status.last_warning_reason == "EWMA_PROFIT_PER_BAR_DEGRADED"

    service.process_closed_bar("2026-04-29T12:40:00Z", 100.8, model_direction="SHORT", confidence=1.0)
    cleared = service.process_closed_bar("2026-04-29T12:45:00Z", 99.6, model_direction="LONG", confidence=1.0)

    assert cleared.status.mode == "LIVE"
    assert cleared.status.last_warning_reason is None


def test_service_blocks_trading_when_data_is_stale(tmp_path):
    service = _make_service(tmp_path, freshness_timeout_sec=60, min_bars_for_health=10)
    service.arm_trading(baseline_profit_per_bar=0.8, actor="tester")
    service.process_closed_bar("2026-04-29T12:30:00Z", 100.0, model_direction="FLAT", confidence=0.0)

    status = service.check_freshness(now="2026-04-29T12:31:01Z")

    assert status.mode == "SAFE_STOP"
    assert status.last_safe_stop_reason == "STALE_DATA"


def test_service_distinguishes_controlled_disarm_from_emergency_stop(tmp_path):
    service = _make_service(tmp_path, min_bars_for_health=10)
    service.arm_trading(baseline_profit_per_bar=0.8, actor="tester")
    service.process_closed_bar("2026-04-29T12:30:00Z", 100.0, model_direction="LONG", confidence=1.0)

    service.disarm_trading(actor="tester", reason="operator_off")
    disarmed = service.process_closed_bar("2026-04-29T12:35:00Z", 99.5, model_direction="SHORT", confidence=1.0)

    assert [intent.action for intent in disarmed.order_intents] == ["EXIT_LONG"]
    assert disarmed.status.mode == "OBSERVE"
    assert disarmed.status.runtime_state.position.side == "FLAT"

    service.arm_trading(actor="tester")
    service.emergency_stop("operator_emergency", actor="tester")
    stopped = service.process_closed_bar("2026-04-29T12:40:00Z", 99.0, model_direction="LONG", confidence=1.0)

    assert stopped.order_intents == ()
    assert stopped.status.mode == "EMERGENCY_STOP"
    assert stopped.status.last_safe_stop_reason == "operator_emergency"


def test_service_restores_runtime_state_on_restart(tmp_path):
    service = _make_service(tmp_path, min_bars_for_health=10)
    service.arm_trading(baseline_profit_per_bar=0.8, actor="tester")
    service.process_closed_bar("2026-04-29T12:30:00Z", 100.0, model_direction="LONG", confidence=1.0)

    restored = _make_service(tmp_path, min_bars_for_health=10)

    assert restored.status.runtime_state.position.side == "LONG"
    assert restored.status.runtime_state.baseline is not None
    assert restored.status.mode == "LIVE"


def test_service_persists_exit_policy_into_runtime_state_and_journal(tmp_path):
    service = _make_service(tmp_path, min_bars_for_health=10, exit_policy="legacy_flat_exit")
    service.arm_trading(baseline_profit_per_bar=0.8, actor="tester")
    service.process_closed_bar("2026-04-29T12:30:00Z", 100.0, model_direction="LONG", confidence=1.0)

    state_payload = json.loads((tmp_path / "runtime.state.json").read_text(encoding="utf-8"))
    journal_lines = (tmp_path / "execution.journal.jsonl").read_text(encoding="utf-8").strip().splitlines()
    journal_payloads = [json.loads(line) for line in journal_lines if line.strip()]

    assert state_payload["extra"]["exit_policy"] == "flat_on_weak_signal"
    assert journal_payloads
    assert all(event["payload"].get("exit_policy") == "flat_on_weak_signal" for event in journal_payloads)